"""SPHEREx instrument adapter — all LVF / subchannel specifics behind one interface.

The generic run engine (``selfcal_scripts.runner``) treats an instrument as a
black box that turns a run config into the geometry the solver/mosaicker need:
the per-"job" valid region, chunk maps, adjacency builders, aux maps, an offset
renderer, and an optional wavelength-append hook. Everything SPHEREx-LVF-specific
— subchannel windows, stripped chunk maps, BC/BW wavelength, ``wav_coadd``,
column/subchannel adjacency, the H2RG readout-channel map — lives here, so the
engine never imports it and a broadband instrument (no LVF) can plug in with none
of this baggage. See ``selfcal.instruments.base.Instrument`` for the contract.

Methods take plain dicts/args (an ``inst_cfg`` mapping = the TOML ``[instrument]``
table), not the runner's RunConfig, so the package stays independent of the runner.
"""
import os
from dataclasses import dataclass
from functools import partial

import numpy as np

from .spherex_utility import (
    load_calibration, load_lvf_params, make_stripped_chunk_map,
    make_stripped_chunk_valid_mask, fast_vertical_dist,
    compute_column_adjacency, compute_subchannel_adjacency,
    compute_column_polynomial_chains, compute_subchannel_polynomial_chains,
    make_spherex_stripped_offset_map)
from .wavemap import wav_coadd

# SPHEREx per-band spectral calibration (BC/BW maps) default location.
SPHEREX_CALIB_DIR = '/data3/SPHEREx/SpecCal_202509/ParameterFiles'

# Named subchannel windows -> (inclusive-low, exclusive-high) subch index range.
# Aromatic/Aliphatic are stable; the PAH-fit window is run-dependent (different
# production runs have chosen different subchannel ranges), so it is NOT a
# global preset — such runs set an explicit `subch_window = [lo, hi]` in their
# config instead.
SUBCH_WINDOWS = {
    'Aromatic': (225, 236),
    'Aliphatic': (249, 260),
}


@dataclass(frozen=True)
class Job:
    """One unit of the channel loop: a name (feeds the cal/mosaic filename) + a
    spatial selection (a subchannel window or a list of channel ids)."""
    name: str
    kind: str            # 'window' | 'channels'
    value: object        # (lo, hi) for 'window'; list[int] for 'channels'


def make_readout_chunk_map(det_shape=(2040, 2040), col_start=60, col_width=64):
    """Per-readout-channel chunk map at detector resolution (H2RG, post 4px trim).
    Chunk 0 covers the first ``col_start`` reference columns; then one chunk per
    ``col_width``-wide readout column, plus a final partial chunk for any
    remaining columns. Returns (chunk_map int32, n_chunks)."""
    H, W = det_shape
    chunk_map = np.full(det_shape, -1, dtype=np.int32)
    chunk_map[:, :col_start] = 0
    n_full = (W - col_start) // col_width
    for i in range(n_full):
        x0 = col_start + i * col_width
        chunk_map[:, x0: x0 + col_width] = i + 1
    right_start = col_start + n_full * col_width
    n_chunks = n_full + 1
    if right_start < W:
        chunk_map[:, right_start:] = n_chunks
        n_chunks += 1
    assert (chunk_map >= 0).all(), "every pixel must be assigned a readout channel"
    return chunk_map, n_chunks


def upsample_chunk_map(det_chunk_map, factor):
    """Replicate each detector pixel into a (factor x factor) block, preserving ids."""
    if factor == 1:
        return det_chunk_map
    return np.kron(det_chunk_map, np.ones((factor, factor), dtype=det_chunk_map.dtype))


class SPHERExInstrument:
    """SPHEREx (LVF) instrument adapter. ``capabilities`` lets modes that need
    LVF features (per-pixel wavelength, subchannels) declare a requirement that
    is checked against the instrument; a broadband adapter would omit them."""

    name = 'spherex'
    capabilities = frozenset({'wavelength', 'subchannel'})

    # ---- jobs (the channel loop) -------------------------------------------
    def jobs(self, inst_cfg):
        """Expand the [instrument] selection keys into a list of Job.
        Exactly one of: windows (named presets) / subch_window (+window_name) /
        channels / channel_range."""
        windows = inst_cfg.get('windows')
        subch_window = inst_cfg.get('subch_window')
        channels = inst_cfg.get('channels')
        crange = inst_cfg.get('channel_range')
        window_defs = inst_cfg.get('window_defs', {})
        if windows is not None:
            out = []
            for w in windows:
                if w in window_defs:
                    lo, hi = window_defs[w]
                elif w in SUBCH_WINDOWS:
                    lo, hi = SUBCH_WINDOWS[w]
                else:
                    raise ValueError(
                        f"unknown window {w!r}; add it to [instrument.window_defs] "
                        f"or use subch_window")
                out.append(Job(name=w, kind='window', value=(int(lo), int(hi))))
            return out
        if subch_window is not None:
            lo, hi = subch_window
            name = inst_cfg.get('window_name', f'subch{lo}_{hi}')
            return [Job(name=name, kind='window', value=(int(lo), int(hi)))]
        if crange is not None:
            channels = [[i] for i in range(int(crange[0]), int(crange[1]))]
        if channels is not None:
            return [Job(name='Ch' + '-'.join(map(str, c)), kind='channels',
                        value=[int(x) for x in c]) for c in channels]
        raise ValueError("[instrument] needs one of: windows / subch_window / "
                         "channels / channel_range")

    # ---- detector-level geometry (built once per run) ----------------------
    def detector_inputs(self, inst_cfg, oversample):
        """LVF params, BC/BW, detector + grid stripped chunk maps, arc edges.
        NO adjacency (offset-structure-specific -> the mode builds it)."""
        det = inst_cfg['detector']
        ns, nch, ncol = inst_cfg['num_sub'], inst_cfg['num_ch'], inst_cfg['num_col']
        lvf_params = load_lvf_params(f'lvf_params_D{det}.npy')
        det_BC, det_BW = load_calibration(
            band=det, calibration_dir=inst_cfg.get('calib_dir', SPHEREX_CALIB_DIR))
        grid_chunk_map, _, _, _ = make_stripped_chunk_map(
            det, num_subchannels=ns, num_channels=nch, num_columns=ncol,
            oversample_factor=oversample, lvf_params=lvf_params)
        det_chunk_map, _, r_edges, x_edges = make_stripped_chunk_map(
            det, num_subchannels=ns, num_channels=nch, num_columns=ncol,
            oversample_factor=1, lvf_params=lvf_params)
        return {'lvf_params': lvf_params, 'det_BC': det_BC, 'det_BW': det_BW,
                'grid_chunk_map': grid_chunk_map, 'det_chunk_map': det_chunk_map,
                'r_edges': r_edges, 'x_edges': x_edges}

    # ---- per-job geometry (valid masks + edge-distance weights) ------------
    def channel_inputs(self, inst_cfg, det_inputs, job):
        ns, nch, ncol = inst_cfg['num_sub'], inst_cfg['num_ch'], inst_cfg['num_col']
        det_chunk_map = det_inputs['det_chunk_map']
        grid_chunk_map = det_inputs['grid_chunk_map']
        kw = dict(num_subchannels=ns, num_channels=nch, num_columns=ncol)
        if job.kind == 'window':
            lo, hi = job.value
            sel = dict(subch=np.arange(lo, hi))
        elif job.kind == 'channels':
            sel = dict(ch=job.value)
        else:
            raise ValueError(f"unknown job kind {job.kind!r}")
        cvm_pad = make_stripped_chunk_valid_mask(**sel, **kw, subchannel_padding=1)
        cvm = make_stripped_chunk_valid_mask(**sel, **kw, subchannel_padding=0)

        det_valid_mask = cvm[det_chunk_map]
        det_valid_weight = fast_vertical_dist(det_valid_mask)
        if np.max(det_valid_weight) > 0:
            det_valid_weight /= np.max(det_valid_weight)
        det_valid_mask_padded = cvm_pad[det_chunk_map]
        grid_valid_mask = cvm[grid_chunk_map]
        grid_valid_weight = fast_vertical_dist(grid_valid_mask)
        if np.max(grid_valid_weight) > 0:
            grid_valid_weight /= np.max(grid_valid_weight)
        return {'chunk_valid_mask_padded': cvm_pad, 'chunk_valid_mask': cvm,
                'det_valid_mask': det_valid_mask, 'grid_valid_mask': grid_valid_mask,
                'det_valid_mask_padded': det_valid_mask_padded,
                'det_valid_weight': det_valid_weight, 'grid_valid_weight': grid_valid_weight}

    # ---- adjacency + poly-chain builders (modes call what they need) -------
    def column_adjacency(self, det_chunk_map, num_columns):
        return compute_column_adjacency(det_chunk_map, num_columns)

    def subchannel_adjacency(self, det_chunk_map, num_columns):
        return compute_subchannel_adjacency(det_chunk_map, num_columns)

    def column_poly_chains(self, det_chunk_map, num_columns, degree=1):
        return compute_column_polynomial_chains(det_chunk_map, num_columns, degree=degree)

    def subchannel_poly_chains(self, num_subchannels, num_columns, degree, lo, hi):
        return compute_subchannel_polynomial_chains(
            num_subchannels=num_subchannels, num_columns=num_columns,
            degree=degree, subch_lo=lo, subch_hi=hi)

    def subchannel_poly_basis(self, det_chunk_map, num_columns, degree, lo, hi):
        """Hard poly-basis descriptor for a per-column subchannel polynomial
        offset (the ``poly_basis`` dict consumed by the instrument-agnostic core
        in ``selfcal.models.offset_basis``). This is the ONLY place the SPHEREx
        chunk encoding ``chunk = subchannel*num_col + column`` is inverted:
        ``chunk_coord`` = subchannel (the polynomial coordinate), ``chunk_group``
        = column (one independent polynomial per column). The core sees only the
        abstract coord/group arrays. Used by spectral modes whose offset is a
        degree-``degree`` Chebyshev in subchannel over the window ``[lo, hi]``."""
        n_chunks = int(det_chunk_map.max()) + 1
        chunk_ids = np.arange(n_chunks)
        return {
            'degree': int(degree),
            'num_groups': int(num_columns),
            'coord_lo': int(lo), 'coord_hi': int(hi),
            'chunk_coord': chunk_ids // int(num_columns),
            'chunk_group': chunk_ids % int(num_columns),
        }

    # ---- readout-channel geometry (k2 mode) --------------------------------
    def readout_chunk_map(self, det_shape, col_start=60, col_width=64):
        return make_readout_chunk_map(det_shape, col_start=col_start, col_width=col_width)

    def upsample_chunk_map(self, det_chunk_map, factor):
        return upsample_chunk_map(det_chunk_map, factor)

    # ---- precompute geometry params (rarely-run generator) -----------------
    def precompute(self, inst_cfg):
        """Generate + save per-detector LVF params (the rarely-run `precompute`
        task of the generic runner).
        Loops the detectors in inst_cfg['detectors']; saves lvf_params_D{N}.npy
        via spherex_utility.save_lvf_params (canonical package data dir, or
        inst_cfg['lvf_output_dir'] / $SELFCAL_LVF_PARAMS_DIR override)."""
        from .spherex_utility import make_fiducial_chunk_map, save_lvf_params
        ns = inst_cfg.get('num_sub', 10)
        nch = inst_cfg.get('num_ch', 34)
        out_dir = inst_cfg.get('lvf_output_dir')  # None -> canonical resolution
        calib_dir = inst_cfg.get('calib_dir', SPHEREX_CALIB_DIR)
        for det in inst_cfg['detectors']:
            det_BC, _ = load_calibration(band=det, calibration_dir=calib_dir)
            _, lvf_params = make_fiducial_chunk_map(
                det, det_BC, num_subchannels=ns, num_channels=nch, oversample_factor=1)
            lvf_params['filename'] = f'lvf_params_D{det}.npy'
            save_lvf_params(lvf_params, output_dir=out_dir)

    # ---- frame tag (cal/mosaic filename component) -------------------------
    def frame_tag(self, inst_cfg):
        return (f"Detector{inst_cfg['detector']}_NumSub{inst_cfg['num_sub']}"
                f"_NumCh{inst_cfg['num_ch']}_NumCol{inst_cfg['num_col']}")

    # ---- aux maps for spectral modes (per-pixel wavelength) ----------------
    def aux(self, det_inputs):
        return [det_inputs['det_BC'], det_inputs['det_BW']]

    # ---- mosaic helpers ----------------------------------------------------
    def offset_render(self, inst_cfg, det_inputs, channel_inputs):
        """Smooth subchannel-arc offset renderer for the mosaic (per job)."""
        ns, nch, ncol = inst_cfg['num_sub'], inst_cfg['num_ch'], inst_cfg['num_col']
        return partial(
            make_spherex_stripped_offset_map,
            chunk_valid_mask=channel_inputs['chunk_valid_mask'],
            lvf_params=det_inputs['lvf_params'], r_edges=det_inputs['r_edges'],
            x_edges=det_inputs['x_edges'], tot_subchannels=ns * nch + 2,
            num_columns=ncol, fill_invalid=True)

    def wavelength_append(self, det_inputs, mm, maps, sigma):
        """LVF wavelength coaddition -> append wav_mean/wav_std maps (full mosaic
        mode). The generic engine calls this only if the instrument provides it."""
        import time
        print("Coadding wavelength maps...")
        t00 = time.time()
        wav_mean, wav_std = wav_coadd(
            det_inputs['det_BC'], det_inputs['det_BW'],
            mean_map=maps['mean_map']['data'], std_map=maps['std_map']['data'],
            reproj_list=mm.reproj_list, cache_list=mm.cached_list,
            ref_shape=maps['mean_map']['data'].shape, sigma=sigma,
            batch_size=40, max_workers=30)
        print(f"Wavelength coaddition finished in {time.time() - t00:.2f} seconds.")
        mm.append_maps({'wav_mean_map': {'data': wav_mean, 'unit': 'um'},
                        'wav_std_map': {'data': wav_std, 'unit': 'um'}})
