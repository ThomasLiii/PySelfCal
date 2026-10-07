"""SPHEREx instrument — all LVF / subchannel specifics behind the Instrument contract.

The generic run engine treats an instrument as a black box that turns a run
config into the geometry the solver/mosaicker need. Everything SPHEREx-LVF-
specific — subchannel windows, the stripped (arc) chunk map and its
``(subchannel, column)`` chunk axes, the H2RG readout-channel map, the BC/BW
wavelength maps, the smooth arc offset renderer, the LVF wavelength coadd, the
zodi anchor, the FINAST astrometry filter — lives here, so the engine never
imports it and another instrument plugs in with none of this baggage. See
:mod:`selfcal.instruments.base` for the contract.

Methods take plain dicts/args (an ``inst_cfg`` mapping = the TOML ``[instrument]``
table), not the runner's RunConfig, so the package stays independent of the runner.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from functools import partial

import numpy as np

from ...models.offset_structure import ChunkAxes
from ..base import (Instrument, register_instrument, Job as _Job, ChunkMap, DetectorGeometry,
                    JobGeometry, ExposureLayout)
from .spherex_utility import (
    load_calibration, load_lvf_params, make_stripped_chunk_map,
    make_stripped_chunk_valid_mask, fast_vertical_dist,
    make_spherex_stripped_offset_map)
from .wavemap import wav_coadd

logger = logging.getLogger(__name__)

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
class Job(_Job):
    """One unit of the channel loop: a name (feeds the cal/mosaic filename) + a
    spatial selection (``kind`` 'window' -> ``value`` = (lo, hi) subchannels;
    'channels' -> a list of channel ids)."""


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
    # Internal invariant: verifies this function's own tiling of the map it just
    # built left no pixel unassigned (not caller-input validation) -> keep assert.
    assert (chunk_map >= 0).all(), "every pixel must be assigned a readout channel"
    return chunk_map, n_chunks


def upsample_chunk_map(det_chunk_map, factor):
    """Replicate each detector pixel into a (factor x factor) block, preserving ids."""
    if factor == 1:
        return det_chunk_map
    return np.kron(det_chunk_map, np.ones((factor, factor), dtype=det_chunk_map.dtype))


@register_instrument('spherex')
class SPHERExInstrument(Instrument):
    """SPHEREx (LVF) instrument. ``capabilities`` lets modes that need LVF
    features (per-pixel wavelength, a spectral chunk axis) declare a
    requirement that is checked against the instrument."""

    capabilities = frozenset({'wavelength', 'spectral_axis', 'subchannel'})

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

    # ---- frame tag (cal/mosaic filename component) -------------------------
    def frame_tag(self, inst_cfg):
        """Return the product-name tag ``Detector<d>_NumSub<s>_NumCh<c>_NumCol<k>``.

        The numbers are the required ``[instrument]`` keys ``detector``,
        ``num_sub``, ``num_ch`` and ``num_col``, e.g.
        ``Detector4_NumSub10_NumCh34_NumCol10``. A job's products are named
        ``cal_<tag>_<job><suffix>.h5`` and ``mosaic_<tag>_<job><suffix>.fits``
        (:class:`~selfcal.run.engine.RunContext`), where ``<job>`` is
        the job name (``Ch17``, ``Aromatic``, ...)."""
        return (f"Detector{inst_cfg['detector']}_NumSub{inst_cfg['num_sub']}"
                f"_NumCh{inst_cfg['num_ch']}_NumCol{inst_cfg['num_col']}")

    # ---- raw exposures (reproject task) ------------------------------------
    def exposure_layout(self, inst_cfg):
        """L2b exposures: science in extension 1, DQ bitmask in extension 2, one
        detector per file; keep only exposures with a converged astrometric
        solution (``FINAST == 0``)."""
        return ExposureLayout(
            sci_ext=[1], dq_ext=[2], detector_ids=[0], ref_use_ext=(1,),
            header_predicate=lambda h: h.get('FINAST', 2) == 0,
            header_keys=('FINAST',), header_ext=1,
            cache_tag=f"finast_D{inst_cfg['detector']}")

    # ---- detector-level geometry (built once per run) ----------------------
    def detector_geometry(self, inst_cfg, oversample):
        """LVF params, BC/BW, the stripped chunk map at detector + grid
        resolution with its (subchannel, column) axes, the readout-channel map.
        NO adjacency (offset-structure-specific -> the mode builds it).
        ``[instrument]`` ``calib_dir`` / ``lvf_dir`` choose the directories of the
        calibration maps and the LVF parameters (default: see
        :func:`~selfcal.instruments.spherex.spherex_utility.load_lvf_params`)."""
        det = inst_cfg['detector']
        ns, nch, ncol = inst_cfg['num_sub'], inst_cfg['num_ch'], inst_cfg['num_col']
        lvf_params = load_lvf_params(f'lvf_params_D{det}.npy', input_dir=inst_cfg.get('lvf_dir'))
        det_BC, det_BW = load_calibration(
            band=det, calibration_dir=inst_cfg.get('calib_dir'))     # None: $SELFCAL_SPHEREX_CALIB_DIR, else the default
        grid_chunk_map, _, _, _ = make_stripped_chunk_map(
            det, num_subchannels=ns, num_channels=nch, num_columns=ncol,
            oversample_factor=oversample, lvf_params=lvf_params)
        det_chunk_map, _, r_edges, x_edges = make_stripped_chunk_map(
            det, num_subchannels=ns, num_channels=nch, num_columns=ncol,
            oversample_factor=1, lvf_params=lvf_params)
        n_sub = (int(det_chunk_map.max()) + 1) // int(ncol)      # == ns * nch + 2 (padding subchannels)
        # The chunk encoding chunk = subchannel * num_col + column, expressed
        # ONCE as chunk axes: subchannels change along y (arcs), columns along x.
        sub_axes = ChunkAxes.row_major(('subchannel', 'column'), (n_sub, int(ncol)), ('y', 'x'))
        stripped = ChunkMap(name='subchannel', det=det_chunk_map, grid=grid_chunk_map, axes=sub_axes,
                            adjacency_axes=('column',), spectral_axis='subchannel', group_axis='column')
        det_ro, n_ro = make_readout_chunk_map(det_chunk_map.shape)
        readout = ChunkMap(name='readout', det=det_ro, grid=upsample_chunk_map(det_ro, oversample),
                           axes=ChunkAxes.row_major(('readout',), (n_ro,), ('x',)))
        return DetectorGeometry(
            shape=det_chunk_map.shape, chunk_maps={'subchannel': stripped, 'readout': readout},
            primary='subchannel', aux={'BC': det_BC, 'BW': det_BW}, wavelength_key='BC', width_key='BW',
            extra={'lvf_params': lvf_params, 'r_edges': r_edges, 'x_edges': x_edges,
                   'num_sub': ns, 'num_ch': nch, 'num_col': ncol})

    # ---- per-job geometry (valid masks + edge-distance weights) ------------
    def job_geometry(self, inst_cfg, geom, job):
        """Return the valid masks and the solve and mosaic weights of one job's subchannels.

        A ``'window'`` job (``value = (lo, hi)``) selects subchannels ``lo`` to
        ``hi - 1``, indexed over all ``num_sub * num_ch + 2`` subchannels of the
        stripped map (``0`` and the last are the padding subchannels). A
        ``'channels'`` job selects the ``num_sub`` subchannels of each listed
        channel (numbered from 1), and its padded set adds one subchannel on each
        side, which overlaps the neighbouring channels for stitching; a window
        job's padded and strict sets are the same. A selected subchannel is
        selected in every column. Any other ``kind`` raises ``ValueError``.

        ``det_valid_weight``, the solve's weight, is the padded set's 0/1 mask on
        the detector grid. The mosaic's ``grid_valid_weight`` (on the primary
        map's ``grid``) tapers the strict set linearly with each pixel's distance
        in rows to the set's edge
        (:func:`~selfcal.instruments.spherex.spherex_utility.fast_vertical_dist`),
        divided by its maximum. ``chunk_valid`` / ``chunk_valid_strict`` hold the
        padded / strict sets per chunk of the primary map, and ``det_valid_mask``
        / ``grid_valid_mask`` the strict set on the two grids."""
        ns, nch, ncol = inst_cfg['num_sub'], inst_cfg['num_ch'], inst_cfg['num_col']
        det_chunk_map = geom.chunk_map.det
        grid_chunk_map = geom.chunk_map.grid
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
        det_valid_mask_padded = cvm_pad[det_chunk_map]
        grid_valid_mask = cvm[grid_chunk_map]
        grid_valid_weight = fast_vertical_dist(grid_valid_mask)
        if np.max(grid_valid_weight) > 0:
            grid_valid_weight /= np.max(grid_valid_weight)
        # The solve weights every pixel of the padded window equally (the
        # padding subchannels overlap the neighbouring jobs for stitching); the
        # mosaic tapers with the distance to the window's arc edges.
        return JobGeometry(det_valid_weight=det_valid_mask_padded, grid_valid_weight=grid_valid_weight,
                           chunk_valid=cvm_pad, chunk_valid_strict=cvm,
                           det_valid_mask=det_valid_mask, grid_valid_mask=grid_valid_mask)

    # ---- mosaic hooks ----------------------------------------------------------
    def offset_renderer(self, inst_cfg, geom, jobgeom, map_name=None, render=None):
        """Smooth subchannel-arc offset renderer for the mosaic (per job) — for
        the stripped map; the readout map renders block-constant."""
        if map_name not in (None, geom.primary):
            return None
        ns, nch, ncol = inst_cfg['num_sub'], inst_cfg['num_ch'], inst_cfg['num_col']
        x = geom.extra
        return partial(
            make_spherex_stripped_offset_map,
            chunk_valid_mask=jobgeom.chunk_valid_strict,
            lvf_params=x['lvf_params'], r_edges=x['r_edges'], x_edges=x['x_edges'],
            tot_subchannels=ns * nch + 2, num_columns=ncol, fill_invalid=True)

    def aux_coadds(self, geom):
        """(band centre, band width) LVF maps, coadded by the mosaic's sigma-clip pass."""
        return geom.aux['BC'], geom.aux['BW']

    def finalize_mosaic(self, geom, mm, maps, sigma):
        """LVF wavelength maps for the full mosaic. When ``make_mosaic`` was given
        the band maps (``wav_maps=self.aux_coadds(...)``) it has already coadded
        them inside the sigma-clip pass and this only labels the units;
        otherwise the standalone ``wav_coadd`` runs over the intermediate cache
        (the pre-2026-09 path, which needs ``cache_intermediate``)."""
        if 'wav_mean_map' in maps and maps['wav_mean_map'].get('data') is not None:
            for k in ('wav_mean_map', 'wav_std_map'):
                mm.maps[k]['unit'] = 'um'
            return
        import time
        logger.info("Coadding wavelength maps...")
        t00 = time.time()
        wav_mean, wav_std = wav_coadd(
            geom.aux['BC'], geom.aux['BW'],
            mean_map=maps['mean_map']['data'], std_map=maps['std_map']['data'],
            reproj_list=mm.reproj_list, cache_list=mm.cached_list,
            ref_shape=maps['mean_map']['data'].shape, sigma=sigma,
            batch_size=40, max_workers=30)
        logger.info(f"Wavelength coaddition finished in {time.time() - t00:.2f} seconds.")
        mm.append_maps({'wav_mean_map': {'data': wav_mean, 'unit': 'um'},
                        'wav_std_map': {'data': wav_std, 'unit': 'um'}})

    def data_unit(self, inst_cfg):
        """Return ``'MJy/sr'``, the surface-brightness unit of SPHEREx data, for any ``inst_cfg``.

        The mosaic writes it as the ``BUNIT`` of its mean, std and sigma-clipped
        mean maps."""
        return 'MJy/sr'

    # ---- named coefficients -------------------------------------------------
    def coefficient_catalog(self):
        """Return the catalogue of named SPHEREx coefficients; its one entry is ``pah_3p29``.

        ``pah_3p29``
        (:func:`~selfcal.instruments.spherex.line_catalog.pah_3p29_coefficient`)
        is a Gaussian of the band-centre map ``BC`` at the PAH 3.29 um feature,
        whose per-observation width combines the band-width map ``BW`` with the
        intrinsic PAH width. A ``[model]`` term selects it with
        ``coefficient = { catalog = "pah_3p29" }``, optionally overriding the
        factory's ``center`` and ``sigma`` (um); the spectral modes select an
        entry with ``[params].line`` (default ``pah_3p29``) when they are given
        no ``lines`` or ``line_template_npz``. Each call returns a new copy of
        :data:`~selfcal.instruments.spherex.line_catalog.CATALOG`."""
        from .line_catalog import CATALOG
        return dict(CATALOG)

    # ---- post-calibration hooks ------------------------------------------------
    def postcal_hooks(self, cfg):
        """The optional zodi anchor (non-mutating; records into the per-detector
        anchor file). Active only when the run config's ``[zodi].pred_dir`` is set."""
        if cfg.zodi.get('pred_dir'):
            return [zodi_anchor_hook]
        return []

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
        calib_dir = inst_cfg.get('calib_dir')               # None: $SELFCAL_SPHEREX_CALIB_DIR, else the default
        for det in inst_cfg['detectors']:
            det_BC, _ = load_calibration(band=det, calibration_dir=calib_dir)
            _, lvf_params, _ = make_fiducial_chunk_map(
                det, det_BC, num_subchannels=ns, num_channels=nch, oversample_factor=1)
            lvf_params['filename'] = f'lvf_params_D{det}.npy'
            save_lvf_params(lvf_params, output_dir=out_dir)


def zodi_anchor_hook(ctx, job, cal_path, mosaic_path):
    """Post-cal zodi anchor for a single-channel job: fit the cal's frame
    scalars against the zodipy prediction ``zodi_pred_<stem>.npz`` in
    ``[zodi].pred_dir`` and record the channel in ``<run>/zodi_anchor/anchor_D<n>.h5``."""
    from ...zodi_anchor import fit_anchor_for_channel, append_anchor_channel
    cfg = ctx.cfg
    detector = cfg.instrument_cfg['detector']
    job_tag, cal_file = ctx.stem(job), ctx.cal_file(job)
    z = cfg.zodi
    npz_path = os.path.join(z['pred_dir'], f'zodi_pred_{job_tag}.npz')
    m = re.search(r'_Ch(\d+)_', cal_file)
    if not os.path.exists(npz_path):
        print(f"Zodi anchor skipped for {job_tag}: {npz_path} not found.")
        return
    if m is None:
        print(f"Zodi anchor skipped for {job_tag}: not a single-channel job "
              f"(cannot parse _Ch<n>_ from {cal_file}).")
        return
    ch_int = int(m.group(1))
    clip_defaults = dict(clip_window_days=z.get('clip_window_days', 7.0),
                         clip_sigma=z.get('clip_sigma', 3.0),
                         clip_iters=z.get('clip_iters', 2))
    print(f"Fitting zodi anchor from {npz_path}...")
    fit = fit_anchor_for_channel(cal_path, npz_path, **clip_defaults)
    run_dir = os.path.dirname(ctx.pipeline_config.cal_dir.rstrip('/'))
    anchor_path = os.path.join(run_dir, 'zodi_anchor', f'anchor_D{detector}.h5')
    append_anchor_channel(anchor_path, detector, ctx.pipeline_config.run_name, ch_int,
                          fit, clip_defaults, anchor_method='raw')
    print(f"  Ch{ch_int}: C={fit['intercept']:.4g} MJy/sr, slope={fit['slope']:.4f}, "
          f"r={fit['pearson_r']:.4f}, inliers={fit['n_inliers']}/"
          f"{fit['n_inliers']+fit['n_outliers']}  -> {anchor_path}")
