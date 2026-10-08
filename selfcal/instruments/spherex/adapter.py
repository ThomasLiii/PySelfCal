"""The ``spherex`` instrument of the TOML run configs.

Everything SPHEREx-LVF-specific of a run (the stripped (arc) chunk map and its ``(subchannel,
column)`` chunk axes, the H2RG readout-channel map, the BC/BW wavelength maps, a job's subchannel
masks, the smooth arc offset renderer, the LVF wavelength coadd, the FINAST astrometry filter) is
the contract of :class:`~selfcal.instruments.spherex.settings.SPHEREx`. The registered
:class:`SPHERExInstrument` reads a config's ``[instrument]`` table as those settings
(:meth:`~selfcal.instruments.spherex.settings.SPHEREx.from_table`, the job selectors with
:meth:`~selfcal.instruments.spherex.settings.SPHEREx.jobs_from_table`) and calls their methods.
Here are the named subchannel windows and the readout-channel map the settings use, and what
only a TOML config has: the post-calibration zodi anchor (``[zodi]``) and the ``precompute``
task. See :mod:`selfcal.instruments.base` for the engine's interface.

Methods take plain dicts/args (an ``inst_cfg`` mapping = the TOML ``[instrument]``
table), not the runner's RunConfig, so the package stays independent of the runner.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass

import numpy as np

from ..base import Instrument, register_instrument
from ..base import Job as _Job
from .spherex_utility import load_calibration

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


def _settings(inst_cfg):
    """The ``[instrument]`` table as :class:`~selfcal.instruments.spherex.settings.SPHEREx` settings."""
    from .settings import SPHEREx  # loaded on first use (see the package docstring)
    return SPHEREx.from_table(inst_cfg)


@register_instrument('spherex')
class SPHERExInstrument(Instrument):
    """SPHEREx (LVF) instrument: the ``[instrument]`` table read as
    :class:`~selfcal.instruments.spherex.settings.SPHEREx` settings, whose methods do the work.
    ``capabilities`` (those of the settings) lets modes that need LVF features (per-pixel
    wavelength, a spectral chunk axis) declare a requirement that is checked against the
    instrument."""

    @property
    def capabilities(self):
        from .settings import SPHEREx
        return SPHEREx.capabilities

    # ---- jobs (the channel loop) -------------------------------------------
    def jobs(self, inst_cfg):
        """Expand the [instrument] selection keys into a list of Job.
        Exactly one of: windows (named presets) / subch_window (+window_name) /
        channels / channel_range
        (:meth:`~selfcal.instruments.spherex.settings.SPHEREx.jobs_from_table`)."""
        from .settings import SPHEREx
        return SPHEREx.jobs_from_table(inst_cfg)

    # ---- frame tag (cal/mosaic filename component) -------------------------
    def frame_tag(self, inst_cfg):
        """Return the product-name tag ``Detector<d>_NumSub<s>_NumCh<c>_NumCol<k>``.

        The numbers are the required ``[instrument]`` keys ``detector``,
        ``num_sub``, ``num_ch`` and ``num_col``, e.g.
        ``Detector4_NumSub10_NumCh34_NumCol10``. A job's products are named
        ``cal_<tag>_<job><suffix>.h5`` and ``mosaic_<tag>_<job><suffix>.fits``
        (:class:`~selfcal.run.engine.RunContext`), where ``<job>`` is
        the job name (``Ch17``, ``Aromatic``, ...)."""
        return _settings(inst_cfg).product_tag

    # ---- raw exposures (reproject task) ------------------------------------
    def exposure_layout(self, inst_cfg):
        """L2b exposures: science in extension 1, DQ bitmask in extension 2, one
        detector per file; keep only exposures with a converged astrometric
        solution (``FINAST == 0``). Only ``detector`` is read: a reprojection
        table needs nothing else."""
        from .settings import SPHEREx
        return SPHEREx(inst_cfg['detector']).layout()

    # ---- detector-level geometry (built once per run) ----------------------
    def detector_geometry(self, inst_cfg, oversample):
        """LVF params, BC/BW, the stripped chunk map at detector + grid
        resolution with its (subchannel, column) axes, the readout-channel map
        (:meth:`~selfcal.instruments.spherex.settings.SPHEREx.geometry`)."""
        return _settings(inst_cfg).geometry(oversample)

    # ---- per-job geometry (valid masks + edge-distance weights) ------------
    def job_geometry(self, inst_cfg, geom, job):
        """The valid masks and the solve and mosaic weights of one job's subchannels
        (:meth:`~selfcal.instruments.spherex.settings.SPHEREx.job_geometry`)."""
        return _settings(inst_cfg).job_geometry(geom, job)

    # ---- mosaic hooks ----------------------------------------------------------
    def offset_renderer(self, inst_cfg, geom, jobgeom, map_name=None, render=None):
        """Smooth subchannel-arc offset renderer for the mosaic (per job) — for
        the stripped map; the readout map renders block-constant."""
        return _settings(inst_cfg).offset_renderer(geom, jobgeom, map_name=map_name, render=render)

    def aux_coadds(self, geom):
        """(band centre, band width) LVF maps, coadded by the mosaic's sigma-clip pass."""
        from .settings import SPHEREx
        return SPHEREx.aux_coadds(geom)

    def finalize_mosaic(self, geom, mm, maps, sigma):
        """LVF wavelength maps for the full mosaic
        (:meth:`~selfcal.instruments.spherex.settings.SPHEREx.finalize_mosaic`)."""
        from .settings import SPHEREx
        SPHEREx.finalize_mosaic(geom, mm, maps, sigma)

    def data_unit(self, inst_cfg):
        """Return ``'MJy/sr'``, the surface-brightness unit of SPHEREx data, for any ``inst_cfg``.

        The mosaic writes it as the ``BUNIT`` of its mean, std and sigma-clipped
        mean maps."""
        from .settings import SPHEREx
        return SPHEREx.unit

    # ---- named coefficients -------------------------------------------------
    def coefficient_catalog(self):
        """Return the catalogue of named SPHEREx coefficients; its one entry is ``pah_3p29``
        (:meth:`~selfcal.instruments.spherex.settings.SPHEREx.coefficient_catalog`)."""
        from .settings import SPHEREx
        return SPHEREx.coefficient_catalog()

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
    from ...zodi_anchor import append_anchor_channel, fit_anchor_for_channel
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
