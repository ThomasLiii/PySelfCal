"""The generic run engine — mode- and instrument-agnostic.

Each ``run_*`` function reads only a :class:`RunConfig`, an
:class:`~selfcal.instruments.base.Instrument`, and (for cal/tiled) a
:class:`~selfcal_scripts.runner.modes.base.CalMode`. The names here never mention
a telescope or a calibration variant: the instrument turns the config into
geometry, the mode turns geometry into the offset/sky/x0/mosaic recipe, and this
file just sequences staging → setup_lsqr → apply_lsqr → save → mosaic → cleanup.

Edits to this engine must keep calibration output byte-identical: re-run the
regression configs ``cache/refactor_gate/gate_continuum.toml`` and
``gate_spectral.toml`` through ``run.py`` and diff the resulting ``cal_*.h5``
against ``cache/refactor_gate/goldens/`` with
``selfcal_scripts/drivers/diff_cal_h5.py``. All numeric choices live in the
TOML config and the mode/instrument objects; this file only sequences
staging → setup_lsqr → apply_lsqr → save → mosaic → cleanup.
"""
import gc
import glob as glob_module
import os
import re
import shutil
import time

from selfcal.pipeline import pipeline_wrapper
from selfcal._state import set_hdd_io_limit

from . import staging
from .config import get_instrument, get_postprocess
from .modes import get_mode


def _check_requires(mode, inst):
    missing = [c for c in mode.requires if c not in inst.capabilities]
    if missing:
        raise ValueError(
            f"mode {mode.name!r} requires instrument capabilities {missing} "
            f"that {inst.name!r} does not provide (has {sorted(inst.capabilities)})")


def _make_config(cfg):
    return pipeline_wrapper.PipelineConfig(
        output_dir=cfg.output_dir,
        run_name=cfg.resolved_run_name(),
        resolution_arcsec=cfg.resolution_arcsec)


def _calibration_kwargs(cfg):
    """The [calibration] table + the resolved (named) postprocess func."""
    kw = dict(cfg.calibration)
    kw['postprocess_func'] = get_postprocess(cfg.postprocess)
    return kw


# ---------------------------------------------------------------------------
# Standard per-job calibration (task = 'cal'): one setup_lsqr + apply_lsqr +
# save + optional mosaic per instrument job; the mode picks the recipe
# (continuum / pahfit / k2_readout — see selfcal_scripts/runner/modes/).
# ---------------------------------------------------------------------------
def run_calibration(cfg):
    inst = get_instrument(cfg.instrument)
    mode = get_mode(cfg.mode)
    _check_requires(mode, inst)

    selfcal_config = _make_config(cfg)
    if cfg.reproj_override:
        # Run directly against an already-staged reproj dir — used by the
        # byte-equality regression configs in cache/refactor_gate/ and for
        # manual re-runs; skips NVMe staging and cleanup.
        nvme = cfg.reproj_override
        set_hdd_io_limit(None)
    else:
        nvme = staging.prepare_nvme(cfg, selfcal_config.reproj_dir, selfcal_config.run_name)

    cal_kwargs = _calibration_kwargs(cfg)
    detector = cfg.instrument_cfg.get('detector')
    frame_tag = inst.frame_tag(cfg.instrument_cfg)
    det_inputs = inst.detector_inputs(cfg.instrument_cfg, cfg.oversample)
    cal_paths = []

    for job in inst.jobs(cfg.instrument_cfg):
        t0 = time.time()
        print(f"Processing {job.name} for detector {detector}...")
        job_tag = f'{frame_tag}_{job.name}{cfg.suffix}'
        cal_file = f'cal_{job_tag}.h5'
        mos_file = f'mosaic_{job_tag}.fits'
        cache_dir = f'{cfg.cache_dir}cache_{job_tag}'

        ch_inputs = inst.channel_inputs(cfg.instrument_cfg, det_inputs, job)
        cal_path = os.path.join(selfcal_config.cal_dir, cal_file)

        # ------------------------- Calibration -------------------------
        if os.path.exists(cal_path):
            print(f"Calibration file {cal_path} already exists. Skipping calibration.")
        else:
            cc = pipeline_wrapper.Calibrator(selfcal_config, reproj_dir=nvme)
            if cfg.n_frames:
                cc.reproj_list = sorted(
                    glob_module.glob(os.path.join(nvme, '*.h5')))[:cfg.n_frames]
            n_frames = len(cc.reproj_list)
            offset_model = mode.build_offset_model(cfg, inst, det_inputs, ch_inputs, job, n_frames)
            sky_model = mode.build_sky_model(cfg, inst, det_inputs)
            det_aux = mode.det_aux(cfg, inst, det_inputs)
            cc.setup_lsqr(
                offset_model=offset_model,
                grid_valid_weight=ch_inputs['det_valid_mask_padded'],
                oversample_factor=1,
                sky_model=sky_model,
                det_aux=det_aux,
                batch_spill_dir=cfg.cache_dir,
                **cal_kwargs)
            # List-pop hand-off: keeping a plain `x0` local would pin the
            # full-layout f64 vector for the entire solve (see Calibrator.apply_lsqr).
            _x0_owned = [mode.x0(cfg, cc)]
            cc.apply_lsqr(x0=_x0_owned.pop(), use_float32=True, n_threads=cfg.apply_n_threads, **cfg.lsqr)
            mode.configure(cfg, cc)
            # Save with original HDD paths so the cal stays valid after NVMe cleanup.
            nvme_list = cc.reproj_list
            cc.reproj_list = staging.remap_to_nvme(nvme_list, selfcal_config.reproj_dir)
            cal_path = cc.save_calibration(cal_file=cal_file)
            cc.reproj_list = nvme_list
            del cc

        # ------------------------- Mosaicking --------------------------
        if not cfg.skip_mosaic and mode.mosaic_mode != 'none':
            chunk_maps, det_offset_funcs = mode.mosaic_geometry(cfg, inst, det_inputs, ch_inputs)
            mm = pipeline_wrapper.Mosaicker(selfcal_config, reproj_dir=nvme)
            mm.load_calibration(cal_path=cal_path)
            mm.reproj_list = staging.remap_to_nvme(mm.reproj_list, nvme)
            maps = mm.make_mosaic(
                chunk_maps=chunk_maps,
                grid_valid_weight=ch_inputs['grid_valid_weight'],
                oversample_factor=cfg.oversample,
                det_offset_funcs=det_offset_funcs,
                cache_dir=cache_dir,
                **cfg.mosaic)
            if mode.mosaic_mode == 'full':
                inst.wavelength_append(det_inputs, mm, maps, cfg.mosaic['sigma'])
            mm.save_mosaic(mos_file=mos_file, overwrite=True)
            if cfg.zodi.get('pred_dir'):
                _run_zodi_anchor(cfg, selfcal_config, detector, cal_path, cal_file, job_tag)
            del mm, maps
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)

        cal_paths.append(cal_path)
        gc.collect()
        print(f"Finished {job.name} for detector {detector} in {time.time() - t0:.2f} seconds.")
        print("-" * 50 + "\n")

    if not cfg.reproj_override:
        staging.cleanup_nvme(cfg, nvme)
    return cal_paths


def _run_zodi_anchor(cfg, selfcal_config, detector, cal_path, cal_file, job_tag):
    """Optional post-cal zodi anchor (non-mutating; records into the per-detector
    anchor file). Active only when [zodi].pred_dir is set."""
    from selfcal.zodi_anchor import fit_anchor_for_channel, append_anchor_channel
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
    run_dir = os.path.dirname(selfcal_config.cal_dir.rstrip('/'))
    anchor_path = os.path.join(run_dir, 'zodi_anchor', f'anchor_D{detector}.h5')
    append_anchor_channel(anchor_path, detector, selfcal_config.run_name, ch_int,
                          fit, clip_defaults, anchor_method='raw')
    print(f"  Ch{ch_int}: C={fit['intercept']:.4g} MJy/sr, slope={fit['slope']:.4f}, "
          f"r={fit['pearson_r']:.4f}, inliers={fit['n_inliers']}/"
          f"{fit['n_inliers']+fit['n_outliers']}  -> {anchor_path}")


# ---------------------------------------------------------------------------
# Tiled calibration (task = 'tiled'): stage + solve each tile independently,
# then Fisher-weighted stitch of the per-tile cal files.
# ---------------------------------------------------------------------------
def run_tiled(cfg):
    from selfcal.pipeline.tiled import make_tile_grid, TiledCalibration, TileSpec

    inst = get_instrument(cfg.instrument)
    mode = get_mode(cfg.mode)
    _check_requires(mode, inst)
    t = cfg.tiled

    selfcal_config = _make_config(cfg)
    cal_kwargs = _calibration_kwargs(cfg)
    set_hdd_io_limit(cfg.hdd_io_limit)
    if t.get('rss_guardrail', True):
        staging.start_rss_guardrail()
        staging.rss_checkpoint('startup')

    ref_shape = tuple(t['ref_shape'])
    # Two tiling modes:
    #  - a uniform grid: `grid` = [n_y, n_x] with `overlap_px` (make_tile_grid);
    #  - explicit tiles: `tiles` = list of {name, bbox=[y0,y1,x0,x1]}, arbitrary
    #    and possibly OVERLAPPING. Overlap matters for spectral fits: with
    #    disjoint tiles and frame_filter='center', a pixel near a seam only
    #    receives frames whose footprint center fell on its side, truncating
    #    its per-pixel wavelength coverage and blanking the fit mask there;
    #    overlapping bboxes let seam pixels take frames from both
    #    neighbouring tiles. The Fisher stitch is tile-shape-agnostic, so
    #    overlapping tiles need no special handling.
    if t.get('tiles'):
        tiles = [TileSpec(name=spec['name'], bbox=tuple(spec['bbox']))
                 for spec in t['tiles']]
        print(f"[tiled] {len(tiles)} explicit tiles (from [tiled].tiles)", flush=True)
    else:
        tiles = make_tile_grid(ref_shape, t['grid'][0], t['grid'][1],
                               overlap_px=t['overlap_px'], names=t['tile_names'])
    # Optional partial run: build the full grid (so each tile's bbox is correct),
    # then restrict the run to a subset of tile names. A partial run skips the
    # stitch (a single tile is already a full cal-shaped h5 over its region).
    only_tiles = t.get('only_tiles')
    if only_tiles:
        all_names = [tile.name for tile in tiles]
        tiles = [tile for tile in tiles if tile.name in only_tiles]
        if not tiles:
            raise ValueError(f"only_tiles={only_tiles} matched no tile in {all_names}")
        print(f"[tiled] only_tiles={only_tiles}: partial run, stitch skipped.", flush=True)
    print("[tiled] tiles:", flush=True)
    for tile in tiles:
        print(f"    {tile.name}: bbox={tile.bbox}", flush=True)

    full_reproj_dir = t['full_reproj_dir']
    frame_glob = t.get('frame_glob', 'exp_*_det_00.h5')
    hdd_reproj_files = sorted(
        glob_module.glob(os.path.join(full_reproj_dir, frame_glob)),
        key=lambda p: int(os.path.basename(p).split('_')[1]))
    print(f"[tiled] {len(hdd_reproj_files)} reproj files in {full_reproj_dir}", flush=True)

    det_inputs = inst.detector_inputs(cfg.instrument_cfg, cfg.oversample)
    job = inst.jobs(cfg.instrument_cfg)[0]
    ch_inputs = inst.channel_inputs(cfg.instrument_cfg, det_inputs, job)
    frame_tag = inst.frame_tag(cfg.instrument_cfg)

    tiled = TiledCalibration(hdd_reproj_files, tiles,
                             frame_filter=t.get('frame_filter', 'center'),
                             halo=t.get('halo', 0))
    assignment = tiled.assign_frames()
    for tile in tiles:
        files, _ = assignment[tile.name]
        print(f"[tiled] {tile.name}: {len(files)} frames "
              f"({100*len(files)/len(hdd_reproj_files):.1f}% of {len(hdd_reproj_files)})",
              flush=True)

    nvme = os.path.join(cfg.cache_dir, t['nvme_subdir'])
    os.makedirs(nvme, exist_ok=True)

    def run_tile(tile, files):
        cal_file = f'cal_{frame_tag}_{job.name}{cfg.suffix.format(tile=tile.name)}.h5'
        cal_path = os.path.join(selfcal_config.cal_dir, cal_file)
        if os.path.exists(cal_path):
            print(f"[tiled] [{tile.name}] cal exists, skipping: {cal_path}", flush=True)
            return cal_path
        t0 = time.time()
        print(f"\n[tiled] === {tile.name}: staging {len(files)} frames -> NVMe ===", flush=True)
        set_hdd_io_limit(cfg.hdd_io_limit)
        staging.stage_files(files, nvme, cfg.hdd_io_limit)
        set_hdd_io_limit(None)
        nvme_frame_list = sorted(os.path.join(nvme, os.path.basename(f)) for f in files)

        cc = pipeline_wrapper.Calibrator(selfcal_config, reproj_dir=nvme)
        cc.reproj_list = list(nvme_frame_list)
        n_frames = len(cc.reproj_list)
        offset_model = mode.build_offset_model(cfg, inst, det_inputs, ch_inputs, job, n_frames)
        sky_model = mode.build_sky_model(cfg, inst, det_inputs)
        det_aux = mode.det_aux(cfg, inst, det_inputs)

        staging.rss_checkpoint(f'{tile.name} pre-setup_lsqr')
        cc.setup_lsqr(
            offset_model=offset_model,
            grid_valid_weight=ch_inputs['det_valid_mask_padded'],
            oversample_factor=1,
            sky_model=sky_model,
            det_aux=det_aux,
            batch_spill_dir=cfg.cache_dir,
            **cal_kwargs)
        staging.rss_checkpoint(f'{tile.name} post-setup_lsqr')
        # List-pop hand-off: keeping a plain `x0` local would pin the
        # full-layout f64 vector for the entire solve (see Calibrator.apply_lsqr).
        _x0_owned = [mode.x0(cfg, cc)]
        staging.rss_checkpoint(f'{tile.name} pre-apply_lsqr')
        cc.apply_lsqr(x0=_x0_owned.pop(), use_float32=True, n_threads=cfg.apply_n_threads, **cfg.lsqr)
        staging.rss_checkpoint(f'{tile.name} post-apply_lsqr')
        mode.configure(cfg, cc)

        nvme_list = cc.reproj_list
        cc.reproj_list = staging.remap_to_nvme(nvme_list, selfcal_config.reproj_dir)
        cc.save_calibration(cal_file=cal_file)
        cc.reproj_list = nvme_list
        del cc
        gc.collect()
        print(f"[tiled] === {tile.name} cal saved to {cal_path} ({time.time()-t0:.1f}s) ===",
              flush=True)
        return cal_path

    cal_paths = tiled.run(run_tile, sequential=True)
    if only_tiles:
        print(f"[tiled] partial run complete ({only_tiles}); per-tile cals: {cal_paths}. "
              f"Stitch skipped — re-run without only_tiles to build + stitch all tiles.",
              flush=True)
        return {'tiles': cal_paths, 'stitched': None, 'assignment': assignment}
    stitched = os.path.join(selfcal_config.cal_dir,
                            f'cal_{frame_tag}_{job.name}{t["stitched_suffix"]}.h5')
    if os.path.exists(stitched):
        print(f"[tiled] stitched cal exists, skipping stitch: {stitched}", flush=True)
    else:
        print(f"\n[tiled] stitching {len(cal_paths)} tile cals -> {stitched}", flush=True)
        tiled.stitch(cal_paths, stitched, ref_shape=ref_shape, line=t.get('line', True))
    print(f"[tiled] DONE. stitched cal: {stitched}", flush=True)
    return {'tiles': cal_paths, 'stitched': stitched, 'assignment': assignment}


# ---------------------------------------------------------------------------
# Reprojection (run_reproject).
# ---------------------------------------------------------------------------
def run_reprojection(cfg):
    import numpy as np
    from selfcal.io.exposure_filter import filter_exposures_by_header

    inst = get_instrument(cfg.instrument)
    selfcal_config = _make_config(cfg)
    r = cfg.reproject
    detector = cfg.instrument_cfg['detector']

    file_pattern = r['file_pattern'].format(detector=detector)
    exposure_list = sorted(
        sum((glob_module.glob(d + file_pattern) for d in r['input_dirs']), []))
    print(f"Globbed {len(exposure_list)} candidate exposures")

    finast_cache = os.path.join(
        selfcal_config.output_dir, '_exposure_cache', f'finast_D{detector}.json')
    exposure_list, dropped = filter_exposures_by_header(
        exposure_list,
        predicate=lambda h: h.get('FINAST', 2) == 0,
        keys=['FINAST'], ext=1, cache_path=finast_cache,
        max_workers=r.get('header_filter_workers', 16))
    print(f"Kept {len(exposure_list)} exposures, dropped {len(dropped)} for poor astrometry")

    rr = pipeline_wrapper.Reprojector(selfcal_config, exposure_list=exposure_list)
    rr.define_reference(padding_pixels=r.get('padding_pixels', 100),
                        use_ext=r.get('use_ext', [1]),
                        source_ref_path=r.get('source_ref_path'))

    max_workers = r.get('max_workers', 50)
    inner_parallel = r.get('inner_parallel', 1)
    print(f"Running reprojection with max_workers={max_workers}, "
          f"reproject_kwargs.parallel={inner_parallel}")
    rr.run_reproject(max_workers=max_workers,
                     reproj_func=r.get('reproj_func', 'exact'),
                     padding_percentage=r.get('padding_percentage', 0.05),
                     sci_ext_list=r.get('sci_ext_list', [1]),
                     dq_ext_list=r.get('dq_ext_list', [2]),
                     exp_idx_list=np.arange(0, len(exposure_list)),
                     det_idx_list=[0] * len(exposure_list),
                     replace_existing=r.get('replace_existing', False),
                     reproject_kwargs={'parallel': inner_parallel})
    rr.status()
    print("Reprojection complete")


# ---------------------------------------------------------------------------
# Precompute instrument geometry params (precompute_lvf_params).
# ---------------------------------------------------------------------------
def run_precompute(cfg):
    inst = get_instrument(cfg.instrument)
    inst.precompute(cfg.instrument_cfg)


def run_npass(cfg):
    """N-pass alternating solve (task = 'npass'); see runner/npass.py."""
    from selfcal_scripts.runner.npass import run_npass as _run
    return _run(cfg, run_calibration=run_calibration, run_tiled=run_tiled)


_TASKS = {
    'cal': run_calibration,
    'tiled': run_tiled,
    'npass': run_npass,
    'reproject': run_reprojection,
    'precompute': run_precompute,
}


def run(cfg):
    if cfg.task not in _TASKS:
        raise ValueError(f"unknown task {cfg.task!r}; known: {sorted(_TASKS)}")
    return _TASKS[cfg.task](cfg)
