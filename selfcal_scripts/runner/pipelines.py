"""The tasks of the generic run engine — mode- and instrument-agnostic.

Each task reads only a :class:`RunConfig`, resolved once into a
:class:`~.engine.RunContext`, and composes the two primitives of
:mod:`.engine` (``solve_job``, ``mosaic_job``):

    cal        per job: solve (skipped when the cal exists) + optional mosaic;
               with ``[tiling]``: per tile solve + Fisher stitch (no mosaic)
    mosaic     per job: mosaic of an existing cal
    npass      the N-pass alternating solve (INIT = a ``cal`` run) — see npass.py
    reproject  FITS exposures -> reprojected frames + ref.fits
    precompute the instrument's rarely-run geometry generator

The names here never mention a telescope or a calibration variant: the
instrument turns the config into geometry, the mode turns geometry into the
offset/sky/x0/mosaic recipe, and this file just sequences
staging -> solve -> save -> mosaic -> cleanup.

Edits must keep calibration output byte-identical: run the gate set
(``workspace/unify/scripts/run_gates.sh``, or the ``cache/refactor_gate``
configs through ``run.py`` + ``selfcal_scripts/drivers/diff_cal_h5.py``).
"""
import gc
import glob as glob_module
import os
import re
import time

from . import staging
from .engine import (RunContext, CalResult, solve_job, mosaic_job, stage_run, unstage_run,
                     frame_list, tile_assignment)


# ---------------------------------------------------------------------------
# task = 'cal'
# ---------------------------------------------------------------------------
def run_calibration(cfg):
    """Per-job calibration (+ optional mosaic), or the tiled variant when the
    config carries a ``[tiling]`` table."""
    ctx = RunContext.build(cfg)
    if cfg.tiling:
        return _run_tiled(ctx)
    return _run_plain(ctx)


def _run_plain(ctx):
    cfg = ctx.cfg
    frame_dir = stage_run(ctx)
    cal_paths, mosaic_paths = [], []
    for job in ctx.jobs():
        t0 = time.time()
        print(f"Processing {job.name} for detector {ctx.detector}...")
        ch_inputs = ctx.channel_inputs(job)
        cal_path = ctx.cal_path(job)
        if os.path.exists(cal_path):
            print(f"Calibration file {cal_path} already exists. Skipping calibration.")
        else:
            frames = frame_list(frame_dir, cfg.n_frames) if cfg.n_frames else None
            cal_path = solve_job(ctx, job, ch_inputs, frame_dir=frame_dir, frames=frames,
                                 cal_file=ctx.cal_file(job),
                                 hdd_reproj_dir=ctx.pipeline_config.reproj_dir)
        if not cfg.skip_mosaic and ctx.mode.mosaic_mode != 'none':
            mosaic_paths.append(mosaic_job(
                ctx, job, ch_inputs, cal_path=cal_path, frame_dir=frame_dir,
                mos_file=ctx.mosaic_file(job), cache_dir=ctx.mosaic_cache_dir(job)))
            if cfg.zodi.get('pred_dir'):
                _run_zodi_anchor(ctx, job, cal_path)
        cal_paths.append(cal_path)
        gc.collect()
        print(f"Finished {job.name} for detector {ctx.detector} in {time.time() - t0:.2f} seconds.")
        print("-" * 50 + "\n")
    unstage_run(ctx, frame_dir)
    return CalResult(cal_paths=cal_paths, mosaic_paths=mosaic_paths)


def _run_zodi_anchor(ctx, job, cal_path):
    """Optional post-cal zodi anchor (non-mutating; records into the per-detector
    anchor file). Active only when [zodi].pred_dir is set."""
    from selfcal.zodi_anchor import fit_anchor_for_channel, append_anchor_channel
    cfg, detector = ctx.cfg, ctx.detector
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


# ---------------------------------------------------------------------------
# task = 'cal' with [tiling]: stage + solve each tile, then Fisher-stitch
# ---------------------------------------------------------------------------
def _run_tiled(ctx):
    cfg = ctx.cfg
    t = cfg.tiling
    staging.set_hdd_io_limit(cfg.hdd_io_limit)
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
    # A partial run (`only_tiles`) builds the full grid first (so each tile's
    # bbox is correct) and skips the stitch (a single tile is already a full
    # cal-shaped h5 over its region).
    tiled, tiles, only_tiles, assignment = tile_assignment(t, ref_shape)
    if t.get('tiles'):
        print(f"[tiled] {len(tiles)} explicit tiles (from [tiling].tiles)", flush=True)
    if only_tiles:
        print(f"[tiled] only_tiles={only_tiles}: partial run, stitch skipped.", flush=True)
    print("[tiled] tiles:", flush=True)
    for tile in tiles:
        print(f"    {tile.name}: bbox={tile.bbox}", flush=True)
    n_all = len(tiled.reproj_files)
    print(f"[tiled] {n_all} reproj files in {t['full_reproj_dir']}", flush=True)
    for tile in tiles:
        files, _ = assignment[tile.name]
        print(f"[tiled] {tile.name}: {len(files)} frames "
              f"({100*len(files)/n_all:.1f}% of {n_all})", flush=True)

    nvme = ctx.tiling_nvme_dir()
    os.makedirs(nvme, exist_ok=True)
    results = {}
    for job in ctx.jobs():
        ch_inputs = ctx.channel_inputs(job)

        def run_tile(tile, files):
            cal_file = ctx.tile_cal_file(job, tile)
            cal_path = os.path.join(ctx.pipeline_config.cal_dir, cal_file)
            if os.path.exists(cal_path):
                print(f"[tiled] [{tile.name}] cal exists, skipping: {cal_path}", flush=True)
                return cal_path
            t0 = time.time()
            print(f"\n[tiled] === {tile.name}: staging {len(files)} frames -> NVMe ===", flush=True)
            with staging.hdd_throttle(cfg.hdd_io_limit):
                staging.stage_files(files, nvme, cfg.hdd_io_limit)
            frames = sorted(os.path.join(nvme, os.path.basename(f)) for f in files)
            cal_path = solve_job(
                ctx, job, ch_inputs, frame_dir=nvme, frames=frames, cal_file=cal_file,
                hdd_reproj_dir=ctx.pipeline_config.reproj_dir,
                checkpoint=lambda label: staging.rss_checkpoint(f'{tile.name} {label}'))
            print(f"[tiled] === {tile.name} cal saved to {cal_path} ({time.time()-t0:.1f}s) ===",
                  flush=True)
            return cal_path

        tile_cals = tiled.run(run_tile, sequential=True)
        if only_tiles:
            print(f"[tiled] partial run complete ({only_tiles}); per-tile cals: {tile_cals}. "
                  f"Stitch skipped — re-run without only_tiles to build + stitch all tiles.",
                  flush=True)
            results[job.name] = CalResult(cal_paths=list(tile_cals.values()), tiles=tile_cals,
                                          stitched=None, assignment=assignment)
            continue
        stitched = ctx.stitched_cal_path(job)
        if os.path.exists(stitched):
            print(f"[tiled] stitched cal exists, skipping stitch: {stitched}", flush=True)
        else:
            print(f"\n[tiled] stitching {len(tile_cals)} tile cals -> {stitched}", flush=True)
            tiled.stitch(tile_cals, stitched, ref_shape=ref_shape, line=t.get('line', True))
        print(f"[tiled] DONE. stitched cal: {stitched}", flush=True)
        results[job.name] = CalResult(cal_paths=list(tile_cals.values()), tiles=tile_cals,
                                      stitched=stitched, assignment=assignment)
    if len(results) == 1:
        return next(iter(results.values()))
    return CalResult(cal_paths=[p for r in results.values() for p in r.cal_paths],
                     tiles={f'{j}/{n}': p for j, r in results.items() for n, p in r.tiles.items()},
                     stitched=None, assignment=assignment)


# ---------------------------------------------------------------------------
# task = 'mosaic': mosaic of an existing cal, per job
# ---------------------------------------------------------------------------
def run_mosaic(cfg):
    ctx = RunContext.build(cfg)
    frame_dir = stage_run(ctx)
    cal_paths, mosaic_paths = [], []
    for job in ctx.jobs():
        t0 = time.time()
        cal_path = ctx.cal_path(job)
        if not os.path.exists(cal_path):
            raise FileNotFoundError(f"task 'mosaic' needs the cal file {cal_path}")
        ch_inputs = ctx.channel_inputs(job)
        print(f"Mosaicking {job.name} for detector {ctx.detector} from {cal_path}...")
        mosaic_paths.append(mosaic_job(
            ctx, job, ch_inputs, cal_path=cal_path, frame_dir=frame_dir,
            mos_file=ctx.mosaic_file(job), cache_dir=ctx.mosaic_cache_dir(job)))
        if cfg.zodi.get('pred_dir'):
            _run_zodi_anchor(ctx, job, cal_path)
        cal_paths.append(cal_path)
        gc.collect()
        print(f"Finished {job.name} for detector {ctx.detector} in {time.time() - t0:.2f} seconds.")
    unstage_run(ctx, frame_dir)
    return CalResult(cal_paths=cal_paths, mosaic_paths=mosaic_paths)


# ---------------------------------------------------------------------------
# task = 'reproject'
# ---------------------------------------------------------------------------
def run_reprojection(cfg):
    import numpy as np
    from selfcal.io.exposure_filter import filter_exposures_by_header
    from selfcal.pipeline import pipeline_wrapper

    ctx = RunContext.build(cfg, need_mode=False, need_geometry=False)
    r = cfg.reproject
    detector = cfg.instrument_cfg['detector']

    file_pattern = r['file_pattern'].format(detector=detector)
    exposure_list = sorted(
        sum((glob_module.glob(d + file_pattern) for d in r['input_dirs']), []))
    print(f"Globbed {len(exposure_list)} candidate exposures")

    finast_cache = os.path.join(
        ctx.pipeline_config.output_dir, '_exposure_cache', f'finast_D{detector}.json')
    exposure_list, dropped = filter_exposures_by_header(
        exposure_list,
        predicate=lambda h: h.get('FINAST', 2) == 0,
        keys=['FINAST'], ext=1, cache_path=finast_cache,
        max_workers=r.get('header_filter_workers', 16))
    print(f"Kept {len(exposure_list)} exposures, dropped {len(dropped)} for poor astrometry")

    rr = pipeline_wrapper.Reprojector(ctx.pipeline_config, exposure_list=exposure_list)
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
    return ctx.pipeline_config.reproj_dir


# ---------------------------------------------------------------------------
# task = 'precompute': the instrument's rarely-run geometry generator
# ---------------------------------------------------------------------------
def run_precompute(cfg):
    from .config import get_instrument
    inst = get_instrument(cfg.instrument)
    inst.precompute(cfg.instrument_cfg)


# ---------------------------------------------------------------------------
# task = 'npass': the N-pass alternating solve (scheduler in npass.py)
# ---------------------------------------------------------------------------
def run_npass(cfg):
    from .npass import run_npass as _run
    return _run(cfg, run_calibration=run_calibration)


_TASKS = {
    'cal': run_calibration,
    'mosaic': run_mosaic,
    'npass': run_npass,
    'reproject': run_reprojection,
    'precompute': run_precompute,
}


def run(cfg):
    if cfg.task not in _TASKS:
        raise ValueError(f"unknown task {cfg.task!r}; known: {sorted(_TASKS)}")
    return _TASKS[cfg.task](cfg)
