"""The tasks of the run engine — instrument-agnostic.

Each task reads only a :class:`~selfcal.run.runspec.RunSpec`, resolved once into a
:class:`~.engine.RunContext` (the action's own, from its plan, or built here), and composes the
two primitives of :mod:`.engine` (``solve_job``, ``mosaic_job``):

    cal        per job: solve (skipped when the cal exists) + optional mosaic;
               with tiles: per tile solve + Fisher stitch (no mosaic)
    mosaic     per job: mosaic of an existing cal
    npass      the N-pass alternating solve (INIT = a ``cal`` run) — see npass.py
    reproject  raw exposures -> reprojected frames + ref.fits, read the way the
               instrument's exposure layout says

The names here never mention a telescope: the instrument turns its settings into geometry and
an exposure layout, the model into the solver's objects, and this file just sequences
staging -> solve -> save -> mosaic -> cleanup.

Edits must keep calibration output byte-identical: run the gate set
(``selfcal_scripts/gates/run_gates.sh`` and ``run_m13_gate.sh``; see the
README there).
"""
import gc
import glob as glob_module
import os
import time

from . import staging
from .engine import (
    CalResult,
    RunContext,
    announce,
    frame_list,
    mosaic_job,
    solve_job,
    stage_run,
    tile_assignment,
    unstage_run,
)


# ---------------------------------------------------------------------------
# task = 'cal'
# ---------------------------------------------------------------------------
def run_calibration(spec, ctx=None):
    """Per-job calibration (+ optional mosaic), or the tiled variant when the run has tiles."""
    ctx = ctx or RunContext.build(spec)
    if spec.tiling is not None:
        return _run_tiled(ctx)
    return _run_plain(ctx)


def _run_plain(ctx):
    spec = ctx.spec
    frame_dir = stage_run(ctx)
    cal_paths, mosaic_paths = [], []
    for job in ctx.jobs():
        t0 = time.time()
        print(f"Processing {job.name} ({ctx.frame_tag})...")
        jobgeom = ctx.job_geometry(job)
        cal_path = ctx.cal_path(job)
        if os.path.exists(cal_path):
            print(f"Calibration file {cal_path} already exists. Skipping calibration.")
        else:
            if spec.frames.files:
                frames = [os.path.join(frame_dir, os.path.basename(f)) for f in spec.frames.files]
            else:
                frames = frame_list(frame_dir, spec.frames.first_n) if spec.frames.first_n else None
            cal_path = solve_job(ctx, job, jobgeom, frame_dir=frame_dir, frames=frames,
                                 cal_file=ctx.cal_file(job),
                                 hdd_reproj_dir=ctx.pipeline_config.reproj_dir)
            announce(spec, 'cal', cal_path, job=job,
                     frames=frames if frames is not None else frame_list(frame_dir))
        if spec.make_mosaic:
            mos_path = ctx.mosaic_path(job)
            if spec.reuse_mosaics and os.path.exists(mos_path):
                print(f"Mosaic {mos_path} exists and is current. Skipping the coadd.")
            else:
                mos_path = mosaic_job(
                    ctx, job, jobgeom, cal_path=cal_path, frame_dir=frame_dir,
                    mos_file=ctx.mosaic_file(job), cache_dir=ctx.mosaic_cache_dir(job))
                announce(spec, 'mosaic', mos_path, job=job, cal=cal_path, frame_dir=frame_dir)
            mosaic_paths.append(mos_path)
        cal_paths.append(cal_path)
        gc.collect()
        print(f"Finished {job.name} ({ctx.frame_tag}) in {time.time() - t0:.2f} seconds.")
        print("-" * 50 + "\n")
    unstage_run(ctx, frame_dir)
    return CalResult(cal_paths=cal_paths, mosaic_paths=mosaic_paths)


# ---------------------------------------------------------------------------
# task = 'cal' with tiles: stage + solve each tile, then Fisher-stitch
# ---------------------------------------------------------------------------
def _run_tiled(ctx):
    spec = ctx.spec
    t = spec.tiling
    io_limit = spec.frames.io_limit
    if not spec.scratch:
        raise ValueError("a tiled run stages each tile's frames in its scratch area, which it has none of")
    staging.set_hdd_io_limit(io_limit)
    if t.memory_guard:
        staging.start_rss_guardrail()
        staging.rss_checkpoint('startup')

    # Two tiling modes:
    #  - a uniform grid: `grid` = (n_y, n_x) with `overlap` (make_tile_grid);
    #  - explicit tiles: (name, bbox=(y0, y1, x0, x1)) pairs, arbitrary
    #    and possibly OVERLAPPING. Overlap matters for spectral fits: with
    #    disjoint tiles and assign='center', a pixel near a seam only
    #    receives frames whose footprint center fell on its side, truncating
    #    its per-pixel wavelength coverage and blanking the fit mask there;
    #    overlapping bboxes let seam pixels take frames from both
    #    neighbouring tiles. The Fisher stitch is tile-shape-agnostic, so
    #    overlapping tiles need no special handling.
    # A partial run (`only`) builds the full grid first (so each tile's
    # bbox is correct) and skips the stitch (a single tile is already a full
    # cal-shaped h5 over its region).
    tiled, tiles, only_tiles, assignment = tile_assignment(t)
    if t.tiles is not None:
        print(f"[tiled] {len(tiles)} explicit tiles", flush=True)
    if only_tiles:
        print(f"[tiled] only {list(only_tiles)}: partial run, stitch skipped.", flush=True)
    print("[tiled] tiles:", flush=True)
    for tile in tiles:
        print(f"    {tile.name}: bbox={tile.bbox}", flush=True)
    n_all = len(tiled.reproj_files)
    print(f"[tiled] {n_all} reproj files in {t.frames_dir}", flush=True)
    for tile in tiles:
        files, _ = assignment[tile.name]
        print(f"[tiled] {tile.name}: {len(files)} frames "
              f"({100*len(files)/n_all:.1f}% of {n_all})", flush=True)

    nvme = staging.claim(t.stage_dir, t.frames_dir)
    results = {}
    for job in ctx.jobs():
        jobgeom = ctx.job_geometry(job)

        def run_tile(tile, files):
            cal_file = ctx.tile_cal_file(job, tile)
            cal_path = os.path.join(ctx.pipeline_config.cal_dir, cal_file)
            if os.path.exists(cal_path):
                print(f"[tiled] [{tile.name}] cal exists, skipping: {cal_path}", flush=True)
                return cal_path
            t0 = time.time()
            print(f"\n[tiled] === {tile.name}: staging {len(files)} frames -> NVMe ===", flush=True)
            with staging.hdd_throttle(io_limit):
                staging.stage_files(files, nvme, io_limit)
            frames = sorted(os.path.join(nvme, os.path.basename(f)) for f in files)
            cal_path = solve_job(
                ctx, job, jobgeom, frame_dir=nvme, frames=frames, cal_file=cal_file,
                hdd_reproj_dir=ctx.pipeline_config.reproj_dir,
                checkpoint=lambda label: staging.rss_checkpoint(f'{tile.name} {label}'))
            announce(spec, 'cal', cal_path, job=job, frames=frames, tile=tile)
            print(f"[tiled] === {tile.name} cal saved to {cal_path} ({time.time()-t0:.1f}s) ===",
                  flush=True)
            return cal_path

        tile_cals = tiled.run(run_tile, sequential=True)
        if only_tiles:
            print(f"[tiled] partial run complete ({list(only_tiles)}); per-tile cals: {tile_cals}. "
                  f"Stitch skipped — run every tile to build + stitch them all.", flush=True)
            results[job.name] = CalResult(cal_paths=list(tile_cals.values()), tiles=tile_cals,
                                          stitched=None, assignment=assignment)
            continue
        stitched = ctx.stitched_cal_path(job)
        if os.path.exists(stitched):
            print(f"[tiled] stitched cal exists, skipping stitch: {stitched}", flush=True)
        else:
            print(f"\n[tiled] stitching {len(tile_cals)} tile cals -> {stitched}", flush=True)
            tiled.stitch(tile_cals, stitched, ref_shape=t.ref_shape, line=t.stitch_line)
            announce(spec, 'stitched', stitched, job=job, tiles=tile_cals)
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
def run_mosaic(spec, ctx=None):
    """Run task ``mosaic``: coadd the frames of each job's existing cal file into a mosaic.

    The cal is the job's ``cal_<stem>.h5`` or, when the run names one, ``spec.cal_override`` (the
    same file for every job, e.g. a cal solved on another grid; its frames without a reprojected
    file in this run are dropped). A missing cal raises ``FileNotFoundError``. The frames are
    staged as for task ``cal``, and each mosaic is written as ``mosaic_<stem>.fits`` (replacing an
    existing one unless the run keeps it). Returns a :class:`~.engine.CalResult` with the cal and
    mosaic path of each job.
    """
    ctx = ctx or RunContext.build(spec)
    frame_dir = stage_run(ctx)
    cal_paths, mosaic_paths = [], []
    for job in ctx.jobs():
        t0 = time.time()
        cal_path = spec.cal_override or ctx.cal_path(job)
        if not os.path.exists(cal_path):
            raise FileNotFoundError(f"task 'mosaic' needs the cal file {cal_path}")
        jobgeom = ctx.job_geometry(job)
        mos_path = ctx.mosaic_path(job)
        if spec.reuse_mosaics and os.path.exists(mos_path):
            print(f"Mosaic {mos_path} exists and is current. Skipping the coadd.")
        else:
            print(f"Mosaicking {job.name} ({ctx.frame_tag}) from {cal_path}...")
            mos_path = mosaic_job(
                ctx, job, jobgeom, cal_path=cal_path, frame_dir=frame_dir,
                mos_file=ctx.mosaic_file(job), cache_dir=ctx.mosaic_cache_dir(job))
            announce(spec, 'mosaic', mos_path, job=job, cal=cal_path, frame_dir=frame_dir)
        mosaic_paths.append(mos_path)
        cal_paths.append(cal_path)
        gc.collect()
        print(f"Finished {job.name} ({ctx.frame_tag}) in {time.time() - t0:.2f} seconds.")
    unstage_run(ctx, frame_dir)
    return CalResult(cal_paths=cal_paths, mosaic_paths=mosaic_paths)


# ---------------------------------------------------------------------------
# task = 'reproject': raw exposures -> reprojected frames, per the instrument's layout
# ---------------------------------------------------------------------------
def run_reprojection(spec, ctx=None):
    """Run task ``reproject``: reproject the raw exposures onto the run's reference grid.

    The exposures are the sorted matches of the run's patterns, read as the instrument's
    :meth:`~selfcal.instruments.contract.Instrument.layout` says. An instrument with a header
    filter drops exposures first (SPHEREx keeps ``FINAST == 0``), caching the header reads in
    ``<output_dir>/_exposure_cache/<cache_tag>.json``. The reference WCS is the run's ``ref.fits``:
    reused when it exists, otherwise derived from the run's reference file or fitted to the
    exposures, then written. Each (exposure, detector) frame becomes one
    ``exp_<exposure>_det_<detector>.h5`` file in the run's ``reprojected/`` directory; files
    already there are skipped unless the run replaces them, and a verifying run load-tests every
    file afterwards and quarantines broken ones. Returns the ``reprojected/`` directory.
    """
    import numpy as np

    from selfcal.io.exposure_filter import filter_exposures_by_header
    from selfcal.pipeline import pipeline_wrapper

    ctx = ctx or RunContext.build(spec, need_geometry=False)
    r = spec.reproject
    layout = spec.instrument.layout()
    exposure_list = sorted(sum((glob_module.glob(p) for p in r.exposures), []))
    print(f"Globbed {len(exposure_list)} candidate exposures")

    if layout.header_predicate is not None:
        cache = os.path.join(ctx.pipeline_config.output_dir, '_exposure_cache', f'{layout.cache_tag}.json')
        exposure_list, dropped = filter_exposures_by_header(
            exposure_list,
            predicate=layout.header_predicate,
            keys=list(layout.header_keys), ext=layout.header_ext, cache_path=cache, max_workers=16)
        print(f"Kept {len(exposure_list)} exposures, dropped {len(dropped)} by the instrument's "
              f"header filter")

    rr = pipeline_wrapper.Reprojector(ctx.pipeline_config, exposure_list=exposure_list)
    rr.define_reference(padding_pixels=r.padding, use_ext=list(layout.ref_use_ext),
                        source_ref_path=r.reference, reader=layout.reader)
    print(f"Running reprojection with max_workers={r.workers}, reproject_kwargs.parallel=1")
    rr.run_reproject(max_workers=r.workers,
                     reproj_func=r.method,
                     padding_percentage=r.padding_fraction,
                     sci_ext_list=list(layout.sci_ext),
                     dq_ext_list=None if layout.dq_ext is None else list(layout.dq_ext),   # None: every pixel valid
                     exp_idx_list=np.arange(0, len(exposure_list)),
                     det_idx_list=list(layout.detector_ids),
                     replace_existing=r.replace,
                     reproject_kwargs={'parallel': 1},
                     reader=layout.reader)
    if r.verify:
        # Load-test every frame; broken ones are quarantined and logged.
        rr.check_reproj_files(quarantine=True)
        rr.get_reproj_files()
    rr.status()
    print("Reprojection complete")
    return ctx.pipeline_config.reproj_dir


# ---------------------------------------------------------------------------
# task = 'npass': the N-pass alternating solve (scheduler in npass.py)
# ---------------------------------------------------------------------------
def run_npass(spec, ctx=None):
    """Run task ``npass``: the N-pass alternating solve, with :func:`run_calibration` as pass 1.

    Delegates to :func:`.npass.run_npass` and returns its dict: ``products`` (pass number to
    product: the list of pass-1 cals, then one file per pass), ``final`` (the latest sky product)
    and, unless ``n = 1``, ``monitor`` (the path of the per-pass monitor JSON).
    """
    from .npass import run_npass as _run
    return _run(spec, ctx, run_calibration=run_calibration)


_TASKS = {
    'cal': run_calibration,
    'mosaic': run_mosaic,
    'npass': run_npass,
    'reproject': run_reprojection,
}


def run(spec, ctx=None):
    """Run the task of ``spec`` (a :class:`~selfcal.run.runspec.RunSpec`) and return its result.

    - ``cal``: :func:`run_calibration`, a :class:`~.engine.CalResult`;
    - ``mosaic``: :func:`run_mosaic`, a :class:`~.engine.CalResult`;
    - ``npass``: :func:`run_npass`, the scheduler's dict of products;
    - ``reproject``: :func:`run_reprojection`, the reprojected-frame directory.

    ``ctx``: the run's :class:`~.engine.RunContext` when the action's plan built it (built here
    otherwise). Any other task raises ``ValueError``.
    """
    if spec.task not in _TASKS:
        raise ValueError(f"unknown task {spec.task!r}; known: {sorted(_TASKS)}")
    return _TASKS[spec.task](spec, ctx)
