#!/usr/bin/env python
"""Set up an expansion reprojection run on the Python API: its expanded canvas, and the old run's frames moved onto it.

A tool over a reproject run script, not a run of its own (it has no settings; tests/test_run_scripts.py lists it in
TOOLS). It made the NEP and SEP expansion fields of 2026-10 (from workspace/survey-expansion/scripts/, where the
campaign's chain scripts still call their copy). For a reproject run script (FIELD + REPROJECT, e.g.
selfcal_scripts/runs/reproject_nep_expansion_d2.py) whose REPROJECT['reference'] is an existing run's ref.fits (the
old run):
 1. builds the exposure list exactly as the library's reproject task does (selfcal.run.pipelines.run_reprojection:
    the sorted glob of the patterns, the instrument's header filter with its cache) -- a frame file's name is its
    index in this list;
 2. writes the run's ref.fits with Reprojector.define_reference, the call the task makes (padding, the layout's
    use_ext and reader, source_ref_path = the old ref.fits): the old projection, CRPIX shifted by an integer (dx, dy),
    the canvas sized to the exposures;
 3. copies every frame of the old run into the new run's reprojected/ under the name of ITS index in the new list
    (matched by source FITS path), with ref_coords + (dy, dy, dx, dx) and the shift recorded in the attributes
    canvas_shift_dxdy / canvas_shift_source / canvas_shift_iso (nothing else in a frame depends on the canvas);
 4. writes <run>/expansion_map.csv (new index, new file, source FITS, old file).
Then the reproject run script (its launcher) loads that ref.fits, finds the moved frames and reprojects only the rest.
Resumable: frames already moved with this shift are skipped.

Usage (from the repository root): PYTHONPATH=$PWD python selfcal_scripts/runs/expand_field.py <run script>
[--workers N] [--dry-run]   (--dry-run reads every header and old frame, writes only the header cache)
"""
import argparse
import csv
import datetime
import glob
import importlib.util
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor

import h5py
import hdf5plugin  # noqa: F401  (the frames are Zstd-compressed)
import numpy as np

from selfcal.geometry import wcs_helper
from selfcal.io.exposure_filter import filter_exposures_by_header
from selfcal.io.reproj import reproj_basename
from selfcal.pipeline import pipeline_wrapper
from selfcal.run.engine import RunContext


def source_of(path):
    with h5py.File(path, "r") as f:
        v = f.attrs["file_path"]
    return v.decode() if isinstance(v, bytes) else str(v)


def move(task):
    """Copy one old frame to its new name with ref_coords shifted; 'skip' if already done."""
    import shutil
    old, new, dx, dy, iso = task
    if os.path.exists(new):
        with h5py.File(new, "r") as f:
            done = np.array_equal(f.attrs.get("canvas_shift_dxdy", []), [dx, dy]) and \
                f.attrs.get("canvas_shift_source", "") == old
        if done:
            return "skip"
        raise RuntimeError(f"{new} exists and is not this migration's copy of {old}")
    tmp = new + ".tmp"
    shutil.copyfile(old, tmp)
    with h5py.File(tmp, "r+") as f:
        rc = np.asarray(f.attrs["ref_coords"])
        f.attrs["ref_coords"] = (rc + np.array([dy, dy, dx, dx])).astype(rc.dtype)
        f.attrs["canvas_shift_dxdy"] = np.array([dx, dy], dtype=np.int32)
        f.attrs["canvas_shift_source"] = old
        f.attrs["canvas_shift_iso"] = iso
    os.replace(tmp, new)
    return "ok"


def main(script, workers, dry):
    sp = importlib.util.spec_from_file_location("reproject_run", script)
    mod = importlib.util.module_from_spec(sp)
    sp.loader.exec_module(mod)
    spec = mod.FIELD.reprojection_spec(**mod.REPROJECT)
    ctx = RunContext.build(spec, need_geometry=False)
    pc, r = ctx.pipeline_config, spec.reproject
    layout = spec.instrument.layout()
    old_ref = r.reference
    old_run = os.path.dirname(old_ref)
    run_dir = os.path.dirname(pc.ref_path)

    # 1. the task's exposure list
    exps = sorted(sum((glob.glob(p) for p in r.exposures), []))
    n_glob = len(exps)
    if layout.header_predicate is not None:
        cache = os.path.join(pc.output_dir, '_exposure_cache', f'{layout.cache_tag}.json')
        exps, _ = filter_exposures_by_header(exps, predicate=layout.header_predicate, keys=list(layout.header_keys),
                                             ext=layout.header_ext, cache_path=cache, max_workers=16)
    print(f"run {pc.run_name}: {n_glob} exposures globbed, {len(exps)} kept by the header filter", flush=True)

    # 2. the canvas: the task's own define_reference (writes ref.fits unless dry)
    if dry:
        new_wcs, new_shape = (wcs_helper.load_from_fits(pc.ref_path) if os.path.exists(pc.ref_path) else
                              wcs_helper.derive_reference_from(old_ref, exps, r.padding,
                                                               use_ext=list(layout.ref_use_ext), reader=layout.reader))
    else:
        os.makedirs(run_dir, exist_ok=True)
        rr = pipeline_wrapper.Reprojector(pc, exposure_list=exps)
        rr.define_reference(padding_pixels=r.padding, use_ext=list(layout.ref_use_ext), source_ref_path=old_ref,
                            reader=layout.reader)
        new_wcs, new_shape = rr.ref_wcs, rr.ref_shape
    old_wcs, old_shape = wcs_helper.load_from_fits(old_ref)
    if not wcs_helper.projections_match(new_wcs, old_wcs):
        raise SystemExit("the new canvas does not share the old projection")
    d = new_wcs.wcs.crpix - old_wcs.wcs.crpix
    dx, dy = int(round(d[0])), int(round(d[1]))
    if max(abs(d[0] - dx), abs(d[1] - dy)) > 1e-6:
        raise SystemExit(f"CRPIX shift {d} is not an integer")
    print(f"canvas {new_shape[1]} x {new_shape[0]} (old {old_shape[1]} x {old_shape[0]}); old pixel (x, y) -> new "
          f"(x + {dx}, y + {dy}); old canvas inside the new one: "
          f"{dx >= 0 and dy >= 0 and dx + old_shape[1] <= new_shape[1] and dy + old_shape[0] <= new_shape[0]}"
          f"{'' if dry else '; written ' + pc.ref_path}", flush=True)

    # 3. old frames -> their index in the new list
    olds = sorted(glob.glob(os.path.join(old_run, "reprojected", "*.h5")))
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("forkserver")) as ex:
        srcs = list(ex.map(source_of, olds, chunksize=64))
    idx = {p: i for i, p in enumerate(exps)}
    by_name = {}
    for i, p in enumerate(exps):
        by_name.setdefault(os.path.basename(p), []).append(i)
    rows, missing, by_basename = [], [], 0
    for old, src in zip(olds, srcs):
        i = idx.get(src)
        if i is None:
            cands = by_name.get(os.path.basename(src), [])
            if len(cands) != 1:
                missing.append(src)
                continue
            i, by_basename = cands[0], by_basename + 1
        rows.append((i, reproj_basename(i, 0), src, old))
    news = [row[0] for row in rows]
    print(f"old run {os.path.basename(old_run)}: {len(olds)} frames; matched {len(rows)} ({by_basename} by file name "
          f"only), not in the new list {len(missing)}, duplicate new indices {len(news) - len(set(news))}", flush=True)
    if missing or len(news) != len(set(news)):
        raise SystemExit(f"cannot place every old frame: missing {missing[:3]}")
    moved = sorted(news)
    print(f"new indices of the old frames: {moved[0]}..{moved[-1]}; renamed "
          f"{sum(old.split('/')[-1] != new for _, new, _, old in rows)}; frames left for the reproject task: "
          f"{len(exps) - len(rows)}", flush=True)
    for row in rows[:3] + rows[-3:]:
        print(f"   {os.path.basename(row[3])} -> {row[1]}  ({os.path.basename(row[2])})")
    if dry:
        return

    reproj_dir = pc.reproj_dir
    os.makedirs(reproj_dir, exist_ok=True)
    expected = {name for _, name, _, _ in rows}
    stray = [f for f in os.listdir(reproj_dir) if f.endswith(".h5") and f not in expected]
    if stray:
        raise SystemExit(f"{len(stray)} files in {reproj_dir} that this migration would not write, e.g. {stray[:3]}")
    with open(os.path.join(run_dir, "expansion_map.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["new_index", "new_file", "source_fits", "old_file"])
        w.writerows(rows)
    iso = datetime.datetime.now().isoformat(timespec="seconds")
    tasks = [(old, os.path.join(reproj_dir, name), dx, dy, iso) for _, name, _, old in rows]
    t0, counts = datetime.datetime.now(), {"ok": 0, "skip": 0}
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("forkserver")) as ex:
        for k, res in enumerate(ex.map(move, tasks, chunksize=16), 1):
            counts[res] += 1
            if k % 2000 == 0 or k == len(tasks):
                print(f"   moved {k}/{len(tasks)} ({counts}) in {(datetime.datetime.now() - t0).seconds / 60:.1f} min",
                      flush=True)
    print(f"DONE: {len(tasks)} old frames on the new canvas in {reproj_dir}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("script")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(a.script, a.workers, a.dry_run)
