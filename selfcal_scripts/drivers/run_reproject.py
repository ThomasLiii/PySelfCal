"""Reprojection driver — reads raw SPHEREx FITS exposures and writes one
HDF5 cutout per (exposure, detector) onto a common reference WCS.

Customization (the knobs you'll typically change):
  - `frame_setting['Detector']` — which SPHEREx detector (1..6).
  - `qr1_dir` / `qr2_dir` / `file_pattern` — where to find input exposures.
  - `SOURCE_REF_PATH` — share projection with another run (see below).
  - `selfcal_config.run_name` / `output_dir` — where outputs live.

Outputs land under
``{output_dir}/{run_name}/`` with subdirs ``reprojected/``, plus a
``ref.fits`` and a ``reprojected/manifest.json`` describing the run.

Features (set up so you can edit, run, Ctrl-C, and re-run safely):

1. **Resume after Ctrl-C / crash.** `run_reproject` writes
   ``reprojected/manifest.json`` listing every intended (exposure,
   extension) task. On every invocation it scans the output dir and
   dispatches only tasks whose final h5 doesn't exist. Workers write to
   ``output.h5.tmp.<pid>`` then ``os.replace`` — a kill never leaves
   half-written files. **To force redo, pass ``replace_existing=True`` or
   delete the specific h5 you want regenerated.** See
   ``Reprojector.status()`` for a snapshot.

2. **FINAST header-filter cache.** The slow ``fits.open`` loop that
   rejects poor-astrometry exposures is cached at
   ``{output_dir}/_exposure_cache/finast_D{N}.json`` keyed on
   ``(path, mtime, ext, keys)``. **Identical reruns skip every FITS
   header open.** Delete the json to force re-read.

3. **Cross-run projection sharing.** Set ``SOURCE_REF_PATH`` to point at
   another run's ``ref.fits``. The new run will load that file's
   projection (CRVAL/CDELT/CTYPE/PC) and recompute only CRPIX + shape to
   fit *this* run's exposure footprint. Useful when you want detectors /
   epochs / cuts that share the same pixel grid. Set to ``None`` to fit
   an optimal frame from scratch.

   Sanity-check: if a ``ref.fits`` already exists at this run's
   ``ref_path``, ``define_reference`` will assert it has the same
   projection as ``SOURCE_REF_PATH`` (raises if mismatched). Pass
   ``verify_projection=False`` to skip.

4. **Failure log.** Worker errors append to
   ``reprojected/failed.jsonl`` as JSON-per-line records with
   ``exp_idx``, ``det_idx``, ``input_fits``, ``error``. Investigate /
   quarantine via ``Reprojector.check_reproj_files(quarantine=True)``.

5. **Determinism.** Worker outputs are sorted before being returned,
   and ``Reprojector.reproj_list`` is always the sorted union of
   pre-existing and newly-completed outputs.

Drivers under selfcal_scripts/drivers/ pin
``OMP/MKL/OPENBLAS_NUM_THREADS=1`` so only the in-process pool is
parallel. Preserve that when adding launchers.

Quick recipes:
  - **First time on a new detector:** edit ``frame_setting['Detector']``
    and run. Builds a fresh ref.fits.
  - **Resume after Ctrl-C:** just re-run; pending tasks dispatch only.
  - **Force re-reproject everything:** pass ``replace_existing=True`` to
    ``rr.run_reproject(...)``, OR delete ``reprojected/`` first.
  - **New detector that shares grid with an existing run:** set
    ``SOURCE_REF_PATH`` to that run's ``ref.fits`` and leave the local
    ``ref.fits`` absent — ``define_reference`` will derive a new one
    sized to this footprint.
  - **Audit a half-run:** ``Reprojector(cfg).status()`` (no need to
    re-glob exposures).
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import glob
import numpy as np

from SelfCal import PipelineWrapper
from SelfCal.exposure_filter import filter_exposures_by_header

frame_setting = {
    'Detector': 4,
    'NumSub': 10,
    'NumCh': 34,
    'NumCol': 3,
}
selfcal_config = PipelineWrapper.PipelineConfig(
    output_dir='/mnt/md124/thomasli/selfcal/outputs/',
    run_name=f'SPHEREx_NEP_2026W17_D{frame_setting["Detector"]}_6p2arcsec',
    resolution_arcsec=6.2
)

# Optional: point a new run at an existing ref.fits to share the projection
# (same CRVAL/CDELT/CTYPE/PC) across runs that cover different areas around
# the same center. The new run's ref_shape + CRPIX are computed from its
# own exposure list. Leave None to fit an optimal frame from scratch.
SOURCE_REF_PATH = '/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D5_6p2arcsec/ref.fits'
# e.g. '/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D5_6p2arcsec/ref.fits'

qr1_dir = '/mnt/md124/SPHEREx/SPHEREx_nep_data/qr1_newgain'
qr2_dir = '/mnt/md124/SPHEREx/SPHEREx_nep_data/qr2'
file_pattern = f'/*/*/*/*D{frame_setting["Detector"]}*.fits'

exposure_list = sorted(glob.glob(qr1_dir + file_pattern) +
                       glob.glob(qr2_dir + file_pattern))
print(f"Globbed {len(exposure_list)} candidate exposures")

# Drop poor-astrometry exposures (FINAST != 0). The (path, mtime)-keyed
# cache reuses prior header reads on identical reruns, killing the slow
# per-file fits.open loop. Cache lives alongside outputs; safe to delete.
finast_cache = os.path.join(
    selfcal_config.output_dir, '_exposure_cache',
    f'finast_D{frame_setting["Detector"]}.json')
exposure_list, dropped = filter_exposures_by_header(
    exposure_list,
    predicate=lambda h: h.get('FINAST', 2) == 0,
    keys=['FINAST'],
    ext=1,
    cache_path=finast_cache,
    max_workers=16,
)
print(f"Kept {len(exposure_list)} exposures, dropped {len(dropped)} for poor astrometry")

# Initialize Reprojector and run reprojection
rr = PipelineWrapper.Reprojector(selfcal_config, exposure_list=exposure_list)
rr.define_reference(padding_pixels=100, use_ext=[1],
                    source_ref_path=SOURCE_REF_PATH)

# The reproject library can also parallelize internally; keep one level of
# parallelism to avoid oversubscription and high kernel/system CPU time.
reproject_max_workers = 50
reproject_inner_parallel = 1
print(
    f"Running reprojection with max_workers={reproject_max_workers}, "
    f"reproject_kwargs.parallel={reproject_inner_parallel}"
)

# run_reproject is resume-safe: it writes a manifest of the intended task
# set, filters already-done outputs before dispatch (so a Ctrl-C + rerun
# is cheap), writes h5 files atomically (.tmp + rename), and appends any
# worker failures to {reproj_dir}/failed.jsonl.
rr.run_reproject(max_workers=reproject_max_workers,
                 reproj_func='exact',
                 padding_percentage=0.05,
                 sci_ext_list=[1],
                 dq_ext_list=[2],
                 exp_idx_list=np.arange(0, len(exposure_list)),
                 det_idx_list=[0] * len(exposure_list),
                 replace_existing=False,
                 reproject_kwargs={'parallel': reproject_inner_parallel}
                )

# One-line summary of this run's reprojection state.
rr.status()
print("Reprojection complete")