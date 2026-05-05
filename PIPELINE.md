# SelfCal pipeline runbook

Operational + on-disk-schema reference for the SelfCal calibration /
mosaicking pipeline. Companion to [SelfCal/README.md](SelfCal/README.md),
which documents the *code architecture* (module layout, shared-memory
hand-off, parallel SpMV, `_prep_subframe`, etc.). Read that first when
modifying anything inside `SelfCal/`. Read this file when running the
pipeline, tuning hyperparameters, or working with its outputs.

## Calibration pipeline tuning

The selfcal model is per-pixel: `observed = sky + offset[frame, chunk]`.
Each frame gets one scalar offset per chunk; zodi removal quality is
dominated by the offset model's spatial resolution.

`frame_setting` chunk geometry ([selfcal_scripts/run_cal_v2.py](selfcal_scripts/run_cal_v2.py),
`notebooks/benchmark_pipeline.py`):

- `NumSub`, `NumCh` — wavelength (radial) divisions; 10×34 is well-tuned.
  `make_fiducial_chunk_map` asserts `num_channels % 17 == 0` because the
  channel edges come from the 17-band `spherex_channels.csv` table;
  `NumCh=34` is the 17 edges interpolated 2×.
- `NumCol` — spatial divisions perpendicular to wavelength. **Primary knob
  for zodi-gradient resolution.** Too few (1–3) leaves intra-column
  residuals; too many (≥9) adds noise artifacts because each chunk has
  fewer pixels to estimate from. `NumCol=5` is the current default for
  det 5/6 where the spatial gradient is steepest.

`calibration_kwargs`:

- `offset_regularization=True` + `reg_weight` + `adj_info` adds
  `reg_weight * (O_i - O_j) = 0` rows to the LSQR matrix. Two adjacency
  builders in `SPHERExUtility.py`:
  - `compute_column_adjacency(det_chunk_map, num_columns)` — pairs chunks
    at same subchannel, adjacent columns. Smooths *across columns at
    fixed wavelength*. **This is the default.**
  - `compute_subchannel_adjacency(...)` — pairs at same column, adjacent
    subchannels. Tried as an alternative for det 5/6 to free the spatial
    gradient, but produced worse results — column adjacency wins in
    practice.
- `weighted_damping=True` + `damp_weight` damps **sky pixels** (not
  offsets) toward zero, weighted by `sqrt(damp_weight * coverage)`.

`lsqr_kwargs`:

- Use `compute_x0_from_Ab` (`MakeMap.py`) for the warm start. With it,
  `iter_lim=10–50` is enough — watch the `show=True` residual prints to
  confirm convergence; raising `iter_lim` further doesn't help once the
  solver has plateaued.
- `precondition=True` (column-norm) is essential — much faster
  convergence.
- **Always wrap `apply_lsqr` in
  `with threadpool_limits(limits=8, user_api='blas')`.** Scripts pin
  `OMP/MKL/OPENBLAS_NUM_THREADS=1` for the in-process LSQR threadpool,
  but scipy's lsqr/lsmr can otherwise grab all cores via BLAS and
  contend. (Note: when `n_threads > 1`, `apply_lsqr` builds a custom
  row-block-parallel `LinearOperator` (splits both `A` and `A^T` as CSR
  across a `ThreadPoolExecutor`) and *internally* wraps the solve in
  `threadpool_limits(limits=1, user_api='blas')` so BLAS doesn't fight
  the SpMV threads. The outer wrap is mostly insurance for the
  `n_threads=1` path.)

`det_offset_func(chunk_map, chunk_offset) -> 2D detector map` controls
offset interpolation **during mosaicking only** — LSQR always solves
block-constant chunk offsets regardless. Default (`None`) renders chunks
with `chunk_to_det` (block-constant, visible edges); SPHEREx uses
`make_spherex_stripped_offset_map` (mean-preserving 2D spline over
`r_edges, x_edges`).

`run_cal_v2.py:chs` accepts three forms: a list of single-channel lists
(e.g. `[[14], [15]]`, one calibration run per entry), a list of
multi-channel groups (joint solve over those channels' subchannels), or
the strings `'Aromatic'` / `'Aliphatic'`, which select hardcoded
subchannel ranges 225–235 / 249–259 covering the PAH bands.
`frame_setting_str` is auto-built as
`'_'.join(f'{k}{v}' for k,v in frame_setting.items())`, so calibration
filenames are deterministically
`cal_Detector{D}_NumSub{S}_NumCh{C}_NumCol{Co}_Ch{ch}{FILE_SUFFIX}.h5`.

## Advanced Calibrator solve modes

Beyond the default per-frame, per-chunk solve, `Calibrator.setup_lsqr`
supports three restricted-solve modes that aren't currently exercised by
`run_cal_v2.py` but exist in the API:

- **Locked offsets via `det_groups`.** Pass an array of length
  `num_frames` giving a group ID per frame; frames in the same group
  share one offset vector, plus each frame keeps an extra free per-frame
  scalar (one column applied as a bias to every valid pixel of that
  frame). Reduces unknowns from `num_frames * num_chunks` to
  `num_groups * num_chunks + num_frames`. Useful when the same pointing
  repeats and the spatial pattern is presumed constant within a group
  while the level can drift. Recover the pre-expansion offsets with
  `Calibrator.get_det_offset()` for use as a template seed.
- **Template-amplitude mode via `det_template`.** Requires `det_groups`
  also. Fixes the spatial pattern from a previously-solved
  `(num_groups, num_chunks)` template and solves only one scalar
  amplitude `alpha` per frame plus the per-frame scalar. Spatial
  regularization rows are skipped automatically.
- **Mean-offset constraint via `mean_offsets`.** Length-`num_frames`
  array of target mean values; `setup_lsqr` appends soft constraint rows
  pulling each frame's chunk-offset mean toward the target. Constraint
  weight is hardcoded at 10.0 in `lsqr.py`. Pair with
  `compute_offsets_guess(reproj_list, det_chunk_map)` from
  `SPHERExUtility.py`, which reads raw FITS frames directly (no
  reprojection) and emits a per-frame, per-chunk mean usable as either
  the target or as part of `x0`.

`compute_x0_from_Ab(A, b, ref_shape, num_offset_groups)` (in
`SelfCal/solution.py`, re-exported via `MakeMap.py`) returns a warm-start
`x0` with `sky=0` and offsets seeded from the diagonal least-squares
estimate `A_off^T b / diag(A_off^T A_off)`. Cheap (no FITS re-reads —
derived directly from the already-built sparse matrix). The
`run_cal_v2.py` reference flow uses it.

## NVMe staging pattern

Reprojected `.h5` files live on RAID (HDD); parallel reads thrash the
heads. The pattern in `run_cal_v2.py` and `notebooks/benchmark_pipeline.py`:

1. `set_hdd_io_limit(20)` — throttle the initial HDD copy
2. Copy `*.h5` to `{CACHE_DIR}/reproj_nvme_{run_name}/` via
   `ThreadPoolExecutor`
3. `set_hdd_io_limit(None)` — NVMe handles massive parallelism
4. Pass `reproj_dir=nvme_reproj_dir` to `Calibrator` / `Mosaicker`
5. Before `save_calibration`, swap `cc.reproj_list` basenames back to HDD
   paths so the cal file remains valid after NVMe cleanup
6. `shutil.rmtree(nvme_reproj_dir)` at the end

`set_hdd_io_limit(n)` installs a `multiprocessing.BoundedSemaphore` in
`SelfCal/_state.py:_hdd_io_semaphore`, which both `ThreadPoolExecutor`
workers and forked `Pool` workers acquire inside `load_reproj_file`. So
the throttle works regardless of which parallelism mode the consumer
uses, and `set_hdd_io_limit(None)` takes effect immediately for any
subsequent reads.

## `cal_*.h5` schema

Written by `Calibrator.save_calibration`, read by analysis scripts and
`Mosaicker.load_calibration`:

- `skymap` — `(ref_h, ref_w)` float32 — solved sky map
- `offset` — `(num_frames, num_chunks)` float32 — per-frame per-chunk
  offsets
- `skymap_coverage` — `(ref_h, ref_w)` int32 — frames touching each pixel
- `offset_coverage_frac` — `(num_frames, num_chunks)` float32 — fraction
  of chunk pixels actually covered per frame
- `reproj_list` (attr) — list of HDD paths to the reprojected files

## Reprojected `*.h5` schema

Written by `Reprojector.run_reproject` (one file per (exposure,
detector)). Filename pattern: `exp_{exp_idx:04d}_det_{det_idx:02d}.h5`;
`load_reproj_file` parses the indices back out of the basename — keep the
pattern stable. Compressed with Zstd + byte-shuffle via `hdf5plugin`.

Datasets:
- `sub_data` `(sub_w, sub_w)` float32 — reprojected science image.
- `sub_foot` `(sub_w, sub_w)` float16 — fractional footprint from the
  `reproject` library.
- `sub_bitmask` `(sub_w, sub_w)` int32 — DQ bitmask after reprojection
  (per-bit reprojected as float, thresholded at 0.01, then re-packed).
- `sub_mapping` `(2, sub_w, sub_w)` float32 — for each subframe pixel,
  the (x, y) sample location in the original *detector* frame. Used by
  every consumer to (a) build the bilinear-interp sparse matrix back to
  the chunk map and (b) sample per-pixel `det_BC` / `det_BW` in
  `wav_coadd`.

Attributes:
- `sub_header` (bytes) / `det_header` (bytes) — FITS headers as strings;
  `load_reproj_file` reconstructs `sub_wcs` / `det_wcs` on demand.
- `file_path` (str) — path to the source FITS the subframe came from.
- `ref_coords` `(4,)` int32 — `[y_min, y_max, x_min, x_max]` in the
  reference frame, where `sub_data` should be splatted back. Can extend
  outside the mosaic; `MapHelper.compute_crop` handles the clip.

Sub-frame side length sized to fit the detector diagonal at mosaic
resolution:
`sub_width = ceil(sqrt(2) * det_width / (ref_reso/det_reso) * (1 + 2*padding_percentage))`.

Cached intermediates from `coadd.compute_coadd_map(mode='cache')` live in
`cache_dir` as `cached_<original>.h5` and follow the same schema, but the
arrays are tightly cropped to the nonzero-weight bbox; an extra
`sub_bbox` `[rmin, rmax, cmin, cmax]` records that crop in original
sub-frame coordinates so `wav_coadd` can crop `sub_mapping` to match
before `map_coordinates`.

## Mosaic `*.fits` schema

Written by `Mosaicker.save_mosaic`. Multi-extension FITS — primary HDU is
empty; each map is its own `ImageHDU` carrying the mosaic WCS in the
header. `EXTNAME` is one of:

- `MEAN_MAP`, `MEAN_MAP_WEIGHT` — weighted mean and `sum(weight)`.
- `STD_MAP`, `STD_MAP_WEIGHT` — weighted std and weight.
- `SC_MEAN_MAP`, `SC_MEAN_MAP_WEIGHT` — sigma-clipped mean and weight.
- `WAV_MEAN_MAP`, `WAV_STD_MAP` — appended via `Mosaicker.append_maps`
  after `wav_coadd` (BUNIT=`um`).

Header keys to know:
- `BUNIT` — taken from `Mosaicker.maps[name]['unit']` (`'MJy/sr'` for sky
  maps, `'um'` for wavelength, `'Weight'` for the `_WEIGHT` companions).
- `MEANOFF` — global mean of valid offsets at mosaic time
  (`np.mean(offset[offset_coverage_frac >= valid_chunk_thresh])`),
  stamped on every map HDU. If `normalize_offset=True` was used, this is
  the value subtracted from the offsets before they were applied — add
  it back to recover absolute brightness.
