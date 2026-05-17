# SelfCal pipeline runbook

Operational + on-disk-schema reference for the SelfCal calibration /
mosaicking pipeline. Companion to [SelfCal/README.md](SelfCal/README.md),
which documents the *code architecture* (module layout, shared-memory
hand-off, parallel SpMV, `_prep_subframe`, etc.). Read that first when
modifying anything inside `SelfCal/`. Read this file when running the
pipeline, tuning hyperparameters, or working with its outputs.

## Calibration model

The selfcal model is per-pixel:

```
observed[i] = sky(p_i) + Σ_m offset^(m)[g_m(k), c_m(i)] + scalar[k] + noise
```

where:
- `m = 0..K-1` indexes **chunk maps**. Each map contributes one additive offset block (K=1 is the legacy single-map case).
- `g_m(k)` is the frame→group mapping for map `m` (defaults to identity; can lock multiple frames to share an offset vector via `det_groups_list[m]`).
- `c_m(i)` is the chunk ID of pixel `i` under map `m`.
- `scalar[k]` is an optional **per-frame DC scalar** added when `use_per_frame_scalar=True` (default in `run_cal_v2.py`). It absorbs per-frame brightness shifts so the chunk offsets only carry within-frame structure.

For the K=1 default case the model collapses to `sky + offset[frame, chunk] + scalar[frame]`. Zodi removal quality is dominated by the offset model's spatial resolution.

## Calibration pipeline tuning

`frame_setting` chunk geometry ([selfcal_scripts/run_cal_v2.py](selfcal_scripts/run_cal_v2.py)):

- `NumSub`, `NumCh` — wavelength (radial) divisions; 10×34 is well-tuned.
  `make_fiducial_chunk_map` asserts `num_channels % 17 == 0` because the
  channel edges come from the 17-band `spherex_channels.csv` table;
  `NumCh=34` is the 17 edges interpolated 2×.
- `NumCol` — spatial divisions perpendicular to wavelength. **Primary knob
  for zodi-gradient resolution.** Too few (1–3) leaves intra-column
  residuals; too many (≥9) adds noise artifacts because each chunk has
  fewer pixels to estimate from. Production uses `NumCol=1` for narrow
  channels (relying on the per-frame scalar + adjacency reg) or `NumCol=3-10`
  for wider channels / poly-constrained runs.

`calibration_kwargs` (production defaults in `run_cal_v2.py`):

```python
{
  'apply_mask': True, 'apply_weight': False,
  'outlier_thresh': 5.0, 'ignore_list': [],
  'batch_size': 50,           # tuned 2026-05 from 20
  'offset_regularization': True,
  'reg_weights': [0.1],       # list — one per chunk map (K-element)
  'weighted_damping': True,
  'damp_weight': 0.1,
  'max_workers': 48,          # tuned 2026-05 from 32
  'postprocess_func': None,
}
```

Plus the always-list `setup_lsqr` arguments:

```python
cc.setup_lsqr(
    chunk_maps=[det_chunk_map],       # list of K chunk maps
    adj_infos=[adj_info],              # list, one per map; None to skip
    poly_constraints_list=None,        # optional, see below
    mean_offsets_list=[np.zeros(num_frames)],  # mean-anchor for each map
    use_per_frame_scalar=True,         # adds per-frame DC scalar column
    ...,
    **calibration_kwargs,
)
```

Key knobs:

- **`reg_weights[m]`** + **`adj_infos[m]`** adds `reg_weights[m] * (O_i - O_j) = 0` rows to LSQR for adjacent chunk pairs on map `m`. Two builders in `SPHERExUtility.py`:
  - `compute_column_adjacency(det_chunk_map, num_columns)` — pairs chunks at same subchannel, adjacent columns. **The default.** Returns `(empty, empty)` for `NumCol=1`; `setup_lsqr` demotes empty adj_info to `None` automatically.
  - `compute_subchannel_adjacency(...)` — pairs at same column, adjacent subchannels.

- **`poly_constraints_list[m]`** (optional) — list of constraint dicts that enforce polynomial offset behavior along supplied chunk chains. Each dict is `{'chains': (n_chains, L) int array, 'stencil': (L,) float array, 'weight': float}` and adds `weight * Σ_ℓ stencil[ℓ] · O[chains[r, ℓ]] = 0` rows per frame, per chain. For SPHEREx column linearity: `compute_column_polynomial_chains(det_chunk_map, num_columns, degree=1)` returns `(chains, stencil)` with stencil `[1, -2, 1]` and chain length `degree+2`. See [SelfCal/SPHERExUtility.py](SelfCal/SPHERExUtility.py).

- **`mean_offsets_list[m]`** — per-frame mean-offset soft constraint with weight 10.0 (hardcoded in `lsqr.py`). When using `use_per_frame_scalar=True`, anchor every map to mean-zero so all per-frame DC ends up in the scalar column.

- **`use_per_frame_scalar=True`** — adds an explicit `num_frames` block to `x` (one scalar per frame) decoupled from `det_groups_list`. Combined with mean-zero anchors on all maps, this pushes per-frame DC entirely into the scalar so chunk offsets only carry within-frame structure. **Required for narrow channels** (D3 Ch17 etc.) where sparse chunk coverage was previously letting per-frame DC leak into scan-stripe residuals.

- **`weighted_damping=True`** + **`damp_weight`** damps **sky pixels** (not offsets) toward zero, weighted by `sqrt(damp_weight * coverage)`.

`lsqr_kwargs`:

- Use **`compute_x0_scalar_only(A, b, ref_shape, scalar_col_start=cc.col_bases[len(cc.chunk_maps)])`** for the warm start when `use_per_frame_scalar=True`. It seeds *only* the scalar block from the diagonal-LS estimate (≈ weighted mean of valid `b` per frame), leaving chunks and sky at 0. Critical to avoid scan-stripe regressions on narrow channels.
- For runs without the per-frame scalar, use the older `compute_x0_from_Ab(A, b, ref_shape)` — diagonal-LS over the full offset region.
- `iter_lim=50` is typical with the warm start. Watch the `show=True` residual prints (`arnorm` should drop to ~1 or below) to confirm convergence.
- `precondition=True` (column-norm) is essential — much faster convergence.
- `apply_lsqr` builds a custom row-block-parallel `LinearOperator` when `n_threads > 1`, with BLAS pinned to a single thread via `threadpool_limits(limits=1, user_api='blas')` so BLAS doesn't fight the SpMV threads. Default `n_threads=48` (tuned 2026-05).

`det_offset_funcs[m]` (in `Mosaicker.make_mosaic`) controls **mosaic-time** offset rendering — LSQR always solves block-constant chunk offsets regardless. Default (`None`) renders chunks with `chunk_to_det` (block-constant, visible edges); SPHEREx LVF maps use `make_spherex_stripped_offset_map` (mean-preserving 2D spline over `r_edges, x_edges`). For multi-map mosaics, each map gets its own `det_offset_func` (or `None`), and `_prep_subframe` sums their grid contributions before a single `det_to_sub` interp.

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
supports restricted-solve modes that aren't currently exercised by
`run_cal_v2.py` but exist in the API:

- **Locked offsets via `det_groups_list[m]`.** Pass an array of length
  `num_frames` giving a group ID per frame for map `m`; frames in the same
  group share one offset vector. Reduces unknowns from
  `num_frames * num_chunks_m` to `num_groups_m * num_chunks_m`. Useful when
  the same pointing repeats and the spatial pattern is presumed constant
  within a group. Recover the pre-expansion offsets with
  `Calibrator.get_det_offset(m)`. **K=2 use case**: pair a free per-frame
  map at `m=0` with a `det_groups_list[1]=zeros` map at `m=1` to capture a
  detector-fixed pattern shared across all frames (e.g., readout-channel
  stripes — see `selfcal_scripts/run_cal_v2_k2_readout.py`).
- **Template-amplitude mode via `det_templates[m]`.** Requires
  `det_groups_list[m]` also. Fixes the spatial pattern from a
  previously-solved `(num_groups, num_chunks_m)` template and solves only
  one scalar amplitude `alpha` per frame. Spatial regularization rows are
  skipped automatically for this map.
- **Mean-offset constraint via `mean_offsets_list[m]`.** Length-`num_frames`
  array of target mean values; `setup_lsqr` appends soft constraint rows
  pulling each frame's chunk-offset mean toward the target. Constraint
  weight is hardcoded at 10.0 in `lsqr.py`. For K≥2 the mean-anchor on
  maps 1..K-1 is how you break the K-1 shift degeneracy (see
  [SelfCal/README.md](SelfCal/README.md)).

`compute_x0_scalar_only(A, b, ref_shape, scalar_col_start)` (in
`SelfCal/solution.py`) returns a warm-start `x0` with sky+offsets=0 and
the per-frame scalar block seeded from the diagonal-LS estimate. Use
this whenever `use_per_frame_scalar=True`. For runs without the scalar,
`compute_x0_from_Ab(A, b, ref_shape)` is the older full-offset warm
start.

## NVMe staging pattern

Reprojected `.h5` files live on RAID (HDD); parallel reads thrash the
heads. The pattern in `run_cal_v2.py`:

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

## `cal_*.h5` schema (multi-chunk-map)

Written by `Calibrator.save_calibration`, read by analysis scripts (via
`zodi_utils.load_cal_offsets`) and `Mosaicker.load_calibration`. Schema
varies by `num_maps`:

**Top-level (always present):**
- `skymap` — `(ref_h, ref_w)` float32 — solved sky map
- `skymap_coverage` — `(ref_h, ref_w)` int32 — frames touching each pixel
- `reproj_list` — list of HDD paths to the reprojected files (dataset of bytes)
- `num_maps` (attr) — number of chunk maps `K`
- `frame_scalar` — `(num_frames,)` float32 — per-frame DC scalar (only when `use_per_frame_scalar=True`)

**Groups (one dataset per map):**
- `offsets/map_{m}` — `(num_frames, num_chunks_m)` float32 — per-frame per-chunk offsets, **expanded** from groups to per-frame
- `offset_coverage/map_{m}` — `(num_frames, num_chunks_m)` int32 — pixel count per (frame, chunk)
- `offset_coverage_frac/map_{m}` — `(num_frames, num_chunks_m)` float32 — fraction of chunk pixels actually covered per frame
- `chunk_maps/map_{m}` — `(det_h, det_w)` int — the chunk_map array used for map `m` (stored for analysis reproducibility)

**Legacy schema (pre-multi-chunk-maps, still readable):**
Top-level `offset`, `offset_coverage`, `offset_coverage_frac` (no `offsets/` group, no `num_maps` attr, no `frame_scalar`). Both `Mosaicker.load_calibration` and `zodi_utils.load_cal_offsets` detect the schema and adapt; the latter folds `frame_scalar` into map-0 offsets for analysis-side compatibility with the legacy single-map subtraction semantics.

`selfcal_scripts/diff_cal_h5.py` understands both schemas — pass a legacy file and a new file and it compares the underlying arrays correctly.

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
- `MEANOFF` — global mean of valid map-0 offsets at mosaic time
  (`np.mean(offsets[0][offset_coverage_frac >= valid_chunk_thresh])`),
  stamped on every map HDU. If `normalize_offset=True` was used, this is
  the value subtracted from the offsets before they were applied — add
  it back to recover absolute brightness.

## Regression testing

`selfcal_scripts/run_cal_baseline_test.py` is the canonical regression harness. It defines `TEST_VARIANTS` (`poly_off`, `poly_k1`, `poly_k2`, `oldx0_off`, `scalar_off`, …) so the same script can produce side-by-side cal files for different solver configurations. Pair with `selfcal_scripts/diff_cal_h5.py` for element-wise diffs (schema-aware: legacy vs new, or new vs new).

Phase-level wall/RSS/IO benchmarking: `selfcal_scripts/benchmark_d3_ch17_{poly,numcol3,tuned,mid}.py`. Each writes `figures/benchmark/d3_ch17_{variant}_{summary.txt,samples.json,timeline.png}` and is parameterized by `max_workers` / `batch_size` / `n_threads` so you can run a tuning sweep quickly.
