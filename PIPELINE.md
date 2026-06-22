# SelfCal pipeline runbook

Operational + on-disk-schema reference for the SelfCal calibration /
mosaicking pipeline. Companion to [selfcal/README.md](selfcal/README.md),
which documents the *code architecture* (module layout, shared-memory
hand-off, parallel SpMV, `_prep_subframe`, etc.). Read that first when
modifying anything inside `selfcal/`. Read this file when running the
pipeline, tuning hyperparameters, or working with its outputs.

## Running

Runs are driven by a **TOML config + the generic runner** — no editing Python:

```bash
./selfcal_scripts/run.sh selfcal_scripts/configs/<run>.toml          # or launch/<run>.sh
./selfcal_scripts/run.sh selfcal_scripts/configs/<run>.toml --dry-run  # resolve jobs+mode only
```

The knobs documented below still exist — they moved from dict literals at the top
of a driver into TOML tables:

| Was (driver dict / literal) | Now (TOML) |
| --- | --- |
| `frame_setting` (Detector / NumSub / NumCh / NumCol) | `[instrument]` |
| `chs` (channel / window selection) | `[instrument]` `windows` / `channels` / `channel_range` / `subch_window` |
| `calibration_kwargs` | `[calibration]` + per-block knobs in `[params]` (`reg_weight`, `poly_degree`/`poly_weight`) |
| `lsqr_kwargs` | `[lsqr]` |
| `mosaic_kwargs` | `[mosaic]` |
| `FILE_SUFFIX`, oversample, NVMe staging | top-level `suffix` / `oversample` / `staging` / `keep_nvme` |
| offset-model **structure** (adjacency choice, single vs dual poly, K=2 block) | the **mode** (`mode = "..."`; `selfcal_scripts/runner/modes/`) |

The offset-model *structure* (which adjacency, poly groups, K=2 readout block) is
chosen by the **mode**, not a flat kwarg — that is the one conceptual change. The
schema + how to add a mode/instrument is in
[`selfcal_scripts/configs/README.md`](selfcal_scripts/configs/README.md). The rest
of this file explains what each knob *does*; the names match the TOML keys.

## Calibration model

The selfcal model is per-pixel:

```
observed[i] = sky(p_i) + Σ_m offset^(m)[g_m(k), c_m(i)] + scalar[k] + noise
```

where:
- `m = 0..K-1` indexes **chunk maps**. Each map contributes one additive offset block (K=1 is the legacy single-map case).
- `g_m(k)` is the frame→group mapping for map `m` (defaults to identity; can lock multiple frames to share an offset vector via `det_groups_list[m]`).
- `c_m(i)` is the chunk ID of pixel `i` under map `m`.
- `scalar[k]` is an optional **per-frame DC scalar** added when `use_per_frame_scalar=True` (set by the continuum / pahfit / tiled modes). It absorbs per-frame brightness shifts so the chunk offsets only carry within-frame structure.

For the K=1 default case the model collapses to `sky + offset[frame, chunk] + scalar[frame]`. Zodi removal quality is dominated by the offset model's spatial resolution.

## Calibration pipeline tuning

Chunk geometry (the `[instrument]` TOML table):

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

`[calibration]` (production defaults, e.g. the `d4_aromatic` config) — passed
verbatim as `setup_lsqr` kwargs. Per-block knobs (`reg_weight`, `poly_degree`/
`poly_weight`) live in `[params]` and the mode lowers them onto the `OffsetBlock`:

```toml
[calibration]
apply_mask = true
apply_weight = true          # d4_aromatic/pahfit use Poisson weighting; d5/k2 false
outlier_thresh = 5.0
ignore_list = []             # [21] for pahfit/tiled (drop source-mask bit)
batch_size = 50              # tuned 2026-05 from 20
offset_regularization = true
weighted_damping = true
damp_weight = 0.1
max_workers = 48             # tuned 2026-05 from 32

[params]
reg_weight = 0.1             # adjacency-smoothness weight (per chunk map)
poly_degree = 1              # omit poly_weight to disable the column poly-constraint
poly_weight = 0.5
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

- **`poly_constraints_list[m]`** (optional) — list of constraint dicts that enforce polynomial offset behavior along supplied chunk chains. Each dict is `{'chains': (n_chains, L) int array, 'stencil': (L,) float array, 'weight': float}` and adds `weight * Σ_ℓ stencil[ℓ] · O[chains[r, ℓ]] = 0` rows per frame, per chain. For SPHEREx column linearity: `compute_column_polynomial_chains(det_chunk_map, num_columns, degree=1)` returns `(chains, stencil)` with stencil `[1, -2, 1]` and chain length `degree+2`. See [selfcal/instruments/spherex/spherex_utility.py](selfcal/instruments/spherex/spherex_utility.py).

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

**Channel / window selection** lives in `[instrument]` (de-mixed into typed keys,
one per run): `channels = [[14],[15]]` (one calibration run per entry),
`channel_range = [23, 35]` (expands to single-channel jobs), `windows =
["Aromatic","Aliphatic"]` (named subchannel ranges 225–235 / 249–259 covering the
PAH bands, registry in the SPHEREx adapter), or `subch_window = [lo, hi]` +
`window_name` (an explicit subchannel range, e.g. the PAHfit 210–250 / 200–260
windows). The instrument's `frame_tag` is `Detector{D}_NumSub{S}_NumCh{C}_NumCol{Co}`,
so cal filenames are deterministically
`cal_{frame_tag}_{job}{suffix}.h5` (job = `Ch<n>` or the window name).

## Advanced Calibrator solve modes

Beyond the default per-frame, per-chunk solve, `Calibrator.setup_lsqr`
supports restricted-solve modes that the mainline `continuum` mode does not use
but exist in the API (the `k2_readout` mode uses `det_groups_list`):

- **Locked offsets via `det_groups_list[m]`.** Pass an array of length
  `num_frames` giving a group ID per frame for map `m`; frames in the same
  group share one offset vector. Reduces unknowns from
  `num_frames * num_chunks_m` to `num_groups_m * num_chunks_m`. Useful when
  the same pointing repeats and the spatial pattern is presumed constant
  within a group. Recover the pre-expansion offsets with
  `Calibrator.get_det_offset(m)`. **K=2 use case**: pair a free per-frame
  map at `m=0` with a `det_groups_list[1]=zeros` map at `m=1` to capture a
  detector-fixed pattern shared across all frames (e.g., readout-channel
  stripes — the `k2_readout` mode / `configs/k2_readout.toml`).
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
  [selfcal/README.md](selfcal/README.md)).

`compute_x0_scalar_only(A, b, ref_shape, scalar_col_start)` (in
`selfcal/core/solution.py`) returns a warm-start `x0` with sky+offsets=0 and
the per-frame scalar block seeded from the diagonal-LS estimate. Use
this whenever `use_per_frame_scalar=True`. For runs without the scalar,
`compute_x0_from_Ab(A, b, ref_shape)` is the older full-offset warm
start.

## NVMe staging pattern

Reprojected `.h5` files live on RAID (HDD); parallel reads thrash the
heads. The pattern (in `selfcal_scripts/runner/staging.py`, driven by the
top-level `staging` / `keep_nvme` / `hdd_io_limit` config keys):

1. `set_hdd_io_limit(20)` — throttle the initial HDD copy
2. Copy `*.h5` to `{CACHE_DIR}/reproj_nvme_{run_name}/` via
   `ThreadPoolExecutor`
3. `set_hdd_io_limit(None)` — NVMe handles massive parallelism
4. Pass `reproj_dir=nvme_reproj_dir` to `Calibrator` / `Mosaicker`
5. Before `save_calibration`, swap `cc.reproj_list` basenames back to HDD
   paths so the cal file remains valid after NVMe cleanup
6. `shutil.rmtree(nvme_reproj_dir)` at the end

`set_hdd_io_limit(n)` installs a `multiprocessing.BoundedSemaphore` in
`selfcal/_state.py:_hdd_io_semaphore`, which both `ThreadPoolExecutor`
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

`selfcal_scripts/drivers/diff_cal_h5.py` understands both schemas — pass a legacy file and a new file and it compares the underlying arrays correctly.

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

## Zodi anchor stage (absolute brightness)

The LSQR solve leaves a global additive degeneracy (`sky += C`,
`frame_scalar -= C` is invariant). The **zodi anchor** fixes `C` per
channel by matching the solved per-frame DC to a Kelsall/zodipy
prediction. It is **non-mutating**: `cal_*.h5` and `mosaic_*.fits` stay
pristine; the fit lands in a per-detector anchor file
`<run>/zodi_anchor/anchor_D{N}.h5` and is applied at read time. Full
detail: [`selfcal_scripts/zodi_anchor/README.md`](selfcal_scripts/zodi_anchor/README.md).

Ordering (after cal+mosaic exist):

```bash
# 1. Per-frame zodi predictions — EXPENSIVE, runs in the selfcal-zodipy
#    env (zodipy needs numpy<2). Writes <run>/zodi_preds/zodi_pred_*.npz.
/home/thomasli/anaconda3/envs/selfcal-zodipy/bin/python \
    selfcal_scripts/zodi_anchor/build_predictions_all_channels.py --detector N ...

# 2. Fit the anchor (cheap; selfcal env). Writes <run>/zodi_anchor/anchor_D{N}.h5.
#    Add --smooth ONLY for atmospheric detectors (D1 He I/OI; D2) — see below.
python selfcal_scripts/zodi_anchor/build_anchor.py --run-dir <run> [--smooth]

# (alternatively, a cal config with [zodi].pred_dir set writes the anchor
#  inline per channel as the cal loop runs — step 2 then already done.)

# 3. (optional) slope smoothing as a separate, inspectable step:
python selfcal_scripts/zodi_anchor/smooth_anchor.py --run-dir <run> --dry-run --plot
python selfcal_scripts/zodi_anchor/smooth_anchor.py --run-dir <run>
```

Consuming the anchor (pipeline outputs stay pristine):

```python
from selfcal.zodi_anchor import load_anchor, load_anchored_mosaic
anchor = load_anchor('<run>/zodi_anchor/anchor_D1.h5')
data, hdr = load_anchored_mosaic('<run>/mosaic/mosaic_..._Ch11_...fits', anchor)  # +C in memory
# or arrays directly: anchor.C(ch), anchor.apply_to_mosaic_array(...), .apply_to_cal_scalar(...)
```

For a materialized FITS (ds9 / sharing), `materialize_anchored_mosaic.py
--run-dir <run>` writes anchored copies to `<run>/anchored_mosaics/`
(never overwrites the pipeline mosaic).

**Slope smoothing scope:** `--smooth` / `smooth_anchor.py` smooths the
per-channel slope across wavelength and overrides airglow-contaminated
channels (low Pearson r). Use it ONLY for detectors with atmospheric
contamination (D1 He I 1083 + OI 8446; D2 once mosaicked). Do NOT smooth
D4/D5 — their low-r channels are real astrophysical features (D4 PAH at
3.3 μm), not contamination. C is never smoothed (it carries the airglow);
only the slope is.

## Regression testing

`selfcal_scripts/benchmarks/run_cal_baseline_test.py` is the canonical regression harness. It defines `TEST_VARIANTS` (`poly_off`, `poly_k1`, `poly_k2`, `oldx0_off`, `scalar_off`, …) so the same script can produce side-by-side cal files for different solver configurations. Pair with `selfcal_scripts/drivers/diff_cal_h5.py` for element-wise diffs (schema-aware: legacy vs new, or new vs new).

Phase-level wall/RSS/IO benchmarking: `selfcal_scripts/benchmarks/benchmark_d3_ch17_{poly,numcol3,tuned,mid}.py`. Each writes `figures/benchmark/d3_ch17_{variant}_{summary.txt,samples.json,timeline.png}` and is parameterized by `max_workers` / `batch_size` / `n_threads` so you can run a tuning sweep quickly.

### Refactor bit-identity gate (`refactor/selfcal-package`)

The package refactor (`SelfCal/` → `selfcal/`, N-component SkyModel, OffsetModel, tiled wrapper) is verified phase-by-phase against **byte-identical** golden cal files. Two fast fixed-subset gates cover the two code paths the refactor touches; both write to the production `calibration/` dir with a `_gate_*` suffix (no clobber of real cals) and both must be re-run with **identical** `--n-frames` / `--max-workers` / `--batch-size` (and `--iter-lim` for the spectral gate) across baseline and candidate — these set the float accumulation order, so changing them re-baselines.

- **Continuum** — `selfcal_scripts/benchmarks/regress_cal.py` (D3 Ch17, NumCol3, K=1 + per-frame scalar, 300 frames). Needs the D3 reproj subset staged at `cache/reproj_nvme_SPHEREx_nep_qr2_det3_6p2arcsec/`.
- **Spectral** — `selfcal_scripts/benchmarks/regress_cal_spectral.py` (D4 Aromatic_PAHfit, NumCol5, `spectral_fit=True`, per-frame scalar + linear column poly, 150 frames). Uses the already-staged `cache/reproj_nvme_pahfit_sanity_1k/`. Exercises the 2-block sky layout, per-pixel Gaussian `G(λ)` row coefficients, line-block Fisher, and the `skymap_line*` datasets.

`diff_cal_h5.py` is schema-aware and now compares `skymap_line`, `skymap_line_coverage`, `skymap_line_fisher`, and `skymap_fisher` when present (and flags a line block dropped by one side). Procedure per phase:

```bash
PY=/home/thomasli/anaconda3/envs/selfcal/bin/python   # dev env (has hdf5plugin etc.)
# Establish goldens once on the pre-refactor tip:
$PY selfcal_scripts/benchmarks/regress_cal.py          --suffix _gate_golden
$PY selfcal_scripts/benchmarks/regress_cal_spectral.py --suffix _gate_golden
# After each phase, re-run with a new suffix and diff (must be ALL DATASETS BYTE-EQUAL):
$PY selfcal_scripts/benchmarks/regress_cal.py          --suffix _gate_phaseN
$PY selfcal_scripts/drivers/diff_cal_h5.py  <cal_dir>/cal_..._Ch17_gate_golden.h5  <cal_dir>/cal_..._Ch17_gate_phaseN.h5
$PY selfcal_scripts/benchmarks/regress_cal_spectral.py --suffix _gate_phaseN
$PY selfcal_scripts/drivers/diff_cal_h5.py  <cal_dir>/cal_..._AromaticPAHfit_gate_golden.h5  <cal_dir>/cal_..._AromaticPAHfit_gate_phaseN.h5
```

The full comprehensive sweep (`run_cal_baseline_test.py`, all variants × all frames) is the heavier comprehensive check; the two fast gates above are the per-phase tripwire.
