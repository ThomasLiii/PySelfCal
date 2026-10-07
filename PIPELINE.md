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
| offset-model **structure** (adjacency choice, single vs dual poly, K=2 block) | the **mode** (`mode = "..."`; `selfcal/run/modes/`) |

The offset-model *structure* (which adjacency, poly groups, K=2 readout block) is
chosen by the **mode**, not a flat kwarg — that is the one conceptual change. The
schema + how to add a mode/instrument is in
[`selfcal_scripts/configs/README.md`](selfcal_scripts/configs/README.md). The rest
of this file explains what each knob *does*; the names match the TOML keys.

## Calibration model

The selfcal model is per observation (one value of frame `k` on reference pixel `p_i`):

```
observed[i] = Σ_j sky_j(p_i)·c_j(v_i) + Σ_m Σ_n offset^(m)[g_m(k), c_m(i), n]·φ_mn(v_i) + scalar[k] + noise
```

where:
- `v_i` are the observation's **data variables** (detector maps such as SPHEREx's band centre,
  per-frame values such as the time, reference-grid maps, stored layers, the built-in
  coordinates, functions of those); `c_j` and `φ_mn` are any known functions of them
  (`selfcal/models/variables.py`, `[model.variables]`).
- `j` indexes **sky terms** (one map each; `c = 1` for a constant sky).
- `m = 0..K-1` indexes **chunk maps**. Each map contributes one additive offset block (K=1 is the legacy single-map case); `φ = 1` (one function) is the classic chunk offset.
- `g_m(k)` is the frame→group mapping for map `m` (defaults to identity; can lock multiple frames to share an offset vector via `det_groups_list[m]`, or group by any per-frame value).
- `c_m(i)` is the chunk ID of pixel `i` under map `m`.
- `scalar[k]` is an optional **per-frame DC scalar** added when `use_per_frame_scalar=True` (set by the continuum / spectral / tiled modes). It absorbs per-frame brightness shifts so the chunk offsets only carry within-frame structure.

For the K=1 default case the model collapses to `sky + offset[frame, chunk] + scalar[frame]`. Zodi removal quality is dominated by the offset model's spatial resolution. User priors (any linear rows on the unknowns) and an observation weight (a function of data variables) complete the model; see `selfcal_scripts/configs/README.md` and `docs/bring_your_own_telescope.md`.

## Calibration pipeline tuning

Chunk geometry (the `[instrument]` TOML table):

- `num_sub`, `num_ch` (`NumSub`, `NumCh` in the product names) — wavelength (radial) divisions; 10×34 is well-tuned.
  `make_fiducial_chunk_map` asserts `num_channels % 17 == 0` because the
  channel edges come from the SPHEREx channel table (17 channels per band),
  `selfcal/instruments/spherex/data/spherex_channels.csv`, shipped with the package;
  `NumCh=34` is the 17 edges interpolated 2×.
- `num_col` (`NumCol`) — spatial divisions perpendicular to wavelength. **Primary knob
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

Programmatically, the offset structure is an `OffsetModel` of one `OffsetBlock` per chunk map
(the runner builds it from the mode; `Calibrator.setup_lsqr` still accepts the older flat per-map
lists `chunk_maps=` / `adj_infos=` / `reg_weights=` / `poly_constraints_list=` /
`mean_offsets_list=` / `det_groups_list=` / `det_templates=`, deprecated — they remain the native
arguments of the core `selfcal.core.system.setup_lsqr`, one list entry per map):

```python
cc.setup_lsqr(
    offset_model=OffsetModel(
        blocks=(OffsetBlock(chunk_map=det_chunk_map,           # chunk_maps[m]
                            adj_info=adj_info, reg_weight=0.1,  # adj_infos[m], reg_weights[m]
                            poly_constraints=None,              # poly_constraints_list[m]
                            mean_offset=np.zeros(num_frames)),),  # mean_offsets_list[m]
        use_per_frame_scalar=True),                             # per-frame DC scalar column
    sky_model=SkyModel.continuum_only(),
    **calibration_kwargs,
)
```

Key knobs (per map `m`; the block field is named in parentheses):

- **`reg_weights[m]`** + **`adj_infos[m]`** (`reg_weight`, `adj_info`) adds `reg_weights[m] * (O_i - O_j) = 0` rows to LSQR for adjacent chunk pairs on map `m`. Two builders in `selfcal/instruments/spherex/spherex_utility.py` (the modes build the same pairs with `selfcal.models.offset_structure.adjacency_along`):
  - `compute_column_adjacency(det_chunk_map, num_columns)` — pairs chunks at same subchannel, adjacent columns. **The default.** Returns `(empty, empty)` for `NumCol=1`; `setup_lsqr` demotes empty adj_info to `None` automatically.
  - `compute_subchannel_adjacency(...)` — pairs at same column, adjacent subchannels.

- **`poly_constraints_list[m]`** (`poly_constraints`; optional) — list of constraint dicts that enforce polynomial offset behavior along supplied chunk chains. Each dict is `{'chains': (n_chains, L) int array, 'stencil': (L,) float array, 'weight': float}` and adds `weight * Σ_ℓ stencil[ℓ] · O[chains[r, ℓ]] = 0` rows per frame, per chain. For SPHEREx column linearity: `compute_column_polynomial_chains(det_chunk_map, num_columns, degree=1)` returns `(chains, stencil)` with stencil `[1, -2, 1]` and chain length `degree+2`. See [selfcal/instruments/spherex/spherex_utility.py](selfcal/instruments/spherex/spherex_utility.py).

- **`mean_offsets_list[m]`** (`mean_offset`) — per-frame mean-offset soft constraint with weight 10.0 (hardcoded in `selfcal/core/system.py`). When using `use_per_frame_scalar=True`, anchor every map to mean-zero so all per-frame DC ends up in the scalar column.

- **`use_per_frame_scalar=True`** — adds an explicit `num_frames` block to `x` (one scalar per frame) decoupled from `det_groups_list`. Combined with mean-zero anchors on all maps, this pushes per-frame DC entirely into the scalar so chunk offsets only carry within-frame structure. **Required for narrow channels** (D3 Ch17 etc.) where sparse chunk coverage was previously letting per-frame DC leak into scan-stripe residuals.

- **`weighted_damping=True`** + **`damp_weight`** damps **sky pixels** (not offsets) toward zero, weighted by `sqrt(damp_weight * coverage)`.

`lsqr_kwargs`:

- Use **`compute_x0_scalar_only(A, b, ref_shape, scalar_col_start=cc.col_bases[len(cc.chunk_maps)], num_sky_blocks=cc.num_sky_blocks, active_mask=cc.active_mask)`** for the warm start when `use_per_frame_scalar=True`. It seeds *only* the scalar block from the diagonal-LS estimate (≈ weighted mean of valid `b` per frame), leaving chunks and sky at 0. Critical to avoid scan-stripe regressions on narrow channels. `active_mask` is required because `setup_lsqr` compacts the zero columns by default; `x0` comes back in the full column layout `apply_lsqr` expects.
- For runs without the per-frame scalar, use the older `compute_x0_from_Ab(A, b, ref_shape, active_mask=cc.active_mask)` — diagonal-LS over the full offset region.
- `iter_lim=50` is typical with the warm start. Watch the `show=True` residual prints (`arnorm` should drop to ~1 or below) to confirm convergence.
- `precondition=True` (column-norm) is essential — much faster convergence.
- **The transpose product, and what "statistical equality" means here.**
  `A^T @ y` is a scatter into output columns, so it cannot be threaded
  without changing the order in which each column's contributions are
  summed. The default kernel is ROW-SPLIT: each thread owns a contiguous row
  range and a private output buffer, and the buffers are reduced in a fixed
  order afterwards. That is deterministic for a given thread count (the same
  config on the same machine reproduces the same bytes) but not bit-identical
  to the one-chain sequential product: a float32 reassociation of ~1e-7 per
  product that reaches the converged maps at ~1e-6 of their own values;
  integer coverage, Fisher and separability outputs are untouched. Measured
  on the real 1k-frame matrix it is 5.2x faster than the sequential kernel
  (2.07 s vs 10.69 s per product). Thread count: as many as the matvec uses,
  capped so the private buffers stay under `SELFCAL_RMATVEC_BUFFER_GB`
  (default 16); `SELFCAL_PARALLEL_RMATVEC=<n>` pins it, and `=1` selects the
  sequential kernel — the byte-exact verification mode the pre-2026-09-10
  goldens were made with. In practice the count equals `apply_n_threads` for
  every production tile: the buffer cap binds only when compaction is off
  (803 M uncompacted columns x 32 threads would want 103 GB), so all tiles of
  one mosaic are treated identically — but **keep `apply_n_threads` fixed
  across the tiles you intend to stitch**, since the solution depends on the
  count at the reassociation level. How much depends on how converged the
  solve is: a converged 1k-frame solve moves by ~1e-6 of each pixel's noise,
  while a deliberately under-converged 30-iteration template fit moves by
  ~1e-3 of it — and two different thread counts differ from each other by as
  much as either differs from the sequential kernel.
- **Column compaction is always on**, template-mode maps included. The solve
  runs in the active column space (e.g. 38 M of 803 M columns on a 1k-frame
  tile), which removes ~17 GB of n-space vectors and is what makes the
  row-split buffers affordable. The compact solve differs from the
  uncompacted one at ~5e-6 relative (n-space reductions regroup), the same
  class of difference as the row-split product. `compact_zero_columns=False`
  is a debugging escape hatch.
- **Column-partitioned storage.** Above the `SELFCAL_BLOCK_NNZ` threshold,
  `setup_lsqr` writes the matrix as **storage blocks x column ranges** (one
  block per spill batch and per constraint block), placing rows with scipy's
  `coo_tocsr` in one pass per block. The default is ONE range, i.e.
  block-major storage with one int32 indptr per block (the same bytes per row
  as a plain `BlockCSR`, but built without the int64 global indptr and with
  the per-row sorts threaded). `SELFCAL_RMATVEC_SPLIT=<T>` cuts T column
  ranges, which only the sequential verification kernel uses (one thread per
  range folds its columns in the sequential order — the byte-equal parallel
  transpose product of the 2026-09 byte-equality rounds); T ranges hold
  `(T-1) * 4 B/row` more indptr, and `SELFCAL_SPLIT_EXTRA_GB` (default 24)
  halves T until that fits.
- **Memory env knobs** (defaults need no tuning): `SELFCAL_BLOCK_NNZ` —
  nnz threshold at which `setup_lsqr` emits int32 block storage instead of a
  unified CSR (default `2**31`, the point where scipy would force int64
  indices; outputs are bit-identical either way). `SELFCAL_SPILL_MIN_GB`
  (default 4) / `SELFCAL_SPILL_DIR` (default system tmp) — `Calibrator.apply_lsqr`
  spills `pixel_counts`/`pixel_fisher`/`pixel_cross` to scratch for the
  duration of the solve when they exceed the threshold (exact byte
  round-trip; ~1 min I/O against a multi-hour production solve).
- **Process pools & the parallel scatter.** The assembly and Phase-4a CSR
  scatter pools (and the frame check of the `reproject` task) run on the
  **forkserver** start method (`selfcal.core.shmbuf.worker_pool_context`;
  `SELFCAL_MP_START_METHOD`, default `forkserver`; `fork` is a debugging
  escape hatch only); the other pools (coadd, reprojection, N-pass OFFSET
  refit, the standalone `wav_coadd`) use `multiprocessing`'s default start
  method (`fork` on Linux before Python 3.14). Forking a pool from the
  runner's multi-threaded process can hand a child the stderr lock in a
  locked state (the RSS guardrail prints
  every 15 s); the child then hangs at exit and the parent joins it forever —
  a production tile lost 6 h to this on 2026-09-09. Consequences: (1) entry
  scripts MUST keep their run code under `if __name__ == "__main__":`
  (children re-import the main module, the standard multiprocessing rule);
  (2) shared arrays reach workers as explicit `selfcal.core.shmbuf.SharedBuffer`
  handles (memfd-backed, fd-passed — no `/dev/shm` size cap), never by fork
  inheritance. The scatter: `SELFCAL_SCATTER_WORKERS` (default `min(8,
  max_workers)`; `0`/`1` = the byte-identical serial path) and
  `SELFCAL_SCATTER_TIMEOUT_S` (default 1800 per batch) — on timeout or a
  broken pool the remaining batches are re-scattered serially, so an
  unattended run degrades to slow, never to a hang. Serial and parallel
  scatters are element-wise identical (pure data movement).
- `apply_lsqr` builds a custom row-block-parallel `LinearOperator` when `n_threads > 1`, with BLAS pinned to a single thread via `threadpool_limits(limits=1, user_api='blas')` so BLAS doesn't fight the SpMV threads. The runner passes `apply_n_threads` (default 48, tuned 2026-05); `Calibrator.apply_lsqr` itself defaults to `n_threads=32`.
- The solver's elementwise vector updates (`x += t1*w`, `u *= alfa`, ...)
  run across `SELFCAL_VEC_THREADS` threads (default 8; serial below 16 M
  elements). Each element depends only on its own inputs, so the split is
  bit-identical; the reductions (norms) stay one ordered pass.

`det_offset_funcs[m]` (in `Mosaicker.make_mosaic`) controls **mosaic-time** offset rendering — LSQR solves block-constant chunk offsets regardless (except for an offset term with a `coefficient` or `basis`, which the mosaic subtracts per observation instead). Default (`None`) renders chunks with `chunk_to_det` (block-constant, visible edges); SPHEREx LVF maps use `make_spherex_stripped_offset_map` (mean-preserving 2D spline over `r_edges, x_edges`). For multi-map mosaics, each map gets its own `det_offset_func` (or `None`), and `_prep_subframe` sums their grid contributions before a single `det_to_sub` interp.

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
but exist in the API (the `two_block_fixed` mode, preset `k2_readout`, uses
`det_groups_list`; in a `[model]` table they are the `fixed` / `grouped` offset kinds). Each is
an `OffsetBlock` field (named in parentheses):

- **Locked offsets via `det_groups_list[m]`** (`det_groups`). Pass an array of length
  `num_frames` giving a group ID per frame for map `m`; frames in the same
  group share one offset vector. Reduces unknowns from
  `num_frames * num_chunks_m` to `num_groups_m * num_chunks_m`. Useful when
  the same pointing repeats and the spatial pattern is presumed constant
  within a group. Recover the pre-expansion offsets with
  `Calibrator.get_det_offset(m)`. **K=2 use case**: pair a free per-frame
  map at `m=0` with a `det_groups_list[1]=zeros` map at `m=1` to capture a
  detector-fixed pattern shared across all frames (e.g., readout-channel
  stripes — the `two_block_fixed` mode / `configs/k2_readout.toml`).
- **Template-amplitude mode via `det_templates[m]`** (`template`). Requires
  `det_groups_list[m]` also. Fixes the spatial pattern from a
  previously-solved `(num_groups, num_chunks_m)` template and solves only
  one scalar amplitude `alpha` per frame. Spatial regularization rows are
  skipped automatically for this map.
- **Mean-offset constraint via `mean_offsets_list[m]`** (`mean_offset`). Length-`num_frames`
  array of target mean values; `setup_lsqr` appends soft constraint rows
  pulling each frame's chunk-offset mean toward the target. Constraint
  weight is hardcoded at 10.0 in `selfcal/core/system.py`. For K≥2 the
  mean-anchor on maps 1..K-1 is how you break the K-1 shift degeneracy.

`compute_x0_scalar_only(A, b, ref_shape, scalar_col_start, num_sky_blocks, active_mask)` (in
`selfcal/core/solution.py`) returns a warm-start `x0` with sky+offsets=0 and
the per-frame scalar block seeded from the diagonal-LS estimate. Use
this whenever `use_per_frame_scalar=True`. For runs without the scalar,
`compute_x0_from_Ab(A, b, ref_shape, num_sky_blocks, active_mask)` is the older full-offset warm
start. Pass `active_mask` whenever `setup_lsqr` compacted the zero columns (the default).

## N-pass alternating solve (task `npass`)

One formalism for the spectral calibrations (SEP PAH J=2, NEP multi-line J=4),
implemented as the runner task `npass` (`selfcal/run/npass.py`;
primitives in `selfcal/pipeline/npass.py`; config table `[passes]`, see
`selfcal_scripts/configs/README.md`). Model per frame *k*, map pixel *p*:

```
d_k(p) = Σ_j S_j(p)·c_j(λ_k(p)) + Σ_d a_{k,col(p),d} B_d(sub(p)) + s_k
```

J sky blocks `S_j` (continuum + line amplitudes; `c_j` = the line template at
the observation's BC), a hard Chebyshev offset shape per column in subchannel,
and a per-frame scalar. Given the offsets the sky is **block-diagonal** (one
J×J normal system per pixel); given the sky the offsets are **independent per
frame**. The joint problem is therefore solved by alternating least squares
with each half exact:

| pass | type | solves | mechanism | tiles |
| --- | --- | --- | --- | --- |
| 1 | INIT | `S, a, s` jointly | the joint LSQR of the `cal` task (tiled when `[tiling]` is present) | yes (memory) |
| even | SKY | `S` given `a, s` | per-tile moment dumps (Σw², Σw²c_j, Σw²c_ic_j, Σw²v, Σw²c_jv) summed, one per-pixel closed-form solve (`solve_sky_closed_form`) | no — exact full-field |
| odd ≥ 3 | OFFSET | `a, s` given `S` | dense least squares per frame against the one global sky (deg 4, per-subchannel clip, bright-sky exclusion) | no |

`n = 1` **is** the legacy single solve (byte-equal; regression gate on a NEP
production tile). `n = 2` is the two-pass recipe; `n = 4` the SEP 4-pass
product. Why not just the joint LSQR: it *semi-converges* — the offsets
converge fast, the low-wavelength-diversity pixels' continuum↔line split does
not, and past that point the iterate drifts along the exact null spaces
(uniform line floor ↔ static detector pattern; uniform sky ↔ scalars). Each
SKY/OFFSET pass is an exact block minimization, so the objective is
non-increasing in `n`, but drift along the null spaces is not excluded — the
runner records per-pass monitors in `<stem>_npass_monitor.json` (per-block
median / % positive / step RMS, offset DC, residual RMS, bright-cut
fallbacks) and `stop_tol` can stop early; pick `n` from those, not by
assumption. The remaining zero points (line floor, continuum DC) are
unobservable from the data in any `n` and need the post-hoc anchors
(`selfcal/line_floor.py`, `selfcal/zodi_anchor.py`).

Why the SKY passes need no tiles: a pixel's normal equations are sums over its
observations, so per-tile dumps over **disjoint** frame sets are additive and
summing them is identical to a single full-field solve — no seam can exist.
Overlapping tile bboxes are de-duplicated first-tile-wins. The OFFSET pass
reads every frame of the field (`[tiling].full_reproj_dir`). Verified at full
scale on the NEP (17,647 frames, J=4): re-running a SKY pass from the same
offsets with a completely different partition (3 vertical bands instead of 6
blocks) reproduced the product to float32 rounding — 4–87 differing elements
of 160.6 M, max 4.7e-10 against a p99 signal of 1.7–4.1e-2, Fisher and
coverage byte-equal, and the median difference **exactly zero in every
distance bin from either partition's boundaries**.

**The OFFSET basis must resolve the window** (`[passes].offset.segments`). The
per-frame refit is a degree-`poly_degree` Chebyshev per column over the whole
`spectral_poly_lo..hi` window. On the SEP, the same degree 4 over the 121-subchannel
multi-line window (200–320) captured 3–5× less of the per-frame structure at the
~15-subchannel scale in the aromatic band than over the 60-subchannel aromatic
window (200–259): the wide fit is constrained by the red-end data, so red-end
residuals pull the polynomial on the aromatic band, and what it cannot follow
stays in the residual and projects onto the adjacent line templates as
frame-coherent stripes (0.64 ×10⁻³ MJy/sr rms at 64–1024 px in the dim sky,
aromatic–aliphatic stripe correlation +0.52, identical whether the per-pixel
model is J=2 or J=4). Raising the global degree is *not* the answer — degree 8
over 121 subchannels extrapolated to ±200 MJy/sr in frames with partial red-end
coverage. `segments = [[200, 259], [260, 320]]` fits an independent degree-4
shape on each range (the aromatic band gets exactly the narrow-window basis, the
red end its own), with nothing to extrapolate. One segment equal to the window is
bit-identical to the unsegmented basis.

**Ordering matters when INIT is tiled** (`[passes].order`, default
`sky_first`). Each INIT tile is an independent joint solve, so it picks its own
gauge along the near-null directions; frames in neighbouring tiles come out on
mutually inconsistent gauges. A SKY pass fed those offsets has to compromise,
which puts smooth footprint-scale lobes within ~1 frame footprint of every INIT
tile edge — and the next OFFSET pass then fits *to* the lobed sky, so the pair
is self-consistent and the alternation drains it only slowly (still visible at
pass 4). `order = "offset_first"` runs INIT → OFFSET → SKY → …, re-levelling
every frame against the one stitched INIT sky before any exact sky exists; its
first sky has no lobes (measured on the NEP: the pass-3-to-pass-5 step shows
0.9–1.2× the far-field level at the edges, i.e. flat, versus 2.5–3.8× for the
sky-first chain's maps). Use it whenever pass 1 is tiled; with an untiled INIT
there is no gauge mismatch to fix and the extra OFFSET pass is close to a
no-op. Note the parity: `offset_first` ends on a sky for **odd** `n`
(`sky_first` for even `n`) — the runner warns when a schedule ends on an OFFSET
pass, whose product carries no sky map.

Products: `<stem>_pass{i}sky.h5` (v3 sky-only cal: `sky/<name>`, Fisher,
coverage, `sky_separability/<name>`; written by the same
`selfcal/io/cal_writer.write_sky_groups` as `save_calibration`) and
`<stem>_pass{i}off.h5` (`offsets/map_0` + `frame_scalar` + `fit_ok` +
`resid_rms`, consumable by `OffsetSubtractor`). Re-running the same config
resumes at the first missing product. Hooks are POSTprocess functions
(weights are computed on the raw data first); `SkySubtractor.window` handles
subframes overhanging any map edge (a negative `ref_coords` start is a Python
negative slice — the SEP LMC-streak bug).

Wall on the 192-core box: SEP (19,269 frames, J=2): SKY pass ≈ 1.5 h (two
halves + combine), OFFSET ≈ 50 min; NEP 1k probe (J=4, 121 subch): SKY ≈ 45
min, OFFSET ≈ 4 min.

## NVMe staging pattern

Reprojected `.h5` files live on RAID (HDD); parallel reads thrash the
heads. The pattern (in `selfcal/run/staging.py`, driven by the
top-level `staging` / `keep_nvme` / `hdd_io_limit` config keys):

1. `set_hdd_io_limit(20)` — throttle the initial HDD copy
2. Copy `*.h5` to `{CACHE_DIR}/reproj_nvme_{run_name}/` via
   `ThreadPoolExecutor`. Each frame is copied under a temporary name and
   renamed when complete, so an interrupted copy never leaves a truncated
   frame; a complete copy already there (same size) is kept, so a staging
   resumes. The directory is marked with a `.selfcal-staging.json` file when
   it is created; a directory that holds files but no marker (frames linked
   by hand, a test fixture, another tool's copy) is refused.
3. `set_hdd_io_limit(None)` — NVMe handles massive parallelism
4. Pass `reproj_dir=nvme_reproj_dir` to `Calibrator` / `Mosaicker`
5. Before `save_calibration`, swap `cc.reproj_list` basenames back to HDD
   paths so the cal file remains valid after NVMe cleanup
6. `shutil.rmtree(nvme_reproj_dir)` at the end, unless `keep_nvme = true` or
   `staging = "reuse"`, and only for a marked directory

A tiled run stages each tile's frames into `[tiling].nvme_subdir` under
`cache_dir`, under the same marker rule, and never deletes it.

`set_hdd_io_limit(n)` installs a `multiprocessing.BoundedSemaphore` in
`selfcal/_state.py:_hdd_io_semaphore`, which `ThreadPoolExecutor` workers
and fork-started `Pool` workers (reprojection) acquire inside
`load_reproj_file`; the forkserver pools (assembly, scatter, coadd, the
N-pass refit) start without it, so their reads are not throttled.
`set_hdd_io_limit(None)` takes effect immediately for any subsequent reads.

## `cal_*.h5` schema (multi-chunk-map)

Written by `Calibrator.save_calibration`. Read it with
`selfcal.io.calfile.CalFile` (`sky(name)`, `offsets`, `frame_scalar`,
`total_offsets()`, `reproj_list`, ...), which resolves every layout below (and
the stitched / N-pass products); `Mosaicker.load_calibration` and the analysis
scripts' `zodi_utils.load_cal_offsets` are its consumers. Schema varies by
`num_maps`:

**Top-level (always present):**
- `sky/<name>`, `sky_coverage/<name>`, `sky_fisher/<name>` — `(ref_h, ref_w)` — map, coverage and
  Fisher information of each sky term (schema v3: attrs `schema_version = 3`, `num_sky_blocks`,
  `sky_components` = the term names, first term first); `sky_separability/<name>` for every term
  after the first when there are several
- `skymap` — `(ref_h, ref_w)` float32 — solved sky map (a hard link to the first sky term; `skymap_coverage` / `skymap_fisher` likewise, and `skymap_line*` link the last term when there are several)
- `skymap_coverage` — `(ref_h, ref_w)` int64 — frames touching each pixel
- `reproj_list` — list of HDD paths to the reprojected files (dataset of bytes)
- `num_maps` (attr) — number of chunk maps `K`
- `frame_scalar` — `(num_frames,)` float32 — per-frame DC scalar (only when the solve has one: `use_per_frame_scalar=True`, or a map with `det_groups`)

**Groups (one dataset per map):**
- `offsets/map_{m}` — `(num_frames, num_chunks_m)` float32 — per-frame per-chunk offsets, **expanded** from groups to per-frame
- `offset_coverage/map_{m}` — `(num_frames, num_chunks_m)` int64 — pixel count per (frame, chunk) (int32 ones for a template or polynomial-basis map)
- `offset_coverage_frac/map_{m}` — `(num_frames, num_chunks_m)` float64 — fraction of chunk pixels actually covered per frame (float32 ones for a template or polynomial-basis map)
- `chunk_maps/map_{m}` — `(det_h, det_w)` int — the chunk_map array used for map `m` (stored for analysis reproducibility)
- a map whose offset term carries known functions of data variables (a `coefficient` or a
  `basis` of `n` functions) stores its unknowns: `offsets/map_{m}` is `(num_frames,
  num_chunks_m * n)` (chunk-major: column `c * n + k`), with attrs `n_basis` and `basis` (the
  function and its variables); `CalFile.offset_basis(m)` returns `(n, description)`. Such an
  offset is not constant per chunk — the mosaic evaluates it per observation
  (`selfcal.pipeline.model_eval.BasisOffsetSubtractor`).

**Legacy schema (pre-multi-chunk-maps, still readable):**
Top-level `offset`, `offset_coverage`, `offset_coverage_frac` (no `offsets/` group, no `num_maps` attr, no `frame_scalar`). Both `Mosaicker.load_calibration` and `zodi_utils.load_cal_offsets` detect the schema and adapt; the latter folds `frame_scalar` into map-0 offsets for analysis-side compatibility with the legacy single-map subtraction semantics.

`selfcal_scripts/drivers/diff_cal_h5.py` understands both schemas — pass a legacy file and a new file and it compares the underlying arrays correctly.

## Reprojected `*.h5` schema

Written by `Reprojector.run_reproject` (one file per (exposure,
detector)), or directly by `selfcal.io.frames.write_frame` for data that do not
come from a WCS imager — the file is the solver's input contract. Filename
pattern: `exp_{exp_idx:04d}_det_{det_idx:02d}.h5`; `load_reproj_file` parses
the indices back out of the basename — keep the pattern stable. Compressed
with Zstd + byte-shuffle via `hdf5plugin`. Raw exposures are read through the
instrument's exposure reader (`ExposureLayout.reader`, default: FITS science +
DQ extensions), which may also return extra per-pixel planes and the
detector coordinates of every pixel (a focal plane, a windowed read-out).

Datasets:
- `sub_data` `(sub_w, sub_w)` float32 — reprojected science image.
- `sub_foot` `(sub_w, sub_w)` float16 — fractional footprint from the
  `reproject` library.
- `sub_bitmask` `(sub_w, sub_w)` int32 — DQ bitmask after reprojection
  (per-bit reprojected as float, thresholded at 0.01, then re-packed).
- `sub_mapping` `(2, sub_w, sub_w)` float32 — for each subframe pixel,
  the (x, y) sample location in the original *detector* frame. Used by
  every consumer to (a) build the bilinear-interp sparse matrix back to
  the chunk map and (b) sample per-pixel `det_BC` / `det_BW` for the
  wavelength maps (in the mosaic's cache pass, or in `wav_coadd`). With a
  reader that returns coordinates, these are the reader's (e.g. focal-plane)
  coordinates.
- `layers/<name>` `(sub_w, sub_w)` float32 — optional per-observation planes
  (a variance, a per-frame wavelength map, ...) from the reader, reprojected
  bilinearly; the source of *layer* data variables (`[model.variables]
  name = { layer = "name" }`).

Attributes:
- `sub_header` (bytes) / `det_header` (bytes) — FITS headers as strings;
  `load_reproj_file` reconstructs `sub_wcs` / `det_wcs` on demand. The
  keywords of `det_header` are the source of *header* frame variables
  (`[model.variables] time = { header = "MJD-AVG" }`,
  `selfcal.io.frames.frame_header_values`).
- `file_path` (str) — path to the source FITS the subframe came from.
- `ref_coords` `(4,)` int32 — `[y_min, y_max, x_min, x_max]` in the
  reference frame, where `sub_data` should be splatted back. Can extend
  outside the mosaic; `selfcal.geometry.map_helper.compute_crop` handles the clip.

Sub-frame side length sized to fit the detector diagonal at mosaic
resolution:
`sub_width = ceil(sqrt(2) * max(det_height, det_width) / (ref_reso/det_reso) * (1 + 2*padding_percentage))`.

Cached intermediates from the mosaic's cache pass live in `cache_dir` as
`cached_<original>.h5` in the **sparse** format (`attrs['format'] =
'sparse-v1'`): only the frame's nonzero-weight pixels are stored —
`ref_coords` (the nonzero-weight bbox in reference coordinates),
`sub_bbox` `[rmin, rmax, cmin, cmax]` (the same bbox in original sub-frame
coordinates), `attrs['shape']` (bbox shape), `mask` (packed bits of
`weight != 0` over the bbox, row-major), and value vectors in that order:
`data`, `weight`, optional `aux` `(K, n)`, and `bc` / `bw` (per-pixel LVF
band centre / width, present when the mosaic was asked for wavelength
maps). `coadd.read_cached_frame` / `load_cached_frame_dense` read this and
the legacy dense format (cropped `sub_data` / `sub_weight` arrays).

## Mosaic `*.fits` schema

Written by `Mosaicker.save_mosaic`. Multi-extension FITS — primary HDU is
empty; each map is its own `ImageHDU` carrying the mosaic WCS in the
header. `EXTNAME` is one of:

- `MEAN_MAP`, `MEAN_MAP_WEIGHT` — weighted mean and `sum(weight)`.
- `STD_MAP`, `STD_MAP_WEIGHT` — weighted std and weight.
- `SC_MEAN_MAP`, `SC_MEAN_MAP_WEIGHT` — sigma-clipped mean and weight.
- `WAV_MEAN_MAP`, `WAV_STD_MAP` — LVF wavelength maps (BUNIT=`um`),
  coadded inside the sigma-clip pass (`make_mosaic(wav_maps=...)`) or, when
  sigma clipping is off, by the standalone `wav_coadd` over the cache.

Header keys to know:
- `BUNIT` — taken from `Mosaicker.maps[name]['unit']` (the instrument's data
  unit for the sky maps — `'MJy/sr'` for SPHEREx, `'electron'` for Euclid —
  `'um'` for wavelength, `'Weight'` for the `_WEIGHT` companions).
- `MEANOFF` — global mean of valid map-0 offsets at mosaic time
  (`np.mean(offsets[0][offset_coverage_frac >= valid_chunk_thresh])`),
  stamped on every map HDU. If `normalize_offset=True` was used, this is
  the value subtracted from the offsets before they were applied — add
  it back to recover absolute brightness.

## Product sidecars (`<product>.json`)

A product written by an action of the Python API (a cal file, a tile cal, a stitched cal, an
N-pass product, a mosaic) has a sidecar next to it, `<product>.json`, written after the product
(which is itself written under a temporary name and renamed when complete):

| key | content |
| --- | --- |
| `inputs` | the settings and files that decided the product's bytes: instrument, reference grid (`ref.fits`, by content), job, model (template and map files by content), fit, solve numerics and frames (by name) for a cal; plus tile box and assignment for a tile cal; tile fingerprints for a stitched cal; first-pass fingerprint, pass number/type and pass settings for an N-pass product; cal fingerprint, reference grid, model, coadd, coadd numerics and frames for a mosaic. A function sent by value counts by its source, defaults and closure values; a hook object by its class and state |
| `fingerprint` | SHA-256 of the canonical JSON of `inputs` |
| `size`, `mtime_ns` | the product when the sidecar was written: a product written again afterwards (another size or modification time) is refused as `changed` |
| `record` | the action's record that made it |
| `adopted` | true when `field.adopt(...)` recorded a product made without one |

An action reuses a product only when its fingerprint matches what it would make; one made by other
inputs, or one without a sidecar (a TOML run's), is refused until it is adopted or the action
passes `overwrite=True`. TOML runs neither write nor read sidecars. The N-pass work directory keeps
`intermediates.json`, the fingerprints its moment dumps and sky exports were made from; the Python
API deletes those that differ before a run.

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

The byte-equality gates live in `selfcal_scripts/gates/` (see its README): `run_gates.sh <tag>` runs
the continuum and spectral cal gates (D3 Ch17 / D4 AromaticPAHfit, vs `*_gate_golden_stat.h5`), the
end-to-end D3 Ch17 cal + full mosaic, the npass n=3 probe (INIT + closed-form SKY + per-frame OFFSET
refit) and the Euclid EDFN recipe, each compared dataset by dataset with `gates/h5_diff.py` /
`gates/fits_diff.py` (or `selfcal_scripts/drivers/diff_cal_h5.py`); `run_m13_gate.sh` runs the npass
n=1 gate on the NEP M13 tile; `mode_lowering_snapshot.py` checks what every mode lowers to without a
solve. The pre-runner harnesses (`run_cal_baseline_test.py`, `regress_cal*.py`, the
`benchmark_d3_ch17_*` timing scripts) are archived under the gitignored `archive/scripts/benchmarks/`.
