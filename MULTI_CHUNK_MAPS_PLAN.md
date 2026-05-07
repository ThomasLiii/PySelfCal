# Plan: Multi-Chunk-Map Additive Offsets

## Context

The SelfCal pipeline currently solves the additive model `d_i = s(p_i) + o[frame_or_group, c(i)] + ε`, where `c(i)` is the chunk ID at pixel `i` under a single user-supplied `chunk_map`. This is enough for one spatial parameterization at a time, but real detector systematics often have multiple uncorrelated structures (e.g., per-channel slow drift × per-column gradient × per-detector-fixed pattern). With one chunk map you can only resolve one of them at a time.

**Goal**: extend to `d_i = s(p_i) + Σ_m o^(m)[frame_or_group_m, c_m(i)] + ε` with `K ≥ 1` chunk maps fed in simultaneously. Each map independently supports the existing advanced solve modes (`det_groups`, `det_template`, `mean_offsets`, spatial adjacency regularization). The system stays linear, so LSQR continues to work — just more columns.

This supersedes the abandoned `feat/multiplicative-corrections` branch (gain feature). Branch will be renamed `feat/multi-chunk-maps`.

**User-confirmed scope decisions:**
- API is *always-list* (no polymorphic ndarray-or-list). Existing single-map call sites must wrap their map as `[chunk_map]`. This eliminates a code path and simplifies internals.
- Cal file always writes the new `offsets/` group schema (no dual schema). Analysis scripts that read `cal_*.h5` are updated to handle both old (top-level `offset`) and new (`offsets/map_m`) layouts so existing on-disk results stay readable.

## Mathematical model

```
d_i = s(p_i) + Σ_{m=0..K-1} o^(m)[g_m(k), c_m(i)] + ε_i
```

- `s` — common sky map (one block in `x`)
- `o^(m)` — per-map offset, indexed by frame-or-group `g_m(k)` and chunk `c_m(i)` of map `m`
- `g_m(k)` is the frame→group mapping for map `m` (defaults to identity, optionally `det_groups[m]`)
- `c_m(i)` is the chunk ID for pixel `i` under map `m`'s `chunk_map`

Per-map adjacency reg, mean-offset constraints, and template-amplitude mode are independent across `m`. The per-frame scalar bias column stays single (one per frame, added if any map uses `det_groups`) — adding K scalars per frame would be degenerate.

## `x` vector layout (single source of truth)

```
[ sky | offset_block_0 | offset_block_1 | ... | offset_block_{K-1} | scalar_block ]

col_bases[m] = num_sky + Σ_{m'<m} (N_groups[m'] · N_chunks[m'])
             = num_sky + Σ_{m'<m} (N_frames if det_template[m'] is None else 1) · N_chunks[m'] (for non-template; template mode block size = num_frames since it's α per frame)

scalar_col_start = col_bases[K]   (only set if any map uses det_groups)
```

`col_bases` is computed once in `setup_lsqr` and passed to every worker; it's the canonical contract between matrix construction and `parse_x` / `compute_x0_from_Ab`.

## Critical files to modify

### Pipeline core (LSQR build/solve)

- **`SelfCal/lsqr.py`**
  - `_prep_lsqr` (line 17): loop over maps `m=0..K-1`. For each, emit one `(O_rows, O_cols, O_data)` block using `chunk_contribs[m]`, `frame_to_groups[m][index]`, `num_chunks[m]`, `det_templates[m]`, with column index offset by `col_bases[m]`. Adjacency reg rows are per-map, skipped for `det_template[m] is not None` and for `reg_weight[m] == 0`. Track per-map row-offset arithmetic carefully — current code starts adj rows at `num_valid_pixels`; with K maps it's `num_valid_pixels + Σ_{m'<m} num_constraints[m']`.
  - `_prep_lsqr_batch_worker` (line 153): SHM reconstruction loops over `task_params['chunk_maps_meta']` (list of `(name, shape, dtype)` tuples) and `'adj_metas'` similarly, populating `shm_arrays['chunk_maps']` and `shm_arrays['adj_infos']` as length-K lists.
  - `setup_lsqr` (line 226): polymorphic input normalization removed (always list). Per-map SHM segments (`chunk_map_shm_name_{m}`, `adj_shm_name_{m}_{0,1}`). Mean-offset constraint rows (block ~lines 450–472) iterate per map; weight stays default 10.0 but should accept `mean_offset_weight` as length-K list to allow per-map tuning.
  - `apply_lsqr` (line 567): unchanged — it's column-layout-agnostic.

- **`SelfCal/subframe.py`**
  - `_prep_subframe` (line 10): accept `chunk_maps` (list of K arrays) instead of `chunk_map`. Build `chunk_contribs` = list of K sparse matrices via existing `compute_chunk_contrib` ([`MapHelper.py:323`](../../spherex/selfcal/SelfCal/MapHelper.py#L323)) called once per map. `interp_matrix` is built once and reused — only the per-map chunk contribution call repeats.
  - For the **mosaic path** (when called by `coadd.compute_coadd_map` rather than `_prep_lsqr`): replace the single-`chunk_offset` block (lines 54–59) with a loop that accumulates `Σ_m det_offset_funcs[m](chunk_maps[m], chunk_offsets[m])` into `total_grid_offset`, then a single `det_to_sub` and subtraction. Cache `interp_matrix` shape so duplicate work is avoided when shapes coincide.

- **`SelfCal/coadd.py`**
  - `compute_coadd_map` (line 437): replace `offset_list` (single ndarray of shape `(num_frames, num_chunks)`) with `offset_lists` (list of K such arrays). At per-frame slicing, build `current_offsets = [offset_lists[m][frame_idx] for m in range(K)]`. Pass `chunk_maps` and `det_offset_funcs` lists into `_prep_subframe`.

- **`SelfCal/solution.py`**
  - `parse_x(x, ref_shape, num_offset_groups_list, num_chunks_list, num_frames=None)` — returns `(skymap, [det_offset_0, ..., det_offset_{K-1}], frame_scalar)`. Returning a list (not dict) preserves index ordering matching the input `chunk_maps`.
  - `encode_x(skymap, offsets_list)` — concatenates sky + flattened per-map offsets in order.
  - `compute_x0_from_Ab(A, b, ref_shape, col_bases)` — diagonal LS warm start works unchanged in spirit (per-column); accept `col_bases` so it knows where sky ends. No per-map logic needed inside.

### User-facing API (Calibrator / Mosaicker)

- **`SelfCal/PipelineWrapper.py`**
  - `Calibrator.setup_lsqr` (line 134): signature changes to `chunk_maps` (list), `det_groups_list`, `det_templates`, `mean_offsets_list`, `adj_infos`, `reg_weights`, `offset_regularization_list` — all length K. Validate lengths up front (raise `ValueError` with clear message if mismatched). Store on self: `self.chunk_maps`, `self.frame_to_groups`, `self.num_offset_groups_list`, `self.num_chunks_list`, `self.col_bases`, `self.det_templates`. Drop the now-redundant single-form attrs (`self.chunk_map`, `self.num_offset_groups`, etc.) — break callers cleanly rather than maintain compat shims.
  - `Calibrator.apply_lsqr` (line 158): use `self.col_bases` instead of `self.num_offset_groups` for parsing. Returns are now lists.
  - `Calibrator.save_calibration` (line 206): always write the new schema (see below).
  - `Calibrator.load_calibration`: detect new vs. old schema (see Analysis scripts section); new-schema runs populate `self.offsets` (list), old-schema runs populate `self.offsets = [old_offset]` so internal code is uniform.
  - `Calibrator.get_det_offset()`: returns the list now. If callers want a specific map, they index it: `cc.get_det_offset()[m]`.
  - `Mosaicker.make_mosaic` (line 299): signature `make_mosaic(chunk_maps, ..., det_offset_funcs=...)` — lists, not single arrays. `_prep_subframe` accumulates over maps.
  - `_expand_offset` (~lines 194–204): per-map expansion (groups → frames). Operates on `self.offsets` list.

### Cal file schema (new only)

`Calibrator.save_calibration` writes:

```
cal_*.h5
├── skymap                    (ref_h, ref_w) float32
├── skymap_coverage           (ref_h, ref_w) int32
├── reproj_list               attribute, list of source paths
├── num_maps                  attribute, int = K
├── offsets/
│   ├── map_0                 (num_frames, num_chunks_0) float32
│   ├── map_1                 (num_frames, num_chunks_1) float32
│   └── ...
├── offset_coverage/
│   ├── map_0                 (num_frames, num_chunks_0) int32
│   └── ...
├── offset_coverage_frac/
│   ├── map_0                 (num_frames, num_chunks_0) float32
│   └── ...
└── chunk_maps/
    ├── map_0                 (det_h, det_w) int32 — the chunk_map array used
    └── ...
```

`chunk_maps/map_m` is stored so analysis can recover them without round-tripping config. `frame_scalar` is stored at top level if `num_scalar_cols > 0` (`(num_frames,)` float32) — single block, not per-map.

### Analysis side (read-only consumers)

The analysis scripts under `analysis/analysis_script/` read cal files via `zodi_utils.py` and direct `h5py.File` calls. These need to handle **both** schemas (old top-level `offset` for existing on-disk results, new `offsets/map_m` for fresh runs).

- **`analysis/analysis_script/zodi_utils.py`**: add `load_cal_offsets(cal_path)` helper that returns a dict `{0: offset_0, 1: offset_1, ...}` (or single `{0: offset_0}` for old files). Update `cal_path()` callers (`build_multichannel_cache.py`, `build_perchunk_coords.py`, `meeting_plots.py`, `verify_stack.py`, `plot_chunkmap.py`, the `task*` and `plot_*` one-offs) to consume the dict and pick the map index they want — usually `[0]` for legacy single-map analyses.
- The `CAL_FILE_TEMPLATE` naming pattern (`cal_Detector{D}_NumSub10_NumCh34_NumCol3_Ch{ch}_*.h5`) doesn't need changes; the new schema lives inside the file, not the filename. A `NumMaps` field could be appended to the template if the user later wants distinct filenames per K, but defer that until needed.
- **`SelfCal/PipelineWrapper.py:Calibrator.load_calibration`**: schema-detection logic at the top — if `'offsets'` group exists, read new schema; else read old top-level `offset` dataset and wrap as `self.offsets = [old_offset]`. Same for coverage arrays.

### Driver script

- **`selfcal_scripts/run_cal_v2.py`**: wrap the existing `det_chunk_map` (and any `det_template`, `frame_to_group`, `mean_offsets`, `adj_info`) in length-1 lists at the call site. Since the user's typical run is K=1 today, this is the minimum migration needed for existing pipelines to continue working under the new API.

## Reused existing utilities

- **`MapHelper.py:323` `compute_chunk_contrib`** — call once per map in `_prep_subframe`.
- **`MapHelper.py` `make_linear_interp_matrix`** — built once per subframe, shared across maps.
- **`SPHERExUtility.py` `compute_column_adjacency` / `compute_subchannel_adjacency`** — call once per map; the user picks which adjacency style fits each map.
- **`solution.py:30` `compute_x0_from_Ab`** — diagonal LS warm start; only signature changes (accept `col_bases`).
- **`_state.py` `_hdd_io_semaphore`** — unchanged; throttle still applies per file regardless of K.

## Identifiability — required defaults

When K ≥ 2, the system has up to K-1 dimensions of redundancy: a constant `c` shifted into `o^(0)` and `-c` into `o^(1)` (over their shared spatial support) leaves the data unchanged. Mandatory mitigations, recommended in the docstring of `Calibrator.setup_lsqr` and enforced as defaults in `run_cal_v2.py`:

1. **Anchor all but one map's per-frame mean to 0** via `mean_offsets_list = [None, np.zeros(num_frames), np.zeros(num_frames), ...]`. The "primary" map (m=0) stays free; the rest are mean-zero per frame, breaking K-1 degrees of freedom.
2. **Per-map adjacency reg**: every map with spatial structure should have `reg_weights[m] > 0`.
3. **Document the sharp edge**: if `chunk_maps[m]` is a strict refinement of `chunk_maps[0]` (e.g., m=0 is per-channel, m=1 is per-(channel,column)), then m=1's mean-zero constraint must align with m=0's partition (mean-zero per channel, not per frame). Otherwise the constraints fight. Provide a helper `make_per_partition_mean_zero(chunk_map_fine, chunk_map_coarse)` if this case is common.
4. **Identical maps are unsolvable** regardless of regs — flag in the docstring, raise a warning if `setup_lsqr` detects two maps with identical content.

## Implementation order (4 staged commits/PRs)

The internals carry the regression risk; the user API is mechanical. Stage them so the K=1 case is validated bit-for-bit before K>1 ever runs.

**Commit 1 — Internal refactor, K hardcoded to 1.**
Convert `setup_lsqr` / `_prep_lsqr` / `_prep_subframe` / `parse_x` / `compute_x0_from_Ab` to operate on `chunk_maps = [chunk_map]`, `col_bases`, list-of-1 internally. User API in `Calibrator.setup_lsqr` and `Mosaicker.make_mosaic` unchanged. Cal file schema unchanged. Verify byte-identical `cal_*.h5` against pre-refactor output on a representative `run_cal_v2.py` config (Test A below).

**Commit 2 — Lift K>1 in core.**
Allow K-element lists end-to-end through the LSQR pipeline. `Calibrator` API still single-map externally. Add a hand-built K=2 smoke test: pass `chunk_maps=[m, m]` (duplicate) and confirm the solver converges with reg + mean_offsets in place — expected behavior is offsets split between the two maps with mean reflecting `mean_offsets` constraints.

**Commit 3 — User-facing always-list API.**
Update `Calibrator.setup_lsqr`, `Calibrator.save_calibration`, `Calibrator.load_calibration`, `Mosaicker.make_mosaic` signatures. Migrate `run_cal_v2.py`. Cal schema flips to new layout. `Calibrator.load_calibration` and analysis-side `zodi_utils.py` get dual-schema readers (old top-level `offset` + new `offsets/` group).

**Commit 4 — Mosaicker multi-map application.**
`_prep_subframe` mosaic-path accumulation, `coadd.compute_coadd_map` `offset_lists`, identifiability helpers (`make_per_partition_mean_zero`).

Each commit is independently runnable (`run_cal_v2.py` works after each), with Commit 1 being the regression gate.

## Branch rename

Before Commit 1, rename the active branch:

```bash
git branch -m feat/multiplicative-corrections feat/multi-chunk-maps
git push origin --delete feat/multiplicative-corrections
git push -u origin feat/multi-chunk-maps
```

(I'll execute these once the plan is approved, before starting Commit 1.)

## Verification

No test suite exists; integration scripts are the oracles.

**Test A — regression gate (Commit 1)**
Save baseline: run `python selfcal_scripts/run_cal_v2.py` on the current `main` and copy the resulting `cal_*.h5` aside. After Commit 1, re-run with the same config and diff every dataset element-wise:
```python
import h5py, numpy as np
with h5py.File('baseline.h5') as a, h5py.File('refactored.h5') as b:
    for key in ['offset', 'skymap', 'skymap_coverage', 'offset_coverage', 'offset_coverage_frac']:
        np.testing.assert_array_equal(a[key][...], b[key][...])
```
Any element-wise diff = bug. `np.allclose` with `rtol=0, atol=0` (i.e., true equality) — the operations are deterministic given fixed seeds and fixed thread counts.

**Test B — K=2 sanity (Commit 2)**
Build a 1-chunk dummy map alongside the real map: `chunk_map_dummy = np.zeros_like(chunk_map_real, dtype=np.int32)`. Solve with `chunk_maps=[real, dummy]`, `mean_offsets_list=[None, np.zeros(num_frames)]`. The dummy map's offsets should be near-zero (because mean-zero is enforced); the real map's should match the original single-map solve to within solver tolerance. Then flip: `mean_offsets_list=[np.zeros(num_frames), None]` — now the dummy absorbs per-frame DC and the real map has mean-zero per frame. Sum `real_offset[k, c] + dummy_offset[k, 0]` should approximately equal the original single-map `offset[k, c]` for every `(k, c)`.

**Test C — Mosaicker (Commit 4)**
Run `Mosaicker.make_mosaic(chunk_maps=[real, dummy], chunk_offsets=[off_real, off_dummy])` using the cal from Test B. Compare against single-map mosaic from the original `cal_*.h5` — should match within solver tolerance. Use `notebooks/spherex_selfcal_demo.ipynb` as the integration harness.

**Test D — Identifiability stress (Commit 4)**
Two physically meaningful maps (e.g., column-adjacency map + subchannel-adjacency map) on the same detector. Run with full default regs vs. without `mean_offsets`. The unconstrained run should hit `iter_lim` with a high LSQR residual norm — flag as expected, document in docstring.

There's no automated CI for any of this. After each commit, the human runs Test A (mandatory) and a quick spot-check of `meeting_plots.py` output (does Fig 3b still look right?) before continuing.
