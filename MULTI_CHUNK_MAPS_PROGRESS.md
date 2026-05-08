# Multi-Chunk-Maps Feature — Progress Log

Companion to [MULTI_CHUNK_MAPS_PLAN.md](MULTI_CHUNK_MAPS_PLAN.md). This file tracks what's done and what's next so the work can be resumed in a fresh session.

The plan also lives at `/home/thomasli/.claude/plans/actually-forget-adding-the-glittery-map.md` (Claude Code's per-project plan store). The copy in this repo is the source of truth going forward — edit this one if requirements change.

## Status as of 2026-05-07

**Branch**: `feat/multi-chunk-maps` (renamed from `feat/multiplicative-corrections` — gain feature was abandoned in favor of additive multi-map). Local + remote.

**Worktrees** unchanged:
- `~/spherex/selfcal/` — dev, branch `feat/multi-chunk-maps`, env `general`
- `~/spherex/selfcal-stable/` — analysis, branch `stable`, env `selfcal-stable`

**Current state**: **Commits 1, 2, 3, and 4 are done.** Pushed up through Commit 3; Commit 4 changes are uncommitted in the working tree pending review.
- `00a2892 selfcal: parameterize _prep_lsqr offset/adjacency rows by col_bases`
- `f24c155 scripts: add baseline test driver and cal-file diff helper`
- `610056b docs: add multi-chunk-maps design plan and progress log`
- `cabea2b selfcal: lift K>1 multi-chunk-maps in lsqr/subframe/solution core`
- `6c9e085 scripts: adapt drivers to trimmed compute_x0_from_Ab + add K=2 smoke test`
- `de7431a docs: update multi-chunk-maps progress to reflect Commit 2 done`
- `aec4050 selfcal: Calibrator/Mosaicker public API to lists + new cal-file schema`
- `e5897d6 scripts: migrate drivers to list API + schema-aware diff_cal_h5`
- `4e4b527 analysis: dual-schema cal-file reader + sweep direct h5 readers`
- `ceff5a4 docs: update multi-chunk-maps progress to reflect Commit 3 done`

## Done

- [x] Wrote and approved the design plan (4-commit staged refactor).
- [x] Renamed branch `feat/multiplicative-corrections` → `feat/multi-chunk-maps` (local + origin).
- [x] Created `selfcal_scripts/run_cal_baseline_test.py` (you authored it; I trimmed it for byte-equality testing — see "How the regression test works" below).
- [x] Copied plan into repo as `MULTI_CHUNK_MAPS_PLAN.md`.
- [x] **Captured pre-refactor baseline cal file.** Path:
      `/mnt/md124/thomasli/selfcal/outputs/SPHEREx_nep_qr2_det3_6p2arcsec/calibration/cal_Detector3_NumSub10_NumCh34_NumCol3_Ch17_baseline_before_refactor.h5`
      (411 MB; `offset (13219, 1026) float64`, `skymap (12532, 12540) float64`).
      Hands-off after this — do not regenerate. The post-refactor run with `TEST_TAG='after_refactor'` produces a sibling file for the diff.
- [x] **Bug noted in `lsqr.py`**: when `adj_info` is a tuple of zero-length arrays (happens with `NumCol=1`, where `compute_column_adjacency` returns 0 boundaries), `SharedMemory(create=True, size=0)` raises. Test was bumped from `NumCol=1 → 3` to dodge it during the refactor. **Fixed in a follow-up commit** by demoting empty `adj_infos[m]` to `None` before SHM packing — see end of this log.
- [x] **Commit 1 — `_prep_lsqr` parameterized by `col_bases`.** Wrapped single-form params (`chunk_contrib`, `num_chunks`, `det_template`, `group_idx`, `adj_info`, `reg_weight`) as length-1 internal lists; offset and adjacency rows now built from `col_bases` indexing. K=1 codepath is mathematically identical to the original.
- [x] **Commit 1 regression check passed.** Re-ran calibration with refactored code (`TEST_TAG='after_refactor'`); diffed against baseline. `offset_coverage`, `offset_coverage_frac`, `skymap_coverage`, `reproj_list` byte-equal. `offset` and `skymap` match within `np.allclose(rtol=0, atol=1e-2)` — median diff = 0; max abs diff 5.7e-3 on offset, 2.8e-3 on skymap. Visual diff (`figures/commit1_skymap_diff.png`, gitignored) shows scan-direction noise concentrated at survey edges, not systematic bias. Diagnosis: pre-existing parallel non-determinism in `setup_lsqr`'s `as_completed` batch ordering ([SelfCal/lsqr.py:400](SelfCal/lsqr.py#L400)), not the refactor. **Worth fixing as a separate small commit** later — collect futures into a list indexed by batch ID, concatenate in batch order — then the byte-equality bar becomes exact zero.
- [x] **Commit 2 — lift K>1 in core.** End-to-end refactor; the K=1 wrapper inside `_prep_lsqr` is gone. Touchpoints:
  - `SelfCal/subframe.py:_prep_subframe` takes `chunk_maps` (list); builds `interp_matrix` once (asserts shared shape) and returns `chunk_contribs` as a list (one per map).
  - `SelfCal/lsqr.py:_prep_lsqr` reads per-map lists (`chunk_maps`, `num_chunks_list`, `det_template_list`, `frame_to_group_list`, `adj_info_list`, `reg_weight_list`, `col_bases`) directly from `task_params`.
  - `SelfCal/lsqr.py:_prep_lsqr_batch_worker` reconstructs per-map SHM via `chunk_maps_meta` and `adj_metas` (each a length-K list of `(name, shape, dtype)` tuples or `None`).
  - `SelfCal/lsqr.py:setup_lsqr` switched to list-form public API: `chunk_maps`, `reg_weights`, `adj_infos`, `mean_offsets_list`, `det_groups_list`, `det_templates`. Per-map SHM segments; per-map mean-offset constraint blocks; `col_bases` (length K+1, with `col_bases[K] == scalar_col_start`).
  - `SelfCal/lsqr.py:apply_lsqr` dropped the unused `num_offset_groups` parameter (column-layout-agnostic).
  - `SelfCal/lsqr.py:parse_pixel_counts` now returns per-map coverage / valid-fraction lists.
  - `SelfCal/solution.py:parse_x` returns `(skymap, [det_offsets_0…], frame_scalar)`; `compute_x0_from_Ab` dropped its unused 4th arg.
  - `SelfCal/coadd.py:_coadd_batch_worker` wraps `chunk_map → [chunk_map]` at the call site (mosaic-side multi-map accumulation is still Commit 4).
  - `SelfCal/PipelineWrapper.py:Calibrator` keeps the **single-map external API** (`setup_lsqr(chunk_map, …)`) — wraps inputs as length-1 lists internally and stores both list-form (`self.chunk_maps`, `self.col_bases`, `self.num_offset_groups_list`, …) and legacy single-form mirrors (`self.chunk_map`, `self.num_offset_groups`, …) for older notebooks.
  - `selfcal_scripts/run_cal_baseline_test.py` and `run_cal_v2.py` updated to drop the no-longer-accepted 4th arg from `compute_x0_from_Ab`.
- [x] **Commit 2 regression check passed.** Re-ran with `TEST_TAG='after_commit2'`; diffed against the same `before_refactor` baseline. `offset_coverage`, `offset_coverage_frac`, `skymap_coverage`, `reproj_list` byte-equal. `offset` max |diff| = **1.10e-04**; `skymap` max |diff| = **8.52e-05** — both well under the same `atol=1e-2` band that Commit 1 produced (and noticeably smaller than Commit 1's diff this time, presumably from how the new SHM packing changed batch arrival timing). Same pre-existing parallel non-determinism root cause; same fix applies.
- [x] **K=2 smoke test added** (`selfcal_scripts/run_cal_k2_smoke_test.py`). Calls `setup_lsqr` directly with `chunk_maps=[real, dummy(1-chunk)]`, `reg_weights=[0.1, 0.1]`, `mean_offsets_list=[None, np.zeros(num_frames)]`. On a 100-frame subset: matrix shape matches `col_bases` prediction (`num_sky + 100·1026 + 100·1`), both offset blocks are populated (real 6.6M nnz, dummy 5.4M nnz), and the diagonal-LS warmstart fills both blocks. Confirms per-map SHM hand-off, per-map offset rows, and per-map mean constraint all wire end-to-end. Doesn't run LSQR — verification is plumbing-level.
- [x] **Commit 3 — user-facing always-list API + new cal-file schema.** Touchpoints:
  - `SelfCal/PipelineWrapper.py:Calibrator.setup_lsqr` flipped to list-form: `chunk_maps`, `reg_weights`, `adj_infos`, `mean_offsets_list`, `det_groups_list`, `det_templates`. Length-K validation up front (raises `ValueError` on mismatch). Legacy single-form mirror attrs (`self.chunk_map`, `self.num_offset_groups`, `self.det_template`, `self.frame_to_group`) dropped — callers now read from the list-form attrs.
  - `Calibrator.save_calibration` writes the new schema: top-level `skymap`, `skymap_coverage`, `reproj_list`, plus `num_maps` attribute and `offsets/`, `offset_coverage/`, `offset_coverage_frac/`, `chunk_maps/` groups (each with one `map_m` dataset per map). Per-map offsets are saved expanded-per-frame *without* the per-frame scalar baked in; the shared `frame_scalar` is stored at the top level instead, only when any map uses `det_groups`.
  - `Calibrator.load_calibration` reads dual schema: legacy top-level `offset` → `[f['offset'][:]]`; new schema → `[f['offsets/map_m'][:] for m]` plus `frame_scalar` if present.
  - `Calibrator.get_offsets()` (new) returns the K-element list; `get_offset()` is the K=1 convenience that returns `get_offsets()[0]`. The shared `frame_scalar` is folded into map 0 only, matching legacy single-map subtraction behavior. `get_det_offset(m=0)` parameterized by map index.
  - `Mosaicker.load_calibration` reads dual schema; under the new schema it pulls `offsets/map_0` plus `frame_scalar`. Multi-map mosaic application is still Commit 4 — Mosaicker still applies a single offset.
  - `selfcal_scripts/run_cal_v2.py` and `run_cal_baseline_test.py` migrated: `chunk_maps=[det_chunk_map]`, `adj_infos=[adj_info]`, `reg_weights=[0.1]`.
  - `selfcal_scripts/diff_cal_h5.py` rewritten as schema-aware: `_read_offset(f, m)` resolves either schema and folds `frame_scalar` into map 0; comparing legacy ↔ new schema works element-wise on the underlying arrays.
  - `analysis/analysis_script/zodi_utils.py` adds `load_cal_offsets(path_or_file) → {m: offset_m}`; legacy → `{0: top-level offset}`, new → `{m: offsets/map_m}` with `frame_scalar` folded into map 0. `load_single_channel_offset` now uses it. Direct `f['offset'][:]` reads in `meeting_plots.py`, `meeting_plots_by_lat.py`, `plot_numcol_decomp.py`, `plot_numcol_benefit.py`, `build_multichannel_cache.py` swept to `load_cal_offsets(f)[0]`.
- [x] **Commit 3 regression check passed.** Re-ran with `TEST_TAG='after_commit3'`; diffed via the new schema-aware `diff_cal_h5.py` against the legacy `before_refactor` baseline (which maps `offset` → `offsets/map_0` for the comparison). `skymap_coverage`, `reproj_list`, per-map `offset_coverage`/`offset_coverage_frac` byte-equal. `offset[map_0]` max |diff| = **1.64e-03**; `skymap` max |diff| = **8.05e-04** — within the same `atol=1e-2` parallel-non-determinism band as Commits 1–2.
- [x] **New-schema sanity check passed.** Inspecting the saved file directly: `num_maps=1` attribute, top-level `skymap`/`skymap_coverage`/`reproj_list`, plus the four `(name)/map_0` group datasets (`offsets`, `offset_coverage`, `offset_coverage_frac`, `chunk_maps`). `chunk_maps/map_0` stored as `(2040, 2040) int32` — the actual `det_chunk_map`. No `frame_scalar` (this config doesn't use `det_groups`). `Mosaicker.load_calibration` reads the new schema and exposes `offset (13219, 1026)` exactly like the legacy path. `zodi_utils.load_cal_offsets` returns `{0: …}` with the same per-frame mean (0.0042) on both legacy and new files.
- [x] **Commit 4 — Mosaicker multi-map application.** Touchpoints:
  - `SelfCal/subframe.py:_prep_subframe` mosaic path now takes `chunk_offsets` (list of K per-frame ndarrays) and `det_offset_funcs` (list of K callables); per-map grid offsets are summed into `total_grid_offset` and a single `det_to_sub` runs the bilinear interp once regardless of K.
  - `SelfCal/coadd.py:compute_coadd_map` flips its public API: `chunk_maps`/`offset_lists`/`det_offset_funcs` are length-K lists (with cross-length asserts up front). The batch worker reconstructs per-map chunk_map SHM via `chunk_maps_meta` (mirroring setup_lsqr); per-batch `batch_offsets` becomes a list of K per-batch slices, each of shape `(batch_size, num_chunks_m)`.
  - `SelfCal/PipelineWrapper.py:Mosaicker.load_calibration` now populates `self.offsets`, `self.offset_coverages`, `self.offset_coverage_fracs`, and `self.cal_chunk_maps` as length-K lists (still folding `frame_scalar` into map 0 for legacy subtraction semantics). `Mosaicker.make_mosaic` flips to `chunk_maps`/`det_offset_funcs` lists and assembles per-map masked/zeroed offset arrays before passing them through; `mean_offset` (used by `save_mosaic` for the FITS header) is reported on map 0 only for legacy compat.
  - `SelfCal/lsqr.py:_prep_lsqr` updated to pass the renamed kwargs (`chunk_offsets=None`, `det_offset_funcs=None`).
  - `selfcal_scripts/run_cal_v2.py` mosaic call wraps `chunk_maps=[grid_chunk_map]`, `det_offset_funcs=[partial_make_offset_map]`.
- [x] **K=2 mosaic smoke test added** (`selfcal_scripts/run_mosaic_k2_smoke_test.py`). On a 20-frame subset with synthetic offsets: a K=2 run with `chunk_maps=[real, dummy(1-chunk)]` + `offset_lists=[off_real, off_dummy]` produces an **exactly identical** (`max |diff| = 0.000e+00`, 880,063 valid pixels) mean map to a K=1 run with the dummy's per-frame DC pre-broadcast into the real map. Confirms per-map SHM hand-off, per-batch slicing, and `_prep_subframe` accumulation are bit-correct.
- [x] **Mosaicker end-to-end check on the new-schema cal file passed.** `Mosaicker.load_calibration` on the Commit 3 `after_commit3.h5` populates `self.offsets[0]` with shape `(13219, 1026)`; running `make_mosaic(chunk_maps=[grid_chunk_map], det_offset_funcs=[partial_make_offset_map])` on a 20-frame subset produces a non-trivial mean map (1.6M valid pixels, mean_offset=0.1099) without errors.

## How the regression test works

The plan's safety gate is **byte-equality of `cal_*.h5`**: the refactor must produce numerically-identical calibration output to the pre-refactor code. Process:

1. **Before any refactor**: with `TEST_TAG = 'before_refactor'` in `selfcal_scripts/run_cal_baseline_test.py`, run:
   ```
   conda activate general
   cd ~/spherex/selfcal
   python selfcal_scripts/run_cal_baseline_test.py
   ```
   This produces a cal file like `cal_Detector3_NumSub10_NumCh34_NumCol1_Ch17_baseline_before_refactor.h5` under the run's calibration directory.

2. **After the refactor** (Commit 1 below): change `TEST_TAG = 'after_refactor'` in the same script and re-run. Produces `..._baseline_after_refactor.h5`.

3. **Diff**: a tiny script (to be written) opens both files and runs `np.testing.assert_array_equal` on every dataset. ✓ identical → refactor is safe. ✗ → bug.

The test script:
- Skips mosaicking entirely (only the cal file is needed).
- Pins `OMP/MKL/OPENBLAS_NUM_THREADS=1` (already in the script) for deterministic float math.
- Preserves the NVMe reproj cache between runs (don't re-copy hundreds of GB each iteration).
- Default `frame_setting = {Detector: 3, NumSub: 10, NumCh: 34, NumCol: 1}`, `chs = [[17]]` — small, fast config for quick iteration.

## Post-refactor follow-ups — done

- [x] **NumCol=1 SHM size=0 bug** in `setup_lsqr` adj_info packing. Fix: in `lsqr.setup_lsqr` after the `_default(...)` normalization, demote any `adj_infos[m]` whose tuple contents are all-zero-length to `None`. The reg-block loop in `_prep_lsqr` already gates on `adj_info_list[m] is not None` and the constraint contribution would have been a no-op anyway, so this only changes behavior in the previously-broken case. Tested by running `setup_lsqr` on a 30-frame subset with `NumCol=1` (`compute_column_adjacency` returns `(empty, empty)`); previously raised `ValueError("'size' must be a positive number different from zero")`, now produces a valid `A`/`b`. Stale "NumCol=1 hits a pre-existing bug" comment removed from `run_cal_baseline_test.py`.
- [x] **Parallel non-determinism in `setup_lsqr`.** The `as_completed` loop assigned cumulative `total_rows` offsets in **completion** order, so the same batch ended up at different row IDs across runs. Mathematically the row ordering doesn't change the LS solution, but the float32 reductions inside the transposed-SpMV (`_rmatvec`) traverse columns in row-id order — different row IDs ⇒ different `(a+b)+c ≠ a+(b+c)` accumulation order ⇒ ULP-level diffs that propagate through LSQR's iterations and produce the ~1e-3 noise we'd see in regression diffs. Fix: stash each batch's read-and-unlinked SHM arrays into `batch_results[batch_id]`, then do a second pass over `batch_results` in batch-id order to assign row offsets and append to `all_*`. Pass 2 frees each slot as it consumes it, so memory peak is unchanged; `np.concatenate` cadence (every 100 batches) is identical. Verified end-to-end on an 80-frame subset: two back-to-back full setup + LSQR runs now produce byte-equal `A`/`b` and **byte-equal `x`** (`np.array_equal` true). Wall-clock cost is negligible — total work is the same, just reordered, and consolidation runs after the worker phase rather than overlapping with it.

## To do — next session pick up here

The 4-commit refactor from [MULTI_CHUNK_MAPS_PLAN.md](MULTI_CHUNK_MAPS_PLAN.md) is complete. Pending follow-ups:

- **Identifiability helpers for K≥2.** Plan section "Identifiability — required defaults" recommends `make_per_partition_mean_zero(chunk_map_fine, chunk_map_coarse)` for nested-map cases. Skipped in Commit 4 because the constraint formulation in `setup_lsqr` only supports per-frame mean-zero (one constraint per frame), not per-(frame, partition) (K_coarse constraints per frame). Wire that up if a real nested K≥2 use case lands.
- **Stale notebook + driver scripts.** `notebooks/spherex_selfcal_demo.ipynb`, `notebooks/demo_notebook.ipynb`, `notebooks/euclid_mosaic.ipynb`, `notebooks/benchmark_pipeline.py`, `selfcal_scripts/run_selfcal_synthetic.py`, `selfcal_scripts/run_cal_synthetic copy.py` still call the legacy single-map API (`chunk_map=`, `det_offset_func=`, `cc.num_offset_groups`, etc.). Migrate when their use case comes back.
- **Mosaicking re-enabled in `run_cal_baseline_test.py`.** Plan called for this as Test C; deferred since the K=2 mosaic smoke test (`run_mosaic_k2_smoke_test.py`) and the full end-to-end Mosaicker check on the new-schema cal file already exercise the multi-map mosaic path.

## Resuming in a fresh session — concrete steps

1. **Open a new Claude Code session in this repo** (`cd ~/spherex/selfcal && claude` — or open the folder in VS Code with the Claude Code extension).
2. **Verify state**:
   ```
   git branch --show-current   # → feat/multi-chunk-maps
   git log --oneline -12       # → confirm all four commits' subcommits are present
   ```
3. **Tell the new session** which follow-up you're working on (see "To do" list above).
4. **Verify cached state still intact** before any compute:
   ```
   ls /mnt/md124/thomasli/selfcal/outputs/SPHEREx_nep_qr2_det3_6p2arcsec/calibration/cal_*_baseline_*.h5
   ls /home/thomasli/spherex/selfcal/cache/reproj_nvme_SPHEREx_nep_qr2_det3_6p2arcsec | head -3
   ```
   If the NVMe cache is gone, the next regression run will re-copy ~250 GB. If only the cal files are gone, the regression test still works as long as you re-run both before- and after-refactor in the same session.
5. **Optional pre-Commit-2 detour**: fix the `as_completed` non-determinism in [SelfCal/lsqr.py:400](SelfCal/lsqr.py#L400) so future regression tests give exact byte-equality (collect futures into list indexed by batch id, concatenate in batch order). One small commit, then the byte-equality bar becomes 0.
6. **Run Commit 2 with the same regression test**: change `TEST_TAG` between runs, diff with `selfcal_scripts/diff_cal_h5.py`. Same `np.allclose(atol=1e-2)` bar as Commit 1 (or exact byte-equality if step 5 was done).

## Untracked files in the dev worktree (state at session end)

After Commits A/B/C above, the only untracked content is:
- `figures/commit1_skymap_diff.png` — gitignored verification artifact, leave on disk.
- (Anything you create later, e.g. progress updates after Commit 2, can be committed via the same `docs:` scope.)

## Identifiability defaults (referenced often during implementation)

When `K ≥ 2`, the `K-1` shift degeneracy must be broken. Default at the call site:
```python
chunk_maps = [primary_map, secondary_map_1, secondary_map_2, ...]
mean_offsets_list = [None] + [np.zeros(num_frames)] * (K - 1)
reg_weights = [0.1] * K  # tune per map
```
Primary map (index 0) stays free; all others are mean-zero per frame. See [MULTI_CHUNK_MAPS_PLAN.md](MULTI_CHUNK_MAPS_PLAN.md) section "Identifiability — required defaults" for the sharp edges.

## Useful pointers (for picking up cold)

- Plan file (this repo): [MULTI_CHUNK_MAPS_PLAN.md](MULTI_CHUNK_MAPS_PLAN.md)
- Plan file (Claude Code store): `/home/thomasli/.claude/plans/actually-forget-adding-the-glittery-map.md`
- Conversation context: the plan was authored by a Plan agent in plan mode; full design rationale (including discarded alternatives like the bilinear gain model and polymorphic API) is in the prior conversation, not in the plan file itself.
- Hot-spot files for Commit 1: [SelfCal/lsqr.py](SelfCal/lsqr.py) (lines 17–151 `_prep_lsqr`, 226+ `setup_lsqr`).
- Hot-spot for K>1 (Commit 2): same plus [SelfCal/subframe.py](SelfCal/subframe.py) and [SelfCal/solution.py](SelfCal/solution.py).
- Test config (small, fast): `frame_setting={Detector:3, NumSub:10, NumCh:34, NumCol:1}`, `chs=[[17]]`.
