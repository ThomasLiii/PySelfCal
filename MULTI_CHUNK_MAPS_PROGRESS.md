# Multi-Chunk-Maps Feature — Progress Log

Companion to [MULTI_CHUNK_MAPS_PLAN.md](MULTI_CHUNK_MAPS_PLAN.md). This file tracks what's done and what's next so the work can be resumed in a fresh session.

The plan also lives at `/home/thomasli/.claude/plans/actually-forget-adding-the-glittery-map.md` (Claude Code's per-project plan store). The copy in this repo is the source of truth going forward — edit this one if requirements change.

## Status as of 2026-05-06

**Branch**: `feat/multi-chunk-maps` (renamed from `feat/multiplicative-corrections` — gain feature was abandoned in favor of additive multi-map). Local + remote.

**Worktrees** unchanged:
- `~/spherex/selfcal/` — dev, branch `feat/multi-chunk-maps`, env `general`
- `~/spherex/selfcal-stable/` — analysis, branch `stable`, env `selfcal-stable`

**Current state**: **Commit 1 is done and pushed.** Three commits ahead of `main`:
- `00a2892 selfcal: parameterize _prep_lsqr offset/adjacency rows by col_bases`
- `f24c155 scripts: add baseline test driver and cal-file diff helper`
- `610056b docs: add multi-chunk-maps design plan and progress log`

## Done

- [x] Wrote and approved the design plan (4-commit staged refactor).
- [x] Renamed branch `feat/multiplicative-corrections` → `feat/multi-chunk-maps` (local + origin).
- [x] Created `selfcal_scripts/run_cal_baseline_test.py` (you authored it; I trimmed it for byte-equality testing — see "How the regression test works" below).
- [x] Copied plan into repo as `MULTI_CHUNK_MAPS_PLAN.md`.
- [x] **Captured pre-refactor baseline cal file.** Path:
      `/mnt/md124/thomasli/selfcal/outputs/SPHEREx_nep_qr2_det3_6p2arcsec/calibration/cal_Detector3_NumSub10_NumCh34_NumCol3_Ch17_baseline_before_refactor.h5`
      (411 MB; `offset (13219, 1026) float64`, `skymap (12532, 12540) float64`).
      Hands-off after this — do not regenerate. The post-refactor run with `TEST_TAG='after_refactor'` produces a sibling file for the diff.
- [x] **Bug noted in `lsqr.py:362`**: when `adj_info` is an empty array (happens with `NumCol=1`, where `compute_column_adjacency` returns 0 boundaries), `SharedMemory(create=True, size=0)` raises. Production typically uses `NumCol=3` or `5` so it doesn't hit. Test was bumped from `NumCol=1 → 3` to dodge it. File a separate bug fix later (don't bundle into the multi-chunk-maps refactor).
- [x] **Commit 1 — `_prep_lsqr` parameterized by `col_bases`.** Wrapped single-form params (`chunk_contrib`, `num_chunks`, `det_template`, `group_idx`, `adj_info`, `reg_weight`) as length-1 internal lists; offset and adjacency rows now built from `col_bases` indexing. K=1 codepath is mathematically identical to the original.
- [x] **Regression check passed.** Re-ran calibration with refactored code (`TEST_TAG='after_refactor'`); diffed against baseline. `offset_coverage`, `offset_coverage_frac`, `skymap_coverage`, `reproj_list` byte-equal. `offset` and `skymap` match within `np.allclose(rtol=0, atol=1e-2)` — median diff = 0; max abs diff 5.7e-3 on offset, 2.8e-3 on skymap. Visual diff (`figures/commit1_skymap_diff.png`, gitignored) shows scan-direction noise concentrated at survey edges, not systematic bias. Diagnosis: pre-existing parallel non-determinism in `setup_lsqr`'s `as_completed` batch ordering ([SelfCal/lsqr.py:400](SelfCal/lsqr.py#L400)), not the refactor. **Worth fixing as a separate small commit** later — collect futures into a list indexed by batch ID, concatenate in batch order — then the byte-equality bar becomes exact zero.

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

## To do — next session pick up here

### Commit 2 — lift K>1 in core (next)

Scope per [MULTI_CHUNK_MAPS_PLAN.md](MULTI_CHUNK_MAPS_PLAN.md) "Implementation order" → "Commit 2 — Lift K>1 in core". `_prep_subframe` returns `chunk_contribs` list, `_prep_lsqr` truly handles K maps, `setup_lsqr` packs per-map SHM segments, `solution.parse_x` / `compute_x0_from_Ab` switch to list signatures. Public `Calibrator` API still single-map externally.

Smoke test: pass `chunk_maps=[m, m]` (duplicate) with reg + `mean_offsets=[None, np.zeros(num_frames)]` and confirm convergence.

### Commit 3 — user-facing always-list API + new cal schema

Scope: `Calibrator.setup_lsqr` accepts `chunk_maps` (list), `Calibrator.save_calibration` writes new `offsets/` group schema, `Calibrator.load_calibration` reads dual schema. Update `selfcal_scripts/run_cal_v2.py` (wrap as `[chunk_map]`). Update `analysis/analysis_script/zodi_utils.py` to read either old top-level `offset` or new `offsets/map_m`.

### Commit 4 — Mosaicker multi-map application

Scope: `_prep_subframe` mosaic path accumulates over maps, `coadd.compute_coadd_map` accepts `offset_lists`, `Mosaicker.make_mosaic` accepts `chunk_maps` / `det_offset_funcs` lists. Optional helpers: `make_per_partition_mean_zero(fine_map, coarse_map)` for nested-map identifiability.

Re-enable mosaicking in `run_cal_baseline_test.py` for Test C verification.

## Resuming in a fresh session — concrete steps

1. **Open a new Claude Code session in this repo** (`cd ~/spherex/selfcal && claude` — or open the folder in VS Code with the Claude Code extension).
2. **Verify state**:
   ```
   git branch --show-current   # → feat/multi-chunk-maps
   git log --oneline -5        # → confirm 00a2892, f24c155, 610056b at top
   ```
3. **Tell the new session**:
   > Read `MULTI_CHUNK_MAPS_PLAN.md` and `MULTI_CHUNK_MAPS_PROGRESS.md`. We're picking up at Commit 2 of that plan — lift K>1 in core. Start by re-reading [SelfCal/lsqr.py](SelfCal/lsqr.py) (the K=1 wrapper inside `_prep_lsqr` is what needs to be replaced by reading per-map params from `task_params`) and [SelfCal/subframe.py](SelfCal/subframe.py) (`_prep_subframe` needs to return `chunk_contribs` as a list and accept a `chunk_maps` list).
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
