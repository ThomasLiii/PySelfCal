# Multi-Chunk-Maps Feature — Progress Log

Companion to [MULTI_CHUNK_MAPS_PLAN.md](MULTI_CHUNK_MAPS_PLAN.md). This file tracks what's done and what's next so the work can be resumed in a fresh session.

The plan also lives at `/home/thomasli/.claude/plans/actually-forget-adding-the-glittery-map.md` (Claude Code's per-project plan store). The copy in this repo is the source of truth going forward — edit this one if requirements change.

## Status as of 2026-05-06

**Branch**: `feat/multi-chunk-maps` (renamed from `feat/multiplicative-corrections` — gain feature was abandoned in favor of additive multi-map). Local + remote.

**Worktrees** unchanged:
- `~/spherex/selfcal/` — dev, branch `feat/multi-chunk-maps`, env `general`
- `~/spherex/selfcal-stable/` — analysis, branch `stable`, env `selfcal-stable`

**Current state**: 0 commits on `feat/multi-chunk-maps` beyond `main`. The branch was pushed empty after rename. All planning + setup work is captured in untracked files in the dev worktree (the test script, this file, the plan copy). Nothing functional has changed yet in `SelfCal/`.

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

### Immediate (before any code changes)

1. **Run the baseline.** With `TEST_TAG = 'before_refactor'` (already set), kick off `python selfcal_scripts/run_cal_baseline_test.py`. Confirm it produces the cal file and note the path. **This is your `cal_baseline_before.h5` snapshot — keep it safe.**

### Commit 1 — minimal `_prep_lsqr` refactor (option B from the conversation)

Scope: refactor only `_prep_lsqr`'s column indexing in [SelfCal/lsqr.py:17-151](SelfCal/lsqr.py#L17-L151) to use `col_bases` (length-K+1 array) and `num_chunks_list` / `num_offset_groups_list` / `frame_to_groups_list` / `det_templates_list` internally. K is forced to 1; everything is wrapped as length-1 lists at the entry of `setup_lsqr` ([SelfCal/lsqr.py:226](SelfCal/lsqr.py#L226)). Public API of `setup_lsqr` unchanged. `_prep_subframe` unchanged. `solution.py` / `coadd.py` / `PipelineWrapper.py` unchanged.

Files touched: `SelfCal/lsqr.py` only.

Key lines to change:
- [SelfCal/lsqr.py:89](SelfCal/lsqr.py#L89): `O_cols = num_sky + (group_idx * num_chunks) + chunk_idx` → loop over maps with `col_bases[m] + (group_idx_list[m] * num_chunks_list[m]) + chunk_idx_list[m]`.
- [SelfCal/lsqr.py:101-108](SelfCal/lsqr.py#L101-L108): adjacency reg rows — also per-map, with their own row-offset arithmetic.
- `setup_lsqr`: at the entry, normalize all single-form params to length-1 lists; compute `col_bases` once; pack into task_params for workers.

After the refactor:
1. Smoke test: `python -c "from SelfCal import PipelineWrapper; print('ok')"`.
2. Run baseline test with `TEST_TAG = 'after_refactor'`.
3. Diff against `before_refactor` cal file (write a small helper script — see template below).

Diff script template (place in `selfcal_scripts/diff_cal_h5.py` when needed):
```python
import sys, h5py, numpy as np
def diff(a_path, b_path):
    with h5py.File(a_path) as A, h5py.File(b_path) as B:
        keys = sorted(set(A.keys()) | set(B.keys()))
        for k in keys:
            if k not in A: print(f'MISSING in A: {k}'); continue
            if k not in B: print(f'MISSING in B: {k}'); continue
            try:
                np.testing.assert_array_equal(A[k][...], B[k][...])
                print(f'OK  {k}  shape={A[k].shape}  dtype={A[k].dtype}')
            except AssertionError as e:
                print(f'DIFF {k}: {e}')
if __name__ == '__main__':
    diff(sys.argv[1], sys.argv[2])
```

### Commit 2 — lift K>1 in core

Scope per [MULTI_CHUNK_MAPS_PLAN.md](MULTI_CHUNK_MAPS_PLAN.md) "Implementation order" → "Commit 2 — Lift K>1 in core". `_prep_subframe` returns `chunk_contribs` list, `_prep_lsqr` truly handles K maps, `setup_lsqr` packs per-map SHM segments, `solution.parse_x` / `compute_x0_from_Ab` switch to list signatures. Public `Calibrator` API still single-map externally.

Smoke test: pass `chunk_maps=[m, m]` (duplicate) with reg + `mean_offsets=[None, np.zeros(num_frames)]` and confirm convergence.

### Commit 3 — user-facing always-list API + new cal schema

Scope: `Calibrator.setup_lsqr` accepts `chunk_maps` (list), `Calibrator.save_calibration` writes new `offsets/` group schema, `Calibrator.load_calibration` reads dual schema. Update `selfcal_scripts/run_cal_v2.py` (wrap as `[chunk_map]`). Update `analysis/analysis_script/zodi_utils.py` to read either old top-level `offset` or new `offsets/map_m`.

### Commit 4 — Mosaicker multi-map application

Scope: `_prep_subframe` mosaic path accumulates over maps, `coadd.compute_coadd_map` accepts `offset_lists`, `Mosaicker.make_mosaic` accepts `chunk_maps` / `det_offset_funcs` lists. Optional helpers: `make_per_partition_mean_zero(fine_map, coarse_map)` for nested-map identifiability.

Re-enable mosaicking in `run_cal_baseline_test.py` for Test C verification.

## Untracked files in the dev worktree (state at session end)

These are working files; commit or remove deliberately.

- `selfcal_scripts/run_cal_baseline_test.py` — the test driver. Keep on the branch (will go in Commit 1 alongside the diff helper).
- `MULTI_CHUNK_MAPS_PLAN.md` — copy of the design plan. Worth committing so reviewers can see the design alongside code.
- `MULTI_CHUNK_MAPS_PROGRESS.md` — this file. Probably keep it untracked or `.gitignore` it; it's session-state, not project documentation.

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
