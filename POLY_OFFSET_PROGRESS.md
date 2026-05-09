# Polynomial-Offset-Constraint Feature — Progress Log

Companion to [POLY_OFFSET_PLAN.md](POLY_OFFSET_PLAN.md). This file tracks what's done and what's next so the work can be resumed in a fresh session.

## Status as of session start

**Branch**: `feat/poly-offset-constraint` (created from `main @ 3f451cf` after the multi-chunk-maps PR #8 was merged). Local + remote, with upstream tracking.

**Worktrees** unchanged from prior work:
- `~/spherex/selfcal/` — dev, branch `feat/poly-offset-constraint`, env `general`
- `~/spherex/selfcal-stable/` — analysis, branch `stable` (intentionally left at the pre-multi-chunk-maps commit until promoted), env `selfcal-stable`

**Current state**: 0 commits on `feat/poly-offset-constraint` beyond `main`. Working tree has:
- `MULTI_CHUNK_MAPS_PLAN.md`, `MULTI_CHUNK_MAPS_PROGRESS.md` — staged for deletion (the multi-chunk-maps work is fully merged; these docs are superseded by [POLY_OFFSET_PLAN.md](POLY_OFFSET_PLAN.md) / this file).
- `POLY_OFFSET_PLAN.md`, `POLY_OFFSET_PROGRESS.md` — new untracked docs for this feature.

Nothing functional in `SelfCal/` has changed yet.

## Done

- [x] Cleaned up branches after multi-chunk-maps merge: deleted `feat/multi-chunk-maps` (local + origin), synced local `main` to `origin/main`, created `feat/poly-offset-constraint`. Stable not advanced (deliberate — keep analysis env on the old code until you're ready to promote).
- [x] Wrote the polynomial-offset-constraint plan ([POLY_OFFSET_PLAN.md](POLY_OFFSET_PLAN.md)) — math, API, file touchpoints, SPHEREx helper, regression strategy, 3-commit implementation order.

## To do — next session pick up here

The full sequence is in [POLY_OFFSET_PLAN.md](POLY_OFFSET_PLAN.md) under "Implementation order — 3 commits". Summary:

### Commit 1 — Core wiring (regression-equal with `poly_constraints_list=None`)

Files:
- `SelfCal/lsqr.py`
  - `setup_lsqr`: add `poly_constraints_list=None` param; default-fill to `[None] * K`; validate length and per-entry shape (`chains.shape[1] == len(stencil)`); thread through to workers via task params.
  - `_prep_lsqr`: inside the existing `if offset_regularization:` block, add a parallel inner loop over `poly_constraint_list[m]` (length 0..G_m). Emit `len(chains)` constraint rows per group, with column entries indexed via `col_bases[m] + group_idx_list[m] * num_chunks_list[m] + chains[r, ℓ]` and data values `weight * stencil[ℓ]`. Skip in template mode.
  - `_prep_lsqr_batch_worker`: forward the new task param. **No SHM** (small data; pickling is fine).
- `SelfCal/PipelineWrapper.py`:`Calibrator.setup_lsqr`: add the param; document the dict format in the docstring.

Verification: run `selfcal_scripts/run_cal_baseline_test.py` with the new code and `poly_constraints_list=None` (default). Diff against the existing baseline `cal_*_baseline_before_refactor.h5` (or generate a new `cal_*_baseline_before_poly.h5` if the prior baseline is gone). Pass criterion: same `np.allclose(rtol=0, atol=1e-2)` band that the multi-chunk-maps refactor cleared.

### Commit 2 — SPHEREx helper + plumbing smoke test

Files:
- `SelfCal/SPHERExUtility.py`: add `compute_column_polynomial_chains(det_chunk_map, num_columns, degree=1)`. Returns `(chains, stencil)` where `chains` is `(num_subchannels * (num_columns - degree), degree + 1)` int32 and `stencil` is the (degree+1)-th finite-difference operator. Raises if `num_columns <= degree`.
- `selfcal_scripts/run_cal_polyconstraint_smoke_test.py`: build a tiny synthetic problem (4 chunks, 1 frame), feed one linear-stencil constraint group, verify the produced `A` matrix has the expected extra rows with the expected column entries and values. No LSQR run.

### Commit 3 — Driver opt-in + integration check

Files:
- `selfcal_scripts/run_cal_v2.py`: add a commented-out block showing how to enable polynomial column-linearity for SPHEREx using the helper.
- `selfcal_scripts/run_cal_baseline_test.py`: support an optional `TEST_TAG='with_poly'` variant that turns on the constraint, so future regression tests can compare with/without.

Then run a real-config calibration with the constraint enabled (`weight=0.5`, `degree=1` over column chains in each subchannel), extract the per-frame offsets, and visualize: for each subchannel, plot column index vs offset for a random sample of frames. With the constraint, lines should be visually linear; without, lines have curvature. (Or compute the second-difference RMS per subchannel as a quantitative metric.)

## How the regression test works

Same scaffolding as multi-chunk-maps:

1. **Baseline**: with `poly_constraints_list=None` (the default), run `python selfcal_scripts/run_cal_baseline_test.py` with `TEST_TAG='before_poly'`. Or if the multi-chunk-maps `before_refactor` baseline cal file is still on disk and config is unchanged, reuse it.
2. **After Commit 1**: re-run with `TEST_TAG='after_poly_off'`. Diff with `selfcal_scripts/diff_cal_h5.py`. Pass criterion: same `np.allclose(atol=1e-2)` band as multi-chunk-maps.
3. **After Commit 3 (constraint on)**: separate run with `TEST_TAG='after_poly_on'`. **Not** byte-equal — the constraint is supposed to change the answer. Verify visually + with the second-difference metric (Test B in the plan).

## Resuming in a fresh session — concrete steps

1. **Open Claude Code in the dev worktree**:
   ```
   cd ~/spherex/selfcal && claude
   ```
   (Or open the folder in VS Code with the Claude Code extension.)

2. **Verify state**:
   ```
   git branch --show-current   # → feat/poly-offset-constraint
   git log --oneline -5        # → top of main (3f451cf and below); 0 commits ahead yet
   git status --short          # → MULTI_CHUNK_MAPS_*.md staged for deletion; POLY_OFFSET_*.md untracked
   ```

3. **Tell the new session**:
   > Read `POLY_OFFSET_PLAN.md` and `POLY_OFFSET_PROGRESS.md`. We're picking up at Commit 1 — wire `poly_constraints_list` through `setup_lsqr` / `_prep_lsqr` / `Calibrator.setup_lsqr`. With the param defaulting to `None`, behavior is unchanged. Run the regression test in the plan before pushing.

4. **Verify cached state still intact** before any compute:
   ```
   ls /mnt/md124/thomasli/selfcal/outputs/SPHEREx_nep_qr2_det3_6p2arcsec/calibration/cal_*_baseline_*.h5
   ls /home/thomasli/spherex/selfcal/cache/reproj_nvme_SPHEREx_nep_qr2_det3_6p2arcsec | head -3
   ```
   The NVMe reproj cache from the multi-chunk-maps test (~250 GB) and the `cal_*_baseline_before_refactor.h5` from before the merge should still be present unless you deliberately cleaned them up.

5. **Commit cadence**: stage the doc deletions + the new plan/progress docs as the *first* commit on this branch (`docs: add poly-offset-constraint plan; remove superseded multi-chunk-maps docs`). Then the three implementation commits from the plan follow.

## Useful pointers

- Plan file: [POLY_OFFSET_PLAN.md](POLY_OFFSET_PLAN.md)
- Hot-spot files for Commit 1: [SelfCal/lsqr.py](SelfCal/lsqr.py) (the `_prep_lsqr` adjacency block is the pattern to mirror; `setup_lsqr` accepts the new param).
- Hot-spot for Commit 2: [SelfCal/SPHERExUtility.py](SelfCal/SPHERExUtility.py) (mirror the structure of `compute_column_adjacency`).
- Test config for the regression run: `frame_setting={Detector:3, NumSub:10, NumCh:34, NumCol:3}`, `chs=[[17]]` — same as the prior multi-chunk-maps regression run.
- Multi-chunk-maps reference (already merged into `main`): the `adj_info` packing in `setup_lsqr`, the per-map row-offset arithmetic in `_prep_lsqr`, and the `col_bases` / `frame_to_group_list` indexing are all directly reused.

## Untracked / staged-for-deletion files

- `MULTI_CHUNK_MAPS_PLAN.md`, `MULTI_CHUNK_MAPS_PROGRESS.md` — staged for deletion via `git rm`. Will be removed in the first commit on this branch.
- `POLY_OFFSET_PLAN.md`, `POLY_OFFSET_PROGRESS.md` — new docs for this feature.
- `figures/commit1_skymap_diff.png` — gitignored verification artifact from the multi-chunk-maps work; leave on disk (or `rm` if you want a clean figures/ dir).
