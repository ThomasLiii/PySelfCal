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
- [x] **Commit 1 — core wiring** ([SelfCal/lsqr.py](SelfCal/lsqr.py), [SelfCal/PipelineWrapper.py](SelfCal/PipelineWrapper.py)). `poly_constraints_list=None` defaults; validation in `setup_lsqr`; per-map loop in `_prep_lsqr` mirrors the adjacency-reg block (same `offset_base_m` indexing, same `reg_row_offset` arithmetic, template-mode skip, RHS=0). No SHM — chains/stencils ride along in `common_params` and pickle directly to workers.
- [x] **Test A passed** for Commit 1: full Ch17 calibration with default (`None`) ran in 2300s. Diff vs `cal_*_baseline_after_commit3.h5`: skymap/offset both clear `np.allclose(rtol=0, atol=1e-2)` with zero elements over threshold (max abs 4.49e-3 / 9.24e-3, mean 2.0e-5 / 2.9e-6). Coverage / chunk_maps / reproj_list byte-equal. Same band as the multi-chunk-maps refactor cleared (parallel-LSQR float32 reduction noise; not affected by this commit).
- [x] **Commit 2 — SPHEREx helper + smoke test** ([SelfCal/SPHERExUtility.py](SelfCal/SPHERExUtility.py), [selfcal_scripts/run_cal_polyconstraint_smoke_test.py](selfcal_scripts/run_cal_polyconstraint_smoke_test.py)). `compute_column_polynomial_chains(chunk_map, num_columns, degree=1)` returns `(num_subchannels*(num_columns-degree-1), degree+2)` int64 chains and the `(degree+2,)` finite-difference stencil; raises if `num_columns < degree+2`. Note math correction vs. the plan's docstring template: chain length is `L = degree + 2` (3 for linear, 4 for quadratic), not `degree + 1`.
- [x] **Smoke test passed**: helper produces the expected `(342, 3)` chains and `[1, -2, 1]` stencil for `(num_columns=3, degree=1)`; `setup_lsqr` with one chain `[[0, 1, 2]]` and weight `2.5` emits exactly one extra row at cols `(num_sky+0, +1, +2)` with values `(2.5, -5.0, 2.5)` and RHS 0. Column count unchanged.
- [x] **Commit 3 — driver opt-in + integration check** ([selfcal_scripts/run_cal_baseline_test.py](selfcal_scripts/run_cal_baseline_test.py)). Added `TEST_VARIANTS` list (default `['poly_k1', 'poly_k2']`), `build_detector_stripe_map(shape, mid_width=64, edge_width=60)` for the K=2 detector-fixed map (32 stripes: 60 + 30×64 + 60 = 2040), and `build_variant_config(variant, ...)` that dispatches the per-variant solver config. **poly_k1**: K=1, NumCol=10, linear column-poly constraint (`weight=0.5`) + adjacency (`weight=0.1`). **poly_k2**: same map 0 + constraint, plus a detector-fixed 64-px stripe map shared across frames (`det_groups_list[1]=zeros`, mean-anchor=0).
- [x] **poly_k1 ran in 2371s**. Test B passes: across the 4.1% of (frame, subchannel) pairs with coverage, median `nonlinearity-RMS / offset-RMS = 0.025` and 90th-percentile `0.05` — column offsets are visually linear, as the constraint demands.
- [x] **poly_k2 ran in 2389s** and resolves a real left/right detector asymmetry into map 1: the 32 shared stripe values flip sign at chunk index 16 (~`+0.02` for chunks 0–15, ~`-0.02` for chunks 16–31). With map 1 absorbing this pattern, map 0's `max |o|` collapsed from `360.8` (k1) to `2.7` (k2) and the sky map's dynamic range tightened from `[-0.30, +0.49]` to `[-0.06, +0.07]`. Active-cols delta = 13,250 = 13219 per-frame scalars + 32 stripes − 1 (rank-deficient by global-DC degeneracy, expected). Cal files: `cal_*_baseline_poly_k1.h5`, `cal_*_baseline_poly_k2.h5`.

## To do — next session pick up here

Commits 1–3 from the plan are complete and pushed. Possible follow-ups:
- Mosaic the `poly_k2` calibration (this branch only ran calibration; no mosaic yet) and inspect whether the residual structure in the sky map shrinks where map 1 isolated detector asymmetry.
- Try a higher-degree poly constraint (`degree=2`) on the column chain and see whether map 0's nonlinearity floor (~2.5e-3 RMS) drops further.
- Add `compute_subchannel_polynomial_chains` (cross-subchannel, fixed column) — sketched in [POLY_OFFSET_PLAN.md](POLY_OFFSET_PLAN.md) but not implemented.
- Promote `stable` if the `poly_k2` skymap looks good (currently held at the pre-multi-chunk-maps commit `868ae1d`).

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
