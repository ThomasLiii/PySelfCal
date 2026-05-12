# Plan: Polynomial-Order Constraints on Per-Map Chunk Offsets

## Context

Multi-chunk-maps (now merged into `main` via PR #8) lets `Calibrator.setup_lsqr` accept K independent chunk maps and solve for a per-frame offset vector for each. The existing **adjacency regularization** (`adj_infos[m]` + `reg_weights[m]`) penalizes differences between neighboring chunks within map `m`:

```
λ · (o[k, c_i] - o[k, c_j]) = 0      for each adjacent pair (c_i, c_j)
```

This is implicitly a **constant prior** — neighbors are pulled toward equal values. Useful for column-adjacency smoothness in SPHEREx, but it cannot express stronger structural assumptions like "the columns within a single subchannel should vary *linearly* in column index" — which is physically motivated for SPHEREx LVF wavelength-axis offsets and for any imager where a low-order polynomial captures detector trends.

**Goal**: add **polynomial-order constraints** on arbitrary user-supplied *chains* of chunks. The constraint enforces that the offsets along each chain follow a polynomial of a chosen degree N. For N=1 (linear) over a chain of three chunks, the rows added to `A` encode the discrete second-difference `o[c_0] - 2·o[c_1] + o[c_2] = 0`. For arbitrary N, the (N+1)-th finite-difference operator is the stencil. Existing adjacency regularization is the N=0 (constant) case with stencil `[1, -1]` on chains of length 2 — the new mechanism is a strict generalization.

The constraint generation is **generic** (any user can supply chains + stencil + weight), with **SPHEREx-specific helpers** in `SPHERExUtility.py` for the common cases (column-linearity within subchannels; cross-subchannel linearity within a column; etc.).

## Mathematical model

For map `m`, the new constraints take the form, per frame `k`:

```
λ_g · Σ_{ℓ=0..L-1} stencil[ℓ] · o^(m)[g_m(k), chains[r, ℓ]]  =  0
```

for each chain `r` and each constraint group `g`, where:
- `chains` is an `(num_chains, L)` int array of chunk indices for map `m`.
- `stencil` is a length-`L` float array of coefficients.
- `λ_g` is the per-group regularization weight.
- `L = N + 1` for a polynomial-of-degree-N constraint.

A polynomial of degree N in a sequence is annihilated by the (N+1)-th finite difference. Common stencils:

| N | L | stencil | meaning |
|---|---|---|---|
| 0 | 2 | `[1, -1]` | constant (existing `adj_info`) |
| 1 | 3 | `[1, -2, 1]` | linear |
| 2 | 4 | `[1, -3, 3, -1]` | quadratic |
| N | N+1 | `(-1)^k · C(N+1, k)`, k=0..N+1 | degree-N |

For SPHEREx with `NumCol=3` subchannels, the chain for subchannel `s` is `[chunk(s, col=0), chunk(s, col=1), chunk(s, col=2)]`. Linear constraint puts one row per (subchannel, frame). For `NumCol=5`, three sliding windows of length 3 per subchannel.

The constraint is **soft** (added as weighted rows in the LSQR least-squares system), tuned by `λ_g`. Hard polynomial constraints would need a different formulation (Lagrange / KKT); soft is sufficient and consistent with the existing reg.

## API design

### New `setup_lsqr` parameter

```python
poly_constraints_list: Optional[List[Optional[List[dict]]]] = None
```

- **Outer length K** (one entry per chunk map; parallel to `chunk_maps`).
- **Each entry** is `None` (no polynomial constraints on this map) or a `list of constraint groups`.
- **Each constraint group** is a dict:
  ```python
  {
      'chains':  np.ndarray of shape (num_chains, L), int,
      'stencil': np.ndarray of shape (L,), float,
      'weight':  float,
  }
  ```
  Multiple groups per map allow stacking different stencils (e.g., column-linearity *and* cross-subchannel adjacency on the same map).

When `None` or absent, behavior is identical to current code (back-compat with all existing callers).

`adj_infos` and `reg_weights` are **left unchanged**. Both regularization mechanisms coexist; adjacency reg can be expressed as a poly-constraint with `stencil=[1, -1]` for callers that want the unified path, but the legacy form keeps working.

### Driver-side example (SPHEREx column linearity, `NumCol=3`)

```python
from SelfCal.SPHERExUtility import compute_column_polynomial_chains

chains, stencil = compute_column_polynomial_chains(
    det_chunk_map, num_columns=3, degree=1
)
# chains.shape = (num_subchannels, 3)
# stencil = np.array([1.0, -2.0, 1.0])

cc.setup_lsqr(
    chunk_maps=[det_chunk_map],
    adj_infos=[adj_info],            # existing column adjacency (constant prior, [1,-1])
    reg_weights=[0.1],
    poly_constraints_list=[[
        {'chains': chains, 'stencil': stencil, 'weight': 0.5},
    ]],
    ...
)
```

A single map can carry both `adj_info` (smoothness) and a poly constraint (linearity). They contribute additive regularization rows.

## SPHEREx helper

In `SelfCal/SPHERExUtility.py`, add:

```python
def compute_column_polynomial_chains(det_chunk_map, num_columns, degree=1):
    """Build chains and stencil for a polynomial-degree constraint along
    columns within each subchannel of a SPHEREx stripped chunk map.

    For a stripped map with `num_columns` columns per subchannel, each subchannel
    s contributes `max(0, num_columns - degree)` sliding-window chains of length
    `degree + 1` over the column indices `[chunk(s, 0), chunk(s, 1), ...]`.
    The stencil is the (degree+1)-th forward finite-difference operator
    (`(-1)^k * C(degree+1, k)` for `k = 0..degree+1`), which annihilates
    polynomials of degree <= degree.

    Parameters
    ----------
    det_chunk_map : (det_h, det_w) int ndarray
        Stripped chunk map produced by `make_stripped_chunk_map`. Chunk IDs
        are assumed to be `subchannel * num_columns + column`.
    num_columns : int
        Number of column subdivisions per subchannel.
    degree : int
        Polynomial degree to enforce (1 = linear, 2 = quadratic, etc.).

    Returns
    -------
    chains : (num_chains, degree+1) int32 ndarray
    stencil : (degree+1,) float64 ndarray
    """
```

Companion helper (optional, defer until needed):

```python
def compute_subchannel_polynomial_chains(det_chunk_map, num_columns, num_subchannels, degree=1):
    """Same idea, but along the subchannel axis at fixed column index."""
```

Both helpers compute chunk-id sequences from the chunk map's known geometry; they don't read the actual chunk_map content beyond `int(det_chunk_map.max())+1` for sanity checks.

For **non-SPHEREx telescopes**, the user computes their own `chains` and `stencil` from their chunk_map's topology and passes them directly. The core mechanism is generic.

## Critical files to modify

### Core (LSQR build)

- **`SelfCal/lsqr.py`**
  - `_prep_lsqr` (~line 17): emit one row block per (map, constraint group, chain). Inside the existing `if offset_regularization:` block, add a parallel inner loop over `poly_constraint_list[m]` (length 0..G_m). For each constraint group `(chains, stencil, weight)`, append `len(chains)` constraint rows. Track row offset as the existing adjacency reg does. Skip in template mode (`det_template_list[m] is not None`), same as adjacency.
  - `_prep_lsqr_batch_worker` (~line 153): forward the new `poly_constraints_list` task param. **No SHM** — chains/stencils are small (kilobytes for typical SPHEREx configs) and pickling per worker is negligible. If a use case ever pushes >100k chains, revisit.
  - `setup_lsqr` (~line 226): add `poly_constraints_list=None` to signature; default-fill to `[None] * K`; validate length == K and per-entry shapes (each constraint dict has matching `chains.shape[1] == len(stencil)`); pack into common task params.

- **`SelfCal/PipelineWrapper.py`**
  - `Calibrator.setup_lsqr`: add `poly_constraints_list=None` to signature; pass through to `MakeMap.setup_lsqr`. Document the constraint dict format in the docstring with an example.

### SPHEREx helpers

- **`SelfCal/SPHERExUtility.py`**
  - Add `compute_column_polynomial_chains(det_chunk_map, num_columns, degree=1)`.
  - (Optional later) `compute_subchannel_polynomial_chains(...)`.

### Driver

- **`selfcal_scripts/run_cal_v2.py`**
  - Demonstrate the new mechanism. Default (commented-out) the `poly_constraints_list` block so existing runs are unchanged unless the user opts in. Add a short paragraph in the file header explaining when to turn it on (zodi gradient resolution, low-flux LVF channels, etc.).

### Tests / regression

- Reuse `selfcal_scripts/run_cal_baseline_test.py` and `selfcal_scripts/diff_cal_h5.py` as before. With `poly_constraints_list=None` (or unset), output must be byte-equal to current behavior — same regression bar as the multi-chunk-maps refactor (`np.allclose(rtol=0, atol=1e-2)` until the parallel-non-determinism fix is verified to give exact equality).
- New smoke test `selfcal_scripts/run_cal_polyconstraint_smoke_test.py` (analogous to `run_cal_k2_smoke_test.py`):
  - Build a tiny synthetic chunk map (e.g., 4x4 with 2 columns × 2 subchannels = 4 chunks), construct chains + linear stencil, call `setup_lsqr` directly, verify the produced `A` matrix has the expected number of constraint rows in the right column positions with the right values. Doesn't need to run LSQR.

## Verification

**Test A — regression gate** (mirrors multi-chunk-maps Commit 1 test):
With `poly_constraints_list=None`, run `python selfcal_scripts/run_cal_baseline_test.py` with `TEST_TAG='before_poly'` to capture a baseline (or reuse the existing `..._baseline_before_refactor.h5` from the multi-chunk-maps test if config is unchanged). After implementing, re-run with `TEST_TAG='after_poly_off'` and diff. Pass criterion: byte-equal coverage / `reproj_list`; offset/skymap match within `atol=1e-2`.

**Test B — constraint actually does something**:
Run with a *strong* poly constraint on column-linearity (`weight=10.0`, much larger than `reg_weight=0.1`), then extract the per-frame, per-chunk offsets and verify that within each subchannel, the column offsets are visually linear in column index. Concretely: for each subchannel `s` and frame `k`, fit `o[k, chunk(s, c)] vs c` to a line and check the residual norm is a small fraction of the offset magnitude. Compare to a baseline run *without* the constraint where the same metric will be much larger.

**Test C — smoke test (plumbing)**:
Tiny synthetic problem (4 chunks, 1 frame). Pass a single linear-stencil constraint group on chains `[[0, 1, 2]]` with weight `1.0`. Verify the resulting `A.toarray()` has exactly one extra row beyond the data and existing adjacency reg, with column entries `(num_sky+0: 1, num_sky+1: -2, num_sky+2: 1)` and RHS 0.

**Test D — sharp-edge identifiability**:
Use a strict polynomial constraint (very high weight) with `mean_offsets[m] = None` (no anchor). Some configurations will become rank-deficient (e.g., a fully-linear constraint on 3 columns + free DC = 1 DOF removed from 3 = 2 DOF remaining; if that's also pinned to zero-mean, the system can lose identifiability for the rest of the offset structure depending on overlap). Document expected behavior; flag cases where the LSQR residual norm doesn't decrease in `iter_lim` iterations.

## Identifiability — sharp edges to document

- **Polynomial constraint + adjacency reg overlap**: linear-stencil `[1,-2,1]` is exactly the second-difference of the constant-stencil `[1,-1]`. They are *not* redundant (the linear stencil is one degree higher), but a system with both at high weight can become numerically stiff. Recommend `weight_poly >= weight_adj` if you want the polynomial assumption to dominate.
- **Insufficient chain length**: polynomial of degree N over a chain of length L < N+1 is undefined. The helper should raise. For `NumCol < degree+1`, no chains are emitted (no constraint contribution).
- **Hard-pinned mean conflict**: if `mean_offsets[m]` pins the per-frame mean to 0 *and* the polynomial constraint forces linear column variation, the per-frame DC is doubly constrained. Soft constraints make this consistent; just be aware that the effective DOF may be lower than naïve counting suggests.

## Implementation order — 3 commits

The mechanism is small enough that 3 staged commits are sufficient.

**Commit 1 — Core wiring, K=1 + None default, regression-equal.**
Files: `SelfCal/lsqr.py` (`setup_lsqr` signature, `_prep_lsqr` constraint emission, batch worker pass-through), `SelfCal/PipelineWrapper.py` (`Calibrator.setup_lsqr`). With `poly_constraints_list=None`, output byte-identical to current. Run the regression test (Test A) before pushing.

**Commit 2 — SPHEREx helper + smoke test.**
Files: `SelfCal/SPHERExUtility.py` (`compute_column_polynomial_chains`), `selfcal_scripts/run_cal_polyconstraint_smoke_test.py`. The smoke test verifies the new mechanism produces the right constraint rows for a tiny hand-built problem; no full pipeline.

**Commit 3 — Driver opt-in + integration check.**
Files: `selfcal_scripts/run_cal_v2.py` (commented-out template for using the helper), `selfcal_scripts/run_cal_baseline_test.py` (optional `--poly` flag or duplicated config block to also run the constraint version of the regression test). Run a real-config calibration with the constraint enabled; verify Test B (offsets are linear within subchannels) by extracting and fitting.

Each commit is independently runnable. Commit 1 is the regression gate (must produce byte-equal output with the constraint disabled). Commits 2–3 add new functionality without changing existing behavior.

## Branch state going in

- Branch: `feat/poly-offset-constraint` (already created off `main` at `3f451cf`, pushed with upstream tracking).
- `main` is at `3f451cf` (PR #8 merged multi-chunk-maps).
- `stable` is intentionally left at the pre-multi-chunk-maps commit (`868ae1d`) — promote when ready, separate from this work.

## Reused existing utilities

- **`SelfCal/lsqr.py:_prep_lsqr`** adjacency reg block — pattern to follow for the new constraint emission. Same per-map row-offset arithmetic, same `det_template_list[m]`-mode skip, same RHS-zero structure.
- **`SelfCal/SPHERExUtility.py:make_stripped_chunk_map`** — produces `det_chunk_map` and the geometry the helper relies on.
- **`SelfCal/SPHERExUtility.py:compute_column_adjacency`** — the existing helper for the constant-prior case; the new helper mirrors its structure.
- **`selfcal_scripts/run_cal_baseline_test.py` + `selfcal_scripts/diff_cal_h5.py`** — the regression-test scaffolding from multi-chunk-maps. Reuse as-is for Test A.

## Out of scope

- Hard polynomial constraints (Lagrange / KKT formulation). Soft constraints are sufficient.
- Per-frame *vs.* per-group constraint application. Constraints follow the existing `frame_to_group_list[m]`/`col_bases[m]` pattern: rows are emitted per frame, but offsets are shared across frames in the same group, so duplicate rows just upweight the constraint by `num_frames_in_group` — same behavior as adjacency reg today.
- Adaptive / data-driven weight selection. Users tune `weight` manually for now.
- Migration of `adj_infos`/`reg_weights` into `poly_constraints_list`. Possible follow-up; not required.
