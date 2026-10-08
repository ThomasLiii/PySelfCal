# SelfCal pipeline runbook

Operational + on-disk-schema reference for the SelfCal calibration /
mosaicking pipeline. Companion to [selfcal/README.md](selfcal/README.md),
which documents the *code architecture* (module layout, shared-memory
hand-off, parallel SpMV, `_prep_subframe`, etc.). Read that first when
modifying anything inside `selfcal/`. Read this file when running the
pipeline, tuning hyperparameters, or working with its outputs.

## Running

A run is a Python run script ([The Python API](docs/guide/python-api.md)); the production ones
are in [`selfcal_scripts/runs/`](selfcal_scripts/runs/), built from the shared fields, recipes and
tilings of [`selfcal_scripts/recipes/`](selfcal_scripts/recipes/):

```bash
./selfcal_scripts/run.sh selfcal_scripts/runs/<run>.py             # or launch/<run>.sh
./selfcal_scripts/run.sh selfcal_scripts/runs/<run>.py --dry-run   # the plan only (selfcal plan)
```

The knobs below are named by the Python settings that carry them (`sc.SPHEREx`, `sc.Offsets`,
`sc.Sky`, `sc.Fit`, `sc.Coadd`, `sc.Numerics`, `sc.Compute`, `sc.Tuning`, `sc.Tiles`, `sc.Passes`)
and, where it helps, by the keyword of the library call they reach (`Calibrator.setup_lsqr`,
`apply_lsqr`, `Mosaicker.make_mosaic`). TOML run configs are no longer run;
[Migrating from TOML](docs/guide/migrating-from-toml.md) maps their keys to these settings.

## Calibration model

The selfcal model is per observation (one value of frame `k` on reference pixel `p_i`):

```
observed[i] = Σ_j sky_j(p_i)·c_j(v_i) + Σ_m Σ_n offset^(m)[g_m(k), c_m(i), n]·φ_mn(v_i) + scalar[k] + noise
```

where:
- `v_i` are the observation's **data variables** (detector maps such as SPHEREx's band centre,
  per-frame values such as the time, reference-grid maps, stored layers, the built-in
  coordinates, functions of those); `c_j` and `φ_mn` are any known functions of them
  (`selfcal/models/variables.py`; `sc.Model(variables=...)`).
- `j` indexes **sky terms** (one map each; `c = 1` for a constant sky).
- `m = 0..K-1` indexes **chunk maps**. Each map contributes one additive offset block (K=1 is the legacy single-map case); `φ = 1` (one function) is the classic chunk offset.
- `g_m(k)` is the frame→group mapping for map `m` (defaults to identity; `sc.Offsets(per=...)` locks several frames to one offset vector: `per="all"`, or any per-frame value; the library's `det_groups_list[m]`).
- `c_m(i)` is the chunk ID of pixel `i` under map `m`.
- `scalar[k]` is an optional **per-frame DC scalar** (`sc.Model(scalar=True)`, the default, and in `sc.continuum` / `sc.spectral`; the library's `use_per_frame_scalar=True`). It absorbs per-frame brightness shifts so the chunk offsets only carry within-frame structure.

For the K=1 default case the model collapses to `sky + offset[frame, chunk] + scalar[frame]`. Zodi removal quality is dominated by the offset model's spatial resolution. User priors (any linear rows on the unknowns) and an observation weight (a function of data variables) complete the model; see [Bring your own telescope](docs/bring_your_own_telescope.md).

## Calibration pipeline tuning

Chunk geometry (`sc.SPHEREx(detector, num_sub=, num_ch=, num_col=)`):

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

The production recipe of the NumCol 10 era, `POLY_K1` of
[`selfcal_scripts/recipes/spherex.py`](selfcal_scripts/recipes/spherex.py) (the channel maps
since 2026-09-29 use `NUMCOL3`: NumCol 3, no column polynomial), with the production machine of
`selfcal_scripts/recipes/site.py`:

```python
import selfcal as sc

POLY_K1 = sc.Recipe(
    sc.continuum(smooth=0.1,                     # adjacency smoothness between neighbouring chunks
                 poly_prior=sc.Poly(1, weight=0.5)),   # the linear column polynomial; omit for none
    fit=sc.Fit(50, clip=5.0),                    # shot_noise_weights=True in d4_aromatic / the PAH fits
    coadd=sc.Coadd(clip=2.0, oversample=2, ignore_flags=[21]),   # drop the source-mask bit
    name="damp0p1_reg0p1_outThresh5_sigma2_polyK1")
ORCA = sc.Compute(scratch, workers=48, coadd_workers=48)        # workers tuned 2026-05 from 32
# sc.Numerics() is production's: 48 solver threads, batches of 50 frames (tuned 2026-05 from 20)
```

The sky damping is each sky term's `sc.Sky(damping=)`: 0.1 for the first term and 0.3 for the
others unless given. The spectral recipes ignore bit 21 in the fit too
(`sc.Fit(ignore_flags=[21])`).

Programmatically, the offset structure is an `OffsetModel` of one `OffsetBlock` per chunk map
(the run engine builds it from the model's offset terms; `Calibrator.setup_lsqr` still accepts the older flat per-map
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

Key knobs (per map `m`; the block field, then the model's setting, in parentheses):

- **`reg_weights[m]`** + **`adj_infos[m]`** (`reg_weight`, `adj_info`; `sc.Offsets(smooth=, smooth_along=)`) adds `reg_weights[m] * (O_i - O_j) = 0` rows to LSQR for adjacent chunk pairs on map `m`. Two builders in `selfcal/instruments/spherex/spherex_utility.py` (the run engine builds the same pairs from a term's `smooth_along` axes with `selfcal.models.offset_structure.adjacency_along`):
  - `compute_column_adjacency(det_chunk_map, num_columns)` — pairs chunks at same subchannel, adjacent columns. **The default.** Returns `(empty, empty)` for `NumCol=1`; `setup_lsqr` demotes empty adj_info to `None` automatically.
  - `compute_subchannel_adjacency(...)` — pairs at same column, adjacent subchannels.

- **`poly_constraints_list[m]`** (`poly_constraints`; `sc.Offsets(poly_prior=sc.Poly(...))`; optional) — list of constraint dicts that enforce polynomial offset behavior along supplied chunk chains. Each dict is `{'chains': (n_chains, L) int array, 'stencil': (L,) float array, 'weight': float}` and adds `weight * Σ_ℓ stencil[ℓ] · O[chains[r, ℓ]] = 0` rows per frame, per chain. For SPHEREx column linearity: `compute_column_polynomial_chains(det_chunk_map, num_columns, degree=1)` returns `(chains, stencil)` with stencil `[1, -2, 1]` and chain length `degree+2`. See [selfcal/instruments/spherex/spherex_utility.py](selfcal/instruments/spherex/spherex_utility.py).

- **`mean_offsets_list[m]`** (`mean_offset`; `sc.Offsets(mean_zero=True)`) — per-frame mean-offset soft constraint with weight 10.0 (hardcoded in `selfcal/core/system.py`). When using `use_per_frame_scalar=True`, anchor every map to mean-zero so all per-frame DC ends up in the scalar column.

- **`use_per_frame_scalar=True`** (`sc.Model(scalar=True)`) — adds an explicit `num_frames` block to `x` (one scalar per frame) decoupled from `det_groups_list`. Combined with mean-zero anchors on all maps, this pushes per-frame DC entirely into the scalar so chunk offsets only carry within-frame structure. **Required for narrow channels** (D3 Ch17 etc.) where sparse chunk coverage was previously letting per-frame DC leak into scan-stripe residuals.

- **`weighted_damping=True`** + **`damp_weight`** (`sc.Sky(damping=)`, per term) damps **sky pixels** (not offsets) toward zero, weighted by `sqrt(damp_weight * coverage)`. Offsets are damped per term with `sc.Offsets(damping=)`.

`lsqr_kwargs`:

- Use **`compute_x0_scalar_only(A, b, ref_shape, scalar_col_start=cc.col_bases[len(cc.chunk_maps)], num_sky_blocks=cc.num_sky_blocks, active_mask=cc.active_mask)`** for the warm start when `use_per_frame_scalar=True`. It seeds *only* the scalar block from the diagonal-LS estimate (≈ weighted mean of valid `b` per frame), leaving chunks and sky at 0. Critical to avoid scan-stripe regressions on narrow channels. `active_mask` is required because `setup_lsqr` compacts the zero columns by default; `x0` comes back in the full column layout `apply_lsqr` expects.
- For runs without the per-frame scalar, use the older `compute_x0_from_Ab(A, b, ref_shape, active_mask=cc.active_mask)` — diagonal-LS over the full offset region.
- `iter_lim=50` (`sc.Fit(50)`) is typical with the warm start. Watch the `show=True` residual prints (`arnorm` should drop to ~1 or below) to confirm convergence.
- **Stopping, and a fixed iteration count.** `sc.Fit(iterations=N, tolerance=t)` reaches the
  solver as `iter_lim = N`, `atol = btol = t` (`tolerance=(atol, btol)` sets them apart), and
  LSQR / LSMR stop at the first of their Paige-Saunders tests: `istop = 1` (|r| / |b| <= btol +
  atol |A| |x| / |b|, a compatible system), `2` (|A^T r| / (|A| |r|) <= atol, a least-squares
  solution), `3` (the estimate of cond(A) exceeds `conlim`, 1e8), `4`-`6` (the same three at
  machine precision) or `7` (the iteration limit). |A| and cond(A) are running estimates that
  grow with the iterations, so the `istop = 2` ratio falls even when the weakest directions are
  still moving (the transfer-function runs of 2026-10: the largest scales of D3 Ch9 kept
  converging for 700 iterations). **`sc.Fit(iterations=N, tolerance=0)` runs exactly N
  iterations**: `atol = btol = 0` and the condition stop off (`conlim = 0`, passed to
  `apply_lsqr`; any other tolerance keeps the solvers' own `conlim = 1e8`, byte for byte). Only
  machine precision (`istop` 4-6) or an exact solution can end such a solve earlier, and the
  record says so.
- **The record of every solve.** `apply_lsqr` records how each solve ran
  (`selfcal.core.solve_record.SolveRecord`; `Calibrator.solve_record`): the method, the
  iterations run (and `iterations_total`, the same until a solve continues another's), `istop`
  and its meaning, the solver's final estimates (`r1norm`, `r2norm`, `arnorm`, `anorm`, `acond`,
  `xnorm`), the tolerances and `conlim`, the wall time (in the action's record only, so the cal
  file stays byte-reproducible), and the **true residual** `|b - A x|`,
  computed once at the end with one product and accumulated in float64 (the estimate `r1norm`
  drifting from it is how the float32-norm failure of 2026-10 showed up). LSQR overwrites `b`,
  so a copy is kept for that product: in memory below `sc.Tuning(spill_min_gb=)` (4 GB), else
  on scratch disk (`spill_dir`) for the solve. The record goes to the cal file's `solve` group
  ([its attributes](#the-solve-group)), to the action's record (`solves`) and, with the solver's
  state at every iteration, to `<field>/records/<cal stem>_history.npz` (`itn`, `r1norm`,
  `r2norm`, `arnorm`, `anorm`, `acond`, `xnorm`, `test1` = |r| / |b|, `test2` = |A^T r| / (|A|
  |r|), `elapsed_s`; row 0 is the starting vector, `itn = 0`), collected from the scalars the
  solver computes anyway: no extra products per iteration. LSMR (`selfcal.core.lsmr`, scipy's
  with this hook, bit-identical) reports one residual norm, as both `r1norm` and `r2norm`.
  Plot convergence with
  `h = np.load(CalFile(cal).solve["history_file"]); plt.semilogy(h["itn"], h["r1norm"])`.
- **Continuing a solve (a warm start from a cal).** `field.calibrate(recipe, start=<cal>)`
  (a cal file, the `Result` of an earlier calibration, or `{job: cal}`) starts each job's
  solve from the solution in that cal instead of the default guess above, to continue a solve
  that has not converged (the D3 Ch9 transfer-function run went 122 → 422 → 722 iterations in
  three runs) or to start a variant from an earlier solution. `x0` is read back as the exact
  inverse of how the cal is written (`selfcal.core.warm_start`): each sky term's `sky/<name>`
  in the model's order (pixels the source did not solve start at 0), each offset term's
  `offsets/map_<m>` frame-major (a `times=` / `basis=` term: one coefficient per chunk and
  function; a term shared by groups of frames: its group's row, checked to be the same on every
  frame of the group), then `frame_scalar`; in physical units and the full column layout, which
  `apply_lsqr` compacts and scales as it does the default `x0` (no extra copy of `A`, one `x0`).
  Columns active now that the source left at zero start at 0; source values of columns with no
  data now are dropped; the log gives both counts. **The source must be a solution of the same
  system**, else the action is refused with what differs: the same frames in the same order, the
  same model (sky terms and their coefficients, offset terms on the same chunk maps, shared over
  the same groups of frames, with the same basis functions), the same reference grid (its shape
  and WCS) and the same job. Every cal records the identity of its system in its `solve` group
  (`system`, `system_identity`); a cal solved before 2026-10-08 has none and is checked by its
  contents only (frames, sky terms, chunk maps, column counts, grid shape), with a warning. The
  plan checks the frames, sky terms, grid shape, chunk maps and job before the solve is set up,
  the solve the rest. Not supported: an offset term with a hard polynomial basis
  (`Offsets(polynomial=...)`; its cal holds the offsets the polynomial expands to, not the
  coefficients), and tiled or N-pass runs (refused). The start's identity (its product
  fingerprint, or the hash of its bytes when it has no current sidecar) enters the new cal's
  fingerprint (the key is absent without a start, so every other cal keeps its fingerprint);
  the new cal records `start_from` (its path), `start_identity` and `iterations_total` (the
  source's total plus this solve's iterations; -1 when the source has no record of its
  iterations), so do the action's record (`settings.start`, `solves`) and a rerun of that record
  replays the start. Give the continuation a recipe name of its own
  (`recipe.replace(fit=sc.Fit(300, tolerance=0), name="it422")`): the cal the action writes is
  never its own start. **A continuation restarts the Krylov space**: LSQR / LSMR solve
  `A dx = b - A x0` from `dx = 0`, so a solve of N iterations continued for M more is not one
  solve of N + M (the search directions start again from the residual of `x0`), and its
  estimates (`anorm`, `acond`; LSQR's `xnorm` = ‖dx‖, LSMR's ‖x‖) are those of the restarted
  solve.
- **Snapshots every k iterations.** `field.calibrate(recipe, snapshots=sc.Snapshots(every=k,
  keep=None))` (or `snapshots=k`) writes each job's solution after iterations k, 2k, ... as
  `<cal dir>/snapshots/<cal stem>_it<NNNN>.h5`, to watch the solution evolve (the TF campaign
  judged the convergence of the largest scales from such snapshots) and to keep a usable state if
  a long run dies. `NNNN` is the **cumulative** iteration (at least four digits): after a warm
  start, the start's `iterations_total` plus this solve's iteration (a start that records no
  count: this solve's iteration, and `iteration = -1`). The iteration the solve stops at gets no
  snapshot (the cal is written then), nor does any iteration of a solve whose cal is current and
  reused (`overwrite=True` solves again). Each snapshot is a **complete cal file** in the final
  cal's schema ([below](#cal_h5-schema-multi-chunk-map)): the sky terms with their coverage,
  Fisher information and separability, the offsets with their coverage, `frame_scalar`, the
  chunk maps, `reproj_list` (the frames' permanent paths), the root attributes, and a `solve`
  group marked `snapshot = True` ([its attributes](#the-solve-group)). Its sky, offset and scalar
  datasets are **bit for bit those of a solve of exactly that many iterations**
  (`sc.Fit(iterations=it, tolerance=0)`): the solvers' iterates do not depend on the iteration
  limit, and the iterate is converted exactly as the end of the solve converts it. So every cal
  reader works on one: `field.mosaic(recipe.replace(name="it0300"), cal=<snapshot>)` mosaics it
  (give the recipe its own name: the mosaic is named after the recipe),
  `field.calibrate(recipe, start=<snapshot>)` continues from it (the snapshot records the system
  identity the warm start checks), `CalFile(<snapshot>)` and the analysis tools read it.
  `keep=m` keeps the last `m` snapshots of the solve (each older one this solve wrote is deleted
  as a new one is written; snapshots of earlier runs are never deleted). **Snapshots are not
  products**: no sidecar, never part of a product's inputs, so a run with snapshots makes the same
  cal, byte for byte, with the same fingerprint, as one without; the setting is an action's,
  recorded in its record (`settings.snapshots`, and per solve `solves[].snapshots`: `every`,
  `keep`, `directory`, the snapshots kept, how many retention deleted, any that could not be
  written) and replayed by a rerun. A snapshot that cannot be written (a full disk) is logged as an
  error and skipped; the solve goes on. Plain calibrations only (`tiles=` and `passes=` are
  refused) and the CSR system `setup_lsqr` builds (always the case in a run). **How**: the solvers
  (`lsqr_inplace`, the vendored `lsmr`) call `callback(itn, x)` every k iterations (nothing when
  unset); `apply_lsqr` wraps the compact, column-scaled `x` in a `selfcal.core.snapshots.Iterate`,
  which reads any block of the physical full layout (`x * M` on the active columns, 0 elsewhere)
  without a second copy of `x`. The parts of the cal that do not depend on `x` are written once,
  before the solve, to a template beside the snapshots (`.<cal stem>_static-<pid>.h5`, removed
  when the solve ends; the pixel state is read memory-mapped where the setup parked it); each
  snapshot is a copy of the template plus the sky maps (written a band of chunk rows at a time,
  the same stored chunks as the cal's), the offsets and `frame_scalar`, written atomically from the
  solver's thread (no process is started). **Size**: a snapshot is about the size of the cal: one
  D3 Ch9 sky map on its 12544 x 12538 grid is ~630 MB as float32 before compression, so budget
  the disk for `keep` (or every) snapshots; the write holds ~10 MB of a sky map at a time
  (196-row bands) plus one offset term.
- **Monitors and stop rules.** The weakest directions of a self-calibration system are usually
  its largest scales: a smooth sky pattern the offsets can nearly absorb (a near-null mode, e.g.
  a field-wide gradient) barely changes `|b - A x|` or `|A^T r|` while it moves, so **residual
  and gradient tests can pass while those scales are still far from their least-squares
  values**. Monitors show it; stop rules are opt-in and say why a solve stopped
  (`selfcal.core.monitor`).
  - **Monitors** — `field.calibrate(recipe, monitor=sc.Monitor(every=m, residual=True,
    gradient=False, large_scale=True, step=None))` (or `monitor=m`) checks each solve at
    iterations 0, m, 2m, ...: `residual`, the **true residual** `|b - A x|` (one product `A x`
    against the copy of `b` the solve keeps for its final residual, accumulated in float64) and the
    ratio of the solver's estimate `r1norm` to it (a drift of the recurrences shows as a ratio
    away from 1); `gradient`, the **true gradient** `|A^T r|` (`|A^T r - damp^2 x|` with damping;
    one product `A^T r` more), with the ratio of `arnorm`, both in the solver's column-scaled
    unknowns; `large_scale`, the **large-scale fit** of each sky term: the least-squares fit of
    the monomials of degree ≤ d (`True`: 2, i.e. 1, x, y, x², xy, y²; or a degree), in the
    reference grid's coordinates scaled to [-1, 1], to the term's covered pixels (its active
    columns) on every `step`-th row and column (default 4), read through
    `selfcal.core.snapshots.Iterate` a row at a time. Recorded: the coefficients (physical units;
    `x` is per half-field), the rms of the fitted surface without its mean, and its relative
    change since the previous check, rms(s_k − s_prev) / rms(s_k) over the sampled pixels, the
    means removed (basis-independent; the mean is left out because a sky term's mean trades with
    the offsets' gauge and would hide the shape). Polynomials rather than a DCT: the covered
    region is a footprint, not the grid's rectangle, so a DCT's modes are neither orthogonal on it
    nor independent of where it sits in the grid, while a polynomial fit on the covered pixels is
    meaningful on any footprint. The checks go to the history file (below), one line each to the
    log, and the last to the action's record (`solves[].monitor`). Monitors **do not change the
    solve**: the cal is the same, byte for byte, with or without them; they are an action's
    setting (recorded, replayed by a rerun, never fingerprinted). Any calibration takes them,
    tiled (one history per tile) and N-pass (the joint solve) included.
  - **Stop rules** — `sc.Fit(iterations=N, stop=sc.Stop(residual=..., gradient=...,
    large_scale=..., lsqr_tests=False, min_iterations=0, combine="all"))`, off by default (the
    field is declared with `added`, so every fingerprint stays while it is unset; set, it enters the
    cal's inputs). The rules, each a number (its threshold) or an object with its parameters:
    `residual=sc.ResidualRule(below, window=10, every=None)`: the relative decrease of `|r|` over
    the last `window` iterations below `below` (of the estimate `r1norm`, every iteration; with
    `every=m`, of the true residual at a check every m iterations, `window` a multiple of m);
    `gradient=sc.GradientRule(below, window=10, every=None)`: the largest `|A^T r|` over the last
    `window` iterations at most `below` times its value at the start of the solve (the window:
    CG-type gradients are not monotone; estimate or true values as above);
    `large_scale=sc.LargeScaleRule(below, every=10, degree=2, step=4)`: the relative change of every
    sky term's large-scale fit below `below` at the last two checks (checks every `every` iterations
    from 0, so the first possible stop is at `2 * every`); `lsqr_tests`: the solver's own tolerance
    tests that count as a rule (`True`, or some of `"compatible"` (istop 1), `"least_squares"` (2),
    `"condition"` (3)). **With a Stop, the solver's tests stop the solve only as one of its rules**
    (`lsqr_tests=False`, the default, turns them off; `Fit(tolerance=0)` turns them off anyway and
    refuses `lsqr_tests`). `combine="all"` (the default) stops when every rule given holds at the
    same iteration, `"any"` when one does; no rule stops the solve before `min_iterations`. A rule
    checked every m iterations keeps its verdict until its next check. The iteration limit stays
    the hard cap, and machine precision (`istop` 4-6, which also catches an exact solution) still
    ends a solve. A solve a rule ended stops with **`istop = 8`**. The stop decision depends on
    the Stop alone (a monitor never feeds a rule): a rule that needs a check runs it at its own
    cadence, with or without a monitor; the checks of both run at the union of their cadences, and
    the large-scale fit is the rule's (a monitor asking another degree or step is refused). A
    large-scale rule is not proof of convergence either: an iterative solver can sit on a plateau
    for a while before a slow mode moves, so give it a strict `below`, combine it with the others,
    and check the history. A warm start (`start=`) starts the rules afresh (the gradient's
    starting value is that of the continuation's start).
  - **Cost** — nothing per iteration (the estimate-based rules read scalars the solver computes
    anyway); a true residual costs one `A x` and one read of `b` (from scratch disk when it is
    parked there), a true gradient one `A^T r` more, each with the transient vector of a solver
    iteration (no copy of `A`, no second `x`); a large-scale check reads `n_pix / step²` pixels
    per sky term, a row at a time. Everything runs in the solver's thread on the operator's thread
    pools (no process is started); the files are written by the main process after the solve.
  - **Records** — the history file gains, for a monitored solve or one with stop rules:
    `check_itn` (this solve's iterations of the checks), `check_iteration` (counted across warm
    starts, like snapshots; -1 when the start's count is unknown), `true_residual` and
    `residual_ratio` (= `r1norm` / truth), `true_gradient` and `gradient_ratio` (= `arnorm` /
    truth), and the large-scale fit, `large_scale_terms`, `large_scale_basis` (`1`, `x`, `y`,
    `x^2`, ...), `large_scale_degree`, `large_scale_step`, `large_scale_samples` (pixels per term),
    `large_scale_coefficients` (checks × terms × monomials), `large_scale_rms`,
    `large_scale_change` (relative to the previous check with a fit) — each only when computed
    at some check, NaN at the others; and the rules' values per iteration, `stop_residual` (the
    relative decrease), `stop_gradient` (the windowed maximum over the start), `stop_large_scale`
    (the larger of the last two changes), `stop_lsqr_tests` (1 when a kept test holds), NaN where
    not evaluated, and `stop_holds` (the combination). A solve with stop rules records in the cal's
    `solve` group ([below](#the-solve-group)) and the action's record `stop_rule`, `stop_iteration`,
    `stop_values` (every rule's state there) and `stop_policy`, and says in the log, in words,
    which rule ended the solve and what it measured (or that none did, with each rule's state at
    the end). A solve with neither keeps exactly the history and the `solve` group it had.
- `precondition=True` (column-norm; `sc.Fit(precondition=True)`, the default) is essential — much faster convergence.
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
  capped so the private buffers stay under 16 GB; `sc.Numerics(rmatvec_threads=n)`
  pins it (the action sets `SELFCAL_PARALLEL_RMATVEC`), and `1` selects the
  sequential kernel — the byte-exact verification mode the pre-2026-09-10
  goldens were made with. In practice the count equals the solver's threads
  (`sc.Numerics(threads)`) for every production tile: the buffer cap binds only when compaction is off
  (803 M uncompacted columns x 32 threads would want 103 GB), so all tiles of
  one mosaic are treated identically — but **keep `sc.Numerics(threads)` fixed
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
  class of difference as the row-split product.
- **Column-partitioned storage.** Above the `SELFCAL_BLOCK_NNZ` threshold,
  `setup_lsqr` writes the matrix as **storage blocks x column ranges** (one
  block per spill batch and per constraint block), placing rows with scipy's
  `coo_tocsr` in one pass per block. The default is ONE range, i.e.
  block-major storage with one int32 indptr per block (the same bytes per row
  as a plain `BlockCSR`, but built without the int64 global indptr and with
  the per-row sorts threaded). `sc.Tuning(split_ranges=T)` (`SELFCAL_RMATVEC_SPLIT`) cuts T column
  ranges, which only the sequential verification kernel uses (one thread per
  range folds its columns in the sequential order — the byte-equal parallel
  transpose product of the 2026-09 byte-equality rounds); T ranges hold
  `(T-1) * 4 B/row` more indptr, and `sc.Tuning(split_extra_gb=)` (`SELFCAL_SPLIT_EXTRA_GB`,
  default 24) halves T until that fits.
- **Memory knobs** (`sc.Tuning`, each a `SELFCAL_*` environment variable;
  defaults need no tuning): `block_nnz` (`SELFCAL_BLOCK_NNZ`) —
  nnz threshold at which `setup_lsqr` emits int32 block storage instead of a
  unified CSR (default `2**31`, the point where scipy would force int64
  indices; outputs are bit-identical either way). `spill_min_gb` (`SELFCAL_SPILL_MIN_GB`,
  default 4) / `spill_dir` (`SELFCAL_SPILL_DIR`, default system tmp) — `Calibrator.apply_lsqr`
  spills `pixel_counts`/`pixel_fisher`/`pixel_cross` to scratch for the
  duration of the solve when they exceed the threshold (exact byte
  round-trip; ~1 min I/O against a multi-hour production solve).
- **Process pools & the parallel scatter.** The assembly and Phase-4a CSR
  scatter pools (and the frame check of `field.reproject(verify=True)`) run on the
  **forkserver** start method (`selfcal.core.shmbuf.worker_pool_context`;
  `sc.Tuning(start_method=)`, `SELFCAL_MP_START_METHOD`, default `forkserver`; `fork` is a debugging
  escape hatch only); the other pools (coadd, reprojection, N-pass OFFSET
  refit, the standalone `wav_coadd`) use `multiprocessing`'s default start
  method (`fork` on Linux before Python 3.14). Forking a pool from the
  engine's multi-threaded process can hand a child the stderr lock in a
  locked state (the RSS guardrail prints
  every 15 s); the child then hangs at exit and the parent joins it forever —
  a production tile lost 6 h to this on 2026-09-09. Consequences: (1) entry
  scripts MUST keep their run code under `if __name__ == "__main__":`
  (children re-import the main module, the standard multiprocessing rule);
  (2) shared arrays reach workers as explicit `selfcal.core.shmbuf.SharedBuffer`
  handles (memfd-backed, fd-passed — no `/dev/shm` size cap), never by fork
  inheritance. The scatter: `sc.Tuning(scatter_workers=)` (`SELFCAL_SCATTER_WORKERS`, default
  `min(8, workers)`; `0`/`1` = the byte-identical serial path) and
  `scatter_timeout_s` (`SELFCAL_SCATTER_TIMEOUT_S`, default 1800 per batch) — on timeout or a
  broken pool the remaining batches are re-scattered serially, so an
  unattended run degrades to slow, never to a hang. Serial and parallel
  scatters are element-wise identical (pure data movement).
- `apply_lsqr` builds a custom row-block-parallel `LinearOperator` when `n_threads > 1`, with BLAS pinned to a single thread via `threadpool_limits(limits=1, user_api='blas')` so BLAS doesn't fight the SpMV threads. The engine passes `sc.Numerics(threads)` (default 48, tuned 2026-05); called directly, `Calibrator.apply_lsqr` defaults to `n_threads=32`.
- The solver's elementwise vector updates (`x += t1*w`, `u *= alfa`, ...)
  run across `sc.Tuning(vector_threads=)` threads (`SELFCAL_VEC_THREADS`, default 8; serial below 16 M
  elements). Each element depends only on its own inputs, so the split is
  bit-identical; the reductions (norms) stay one ordered pass.

`det_offset_funcs[m]` (in `Mosaicker.make_mosaic`) controls **mosaic-time** offset rendering — LSQR solves block-constant chunk offsets regardless (except for an offset term with a `coefficient` or `basis`, which the mosaic subtracts per observation instead). Default (`None`) renders chunks with `chunk_to_det` (block-constant, visible edges); SPHEREx LVF maps use `make_spherex_stripped_offset_map` (mean-preserving 2D spline over `r_edges, x_edges`). For multi-map mosaics, each map gets its own `det_offset_func` (or `None`), and `_prep_subframe` sums their grid contributions before a single `det_to_sub` interp. The engine takes each term's renderer from the instrument's `offset_renderer` (`sc.Offsets(render=)` chooses among them).

**Channel / window selection** is the action's `jobs=`: `spherex.channel(17)` (one
channel), `spherex.channels(1, 34)` (one job per channel), `spherex.group(17, 18)` (one
map over several channels), `spherex.window("Aromatic")` / `spherex.window("Aliphatic")`
(the named subchannel windows 225–235 / 249–259, covering the PAH bands), or
`spherex.window(name, subchannels=range(lo, hi))` (an explicit subchannel range, e.g.
the PAHfit windows `range(210, 250)` / `range(200, 260)`). The SPHEREx product tag is
`Detector{D}_NumSub{S}_NumCh{C}_NumCol{Co}`, so cal filenames are deterministically
`cal_{tag}_{job}_{name}.h5` (job = `Ch<n>` or the window name; `name` the recipe's).

## Advanced Calibrator solve modes

Beyond the default per-frame, per-chunk solve, `Calibrator.setup_lsqr`
supports restricted-solve modes that the mainline `sc.continuum` model does not use
but exist in the API (`sc.two_block` uses `det_groups_list`; in a model they are offset terms
with `per="all"` or `per=<frame variable>`). Each is an `OffsetBlock` field (named in
parentheses):

- **Locked offsets via `det_groups_list[m]`** (`det_groups`). Pass an array of length
  `num_frames` giving a group ID per frame for map `m`; frames in the same
  group share one offset vector. Reduces unknowns from
  `num_frames * num_chunks_m` to `num_groups_m * num_chunks_m`. Useful when
  the same pointing repeats and the spatial pattern is presumed constant
  within a group. Recover the pre-expansion offsets with
  `Calibrator.get_det_offset(m)`. **K=2 use case**: pair a free per-frame
  map at `m=0` with a `det_groups_list[1]=zeros` map at `m=1` to capture a
  detector-fixed pattern shared across all frames (e.g., readout-channel
  stripes — `sc.two_block`, `selfcal_scripts/runs/k2_readout.py`).
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

## N-pass alternating solve

One formalism for the spectral calibrations (SEP PAH J=2, NEP multi-line J=4):
`field.calibrate(recipe, passes=sc.Passes(n, ...))`, run by the engine's `npass` task
(`selfcal/run/npass.py`; primitives in `selfcal/pipeline/npass.py`). Model per frame
*k*, map pixel *p*:

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
| 1 | INIT | `S, a, s` jointly | the joint LSQR of a plain `calibrate` (tiled with `tiles=`) | yes (memory) |
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
scheduler records per-pass monitors in `<stem>_npass_monitor.json` (per-block
median / % positive / step RMS, offset DC, residual RMS, bright-cut
fallbacks) and `sc.Passes(stop_tol=)` can stop early; pick `n` from those, not by
assumption. The remaining zero points (line floor, continuum DC) are
unobservable from the data in any `n` and need the post-hoc anchors
(`selfcal/line_floor.py`, `selfcal/zodi_anchor.py`).

Why the SKY passes need no tiles: a pixel's normal equations are sums over its
observations, so per-tile dumps over **disjoint** frame sets are additive and
summing them is identical to a single full-field solve — no seam can exist.
Overlapping tile bboxes are de-duplicated first-tile-wins. The OFFSET pass
reads every frame of the field (the run's `frames=` directory). Verified at full
scale on the NEP (17,647 frames, J=4): re-running a SKY pass from the same
offsets with a completely different partition (3 vertical bands instead of 6
blocks) reproduced the product to float32 rounding — 4–87 differing elements
of 160.6 M, max 4.7e-10 against a p99 signal of 1.7–4.1e-2, Fisher and
coverage byte-equal, and the median difference **exactly zero in every
distance bin from either partition's boundaries**.

**The OFFSET basis must resolve the window** (`sc.Refit(segments=)`). The
per-frame refit is a degree-`degree` Chebyshev per column (`sc.Refit(degree)`) over the whole
polynomial window of the model. On the SEP, the same degree 4 over the 121-subchannel
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

**Ordering matters when INIT is tiled** (`sc.Passes(order=)`, default
`offset_first`). Each INIT tile is an independent joint solve, so it picks its own
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
(`sky_first` for even `n`) — `sc.Passes` refuses a schedule that ends on an OFFSET
pass, whose product carries no sky map, unless `ends_on_offset=True`.

Products: `<stem>_pass{i}sky.h5` (v3 sky-only cal: `sky/<name>`, Fisher,
coverage, `sky_separability/<name>`; written by the same
`selfcal/io/cal_writer.write_sky_groups` as `save_calibration`) and
`<stem>_pass{i}off.h5` (`offsets/map_0` + `frame_scalar` + `fit_ok` +
`resid_rms`, consumable by `OffsetSubtractor`). Running the same script again
reuses every pass product that is current and resumes at the first missing one.
The subtractors are post-weight frame hooks
(weights are computed on the raw data first); `SkySubtractor.window` handles
subframes overhanging any map edge (a negative `ref_coords` start is a Python
negative slice — the SEP LMC-streak bug).

Wall on the 192-core box: SEP (19,269 frames, J=2): SKY pass ≈ 1.5 h (two
halves + combine), OFFSET ≈ 50 min; NEP 1k probe (J=4, 121 subch): SKY ≈ 45
min, OFFSET ≈ 4 min.

## NVMe staging pattern

Reprojected `.h5` files live on RAID (HDD); parallel reads thrash the
heads. The pattern (in `selfcal/run/staging.py`, driven by
`sc.Compute(stage=, keep_staged=, io_limit=)`):

1. `set_hdd_io_limit(20)` — throttle the initial HDD copy
2. Copy `*.h5` to `{scratch}/reproj_nvme_{field name}/` (or `sc.Compute(stage_dir=)`) via
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
6. `shutil.rmtree(nvme_reproj_dir)` at the end, unless `keep_staged=True` or
   `stage="reuse"`, and only for a marked directory

A tiled run stages each tile's frames into `sc.Compute(stage_dir=)` under
the scratch directory, under the same marker rule, and never deletes it.

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
- `frame_scalar` — `(num_frames,)` float32 — per-frame DC scalar (only when the solve has one: `sc.Model(scalar=True)`, or a map with `det_groups`)
- `solve` — a group with attributes only (no dataset): the record of the solve that made the file
  (cals solved since 2026-10-08; see [The `solve` group](#the-solve-group) below)

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

### The `solve` group

The attributes of a cal file's `solve` group (`selfcal.core.solve_record`, read with
`CalFile.solve`) record how the solve that made the file ran and stopped.

| attribute | content |
| --- | --- |
| `version` | the record's layout (1) |
| `method` | `lsqr` or `lsmr` |
| `iterations`, `iterations_total` | iterations this solve ran; the cumulative count of the solution (equal unless the solve continued another's: then the source's total plus this solve's, -1 when the source's is unknown) |
| `iteration_limit` | `sc.Fit(iterations=)` |
| `istop`, `stop` | the solver's stop code (0-7, see [the tuning notes](#calibration-pipeline-tuning); 8: a stop rule of `sc.Fit(stop=...)` ended the solve) and its meaning |
| `r1norm`, `r2norm`, `arnorm`, `anorm`, `acond`, `xnorm` | the solver's final estimates: ‖b − A x‖, the same with the damping term, ‖Aᵀ r‖, ‖A‖, cond(A), ‖x‖ (in the column-scaled unknowns) |
| `true_residual`, `bnorm` | ‖b − A x‖ and ‖b‖, from one product at the end, accumulated in float64 |
| `atol`, `btol`, `conlim`, `damp` | what the solver ran with (`conlim = 0`: `sc.Fit(tolerance=0)`) |
| `rows`, `columns` | the system's shape (the active columns) |
| `history_file` | the NPZ of the solver's state at every iteration, `<field>/records/<cal stem>_history.npz` |
| `system`, `system_identity` | the identity of the system solved (`selfcal.core.warm_start.System`): the JSON of its frames (in order), reference grid (shape and WCS), sky terms, offset terms (chunk maps, groups of frames, basis functions, columns), per-frame scalar columns, total columns and job, and its sha256; what a later solve's start is checked against (cals solved since 2026-10-08) |
| `start_from`, `start_identity` | a continued solve only: the cal it started from (its path) and that cal's identity, `fingerprint:<sha256>` (its sidecar's) or `sha256:<sha256>` (its bytes) |
| `stop_rule`, `stop_iteration`, `stop_values`, `stop_policy` | a solve with [stop rules](#calibration-pipeline-tuning) only: the rules that ended it (`"gradient"`, `"residual, large_scale"`; `"none"` when it ended otherwise), the iteration they were judged at last (where they ended it, or the last), the JSON of every rule's state there (its value, threshold, window, where it was measured) and the JSON of the rules (`sc.Stop`) |
| `snapshot`, `iteration` | a [snapshot](#calibration-pipeline-tuning) only: `True`, and the cumulative iteration it holds (the start's `iterations_total` plus `iterations`; -1 when unknown). A snapshot has `iterations` (this solve's, at the snapshot), `iterations_total`, `iteration_limit`, the solver's estimates at that iteration (`r1norm` ... `xnorm`, `test1`, `test2`), the tolerances, the shape, the system and the start as above, and no `istop`, `stop`, `true_residual`, `bnorm` or `history_file` |

Every value there is a function of the solve, so a cal file stays byte-identical from run to run;
the solver's wall time (`wall_s`), which is not, is in the action's record (`solves`) only, with
the same values, and in the history (`elapsed_s`).

The group holds attributes only: every dataset and every root attribute of a cal is what it was
before records existed (the byte-equality gates compare those). A stitched cal and an N-pass pass
product have no `solve` group (each tile's cal has its own).

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
  bilinearly; the source of *layer* data variables (`sc.Layer("name")`).

Attributes:
- `sub_header` (bytes) / `det_header` (bytes) — FITS headers as strings;
  `load_reproj_file` reconstructs `sub_wcs` / `det_wcs` on demand. The
  keywords of `det_header` are the source of *header* frame variables
  (`sc.Header("MJD-AVG")`, `selfcal.io.frames.frame_header_values`).
- `file_path` (str) — path to the source FITS the subframe came from.
- `ref_coords` `(4,)` int32 — `[y_min, y_max, x_min, x_max]` in the
  reference frame, where `sub_data` should be splatted back. Can extend
  outside the mosaic; `selfcal.geometry.map_helper.compute_crop` handles the clip.

Sub-frame side length sized to fit the detector diagonal at mosaic
resolution:
`sub_width = ceil(sqrt(2) * max(det_height, det_width) / (ref_reso/det_reso) * (1 + 2*padding_percentage))`.

Cached intermediates from the mosaic's cache pass live in the scratch directory as
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
| `inputs` | the settings and files that decided the product's bytes: instrument, reference grid (`ref.fits`, by content), job, model (template and map files by content), fit, solve numerics and frames (by name) for a cal, and for a continued solve (`calibrate(start=...)`) the start's identity (`start`: its fingerprint, or the sha256 of its bytes without a current sidecar); plus tile box and assignment for a tile cal; tile fingerprints for a stitched cal; first-pass fingerprint, pass number/type and pass settings for an N-pass product; cal fingerprint, reference grid, model, coadd, coadd numerics and frames for a mosaic. A function sent by value counts by its source, defaults and closure values; a hook object by its class and state |
| `fingerprint` | SHA-256 of the canonical JSON of `inputs` |
| `size`, `mtime_ns` | the product when the sidecar was written: a product written again afterwards (another size or modification time) is refused as `changed` |
| `record` | the action's record that made it |
| `adopted` | true when `field.adopt(...)` recorded a product made without one |

An action reuses a product only when its fingerprint matches what it would make; one made by other
inputs, or one without a sidecar (made by a TOML run of an earlier version, say), is refused until
it is adopted or the action passes `overwrite=True`. The N-pass work directory keeps
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

# (alternatively, in the run script after the calibration:
#  spherex.zodi_anchor(result, predictions='<run>/zodi_preds') fits and records
#  the same anchor per channel — step 2 then already done.)

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
pytest, then the gates of `python_gates.py`, written with the Python API, against the float64-norm
goldens (`*golden_f64*`): the continuum and spectral cal gates (D3 Ch17 / D4 AromaticPAHfit), the
end-to-end D3 Ch17 cal + full mosaic, the npass n=3 probe (INIT + closed-form SKY + per-frame OFFSET
refit; no float64 golden yet), the Euclid EDFN recipe and the rerun of the continuum gate's record,
each compared dataset by dataset with `gates/h5_diff.py` / `gates/fits_diff.py` (or
`selfcal_scripts/drivers/diff_cal_h5.py`); `run_m13_gate.sh <tag>` runs the npass n=1 gate on the
NEP M13 tile (`run_gates.sh <tag> m13`). `config_equivalence.py views` / `compare-views` checks what
the engine does with every run script without a solve. The older harnesses
(`run_cal_baseline_test.py`, `regress_cal*.py`, the `benchmark_d3_ch17_*` timing scripts) are
archived under the gitignored `archive/scripts/benchmarks/`.
