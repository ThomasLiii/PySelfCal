# selfcal

A self-calibration and mosaicking pipeline for astronomical imaging — any
telescope, from SPHEREx (linear-variable-filter spectral imaging) and Euclid
(multi-detector broadband) to one that does not exist yet. It solves
simultaneously for sky maps and per-frame / per-chunk instrumental offsets by
casting the problem as a large, sparse linear least-squares problem, then
builds co-added mean / std / sigma-clipped mosaic products. Every function of
the model (sky coefficients, offset bases, weights, groupings, priors) reads
named per-observation *data variables* from pluggable sources, so a new
instrument or model is a set of high-level functions, never a core edit — see
[`../docs/bring_your_own_telescope.md`](../docs/bring_your_own_telescope.md).

The entry point is the **Python API**: a run is a Python script (or a notebook).
The production runs are the scripts of `selfcal_scripts/runs/`:

```bash
./selfcal_scripts/run.sh selfcal_scripts/runs/d4_aromatic.py            # or any run script
./selfcal_scripts/run.sh selfcal_scripts/runs/d4_aromatic.py --dry-run  # the plan: products made, reused or refused
```

A run script builds an **instrument** (`sc.SPHEREx`, `sc.Euclid`, `sc.Camera`, or an
`sc.Instrument` subclass), a **field** (`sc.Field`: one data set and its directory) and a
**recipe** (`sc.Recipe`: the model, the fit, the coadd and the summation layout), then calls
an action (`field.reproject`, `field.calibrate`, `field.mosaic`); see
[The Python API](../docs/guide/python-api.md). The action plans first, then lowers its
objects ([`run/lower.py`](run/lower.py)) into one `RunSpec` per group of jobs
([`run/runspec.py`](run/runspec.py)), in which every value is resolved: the library keywords
of the solve, the solver and the coadd, the frames and their staging, the tiles, the passes.
The run engine reads nothing else. `RunContext` ([`run/engine.py`](run/engine.py)) holds what
a run resolves once: the instrument's detector geometry (built once per action; kept between
actions when the instrument declares `geometry_is_pure`), the `ModelSpec` of the model, each
sky term's damping, every product name; its methods lower the model to the solver's objects
(offset and sky models, data variables, weight, priors, warm start, clip groups, the N-pass
refit basis). The tasks of
[`run/pipelines.py`](run/pipelines.py) (`cal`, tiled or not; `mosaic`; `npass`, scheduled by
[`run/npass.py`](run/npass.py); `reproject`) run on its two primitives, `solve_job` and
`mosaic_job`.

The engine calls the instrument only through the `sc.Instrument` contract
([`instruments/contract.py`](instruments/contract.py)), which the built-in instruments
implement themselves: `sc.SPHEREx`
([`instruments/spherex/settings.py`](instruments/spherex/settings.py)), `sc.Euclid`
([`instruments/euclid/settings.py`](instruments/euclid/settings.py)) and `sc.Camera`
([`instruments/camera.py`](instruments/camera.py)). It never names a telescope or a
calibration variant: a new telescope is an `sc.Instrument` subclass and a new variant an
`sc.Model` (or a function that returns one, as the presets are). Neither touches the engine,
and there is no registry. See [`../PIPELINE.md`](../PIPELINE.md) for tuning knobs.

Whatever the entry point, an end-to-end run flows through the three stage
classes in [`pipeline/pipeline_wrapper.py`](pipeline/pipeline_wrapper.py):

1. `Reprojector` — reproject raw FITS exposures onto a common reference
   WCS, one HDF5 per detector per exposure.
2. `Calibrator` — build and solve the LSQR system for the sky map and
   detector offsets.
3. `Mosaicker` — apply the calibration offsets and co-add into mean / std /
   sigma-clipped mosaics, optionally with a wavelength map.

## What the pipeline does

Given many overlapping detector exposures of the same patch of sky, every
observation `i` (one value of frame `k` on reference pixel `p_i`, seen at a
detector position) is modelled as

```
d_i = Σ_c coeff_c(v_i) * s_c(p_i) + Σ_m Σ_j o^(m)_{g_m(k)}(c_m(i), j) * φ_mj(v_i) + σ_k + eps_i
```

where:
- `v_i` are the observation's **data variables**
  ([`models/variables.py`](models/variables.py)): the built-in coordinates,
  detector maps (SPHEREx: the band-centre and band-width maps), per-frame
  values (time, filter, an angle, a temperature), reference-grid maps, planes
  stored with the frame, and functions of those or of the whole frame.
- `s_c(p)` is the per-pixel amplitude of sky component `c` (shared across all
  frames), and `coeff_c(v_i)` is that component's known coefficient — any
  function of data variables; `coeff = 1` for a constant sky. The set of
  components is a `SkyModel` ([`models/sky_model.py`](models/sky_model.py)).
- `o^(m)_g(c, j)` is an offset for chunk `c` in frame group `g` under chunk
  map `m`, times `n_m` known functions `φ_mj` of data variables (an `OffsetBlock`
  `basis`; `n_m = 1`, `φ = 1` is the classic chunk offset). The sum runs over
  `K` chunk maps.
- `g_m(k)` is the frame→group mapping for map `m` (identity by default; any
  per-frame value groups frames — the detector, the exposure, the night).
- `σ_k` is an optional per-frame DC scalar (added when
  `use_per_frame_scalar=True`).
- `eps_i` is noise.

The calibrator flattens all valid pixels from all reprojected frames into a
single sparse matrix equation `A x = b` where `x` stacks the per-component
sky pixels, then K offset blocks (one per map), then optionally a per-frame
scalar block. LSQR / LSMR returns the offsets that best explain the data
assuming a common underlying sky, which is also recovered as part of `x`.
The exact column layout of `x` is owned by
[`core/layout.py`](core/layout.py)'s `SystemLayout`.

For SPHEREx, the "chunks" are curved arc-shaped subchannels of the linear
variable filter (LVF), optionally split further into vertical columns to
absorb column-dependent offsets. These chunks are generated by the helpers
in [`instruments/spherex/spherex_utility.py`](instruments/spherex/spherex_utility.py).
The multi-chunk-map API lets you stack independent parameterizations — e.g.,
LVF-curved subchannels on map 0 plus detector-fixed readout-channel stripes
on map 1 — and solve them jointly.

## Package layout

The package is organized into focused subpackages: `pipeline/` (orchestration),
`core/` (the sparse solve), `models/` (sky + offset abstractions),
`geometry/` (masking / interpolation / WCS), `io/` (reproject + frame
selection), and `instruments/` (the instruments: the contract and the built-ins). The curated
public API is re-exported from [`__init__.py`](__init__.py):

```python
from selfcal import PipelineConfig, Reprojector, Calibrator, Mosaicker
from selfcal import SkyModel, SkyComponent, ContinuumComponent, Coefficient, ImportedFunction
from selfcal import GaussianProfile, TemplateProfile
from selfcal import OffsetModel, OffsetBlock, Basis, SystemLayout
from selfcal import VariableSet
from selfcal.models.variables import Derived, FrameFunction   # (sc.Derived / sc.FrameFunction: the Python API's sources)
from selfcal import TiledCalibration, TileSpec, make_tile_grid
```

Lower-level functions are imported from their submodules directly. There is no
`MakeMap` facade — the monolithic re-export shim was deleted; consumers import
the real homes below.

### High-level orchestration

- **[`pipeline/pipeline_wrapper.py`](pipeline/pipeline_wrapper.py)** — Dataclass
  + three user-facing classes.
  - `PipelineConfig` — Holds `output_dir`, `run_name`,
    `resolution_arcsec`, and auto-derives `ref_path`, `reproj_dir`,
    `cal_dir`, `mos_dir` underneath the run directory.
  - `Reprojector` — Loads exposure list, defines a reference WCS (either
    from scratch via `find_optimal_frame` or loaded from disk), then runs
    batch reprojection in parallel. Provides `get_reproj_files` for
    re-discovering output HDF5s.
  - `Calibrator(Reprojector)` — Builds the sparse LSQR system
    (`setup_lsqr`), runs the solve (`apply_lsqr`), saves a calibration HDF5
    (`save_calibration`) with the per-component skymap(s), K per-map offset
    arrays, optional per-frame scalar, and coverage information. Always-list
    API for K maps: `chunk_maps=[…]`, `reg_weights=[…]`, `adj_infos=[…]`,
    `det_groups_list`, `det_templates`, `mean_offsets_list`, plus
    `poly_constraints_list` for polynomial-degree constraints along
    user-supplied chunk chains and `use_per_frame_scalar=True` to add an
    explicit per-frame DC scalar block (decoupled from `det_groups_list`).
    The forward-looking spelling bundles these into a single `offset_model=`
    (`OffsetModel`) and `sky_model=` (`SkyModel`); both lower to the exact
    same parallel-list kwargs and are byte-equal. `load_calibration` handles
    the legacy single-map, the multi-map, and the multi-sky-component
    schemas.
  - `Mosaicker(Reprojector)` — Loads a calibration file (multi-schema
    reader), applies the K per-map offsets (each via its own
    `det_offset_funcs[m]`, summed into a single grid offset before
    `det_to_sub`), and produces `mean`, `std`, and sigma-clipped mean maps
    via parallel co-addition. Writes a multi-extension FITS with WCS
    headers via `save_mosaic`, and can append auxiliary maps (e.g.
    wavelength maps) with `append_maps`.

- **[`pipeline/tiled.py`](pipeline/tiled.py)** — `TiledCalibration` +
  `TileSpec` + `make_tile_grid`: a reusable tiled-calibration wrapper that
  splits a large reference frame into overlapping tiles, calibrates each
  tile independently (assigning frames per tile via center / overlap
  filters), then merges the per-tile cal files' sky maps into one stitched
  cal (Fisher-weighted inverse-variance average). Replaces the
  former chunked-NEP copy-paste driver.

### Core computation (`core/`)

The old monolithic `lsqr.py` was split into three focused modules; a
re-export shim ([`core/lsqr.py`](core/lsqr.py)) keeps `selfcal.core.lsqr`
importable for existing call sites, but import from the focused homes for
clarity:

- **[`core/assembly.py`](core/assembly.py)** — `_prep_lsqr` (the per-task row
  builder) and `_prep_lsqr_batch_worker` (the shared-memory batch worker).
- **[`core/system.py`](core/system.py)** — `setup_lsqr` orchestration plus the
  coverage/Fisher parsers (`parse_pixel_counts`, `parse_pixel_fisher`, …) and
  `apply_line_fisher_mask`.
- **[`core/solve.py`](core/solve.py)** — `apply_lsqr` and the thread-parallel
  SpMV `LinearOperator` (`_partition_csr`, `_make_parallel_operator`).
- **[`core/solve_record.py`](core/solve_record.py)** — `SolveRecord` (how a solve
  ran and stopped, its final estimates, the true residual `|b - A x|`) and
  `SolveHistory` (the solver's state at every iteration), filled by the
  solvers: [`core/lsqr_inplace.py`](core/lsqr_inplace.py) and
  [`core/lsmr.py`](core/lsmr.py) (scipy's LSMR with the same hook), both
  bit-identical to scipy's.
- **[`core/warm_start.py`](core/warm_start.py)** — `System` (the identity of a
  solve's system: frames in order, grid, sky and offset terms, columns, job;
  recorded in the cal's `solve` group) and `WarmStart` (a solve continued from
  an earlier cal, `field.calibrate(start=...)`: the cal checked against the
  system, and its solution read back as `x0`, the exact inverse of
  `save_calibration`).
- **[`core/snapshots.py`](core/snapshots.py)** — snapshots of a solve
  (`field.calibrate(snapshots=sc.Snapshots(every=k))`): the solvers call back
  every `k` iterations (`callback(itn, x)`, never at the last); `Iterate` reads
  the compact, column-scaled iterate in the physical full layout block by block
  (no second copy of `x`, each column as the end of the solve converts it);
  `SnapshotWriter` writes `snapshots/<cal stem>_it<NNNN>.h5` in the cal's schema
  (the parts that do not depend on `x` written once to a template, before the
  solve, by `Calibrator.write_cal_static`; the sky maps a band of chunk rows at
  a time, `Calibrator.write_cal_solution`), with retention.

- **[`core/subframe.py`](core/subframe.py)** — `_prep_subframe` is the single
  shared routine that loads an HDF5 reprojected file and produces
  `(ref_coords, sub_data, sub_weight, chunk_contribs, sub_aux)` usable by
  both the LSQR builder and the co-adder. It decodes the bitmask, builds
  *one* bilinear-interpolation sparse matrix from `sub_mapping` (the
  detector-frame interp matrix is shared across all maps; only
  `compute_chunk_contrib(chunk_map[m], interp_matrix)` is called per map),
  optionally subtracts the **sum of per-map per-chunk offsets** via
  `chunk_offsets` + `det_offset_funcs` lists for the mosaic path, applies
  a validity weight, and stamps NaNs to zero. Returns `chunk_contribs` as
  a length-K list (LSQR path). Hooks are provided for `preprocess_func`
  and `postprocess_func`: each receives a `FrameContext` (the frame's
  identity, `sub_data`, `sub_weight`, `sub_mapping`, `ref_coords`, and after
  the offsets `sub_aux`) and returns the new `sub_data` — e.g. Euclid's star
  mask, or the N-pass offset/sky subtractors. (The solve takes both,
  `sc.Fit(raw_frame_hook=, frame_hook=)`; the mosaic only the second,
  `sc.Coadd(frame_hook=)`.)

- **`setup_lsqr` (in [`core/system.py`](core/system.py))** — Sparse matrix
  construction. Builds `A`, `b`, and a per-map `pixel_counts` coverage list
  by running `_prep_lsqr_batch_worker` processes that share large arrays
  (per-map `chunk_maps[m]`, `grid_valid_weight`, per-map `adj_infos[m]`)
  via `multiprocessing.shared_memory` to avoid pickling. Each worker
  emits its partial rows/cols/data/b into new shared segments (or, with
  `batch_spill_dir` — the engine passes the run's scratch directory — into files on
  scratch) which the main process places into one CSR matrix (a `BlockCSR` /
  `ColSplitCSR` above `SELFCAL_BLOCK_NNZ`) **in batch-id order**
  (deterministic across runs). `col_bases` (length `K+1` array) marks
  the column boundary between maps and the optional scalar block; the
  full column layout is computed once by `SystemLayout`. Supported features:
  - **Multi-chunk-map offsets**: K independent offset blocks, summed
    additively in the model. Each map can independently set its own
    adjacency reg, poly constraints, mean anchor, grouping, or
    template.
  - **Grouped offsets** (`det_groups_list[m]`): frames in the same
    group share one offset vector for map `m`. Reduces unknowns from
    `num_frames * num_chunks_m` to `num_groups_m * num_chunks_m`.
  - **Template mode** (`det_templates[m]`): fix the spatial pattern
    and solve only for a per-frame amplitude `alpha` for map `m`.
    Spatial regularization is auto-skipped for template maps.
  - **Spatial adjacency regularization**: adjacent chunks pulled
    toward each other with weight `reg_weights[m]` using pre-computed
    `adj_infos[m]`. Empty adjacency tuples (e.g. `NumCol=1`) are
    auto-demoted to `None`.
  - **Polynomial constraints** (`poly_constraints_list[m]`): each
    element is a list of `{'chains': (n, L) int, 'stencil': (L,)
    float, 'weight': float}` dicts. Each chain adds one row per frame
    enforcing `weight * Σ stencil[ℓ] · O[chains[r, ℓ]] = 0` — a
    polynomial-of-degree-(L-2) annihilator. SPHEREx column-linearity
    helper: `compute_column_polynomial_chains(chunk_map, num_columns,
    degree=1)`.
  - **Per-frame DC scalar** (`use_per_frame_scalar=True`): adds a
    length-`num_frames` block to `x` decoupled from the
    `det_groups_list` path. Combined with mean-zero anchors on each
    map, this pushes per-frame DC entirely into the scalar so chunk
    offsets only carry within-frame structure. Required for narrow
    channels where sparse chunk coverage was previously letting
    per-frame DC leak into scan-stripe residuals.
  - **Spectral / multi-component sky** (`sky_model=`, `det_aux=`):
    one sky block per `SkyModel` component (continuum + arbitrary
    spectral templates), with per-pixel line damping and a Fisher
    coverage map for masking the line at read time
    (`apply_line_fisher_mask`).
  - **Outlier rejection**: nMAD-based per-subframe z-score clip.
  - **Mean-offset constraints** (`mean_offsets_list[m]`): soft
    constraint forcing each frame's map-`m` chunk-offset mean to a
    prescribed value. Weight hardcoded at 10.0.
  - **Coverage-weighted damping**: extra rows that damp sky pixels
    proportionally to `sqrt(coverage)`, suppressing poorly-covered
    regions without over-damping well-covered ones.
  - Constraint rows (mean-offset, sky/offset damping) are assembled by
    the small builders in [`core/constraint_builders.py`](core/constraint_builders.py)
    (`mean_offset_block`, `sky_damping_block`, `offset_damping_block`).

- **`apply_lsqr` (in [`core/solve.py`](core/solve.py))** — Handles
  zero-column elimination, optional float32 downcast, column-norm
  preconditioning, and a custom thread-parallel `LinearOperator`: `A @ x` is
  row-parallel (a row's dot product is its own), and `A^T @ y` is ROW-SPLIT —
  each thread scatters its own rows into a private output buffer and the
  buffers are reduced in a fixed order — with BLAS pinned to a single thread
  via `threadpool_limits`. The row-split product is deterministic for a given
  thread count but not bit-identical to the one-chain sequential scatter
  (a float32 reassociation of ~1e-7 per product, ~1e-6 in the converged
  maps); `SELFCAL_PARALLEL_RMATVEC=1` selects the sequential kernel, the
  byte-exact verification mode. Both `lsmr` and `lsqr` solvers are
  supported. Column-layout-agnostic (works for any K
  and any scalar/no-scalar configuration). Also accepts a
  [`core/blockcsr.py`](core/blockcsr.py) `BlockCSR` (see below).

- **[`core/blockcsr.py`](core/blockcsr.py)** — `BlockCSR`: row-block list
  representation that `setup_lsqr` emits instead of one giant CSR once total
  nnz reaches 2^31 (env `SELFCAL_BLOCK_NNZ` overrides). A unified scipy CSR
  at that scale is forced to int64 indices (+nnz*4 bytes held, +nnz*8 upcast
  copy at construction, +50% index bytes per SpMV); per-block int32 gives
  the same bytes as the unified matrix under either transpose kernel:
  matvec is row-local, and a thread of the row-split product (or the whole
  sequential product, in verification mode) scatters rows in the same global
  row order whatever the block boundaries (`_sparsetools.csc_matvec`).
  `compute_x0_scalar_only` consumes it too;
  `compute_x0_from_Ab` (k2-style full-offset warm starts) intentionally does
  not.

  `ColSplitCSR` in the same module is that storage cut a second way, into
  column ranges: `sub[b][t]` is storage block `b` restricted to columns
  `cuts[t]:cuts[t+1]`, with local int32 column ids and its own indptr. The
  default is one range: block-major storage that `setup_lsqr` emits directly
  (see [`core/system.py`](core/system.py) Phase 3-5), placed by scipy's
  `coo_tocsr` with threaded per-row sorts and no int64 global indptr. The
  transpose product on it is the row-split kernel described under
  `apply_lsqr`; with `SELFCAL_RMATVEC_SPLIT=<T>` ranges and the sequential
  kernel selected, a thread per column range keeps each column's addition
  order, which is the byte-exact verification mode. `partition_block_csr` is
  the reference converter, used by the tests and by the solve-time fallback
  for a plain `BlockCSR`.

- **`parse_pixel_counts` (in [`core/system.py`](core/system.py))** —
  Separates sky-pixel coverage from per-map chunk coverage and returns lists
  of per-map coverage arrays.

- **[`core/coadd.py`](core/coadd.py)** — the coadd engine.
  `run_coadd_schedule(...)` runs every pass a mosaic needs; `compute_coadd_map(mode, ...)`
  is the single-pass API with four modes.
  - `cache`: runs `_prep_subframe` and writes each frame's **nonzero-weight
    pixels** (packed bbox mask + value vectors, `format='sparse-v1'`) — a
    single-channel SPHEREx frame keeps ~1 % of its pixels, so this is ~10x
    smaller than the dense bbox crops it replaced (which are still read).
  - `mean`: weighted mean map, `sum(d*w) / sum(w)`.
  - `std`: weighted standard deviation using a supplied mean map.
  - `sigma_clip`: weighted mean with per-pixel `|d - mean| <= sigma * std`
    clipping, using supplied mean and std maps.
  The schedule fuses the cache pass with the mean accumulation, and (given
  `wav_maps`, the LVF band-centre/width maps) accumulates the wavelength
  sums inside the sigma-clip pass from per-pixel values sampled once in the
  cache pass — there is no separate wavelength pass.
  Accumulators live in `SharedMemory`; each worker accumulates a batch into
  lazily-zeroed full-grid locals (only touched pages materialise) and
  flushes **only the batch's union window, in batch order** through a
  turnstile, so the maps are a pure function of (frames, batch size) —
  bit-reproducible at any worker count. Large read-only arrays (`chunk_map`,
  `grid_valid_weight`, `mean_map`, `std_map`, `det_aux`, band maps) are
  also staged in shared memory.

- **[`core/solution.py`](core/solution.py)** — Small utilities for the `x`
  vector:
  - `parse_x(x, ref_shape, num_offset_groups_list, num_chunks_list, num_frames=None)`
    splits `x` into `(skymap, [det_offset_0, …, det_offset_{K-1}], frame_scalar)`.
    Returns offsets as a list of K ndarrays. `parse_x_sky` exposes the
    per-component sky blocks for multi-component `SkyModel`s.
  - `encode_x(skymap, offsets)` concatenates sky + per-map offsets back
    into a single vector.
  - `compute_x0_from_Ab(A, b, ref_shape)` derives an initial guess
    (sky=0, full offset region from the diagonal least-squares estimate
    `A_off^T b / diag(A_off^T A_off)`) without re-reading any files.
  - `compute_x0_scalar_only(A, b, ref_shape, scalar_col_start)` seeds
    *only* the per-frame scalar block (≈ weighted mean of valid b per
    frame), leaving chunks and sky at 0. Use this when
    `use_per_frame_scalar=True`.

- **[`core/layout.py`](core/layout.py)** — `SystemLayout`: the single source
  of truth for the column layout of `x`
  (`[sky blocks | offset block 0 … K-1 | per-frame scalars]`). Built once and
  shared between `setup_lsqr` (to tell workers their per-map column bases) and
  `Calibrator` (to parse the solved `x` back into maps). Bit-identical to the
  historical inline computation.

### Sky / offset models (`models/`)

- **[`models/sky_model.py`](models/sky_model.py)** — `SkyModel`: an ordered
  tuple of named `SkyComponent`s, one sky block each. Every component is a
  per-pixel map times an optional `Coefficient` — any function of named data
  variables sampled at every observation (the instrument's `aux` maps):
  `Coefficient(variable, function)` takes any picklable callable
  `f(*arrays)` (`ImportedFunction` references one by import path) or an
  object with `evaluate(x, obs)`. No coefficient is `c = 1` (the bit-exact
  identity path). `SpectralComponent(name, profile, wavelength_key)` (alias
  `LineComponent`) is the historical spelling of
  `SkyComponent(name, Coefficient(wavelength_key, profile))`.
  `SkyModel.damp_weights` is the one per-term damping rule.
- **[`models/profiles.py`](models/profiles.py)** — ready-made coefficient
  functions of one variable (`GaussianProfile`, `TemplateProfile`,
  `LinearProfile`, with `QuadratureSigma` for a per-observation Gaussian
  width read from a second variable); their field names are historical.
- **[`models/offset_model.py`](models/offset_model.py)** — `OffsetModel` /
  `OffsetBlock` bundle the parallel length-K offset-config lists into
  one cohesive block per map. `OffsetModel.to_setup_kwargs()` lowers back to
  the exact flat kwargs `setup_lsqr` consumes (numerically identical, gated
  byte-equal). A block's `basis` (`Basis`: `n` known functions of data
  variables) makes its unknowns one coefficient per group × chunk × function;
  `n = 1` is a *coefficient* (a pattern times the temperature, a gain times a
  previous sky). The flat-kwarg API remains supported as the deprecated
  transitional spelling.
- **[`models/variables.py`](models/variables.py)** — data variables:
  `VariableSet` declares the sources of a solve (detector maps, per-frame
  values, reference-grid maps, stored layers, derived functions, frame
  functions); `ObservationVariables` evaluates them lazily for one frame's
  observations — the row assembly, the mosaic and the N-pass share it.
- **[`models/priors.py`](models/priors.py)** — user priors: a function of
  `TermInfo`s returning linear rows on the unknowns (ready-made:
  `frame_smoothness`, `sky_smoothness`, `toward_variable`).

### I/O & state (`io/`, `_state.py`)

- **[`io/reproj.py`](io/reproj.py)** — `load_reproj_file(file_path, fields)`
  reads a reprojected HDF5, transparently handling datasets, attributes, and
  derived `sub_wcs` / `det_wcs` objects, and encodes `(exp_idx, det_idx)`
  parsed from the filename (`reproj_basename` / `parse_reproj_basename` are
  the one place that name is spelled). Respects the global HDD I/O semaphore.
  A frame that cannot be read raises `FrameLoadError` — it is never silently
  dropped.

- **[`io/calfile.py`](io/calfile.py)** — `CalFile`: the one reader of every
  calibration product (v3 named sky blocks, v2, the legacy v1 layout, stitched
  cals, N-pass sky and offset products): `sky(name)`, `sky_coverage`,
  `sky_fisher`, `sky_separability`, `offsets` (per map, per frame),
  `frame_scalar`, `total_offsets()`, `reproj_list`, `chunk_maps`, `fit_ok`.
  The mosaicker and the N-pass readers go through it.

- **[`io/reprojection.py`](io/reprojection.py)** — `batch_reproject(...)`
  iterates over `(exposure, sci_ext, dq_ext)` tasks and calls
  `reproject_interp / reproject_exact / reproject_adaptive` (from the
  `reproject` package) in a process pool. It sizes one square subframe for
  every frame, big enough to contain the diagonal of a sample detector (the
  first exposure's first science entry) after reprojection (with padding).
  For each frame it reprojects the science image, the detector
  pixel coordinates and (when the exposure has one, `dq_ext` not None) the DQ
  bitmask — detectors need not be square — and writes a zstd-compressed HDF5
  file per (exposure, detector)
  containing `sub_data`, `sub_foot`, `sub_bitmask`, `sub_mapping`, and both
  the detector-frame and subframe WCS headers as attributes.

- **[`io/exposure_filter.py`](io/exposure_filter.py)** — Header-driven
  exposure selection: `cached_header_values` (cached FITS header reads) and
  `filter_exposures_by_header` (predicate over header keys).

- **[`io/frame_select.py`](io/frame_select.py)** — Spatial frame selection
  for tiled/windowed solves: `compute_overlapping_frames`,
  `load_ref_coords_table`, `filter_by_center` — pick the reproj files whose
  footprint touches a given chunk bbox (used by `pipeline/tiled.py`).

- **[`_state.py`](_state.py)** — Module-level mutable state shared between
  the worker and parent processes.
  - `_hdd_io_semaphore` — bounded semaphore limiting concurrent HDD
    reads; set via `set_hdd_io_limit(n)`. Essential when many workers do
    random reads on a RAID array, where seek thrashing kills
    throughput.
  - `progress_enabled` — whether library calls draw tqdm progress bars;
    set via `set_progress(enabled)`.

  (The turnstile that orders the coadd workers' per-batch flushes,
  `_coadd_turn` — a `Condition` and per-stripe batch counters — lives in
  [`core/coadd.py`](core/coadd.py) and reaches the workers through the
  `_init_coadd_worker` pool initializer.)

### Geometry, masking, and interpolation helpers (`geometry/`)

- **[`geometry/map_helper.py`](geometry/map_helper.py)** — Low-level
  numerical utilities.
  - Bitmask: `bit_to_bool` / `bool_to_bit`, with optional `ignore_list`
    and per-bit expansion.
  - Weighting: `make_weight` (Poisson weight `1/sqrt(|d| + floor)`), `find_outliers`
    (nMAD-based).
  - Chunk machinery: `chunk_to_det`, `det_to_sub`, `make_linear_interp_matrix`
    (vectorized bilinear-interp sparse CSR matrix), `compute_chunk_contrib`,
    `compute_chunk_adjacency`, `compute_crop`, `make_grid_chunk_map`
    (regular square grid for broadband instruments).
  - Binning: `bin2d`.
  - Splines: `linear_spline`, `mean_preserving_spline` (1D, `pchip` /
    `akima` / `cubic`), `mean_preserving_spline_2d`.
  - Validity: `check_invalid`, `get_valid_bounds`.

- **[`geometry/wcs_helper.py`](geometry/wcs_helper.py)** — Reference frame
  construction.
  - `find_optimal_frame(exposure_list, resolution_arcsec, ...)` loads
    detector WCSs from selected FITS extensions and delegates to
    `reproject.mosaicking.find_optimal_celestial_wcs` with a configurable
    pixel-padding margin.
  - `derive_reference_from` builds a reference WCS aligned to an existing
    one (so cal outputs land on a shared grid); `projections_match` /
    `projection_signature` guard against mismatched projections.
  - `save_to_fits` / `load_from_fits` persist reference frames as a
    zero-filled primary FITS image of the grid's shape carrying the WCS header.

### Instrument-specific helpers (`instruments/`)

- **[`instruments/contract.py`](instruments/contract.py)** — `sc.Instrument`,
  the contract the run engine calls and the only one: `geometry(oversample)`
  (the chunk maps with their axes and the detector maps; `Geometry(...)` builds
  one), `layout()` (how a raw exposure is read), `default_jobs()`,
  `job_geometry(geom, job)` (a job's valid pixels and weights),
  `frame_variables(frames)` / `frame_variable_names()`, `frame_groups(frames)`
  (default: the integer-valued frame variables), `product_tag`, `unit`
  (default `''`), `geometry_files()` (the files its geometry reads) and
  `geometry_is_pure` (a class constant, default False, True on the built-in
  instruments: the geometry depends only on the settings and those files, so
  the engine may keep it between actions; otherwise each action builds it),
  and the optional hooks `offset_renderer`, `aux_coadds`,
  `finalize_mosaic` and `coefficient_catalog`; everything but `geometry` has a
  default. `sc.Job` is one unit of an instrument's loop. A new telescope is a
  frozen-dataclass subclass; the built-in instruments are subclasses too.

- **[`instruments/base.py`](instruments/base.py)** — The typed geometry the
  instruments return: `ChunkMap` (a chunk partition at detector and
  grid resolution with the AXES of its chunk grid —
  `selfcal.models.offset_structure.ChunkAxes` — plus which axes the standard
  block regularises along, which is the spectral axis and which the group
  axis), `DetectorGeometry` (chunk maps by name, named aux maps such as a
  wavelength map), `JobGeometry` (per-job valid weights) and `ExposureLayout`
  (how the reprojection stage reads a raw exposure). The selfcal core takes
  plain arrays and never imports `instruments`.

- **[`instruments/camera.py`](instruments/camera.py)** — `sc.Camera`: any
  single-detector imager without code (detector shape, rectangular chunk grid,
  FITS extensions or a reader, optional detector maps and header variables).
  The test suite's end-to-end runs use it on synthetic exposures.

- **[`instruments/spherex/settings.py`](instruments/spherex/settings.py)** — `sc.SPHEREx`,
  the reference implementation of the instrument contract: builds the
  detector-level geometry (the LVF stripped chunk map with its
  `(subchannel, column)` axes, the readout-channel map, the BC/BW aux maps),
  supplies per-job valid masks + weights, the arc offset renderer for the
  mosaic, the wavelength coadd + finaliser, the L2b exposure layout (FINAST
  filter) and the data unit. Its module also holds the jobs (`channel`,
  `channels`, `group`, `window`), the line terms (`line`), `precompute_lvf`
  (the LVF arc fit) and `zodi_anchor` (the anchor of a calibration's result).
  [`instruments/spherex/line_catalog.py`](instruments/spherex/line_catalog.py)
  holds the named coefficients (`pah_3p29`).

- **[`instruments/spherex/spherex_utility.py`](instruments/spherex/spherex_utility.py)** —
  SPHEREx LVF geometry and chunk-map construction.
  - `load_calibration(band, calibration_dir)` reads the
    `BC` (band-center wavelength) and `BW` (bandwidth) FITS maps for a
    given detector band.
  - `fit_lvf_arcs` / `fit_lvf_params` fit a shared center `(xc, yc)` plus
    per-channel radii `R_i` to the observed iso-wavelength contours in
    the `BC` map, modeling the LVF as concentric arcs. Results can be
    cached to `.npy` with `save_lvf_params` / `load_lvf_params` (the
    `load_lvf_params` path is resolved package-relative from the shipped
    `instruments/spherex/data/lvf_params/`, overridable via
    `$SELFCAL_LVF_PARAMS_DIR`).
  - `make_spherex_chunk_map`, `make_fiducial_chunk_map`,
    `make_stripped_chunk_map` generate pixel-level chunk-ID maps for the
    detector. The `stripped` variants additionally split each arc
    subchannel into `num_columns` vertical strips (chunk ID `= subchannel
    * num_columns + column`), allowing the calibrator to absorb
    column-dependent offsets.
  - `make_fiducial_chunk_mask`, `make_stripped_chunk_valid_mask`
    generate boolean masks over chunk IDs, selecting the subset of
    subchannels to use when solving for a given channel.
  - `compute_column_adjacency` / `compute_subchannel_adjacency` emit the
    `(i, j)` chunk pairs used by the spatial regularization term in
    `setup_lsqr` — e.g., within-subchannel column neighbors only (no arc
    crossings) for column smoothness, or cross-subchannel same-column
    neighbors for vertical smoothness.
  - `compute_column_polynomial_chains(chunk_map, num_columns, degree=1)`
    emits sliding-window chains of length `degree+2` over the column
    indices of each subchannel, plus the matching finite-difference
    stencil (`[1, -2, 1]` for `degree=1`, `[1, -3, 3, -1]` for
    `degree=2`, etc.). Pass to `setup_lsqr(poly_constraints_list=...)`
    to enforce polynomial-of-degree-N offsets across the columns within
    each subchannel. `compute_subchannel_polynomial_chains` is the
    cross-subchannel analogue.
  - `make_spherex_offset_map` / `make_spherex_stripped_offset_map`
    convert a solved per-chunk offset vector into a smooth
    high-resolution offset map using mean-preserving splines in `(R, x)`,
    suitable for subtraction during co-addition.
  - `compute_offsets_guess` computes a fast per-frame, per-chunk mean of
    the raw FITS data (no reprojection) to seed either the LSQR initial
    guess or the mean-offset constraint.
  - `fill_invalid_offsets` fills zeros in a 2-D offset grid via
    Delaunay-linear interpolation with nearest-neighbor fallback.
  - `fast_vertical_dist` computes a per-pixel distance-to-edge metric
    used as a smooth vertical tapering weight.

- **[`instruments/spherex/wavemap.py`](instruments/spherex/wavemap.py)** —
  `wav_coadd(det_BC, det_BW, mean_map, std_map, reproj_list, cache_list,
  ref_shape, sigma, ...)` produces per-pixel `wav_mean` and `wav_std` maps
  (effective wavelength and its spread per mosaic pixel) by running a
  multi-process sigma-clipped weighted coaddition of the LVF band-center and
  band-width values mapped through each exposure's `sub_mapping`. This is
  the standalone (pre-2026-09) path over an intermediate cache of either
  format; the engine now folds the same sums into the mosaic's sigma-clip
  pass (`make_mosaic(wav_maps=...)`, see `core/coadd.py`) and only calls
  `wav_coadd` when sigma clipping is off.

- **[`instruments/euclid/settings.py`](instruments/euclid/settings.py)** —
  `sc.Euclid`: the 16-detector NISP exposure layout, the grid / stripe / tilt
  chunk maps, the spline / strip / ramp renderers, the edge taper and electron
  units. [`instruments/euclid/hooks.py`](instruments/euclid/hooks.py) holds the
  recipe's per-frame hooks, `StarMask` and `ResidualMask`.

- **[`instruments/euclid/exposures.py`](instruments/euclid/exposures.py)** —
  Simple exposure-list helpers for Euclid data: `load_from_radius` (filter a
  VOTable by angular separation to a target), `load_from_csv`,
  `load_from_directory`. Euclid sci/DQ ext + square chunk-map conventions
  live in [`instruments/euclid/conventions.py`](instruments/euclid/conventions.py).

### Post-cal zodi anchor

- **[`zodi_anchor.py`](zodi_anchor.py)** — Core math + anchor-file I/O +
  read-time consumer for the post-hoc zodi anchor (a per-channel
  re-leveling fit applied non-mutatingly at read time, leaving cal/mosaic
  pristine): `fit_anchor_for_channel`, `write_anchor` /
  `append_anchor_channel`, `Anchor` / `load_anchor`,
  `load_anchored_mosaic`. The driver-side build/diagnose tooling lives in
  `selfcal_scripts/zodi_anchor/`.

## Data flow through a single run

```
FITS exposures (+ sci_ext/dq_ext)
   |
   |  Reprojector.run_reproject  -> io.reprojection.batch_reproject
   v
reprojected/*.h5   (sub_data, sub_bitmask, sub_mapping, ref_coords, headers)
   |
   |  Calibrator.setup_lsqr      -> core.system.setup_lsqr (+ core.subframe._prep_subframe)
   v
A (sparse), b, per-map pixel_counts, col_bases
   |
   |  Calibrator.apply_lsqr      -> core.solve.apply_lsqr
   v
x -> (skymap(s), [det_offset_0, ..., det_offset_{K-1}], frame_scalar)
   |
   |  Calibrator.save_calibration  (schema: sky/<component> + offsets/map_m groups + frame_scalar)
   v
calibration/cal_*.h5
   |
   |  Mosaicker.load_calibration -> Mosaicker.make_mosaic -> core.coadd.run_coadd_schedule
   v
mosaic dict: {mean_map, std_map, sc_mean_map, wav_mean_map, wav_std_map}
   |            (wav maps: LVF instruments only, coadded inside the sigma-clip pass)
   v
mosaic/mosaic_*.fits  (multi-extension FITS with WCS and all maps)
```

## Key design decisions worth knowing

- **Shared-memory hand-off to workers.** Large arrays (chunk maps, grid
  weights, adjacency, mean/std maps, det_aux) are staged in
  `multiprocessing.shared_memory.SharedMemory` and rehydrated by each
  worker rather than pickled. This is both faster and dramatically
  reduces peak memory for multi-process pools.
- **Ordered window flush.** Coadd workers accumulate a batch into
  lazily-zeroed local grids and flush only the batch's union window into
  the shared accumulator, in batch order (a turnstile), so contention stays
  flat as workers scale and the maps depend only on the frames and the
  batch size — not on the worker count or completion order.
- **Sparse per-frame payloads.** Everything downstream of `_prep_subframe`
  carries only a frame's nonzero-weight pixels; the per-frame preparation
  itself builds the interpolation matrix, offsets and weights over the
  bounding box of the rows that can carry weight, and the per-frame offset
  render evaluates the spline only at the grid pixels that box reads
  (identical values, ~3 % of the grid).
- **HDD throttle.** A global `BoundedSemaphore` (`set_hdd_io_limit`)
  bounds concurrent HDD reads to avoid RAID seek thrashing. The engine
  typically copies reprojected HDF5s onto NVMe and disables the
  limit before calibration / mosaicking.
- **Sparse intermediate cache.** `core/coadd.py` in `cache` mode stores
  each frame's nonzero-weight pixels only (packed bbox mask + values), a
  small fraction of the full subframe (e.g. a single channel inside a
  multi-channel detector), so the downstream passes are linear in true
  signal and the cache is a few per cent of the frames' size.
- **Zero-column elimination and column-norm preconditioning.**
  `setup_lsqr` drops all-zero columns before the solve (so unseen sky
  pixels and inactive chunks do not bloat the iterate; `apply_lsqr` does it
  for a matrix that arrives uncompacted), then `apply_lsqr` rescales the
  remaining columns to unit norm. This is the standard Jacobi
  preconditioner for least squares and greatly improves LSMR/LSQR
  convergence.
- **One column layout, computed once.** `SystemLayout` is the single
  source of truth for the `x` column blocks, shared between `setup_lsqr`
  and `Calibrator` so the build and the parse cannot drift.
- **Engine ↔ instrument ↔ model separation.** The run engine never
  imports a telescope or names a calibration variant: it calls the
  instrument through the `sc.Instrument` contract
  ([`instruments/contract.py`](instruments/contract.py)) and lowers the
  model (`ModelSpec`), whose offset structure is expressed in the chunk
  map's axes ([`models/offset_structure.py`](models/offset_structure.py)),
  never in a telescope's vocabulary. Adding a telescope = an `sc.Camera`, or
  one `sc.Instrument` subclass; adding a calibration variant = an `sc.Model`
  (or a function that returns one) — neither touches the engine.
- **Resolved once.** An action lowers its objects once into `RunSpec`s, and
  each engine run resolves its `RunContext` once (the N-pass INIT shares its
  scheduler's). The detector geometry is built once per action and shared
  by its plan and its runs. An instrument that declares `geometry_is_pure`
  (its geometry depends only on its settings and its `geometry_files()`:
  the built-in ones) has it kept between actions too: the engine keeps the
  last two, keyed by the instrument's settings, the oversampling, the files
  the geometry reads and the environment variables that locate them, so
  `field.plan` followed by `field.calibrate` builds it once. Any other
  instrument's geometry is built by each action, never kept nor copied.
  Each sky term's damping is decided once (`SkyModel.damp_weights`), and the
  joint solve, the closed-form sky solve and the N-pass SKY pass read the
  same numbers.
- **Models over parallel lists.** `SkyModel` and `OffsetModel` bundle what
  used to be loose integers / parallel length-K kwargs. They lower to the
  identical flat kwargs (gated byte-equal) so the abstraction adds no
  numerical risk.

## Using the pipeline

The supported entry point is the Python API
([The Python API](../docs/guide/python-api.md)). Below it, the three stage
classes can be driven directly; a minimal flow mirrors what the engine does
for one job:

```python
import numpy as np
from selfcal import PipelineConfig, Reprojector, Calibrator, Mosaicker
from selfcal import OffsetModel, OffsetBlock, SkyModel, set_hdd_io_limit
from selfcal.core.solution import compute_x0_scalar_only

cfg = PipelineConfig(
    output_dir='/path/to/outputs',
    run_name='my_run',
    resolution_arcsec=6.2,
)

# Reprojection (usually done once per run)
rr = Reprojector(cfg, exposure_list=fits_paths)
rr.define_reference(padding_pixels=100, use_ext=[1])
rr.run_reproject(max_workers=50, sci_ext_list=[1], dq_ext_list=[2])

# Calibration: one offset map, smooth between neighbouring chunks, the per-frame
# mean anchored at 0, plus an explicit per-frame scalar (the offset's DC)
cc = Calibrator(cfg)
num_frames_run = len(cc.reproj_list)
cc.setup_lsqr(
    offset_model=OffsetModel(
        blocks=(OffsetBlock(chunk_map=det_chunk_map, adj_info=adj_info, reg_weight=0.1,
                            mean_offset=np.zeros(num_frames_run)),),
        use_per_frame_scalar=True),
    sky_model=SkyModel.continuum_only(),
    grid_valid_weight=weight,
    offset_regularization=True,
    weighted_damping=True, damp_weight=0.1,
    outlier_thresh=5.0,
    batch_size=50, max_workers=48,
)
x0 = compute_x0_scalar_only(
    cc.A, cc.b, cc.ref_shape,
    scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
    num_sky_blocks=cc.num_sky_blocks,
    active_mask=cc.active_mask,       # setup_lsqr compacted the zero columns
)
cc.apply_lsqr(x0=x0, iter_lim=50, solver='lsqr', damp=0,
              use_float32=True, n_threads=48)
cal_path = cc.save_calibration(cal_file='cal.h5')

# Mosaic
mm = Mosaicker(cfg)
mm.load_calibration(cal_path)
maps = mm.make_mosaic(
    chunk_maps=[grid_chunk_map],
    grid_valid_weight=grid_w,
    oversample_factor=2,
    det_offset_funcs=[None],          # or per-map smoothing funcs
    make_std_map=True,
    apply_sigma_clipping=True, sigma=2.0,
    cache_batch_size=50, coadd_batch_size=50,
    cache_intermediate=True, max_workers=48,
    wav_maps=(det_BC, det_BW),        # LVF band maps -> wav_mean_map / wav_std_map (optional)
)
mm.save_mosaic(mos_file='mosaic.fits', overwrite=True)
```

Sky terms with coefficients, offset bases, data variables beyond the detector
maps and user priors are further `setup_lsqr` arguments (`sky_model`, a block's
`basis`, `variables`, `priors`); see
[`../docs/bring_your_own_telescope.md`](../docs/bring_your_own_telescope.md). The
flat per-map keyword lists (`chunk_maps=`, `adj_infos=`, ...) are still accepted
but deprecated. For a K=2 example (LVF chunks + detector-fixed readout-channel
stripes shared across all frames), see `sc.two_block` and
[`../selfcal_scripts/runs/k2_readout.py`](../selfcal_scripts/runs/k2_readout.py).

## Dependencies

Declared in [`../pyproject.toml`](../pyproject.toml) (Python >= 3.11). Key
runtime libraries: `numpy`, `scipy`, `astropy`, `reproject`, `h5py`,
`hdf5plugin`, `threadpoolctl`, `tqdm`, `opencv-python` (cv2),
`scikit-image`, `matplotlib`; `mpsplines` is an optional extra
(`[mpsplines]`, git-only), needed only for `interp_1d(method='mp_external')`.

## File index

| File | Purpose |
| --- | --- |
| [`__init__.py`](__init__.py) | Curated public API re-exports + package docstring. |
| [`_state.py`](_state.py) | Shared HDD I/O semaphore and the progress-bar switch. |
| [`config/`](config/__init__.py) | The settings base class of the Python API (`base.py`), function references for the worker processes (`functions.py`), path resolution + `SelfCalConfigError` (`paths.py`). |
| [`priors.py`](priors.py) | Ready-made priors of the Python API (`sc.priors.frame_smoothness`, ...). |
| [`models/model.py`](models/model.py) | The Python API's model: `Model`, `Sky`, `Offsets`, `Poly`, functions, data-variable sources, `Prior`, the presets; lowers to `ModelSpec`. |
| [`run/`](run/__init__.py) | The Python API's actions and the run engine: `recipe.py` (`Recipe`, `Fit`, `Coadd`, `Numerics`, `Clip`, `ChunkGroups`), `schedule.py` (`Tiles`, `Passes`, `Refit`), `compute.py` (`Compute`, `Tuning`), `field.py` (`Field` and its actions), `lower.py` (objects -> `RunSpec`), `runspec.py` (`RunSpec`, the engine's input), `engine.py` (`RunContext`, the geometry cache, `solve_job`, `mosaic_job`), `pipelines.py` (the tasks), `npass.py` (the N-pass scheduler), `staging.py`, `plan.py`, `records.py` (records, `rerun`, submit requests), `result.py`, `convert.py` (an old TOML config -> a run script, `convert_file`), `products.py` (sidecars, fingerprints, `adopt` checks, N-pass intermediates), `equivalence.py` (`engine_view`: what the engine does with a run, for the views), `compare.py`. |
| [`__main__.py`](__main__.py) | The `selfcal` command line: `run`, `plan`, `adopt`, `convert`, `rerun`, `compare`. |
| [`io/atomic.py`](io/atomic.py) | `atomic_path`: products written under a temporary name, renamed when complete. |
| [`instruments/contract.py`](instruments/contract.py), [`instruments/camera.py`](instruments/camera.py) | The instrument contract (`sc.Instrument`, the only interface the engine calls) and `sc.Camera`; SPHEREx and Euclid implement it in `spherex/settings.py`, `euclid/settings.py`. |
| [`zodi_anchor.py`](zodi_anchor.py) | Post-cal zodi anchor math + anchor-file I/O + read-time consumer. |
| [`pipeline/pipeline_wrapper.py`](pipeline/pipeline_wrapper.py) | `PipelineConfig`, `Reprojector`, `Calibrator`, `Mosaicker`. |
| [`pipeline/tiled.py`](pipeline/tiled.py) | `TiledCalibration`, `TileSpec`, `make_tile_grid`, stitching. |
| [`core/assembly.py`](core/assembly.py) | `_prep_lsqr` + shared-memory batch worker. |
| [`core/system.py`](core/system.py) | `setup_lsqr` + coverage/Fisher parsers + line-mask. |
| [`core/solve.py`](core/solve.py) | `apply_lsqr` + thread-parallel SpMV operator. |
| [`core/solve_record.py`](core/solve_record.py) | `SolveRecord`, `SolveHistory`: the record of a solve (the cal's `solve` group, the action's `solves`, `records/<cal stem>_history.npz`). |
| [`core/warm_start.py`](core/warm_start.py) | `System`, `WarmStart`: the identity of a solve's system, and a solve continued from an earlier cal (`calibrate(start=...)`). |
| [`core/snapshots.py`](core/snapshots.py) | `Iterate`, `SnapshotWriter`: the solution every `k` iterations as a cal file (`calibrate(snapshots=...)`). |
| [`core/lsqr_inplace.py`](core/lsqr_inplace.py), [`core/lsmr.py`](core/lsmr.py) | scipy's LSQR (in-place vector updates, float64 norms of long vectors) and LSMR, each with the per-iteration history hook and the iteration callback. |
| [`core/blockcsr.py`](core/blockcsr.py) | `BlockCSR` int32 row-block matrix for nnz >= 2^31; `ColSplitCSR` row-blocks x column-ranges for the bit-equal parallel transpose product. |
| [`core/lsqr.py`](core/lsqr.py) | Back-compat re-export shim over assembly/system/solve. |
| [`core/subframe.py`](core/subframe.py) | Unified `_prep_subframe` used by coadd & LSQR. |
| [`core/coadd.py`](core/coadd.py) | Parallel mean / std / sigma-clip (+ wavelength) coaddition, sparse caching, deterministic flush order. |
| [`core/solution.py`](core/solution.py) | `parse_x`, `encode_x`, `compute_x0_from_Ab`, `compute_x0_scalar_only`. |
| [`core/layout.py`](core/layout.py) | `SystemLayout` — column layout of `x`. |
| [`core/constraint_builders.py`](core/constraint_builders.py) | Mean-offset / sky / offset damping constraint rows. |
| [`models/sky_model.py`](models/sky_model.py) | `SkyModel`, `SkyComponent` (a map times an optional `Coefficient`: any function of data variables), `ImportedFunction`. |
| [`models/profiles.py`](models/profiles.py) | Ready-made coefficient functions: `GaussianProfile`, `TemplateProfile`, `LinearProfile`, `QuadratureSigma`. |
| [`models/offset_model.py`](models/offset_model.py) | `OffsetModel` / `OffsetBlock` per-map offset bundling. |
| [`geometry/map_helper.py`](geometry/map_helper.py) | Bitmask, interp, chunk, spline, and binning utilities. |
| [`geometry/wcs_helper.py`](geometry/wcs_helper.py) | Reference WCS construction / derive / save / load. |
| [`io/reproj.py`](io/reproj.py) | `load_reproj_file` for reprojected HDF5s; the frame-name helpers; `FrameLoadError`. |
| [`io/calfile.py`](io/calfile.py) | `CalFile`, the reader of every calibration product. |
| [`io/reprojection.py`](io/reprojection.py) | Parallel batch reprojection onto the reference WCS. |
| [`io/exposure_filter.py`](io/exposure_filter.py) | Header-driven exposure selection (cached header reads). |
| [`io/frame_select.py`](io/frame_select.py) | Spatial frame selection for tiled / windowed solves. |
| [`models/offset_structure.py`](models/offset_structure.py) | Chunk axes + the generic offset-structure builders (adjacency, polynomial chains, hard basis, group edges). |
| [`models/spec.py`](models/spec.py) | `ModelSpec`: the model as the engine takes it (variables, sky terms, offset terms, weight, priors), built from an `sc.Model`, lowered to `VariableSet` / `SkyModel` / `OffsetModel` / prior callables. |
| [`models/variables.py`](models/variables.py) | `VariableSet`, `ObservationVariables`, `FrameObservations`: named per-observation data variables from any source. |
| [`models/priors.py`](models/priors.py) | `TermInfo`, `ModelPrior` and ready-made prior functions. |
| [`io/frames.py`](io/frames.py) | `ExposureData` + the default FITS reader (the exposure-reader contract), `write_frame` (the frame-file contract), `frame_header_values`. |
| [`pipeline/model_eval.py`](pipeline/model_eval.py) | Evaluating a solved model outside the solve: `BasisOffsetSubtractor` (the mosaic's per-observation subtraction of offset terms with a basis). |
| [`instruments/base.py`](instruments/base.py) | The typed geometry (`ChunkMap`, `DetectorGeometry`, `JobGeometry`, `ExposureLayout`). |
| [`instruments/grid.py`](instruments/grid.py) | Rectangular chunk-grid helpers (`rect_grid_chunk_map`, `upsample_chunk_map`). |
| [`instruments/euclid/settings.py`](instruments/euclid/settings.py) | `sc.Euclid`: the 16-detector NISP exposure layout, grid/stripe/tilt chunk maps, spline/strip/ramp renderers, edge taper, electron units. |
| [`instruments/euclid/hooks.py`](instruments/euclid/hooks.py) | The recipe's per-frame hooks (`StarMask`, `ResidualMask`). |
| [`instruments/euclid/adapter.py`](instruments/euclid/adapter.py) | Euclid helpers: the strip and grid chunk maps, their offset renderers, the edge taper. |
| [`instruments/spherex/settings.py`](instruments/spherex/settings.py) | `sc.SPHEREx` (chunk maps and axes, wavelength maps, renderer, layout), its jobs, `line`, `precompute_lvf`, `zodi_anchor`. |
| [`instruments/spherex/adapter.py`](instruments/spherex/adapter.py) | SPHEREx helpers: the readout-channel chunk map (`make_readout_chunk_map`) and the named subchannel windows (`SUBCH_WINDOWS`). |
| [`instruments/spherex/line_catalog.py`](instruments/spherex/line_catalog.py) | SPHEREx named coefficients (`pah_3p29`) + the `pah_3p29()` SkyModel factory. |
| [`instruments/spherex/spherex_utility.py`](instruments/spherex/spherex_utility.py) | SPHEREx LVF arcs, chunk maps, adjacency, offset-map splines. |
| [`instruments/spherex/wavemap.py`](instruments/spherex/wavemap.py) | Wavelength mean/std maps via multi-process sigma-clipped coadd. |
| [`instruments/euclid/exposures.py`](instruments/euclid/exposures.py) | Euclid exposure-list loaders. |
| [`instruments/euclid/conventions.py`](instruments/euclid/conventions.py) | Euclid sci/DQ ext + square chunk-map conventions. |
