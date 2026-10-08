# Testing

selfcal is checked at three levels:

- **The pytest suite** (`tests/`): fast, self-contained, and run by CI on every pull request.
- **The byte-equality gates** (`selfcal_scripts/gates/`): real-data runs whose products must
  match fixed references byte for byte. They are run by hand on the processing host.
- **The documentation build** (`mkdocs build --strict`), also run by CI.

## The pytest suite

The suite needs no data and no network. Every test builds its own synthetic frames, exposures or
linear systems, so it runs on any checkout:

```bash
pytest                                        # everything, a few minutes
pytest tests/test_sky_model.py                # one file
pytest tests/test_any_telescope.py -k ghost   # tests whose name matches
pytest -x -q                                  # stop at the first failure, quietly
```

`pyproject.toml` sets `testpaths = ["tests"]` and puts the repository root on `sys.path`
(`pythonpath = ["."]`), so `import selfcal` resolves to the checkout under test and shared
helpers are imported as `tests.<module>`. Most test files also run as scripts, for example
`python tests/test_any_telescope.py`.

One test is skipped by default. It reproduces the process-pool fork hang of 2026-09-09 and runs
only with `SELFCAL_TEST_FORK_HAZARD=1`.

### What the tests cover

| Area | File | Checks |
| --- | --- | --- |
| End to end | `test_run_products.py` | products and records: atomic writes; fingerprints; a product reused only when current, refused when made by other inputs or unrecorded, adopted; a record rerun byte-identically; `compare`; `Tuning` applied and byte-neutral; `by_value` functions in a run; `selfcal convert`, its three rules (the converted script's objects, and a toy run of it) and what it refuses; `selfcal plan` and `selfcal adopt` on a run script; `submit` runs detached; a plan of a field without frames; two exposures make a reference grid |
| | `test_run_robustness.py` | products and records off the happy path: a product written again after its sidecar is refused, a mosaic whose cal is gone is made again, two actions in one second keep two records, `rerun --overwrite` remakes a mosaic and keeps the caller's directory, `submit` refuses settings a detached run cannot rebuild, a run-script function reaches the worker processes, hooks and `by_value` functions fingerprint the same in every process, interrupted writes are swept, `compare` sees infinities and file types, `convert` never loses an existing script |
| | `test_python_api.py` | the [Python API](../guide/python-api.md): settings checked when built; models, presets and instruments lowered for the run engine; a run through the API (calibrate and its mosaic, the mosaic action, a tiled solve); a non-square camera without a mask; the run-script rules (the `__main__` guard, importable functions); an instrument's geometry and optional hooks; map variables given as arrays; the engine's rules listed below |
| | `test_run_scripts.py` | every run script of `selfcal_scripts/runs/` builds and lowers without the data |
| | `test_quickstart_example.py` | the [quickstart](../getting-started/quickstart.md) example runs (simulate, quickstart, inspect, damping), its products and the mosaic's extensions are made, and it recovers its injected offsets and scalars |
| | `test_npass_toy.py` | the N-pass solve on a toy field: its products equal the recorded digests |
| | `test_any_telescope.py` | seven instruments and models written with the Python API, adapted with high-level functions only (see [Tutorials](../getting-started/tutorials.md#one-instrument-one-example)) |
| | `test_e2e_offset_recovery.py` | the LSQR solver recovers injected offsets on synthetic data |
| Products | `test_fingerprints.py` | the inputs and the fingerprint of every kind of product, for settings that cover every fingerprinted class, equal the committed golden `tests/data/fingerprints.json` ([Settings and fingerprints](settings.md)) |
| Model | `test_general_model.py` | data variables, offset bases, priors, observation weights and frame hooks, without a solve |
| | `test_model_spec.py` | the model as the engine takes it ([`ModelSpec`][selfcal.models.spec.ModelSpec]) |
| | `test_sky_model.py`, `test_sky_coefficients.py` | sky terms: a map times any coefficient function of data variables; the ready-made coefficient shapes |
| | `test_nsky_roundtrip.py` | a three-component sky model survives the save and the load exactly, with the v3 layout and its v2 aliases |
| | `test_offset_structure.py` | the generic chunk-axes builders reproduce the SPHEREx builders exactly |
| | `test_piecewise_offset_basis.py` | the segmented Chebyshev offset basis |
| Instruments | `test_euclid_instrument.py` | the Euclid exposure layout, chunk maps and axes, edge taper, renderers, hooks and grouped terms |
| | `test_spherex_precompute.py` | `spherex.precompute_lvf` writes the LVF arc parameters |
| | `test_grid_chunk_map.py` | rectangular chunk maps of any size |
| Solver | `test_constraint_builders.py`, `test_grouped_constraints.py` | the constraint rows (mean-offset anchors, sky and offset damping), grouped rows and per-map offset damping |
| | `test_closed_form_sky.py` | the closed-form sky-only solve equals the converged LSQR solution |
| | `test_prep_lsqr_arity.py` | the per-frame preparation returns the same number of values on every path, including a frame with no valid pixel |
| | `test_blockcsr_bitequal.py`, `test_colsplit_bitequal.py`, `test_x0_bitequal.py` | the memory- and speed-optimised matrix paths give the same bytes as the reference paths |
| | `test_rowsplit_rmatvec.py` | the parallel transpose product is deterministic, independent of the matrix storage, and equal to the sequential product up to float32 rounding |
| | `test_lsqr_norm64.py` | the solver's norms of long float32 vectors are accumulated in float64; short and float64 vectors take SciPy's call bit for bit |
| | `test_mp_fork_safety.py` | worker pools use the forkserver start method and pass shared arrays explicitly, so a pool started from a multi-threaded process cannot hang |
| Mosaic | `test_coadd_engine.py` | the coadd engine on synthetic reprojected frames |
| Run engine | `test_tiled.py` | tile geometry and the assignment of frames to tiles |
| | `test_npass_primitives.py` | the primitives of the N-pass alternating solve on a sky-only system |
| | `test_runlog.py` | the per-run log file |
| | `test_staging.py` | frames are staged atomically, only into or out of a directory the pipeline made; a tiled run takes every frame of its directory, in exposure order |
| Code layout | `test_import_direction.py` | layering: `selfcal.config` imports no other layer, the numerical layers never import the instrument layer, the instrument layer never imports the pipeline or run layers, and the run engine never imports the scripts |

`test_python_api.py` also checks the engine's rules: the instrument's geometry is built once for
`field.plan` followed by `field.calibrate`, and the kept geometry is handed out as a copy; every
pass of a run damps the sky as the model says; a setting declared as added later keeps the
fingerprints of existing products; and a model whose offsets are grouped by a frame variable of
the model's own can be mosaicked.

`tests/synthetic_exposures.py` is the shared helper that writes synthetic FITS exposures for the
end-to-end tests.

## Continuous integration

[`.github/workflows/ci.yml`](https://github.com/ThomasLiii/PySelfCal/blob/main/.github/workflows/ci.yml)
runs on every push to `main` and on every pull request:

| Job | Steps | Blocking |
| --- | --- | --- |
| `test` (Python 3.11 and 3.12) | `pip install -e ".[dev]"`, then `pytest -q` | yes |
| | `ruff check .` and `mypy` | no: the code still carries known lint and type findings, so these report without failing |
| `docs` | `pip install -e ".[docs]"`, then `mkdocs build --strict` | yes |

## The byte-equality gates

Every product of selfcal is byte-reproducible from run to run: the per-pixel sums are folded in a
fixed order and the coadd is deterministic. The gates use this to check structural changes. They
run real-data calibrations, written with the Python API, and compare every dataset of the
calibration and every extension of the mosaic with stored references, exactly.

Run them for any change that can alter numbers in the solve or the mosaic: the system build, the
constraint rows, the solver, the model lowering, the coadd, or the engine's handling of frames.
[Regression gates](gates.md) lists the gates and their references, and the engine views that check
a change to the engine without a solve. The gates read data that only exist on the processing
host, so CI does not run them.

## The documentation build

`mkdocs build --strict` fails on a broken link, a missing anchor, or a cross-reference to an
object that does not exist, in a page or in a docstring. Run it after changing docstrings or
pages; [Documentation](documentation.md) explains how the site is built.
