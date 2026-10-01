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
pytest                                        # everything: 110 tests, about two minutes
pytest tests/test_sky_model.py                # one file
pytest tests/test_any_telescope.py -k ghost   # tests whose name matches
pytest -x -q                                  # stop at the first failure, quietly
```

`pyproject.toml` sets `testpaths = ["tests"]` and puts the repository root on `sys.path`
(`pythonpath = ["."]`), so `import selfcal` resolves to the checkout under test and shared
helpers are imported as `tests.<module>`. Most test files also run as scripts, for example
`python tests/test_runner_e2e_toy.py`.

One test is skipped by default. It reproduces the process-pool fork hang of 2026-09-09 and runs
only with `SELFCAL_TEST_FORK_HAZARD=1`.

### What the tests cover

| Area | File | Checks |
| --- | --- | --- |
| End to end | `test_runner_e2e_toy.py` | every runner task on synthetic exposures with the `grid` instrument: `reproject`, `cal` and its mosaic, `mosaic`, `cal` with `[tiling]` and the Fisher stitch; the recovered offsets track the injected ones |
| | `test_quickstart_example.py` | the [quickstart](../getting-started/quickstart.md) example runs and recovers its injected offsets |
| | `test_grid_nomask_nonsquare.py` | a non-square detector without a data-quality extension through reproject, cal and mosaic; the `CalFile` reader and typed hooks on the product |
| | `test_any_telescope.py` | seven instruments and models adapted with high-level functions only (see [Tutorials](../getting-started/tutorials.md#one-instrument-one-example)) |
| | `test_e2e_offset_recovery.py` | the LSQR solver recovers injected offsets on synthetic data |
| Model | `test_general_model.py` | data variables, offset bases, priors, observation weights and frame hooks, without a solve |
| | `test_model_spec.py` | a `[model]` table lowers to the same solver objects as the named presets; `mode = "model"` runs end to end |
| | `test_sky_model.py`, `test_sky_coefficients.py` | sky terms: a map times any coefficient function of data variables; the ready-made coefficient shapes |
| | `test_nsky_roundtrip.py` | a three-component sky model survives the save and the load exactly, with the v3 layout and its v2 aliases |
| | `test_offset_structure.py` | the generic chunk-axes builders reproduce the SPHEREx builders exactly |
| | `test_piecewise_offset_basis.py` | the segmented Chebyshev offset basis |
| Instruments | `test_instrument_registry.py` | built-in and locally registered instruments resolve; mode presets are aliases of the structural recipes |
| | `test_euclid_instrument.py` | the Euclid exposure layout, chunk maps and axes, edge taper, renderers, hooks and grouped terms |
| Solver | `test_constraint_builders.py`, `test_grouped_constraints.py` | the constraint rows (mean-offset anchors, sky and offset damping), grouped rows and per-map offset damping |
| | `test_closed_form_sky.py` | the closed-form sky-only solve equals the converged LSQR solution |
| | `test_prep_lsqr_arity.py` | the per-frame preparation returns the same number of values on every path, including a frame with no valid pixel |
| | `test_blockcsr_bitequal.py`, `test_colsplit_bitequal.py`, `test_x0_bitequal.py` | the memory- and speed-optimised matrix paths give the same bytes as the reference paths |
| | `test_rowsplit_rmatvec.py` | the parallel transpose product is deterministic, independent of the matrix storage, and equal to the sequential product up to float32 rounding |
| | `test_mp_fork_safety.py` | worker pools use the forkserver start method and pass shared arrays explicitly, so a pool started from a multi-threaded process cannot hang |
| Mosaic | `test_coadd_engine.py` | the coadd engine on synthetic reprojected frames |
| Run engine | `test_tiled.py` | tile geometry and the assignment of frames to tiles |
| | `test_npass_primitives.py` | the primitives of the N-pass alternating solve on a sky-only system |
| | `test_runlog.py` | the per-run log file |
| Code layout | `test_import_direction.py` | layering: the numerical layers never import the instrument layer, and the instrument layer never imports the pipeline or runner layers |

`tests/synthetic_exposures.py` is the shared helper that writes synthetic FITS exposures for the
runner tests.

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
run real-data configurations and compare every dataset of the calibration and every extension
of the mosaic with stored references, exactly.

Run them for any change that can alter numbers in the solve or the mosaic: the system build, the
constraint rows, the solver, the model lowering, the coadd, or the runner's handling of
frames. [Regression gates](gates.md) lists the gates, their configs and their references. The gates
read data that only exist on the processing host, so CI does not run them.

## The documentation build

`mkdocs build --strict` fails on a broken link, a missing anchor, or a cross-reference to an
object that does not exist, in a page or in a docstring. Run it after changing docstrings or
pages; [Documentation](documentation.md) explains how the site is built.
