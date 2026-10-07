# Tutorials and examples

Beyond the [quickstart](quickstart.md), the repository holds three kinds of worked material:
notebooks that run the production recipes on real data, small end-to-end examples that each adapt
selfcal to a different kind of instrument, and the run scripts of the production campaigns.

## Notebooks

Both notebooks configure their runs with the Python API (an instrument, an `sc.Field`, an
`sc.Recipe`, then `field.calibrate`), read the calibration and the mosaic through the result
(`result.cal()`, a [`CalFile`][selfcal.io.calfile.CalFile]; `result.mosaic()`) and look at them.
Their paths point at frames already reprojected on the processing host. Elsewhere, edit the paths
in the first cell or set the `SELFCAL_DEMO_FRAMES`, `SELFCAL_DEMO_OUTPUT`, `SELFCAL_DEMO_CACHE`
and `SELFCAL_DEMO_WORKERS` environment variables (the SPHEREx notebook also reads
`SELFCAL_DEMO_N_FRAMES` and `SELFCAL_DEMO_D4_FIELD`). Each has a `REPROJECT` switch, off by
default, that reprojects raw exposures first.

| Notebook | What it shows |
| --- | --- |
| [`spherex_selfcal_demo.ipynb`](https://github.com/ThomasLiii/PySelfCal/blob/main/notebooks/spherex_selfcal_demo.ipynb) | One SPHEREx channel end to end: `sc.continuum` on 300 Detector 3 frames; the contents of the calibration and of the mosaic; the model the preset builds, its [`ModelSpec`][selfcal.models.spec.ModelSpec] and the plan of the run; and a spectral model on Detector 4 (a 3.29 µm aromatic template and a polynomial-basis offset), checked with `field.plan` without solving. |
| [`euclid_mosaic.ipynb`](https://github.com/ThomasLiii/PySelfCal/blob/main/notebooks/euclid_mosaic.ipynb) | Euclid NISP with the EDFN recipe written as an `sc.Model` on `sc.Euclid`: a detector-fixed pattern per detector on a grid of blocks, per-frame column and row stripes, a per-frame scalar; the fitted patterns and the mosaic, on 3 exposures x 16 detectors. |

Run them with Jupyter from the repository root, in an environment where selfcal is installed
(`pip install -e .` plus `jupyter`).

## One instrument, one example

[`tests/test_any_telescope.py`](https://github.com/ThomasLiii/PySelfCal/blob/main/tests/test_any_telescope.py)
calibrates seven different set-ups end to end on synthetic data. Each is an instrument (a camera,
or a small `sc.Instrument` subclass), a model and a few functions, with no change to the package,
and each test checks that the injected signal is recovered. They are the best templates for a new
instrument or model;
[Bring your own telescope](../bring_your_own_telescope.md#3-examples-each-is-a-few-lines-of-python)
explains every one.

| Test | Set-up |
| --- | --- |
| `test_time_variable_sky` | a sky term modulated with the season, read from a header keyword |
| `test_imaging_polarimeter` | an imaging polarimeter: I, Q and U maps from a half-wave-plate angle |
| `test_thermal_pattern_and_gradients` | a detector pattern scaled by the temperature, plus per-frame gradients |
| `test_data_cube_slices` | an integral-field cube read slice by slice, with a line map and inverse-variance weights |
| `test_frames_written_directly` | data that never were images, written as frame files, with per-frame gains |
| `test_amplifier_ghost` | amplifier crosstalk computed from each frame's own data |
| `test_heterogeneous_focal_plane` | detectors of different sizes on one focal plane |

Run one with `pytest -q tests/test_any_telescope.py -k polarimeter`.

## Runs on synthetic exposures

[`tests/test_run_products.py`](https://github.com/ThomasLiii/PySelfCal/blob/main/tests/test_run_products.py)
runs the Python API's actions on a toy field: a product reused, refused and adopted, a record run
again byte for byte, a comparison, a detached run, and the command line (`selfcal plan`,
`selfcal adopt`).
[`tests/test_runner_e2e_toy.py`](https://github.com/ThomasLiii/PySelfCal/blob/main/tests/test_runner_e2e_toy.py)
runs every task of the TOML runner on synthetic FITS exposures with the built-in `grid` instrument:
`reproject`, `cal` (with the mosaic), `mosaic` on an existing calibration, and `cal` with a
`[tiling]` table (two tiles and the Fisher stitch). `python tests/test_runner_e2e_toy.py` runs it
as a script.

## Production runs

[`selfcal_scripts/runs/`](https://github.com/ThomasLiii/PySelfCal/tree/main/selfcal_scripts/runs)
holds the run scripts of the SPHEREx campaigns: continuum channel maps (`numcol3_maps.py`, all 204
maps of the current recipe), spectral line fits, tiled fields and N-pass chains. They share the
fields, recipes and tilings of
[`selfcal_scripts/recipes/spherex.py`](https://github.com/ThomasLiii/PySelfCal/blob/main/selfcal_scripts/recipes/spherex.py)
and the machine of `selfcal_scripts/recipes/site.py`, point at data on the processing host, and
show complete, tested settings for each kind of run: `selfcal plan selfcal_scripts/runs/<name>.py`
prints what one would do.
[`selfcal_scripts/configs/`](https://github.com/ThomasLiii/PySelfCal/tree/main/selfcal_scripts/configs)
holds the same runs as TOML configs; [Run configuration](../guide/configuration.md) lists them and
documents every key.
