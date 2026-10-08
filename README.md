# selfcal

Self-calibration and mosaicking for imaging telescopes. From many overlapping exposures, selfcal
solves jointly, by sparse least squares (LSQR), for the sky and for the instrument's additive
offsets, then coadds the calibrated frames into a mosaic. It was built for the SPHEREx all-sky
survey and also calibrates Euclid. Another imager needs an `sc.Camera(...)`, or a small instrument
class, and no change to the core.

## Install

```bash
pip install -e .            # the library (selfcal) and the production scripts (selfcal_scripts)
pip install -e ".[docs]"    # optional: the documentation toolchain
```

The runtime dependencies are listed in [`pyproject.toml`](pyproject.toml).
[`environment.yml`](environment.yml) is an exact export of the processing host's conda
environment (Linux, Python 3.13), for reproducing that environment.

## Run

A run is a Python script ([the quickstart](docs/getting-started/quickstart.md) is one); the
production runs are in [`selfcal_scripts/runs/`](selfcal_scripts/runs/):

```bash
./selfcal_scripts/run.sh selfcal_scripts/runs/<run>.py
./selfcal_scripts/run.sh selfcal_scripts/runs/<run>.py --dry-run   # the plan only: nothing is computed
```

TOML run configs are no longer run; `selfcal convert run.toml` writes the run script of an old one
([`docs/guide/migrating-from-toml.md`](docs/guide/migrating-from-toml.md)).

## Documentation

The documentation is at <https://thomasliii.github.io/PySelfCal/>. It is built from
[`docs/`](docs/) and the docstrings (`mkdocs serve` previews it after installing the docs extra),
and its sources can also be read here:

- [`docs/getting-started/quickstart.md`](docs/getting-started/quickstart.md): a first run on
  simulated data.
- [`docs/guide/concepts.md`](docs/guide/concepts.md): how selfcal works.
- [`docs/guide/python-api.md`](docs/guide/python-api.md): the Python API, every setting.
- [`docs/bring_your_own_telescope.md`](docs/bring_your_own_telescope.md): new instruments and
  models.
- [`PIPELINE.md`](PIPELINE.md): the operational runbook (tuning, file schemas, NVMe staging).
- [`selfcal/README.md`](selfcal/README.md): the code architecture.
- [`CLAUDE.md`](CLAUDE.md): repository layout and conventions.
