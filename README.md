# selfcal

Self-calibration and mosaicking for imaging telescopes. From many overlapping exposures, selfcal
solves jointly, by sparse least squares (LSQR), for the sky and for the instrument's additive
offsets, then coadds the calibrated frames into a mosaic. It was built for the SPHEREx all-sky
survey and also calibrates Euclid. Another imager needs an `[instrument]` table in a run config, or
a small instrument class, and no change to the core.

## Install

```bash
pip install -e .            # the library (selfcal) and the run engine (selfcal_scripts)
pip install -e ".[docs]"    # optional: the documentation toolchain
```

The runtime dependencies are listed in [`pyproject.toml`](pyproject.toml).
[`environment.yml`](environment.yml) is an exact export of the processing host's conda
environment (Linux, Python 3.13), for reproducing that environment.

## Run

One TOML config per run, one command:

```bash
./selfcal_scripts/run.sh selfcal_scripts/configs/<run>.toml
./selfcal_scripts/run.sh selfcal_scripts/configs/<run>.toml --dry-run   # resolve jobs and mode only
```

## Documentation

The documentation site is built from [`docs/`](docs/) and the docstrings: `mkdocs serve` after
installing the docs extra. Its sources can also be read here:

- [`docs/getting-started/quickstart.md`](docs/getting-started/quickstart.md): a first run on
  simulated data.
- [`docs/guide/concepts.md`](docs/guide/concepts.md): how selfcal works.
- [`selfcal_scripts/configs/README.md`](selfcal_scripts/configs/README.md): the run configuration.
- [`docs/bring_your_own_telescope.md`](docs/bring_your_own_telescope.md): new instruments and
  models.
- [`PIPELINE.md`](PIPELINE.md): the operational runbook (tuning, file schemas, NVMe staging).
- [`selfcal/README.md`](selfcal/README.md): the code architecture.
- [`CLAUDE.md`](CLAUDE.md): repository layout and conventions.
