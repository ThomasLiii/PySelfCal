# Installation

selfcal is installed from a checkout of its repository. The checkout holds more than the library:
the production run scripts, their shared recipes and the launcher (`selfcal_scripts/`), the
regression gates, the tools and the tests. `pip install` puts both packages on the path, and the
production runs are started from the checkout.

## Requirements

- **Python 3.11 or newer.** The coadd engine (`selfcal/core/coadd.py`) indexes arrays with star
  expressions (`arr[*idx]`), which Python 3.10 cannot parse. CI tests 3.11 and 3.12; the
  processing host runs 3.13.
- **Runtime dependencies** (from `pyproject.toml`; pip installs them): astropy, h5py, hdf5plugin,
  matplotlib, numpy, opencv-python, reproject, scikit-image, scipy, threadpoolctl and tqdm.
- **Memory and cores** scale with the run. The test suite and the [quickstart](quickstart.md) run
  on a laptop. Production SPHEREx channel maps are solved on a 192-core machine with hundreds of
  GB of memory; [the pipeline runbook](../guide/pipeline.md) has the tuning numbers.

## Install

```bash
git clone https://github.com/ThomasLiii/PySelfCal.git
cd PySelfCal
pip install -e .
```

`pip install -e .` installs two packages in editable mode: `selfcal` (the library, its Python
API and the run engine) and `selfcal_scripts` (the production run scripts and recipes, the gates
and the tools), and the `selfcal` command. Optional extras:

| Extra | Adds | For |
| --- | --- | --- |
| `pip install -e ".[dev]"` | pytest, pytest-cov, ruff, mypy | running the tests and the linters |
| `pip install -e ".[docs]"` | MkDocs, Material for MkDocs, mkdocstrings and its plugins | building this site |
| `pip install -e ".[mpsplines]"` | `mpsplines` (installed from GitHub; it is not on PyPI) | the external mean-preserving interpolator, `interp_1d(method='mp_external')`; the default offset rendering does not need it |

### Reproducing the processing host's environment

[`environment.yml`](https://github.com/ThomasLiii/PySelfCal/blob/main/environment.yml) is an exact
export of the conda environment used on the processing host: Linux x86-64 builds, Python 3.13,
every package pinned. Use it only to reproduce that environment; on another system, the
`pip install` above is enough.

```bash
conda env create -f environment.yml -n selfcal
conda activate selfcal
pip install -e .
```

## Check the install

```bash
python -c "import selfcal, selfcal.run; print(selfcal.__file__)"
pytest -q tests/test_quickstart_example.py  # about a minute
```

The second command runs the [quickstart](quickstart.md) in temporary directories: a complete
reprojection, calibration and mosaic of simulated exposures, checked against the injected offsets.
<!-- check: what test_quickstart_example.py checks once its TOML half is gone --> The whole suite
(`pytest`) takes a few minutes and needs no data. [Testing](../developer/testing.md) describes it.

## Threads and processes

selfcal parallelises with its own worker processes (`sc.Compute(workers=...)`) and with the thread
pool of the LSQR matrix-vector products (`sc.Numerics(threads)`). BLAS libraries that also start
one thread per core would oversubscribe the machine, so every action of the Python API limits them
to one thread per process before it starts a worker, and `selfcal run script.py` sets
`OMP_NUM_THREADS`, `OPENBLAS_NUM_THREADS`, `MKL_NUM_THREADS`, `VECLIB_MAXIMUM_THREADS` and
`NUMEXPR_NUM_THREADS` to 1 before numpy is imported.

Put a script's work under `if __name__ == "__main__":`. The worker pools use the `forkserver`
start method, which imports the main module again in each worker; an action started outside the
guard is refused. When you call the library's functions directly rather than through a field's
actions, set the thread variables yourself before the first `import numpy`, for example in the
shell.

## Optional: zodiacal-light predictions

The [zodiacal-light anchor](../tools/zodi-anchor.md) fits the calibrated levels to predictions of
the zodiacal light made with [zodipy](https://github.com/Cosmoglobe/zodipy). zodipy needs
numpy < 2, so the predictions are computed in a separate environment (`selfcal-zodipy` on the
processing host) and handed over as files; the fit and its use run in the main environment.

## Building this documentation

```bash
pip install -e ".[docs]"
mkdocs serve                                          # live preview at http://127.0.0.1:8000
DISABLE_MKDOCS_2_WARNING=true mkdocs build --strict   # what CI runs
```

[Documentation](../developer/documentation.md) explains how the site is put together.
