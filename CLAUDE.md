# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. The
user's working rules and the index of the project's skills, for every tree under `~/selfcal-project`, are in
`/home/thomasli/selfcal-project/CLAUDE.md` (outside the repository).

## Repo overview

`selfcal` is a sparse-LSQR self-calibration and mosaicking pipeline for SPHEREx (with Euclid helpers), driven by a
Python API (`import selfcal as sc`). Read first:

- [selfcal/README.md](selfcal/README.md): the module-level **code architecture**. It covers the data flow
  `FITS → reprojected/*.h5 → cal_*.h5 → mosaic_*.fits`, what each submodule does, and the design decisions. Read it
  before touching `selfcal/`.
- [PIPELINE.md](PIPELINE.md): the **operational runbook**. It covers:
  - the tuning knobs, by their Python names;
  - the advanced solve modes;
  - NVMe staging;
  - the on-disk schemas of `cal_*.h5`, reprojected `*.h5`, mosaic `*.fits` and the product sidecars.
- The documentation site, https://thomasliii.github.io/PySelfCal/ (sources in `docs/`). Its main pages are the Python
  API guide `docs/guide/python-api.md`, `docs/bring_your_own_telescope.md` and
  `docs/developer/{settings,testing,documentation}.md`.

## Layout

| Dir | What's there |
| --- | --- |
| `selfcal/` | The package. `run/` holds the API's actions (`field.py`, `recipe.py`, `compute.py`, `plan.py`, `records.py`, `products.py`) and the run engine (`lower.py` → `runspec.py` → `engine.py`, plus `pipelines.py`, `npass.py`, `staging.py`). Also `npass/`, `core/` (assembly, solve, coadd, gauge, deflation), `models/`, `geometry/`, `io/` (frames, reprojection, `calfile.CalFile`), `instruments/` (the `sc.Instrument` contract, `camera`, `spherex/`, `euclid/`), `config/` (the settings base) and `zodi_anchor`. See [selfcal/README.md](selfcal/README.md). |
| `selfcal_scripts/recipes/` | The campaigns' shared settings. `site.py` is this machine (`OUTPUTS`, `NVME`, `ORCA`). `spherex.py` holds the fields `nep(d)` / `sep(d)` / `nep_qr2(d)` (the detector only), the chunk maps every recipe carries (`maps(num_sub=10, num_ch=17, num_col=1)`: the LVF subchannels, primary, and the detector columns; `on_maps(recipe, ...)` moves a recipe to another partition) and the recipes: `CHANNEL_MAPS`, `SLOPE_GAUGE`, `SLOPE_GAUGE_VARWEIGHT` (weighted by the frames' `variance` layer), `SLOPE_GAUGE_COLUMNS` / `SLOPE_GAUGE_VARWEIGHT_COLUMNS` (plus per-column constants shared by every frame), `SUBCH_VARWEIGHT_COLUMNS` (free subchannel constants, no per-frame scalar) and `SUBCH_VARWEIGHT_COLUMNS_FINAL` (plus `exclude=SkySignature('all')` on the x-slopes and `Fit(min_observations=2)`: the final channel recipe of 2026-10), `continuous_final(channel)` (its per-frame subchannel constants continuous across the subchannels, one channel job), `NUMCOL3`, `POLY_K1`, `multiline3(...)`, ... |
| `selfcal_scripts/spherex/` | The SPHEREx model side, built from the API: `chunks.py` (the chunk-map functions `subchannels`, `subchannel_map(num_sub, num_ch, num_col)`, `detector_columns`, `readouts`), `jobs.py` (`channel(c)`, `channels(a, b)`, `group(...)`, `window(name, range)`, `WINDOWS`), `terms.py` (`frame_slope`, `spectral_offsets`, `line`, `pah_3p29`, ...). |
| `selfcal_scripts/runs/`, `launch/`, `run.sh` | One run script per run, with `FIELD`/`FIELDS`, `RECIPE` and `RUN` (or `REPROJECT`) at top level. They include the fiducial channel maps (`channel_maps.py`), the NEP/SEP multi-line fits, the reprojections, `precompute.py`, and the tool `expand_field.py`. `launch/<run>.sh` runs `run.sh` on its run script. |
| `selfcal_scripts/gates/` | The byte-equality gates (`run_gates.sh <tag>`, `make_goldens.sh`, `python_gates.py`), the engine views (`config_equivalence.py`) and the diff tools ([README](selfcal_scripts/gates/README.md)). |
| `selfcal_scripts/zodi_anchor/` | The post-hoc zodi anchor. It is non-mutating: anchor files are applied at read time (`selfcal/zodi_anchor.py`) ([README](selfcal_scripts/zodi_anchor/README.md)). |
| `selfcal_scripts/transfer_function/` | The transfer-function kit for collaborators ([README](selfcal_scripts/transfer_function/README.md)). |
| `selfcal_scripts/line_templates/` | `build_line_template.py`, which regenerates the shipped line templates. |
| `selfcal/instruments/spherex/data/` | Shipped SPHEREx data: the LVF params `lvf_params_D{1..6}.npy` (override with `$SELFCAL_LVF_PARAMS_DIR`; regenerate with `runs/precompute.py`), the channel table, the line templates. |
| `docs/` + `mkdocs.yml` | The documentation site (MkDocs Material + mkdocstrings). The guides that live next to the code are imported in place by `docs/scripts/gen_pages.py` (`docs/developer/documentation.md`). |
| `tests/` | The pytest suite: self-contained (no `/mnt`), about 14 minutes. |
| `notebooks/`, `examples/` | `spherex_selfcal_demo.ipynb`, `euclid_mosaic.ipynb`; `examples/quickstart/`. |
| `analysis/analysis_script/` | The 2026-05 analyses of the offset terms (`meeting_plots.py`, `verify_stack.py`, `zodi_utils.py`, `multiline/`). Not active since 2026-10. `zodi_utils.CAL_FILE_TEMPLATE` points at the QR2 NumCol3 cals. Archived one-offs (`archive/analysis_script/`) run with `PYTHONPATH=analysis/analysis_script`. |
| `cache/`, `figures/`, `archive/` | Gitignored: scratch and NVMe staging (`cache/reproj_nvme_*`), old renders, old scripts. |
| `workspace/` | Gitignored link to the shared `/home/thomasli/selfcal-project/workspace`, one `workspace/<slug>/` per effort (`brief.md`, `report.md`, `scripts/`, `figures/`, `logs/`; `workspace/README.md`). **`workspace/DATA_LOCATIONS.md` tracks every data set.** Read it to find data, and update it whenever products are made, delivered, moved or deleted. |
| `TODO.md` | Gitignored scratch notes. |

Pipeline outputs are not in the repository. They live in
`/mnt/md124/thomasli/selfcal/outputs/<field>/{ref.fits, reprojected/, calibration/, mosaic/, records/, logs/}`, and
`/data3` is a symlink to `/mnt/md124`. Never `git add` a `cal_*.h5` or any other product.

## Common commands

```bash
pip install -e ".[docs,dev]"                                        # editable install with the docs and test tools
./selfcal_scripts/run.sh selfcal_scripts/runs/<name>.py [args]      # a run (or selfcal_scripts/launch/<name>.sh)
./selfcal_scripts/run.sh selfcal_scripts/runs/<name>.py --dry-run   # its plan: products made / reused / refused
selfcal plan|run|adopt|rerun|compare|convert ...                    # the CLI (plan/adopt read FIELD(S), RECIPE, RUN)
pytest                                                              # the test suite
selfcal_scripts/gates/run_gates.sh <tag>                            # pytest + the default gates, once before a merge
DISABLE_MKDOCS_2_WARNING=true mkdocs build --strict -d <scratch>    # the docs build CI runs (site/ is not ignored)
```

Machine settings (`sc.Compute`: workers, staging) never change a product's bytes. The `sc.Numerics()` defaults
(48 threads, batch 50) are the 2026-05 tuning, and a recipe can set its own: the fiducial uses
`Numerics(30, batch=40)`. Tuning notes are in PIPELINE.md.

## The Python API, in short

A run is a Python script: `sc.Field(path, instrument, pixel_scale, compute=sc.Compute(scratch, workers=))` and
its actions: `field.reproject(...)`, `field.plan(recipe, jobs=)`, `field.calibrate(recipe, jobs=, tiles=, passes=,
frames=)`, `field.mosaic(recipe, jobs=)` and `field.submit(recipe, jobs=)` (detached). `jobs=` is required by every action
but `reproject`.
- **Recipes:** `sc.Recipe(model, fit=sc.Fit(), coadd=sc.Coadd(), numerics=sc.Numerics(), name=<suffix>)`.
- **Models:** `sc.continuum()`, `sc.spectral(lines, polynomial=sc.Poly(...))`, or
  `sc.Model(sky=[sc.Sky(...)], offsets=[sc.Offsets(...)], chunk_maps={name: sc.ChunkMap(fn, tag=, **params)})`.
  The chunk maps are the model's (the first is primary: its tag joins the product names); a model without them gets the
  instrument's `default_chunk_maps()`.
- **Instruments:** `sc.SPHEREx(d)` (the detector only, no default maps; its maps from `selfcal_scripts.recipes.spherex.maps`,
  its jobs from `selfcal_scripts.spherex.jobs`: `jobs.channel(17)`, `jobs.channels(1, 17)`, `jobs.window(name, range(lo, hi))`),
  `sc.Euclid()` (`jobs=sc.Job(band)`), `sc.Camera(shape, chunks=)` (`jobs=sc.Job('All')`).
- **Jobs:** `sc.Job(name, window={axis: range}, strict={axis: range})`, a box in the primary map's chunk axes.

The guide is `docs/guide/python-api.md`. Every change keeps these contracts:

- **Work goes under `if __name__ == "__main__":`, and functions must be importable** (module-level `def`), because
  forkserver workers re-import the script. Notebook cells use `sc.by_value(fn)`.
- **Every action plans first and pins the BLAS threads.** `run.sh` and `selfcal run` set
  `OMP/MKL/OPENBLAS_NUM_THREADS=1` before numpy loads. Every action writes `<field>/records/*.json` and one log per
  process.
- **Products are written atomically, each with a sidecar `<product>.json`** (its inputs and fingerprint,
  `selfcal/run/products.py`). A product is reused only when current, and refused when made by other inputs or
  unrecorded; `field.adopt` / `selfcal adopt` records TOML-made ones. `overwrite=True` remakes it.
- **Settings are frozen dataclasses checked when built** (`selfcal.config.base.Config`). A new field on a fingerprinted
  class is declared `added(default, since=...)`, so existing products keep their fingerprints
  (`tests/test_fingerprints.py`, `docs/developer/settings.md`).
- **A new calibration variant is an `sc.Model`, and a new telescope an `sc.Camera(...)` or an `sc.Instrument`
  subclass.** A variant can also be a function returning a model, like `sc.continuum`, `sc.spectral` or
  `sc.two_block`. Neither touches the engine, and nothing is registered.
  - Instrument specifics live in `selfcal/instruments/<name>/` (SPHEREx: `spherex/settings.py`); the SPHEREx chunk
    maps, jobs and terms are the model side, in `selfcal_scripts/spherex/`.
  - The engine reaches the instrument only through `sc.Instrument` (`selfcal/instruments/contract.py`).
- **The library, the gates and the tests never read or write `workspace/`.**
- **TOML run configs are gone** (2026-10-07). `selfcal convert x.toml` writes a run script, unchecked. What was removed
  with TOML is listed in `docs/guide/migrating-from-toml.md` and `workspace/TOML_TO_PYTHON.md`.

## Recipes and products

- **Recipes:** `selfcal_scripts/recipes/spherex.py` holds every campaign's recipe. Read the current one there, because
  it changes with the work. The fiducial channel maps are `CHANNEL_MAPS` (`SLOPE_GAUGE` plus the wavelength maps) on
  17 channels x 1 column. Production runs 17x1 or 34x1, as the recipe says.
- **Names:** `cal_Detector{D}_NumSub10_NumCh{17|34}_NumCol{C}_<job>_<recipe name>.h5`, and `mosaic_<same>.fits`.
- **SPHEREx chunk ids:** on the subchannel map (`selfcal_scripts/spherex/chunks.py`), `subchannel * num_col + column`,
  with `num_sub * num_ch + 2` subchannels. The first and the last lie beyond the band's edges (`make_stripped_chunk_map`).
  The `NumSub/NumCh/NumCol` part of the names is that map's tag.
- **Reproducibility:** every product is byte-reproducible run to run. The moment folds are deterministic since
  2026-09-28, and the coadd depends only on the frames and the batch sizes.

## Regression

- **The test suite:** `tests/`. CI (`.github/workflows/ci.yml`) runs it on Python 3.11 and 3.12, with the strict docs
  build; ruff and mypy run advisory.
- **The gates:** `run_gates.sh <tag>` runs pytest and the default byte-equality gates against the float64 goldens
  (`selfcal_scripts/gates/README.md`).
  - A change to code that no gate covers adds a short gate.
  - Goldens are remade only for an intended numerical change, with `make_goldens.sh` from a committed tree.

## Trees and environments

- `~/selfcal-project/selfcal/`: the main worktree, branch `main`, env `selfcal` (which has pytest and the docs tools).
  Feature branches are made here.
- `~/selfcal-project/selfcal-stable/`: branch `stable`, env `selfcal-stable` (the analysis env). It is advanced only
  deliberately.
- The other worktrees (`git worktree list`) hold long-lived branches, and all share `.git`. The env `selfcal-memopt`
  imports the `selfcal-memopt` tree.

## Quirks

- `find_outliers` (`selfcal/geometry/map_helper.py`) warns `All-NaN slice encountered` for fully masked subframes. It
  is harmless.
