# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo overview

Two coupled efforts share this tree:

1. **`SelfCal/` — the pipeline.** Sparse-LSQR self-calibration + mosaicking for SPHEREx (with Euclid helpers). Two reference docs:
   - [SelfCal/README.md](SelfCal/README.md) — module-level **code architecture**: the data flow `FITS → reprojected/*.h5 → cal_*.h5 → mosaic_*.fits`, what each submodule does, and design decisions (multi-chunk-map LSQR build, shared-memory worker hand-off, parallel SpMV `LinearOperator`, tight-bbox cache crops, zero-column elimination, etc.). Read this before touching anything in `SelfCal/`.
   - [PIPELINE.md](PIPELINE.md) — **operational runbook**: tuning knobs (`NumCol`, `reg_weights`, `damp_weight`, adjacency choice, polynomial constraints), advanced solve modes (`det_groups_list`, `det_templates`, `mean_offsets_list`, `use_per_frame_scalar`), the NVMe staging pattern, and the on-disk schemas of `cal_*.h5`, reprojected `*.h5`, and mosaic `*.fits`. Read this when running the pipeline or working with its outputs.
2. **`analysis/analysis_script/` — diagnostics on the calibration outputs.** Loads the per-frame, per-chunk offset terms from `cal_*.h5` files via `zodi_utils.load_cal_offsets` (handles both legacy single-map and multi-chunk-map schemas) and characterizes the temporal/spatial structure (annual sine fits, residual scans, detector-fixed pattern extraction, raw-stack verification, etc.). This is where most ad-hoc plotting and analysis lives.

The pipeline emits `cal_Detector{D}_NumSub10_NumCh34_NumCol{C}_Ch{ch}{suffix}.h5` files; the analysis scripts consume them.

## Layout

| Dir | What's there |
| --- | --- |
| `SelfCal/` | Library code (PipelineWrapper, lsqr, coadd, subframe, solution, SPHERExUtility, …). See [SelfCal/README.md](SelfCal/README.md). |
| `selfcal_scripts/` | Top-level drivers only: `run_reproject.py`, `run_cal_v2.py`, `run_cal_baseline_test.py` (regression harness with `TEST_VARIANTS`), `precompute_lvf_params.py`, `diff_cal_h5.py` (schema-aware cal-file diff). Cached LVF params in `lvf_params/lvf_params_D{1..6}.npy`. Tuning + schema docs in [PIPELINE.md](PIPELINE.md). |
| `selfcal_scripts/benchmarks/` | Phase-level wall/RSS/IO benchmark harness: `benchmark_d3_ch17_{poly,numcol3,tuned,mid}.py`. `..._{numcol3,mid,tuned}` import `PhaseTracker` from `..._poly`; keep the 4 together. |
| `selfcal_scripts/experiments/` | Non-mainline experiments: `run_cal_v2_k2_readout.py` (D5 K=2 readout-stripe variant). |
| `selfcal_scripts/cc_scripts/` | Cross-channel analysis/plotting one-offs (`plot_d5_ch3_*`, `run_mosaic_polyconstraint.py`). |
| `analysis/analysis_script/` | Canonical analysis: `meeting_plots.py`, `verify_stack.py` (deliverables) plus their cache builders `build_multichannel_cache.py`, `build_perchunk_coords.py` and the shared `zodi_utils.py`. Historical one-offs live in `analysis/analysis_script/archive/`. |
| `notebooks/` | `spherex_selfcal_demo.ipynb` (working demo) and `euclid_mosaic.ipynb`. Older demos in `notebooks/archive/`. |
| `figures/` | Gitignored. Per-channel analysis plots, benchmark timelines (`figures/benchmark/`), etc. |
| `archive/` | Gitignored top-level archive. Holds stale top-level `analysis/` notebooks/lists (`analysis.ipynb`, `aliph_arom.ipynb`, `offset_analysis.ipynb`, `euclid_mosaic_copy.ipynb`, `reproj_list_nep*.txt`, `fit_params_with_scatter_n*.json`) plus pre-split pipeline scripts. Ignore unless asked. |

Pipeline outputs are *not* in this repo — they live under `/mnt/md124/thomasli/selfcal/outputs/{run_name}/calibration/` (path is hard-coded in `analysis/analysis_script/zodi_utils.py:CAL_OUTPUT_BASE`).

## Common commands

Install editable: `pip install -e .` (uses `pyproject.toml`; package is just `SelfCal`).

Pipeline runs are normally launched via `selfcal_scripts/run_cal_v2.py` (calibration + mosaic) and `selfcal_scripts/run_reproject.py` (reprojection). Both prepend the repo root to `sys.path` and import `SelfCal.PipelineWrapper`. Edit the `frame_setting` / config block at the top, then run with `python selfcal_scripts/run_cal_v2.py`. They pin `OMP/MKL/OPENBLAS_NUM_THREADS=1` so the in-process LSQR threadpool is the only parallelism — preserve that when adding launchers.

**Production defaults (tuned 2026-05)**: `max_workers=48`, `batch_size=50` (cal), `cache_batch_size=50`, `coadd_batch_size=50` (mosaic), `n_threads=48` (apply_lsqr). On a 192-physical / 384-logical-core box, this is ~15% faster than the prior 32/20/30 defaults; the sweet spot is below the physical-core count because each `setup_lsqr` / `compute_coadd_map` call spawns a fresh `ProcessPoolExecutor`, and at very large worker counts the per-pool spawn time eats the parallelism gain. Full sweep + analysis: `figures/benchmark/d3_ch17_tuning_sweep.md` (gitignored).

Analysis runs, all from `analysis/analysis_script/`:

```bash
# One-off per detector D ∈ {1,2,3,4,5}, in dependency order:
python build_multichannel_cache.py --detector D     # ~3 min, FITS header reads, writes cache/multichannel_det{D}.pkl
python build_perchunk_coords.py --detector D        # ~3 min, WCS + ecliptic geometry per (exp, chunk)
python meeting_plots.py --detector D                # ~50 s, fig1/2/3 + cache/meeting_det{D}_per_chunk_fits.npz
python verify_stack.py --detector D --n-frames N    # ~5 min for full stack; --reuse-cache to re-render

# Re-running the figure with a cached stack/fits costs seconds; the heavy
# step is only the FITS reads.
```

Archived analysis scripts (in `analysis/analysis_script/archive/`) all do `from zodi_utils import ...`. To run one, prepend `PYTHONPATH` so Python finds the shared `zodi_utils.py` from its new directory:
```bash
PYTHONPATH=analysis/analysis_script python analysis/analysis_script/archive/<script>.py
```

There is no test suite. There is no linter configured. Regression for the pipeline is via `selfcal_scripts/run_cal_baseline_test.py` (produces named `cal_*.h5` files per `TEST_TAG` / variant) + `selfcal_scripts/diff_cal_h5.py` (element-wise dataset diff, schema-aware).

For SelfCal-pipeline tuning, advanced solve modes, the NVMe staging pattern, and the on-disk schemas of the cal / reprojected / mosaic files, see [PIPELINE.md](PIPELINE.md). For module-level code architecture, see [SelfCal/README.md](SelfCal/README.md).

## How `analysis/analysis_script/` is wired

All analysis scripts share `zodi_utils.py`:

- `cal_path(detector, channel)` → the `cal_*.h5` calibration file (lives off-tree under `/mnt/md124/...`).
- `load_cal_offsets(path_or_file)` → `{m: offset_m}` dict; handles both legacy top-level `offset` and new `offsets/map_m` schemas, folds `frame_scalar` into map 0 for legacy-compatible subtraction semantics.
- `data_path(name)` → `analysis/analysis_script/cache/<name>` for `.pkl` / `.npz` caches.
- `fig_path(name)` → `analysis/analysis_script/figures/<name>` for plots.
- `sine_model`, `fit_sine` (frequency fixed at `1/SIDEREAL_YEAR_DAYS`), `compute_ecliptic_geometry`.

Per-detector cache files used by most scripts:

- `cache/multichannel_det{D}.pkl` — long-format DataFrame, one row per (exposure, channel), with MJD, ecliptic/sun geometry, per-column means.
- `cache/perchunk_coords_det{D}.npz` — `(n_exp, n_chunks)` arrays of `helio_lon`, `elongation`, `mjd`. Optional but enables per-chunk residual scans in `meeting_plots.py` fig 2; the script falls back to per-exposure central pointing if missing.
- `cache/detector_templates_det{D}.pkl` — per-channel `(342, 3)` mean-removed offset templates.
- `cache/meeting_det{D}_per_chunk_fits.npz` — output of `meeting_plots.py`; `verify_stack.py` reads its `D` array to set the diverging-colour vmax in panel (b) so the two scripts use the same scale.

`meeting_plots.py` and `verify_stack.py` are the canonical "deliverable" scripts for the current meeting set. Older one-off analyses (`task1`–`task4`, `plot_grad_*`, `plot_clean_spatial.py`, `plot_numcol_*`, `meeting_plots_by_lat.py`, `analyze_zodi_spatial.py`, `build_metadata.py`, `plot_chunkmap.py`) live in `analysis/analysis_script/archive/` — see the `PYTHONPATH` note above for re-running them.

## Key conventions inside the analysis code

- **Chunk indexing.** With `NUM_SUB=10`, `NUM_CH=34`, `NUM_COL=3`, `TOT_SUB = 10*34 + 2 = 342` (the +2 are the above-first / below-last padding subchannels). Chunk id = `subchannel * NUM_COL + column`, total `342 * 3 = 1026` chunks per detector. (Production now also runs at `NumCol=10` for higher spatial resolution — total `342 * 10 = 3420` chunks.)
- **Padded vs unpadded subchannels.** `make_stripped_chunk_valid_mask(..., subchannel_padding=0)` returns the strict per-channel valid set; `padding=1` extends it by one subchannel each side so adjacent channels overlap (used for stitching in `meeting_plots.py`).
- **Smooth chunk boundaries.** The chunk map is built from circular arcs `y = -sqrt(R² − (x − xc)²) + yc` with `(xc, yc)` and `R[i]` from `lvf_params` (`r_edges` returned by `make_stripped_chunk_map`). Plot scripts that overlay chunk geometry (`plot_chunkmap.py`) draw these analytic arcs rather than per-pixel jagged edges.
- **Reverse-rendering chunk values to the detector grid.** `make_spherex_stripped_offset_map` runs a mean-preserving 2-D spline (`x_degree=3`, `y_degree=3`) over `(r_edges, x_edges)`. Both `meeting_plots.py` (Fig 3) and `verify_stack.py` (chunkbin-vs-Fig3b comparison) call it with the *same* arguments, so values rendered on the pixel grid are directly comparable.

## Recent analysis state (current `meeting_plots.py` / `verify_stack.py`)

These were rewritten in 2026-04 / 2026-05; old commit history will not match. Conventions to preserve when modifying:

- **`meeting_plots.py` uses one global sine phase.** `find_global_phi` does a 1-D minimization of summed OLS SSR across every (channel, sub, col) chunk, then `fit_chunks_with_phi` does a per-chunk linear fit `y = A · sin(2π f t + φ_global) + C` with `A` *signed* (do not flip the sign — that would reintroduce per-chunk phase). `f` is fixed at `1/SIDEREAL_YEAR_DAYS`.
- **`verify_stack.py` uses per-subchannel median, no smoothing.** The previous version Gaussian-smoothed the spectrum before subtraction; that was intentionally removed. The two-panel output is `(a) raw stack, (b) stack − per-subchannel-median spectrum` with vmax matched to Fig 3b. There is also a 3-panel sanity-check output `*_chunkbin_vs_fig3b.png` that bins panel (b) per chunk, re-renders through the same mean-preserving spline used for Fig 3b, and prints the RMS difference.
- **`plot_chunkmap.py` zoom is square + uses analytic arcs.** Side-by-side layout, both panels `aspect='equal'`, dpi=300. The zoom default crosses the Left/Mid column boundary (`x_edges[1] = 680`) so subchannel-arc and column structure are both visible.

## Path / environment quirks worth knowing

- Calibration files live on `/mnt/md124` (RAID). They're large; never `git add` anything from `cal_*.h5`.
- Each analysis script does `sys.path.insert(0, _SELFCAL_ROOT)` so it can import from `SelfCal/` without `pip install`. If an import from `SelfCal` fails, check the script's `_PKG_DIR / _SELFCAL_ROOT` two-up logic.
- `zodi_utils.py` hard-codes the calibration directory layout in `CAL_OUTPUT_BASE / CAL_RUN_TEMPLATE / CAL_FILE_TEMPLATE`. Changing the run name or hyperparameter suffix (`damp0p1_reg0p1_outThresh5_sigma2`) requires updating this template.
- `find_outliers` (`MapHelper.py:79-88`) emits `RuntimeWarning: All-NaN slice encountered` for fully-masked subframes. Harmless; suppress with `warnings.filterwarnings` if it's noisy.
- The cal-file naming pattern in `analysis/analysis_script/zodi_utils.py` bakes in a specific `NumCol` value (`NumCol3` in older runs, `NumCol10` for the poly-constraint era). If you regenerate with a different `NumCol`, update `CAL_FILE_TEMPLATE` to match.
- **Two worktrees**: `~/selfcal-project/selfcal/` (dev, env `selfcal`) tracks `main`; `~/selfcal-project/selfcal-stable/` (analysis, env `selfcal-stable`) tracks `stable`. They share `.git`; advance `stable` deliberately when you want the analysis env to pick up new features.
