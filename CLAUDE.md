# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo overview

Two coupled efforts share this tree:

1. **`SelfCal/` — the pipeline.** Sparse-LSQR self-calibration + mosaicking for SPHEREx (with Euclid helpers). Two reference docs:
   - [SelfCal/README.md](SelfCal/README.md) — module-level **code architecture**: the data flow `FITS → reprojected/*.h5 → cal_*.h5 → mosaic_*.fits`, what each submodule does, and design decisions (shared-memory worker hand-off, parallel SpMV `LinearOperator`, tight-bbox cache crops, zero-column elimination, etc.). Read this before touching anything in `SelfCal/`.
   - [PIPELINE.md](PIPELINE.md) — **operational runbook**: tuning knobs (`NumCol`, `reg_weight`, `damp_weight`, adjacency choice), advanced solve modes (`det_groups`, `det_template`, `mean_offsets`), the NVMe staging pattern, and the on-disk schemas of `cal_*.h5`, reprojected `*.h5`, and mosaic `*.fits`. Read this when running the pipeline or working with its outputs.
2. **`analysis/analysis_script/` — diagnostics on the calibration outputs.** Loads the per-frame, per-chunk offset terms from `cal_*.h5` files and characterizes the temporal/spatial structure (annual sine fits, residual scans, detector-fixed pattern extraction, raw-stack verification, etc.). This is where most ad-hoc plotting and analysis lives.

The pipeline emits `cal_Detector{D}_NumSub10_NumCh34_NumCol3_Ch{ch}_*.h5` files; the analysis scripts consume them.

## Layout

| Dir | What's there |
| --- | --- |
| `SelfCal/` | Library code (PipelineWrapper, lsqr, coadd, SPHERExUtility, …). See [SelfCal/README.md](SelfCal/README.md). |
| `selfcal_scripts/` | Drivers: `run_reproject.py`, `run_cal_v2.py`, `precompute_lvf_params.py`. Cached LVF params land in `selfcal_scripts/lvf_params/lvf_params_D{D}.npy`. Tuning + schema docs in [PIPELINE.md](PIPELINE.md). |
| `analysis/analysis_script/` | Analysis scripts + caches + figures (see below). |
| `notebooks/` | Demos / one-offs. `spherex_selfcal_demo.ipynb` is the working demo. |
| `archive/` | Older analysis kept for reference; ignore unless asked. |

Pipeline outputs are *not* in this repo — they live under `/mnt/md124/thomasli/selfcal/outputs/SPHEREx_nep_qr2_det{D}_6p2arcsec/calibration/` (path is hard-coded in `analysis/analysis_script/zodi_utils.py:CAL_OUTPUT_BASE`).

## Common commands

Install editable: `pip install -e .` (uses `pyproject.toml`; package is just `SelfCal`).

Pipeline runs are normally launched via `selfcal_scripts/run_cal_v2.py` (calibration) and `selfcal_scripts/run_reproject.py` (reprojection). Both prepend the repo root to `sys.path` and import `SelfCal.PipelineWrapper`. Edit the `frame_setting` / config block at the top, then run with `python selfcal_scripts/run_cal_v2.py`. They pin `OMP/MKL/OPENBLAS_NUM_THREADS=1` so the in-process LSQR threadpool is the only parallelism — preserve that when adding launchers.

Analysis runs, all from `analysis/analysis_script/`:

```bash
# One-off per detector D ∈ {1,2,3,4,5}, in dependency order:
python build_multichannel_cache.py --detector D     # ~3 min, FITS header reads, writes cache/multichannel_det{D}.pkl
python build_perchunk_coords.py --detector D        # ~3 min, WCS + ecliptic geometry per (exp, chunk)
python meeting_plots.py --detector D                # ~50 s, fig1/2/3 + cache/meeting_det{D}_per_chunk_fits.npz
python verify_stack.py --detector D --n-frames N    # ~5 min for full stack; --reuse-cache to re-render
python plot_chunkmap.py --detector D                # static; chunkmap_det{D}.png

# Re-running the figure with a cached stack/fits costs seconds; the heavy
# step is only the FITS reads.
```

There is no test suite. There is no linter configured.

For SelfCal-pipeline tuning, advanced solve modes, the NVMe staging pattern, and the on-disk schemas of the cal / reprojected / mosaic files, see [PIPELINE.md](PIPELINE.md). For module-level code architecture, see [SelfCal/README.md](SelfCal/README.md).

## How `analysis/analysis_script/` is wired

All analysis scripts share `zodi_utils.py`:

- `cal_path(detector, channel)` → the `cal_*.h5` calibration file (lives off-tree under `/mnt/md124/...`).
- `data_path(name)` → `analysis/analysis_script/cache/<name>` for `.pkl` / `.npz` caches.
- `fig_path(name)` → `analysis/analysis_script/figures/<name>` for plots.
- `sine_model`, `fit_sine` (frequency fixed at `1/SIDEREAL_YEAR_DAYS`), `compute_ecliptic_geometry`.

Per-detector cache files used by most scripts:

- `cache/multichannel_det{D}.pkl` — long-format DataFrame, one row per (exposure, channel), with MJD, ecliptic/sun geometry, per-column means.
- `cache/perchunk_coords_det{D}.npz` — `(n_exp, n_chunks)` arrays of `helio_lon`, `elongation`, `mjd`. Optional but enables per-chunk residual scans in `meeting_plots.py` fig 2; the script falls back to per-exposure central pointing if missing.
- `cache/detector_templates_det{D}.pkl` — per-channel `(342, 3)` mean-removed offset templates.
- `cache/meeting_det{D}_per_chunk_fits.npz` — output of `meeting_plots.py`; `verify_stack.py` reads its `D` array to set the diverging-colour vmax in panel (b) so the two scripts use the same scale.

Scripts in this directory can be grouped by which cache they consume. `meeting_plots.py` and `verify_stack.py` are the canonical "deliverable" scripts for the current meeting set; the `task1`–`task4`, `plot_grad_*`, `plot_clean_spatial.py`, etc. are older one-off analyses.

## Key conventions inside the analysis code

- **Chunk indexing.** With `NUM_SUB=10`, `NUM_CH=34`, `NUM_COL=3`, `TOT_SUB = 10*34 + 2 = 342` (the +2 are the above-first / below-last padding subchannels). Chunk id = `subchannel * NUM_COL + column`, total `342 * 3 = 1026` chunks per detector.
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
- The cal-file naming pattern in `analysis/analysis_script/zodi_utils.py` still bakes in `NumCol3` from earlier production runs; if you regenerate with a different `NumCol`, update `CAL_FILE_TEMPLATE` to match.
