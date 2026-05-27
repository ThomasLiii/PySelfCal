# zodi_anchor scripts

Post-hoc zodiacal-light anchor for SelfCal cal files + mosaics.

The core library lives in [`SelfCal/ZodiAnchor.py`](../../SelfCal/ZodiAnchor.py).
The scripts here orchestrate building per-frame zodi predictions
(via [zodipy](https://github.com/Cosmoglobe/zodipy)), fitting the per-channel
anchor, and producing diagnostic plots.

## Architecture

The anchor is **non-mutating**. SelfCal pipeline outputs (`cal_*.h5`,
`mosaic_*.fits`) stay **pristine**. The fit result is written to a
per-detector **anchor file**, `<run>/zodi_anchor/anchor_D{N}.h5`
(summary-only schema, `ANCHOR_VERSION` in `SelfCal/ZodiAnchor.py`), and
applied to arrays **at read time** by the consumer (`load_anchor` /
`Anchor`). Nothing on disk is rewritten.

Why: an earlier version edited cal/mosaic in place (`skymap += C`,
`frame_scalar -= C`, mosaic `+C`), which conflated calibration with zodi
attribution and made re-runs/analysis confusing. Keeping the fit in a
separate anchor file makes re-anchoring (different clip params,
smoothing/repair variants) a seconds-long rebuild instead of a multi-file
re-edit.

File-name roles:
- `build_predictions*` — produce zodi-prediction `.npz` files (the
  expensive zodipy step; one per cal file or one per channel). Cached in
  `zodi_preds/`.
- `build_anchor.py` — fit the per-channel anchor from pristine cal +
  `zodi_pred_*.npz`, write `anchor_D{N}.h5`. No cal/mosaic mutation.
  (The `run_cal_v2.py` driver does the same per channel inline via
  `append_anchor_channel`.)
- `diag_*` — read-only diagnostics; read the anchor file (+ cal/npz as
  needed). Never modify cal/mosaic files.
- `revert_anchor.py` — historical migration: undo a legacy in-place
  anchor on cal+mosaic (symmetric inverse), returning them to pristine
  state.

## Typical workflow

```
        ┌─────────────────────────────────┐
        │ build_predictions[_all_channels]│  (zodipy; selfcal-zodipy env)
        └──┬──────────────────────────────┘
           │   produces zodi_preds/zodi_pred_*.npz  (cache + env hand-off)
           v
        ┌────────────────────────┐
        │ build_anchor.py        │  (cheap linear fit; selfcal env)
        └──┬─────────────────────┘
           │   writes <run>/zodi_anchor/anchor_D{N}.h5   (PRISTINE cal/mosaic)
           v
        ┌──────────────────────────────────────────────┐
        │ diag_*.py        (read-only; apply at runtime)│
        │ load_anchor()    (consumer for downstream use)│
        └──────────────────────────────────────────────┘
```

A one-time `revert_anchor.py` pass undid the old in-place anchoring on
D1/D4/D5 before this layout was adopted.

## Scripts

### Builders

| Script | Purpose |
|---|---|
| [`build_predictions.py`](build_predictions.py) | Build per-frame zodi predictions for ONE cal file. Writes `zodi_pred_<tag>.npz` with arrays `zodi_pred`, `mjds`, `reproj_list`. Core module — exports `DEFAULT_CALIBRATION_DIR`, `DEFAULT_METADATA_CACHE_TEMPLATE`, `extract_metadata_for_reproj_list`, used by the other builders. |
| [`build_predictions_all_channels.py`](build_predictions_all_channels.py) | Build predictions for all 34 channels of a detector in one go, without needing a per-channel cal file (reads exposure + LVF metadata directly). Faster than running `build_predictions.py` 34 times because the WCS/MJD extraction is shared. |

### Anchor builder + repair

| Script | Purpose |
|---|---|
| [`build_anchor.py`](build_anchor.py) | Fit the per-channel anchor from a run's PRISTINE cals + `zodi_preds/*.npz` via `SelfCal.ZodiAnchor.fit_anchor_for_channel`; write `<run>/zodi_anchor/anchor_D{N}.h5`. No cal/mosaic mutation. Skips any channel whose cal still carries a legacy in-place anchor (run `revert_anchor.py` first). `--run-dir <run> [<run> ...]`. |
| [`repair_anchor.py`](repair_anchor.py) | Phase-1 repair. Flags channels with `pearson_r < --r-threshold` (default 0.9; lower to 0.5 for only the hard blowouts like D1 He I Ch31-33). For flagged channels it **smooths only the slope** (Pearson-r-weighted spline fit to the **clean channels only**, so a blown channel can't leak in) and then **recomputes C** as `mean_full_dc - slope_final*mean_pred` — C is NOT smoothed, so the non-zodi signal (He glow, PAH, DGL) it carries is preserved; only the slope's per-channel bias is removed. Clean channels keep their raw fit. At 0.9 this also de-biases moderate-r channels (D1 OI Ch10-12, D4 PAH Ch23-24) while keeping their C features. Updates the anchor file **in-place**; raw `slope`/`intercept` are preserved (re-runnable). Warns when a flagged channel is outside the clean wavelength span (spline extrapolation, e.g. D1 Ch30-34 at 0.9). `--dry-run`/`--plot` to inspect first; `--s-factor` tunes slope smoothing. Core: `SelfCal.ZodiAnchor.rweighted_spline_repair`. |

### Migration

| Script | Purpose |
|---|---|
| [`revert_anchor.py`](revert_anchor.py) | Undo a legacy in-place anchor (the pre-anchor-file era): skymap `-= C` on covered, `frame_scalar += C`, mosaic `MEAN_MAP`/`SC_MEAN_MAP -= C` on weighted; drops `zodi_anchor_*` attrs + `zodi_anchor_pred` dataset + `ZODIANCH*` headers; stamps `ZODIRVRT`. Idempotent. Dry-run by default; `--apply` to mutate. Round-trip-verified bit-exact. Historical — kept in case legacy files resurface. |

### Diagnostics (read-only)

| Script | Purpose |
|---|---|
| [`diag_zodi_spectrum.py`](diag_zodi_spectrum.py) | Per-detector 4-panel spectrum (mean(full_DC)/mean(zodi_pred)/slope·mean(zodi_pred), C, slope, Pearson r vs wavelength) read **entirely from the anchor file** — instant, no cal/npz I/O. `--anchor` or `--run-dir`; `--max-ch` drops airglow-blown channels. |
| [`diag_plot_cross_channel.py`](diag_plot_cross_channel.py) | Cross-channel continuity: loads pristine cals, applies the anchor **in-memory** from the anchor file, plots per-chunk continuity across the LVF boundaries. `--run-dir` (auto-locates the anchor file) or `--cal-glob` + `--anchor`. |
| [`diag_compare_zodi_vs_scalar.py`](diag_compare_zodi_vs_scalar.py) | Per-channel scatter: `zodi_pred` vs the cal's recovered `full_DC` (frame_scalar + chunk leakage). Re-fits from the pristine cal + npz (matches the stored anchor fit). Sanity-check that the linear fit makes sense. |
| [`diag_compare_models.py`](diag_compare_models.py) | Run multiple zodi IPD models against the same cal files; side-by-side plot + `compare_models_summary.json` of per-model slope/intercept/r per channel. |

## Output locations (not in this directory)

| Output | Where it goes |
|---|---|
| Metadata cache (`metadata_D{N}.h5`) | `<repo>/cache/zodi_anchor/` (gitignored). Path is set by `DEFAULT_METADATA_CACHE_TEMPLATE` in `build_predictions.py`. |
| Diagnostic figures | `<repo>/figures/zodi_anchor/...` (gitignored). All scripts that emit PNGs accept an `--out-dir`. |
| `zodi_pred_<tag>.npz` files | `<run>/zodi_preds/`. The expensive zodipy output; cache + cross-env hand-off (zodipy needs the `selfcal-zodipy`/numpy<2 env; the fit + consumer run in `selfcal`). Referenced by the anchor file (path + sha + len), not copied into it. |
| Anchor file (`anchor_D{N}.h5`) | `<run>/zodi_anchor/`. The fit result (summary-only, one file per detector). Cal+mosaic stay pristine. |
| Anchored cal+mosaic | **Not written.** The shift is applied in-memory by `load_anchor()` consumers. For a materialized FITS (ds9 / publication), a separate opt-in step would write to `<run>/anchored_mosaics/` — never overwriting pipeline outputs. |

## Notes

- The anchor file is **summary-only**; per-frame `full_DC` is recomputed
  from the pristine cal and per-frame `zodi_pred` is read from the
  referenced npz. It stores the npz path + sha1 + length so a consumer can
  re-load and verify identity.
- The fit result is repair-aware: `slope`/`intercept` are the raw
  per-channel linfit; `slope_final`/`C_final` are what consumers apply.
  They equal the raw values until `repair_anchor.py` overwrites the
  flagged channels (`contaminated_flag`, `repair_method='rweighted_spline'`,
  root `anchor_method='rweighted_spline'`). Re-running the repair always
  recomputes from the preserved raw values, so it's safe to re-run with a
  different `--r-threshold` / `--s-factor`.
- Imports between scripts: the `build_predictions*` and
  `diag_compare_models.py` scripts `from build_predictions import ...` for
  shared constants and the `extract_metadata_for_reproj_list` helper (each
  inserts its dir into `sys.path[0]`, so cwd doesn't matter).
  `build_anchor.py` and `diag_*.py` import the anchor core from
  `SelfCal.ZodiAnchor` (resolved via `pip install -e .`).
- The actual anchor math lives in `SelfCal.ZodiAnchor`
  (`fit_anchor_for_channel`, `write_anchor` / `append_anchor_channel`,
  `Anchor`/`load_anchor`). These scripts only orchestrate I/O and CLI
  plumbing.
