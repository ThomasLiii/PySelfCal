# zodi_anchor scripts

Post-hoc zodiacal-light anchor for SelfCal cal files + mosaics.

The core library lives in [`SelfCal/ZodiAnchor.py`](../../SelfCal/ZodiAnchor.py).
The scripts here orchestrate building per-frame zodi predictions
(via [zodipy](https://github.com/Cosmoglobe/zodipy)), fitting the per-channel
anchor, and producing diagnostic plots.

## Sidecar architecture (current)

The anchor is a **non-mutating** post-processing layer. SelfCal pipeline
outputs (`cal_*.h5`, `mosaic_*.fits`) stay **pristine**. The anchor result
is written to a per-detector **sidecar**, `<run>/zodi_anchor/anchor_D{N}.h5`
(summary-only schema, `SIDECAR_VERSION` in `SelfCal/ZodiAnchor.py`), and
applied to arrays **at read time** by the consumer (`load_anchor` /
`Anchor`). Nothing on disk is rewritten.

Why: the old in-place anchor edited cal/mosaic (`skymap += C`,
`frame_scalar -= C`, mosaic `+C`), which conflated calibration with zodi
attribution and made re-runs/analysis confusing. The sidecar split makes
re-anchoring (different clip params, smoothing/repair variants) a
seconds-long sidecar rebuild instead of a multi-file re-edit.

File-name roles:
- `build_*` — produce zodi-prediction `.npz` files (the expensive zodipy
  step; one per cal file or one per channel). Cached in `zodi_preds/`.
- `build_sidecar.py` — fit the per-channel anchor from pristine cal +
  `zodi_pred_*.npz`, write `anchor_D{N}.h5`. No cal/mosaic mutation.
- `diag_*` — read-only diagnostics; consume the sidecar (+ cal/npz as
  needed). Never modify cal/mosaic files.
- `revert_anchor.py` — one-shot migration: undo any legacy in-place anchor
  on cal+mosaic (symmetric inverse), returning them to pristine state.
- `apply_anchor*.py`, `pipeline_multi_channel.py` — **DEPRECATED** (legacy
  in-place mutators; kept only for reference / reading old outputs).

## Typical workflow

```
        ┌─────────────────────────────────┐
        │ build_predictions[_all_channels]│  (zodipy; selfcal-zodipy env)
        └──┬──────────────────────────────┘
           │   produces zodi_preds/zodi_pred_*.npz  (cache + env hand-off)
           v
        ┌────────────────────────┐
        │ build_sidecar.py       │  (cheap linear fit; selfcal env)
        └──┬─────────────────────┘
           │   writes <run>/zodi_anchor/anchor_D{N}.h5   (PRISTINE cal/mosaic)
           v
        ┌──────────────────────────────────────────────┐
        │ diag_*.py        (read-only; apply at runtime)│
        │ load_anchor()    (consumer for downstream use)│
        └──────────────────────────────────────────────┘
```

A one-time migration (`revert_anchor.py`) was used to undo the old
in-place anchoring on D1/D4/D5 before adopting this layout.

## Scripts

### Builders

| Script | Purpose |
|---|---|
| [`build_predictions.py`](build_predictions.py) | Build per-frame zodi predictions for ONE cal file. Writes `zodi_pred_<tag>.npz` with arrays `zodi_pred`, `mjds`, `reproj_list`. Core module — exports `DEFAULT_CALIBRATION_DIR`, `DEFAULT_METADATA_CACHE_TEMPLATE`, `extract_metadata_for_reproj_list`, used by the other builders + pipeline. |
| [`build_predictions_all_channels.py`](build_predictions_all_channels.py) | Build predictions for all 34 channels of a detector in one go, without needing a per-channel cal file (reads exposure + LVF metadata directly). Faster than running `build_predictions.py` 34 times because the WCS/MJD extraction is shared. |

### Sidecar builder

| Script | Purpose |
|---|---|
| [`build_sidecar.py`](build_sidecar.py) | Fit the per-channel anchor from a run's PRISTINE cals + `zodi_preds/*.npz` via `SelfCal.ZodiAnchor.fit_anchor_for_channel`; write `<run>/zodi_anchor/anchor_D{N}.h5`. No cal/mosaic mutation. Skips any channel whose cal still carries a legacy in-place anchor (run `revert_anchor.py` first). `--run-dir <run> [<run> ...]`. |

### Migration

| Script | Purpose |
|---|---|
| [`revert_anchor.py`](revert_anchor.py) | Undo a legacy in-place anchor: symmetric inverse of the old `_shift_cal_file`/`_shift_mosaic_file` (skymap `-= C` on covered, `frame_scalar += C`, mosaic `MEAN_MAP`/`SC_MEAN_MAP -= C` on weighted; drops `zodi_anchor_*` attrs + `zodi_anchor_pred` dataset + `ZODIANCH*` headers; stamps `ZODIRVRT`). Idempotent. Dry-run by default; `--apply` to mutate. Round-trip-verified bit-exact. |

### Diagnostics (read-only)

| Script | Purpose |
|---|---|
| [`diag_zodi_spectrum.py`](diag_zodi_spectrum.py) | Per-detector 4-panel spectrum (mean(full_DC)/mean(zodi_pred)/slope·mean(zodi_pred), C, slope, Pearson r vs wavelength) read **entirely from the sidecar** — instant, no cal/npz I/O. `--sidecar` or `--run-dir`; `--max-ch` drops airglow-blown channels. |
| [`diag_plot_cross_channel.py`](diag_plot_cross_channel.py) | Cross-channel continuity: loads pristine cals, applies the anchor **in-memory** from the sidecar, plots per-chunk continuity across the LVF boundaries. `--run-dir` (auto-locates sidecar) or `--cal-glob` + `--sidecar`. |
| [`diag_compare_zodi_vs_scalar.py`](diag_compare_zodi_vs_scalar.py) | Per-channel scatter: `zodi_pred` vs the cal's recovered `full_DC` (frame_scalar + chunk leakage). Re-fits from the pristine cal + npz (matches the sidecar fit). Sanity-check that the linear fit makes sense. |
| [`diag_compare_models.py`](diag_compare_models.py) | Run multiple zodi IPD models against the same cal files; side-by-side plot + `compare_models_summary.json` of per-model slope/intercept/r per channel. |

### Deprecated (legacy in-place mutators — do not use for new work)

| Script | Status |
|---|---|
| [`apply_anchor.py`](apply_anchor.py) | DEPRECATED. Wrote anchored copies / in-place edits of cal+mosaic. Superseded by the sidecar (`build_sidecar.py` + `load_anchor`). |
| [`apply_anchor_batch.py`](apply_anchor_batch.py) | DEPRECATED. Bulk in-place variant of `apply_anchor.py`. |
| [`pipeline_multi_channel.py`](pipeline_multi_channel.py) | DEPRECATED. Chained build → in-place apply → compare. Replace with `build_predictions* → build_sidecar → diag_*`. |

`SelfCal.ZodiAnchor.apply_anchor_to_file` and the private
`_shift_cal_file`/`_shift_mosaic_file` helpers are likewise legacy; the
sidecar path uses `fit_anchor_for_channel` + `write_sidecar` + `Anchor`.

## Output locations (not in this directory)

| Output | Where it goes |
|---|---|
| Metadata cache (`metadata_D{N}.h5`) | `<repo>/cache/zodi_anchor/` (gitignored). Path is set by `DEFAULT_METADATA_CACHE_TEMPLATE` in `build_predictions.py`. |
| Diagnostic figures | `<repo>/figures/zodi_anchor/...` (gitignored). All scripts that emit PNGs accept an `--out-dir`. |
| `zodi_pred_<tag>.npz` files | `<run>/zodi_preds/`. The expensive zodipy output; cache + cross-env hand-off (zodipy needs the `selfcal-zodipy`/numpy<2 env; the fit + consumer run in `selfcal`). Referenced by the sidecar (path + sha + len), not copied into it. |
| Anchor sidecar (`anchor_D{N}.h5`) | `<run>/zodi_anchor/`. The fit result (summary-only, one file per detector). Cal+mosaic stay pristine. |
| Anchored cal+mosaic | **Not written.** The shift is applied in-memory by `load_anchor()` consumers. For a materialized FITS (ds9 / publication), a separate opt-in step would write to `<run>/anchored_mosaics/` — never overwriting pipeline outputs. |

## Notes

- The anchor is **summary-only** in the sidecar; per-frame `full_DC` is
  recomputed from the pristine cal and per-frame `zodi_pred` is read from
  the referenced npz. The sidecar stores the npz path + sha1 + length so a
  consumer can re-load and verify identity.
- The fit result is repair-aware: `slope`/`intercept` are the raw
  per-channel linfit; `slope_final`/`C_final` are what consumers apply
  (equal to raw until a Phase-1 smoothing/repair pass overwrites them —
  see `todo/zodi_anchor_refactor.md`).
- Imports between scripts: the `build_*` and `diag_compare_models.py`
  scripts `from build_predictions import ...` for shared constants and the
  `extract_metadata_for_reproj_list` helper (each inserts its dir into
  `sys.path[0]`, so cwd doesn't matter). `build_sidecar.py` and
  `diag_*.py` import the anchor core from `SelfCal.ZodiAnchor` (resolved
  via `pip install -e .`).
- The actual anchor math lives in `SelfCal.ZodiAnchor`
  (`fit_anchor_for_channel`, `write_sidecar`, `Anchor`/`load_anchor`).
  These scripts only orchestrate I/O and CLI plumbing.
