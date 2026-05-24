# zodi_anchor scripts

Post-hoc zodiacal-light anchor for SelfCal cal files + mosaics.

The core library lives in [`SelfCal/ZodiAnchor.py`](../../SelfCal/ZodiAnchor.py).
The scripts here orchestrate building per-frame zodi predictions
(via [zodipy](https://github.com/Cosmoglobe/zodipy)), applying the anchor
shift to existing cal+mosaic pairs, and producing diagnostic plots.

File names group by role:
- `build_*` — produce zodi-prediction `.npz` files (one per cal file or
  one per channel).
- `apply_*` — read a zodi-prediction `.npz` + a cal+mosaic, write
  anchored copies (or in-place modifications).
- `pipeline_*` — end-to-end orchestration (build → apply → render).
- `diag_*` — read-only diagnostics; never modify cal/mosaic files.

## Typical workflow

```
                        ┌─────────────────────────┐
                        │ build_predictions.py    │
                        │ (one cal file at a time)│
                        └──┬──────────────────────┘
                           │   or
                        ┌──┴──────────────────────────────┐
                        │ build_predictions_all_channels  │
                        │ (all ch of a detector at once)  │
                        └──┬──────────────────────────────┘
                           │   produces zodi_pred_*.npz
                           v
                  ┌────────────────────────┐
                  │ apply_anchor.py        │  (single)
                  │ apply_anchor_batch.py  │  (bulk, in-place option)
                  └──┬─────────────────────┘
                     │   writes/updates cal+mosaic
                     v
              ┌──────────────────────────────┐
              │ diag_*.py  (read-only plots) │
              └──────────────────────────────┘
```

Or use `pipeline_multi_channel.py` to chain build + apply + diagnostic
in one call for a whole detector.

## Scripts

### Builders

| Script | Purpose |
|---|---|
| [`build_predictions.py`](build_predictions.py) | Build per-frame zodi predictions for ONE cal file. Writes `zodi_pred_<tag>.npz` with arrays `zodi_pred`, `mjds`, `reproj_list`. Core module — exports `DEFAULT_CALIBRATION_DIR`, `DEFAULT_METADATA_CACHE_TEMPLATE`, `extract_metadata_for_reproj_list`, used by the other builders + pipeline. |
| [`build_predictions_all_channels.py`](build_predictions_all_channels.py) | Build predictions for all 34 channels of a detector in one go, without needing a per-channel cal file (reads exposure + LVF metadata directly). Faster than running `build_predictions.py` 34 times because the WCS/MJD extraction is shared. |

### Appliers

| Script | Purpose |
|---|---|
| [`apply_anchor.py`](apply_anchor.py) | CLI wrapper around `SelfCal.ZodiAnchor.apply_anchor_to_file`. Applies one `.npz` to one cal+mosaic, fits the linear `slope * zodi_pred + intercept` with moving-MJD sigma-clip, writes anchored copies (or in-place with `--in-place`). |
| [`apply_anchor_batch.py`](apply_anchor_batch.py) | Apply a built `.npz` set to many cal+mosaic pairs matched by tag. In-place by default; pairs `cal_<TAG>.h5` ↔ `mosaic_<TAG>.fits` automatically. |

### Pipeline

| Script | Purpose |
|---|---|
| [`pipeline_multi_channel.py`](pipeline_multi_channel.py) | End-to-end for one detector: builds per-channel predictions, applies the anchor to each cal+mosaic, optionally writes a `compare_*.png` per channel. Common arg is `--out-dir` where the `.npz` + anchored outputs + figures land. |

### Diagnostics (read-only)

| Script | Purpose |
|---|---|
| [`diag_compare_models.py`](diag_compare_models.py) | Run multiple zodi IPD models against the same cal files; produces a side-by-side plot and a `compare_models_summary.json` of per-model slope/intercept/r per channel. |
| [`diag_compare_zodi_vs_scalar.py`](diag_compare_zodi_vs_scalar.py) | Quick scatter: per-frame `zodi_pred` vs the cal's recovered `full_DC` (frame_scalar + chunk leakage). Quick sanity that the anchor's linear fit makes sense. |
| [`diag_plot_cross_channel.py`](diag_plot_cross_channel.py) | Stitch the anchored `mosaic_Ch*.fits` files across all channels of a detector and plot a cross-channel continuity diagnostic — the anchor should make adjacent channels meet at the LVF boundaries. |

## Output locations (not in this directory)

| Output | Where it goes |
|---|---|
| Metadata cache (`metadata_D{N}.h5`) | `<repo>/cache/zodi_anchor/` (gitignored). Path is set by `DEFAULT_METADATA_CACHE_TEMPLATE` in `build_predictions.py`. |
| Diagnostic figures | `<repo>/figures/zodi_anchor/...` (gitignored). All scripts that emit PNGs accept an `--out-dir`. |
| `zodi_pred_<tag>.npz` files | User-chosen `--out-dir`. Typically `/mnt/md124/.../zodi_preds/` for production runs. |
| Anchored cal+mosaic | Either alongside originals with a suffix, or in-place via `--in-place`. The driver hooks in `run_cal_v2.py` write to the same `calibration/` and `mosaic/` dirs. |

## Notes

- Imports between scripts: `diag_compare_models.py`,
  `build_predictions_all_channels.py`, and `pipeline_multi_channel.py`
  all `from build_predictions import ...` for the shared constants and
  the `extract_metadata_for_reproj_list` helper. The flat layout makes
  this work without `sys.path` munging — run them with cwd anywhere; the
  Python interpreter will pick up `build_predictions.py` from the same
  directory because the scripts insert their dir into `sys.path[0]`.
- The actual anchor math lives in `SelfCal.ZodiAnchor`. These scripts
  only orchestrate I/O and CLI plumbing.
