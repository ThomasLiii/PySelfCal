# Run configs

Each `.toml` here fully describes one pipeline run. Run it with:

```bash
./selfcal_scripts/run.sh selfcal_scripts/configs/<name>.toml
# or the per-run launcher:
./selfcal_scripts/launch/<name>.sh
# validate without running (resolves jobs + mode, no compute):
./selfcal_scripts/run.sh selfcal_scripts/configs/<name>.toml --dry-run
```

The generic engine (`selfcal_scripts/runner/`) reads the config, asks the
**instrument** for geometry and the **mode** for the calibration recipe, and
sequences staging → setup_lsqr → apply_lsqr → save → mosaic. It never references
a telescope or a specific calibration variant by name.

## The configs

| Config | Task / mode | Replaces |
| --- | --- | --- |
| `d4_aromatic` | cal / continuum | `drivers/run_cal.py` |
| `d5` | cal / continuum | `experiments/run_cal_d5.py` |
| `damp0p5` | cal / continuum | `experiments/run_cal_damp0p5.py` |
| `damp_offset` | cal / continuum | `experiments/run_cal_damp_offset.py` |
| `pahfit` | cal / pahfit | `experiments/run_cal_pahfit.py` |
| `k2_readout` | cal / k2_readout | `experiments/run_cal_k2_readout.py` |
| `tiled_nep` | tiled / tiled | `drivers/chunked_NEP/run_cal_tiled_NEP.py` |
| `reproject_d4` | reproject | `drivers/run_reproject.py` |
| `precompute` | precompute | `drivers/precompute_lvf_params.py` |

## Schema

**Top-level (generic)** — `task` (`cal`|`tiled`|`reproject`|`precompute`),
`mode` (cal/tiled only), `output_dir`, `run_name` (may contain `{detector}`),
`resolution_arcsec`, `cache_dir`, `suffix`, `oversample`, `staging`
(`copy`|`reuse`), `keep_nvme`, `hdd_io_limit`, `apply_n_threads`. Optional
operational knobs: `n_frames` (limit to first N sorted reproj files),
`skip_mosaic`, `reproj_override` (run directly against an existing reproj dir,
no staging), `postprocess` (named subframe hook).

**`[instrument]`** — instrument-specific. SPHEREx: `name = "spherex"`, `detector`,
`num_sub`/`num_ch`/`num_col`, `calib_dir`, and exactly one channel selector:
`windows = ["Aromatic","Aliphatic"]` (named subchannel windows) /
`subch_window = [lo, hi]` (+ `window_name`) / `channels = [[1],[2]]` /
`channel_range = [lo, hi]`.

**`[params]`** — mode knobs. continuum/pahfit: `reg_weight`, `poly_degree`,
`poly_weight` (omit `poly_weight` to disable the column poly-constraint),
`line_fisher_threshold` (pahfit). k2_readout: `reg_weight`, `readout_reg_weight`.
tiled: the above + `subch_poly_degree`/`subch_poly_weight`/`subch_poly_lo`/
`subch_poly_hi`/`subch_tot`.

**Stage tables** — passed through verbatim as kwargs: `[calibration]` →
`setup_lsqr`, `[lsqr]` → `apply_lsqr`, `[mosaic]` → `make_mosaic`,
`[zodi]` (optional; set `pred_dir` to enable the post-cal anchor),
`[reproject]` (reproject task), `[tiled]` (tiled task: `ref_shape`, `grid`,
`overlap_px`, `tile_names`, `full_reproj_dir`, `nvme_subdir`, `stitched_suffix`).

## Adding a calibration variant (mode)

Drop a module in `selfcal_scripts/runner/modes/`:

```python
from .base import CalMode, register_mode

@register_mode("my_variant")
class MyVariant(CalMode):
    requires = ()                       # e.g. ("wavelength",) for spectral
    def build_offset_model(self, cfg, inst, det_inputs, ch_inputs, job, n_frames):
        ...                             # build it from inst.* geometry helpers
```

Add it to the import in `modes/__init__.py`. No engine edits. A config then sets
`mode = "my_variant"`.

## Adding a telescope (instrument)

Implement `selfcal.instruments.base.Instrument` (see
`selfcal/instruments/spherex/adapter.py`) and register the name in
`runner/config.get_instrument`. The generic modes (`continuum`) work against any
instrument; LVF-specific modes declare `requires` capabilities the instrument
must provide. Broadband instruments omit `wavelength`/`subchannel` and the
generic engine skips the wavelength append automatically.
