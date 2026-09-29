# Run configs

Each `.toml` here fully describes one pipeline run. Run it with:

```bash
./selfcal_scripts/run.sh selfcal_scripts/configs/<name>.toml
# or the per-run launcher:
./selfcal_scripts/launch/<name>.sh
# validate without running (resolves jobs + mode, no compute):
./selfcal_scripts/run.sh selfcal_scripts/configs/<name>.toml --dry-run
```

**Logs.** Every run (not `--dry-run`) writes its full console output — the
main process, worker processes and any traceback — to
`<output_dir>/<run_name>/logs/<task>_<YYYYmmdd-HHMMSS>_<pid>.log`, headed by
the command, the git commit and the complete config text, while still printing
to the terminal. `--log PATH` picks the file, `--no-log` turns it off. Configs
without an `output_dir`/`run_name` log under `<cache_dir>/logs/`.

The generic engine (`selfcal_scripts/runner/`) reads the config, asks the
**instrument** for geometry and the **mode** for the calibration recipe, and
sequences staging → setup_lsqr → apply_lsqr → save → mosaic. It never references
a telescope or a specific calibration variant by name.

## The configs

| Config | Task / mode | Replaced driver (removed; see git history) |
| --- | --- | --- |
| `d4_aromatic` | cal / continuum | `drivers/run_cal.py` |
| `d5` | cal / continuum | `experiments/run_cal_d5.py` |
| `damp0p5` | cal / continuum | `experiments/run_cal_damp0p5.py` |
| `damp_offset` | cal / continuum | `experiments/run_cal_damp_offset.py` |
| `pahfit` | cal / pahfit | `experiments/run_cal_pahfit.py` |
| `k2_readout` | cal / k2_readout | `experiments/run_cal_k2_readout.py` |
| `tiled_nep` | tiled / tiled | `drivers/chunked_NEP/run_cal_tiled_NEP.py` |
| `multiline_nep` | tiled / multiline | (workspace `spectral-pah-fit` campaign) |
| `sep_d4_npass` | npass / multiline (J=2, W/E tiles, n=8) | the SEP 4-pass chain (workspace `spectral-pah-fit` / `sky-closed-form` campaigns) |
| `nep_d4_npass` | npass / multiline (J=4, 16 overlap tiles, n=4) | passes 2+ on top of `multiline_nep` |
| `nep_d4_probe1k_npass` | npass / multiline (J=4, 1k-frame probe) | the multiline stability probe |
| `reproject_d4` | reproject | `drivers/run_reproject.py` |
| `precompute` | precompute | `drivers/precompute_lvf_params.py` |

## Schema

**Top-level (generic)** — `task` (`cal`|`mosaic`|`npass`|`reproject`|`precompute`;
`tiled` is accepted as an alias of `cal` + `[tiling]`), `mode` (cal/mosaic/npass only; see *Modes* below),
`cal_override` (mosaic task: apply this cal file, e.g. one solved on another grid; frames without a
reprojected file here are dropped), `output_dir`, `run_name` (may contain `{detector}`),
`resolution_arcsec`, `cache_dir`, `suffix`, `oversample`, `staging`
(`copy`|`reuse`), `keep_nvme`, `hdd_io_limit`, `apply_n_threads`. Optional
operational knobs: `n_frames` (limit to first N sorted reproj files),
`skip_mosaic`, `wavelength_coadd` (default `true`; `false` builds the mosaic
without the LVF `wav_mean`/`wav_std` maps — they sigma-clip against the std
map, so leaving it on requires `[mosaic]` `make_std_map` plus either
`apply_sigma_clipping` (coadded inside the sigma-clip pass, no extra pass) or
`cache_intermediate` (standalone coadd over the cache)), `reproj_override` (run directly
against an existing reproj dir, no staging), `postprocess` (named subframe
hook).

**`[instrument]`** — instrument-specific. SPHEREx: `name = "spherex"`, `detector`,
`num_sub`/`num_ch`/`num_col`, `calib_dir`, and exactly one channel selector:
`windows = ["Aromatic","Aliphatic"]` (named subchannel windows) /
`subch_window = [lo, hi]` (+ `window_name`) / `channels = [[1],[2]]` /
`channel_range = [lo, hi]`.

**`[params]`** — mode knobs. continuum/pahfit: `reg_weight`, `poly_degree`,
`poly_weight` (omit `poly_weight` to disable the column poly-constraint),
`line_fisher_threshold` (pahfit). pahfit_subch/pahfit_lvf: the above +
`subch_poly_degree`/`subch_poly_weight`/`subch_poly_lo`/`subch_poly_hi`/`subch_tot`;
pahfit_lvf adds `line_template_npz` (+ `line_template_norm`); pahfit_lvf_polybasis
uses the hard poly-basis offset (`subch_poly_degree`/`_lo`/`_hi`, no weight) with
the template sky (== `multiline` with one `[[params.lines]]` block).
k2_readout: `reg_weight`, `readout_reg_weight`.
tiled: the above + `subch_poly_degree`/`subch_poly_weight`/`subch_poly_lo`/
`subch_poly_hi`/`subch_tot`. multiline: `subch_poly_degree`/`subch_poly_lo`/
`subch_poly_hi` (hard poly-basis offset), `line_fisher_threshold`, and one
`[[params.lines]]` table per spectral block — each with `name`, a profile
(`template_npz` = realistic peak-normalized template, or `center_um` [+`sigma_um`
| `intrinsic_var_um2`] for an analytic Gaussian), and optional per-line
`damp_weight` (falls back to `[calibration].damp_weight_line`).

**Stage tables** — passed through verbatim as kwargs: `[calibration]` →
`setup_lsqr`, `[lsqr]` → `apply_lsqr`, `[mosaic]` → `make_mosaic`,
`[zodi]` (optional; set `pred_dir` to enable the post-cal anchor),
`[reproject]` (reproject task; `check = true` load-tests every frame and quarantines broken ones),
`[hooks]` (per-frame hooks: `pre_cal`, `post_cal`, `post_mosaic`, each `{ name = ..., <params> }`
naming one of the instrument's hook factories or the runner's `mask_bright_pixels`; the callable
receives a `FrameContext`), `[tiling]` (old spelling `[tiled]`; tiles the `cal`
task: `ref_shape`,
`full_reproj_dir`, `nvme_subdir`, `stitched_suffix`, and the tile geometry —
EITHER a uniform grid `grid = [n_y, n_x]` + `overlap_px` + `tile_names`, OR an
explicit `tiles = [{name, bbox=[y0,y1,x0,x1]}, ...]` list of arbitrary/overlapping
tiles for the adaptive-overlap layout; `line` toggles the spectral-block stitch).

**`[passes]`** (npass task — the N-pass alternating solve, see the "N-pass
alternating solve" section of [PIPELINE.md](../../PIPELINE.md)): `n` (number
of passes; `1` == the `cal` solve, tiled or not, byte-equal), `stop_tol`
(stop after a SKY pass whose per-block step RMS is below it; `0` = run all
`n`), `sky_merge` (`combine` = exact additive moments, default; `stitch` =
Fisher stitch, legacy), `order` (`sky_first` default, or `offset_first` =
INIT → OFFSET → SKY → … — prefer it whenever pass 1 is **tiled**, it removes
the seam-adjacent lobes the per-tile INIT gauges otherwise leave in the first
exact sky; it ends on a sky for odd `n`), `keep_moments` (retain the per-tile
moment dumps, ~23 GB each at J=4; they are deleted after the combine by
default), and the per-pass-type clip knobs `init = {outlier_thresh,
subch_clip, ignore_list}` (pass 1 only; omit to reproduce the legacy clip),
`sky = {outlier_thresh, subch_clip}`, `offset = {poly_degree, outlier_thresh,
subch_clip, bright_cut, min_pix, segments}` — `segments` (optional, e.g.
`[[200, 259], [260, 320]]`, inclusive subchannel ranges inside
`subch_poly_lo..hi`) fits an independent degree-`poly_degree` Chebyshev per
column on each segment instead of one over the whole window; use it when the
window is wide, since a single polynomial over ~120 subchannels resolves 2×
less per-frame subchannel structure than the same degree over 60 and a higher
global degree extrapolates wildly wherever a frame's coverage is partial (see
PIPELINE.md); `ridge` (default `0` = plain least squares) adds a Tikhonov term
on the shape/level coefficients, λ² = ridge² × the median diagonal of DᵀD, so a
segment a frame barely covers is held near zero instead of extrapolating —
pair it with `segments` (SEP: `ridge = 0.03`). The joint INIT solve of the `multiline`
mode takes the same segmentation as `[params].subch_poly_segments` (an independent
degree-`subch_poly_degree` shape per column on each segment, plus a level per segment
after the first). Pass 1 is the `cal` task on the same config — tiled when
`[tiling]` is present (its tiles are then the memory tiling of every SKY pass;
overlapping tiles are de-duplicated first-tile-wins). A re-run resumes: passes
whose product exists are skipped.

## The model: sky terms + offset terms

Every calibration fits ``data(frame, pixel) = Σ_j S_j(pixel)·c_j(λ) + Σ_m O_m(frame, chunk) + s(frame)``.
The **S terms** are sky blocks (a continuum, and line terms = per-pixel amplitudes of a spectral
profile), the **O terms** are per-frame offset blocks on the instrument's chunk maps, ``s`` the
per-frame scalar. Each term carries its own priors and constraints. The named modes below are
*presets* of that model; ``mode = "model"`` spells it out in a ``[model]`` table so a new
combination needs no Python:

```toml
mode = "model"

[model]
scalar = true                     # per-frame scalar (the offset's DC)
mosaic = "full"                   # full | no_wav | none

[[model.sky]]
type = "continuum"                # constant per pixel; damping = [calibration].damp_weight

[[model.sky]]
type = "line"                     # per-pixel amplitude x profile(λ)
name = "aromatic"
template = ".../aromatic_3p289.npz"   # or center_um + sigma_um | center_um + intrinsic_var_um2 | catalog = "pah_3p29"
damp_weight = 5e-3                # prior: Tikhonov shrinkage of this block

[[model.offset]]
map = "subchannel"                # a chunk map of the instrument (omit: the primary map)
kind = "free"                     # free (per frame & chunk) | polybasis (Chebyshev along an axis) | fixed (shared by all frames)
reg_weight = 0.1                  # smoothness between neighbouring chunks
adjacency = ["column"]            # along which axes chunks are neighbours (omit: the map's default)
mean_zero = true                  # anchor: the per-frame mean over chunks is 0
poly = [ { axis = "column", degree = 1, weight = 0.5 },                  # soft polynomial constraints
         { axis = "subchannel", degree = 3, lo = 200, hi = 320, weight = 1.0 } ]

[[model.offset]]
map = "readout"
kind = "fixed"                    # one offset vector shared by every frame (detector-fixed pattern)
mean_zero = true
```

Term vocabulary — `kind = "free"`: `reg_weight` + `adjacency` (smoothness), `poly` (shape),
`mean_zero` (anchor); `kind = "polybasis"`: `axis`, `group_axis` (defaults: the map's spectral and
group axes), `degree`, `lo`, `hi`, `segments`; `kind = "fixed"`: as `free`, one vector for all
frames; `kind = "grouped"`: one vector per frame group `groups` (the instrument's frame groupings;
every instrument provides `"detector"`). Any term: `damp` (Tikhonov damping toward 0),
`exact_group_rows` (fixed/grouped: anchor + adjacency rows once per group), `render` (which of the
instrument's mosaic renderers draws it). Line terms: `template` (npz `center_um` + `G`/`G_peaknorm`), or `center_um` + `sigma_um`, or
`center_um` + `intrinsic_var_um2` (per-pixel width from the instrument's band-width map), or
`catalog` (an entry of the instrument's line catalogue) with `line_center` / `line_sigma`.
The axes named here are the ones the instrument's chunk map declares (SPHEREx: `subchannel`,
`column`; the `grid` instrument: `row`, `col`). Implementation: `selfcal.models.spec`.

## Modes (presets of the model)

| mode | offset structure | sky | presets (historical names, same behaviour) |
| --- | --- | --- | --- |
| `continuum` | adjacency along the chunk map's adjacency axes + optional soft polynomial (`poly_weight`, `poly_degree`, `poly_axis`) + mean-zero anchor + per-frame scalar | continuum | — |
| `spectral` | as `continuum` | continuum + line(s): `[[params.lines]]`, or `line_template_npz`, or a catalogue line `line` (default `pah_3p29`) | `pahfit` |
| `spectral_softpoly` | + soft polynomial along the spectral axis: `spectral_poly_degree` / `_lo` / `_hi` / `_weight` (historical `subch_poly_*`) | as `spectral` | `pahfit_subch`, `pahfit_lvf`; `tiled` (column poly always on, spectral poly required, no mosaic) |
| `spectral_polybasis` | hard Chebyshev basis in the spectral axis per group axis (no weight knob), optional `spectral_poly_segments` | as `spectral` | `pahfit_lvf_polybasis`, `multiline` |
| `two_block_fixed` | primary map regularised along its spectral axis + a detector-fixed second map (`second_map`, default `readout`; `second_reg_weight`) | continuum | `k2_readout` |
| `model` | whatever the `[model]` table says | idem | — |

Modes read `[params]` only; the axes they refer to ("column", "subchannel",
"row", "col") are declared by the instrument's chunk map, so a recipe runs on
any instrument that declares the axes it needs (`requires` lists the
capability tags: `wavelength`, `spectral_axis`).

## Adding a calibration variant (mode)

Drop a module in `selfcal_scripts/runner/modes/`:

```python
from .base import CalMode, register_mode, standard_block

@register_mode("my_variant", "my_old_name")      # extra names are aliases
class MyVariant(CalMode):
    requires = ()                       # e.g. ("wavelength",) for spectral
    def model_spec(self, cfg, inst, geom):
        return ModelSpec(sky=(SkyTerm('continuum'),),                     # the S terms
                         offset=(OffsetTerm(kind='free', reg_weight=cfg.params['reg_weight'],
                                            mean_zero=True),),             # the O terms + priors
                         scalar=True)
```

Add it to the import in `modes/__init__.py`. No engine edits. A config then sets
`mode = "my_variant"`. (Most new combinations need no mode at all: write them in a
`[model]` table with `mode = "model"`.)

## Adding a telescope (instrument)

**No code — the built-in `grid` instrument.** Any single-detector imager whose
exposures carry a science image with a celestial WCS and a bitmask extension::

    [instrument]
    name = "grid"
    detector_shape = [2048, 2048]   # rows, cols of the science array
    chunks = [8, 8]                 # offset chunk grid
    sci_ext = 1
    dq_ext = 2
    tag = "MyCam"                   # product-name tag

Then `task = "reproject"` on the exposure directory and `task = "cal"` with
`mode = "continuum"`; `tests/test_runner_e2e_toy.py` is a complete worked
example on synthetic exposures.

**Euclid NISP** is the built-in multi-detector example (`name = "euclid"`, `band`, `chunks`,
`strips`, `tilt_strips`, `edge_zero_px`/`edge_ramp_px`): 16 detectors per exposure, a grid map
plus column/row stripe and tilt maps, spline/strip/ramp renderers, electron units, the
`star_position_mask` / `residual_mask` hooks. `workspace/unify/configs/gate_euclid_unify.toml` is the
frozen EDFN recipe as a `[model]` table.

**With code.** Subclass `selfcal.instruments.Instrument` (five required
methods: `jobs`, `frame_tag`, `exposure_layout`, `detector_geometry`,
`job_geometry`; hooks for the rest), decorate it with
`@register_instrument("name")` — or publish it from your own package through
the `selfcal.instruments` entry-point group — and select it with
`[instrument].name`. `selfcal/instruments/spherex/adapter.py` is the full
reference (chunk axes, wavelength maps, renderer, post-cal hooks),
`selfcal/instruments/grid.py` the minimal one. Modes declare the capability
tags they need; an instrument without them cannot run those modes, and the
engine skips the wavelength coadd when the instrument has none.
