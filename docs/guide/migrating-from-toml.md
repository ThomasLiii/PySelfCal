# Migrating from TOML

Runs are Python run scripts; selfcal no longer reads TOML run configs. Until October 2026 a run
could also be a TOML file (`task`, `mode`, `[instrument]`, `[calibration]`, ...) started by
`selfcal_scripts/run.sh`. Two ways to describe one run kept the engine complicated, so the TOML
form was removed. `selfcal convert` turns an old config into a run script, and `selfcal adopt`
records the products a TOML run made, so that a run script reuses them.

`selfcal_scripts/run.sh` still runs a run script; given a `.toml` file, it stops and names the
`selfcal convert` command instead. <!-- check -->

## Convert a config

```bash
selfcal convert run.toml                         # writes run.py next to the config
selfcal convert run.toml -o my_run.py --force    # another path; --force replaces an existing script
```
<!-- check: the convert options after S2 (--no-check gone?) -->

The script holds the field, the recipe and the action's keyword arguments at its top level
(`FIELD`, `RECIPE`, `RUN`; a reprojection has `REPROJECT`, an LVF precompute `PRECOMPUTE`) and
runs the action under its `__main__` guard, as the [run scripts](python-api.md#the-command-line)
of the Python API do. Every value is written out, defaults included: a TOML table took the
library calls' own defaults (an `[lsqr]` table without `solver` meant LSMR with `damp = 0.01` and
300 iterations), which are not the Python API's (`sc.Fit()`: LSQR, no damping, 50 iterations), so
a converted script depends on no default.

The converter prints a note for every key it dropped or rewrote. Read the notes and the script
before running it; `selfcal plan run.py` then prints what the run would make, reuse or refuse,
without computing anything.

### The script is not checked

Before TOML support was removed, `selfcal convert` imported the script it wrote, lowered its
objects again and compared what the run engine would do with what the TOML config made it do; it
wrote nothing when the two differed. There is no TOML engine left to compare with, so the script
is now written unchecked.

The checked converter is in commit `d288f02` (and the commits before it). To convert with the
check, run that tree's command on the machine that holds the config's data (the check reads the
frames, the reference grid and the instrument's calibration maps):

```bash
git worktree add ../selfcal-toml d288f02
PYTHONPATH=../selfcal-toml python -m selfcal convert run.toml
```

The shipped configs were converted with the check before the removal: they are the run scripts
of `selfcal_scripts/runs/`, each proven to make the engine do what its config did.

### Three switches become the model

Three `[calibration]` switches have no setting of their own in Python. The converter writes what
they did into the model and notes it:

| TOML | Python |
| --- | --- |
| `offset_regularization = false` | every offset term without a hard polynomial gets `smooth=0` and no `poly_prior`: without the switch the engine built no smoothness or polynomial rows |
| `weighted_damping = false` | every sky term gets `damping=0.0`: no sky damping rows |
| a `basis` of one function | `sc.Offsets(times=...)`: one function is a coefficient (`basis=` takes at least two) |

Each rule was proven on toy runs: the TOML config and its converted script made byte-identical
cal files and mosaics.

Other notes:

- `[params]` keys that no code read (a key the mode never looked at, or any key beside a
  `[model]` table) are dropped.
- `[mosaic] sigma` without the sigma clip was read by no pass; it is dropped.
- `[tiling] frame_glob = "exp_*_det_00.h5"`: a tiled run takes every frame of its directory, the
  same frames for a one-detector instrument.
- `[zodi] pred_dir`: the script calls `spherex.zodi_anchor(result, predictions)` after the
  calibration.

## Where the old configs are

The TOML files that were shipped (`selfcal_scripts/configs/*.toml`, the gate configs in
`selfcal_scripts/gates/configs/`, the quickstart's `reproject.toml` and `cal.toml`, the
transfer-function kit's `transfer_function.toml` and its launcher) are in the git history:

```bash
git log --oneline --diff-filter=D -- '*.toml'          # the commit that removed them
git show <commit>^:selfcal_scripts/configs/d5.toml     # one of them, as it was
```

Each shipped config has its run script: `selfcal_scripts/runs/<name>.py` for
`selfcal_scripts/configs/<name>.toml`, the gate functions of `selfcal_scripts/gates/python_gates.py`
for the gate configs, `examples/quickstart/quickstart.py` for the quickstart and
`selfcal_scripts/transfer_function/transfer_function.py` for the transfer-function kit.

## Products a TOML run made

A product made by a TOML run has no sidecar (`<product>.json`, the inputs it was made from), so
a run script refuses it: its plan lists the product as `unrecorded`. `selfcal adopt` records it:

```bash
selfcal adopt run.py      # or, in Python: FIELD.adopt(RECIPE, jobs=...)
```

`adopt` checks each product the script's action would make against the recipe (the frames, the
sky terms, the offset maps and the detector shape of a cal file; the maps of a mosaic) and writes
its sidecar. What a product does not show, the settings of the fit and of the coadd, is taken on
trust: adopt a product only with the script of the config that made it. See
[Products, records and reruns](python-api.md#products-records-and-reruns).

## Where each key went

| TOML | Python |
| --- | --- |
| `task` | the action: `field.reproject`, `field.calibrate` (`tiles=`, `passes=`), `field.mosaic`; `precompute`: `spherex.precompute_lvf(detectors)` |
| `mode` + `[params]`, `[model]` | the model, `sc.Model` or a preset (below) |
| `output_dir` + `run_name`, `resolution_arcsec` | `sc.Field(path, instrument, pixel_scale)` |
| `[instrument]` | `sc.SPHEREx(detector, num_col=, num_sub=, num_ch=, calib_dir=)`, `sc.Euclid(band=, chunks=, strips=, ...)`, `sc.Camera(shape, chunks=, sci_ext=, dq_ext=, tag=)`; the channel or window selection: `jobs=spherex.channel(n)`, `spherex.channels(first, last)`, `spherex.group(...)`, `spherex.window(name, subchannels=range(lo, hi + 1))` |
| `suffix` | `sc.Recipe(name=)`, without the leading `_` |
| `[calibration] outlier_thresh`, `outlier_group_variable`, `outlier_group_edges`, `outlier_groups` | `sc.Fit(clip=sigma)`, `sc.Clip(sigma, variable=, edges=, per=)` |
| `[calibration] apply_mask`, `ignore_list`, `apply_weight` | `sc.Fit(use_mask=, ignore_flags=, shot_noise_weights=)` |
| `[calibration] damp_weight`, `damp_weight_line`; a line's `damp_weight` | `sc.Sky(damping=)` on each sky term |
| `[lsqr] iter_lim`, `atol` / `btol`, `damp`, `solver`, `precondition`, `use_float32` | `sc.Fit(iterations, tolerance=, damp=, method=, precondition=, float32=)` |
| `[params] line_fisher_threshold` | `sc.Fit(line_fisher_threshold=)` |
| `[hooks]` `pre_cal`, `post_cal`, `post_mosaic` | the hook objects, not their names: `sc.Fit(raw_frame_hook=)`, `sc.Fit(frame_hook=)`, `sc.Coadd(frame_hook=)` (Euclid: `StarMask`, `ResidualMask` of `selfcal.instruments.euclid.hooks`) |
| `[mosaic] apply_sigma_clipping` + `sigma`, `make_std_map`, `apply_mask`, `ignore_list`, `apply_weight`, `valid_chunk_thresh`, `apply_offset`, `normalize_offset` | `sc.Coadd(clip=, std=, use_mask=, ignore_flags=, shot_noise_weights=, min_chunk_coverage=, subtract_offsets=, normalize_offsets=)` |
| `oversample`, `wavelength_coadd`, `skip_mosaic` | `sc.Coadd(oversample=, instrument_maps=)`; `coadd=None` |
| `apply_n_threads`, `batch_size`, `cache_batch_size`, `coadd_batch_size` | `sc.Numerics(threads, batch=, mosaic_batch=, coadd_batch=)` |
| `cache_dir`, `staging`, `keep_nvme`, `hdd_io_limit`, `max_workers` (`[calibration]`, `[mosaic]`), `cache_intermediate` | `sc.Compute(scratch, stage=, keep_staged=, io_limit=, workers=, coadd_workers=, cache_frames=)` |
| `n_frames`, `reproj_override` | `calibrate(frames=n)`, `frames=directory`, `frames=sc.frames_in(directory)[:n]` |
| `cal_override` | `field.mosaic(recipe, cal=)` |
| `[tiling]` `grid`, `overlap_px`, `tile_names`, `tiles`, `frame_filter`, `halo`, `only_tiles` | `sc.Tiles(grid=, overlap=, names=, boxes=, assign=, halo=, only=)`; the suffix and `stitched_suffix`: `tile_name=`, `stitched_name=` |
| `[tiling]` `full_reproj_dir`, `nvme_subdir`, `rss_guardrail` | `calibrate(frames=)`, `sc.Compute(stage_dir=, memory_guard=)` |
| `[passes]` `n`, `order`, `stop_tol`, `sky_merge`, `keep_moments` | `sc.Passes(n, order=, stop_tol=, sky_merge=, keep_moments=)` |
| `[passes]` `init`, `sky`, `offset` | `sc.Passes(init_clip=, sky_clip=, offset=sc.Refit(degree, clip=, bright_cut=, min_pixels=, segments=, ridge=))`; `subch_clip = true`: `sc.Clip(sigma, per=sc.ChunkGroups.along("subchannel"))` |
| `[reproject]` | `field.reproject(exposures, reference=, method=, padding=, padding_fraction=, replace=, verify=)` |
| `[zodi] pred_dir` | `spherex.zodi_anchor(result, predictions)` after the calibration |

Two defaults differ. A `[passes]` table without `n` or `order` ran four passes, sky first;
`sc.Passes()` runs three, offset first, and refuses a schedule that ends on an OFFSET pass unless
`ends_on_offset=True`. The converter writes both values out.

### Modes

A mode was a named model with its knobs in `[params]`. Each is a preset or a model in Python:

| mode (historical names) | Python |
| --- | --- |
| `continuum` | `sc.continuum(smooth=reg_weight, poly_prior=sc.Poly(poly_degree, along=poly_axis, weight=poly_weight))` |
| `spectral` (`pahfit`) | `sc.spectral(lines, smooth=, poly_prior=)` |
| `spectral_softpoly` (`pahfit_subch`, `pahfit_lvf`; `tiled` without a mosaic) | `sc.spectral(lines, poly_prior=[..., sc.Poly(spectral_poly_degree, along="subchannel", window=range(lo, hi + 1), weight=spectral_poly_weight)])` |
| `spectral_polybasis` (`pahfit_lvf_polybasis`, `multiline`) | `sc.spectral(lines, polynomial=sc.Poly(spectral_poly_degree, window=range(lo, hi + 1), segments=...))` |
| `two_block_fixed` (`k2_readout`) | `sc.two_block(second=second_map, smooth=reg_weight, second_smooth=second_reg_weight)` |
| `model` | `sc.Model(sky=[...], offsets=[...], scalar=, variables={...}, weight=, priors=[...])` |

The lines of the spectral modes (`[[params.lines]]`) are sky terms: `spherex.line(name, damping=)`
for a shipped template, else `sc.Sky(name, times=sc.template(file))` or
`sc.Sky(name, times=sc.gaussian(center, sigma=...))`; the catalogue line `pah_3p29` is
`sc.Sky("pah_3p29", times=sc.catalog("pah_3p29"))`.

A `[model]` table maps term by term:

| `[model]` | Python |
| --- | --- |
| `[[model.sky]]` `name`, `coefficient`, `damp_weight` | `sc.Sky(name, times=, damping=)` |
| `[[model.offset]]` `map`, `kind = "free"` / `"fixed"` / `"grouped"` + `groups` | `sc.Offsets(on=map, per="frame")` / `per="all"` / `per=groups` |
| `kind = "polybasis"`: `axis`, `group_axis`, `degree`, `lo`, `hi`, `segments` | `sc.Offsets(polynomial=sc.Poly(degree, along=, each=, window=range(lo, hi + 1), segments=))` |
| `reg_weight`, `adjacency`, `poly`, `mean_zero`, `damp`, `exact_group_rows`, `render` | `smooth=`, `smooth_along=`, `poly_prior=sc.Poly(...)`, `mean_zero=`, `damping=`, `exact_group_rows=`, `render=` |
| a `coefficient` or a `basis`: `{ variable, function = "pkg.mod:fn", params }` | `times=fn` or `basis=fn, n=` (the function itself), or `sc.Function(fn, of=variable, **params)` |
| `function = "template"` / `"gaussian"` / `"linear"`; `{ catalog = name }` | `sc.template(...)`, `sc.gaussian(...)`, `sc.linear(...)`; `sc.catalog(name)` |
| `[model.variables]` `header`, `per_frame`, `detector`, `sky`, `sky_cal`, `layer`, `function`, `frame_function` | `sc.Header`, `sc.PerFrame`, `sc.DetectorMap`, `sc.SkyMap`, `sc.SolvedSky`, `sc.Layer`, `sc.Derived`, `sc.FrameFunction` |
| `weight` | `sc.Model(weight=)` |
| `[[model.prior]]` `frame_smoothness`, `sky_smoothness`, `toward_variable`, `"pkg.mod:fn"` | `sc.priors.frame_smoothness(...)`, `sc.priors.sky_smoothness(...)`, `sc.priors.toward(...)`, `sc.Prior(fn, terms, weight=, **params)` |
| `mosaic = "full"` / `"no_wav"` / `"none"` | `sc.Coadd(instrument_maps=True)` / `sc.Coadd(instrument_maps=False)` / `coadd=None` |

## Options removed with TOML

Some options went with the TOML form, because no run used them or because the Python API derives
their value itself. A config that sets one cannot be converted as it is: <!-- check: which ones convert.py refuses and which it notes after S2 -->

| TOML | instead |
| --- | --- |
| `postprocess = "mask_bright_pixels"` | a frame hook of your own, `sc.Fit(frame_hook=...)` |
| a hook named in `[hooks]` | the hook object itself (above) |
| `[lsqr] resume`, `keep_state` | none: a solve starts from its warm start |
| `[calibration] compact_zero_columns` | none: the zero columns are always compacted |
| `[calibration] damp_offset` | each offset term's own `sc.Offsets(damping=)` |
| `[calibration] outlier_aux_key`, `outlier_subchannel_edges` | `sc.Clip(sigma, variable=, edges=)` |
| `[calibration] spectral_fit`, `line_center`, `line_sigma` | a sky term with a coefficient, `sc.Sky(name, times=sc.gaussian(center, sigma=...))` |
| a polynomial prior with only one of `lo`, `hi` | both bounds: `sc.Poly(degree, window=range(lo, hi + 1))` |
| a `suffix` that does not start with `_` | `sc.Recipe(name=)`, joined to the product names with `_` |
| `[tiling] frame_glob`, `line` | none: a tiled run takes every frame of its directory, and the model decides which sky terms are stitched |
| `[reproject] use_ext`, `sci_ext_list`, `dq_ext_list` | the instrument's exposure layout: `sc.Camera(sci_ext=, dq_ext=, reference_ext=)`, or a subclass's `layout()` |
| `[reproject] inner_parallel`, `header_filter_workers` | none: fixed at 1 and 16 |
| `[instrument] name` of a registered or plugin instrument | the instrument object, an `sc.Instrument` subclass ([Bring your own telescope](../bring_your_own_telescope.md#4-with-code-an-instrument)) |
| a `mode` registered outside selfcal | a function that returns an `sc.Model` ([Bring your own telescope](../bring_your_own_telescope.md#2-the-model-is-yours)) |
