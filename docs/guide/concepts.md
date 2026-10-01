# How selfcal works

This page explains what a selfcal run computes, and why, for readers who want to understand the
method before they configure a run. It links the guides with the details:
[Run configuration](configuration.md) (config keys),
[Bring your own telescope](../bring_your_own_telescope.md) (new instruments and models), the
[Pipeline runbook](pipeline.md) (tuning, file formats) and
[Architecture](../developer/architecture.md) (the code). The [Glossary](glossary.md) defines the
terms.

## The problem

A survey observes each part of the sky several times: in different exposures, through different
detectors or different parts of one detector, at different times. Besides the sky, every exposure
carries signals that belong to the exposure rather than to the sky: a bias or dark level, a
foreground that changes from one exposure to the next (for SPHEREx, the zodiacal light), stripes
from the read-out electronics, a pattern fixed to the detector. selfcal calls these additive
signals *offsets*.

Where exposures overlap, one patch of sky is seen through different detector regions in different
exposures. The sky is the same each time, while an offset stays with its detector region and its
exposure. With enough overlap the data alone tell the two apart, so selfcal solves for the sky and
the offsets together, as one sparse linear least-squares problem, instead of modelling the
instrument from first principles. In its simplest form (the `continuum` mode), the value of frame
`k` (one detector of one exposure) on pixel `P` of a common reference grid is modelled as

```text
data[k, P] = S[P] + O[k, c] + s[k] + noise        c: the chunk of the detector on which frame k saw P
```

`S` is the sky map, shared by all frames. `O` holds one unknown per frame and *chunk*, a region of
the detector such as one cell of an 8 x 8 grid. `s` holds one constant per frame, the *per-frame
scalar*. An offset in this sense is

- additive: it never multiplies the data (multiplicative effects such as gains are linearised
  around a previously solved sky; see the [limits](../bring_your_own_telescope.md#6-limits));
- constant over its chunk, unless the model gives it a known shape;
- per frame, or shared by a group of frames (all frames, the frames of one detector, of one night).

Once the offsets are known, selfcal subtracts them from every frame and coadds the corrected frames
into a mosaic.

## From exposures to a mosaic

```text
raw exposures                FITS files, or any format an instrument's exposure reader reads
   │   task reproject        Reprojector.define_reference, Reprojector.run_reproject
   ▼
ref.fits                     the reference grid: a celestial WCS and a shape
reprojected/*.h5             one frame file per detector of each exposure
   │   task cal              Calibrator.setup_lsqr, apply_lsqr, save_calibration
   ▼
calibration/cal_<stem>.h5    sky maps, offsets, per-frame scalars, coverage
   │   task cal or mosaic    Mosaicker.load_calibration, make_mosaic, save_mosaic
   ▼
mosaic/mosaic_<stem>.fits    mean, standard-deviation and sigma-clipped mean maps
```

The paths are relative to the run directory `<output_dir>/<run_name>/`.

1. **Reproject** (task `reproject`). The instrument's exposure layout says how to read a raw file
   and which detectors it holds.
   [`Reprojector.define_reference`][selfcal.pipeline.pipeline_wrapper.Reprojector.define_reference]
   loads `ref.fits` if it exists; otherwise it computes a WCS at `resolution_arcsec` that contains
   every exposure (or derives one from `[reproject] source_ref_path`) and writes it.
   [`Reprojector.run_reproject`][selfcal.pipeline.pipeline_wrapper.Reprojector.run_reproject]
   resamples every detector of every exposure onto that grid
   ([`batch_reproject`][selfcal.io.reprojection.batch_reproject], with the `reproject` package)
   and writes one *frame file*, `exp_<exposure>_det_<detector>.h5`: the values on a box of the
   reference grid (`ref_coords`), the data-quality bit mask (if any) and the detector coordinates
   of every box pixel (`sub_mapping`), which tell the solver where on the detector each value was
   recorded. Data that are not images with a WCS can be written as frame files directly
   ([`write_frame`][selfcal.io.frames.write_frame]).
2. **Calibrate** (task `cal`), once per [job](glossary.md#job): a SPHEREx
   [channel](glossary.md#channel) or subchannel window, or the one job of the `grid` and `euclid`
   instruments. [`Calibrator.setup_lsqr`][selfcal.pipeline.pipeline_wrapper.Calibrator.setup_lsqr]
   ([`setup_lsqr`][selfcal.core.system.setup_lsqr]) reads the frames in parallel worker processes
   and builds the sparse system: one row per observation, plus the rows of the priors.
   [`Calibrator.apply_lsqr`][selfcal.pipeline.pipeline_wrapper.Calibrator.apply_lsqr]
   ([`apply_lsqr`][selfcal.core.solve.apply_lsqr]) solves it with [LSQR or LSMR](glossary.md#lsqr),
   and [`Calibrator.save_calibration`][selfcal.pipeline.pipeline_wrapper.Calibrator.save_calibration]
   writes the [cal file](glossary.md#cal-file), which [`CalFile`][selfcal.io.calfile.CalFile] reads.
   Task `cal` skips the solve of a job whose cal file exists: change `suffix` (or remove the file)
   to solve again.
3. **Mosaic** (task `cal` after the solve, or task `mosaic` from an existing cal file).
   [`Mosaicker.make_mosaic`][selfcal.pipeline.pipeline_wrapper.Mosaicker.make_mosaic] subtracts
   each frame's offsets and coadds the frames
   ([`run_coadd_schedule`][selfcal.core.coadd.run_coadd_schedule]), and
   [`Mosaicker.save_mosaic`][selfcal.pipeline.pipeline_wrapper.Mosaicker.save_mosaic] writes the
   FITS file.

A job's products are named by its [stem](glossary.md#stem), `<frame_tag>_<job><suffix>`: the
instrument's [frame tag](glossary.md#frame-tag), the job name and the config's `suffix`, for
example `MyCam_Chunks8x8_All` for a `grid` camera. The run engine (`selfcal_scripts/run.sh` with
one TOML config per run) runs these steps; the same classes can be used from Python
([Using the pipeline](../developer/architecture.md#using-the-pipeline)). The
[Quickstart](../getting-started/quickstart.md) runs all three on simulated exposures.

## The model

For every *observation* `i`, one value of one frame on one pixel `P` of the reference grid,
recorded at a known position on the detector, the solver fits

```text
data_i = Σ_j S_j[P] · c_j(v_i)  +  Σ_m Σ_k O_m[g_m(frame), chunk_m(i), k] · φ_mk(v_i)  +  s(frame)  +  noise_i
```

- **Sky terms** `S_j` ([`SkyTerm`][selfcal.models.spec.SkyTerm], `[[model.sky]]`): a map on the
  reference grid, shared by every frame, times a known *coefficient* `c_j(v)`: `c = 1` for a plain
  sky map (one value per pixel), a template of the wavelength for the map of an emission feature, a
  sine of the time for a seasonal term. A coefficient is a built-in function (`template`,
  `gaussian`, `linear`), any importable Python function, or a named entry of the instrument
  (SPHEREx: `pah_3p29`).
- **Offset terms** `O_m` ([`OffsetTerm`][selfcal.models.spec.OffsetTerm], `[[model.offset]]`): an
  offset per chunk of chunk map `m`, shared by the frames of a group `g_m` as the term's `kind`
  says: `free` (each frame its own), `fixed` (one for all frames: a pattern fixed to the detector),
  `grouped` (frames with equal values of a [frame variable](glossary.md#frame-variable) such as
  `detector`), or `polybasis` (a polynomial along one chunk axis, fitted by its coefficients). A
  `coefficient` multiplies the offset by a known function `φ` of data variables (a pattern times
  the detector temperature); a `basis` of `n` functions gives `n` unknowns per chunk (a gradient
  across the detector in every frame). With neither, `φ = 1`: the plain chunk offset.
- **Per-frame scalar** `s` (`scalar = true`): one additive constant per frame.
- **Data variables** `v`: named quantities with one value per observation, which every function of
  the model reads by name ([`selfcal.models.variables`](../reference/selfcal/models/variables.md)):
  the built-ins `det_x`, `det_y` (detector position), `sky_x`, `sky_y` (reference pixel) and
  `frame`; the instrument's detector maps (SPHEREx: the band centre `BC` and band width `BW`, also
  called `wavelength` and `bandwidth`) and per-frame values (`exposure`, `detector`); and the
  model's own `[model.variables]` (header keywords, maps, stored planes, functions; see
  [Data variables](../bring_your_own_telescope.md#data-variables)).

The solve seeks the unknowns (the pixels of every sky map, the offsets, the scalars) that minimise

```text
Σ_i  w_i² · (data_i − model_i)²   +   Σ_r  (prior row r)²
```

where `w_i` is the observation's weight (see [Masks, outliers and weights](#masks-outliers-and-weights))
and the prior rows are linear equations on the unknowns that settle what the data leave free (see
[What the data cannot tell apart](#what-the-data-cannot-tell-apart)).
[`SystemLayout`][selfcal.core.layout.SystemLayout] orders the unknowns: the sky maps, then the
offset terms, then the scalars.

Every run builds one [`ModelSpec`][selfcal.models.spec.ModelSpec]. A named *mode* builds it from a
few `[params]` keys (the [modes](configuration.md#modes-presets-of-the-model) are presets of the
general model); `mode = "model"` reads it from a `[model]` table. For example, `mode = "continuum"`
with `[params] reg_weight = 0.1` and no `poly_weight` builds the same model as

```toml
mode = "model"

[model]
scalar = true                 # the per-frame scalar s

[[model.sky]]
name = "continuum"            # no coefficient: c = 1

[[model.offset]]              # on the instrument's primary chunk map
kind = "free"                 # an offset per frame and chunk
reg_weight = 0.1              # smoothness between neighbouring chunks
mean_zero = true              # each frame's mean offset is pulled to 0
```

Either way the smoothness rows also need `[calibration] offset_regularization = true` (see
[What the data cannot tell apart](#what-the-data-cannot-tell-apart)).
[Bring your own telescope](../bring_your_own_telescope.md#2-the-model-is-yours-mode-model)
describes every kind of term, variable and prior, with worked examples.

## Chunk maps and chunk axes

A *chunk map* is an integer image on the detector grid: the value of a pixel is the id of its
chunk, and `-1` marks pixels in no chunk. An instrument defines one or more chunk maps by name
([`ChunkMap`][selfcal.instruments.base.ChunkMap]) and marks one as primary; an offset term that
names no map uses the primary one, and the name `"detector"` makes the whole detector one chunk.
An observation's offset is read at its detector position by bilinear interpolation of the chunk
map, so an observation within a pixel of a chunk edge shares its offset between the two chunks.

*Chunk axes* ([`ChunkAxes`][selfcal.models.offset_structure.ChunkAxes]) are the named coordinates of
the chunks: for each axis, the value of every chunk along it and the image direction in which
neighbouring chunks differ in it. Priors and modes are written in axes ("smoothness along
`column`", "a degree-3 polynomial along `subchannel`, one per `column`"), so a recipe runs on any
instrument whose chunk map declares the axes it needs. A map also names its default adjacency
axes, its *spectral axis* (along which the wavelength changes; none for a broadband instrument) and
its *group axis* (one polynomial per value in a `polybasis` term).

| instrument | chunk maps | axes |
| --- | --- | --- |
| `grid` | `grid`: an `ny` x `nx` grid of rectangles (`chunks = [ny, nx]`), id `row · nx + col` | `row`, `col`; adjacency along both; group axis `row` |
| `spherex` | `subchannel` (primary): [subchannels](glossary.md#subchannel), the arcs of nearly constant wavelength of the [LVF](glossary.md#lvf), each cut into `num_col` vertical columns, id `subchannel · num_col + column`; `readout`: one chunk per read-out channel of the detector | `subchannel` (spectral axis), `column` (group axis and default adjacency); `readout` |
| `euclid` | `grid` (primary): `chunks` x `chunks` squares (default 40); `col_strips`, `row_strips`: vertical and horizontal stripes; `col_tilt`, `row_tilt`: stripes on which a degree-1 `polybasis` term is one linear ramp per frame | `row`, `col`; `strip`, `all` |

Chunk size trades flexibility against noise: smaller chunks absorb finer offset structure, but
fewer pixels constrain each unknown. For SPHEREx, `num_col` is the main knob of this kind (see
[Calibration pipeline tuning](pipeline.md#calibration-pipeline-tuning)). The solve treats an
offset as constant over its chunk (times its known functions, if any); the mosaic may draw it more
smoothly, through the instrument's [offset renderer](glossary.md#offset-renderer) (for SPHEREx, a
mean-preserving spline across the arcs and columns).

## What the data cannot tell apart

Some changes to the unknowns leave every model value unchanged, or almost unchanged. The data
cannot fix them, so the priors (and, for an unconverged solve, the starting point) choose one
solution along each such direction, a choice called a *gauge*:

- **The overall level.** Adding a constant to the sky and subtracting it from every per-frame
  scalar (or every offset) changes nothing, so the data never fix the absolute sky level (see
  [The absolute level](#the-absolute-level)). Sets of frames that share no sky, directly or through
  other frames, each have their own free level.
- **A frame's mean offset.** Adding a constant to all chunk offsets of one frame and subtracting it
  from that frame's scalar changes nothing.
- **A smooth gradient.** A sky gradient is nearly the same as a ramp across the chunks of every
  frame plus per-frame scalars. Offsets constant per chunk follow a ramp only in steps, so the data
  constrain such a gradient only through its variation inside a chunk.
- **Spectral models.** A uniform level of a line map and a fixed detector pattern shaped like the
  line's coefficient are exactly degenerate when the coefficient depends only on the detector
  position, as a template of the wavelength map does
  ([`selfcal.line_floor`](../reference/selfcal/line_floor.md)). Sky terms with overlapping
  coefficients are nearly degenerate at every pixel; with `[[params.lines]]` the spectral modes
  print the normalised Gram matrix of the coefficients and warn when an entry exceeds 0.7 in
  absolute value.

The priors, and the rows each adds to the system:

| prior | config | rows |
| --- | --- | --- |
| mean-zero anchor | `mean_zero = true` (set in the standard offset term of the modes) | per frame (and basis function): `10 · Σ_c O[c] = 0`, over every chunk `c` of the map |
| adjacency smoothing | `reg_weight` (0 by default in `[model]`; the modes read `[params] reg_weight`, default 0.1), `adjacency` | per frame and pair of neighbouring chunks `a`, `b`: `reg_weight · (O[a] − O[b]) = 0` |
| polynomial constraint | `poly = [{ axis, degree, weight, lo, hi }]` (modes: `poly_weight`, `poly_degree`, `poly_axis`, `spectral_poly_*`) | per frame and run of `degree + 2` consecutive chunks along `axis`: `weight · Σ stencil · O = 0`, a finite difference that vanishes on polynomials of degree `degree` or less |
| sky damping | `[calibration] weighted_damping = true` with `damp_weight` (first sky term), `damp_weight_line` (the others; default 3 x `damp_weight`) or a term's own `damp_weight` | per covered pixel: `sqrt(damp_weight · coverage) · S[P] = 0`, with the pixel's [coverage](glossary.md#coverage) |
| offset damping | `damp` of an offset term | per covered unknown: `sqrt(damp · coverage) · O = 0` |
| user priors | `[[model.prior]]`: a ready-made function of [`selfcal.models.priors`](../reference/selfcal/models/priors.md) (`frame_smoothness`, `sky_smoothness`, `toward_variable`) or your own | the linear rows the function returns |

The adjacency and polynomial rows are added only when `[calibration] offset_regularization = true`,
and the sky damping only when `weighted_damping = true`; both are off unless set. What each settles:

- The **mean-zero anchor** moves each frame's mean offset into its scalar, so the chunk offsets
  carry only structure within the frame. The mean runs over every chunk of the map, observed or
  not; for a term whose groups observe different parts of the map (detectors on one focal plane),
  anchor with `damp` instead.
- The **sky damping** pulls the maps toward zero: of all the levels the data allow, it prefers the
  one that makes the maps smallest, a level with no physical meaning. Like any Tikhonov term it also
  shrinks real structure toward zero, by an amount that grows with the weight.
- The **adjacency smoothing** makes steps between neighbouring chunks costly, so smooth structure
  goes to the sky rather than to offset ramps; the sky damping pushes the other way. A
  **polynomial constraint** lets the offset follow a polynomial of its degree along its axis (a
  degree-1 constraint along the columns leaves linear ramps free) and penalises the rest; a
  [`polybasis`](glossary.md#polybasis) term makes that shape exact.
- `[lsqr] damp` damps every unknown (default 0.01; the shipped configs set 0). With a limited
  `iter_lim`, LSQR also stops before it has moved far along weakly constrained directions, where
  the result then depends on the [warm start](glossary.md#warm-start).

### Comparing solutions

Two solves that fit the data equally well can differ in their raw arrays along these directions.
Compare quantities that a change of gauge leaves alone:

- the total offset of each frame and chunk, the offset plus the frame's scalar
  ([`CalFile.total_offsets`][selfcal.io.calfile.CalFile.total_offsets] folds the scalar into the
  first map), rather than either part;
- offsets with each frame's mean over chunks and each chunk's mean over frames removed, which also
  removes a pattern common to every frame, such as the detector ramp of a sky gradient; the test
  suite compares recovered and injected offsets this way (`_degauge` in
  [tests/test_runner_e2e_toy.py](https://github.com/ThomasLiii/PySelfCal/blob/main/tests/test_runner_e2e_toy.py));
- sky maps after removing their mean, or after an external anchor has set their level;
- maps of terms with a coefficient only where their [separability](glossary.md#separability) or
  [Fisher information](glossary.md#fisher) is high enough.

## Masks, outliers and weights

- **Data-quality masks.** With `apply_mask = true`, a sample is dropped when any bit of its
  data-quality mask is set, except the bits listed in `ignore_list`. The mask comes from the
  exposure's `dq_ext` and is resampled bit by bit during the reprojection. `[calibration]` and
  `[mosaic]` each have their own `apply_mask` and `ignore_list`, so the solve can leave out pixels
  that the mosaic uses, or the reverse.
- **Outlier rejection in the solve.** With `[calibration] outlier_thresh`, every sample is scored
  against its frame's median, in units of `1.4826 · MAD` (the median absolute deviation), and
  samples scoring above the threshold are left out of the solve
  ([`find_outliers`][selfcal.geometry.map_helper.find_outliers]). The score uses the frame's own
  values, before any model is subtracted, so it removes compact bright sources and artefacts. When
  a frame's brightness changes strongly across the detector (SPHEREx sees a different wavelength in
  every subchannel), the [grouped clip](glossary.md#grouped-clip) scores each sample within its
  group instead: `outlier_group_edges` bins a data variable, `outlier_group_variable` (by default
  the instrument's wavelength map), and in task `npass` `subch_clip = true` sets one bin per
  subchannel. Samples whose weight, coefficient or basis value is not finite are dropped as well.
- **Per-frame hooks.** `[hooks]` runs a function on every frame, right after it is read (`pre_cal`)
  or after its weights are computed (`post_cal` in the solve, `post_mosaic` in the mosaic): one of
  the instrument's hook factories (Euclid: `star_position_mask`, `residual_mask`) or the runner's
  `mask_bright_pixels`.
- **Weights.** An observation's weight `w_i` multiplies its row of the system, so the fit weights
  the observation by `w_i²`. It is the product of
    - the data-quality mask (0 or 1);
    - the job's [valid weight](glossary.md#valid-weight) on the detector, from the instrument: 1
      everywhere for `grid`; for a SPHEREx channel, 1 on the channel's subchannels and on one more
      subchannel on each side (shared with the neighbouring channels), 0 elsewhere; for Euclid,
      an optional taper at the detector edges (`edge_zero_px`, `edge_ramp_px`);
    - with `apply_weight = true`, `1 / sqrt(|data| + 1e-4)`, which weights bright pixels down as
      shot noise would;
    - the model's `weight`, a function of data variables (for example `1/σ` from a stored variance
      plane).

## The mosaic

The mosaic coadds the corrected frames on the reference grid
([`Mosaicker.make_mosaic`][selfcal.pipeline.pipeline_wrapper.Mosaicker.make_mosaic]). From each
frame it subtracts the chunk offsets of every offset term, drawn on the detector by the
instrument's offset renderer, and the frame's scalar (added to the first map's offsets); offset
terms with a `coefficient` or a `basis` are evaluated and subtracted at every observation
([`BasisOffsetSubtractor`][selfcal.pipeline.model_eval.BasisOffsetSubtractor]). The offset of a
frame and chunk that the solve barely saw (covered fraction below `valid_chunk_thresh`, default
0.01) is set to zero first; the SPHEREx and Euclid spline renderers fill it in from the
neighbouring chunks.

It then accumulates, at every reference pixel, the values `d` of the frames that cover it:

```text
MEAN_MAP     = Σ w·d / Σ w
STD_MAP      = sqrt( Σ w·(d − MEAN_MAP)² / Σ w )                          make_std_map = true
SC_MEAN_MAP  = Σ w·d / Σ w  over the d with |d − MEAN_MAP| ≤ sigma · STD_MAP   apply_sigma_clipping = true
```

Here `w` is the mosaic's weight: the data-quality mask, the job's valid weight for the mosaic
(SPHEREx tapers it linearly toward the edges of the job's subchannels), the factor
`1 / sqrt(|data| + 1e-4)` when `[mosaic] apply_weight = true` (the default), and the square of the
model's `weight`. Each map is written with its summed weight (`MEAN_MAP_WEIGHT`, ...). For SPHEREx
the mosaic also coadds the band-centre and band-width maps over the same clipped observations into
`WAV_MEAN_MAP` and `WAV_STD_MAP`, the effective wavelength of each mosaic pixel and its spread, in
µm (top-level `wavelength_coadd`, on by default). The [mosaic schema](pipeline.md#mosaic-fits-schema)
lists every extension and header key.

The mosaic is not the sky map of the solve (`sky/<name>` in the cal file): it is a weighted mean of
corrected frames, without the sky damping and the solve's outlier rejection, and with its own
weights and clipping. The top-level `oversample` sets how finely the mosaic samples the
detector-plane maps (chunk maps, valid weights, rendered offsets): `oversample` x `oversample`
points per detector pixel. The mosaic stays on the reference grid, and the solve always samples at
one point per pixel.

## Scaling up

A solve holds one matrix row per observation, so its memory and time grow with the number of
frames and their pixels. The [Pipeline runbook](pipeline.md) covers the knobs; in short:

- **Parallelism.** `[calibration] max_workers` and `batch_size` set the processes that build the
  matrix rows and the frames each takes at a time, `apply_n_threads` (top level) the solver's
  threads, `[mosaic] max_workers`, `cache_batch_size` and `coadd_batch_size` the coadd's. With
  `[mosaic] cache_intermediate = true` the mosaic caches every corrected frame once for its later
  passes.
- **NVMe staging.** Before a `cal` or `mosaic` task reads the frames, the runner copies them to
  `<cache_dir>/reproj_nvme_<run_name>` on fast storage (at most `hdd_io_limit` concurrent reads
  from the slow disk) and deletes the copy at the end unless `keep_nvme = true`; the cal file
  records the frames' permanent paths. See [NVMe staging pattern](pipeline.md#nvme-staging-pattern).
- **Tiling.** A `[tiling]` table splits the reference grid into tiles: a grid with `overlap_px`, or
  an explicit list of boxes, which may overlap. Each tile takes the frames whose footprint centre
  falls inside it (or, with `frame_filter = "overlap"`, every frame that overlaps it) and is solved
  on its own; the [Fisher stitch](../reference/selfcal/pipeline/tiled.md) then merges the tile
  skies, every pixel becoming the mean of the tiles that cover it, weighted by their Fisher
  information. The stitched cal holds no per-frame offsets, so a tiled run makes no mosaic. Every
  tile picks its own gauge, which can leave seams; keep `apply_n_threads` the same for all tiles you
  stitch.
- **N-pass solve.** For spectral models, task `npass` alternates exact half-solves. Pass 1 (INIT)
  is the joint solve of task `cal`, tiled or not. SKY passes then solve the sky terms of every pixel
  in closed form, from moments summed over all tiles (one solve for the whole field, so no seams),
  and OFFSET passes refit every frame's polynomial offset and scalar against that one sky.
  `[passes]` sets the number of passes `n` and their order; `n = 1` reproduces task `cal` byte for
  byte. See [N-pass alternating solve](pipeline.md#n-pass-alternating-solve-task-npass).

## The absolute level

The solve cannot set the absolute sky level: a constant moves freely between the sky and the
per-frame scalars. For SPHEREx, the zodiacal-light anchor sets it from outside, channel by
channel. It fits every frame's mean level (its scalar plus the pixel-weighted mean of its chunk
offsets, [`compute_full_dc`][selfcal.zodi_anchor.compute_full_dc]) against a prediction of the
zodiacal light made with zodipy:

```text
frame level[k] = slope · prediction[k] + C        (straight-line fit, outliers clipped in time)
```

The intercept `C` is the constant to add to the sky; the slope, near 1 when the prediction follows
the frames, is a check. The anchor never rewrites the cal file or the mosaic: the fit goes to
`<run>/zodi_anchor/anchor_D<N>.h5` and is applied when a product is read
([`load_anchor`][selfcal.zodi_anchor.load_anchor],
[`load_anchored_mosaic`][selfcal.zodi_anchor.load_anchored_mosaic]). A cal config with
`[zodi] pred_dir` fits it after each channel's mosaic; the
[Zodiacal-light anchor](../tools/zodi-anchor.md) guide gives the full workflow. The uniform level
of a spectral line map is set the same way, from a reference region declared free of emission
([`selfcal.line_floor`](../reference/selfcal/line_floor.md)).

## Reproducibility

A run repeated on the same frames with the same config reproduces its products byte for byte. The
matrix rows and the per-pixel moments are assembled in batch order, whatever the number of worker
processes. The coadd adds its batches into the maps in batch order, so the mosaic depends on the
frames and the batch sizes but not on `max_workers`. The solver's threaded products are
deterministic for a given thread count (`apply_n_threads`); two thread counts give results that
differ at the level of float32 rounding, about 1e-6 of the values of a converged map.
`SELFCAL_PARALLEL_RMATVEC=1` selects a sequential transpose product whose result does not depend
on the thread count.

The maintainers check every change to the pipeline against this: the
[regression gates](../developer/gates.md) rerun fixed calibrations (a continuum and a spectral
SPHEREx solve, an end-to-end cal and mosaic, an N-pass probe, a Euclid recipe) and compare every
dataset with reference products, the *goldens*, byte for byte. The gates need the maintainers'
data; the [test suite](../developer/testing.md) runs anywhere.
