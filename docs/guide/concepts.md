# How selfcal works

This page explains what a selfcal run computes, and why, for readers who want to understand the
method before they configure a run. It links the guides with the details:
[The Python API](python-api.md) (every setting),
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
instrument from first principles. In its simplest form (the model `sc.continuum()`), the value of frame
`k` (one detector of one exposure) on pixel `P` of a common reference grid is modelled as

```text
data[k, P] = S[P] + O[k, c] + s[k] + noise        c: the chunk of the detector on which frame k saw P
```

`S` is the sky map, shared by all frames. `O` holds one unknown per frame and *chunk*, a region of
the detector such as one cell of an 8 x 8 grid. `s` holds one constant per frame, the *per-frame
scalar*: itself an offset term, whose one chunk is the whole detector. An offset in this sense is

- additive: it never multiplies the data (multiplicative effects such as gains are linearised
  around a previously solved sky; see the [limits](../bring_your_own_telescope.md#6-limits));
- constant over its chunk, unless the model gives it a known shape;
- per frame, or shared by a group of frames (all frames, the frames of one detector, of one night).

Once the offsets are known, selfcal subtracts them from every frame and coadds the corrected frames
into a mosaic.

## From exposures to a mosaic

```text
raw exposures                FITS files, or any format an instrument's exposure reader reads
   │   field.reproject       Reprojector.define_reference, Reprojector.run_reproject
   ▼
ref.fits                     the reference grid: a celestial WCS and a shape
reprojected/*.h5             one frame file per detector of each exposure
   │   field.calibrate       Calibrator.setup_lsqr, apply_lsqr, save_calibration
   ▼
calibration/cal_<stem>.h5    sky maps, offsets, coverage
   │   field.calibrate       Mosaicker.load_calibration, make_mosaic, save_mosaic
   │   or field.mosaic
   ▼
mosaic/mosaic_<stem>.fits    mean, standard-deviation and sigma-clipped mean maps
```

The paths are relative to the field's directory (`sc.Field(path, ...)`).

1. **Reproject** (`field.reproject`). The instrument's exposure layout says
   how to read a raw file and which detectors it holds.
   [`Reprojector.define_reference`][selfcal.pipeline.pipeline_wrapper.Reprojector.define_reference]
   loads `ref.fits` if it exists; otherwise it computes a WCS at the field's `pixel_scale` that
   contains every exposure (or, given a FITS file as `reference=`, one with that file's projection
   and pixel scale, re-centred and sized to the exposures) and writes it.
   [`Reprojector.run_reproject`][selfcal.pipeline.pipeline_wrapper.Reprojector.run_reproject]
   resamples every detector of every exposure onto that grid
   ([`batch_reproject`][selfcal.io.reprojection.batch_reproject], with the `reproject` package)
   and writes one *frame file*, `exp_<exposure>_det_<detector>.h5`: the values on a box of the
   reference grid (`ref_coords`), the data-quality bit mask (if any) and the detector coordinates
   of every box pixel (`sub_mapping`), which tell the solver where on the detector each value was
   recorded. Data that are not images with a WCS can be written as frame files directly
   ([`write_frame`][selfcal.io.frames.write_frame]).
2. **Calibrate** (`field.calibrate`), once per [job](glossary.md#job) the action names: the chunks
   of the primary chunk map one solve covers (`sc.Job("All")`: every chunk, the one job of a camera;
   a SPHEREx [channel](glossary.md#channel) or subchannel window). [`Calibrator.setup_lsqr`][selfcal.pipeline.pipeline_wrapper.Calibrator.setup_lsqr]
   ([`setup_lsqr`][selfcal.core.system.setup_lsqr]) reads the frames in parallel worker processes
   and builds the sparse system: one row per observation, plus the rows of the priors.
   [`Calibrator.apply_lsqr`][selfcal.pipeline.pipeline_wrapper.Calibrator.apply_lsqr]
   ([`apply_lsqr`][selfcal.core.solve.apply_lsqr]) solves it with [LSQR or LSMR](glossary.md#lsqr),
   and [`Calibrator.save_calibration`][selfcal.pipeline.pipeline_wrapper.Calibrator.save_calibration]
   writes the [cal file](glossary.md#cal-file), which [`CalFile`][selfcal.io.calfile.CalFile] reads.
   A cal file that exists is reused when it was made by the same inputs (its sidecar,
   `<cal file>.json`, records them) and refused otherwise: give the recipe another `name`, or pass
   `overwrite=True`, to solve again.
3. **Mosaic** (`field.calibrate` after the solve, when the recipe has a `Coadd`, or `field.mosaic`
   from an existing cal file).
   [`Mosaicker.make_mosaic`][selfcal.pipeline.pipeline_wrapper.Mosaicker.make_mosaic] subtracts
   each frame's offsets and coadds the frames
   ([`run_coadd_schedule`][selfcal.core.coadd.run_coadd_schedule]), and
   [`Mosaicker.save_mosaic`][selfcal.pipeline.pipeline_wrapper.Mosaicker.save_mosaic] writes the
   FITS file.

A job's products are named by its [stem](glossary.md#stem), `<frame_tag>_<job>_<name>`: the
[frame tag](glossary.md#frame-tag) (the instrument's tag, then the primary chunk map's), the job
name and the recipe's `name`, for example `MyCam_Chunks8x8_All_v1` for a camera tagged `MyCam` on
its default 8 x 8 grid and a recipe named `v1`. A field's
actions run these steps through the run engine; the same classes can also be used directly
([Using the pipeline](../developer/architecture.md#using-the-pipeline)). The
[Quickstart](../getting-started/quickstart.md) runs all three on simulated exposures.

## The model

For every *observation* `i`, one value of one frame on one pixel `P` of the reference grid,
recorded at a known position on the detector, the solver fits

```text
data_i = Σ_j S_j[P] · c_j(v_i)  +  Σ_m Σ_k O_m[g_m(frame), chunk_m(i), k] · φ_mk(v_i)  +  noise_i
```

- **Sky terms** `S_j` (`sc.Sky(name, times=c)`, [`SkyTerm`][selfcal.models.spec.SkyTerm]): a map
  on the reference grid, shared by every frame, times a known *coefficient* `c_j(v)`: `c = 1` for a
  plain sky map (one value per pixel), a template of the wavelength for the map of an emission
  feature, a sine of the time for a seasonal term. A coefficient is any importable Python function
  of data variables or a built-in shape (`sc.template(file)`, `sc.gaussian(...)`, `sc.linear(...)`;
  the SPHEREx 3.29 µm feature, `terms.pah_3p29()` of `selfcal_scripts/spherex/terms.py`, is a
  Gaussian of the wavelength).
- **Offset terms** `O_m` (`sc.Offsets(name, on=m, per=g)`,
  [`OffsetTerm`][selfcal.models.spec.OffsetTerm]): an offset per chunk of chunk map `m`, shared by
  the frames of a group `g_m`: `per="frame"` (each frame its own), `per="all"` (one for all frames:
  a pattern fixed to the detector) or `per=` a [frame variable](glossary.md#frame-variable) such
  as `"detector"` (the frames with equal values share one). `times=φ` multiplies the
  offset by a known function of data variables (a pattern times the detector temperature);
  `basis=` with `n=` functions gives `n` unknowns per chunk (a gradient across the detector in every
  frame; a polynomial in a coordinate, as the polynomials of that coordinate). With neither, `φ = 1`:
  the plain chunk offset.
- **Per-frame scalar**: one additive constant per frame, the offset term
  `sc.Offsets("scalar", on="detector")` (the map `"detector"` is the whole detector as one chunk).
  A model has one only if it lists it: `sc.continuum` and `sc.spectral` end with it, `sc.two_block`
  has none.
- **Data variables** `v`: named quantities with one value per observation, which every function of
  the model reads by name ([`selfcal.models.variables`](../reference/selfcal/models/variables.md)):
  the built-ins `det_x`, `det_y` (detector position), `sky_x`, `sky_y` (reference pixel),
  `frame`, `value` (the stored value) and `flags` (its flag bits); the instrument's detector maps (SPHEREx: the band centre `BC` and band width `BW`, also
  called `wavelength` and `bandwidth`) and per-frame values (`exposure`, `detector`); and the
  model's own `variables={...}` (header keywords, maps, stored planes, functions; see
  [Data variables](../bring_your_own_telescope.md#data-variables)).

The solve seeks the unknowns (the pixels of every sky map and the offsets) that minimise

```text
Σ_i  w_i² · (data_i − model_i)²   +   Σ_r  (prior row r)²
```

where `w_i` is the observation's weight (see [Masks, outliers and weights](#masks-outliers-and-weights))
and the prior rows are linear equations on the unknowns that settle what the data leave free (see
[What the data cannot tell apart](#what-the-data-cannot-tell-apart)). An offset term's hard
constraints (`exclude=`) restrict its unknowns exactly instead of adding rows.
[`SystemLayout`][selfcal.core.layout.SystemLayout] orders the unknowns: the sky maps, then the
offset terms.

Every run builds one [`ModelSpec`][selfcal.models.spec.ModelSpec]. The presets (`sc.continuum`,
`sc.spectral`, `sc.two_block`) build common models from a few settings, and `sc.Model` spells any
model out term by term; a preset prints as what it is:

```pycon
>>> sc.continuum(smooth=0.1)
Model(offsets=(Offsets(smooth=0.1, exclude=Poly(0)), Offsets('scalar', on='detector')))
>>> sc.continuum(smooth=0.1) == sc.Model(
...     sky=[sc.Sky()],                                   # no coefficient: c = 1
...     offsets=[sc.Offsets(smooth=0.1,                   # per frame and chunk of the primary map,
...                         exclude=sc.Poly(0)),          #   smooth, no constant in any frame
...              sc.Offsets("scalar", on="detector")])    # the per-frame scalar
True
```

A preset is a plain function that returns a `Model`, and so is a calibration variant of your own:
a new variant needs no change to the engine.
[Bring your own telescope](../bring_your_own_telescope.md#2-the-model-is-yours) describes every
kind of term, variable and prior, with worked examples.

## Chunk maps and chunk axes

A *chunk map* is an integer image on the detector grid: the value of a pixel is the id of its
chunk, and `-1` marks pixels in no chunk. The model names its chunk maps
(`sc.Model(chunk_maps={name: sc.ChunkMap(fn, **params)})`, [`ChunkMap`][selfcal.models.model.ChunkMap]):
each is a function of the instrument's detector geometry that returns the partition, a
[`Chunks`][selfcal.instruments.base.Chunks], and the run engine evaluates it for the run. The first
map is the primary one; a model that names none gets the instrument's default maps. An offset term
that names no map uses the primary one, a [job](glossary.md#job) names its chunks on it, and the
name `"detector"` makes the whole detector one chunk.
An observation's offset is read at its detector position by bilinear interpolation of the chunk
map, so an observation within a pixel of a chunk edge shares its offset between the two chunks.

*Chunk axes* ([`ChunkAxes`][selfcal.models.offset_structure.ChunkAxes]) are the named coordinates of
the chunks: for each axis, the value of every chunk along it and the image direction in which
neighbouring chunks differ in it. Priors and offset terms are written in axes ("smoothness along
`column`", "a degree-3 polynomial along `subchannel`, one per `column`"), so a recipe runs on any
instrument whose chunk map declares the axes it needs. A map also names its default adjacency
axes and its *spectral axis* (along which the wavelength changes; none for a broadband instrument).

| instrument | chunk maps | axes |
| --- | --- | --- |
| `sc.Camera` | default `grid`: an `ny` x `nx` grid of rectangles (`chunks=(ny, nx)`), id `row · nx + col` | `row`, `col`; adjacency along both |
| `sc.SPHEREx` | no default; the recipes' maps (`selfcal_scripts/spherex/chunks.py`): `subchannel` (primary): [subchannels](glossary.md#subchannel), the arcs of nearly constant wavelength of the [LVF](glossary.md#lvf), each cut into `num_col` vertical columns, id `subchannel · num_col + column`; `detector_column`: one chunk per detector column (a term on it shared by every frame, `per='all'`, is a detector-fixed column pattern); `readout`: one chunk per read-out channel of the detector | `subchannel` (spectral axis), `column` (default adjacency); `detector_column`; `readout` |
| `sc.Euclid` | default `grid` (primary): `chunks` x `chunks` squares (default 40); `col_strips`, `row_strips`: vertical and horizontal stripes; `col_tilt`, `row_tilt`: stripes whose `strip` axis is the spectral axis | `row`, `col`; `strip`, `all` |

Chunk size trades flexibility against noise: smaller chunks absorb finer offset structure, but
fewer pixels constrain each unknown. For SPHEREx, `num_col` of the subchannel map is the main
knob of this kind (see [Calibration pipeline tuning](pipeline.md#calibration-pipeline-tuning)). The
solve treats an offset as constant over its chunk (times its known functions, if any); the mosaic
may draw it more smoothly, through the chunk map's [offset renderer](glossary.md#offset-renderer)
(for the SPHEREx subchannel map, a mean-preserving spline across the arcs and columns).

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
  coefficients are nearly degenerate at every pixel: when the coefficients of two terms, sampled
  over the window's wavelengths, have a normalised inner product above about 0.7 in absolute
  value, expect cross-talk between their maps.

The priors, and the rows each adds to the system:

| prior | setting | rows |
| --- | --- | --- |
| adjacency smoothing | `sc.Offsets(smooth=w, smooth_along=axes)` (0 by default; the presets' `smooth`, default 0.1) | per frame and pair of neighbouring chunks `a`, `b`: `w · (O[a] − O[b]) = 0` |
| sky damping | `sc.Sky(damping=d)` (default none, `d` = 0) | per covered pixel: `sqrt(d · coverage) · S[P] = 0`, with the pixel's [coverage](glossary.md#coverage) |
| offset damping | `sc.Offsets(damping=d)` | per covered unknown: `sqrt(d · coverage) · O = 0` |
| user priors | `sc.Model(priors=[...])`: a ready-made prior of [`selfcal.priors`](../reference/selfcal/priors.md) (`frame_smoothness`, `sky_smoothness`, `toward`) or your own rows, `sc.Prior(fn, terms)` | the linear rows the function returns |

An offset term may also carry a **hard constraint**, which adds no row:
`sc.Offsets(exclude=sc.Poly(k, along=axis))` removes from the term every polynomial of degree `k`
or less along the chunk axis, per line of the other axes and per group of frames (`along` may be
omitted on a map with one axis). The solver imposes it exactly, by reparametrisation: it solves
only for unknowns that satisfy it
([`ColumnProjector`][selfcal.core.projector.ColumnProjector]), so there is no weight to choose,
and what the term cannot carry goes where the data place it, to other terms or the sky.
`exclude=sc.Poly(0)` needs no `along` on a map with several axes either: the term carries no
constant over all of a group's chunks. In the chunk term of `sc.continuum` and `sc.spectral` it
moves each frame's mean offset into the frame's per-frame scalar, so the
chunk offsets carry only structure within the frame. The constraint acts on the unknowns of the
solve: a chunk that a frame never observed drops out of it, unless a prior row, such as a
smoothness row, makes that chunk an unknown.

What each settles:

- The **sky damping** pulls the maps toward zero: of all the levels the data allow, it prefers the
  one that makes the maps smallest, a level with no physical meaning. Like any Tikhonov term it also
  shrinks real structure toward zero, by an amount that grows with the weight.
- The **adjacency smoothing** makes steps between neighbouring chunks costly, so smooth structure
  goes to the sky rather than to offset ramps; the sky damping pushes the other way. An offset
  that should be a polynomial in a coordinate is written as one exactly, a `basis=` of that
  coordinate's polynomials (`n = degree + 1`), with no weight.
- `sc.Fit(damp=)` damps every unknown (default 0). With a limited number of iterations
  (`sc.Fit(iterations)`), LSQR also stops before it has
  moved far along weakly constrained directions, where the result then depends on the
  [warm start](glossary.md#warm-start).

Hard constraints settle a direction without any row or weight, by removing it from the unknowns
(a reparametrisation the solver applies at every iteration):

- `sc.Offsets(exclude=sc.Poly(k, along=axis))`: the term carries no polynomial of degree `k` or less
  along the chunk axis (for a pattern shared by every frame, the parts that duplicate per-frame
  terms).
- `sc.Offsets(exclude=sc.SkySignature(k))`: the term carries no part of its response to the smooth
  sky fields, the polynomials of degree 1 to `k` on the sky grid. A sky plane appears in every frame
  as a slope along each detector direction, so a per-frame slope term absorbs it nearly whole; the
  plane's signature in the slopes follows each frame's pointing and orientation, which a slope of the
  instrument does not. Without that part, the sky field is fitted by the sky. The constraint is
  measured in the data's metric (each coefficient weighted by its information, so coefficients the
  data barely measure cannot take it up for the rest), and what the term holds the same way in every
  frame (a pattern fixed on the detector) stays free. Whatever the true per-frame slopes do hold
  along the rest of the signature (the survey-mean gradient of a foreground at each position, say)
  goes into the map as that field. The signature comes from the system itself (the deflation's
  setup), so the solve needs `sc.Fit(deflate=sc.Deflate(...))`, and the term must be fitted per frame
  or per group of frames. `sc.SkySignature('all')` takes every pattern of the deflation (its cosines
  too) but the constant, instead of the polynomials up to a degree.
- `sc.Offsets(exclude=sc.Overlap('a', 'b'))`: the term carries no detector pattern that the offset
  terms `a` and `b` can also make with the same coefficients in every frame (their fixed patterns).
  Two terms that can make the same pattern leave the data unable to say which holds it; with the
  exclusion it goes to `a` and `b`. The shared patterns are the intersection of the two terms'
  pattern spaces over the detector pixels the job's observations reach (its window and the one pixel
  around it that bilinear sampling touches), computed from the terms' chunk maps and their functions
  at each pixel centre: a term with one offset per detector pixel gives up every fixed pattern of the
  named terms, a term with one offset per detector column only those constant along each column. The
  named terms must read detector-plane variables only (`det_x`, `det_y`, detector maps and variables
  derived from them). By default every offset counts alike in the constraint. A term with many offsets
  can include some the data barely see (detector pixels that only the corners of the observations'
  bilinear sampling reach): those can take large values that cancel a shared pattern the other offsets
  hold, at almost no cost to the fit, so the constraint holds only on paper and the solve wanders along
  that direction. `sc.Overlap('a', 'b', weights='information')` measures the shared patterns with each
  offset's information (the sum of its squared weighted entries, as the solver scales its columns): an
  offset the data barely see then weighs almost nothing in the constraint and an offset without data
  drops out. The constraint stays exact and needs no threshold.
- Offsets shared by many frames make one block of the deflation's offset solve. Up to 16,384 such
  columns are eliminated directly by a Schur complement; with more (one offset per detector pixel,
  say), the complement no longer fits in memory. `sc.Deflate(offsets=(...))` names the offset terms
  whose response the deflation solves; the others keep a zero response and stay in the space. Listing
  the per-frame terms alone leaves the shared ones out: the patterns' offsets are solved directly frame
  by frame, and what the shared terms take of the patterns is left to the iterations. A shared offset
  fixed on the detector cannot follow a sky pattern that crosses the detector differently in each
  frame, so that share is small except along the sky's constant level.
- `sc.Sky(exclude=shapes)` and `sc.Deflate(exclude=sc.Degenerate(below=))`: the map has no
  component along given (or automatically found) shapes; what the data place there goes to the
  other terms.

### Comparing solutions

Two solves that fit the data equally well can differ in their raw arrays along these directions.
Compare quantities that a change of gauge leaves alone:

- the total offset of each frame and chunk, the sum of the offset terms there (a chunk offset plus
  the frame's scalar), rather than either part;
- offsets with each frame's mean over chunks and each chunk's mean over frames removed, which also
  removes a pattern common to every frame, such as the detector ramp of a sky gradient; the
  quickstart compares recovered and injected offsets this way (`remove_gauge` in
  [examples/quickstart/inspect_results.py](https://github.com/ThomasLiii/PySelfCal/blob/main/examples/quickstart/inspect_results.py));
- sky maps after removing their mean, or after an external anchor has set their level;
- maps of terms with a coefficient only where their [separability](glossary.md#separability) or
  [Fisher information](glossary.md#fisher) is high enough.

## Masks, outliers and weights

- **Data-quality masks.** A sample is dropped when any bit of its data-quality mask is set, except
  the bits listed in `ignore_flags` (`sc.Fit(use_mask=True, ignore_flags=[...])`). The mask comes
  from the exposure's DQ extension and is resampled bit by bit during the reprojection. `sc.Fit`
  and `sc.Coadd` each have their own `use_mask` and `ignore_flags`, so the solve can leave out
  pixels that the mosaic uses, or the reverse.
- **Outlier rejection in the solve.** With `sc.Fit(clip=5.0)`, every sample is scored against its
  frame's median, in units of `1.4826 · MAD` (the median absolute deviation), and samples scoring
  above the threshold are left out of the solve
  ([`find_outliers`][selfcal.geometry.map_helper.find_outliers]). The score uses the frame's own
  values, before any model is subtracted, so it removes compact bright sources and artefacts. When
  a frame's brightness changes strongly across the detector (SPHEREx sees a different wavelength in
  every subchannel), a [grouped clip](glossary.md#grouped-clip) scores each sample within its
  group instead: `sc.Clip(5.0, per="chunk")` (its chunk of the primary map),
  `sc.Clip(5.0, per=sc.ChunkGroups.along("subchannel"))` (the chunks of its subchannel; for
  SPHEREx, binned by wavelength) or `sc.Clip(5.0, variable=..., edges=[...])` (bins of a data
  variable, by default the instrument's wavelength map). The N-pass passes clip the same way
  (`sc.Passes(sky_clip=..., offset=sc.Refit(clip=...))`). Samples whose weight, coefficient or
  basis value is not finite are dropped as well.
- **Coverage floor.** `sc.Fit(min_observations=2)` leaves out of the solve the sky pixels that fewer
  than 2 observations cover, counted after the masks, the clip if any and the weights. A sky pixel
  with a single observation fits that observation exactly and informs no other unknown, so it adds
  an unknown and nothing else. The matrix build counts the observations as it collates the rows and
  zeroes those pixels' rows as it places them (values and right-hand side), with no extra pass over
  the frames; the pixels' columns stay without data (coverage 0, value 0, outside the deflation's
  patterns), so every other unknown gets the solution without those rows. The mosaic still covers
  them. Off by default. `sc.Offsets(..., min_observations=2)` does the same for one offset term with
  one unknown per chunk (a chunk a single observation sees: for a term per detector pixel, a pixel
  next to masked ones): judged on the same counts, its rows zeroed in the same pass, its offsets left
  at zero, which the mosaic then applies to those observations. A hard constraint on the term
  (`exclude=`) holds on the observed offsets alone: an offset without data left in the constraint
  would take whatever value cancels it. Both floors judge the counts the build collates, and those
  counts stand: a zeroed row still counts in the coverage of its other columns.
- **Per-frame hooks.** `sc.Fit(raw_frame_hook=...)` runs a function on every frame right after it
  is read, `sc.Fit(frame_hook=...)` after its weights are computed, and `sc.Coadd(frame_hook=...)`
  the same in the mosaic; for example Euclid's star mask
  (`selfcal.instruments.euclid.hooks.StarMask`, before the fit) and residual mask (`ResidualMask`,
  after).
- **Weights.** An observation's weight `w_i` multiplies its row of the system, so the fit weights
  the observation by `w_i²`. It is the product of
    - the data-quality mask (0 or 1);
    - the job's [valid weight](glossary.md#valid-weight) on the detector, from the instrument: by
      default 1 on the chunks of the primary map inside the job's window, 0 elsewhere (every chunk
      for `sc.Job("All")`; for a SPHEREx channel job, the channel's subchannels and one more
      subchannel on each side, shared with the neighbouring channels); for Euclid, an optional taper
      at the detector edges (`edge_zero_px`, `edge_ramp_px`). It is sampled bilinearly at the
      observation's detector position, like every detector-plane quantity, so an observation whose
      four sampled pixels straddle the window's edge gets the share inside and reads the offsets of
      the pixels just outside; with `sc.Fit(window_edge='inside')` such an observation leaves the
      solve, and the solve reads no detector pixel outside the window (the coadd is unchanged);
    - with `shot_noise_weights=True` (`sc.Fit`), `1 / sqrt(|data| + 1e-4)`, which weights bright
      pixels down as shot noise would;
    - the model's `weight` (`sc.Model(weight=...)`), a function of data variables (for example
      `1/σ` from a stored variance plane).

## The mosaic

The mosaic coadds the corrected frames on the reference grid
([`Mosaicker.make_mosaic`][selfcal.pipeline.pipeline_wrapper.Mosaicker.make_mosaic]). From each
frame it subtracts the chunk offsets of every offset term (the per-frame scalar among them), drawn
on the detector by the chunk map's offset renderer; offset terms with a `coefficient` or a `basis`
are evaluated and subtracted at every observation that carries weight, as in the solve
([`BasisOffsetSubtractor`][selfcal.pipeline.model_eval.BasisOffsetSubtractor]; the frame's other pixels never reach
the coadd). The offset of a
frame and chunk that the solve saw only in part (its covered fraction, counted over the chunk's
pixels that the job can observe, below `sc.Coadd(min_chunk_coverage=0.01)`) is set to zero first.
The SPHEREx subchannel renderer and the Euclid grid spline fill such a chunk in from the frame's
other chunks; where nothing fills it, its pixels in that frame are left out of the coadd.

It then accumulates, at every reference pixel, the values `d` of the frames that cover it:

```text
MEAN_MAP     = Σ w·d / Σ w
STD_MAP      = sqrt( Σ w·(d − MEAN_MAP)² / Σ w )                          sc.Coadd(std=True)
SC_MEAN_MAP  = Σ w·d / Σ w  over the d with |d − MEAN_MAP| ≤ sigma · STD_MAP   sc.Coadd(clip=sigma)
```

Here `w` is the mosaic's weight: the data-quality mask, the job's valid weight for the mosaic
(by default 1 on the chunks inside the job's strict set, its window unless it names one: for a
SPHEREx channel job, the channel's own subchannels), the factor
`1 / sqrt(|data| + 1e-4)` with `sc.Coadd(shot_noise_weights=True)` (off by default), and the
square of the model's `weight`. Each map is written with its summed weight (`MEAN_MAP_WEIGHT`, ...). For SPHEREx
the mosaic also coadds the band-centre and band-width maps over the same clipped observations into
`WAV_MEAN_MAP` and `WAV_STD_MAP`, the effective wavelength of each mosaic pixel and its spread, in
µm (`sc.Coadd(instrument_maps=True)`, the default). The
[mosaic schema](pipeline.md#mosaic-fits-schema) lists every extension and header key.

The mosaic is not the sky map of the solve (`sky/<name>` in the cal file): it is a weighted mean of
corrected frames, without the sky damping and the solve's outlier rejection, and with its own
weights and clipping. `sc.Coadd(oversample=n)` sets how finely the mosaic
samples the detector-plane maps (chunk maps, valid weights, rendered offsets): `oversample` x `oversample`
points per detector pixel. The mosaic stays on the reference grid, and the solve always samples at
one point per pixel.

## Scaling up

A solve holds one matrix row per observation, so its memory and time grow with the number of
frames and their pixels. The [Pipeline runbook](pipeline.md) covers the knobs; in short:

- **Parallelism.** `sc.Compute(workers=)` sets the processes that build the matrix rows and
  `sc.Numerics(batch=)` the frames each takes at a time; `sc.Numerics(threads)` the solver's
  threads; `sc.Compute(coadd_workers=)` and `sc.Numerics(mosaic_batch=, coadd_batch=)` the
  coadd's. With `sc.Compute(cache_frames=True)` (the default) the mosaic caches every corrected
  frame once for its later passes.
- **NVMe staging.** Before an action reads the frames, it copies them to
  `<scratch>/reproj_nvme_<field name>` on fast storage (`sc.Compute(scratch)`; at most `io_limit`
  concurrent reads from the slow disk) and deletes the copy at the end unless `keep_staged=True`;
  the cal file records the frames' permanent paths. Each frame is copied atomically, and an action only stages into, or deletes, a directory
  it created (marked by a `.selfcal-staging.json` file). See
  [NVMe staging pattern](pipeline.md#nvme-staging-pattern).
- **Tiling.** `calibrate(tiles=sc.Tiles(...))` splits the reference grid into
  tiles: a grid (`grid=(ny, nx)`, with `overlap=` pixels), or explicit boxes (`boxes={name: (y0, y1,
  x0, x1)}`), which may overlap. Each tile takes the frames whose footprint centre falls inside it
  (or, with `assign="overlap"`, every frame that overlaps it) and is solved
  on its own; the [Fisher stitch](../reference/selfcal/pipeline/tiled.md) then merges the tile
  skies, every pixel becoming the mean of the tiles that cover it, weighted by their Fisher
  information. The stitched cal holds no per-frame offsets, so a tiled run makes no mosaic. Every
  tile picks its own gauge, which can leave seams; keep `sc.Numerics(threads)` the same for all
  tiles you stitch.
- **N-pass solve.** For spectral models, `calibrate(passes=sc.Passes(n))` alternates exact
  half-solves. Pass 1 (INIT) is the joint solve of a plain `calibrate`, tiled or
  not. SKY passes then solve the sky terms of every pixel
  in closed form, from moments summed over all tiles (one solve for the whole field, so no seams),
  and OFFSET passes refit the offsets against that one sky: the model's own offset terms, whatever
  they are (a polynomial, per-chunk values, terms times known functions, a per-frame scalar), under
  their hard constraints, every frame solved directly from its own rows. Neither kind of pass
  builds the joint matrix.
  `sc.Passes` sets the number of passes `n` and their order; `n = 1` reproduces the plain
  calibration byte for byte. See [N-pass alternating solve](pipeline.md#n-pass-alternating-solve).

## The absolute level

The solve cannot set the absolute sky level: a constant moves freely between the sky and the
frames' levels. For SPHEREx, the zodiacal-light anchor sets it from outside, channel by
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
[`load_anchored_mosaic`][selfcal.zodi_anchor.load_anchored_mosaic]).
`spherex.zodi_anchor(result, predictions)` fits it after a calibration; the
[Zodiacal-light anchor](../tools/zodi-anchor.md) guide gives the full workflow. The uniform level
of a spectral line map is set the same way, from a reference region declared free of emission
([`selfcal.line_floor`](../reference/selfcal/line_floor.md)).

## Reproducibility

A run repeated on the same frames with the same recipe reproduces its products byte for byte. The
matrix rows and the per-pixel moments are assembled in batch order, whatever the number of worker
processes. The coadd adds its batches into the maps in batch order, so the mosaic depends on the
frames and the batch sizes but not on the number of workers. The solver's threaded products are
deterministic for a given thread count; two thread counts give results that differ at the level of
float32 rounding, about 1e-6 of the values of a converged map. That is why the thread count and
the batch sizes belong to the recipe (`sc.Numerics`), whose defaults are production's, while the
machine settings (`sc.Compute`) never change a byte. `sc.Numerics(rmatvec_threads=1)` selects a
sequential transpose product whose result does not depend on the thread count.

The maintainers check every change to the pipeline against this: the
[regression gates](../developer/gates.md) rerun fixed calibrations (a continuum and a spectral
SPHEREx solve, an end-to-end cal and mosaic, an N-pass probe, a Euclid recipe, the production
recipe, a tiled solve) and compare every
dataset with reference products, the *goldens*, byte for byte. The gates need the maintainers'
data; the [test suite](../developer/testing.md) runs anywhere.
