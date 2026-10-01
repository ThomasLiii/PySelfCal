# Glossary

The terms used in selfcal's documentation, run configs and code, in alphabetical order. Each entry
links to the page that explains or implements it; [How selfcal works](concepts.md) puts them
together.

## Adjacency regularisation { #adjacency }

A prior on an offset term: in every frame, each pair of neighbouring chunks gets a row
`reg_weight · (O[a] − O[b]) = 0`. Neighbours are chunks that touch on the detector and differ only
along one of the term's `adjacency` axes (default: the chunk map's own). The rows are added only
with `[calibration] offset_regularization = true`. See
[What the data cannot tell apart](concepts.md#what-the-data-cannot-tell-apart).

## Band centre, band width { #bc-bw }

`BC` and `BW`, the SPHEREx detector maps of the centre and the width of the filter passband at
every detector pixel, read from the spectral calibration files in `[instrument] calib_dir`. They are
data variables, also called `wavelength` and `bandwidth`, and the mosaic coadds them into its
wavelength maps.

## Basis { #basis }

`n` known functions of data variables attached to an offset term (`basis = { variable, function,
params, n }`). The term's unknowns become `n` coefficients per group and chunk, and an
observation's offset is their sum weighted by the functions' values there; a function of `det_x`
and `det_y`, for example, gives every frame its own gradient across the detector. See
[The model](concepts.md#the-model) and
[Terms, weights, priors](../bring_your_own_telescope.md#terms-weights-priors).

## Cal file { #cal-file }

`calibration/cal_<stem>.h5`, the product of a solve: the map of every sky term with its coverage
and Fisher information, the offsets of every frame and chunk for each offset term, the per-frame
scalars and the list of frame files. Read it with [`CalFile`][selfcal.io.calfile.CalFile]; the
[schema](pipeline.md#cal_h5-schema-multi-chunk-map) lists its datasets. See
[From exposures to a mosaic](concepts.md#from-exposures-to-a-mosaic).

## Capability { #capability }

A tag an instrument declares (SPHEREx: `wavelength`, `spectral_axis`, `subchannel`) and a mode can
require. The spectral modes require `wavelength`, so a run that pairs them with an instrument that
does not declare it stops with an error before any work.

## Channel { #channel }

SPHEREx: one of the `num_ch` wavelength bands across a detector, each `num_sub`
[subchannels](#subchannel) wide. Its edges come from the 17 channels per detector of the SPHEREx
channel table, interpolated (`num_ch` must be a multiple of 17; 34 splits each table channel in
two). Channels are numbered from 1 and are the usual SPHEREx jobs (`[instrument] channels` or
`channel_range`; job name `Ch<n>`).

## Chunk { #chunk }

A region of the detector that carries its own offset unknowns, one per frame (or group of frames)
and basis function. The chunks are the values of a [chunk map](#chunk-map). See
[Chunk maps and chunk axes](concepts.md#chunk-maps-and-chunk-axes).

## Chunk axes { #chunk-axes }

The named coordinates of a chunk map's chunks
([`ChunkAxes`][selfcal.models.offset_structure.ChunkAxes]): for each axis, the value of every chunk
along it and the image direction in which neighbouring chunks differ. Priors and modes name axes
(`row` and `col` for `grid`, `subchannel` and `column` for SPHEREx), so they work on any instrument
that declares them. A map also names its adjacency axes, its *spectral axis* (along which the
wavelength changes) and its *group axis* (one polynomial per value in a [polybasis](#polybasis)
term). See [Chunk maps and chunk axes](concepts.md#chunk-maps-and-chunk-axes).

## Chunk map { #chunk-map }

An integer image on the detector grid whose value at each pixel is the id of its chunk (`-1`: no
chunk). An instrument defines one or more by name ([`ChunkMap`][selfcal.instruments.base.ChunkMap])
and marks one as primary; the name `"detector"` makes the whole detector one chunk. Examples: the
`grid` rectangles, the SPHEREx `subchannel` arcs cut into columns, the Euclid stripes. See
[Chunk maps and chunk axes](concepts.md#chunk-maps-and-chunk-axes).

## Coadd { #coadd }

To combine the corrected frames into one map, pixel by pixel, as weighted means; the
[mosaic](#mosaic) is the coadd. See [The mosaic](concepts.md#the-mosaic).

## Coefficient { #coefficient }

A known function of data variables that multiplies a term: `c_j(v)` of a sky term, `φ(v)` of an
offset term (`coefficient = { variable, function, params }`). `function` is a built-in
(`template`, a tabulated function; `gaussian`; `linear`), any importable `"package.module:name"`,
or omitted (the variable itself); `coefficient = { catalog = "<name>" }` takes a named coefficient
of the instrument (SPHEREx: `pah_3p29`). A term without one has `c = 1`. See
[The model](concepts.md#the-model).

## Coverage { #coverage }

The number of observations of an unknown: per sky pixel (`sky_coverage/<name>` in the cal file)
and per frame and chunk (`offset_coverage/map_<m>`, with `offset_coverage_frac/map_<m>` the
fraction of the chunk's pixels the frame covered). The damping rows are weighted by it, and the
mosaic sets to zero the offset of a frame and chunk whose covered fraction is below
`valid_chunk_thresh`.

## Damping { #damping }

A Tikhonov prior that pulls unknowns toward zero. Sky terms, with
`[calibration] weighted_damping = true`: one row `sqrt(damp_weight · coverage) · S[P] = 0` per
covered pixel, with `damp_weight` for the first term and `damp_weight_line` (default 3 x
`damp_weight`) for the others, unless a term sets its own `damp_weight`. Offset terms: `damp`, with
rows of the same form. `[lsqr] damp` is the solver's own damping of every unknown. See
[What the data cannot tell apart](concepts.md#what-the-data-cannot-tell-apart).

## Data-quality mask { #dq-mask }

The integer bit mask of an exposure (`dq_ext`), resampled bit by bit onto the reference grid and
stored in the frame file. With `apply_mask = true` a sample with any bit set is dropped, except the
bits in [`ignore_list`](#ignore-list). See
[Masks, outliers and weights](concepts.md#masks-outliers-and-weights).

## Data variable { #data-variable }

A named quantity with one value per observation, which the functions of a model read by name
([`selfcal.models.variables`](../reference/selfcal/models/variables.md)). Sources: the built-ins
`det_x`, `det_y`, `sky_x`, `sky_y` and `frame`; the instrument's detector maps and frame variables;
and the model's own `[model.variables]` (`header`, `per_frame`, `detector`, `sky`, `sky_cal`,
`layer`, `function`, `frame_function`). See [The model](concepts.md#the-model) and
[Data variables](../bring_your_own_telescope.md#data-variables).

## Detector { #detector }

One detector array of the instrument. Each detector of each exposure becomes its own frame, whose
file name and `detector` frame variable hold the detector's index. A SPHEREx run calibrates one of
the six detectors (`[instrument] detector`); a Euclid NISP exposure holds 16.

## Detector map { #detector-map }

A map on the detector grid, sampled at every observation's detector position: a data variable with
one value per detector pixel. An instrument provides its own (SPHEREx: [`BC`, `BW`](#bc-bw)); a
model adds more with `{ detector = ... }` in `[model.variables]`.

## Exposure { #exposure }

One raw input file, holding one or more detector images. The instrument's exposure layout says
which extensions (or other entries) hold the science values and the data-quality masks, and which
reader parses them (default: FITS extensions). The `exposure` frame variable holds the exposure
index of each frame's file name; task `reproject` numbers the exposures in sorted order. See
[From exposures to a mosaic](concepts.md#from-exposures-to-a-mosaic).

## Fisher information { #fisher }

For a sky term at one pixel, `Σ (w · c)²` over the pixel's observations: the diagonal of the
normal equations, large where the term is well measured. The cal file stores it per term
(`sky_fisher/<name>`) and the tile [stitch](#stitch) weights with it. When a sky term has a
coefficient, the cal file also records a recommended threshold (`line_fisher_threshold`, default
10) for masking the maps when they are read
([`apply_line_fisher_mask`][selfcal.core.system.apply_line_fisher_mask]).

## Frame { #frame }

One detector of one exposure, resampled onto the reference grid: the unit that the offsets and the
per-frame scalar belong to. It is stored as a [frame file](#frame-file). See
[The problem](concepts.md#the-problem).

## Frame file { #frame-file }

`reprojected/exp_<exposure>_det_<detector>.h5`, the solver's input: a frame's values on a box of
the reference grid (`sub_data`, placed by `ref_coords`), its data-quality mask, the detector
coordinates of every box pixel (`sub_mapping`), optional planes (`layers/<name>`) and its headers.
Written by the reprojection or by [`write_frame`][selfcal.io.frames.write_frame]. See
[From exposures to a mosaic](concepts.md#from-exposures-to-a-mosaic) and the
[schema](pipeline.md#reprojected-h5-schema).

## Frame tag { #frame-tag }

The instrument's part of every product name: `Detector<d>_NumSub<s>_NumCh<c>_NumCol<k>` for
SPHEREx, `<tag>_Chunks<ny>x<nx>` for `grid`, `[instrument] tag` (default `EDFN`) for Euclid. See
[stem](#stem).

## Frame variable { #frame-variable }

A data variable with one value per frame: `exposure` and `detector`, a header keyword, the time, a
filter. Frame variables group offsets (`kind = "grouped"`, `groups = "<name>"`) and feed priors such
as `frame_smoothness`.

## Gate { #gate }

A regression check run by the maintainers ([Regression gates](../developer/gates.md)): it reruns a
fixed calibration and compares every dataset of the products with a reference product, the
*golden*, byte for byte. Goldens are regenerated only when a change is meant to alter the numbers.
See [Reproducibility](concepts.md#reproducibility).

## Gauge { #gauge }

A change of the unknowns that leaves every model value unchanged, such as a constant moved between
the sky and the per-frame scalars, and the choice that the priors make along it. Compare solutions
only through quantities that a change of gauge leaves alone. See
[What the data cannot tell apart](concepts.md#what-the-data-cannot-tell-apart).

## Golden { #golden }

A reference product (cal file or mosaic) that a [gate](#gate) compares new products with, byte for
byte. See [Reproducibility](concepts.md#reproducibility).

## Grouped clip { #grouped-clip }

The [outlier rejection](#outlier-threshold) of the solve, scored within groups of samples instead
of over the whole frame: `[calibration] outlier_group_edges` bins the data variable
`outlier_group_variable` (default: the instrument's wavelength map). In task `npass`,
`subch_clip = true` sets the edges to one bin per value of the spectral axis (SPHEREx: per
subchannel). See [Masks, outliers and weights](concepts.md#masks-outliers-and-weights).

## Hook { #hook }

1. A per-frame function chosen in `[hooks]` (`pre_cal`, `post_cal`, `post_mosaic`): it receives a
   [`FrameContext`][selfcal.core.subframe.FrameContext] and returns the frame's values (a `post`
   hook may also return new weights). The instrument's hook factories (Euclid:
   `star_position_mask`, `residual_mask`) and the runner's `mask_bright_pixels` can be chosen.
2. An optional method of an [`Instrument`][selfcal.instruments.base.Instrument] that has a
   default: offset renderer, auxiliary coadds, mosaic finaliser, coefficient catalogue, per-frame
   hook factories, post-calibration hooks (SPHEREx: the zodiacal-light anchor), data unit,
   precompute.

See [Masks, outliers and weights](concepts.md#masks-outliers-and-weights) and
[4. With code: an instrument](../bring_your_own_telescope.md#4-with-code-an-instrument).

## ignore_list { #ignore-list }

The data-quality bits that never flag a sample, given as bit numbers in `[calibration]` and
`[mosaic]` separately; empty means every set bit flags. The shipped SPHEREx configs ignore bit 21,
the source mask, in the mosaic, and the spectral ones in the solve as well
([tuning](pipeline.md#calibration-pipeline-tuning)). See
[Masks, outliers and weights](concepts.md#masks-outliers-and-weights).

## Instrument { #instrument }

The description of a telescope that the run engine works through
([`Instrument`][selfcal.instruments.base.Instrument], chosen by `[instrument] name`). Five methods
are required (jobs, product tag, exposure layout, detector geometry, per-job geometry); hooks have
defaults. Built in: `spherex`, `euclid` and the config-only `grid`. See
[Bring your own telescope](../bring_your_own_telescope.md).

## Job { #job }

One unit of an instrument's loop; each job has its own solve, cal file and mosaic. SPHEREx: a
channel, a group of channels or a subchannel window; `grid`: one job, `All` (`job_name`); Euclid:
one job named after `band`. See
[From exposures to a mosaic](concepts.md#from-exposures-to-a-mosaic).

## Line floor { #line-floor }

The uniform level of a spectral line map, which the data cannot fix: it is degenerate with a
detector pattern shaped like the line's coefficient.
[`selfcal.line_floor`](../reference/selfcal/line_floor.md) sets it after the solve from a
reference region declared free of emission, and keeps it in a file beside the cal file. See
[The absolute level](concepts.md#the-absolute-level).

## LSQR { #lsqr }

The iterative sparse least-squares solver (Paige and Saunders) that `[lsqr] solver = "lsqr"`
selects, in a memory-saving copy that gives the same results as SciPy's; `"lsmr"`, the default
when `solver` is not set, selects SciPy's LSMR (the shipped configs use `lsqr`). `iter_lim` limits
the iterations, `atol` and `btol` are the stopping tolerances. The run engine solves in float32
with `apply_n_threads` threads ([`apply_lsqr`][selfcal.core.solve.apply_lsqr]).

## LVF { #lvf }

Linear variable filter: a filter whose passband changes across the detector, so the wavelength a
pixel sees changes along one direction of the detector. On SPHEREx the lines of constant wavelength
are close to concentric circular arcs; selfcal fits their centre and radii (the LVF parameters,
`lvf_params_D<N>.npy`, shipped with the package and remade by task `precompute`) and builds the
[subchannels](#subchannel) from them.

## Mean-zero anchor { #mean-zero-anchor }

The prior `mean_zero = true` of an offset term: for every frame (and basis function), a row of
weight 10 pulls the mean of the frame's offsets over all chunks of the map to zero, so the frame's
overall level goes to its per-frame scalar. See
[What the data cannot tell apart](concepts.md#what-the-data-cannot-tell-apart).

## Mode { #mode }

The calibration recipe of a run (`mode`): a class registered with `@register_mode` that builds the
run's model from `[params]`. The modes are `continuum`, `spectral`, `spectral_softpoly`,
`spectral_polybasis`, `two_block_fixed` and `model` (the model read from a `[model]` table);
historical names are [presets](#preset). See [The model](concepts.md#the-model) and
[Modes](configuration.md#modes-presets-of-the-model).

## Model { #model }

What a solve fits: sky terms, offset terms, the per-frame scalar, data variables, the observation
weight and priors, held in a [`ModelSpec`][selfcal.models.spec.ModelSpec]. A mode builds it;
`mode = "model"` reads it from the config's `[model]` table. See
[The model](concepts.md#the-model).

## Mosaic { #mosaic }

`mosaic/mosaic_<stem>.fits`: the coadd of the corrected frames on the reference grid, with the
mean (`MEAN_MAP`), standard-deviation (`STD_MAP`) and sigma-clipped mean (`SC_MEAN_MAP`) maps,
each with its summed weight, and for SPHEREx the wavelength maps. Made by task `cal` after the
solve or by task `mosaic`. See [The mosaic](concepts.md#the-mosaic).

## N-pass solve { #n-pass }

Task `npass`, a solve in passes for spectral models: pass 1 (INIT) is the joint solve of task
`cal`; SKY passes then solve the sky terms exactly given the offsets, and OFFSET passes refit every
frame's polynomial offset and scalar given the sky, in turn (`[passes]`: `n`, `order`). See
[Scaling up](concepts.md#scaling-up) and
[N-pass alternating solve](pipeline.md#n-pass-alternating-solve-task-npass).

## NumCol, NumSub, NumCh { #numcol }

The SPHEREx chunk-geometry keys `num_col`, `num_sub` and `num_ch` of `[instrument]`, which
appear in product names as `NumCol<k>`, `NumSub<s>` and `NumCh<c>`: `num_ch`
[channels](#channel) per detector, `num_sub` [subchannels](#subchannel) per channel, `num_col`
vertical columns per subchannel. `num_col` is the main knob for how much spatial structure the
offsets can absorb ([tuning](pipeline.md#calibration-pipeline-tuning)).

## NVMe staging { #nvme-staging }

Copying a run's frame files from the slow disk where they live to fast scratch storage
(`<cache_dir>/reproj_nvme_<run_name>`) before a task reads them in parallel, with at most
`hdd_io_limit` concurrent reads; the copy is deleted at the end unless `keep_nvme = true`. See
[Scaling up](concepts.md#scaling-up).

## Observation { #observation }

One value of one frame on one pixel of the reference grid, recorded at a known detector position;
one row of the solve. A frame holds at most one observation per reference pixel. See
[The model](concepts.md#the-model).

## Observation weight { #weight }

The factor `w` that multiplies an observation's row, so the fit weights the observation by `w²`:
the product of the data-quality mask, the job's [valid weight](#valid-weight), the factor
`1 / sqrt(|data| + 1e-4)` with `apply_weight = true`, and the model's `weight` function of data
variables (which the mosaic applies squared). See
[Masks, outliers and weights](concepts.md#masks-outliers-and-weights).

## Offset { #offset }

An additive signal that belongs to a frame (or a group of frames) and a region of the detector
rather than to the sky: a bias or dark level, a foreground that changes between exposures,
read-out stripes, a pattern fixed to the detector. See [The problem](concepts.md#the-problem).

## Offset renderer { #offset-renderer }

The instrument hook that draws a frame's chunk offsets as a detector image for the mosaic. Default:
constant over each chunk; SPHEREx: a mean-preserving spline in arc radius and detector x for the
`subchannel` map; Euclid: a spline for `grid`, constant stripes, linear ramps. An offset term's
`render` key chooses among the instrument's renderers. See
[Chunk maps and chunk axes](concepts.md#chunk-maps-and-chunk-axes) and
[The mosaic](concepts.md#the-mosaic).

## Offset term { #offset-term }

One block of offsets in the model, on one chunk map, also called an offset map (`[[model.offset]]`,
[`OffsetTerm`][selfcal.models.spec.OffsetTerm]). `kind = "free"`: an offset per frame and chunk;
`"fixed"`: one offset vector shared by all frames; `"grouped"`: one per group of frames with equal
values of `groups`; `"polybasis"`: see [polybasis](#polybasis). It may carry a `coefficient` or a
`basis`, and its priors. Map `m`'s offsets are `offsets/map_<m>` in the cal file. See
[The model](concepts.md#the-model).

## Outlier threshold { #outlier-threshold }

`[calibration] outlier_thresh`: the solve leaves out every sample whose distance from its frame's
median exceeds this many times `1.4826 · MAD` (the median absolute deviation of the frame). The
[grouped clip](#grouped-clip) scores within groups instead; the mosaic has its own
[sigma clipping](#sigma-clipping). See
[Masks, outliers and weights](concepts.md#masks-outliers-and-weights).

## Oversample factor { #oversample }

The top-level `oversample` (default 1): the mosaic samples its detector-plane maps (chunk maps,
valid weights, rendered offsets) on a grid `oversample` times finer than the detector pixels. It
changes neither the mosaic's grid, which is the reference grid, nor the solve. See
[The mosaic](concepts.md#the-mosaic).

## Per-frame scalar { #per-frame-scalar }

One additive constant per frame (`scalar = true`; `frame_scalar` in the cal file). With mean-zero
anchors on the offset terms it carries each frame's overall level, and the chunk offsets carry
only structure within the frame. The mosaic subtracts it with the first offset map. See
[The model](concepts.md#the-model).

## Polybasis { #polybasis }

An offset term kind (`kind = "polybasis"`, used by the `spectral_polybasis` mode): the offset is a
polynomial along one chunk axis (`axis`, default the spectral axis) over `[lo, hi]`, one per value
of `group_axis`, and the solve fits its coefficients (a Chebyshev series of degrees 1 to `degree`;
the per-frame scalar carries the constant). `segments` fits an independent shape on each listed
sub-range. See [What the data cannot tell apart](concepts.md#what-the-data-cannot-tell-apart).

## Polynomial constraint { #polynomial-constraint }

A soft prior on an offset term (`poly = [{ axis, degree, weight, lo, hi }]`; in the modes
`poly_weight`, `poly_degree`, `poly_axis` and `spectral_poly_*`): on every run of `degree + 2`
consecutive chunks along `axis`, a row `weight · Σ stencil · O = 0` that vanishes on polynomials of
degree `degree` or less, so the offset is pulled toward such a polynomial. Added only with
`[calibration] offset_regularization = true`; [polybasis](#polybasis) is its exact form. See
[What the data cannot tell apart](concepts.md#what-the-data-cannot-tell-apart).

## Preconditioning { #preconditioning }

Scaling every column of the system to unit norm before the solve and undoing it afterwards
(`[lsqr] precondition`, default true): the Jacobi preconditioner for least squares, which makes
LSQR converge in fewer iterations. Columns without a single nonzero entry (unknowns that no row
touches) are dropped before the solve.

## Preset { #preset }

A named mode that fills in the model from a few `[params]` keys. The historical SPHEREx names are
presets of the structural modes: `pahfit` of `spectral`; `pahfit_subch` and `pahfit_lvf` of
`spectral_softpoly`; `pahfit_lvf_polybasis` and `multiline` of `spectral_polybasis`; `k2_readout`
of `two_block_fixed`; `tiled` is a variant of `spectral_softpoly` without a mosaic. See
[Modes](configuration.md#modes-presets-of-the-model).

## Prior { #prior }

Linear rows added to the system beside the data rows, to settle what the data leave free: the
priors built into the terms (`damp_weight`, `reg_weight`, `poly`, `mean_zero`, `damp`) and user
priors (`[[model.prior]]`: a function that returns rows on the unknowns of named terms;
ready-made: `frame_smoothness`, `sky_smoothness`, `toward_variable`). See
[What the data cannot tell apart](concepts.md#what-the-data-cannot-tell-apart).

## Reference grid { #reference-grid }

The common pixel grid that every frame is resampled onto and every map is made on: a celestial WCS
and a shape, stored in the run's `ref.fits`. Task `reproject` computes it at `resolution_arcsec`
around the exposures, or derives it from `[reproject] source_ref_path`, unless `ref.fits` already
exists. See [From exposures to a mosaic](concepts.md#from-exposures-to-a-mosaic).

## Reprojection { #reprojection }

Task `reproject`: resampling every detector of every raw exposure onto the reference grid with the
`reproject` package (`[reproject] reproj_func`: `exact`, the default, `interp` or `adaptive`) and
writing the frame files. See [From exposures to a mosaic](concepts.md#from-exposures-to-a-mosaic).

## Run config { #run-config }

The TOML file that describes one run: top-level keys (`task`, `mode`, `output_dir`, `run_name`,
...) and tables (`[instrument]`, `[params]`, `[calibration]`, `[lsqr]`, `[mosaic]`, `[model]`, ...).
Products go to the run directory `<output_dir>/<run_name>/`. See
[Run configuration](configuration.md).

## Separability { #separability }

For each sky term after the first, how well each pixel's observations separate it from the other
terms: the term's Fisher information that is left once the other terms are fitted out of the
pixel, a Schur complement of the pixel's normal equations (`sky_separability/<name>` in the cal
file). A pixel observed many times at a single wavelength has a large
[Fisher information](#fisher) for a line term but zero separability, so line maps are masked on
separability.

## Sigma clipping { #sigma-clipping }

The mosaic's outlier rejection (`[mosaic] apply_sigma_clipping`, `sigma`, with `make_std_map`):
`SC_MEAN_MAP` averages, at each pixel, only the values within `sigma` standard deviations
(`STD_MAP`) of the mean (`MEAN_MAP`). See [The mosaic](concepts.md#the-mosaic).

## Sky term { #sky-term }

A map on the reference grid, shared by every frame, times a known [coefficient](#coefficient) of
data variables (`[[model.sky]]`, [`SkyTerm`][selfcal.models.spec.SkyTerm]); without one, `c = 1`
and the term is a plain sky map, named `continuum` by default. Each term is one map in the cal file
(`sky/<name>`). See [The model](concepts.md#the-model).

## Stem { #stem }

The common part of a job's product names, `<frame_tag>_<job><suffix>`, as in `cal_<stem>.h5` and
`mosaic_<stem>.fits`: the instrument's [frame tag](#frame-tag), the [job](#job) name and the
config's `suffix`. See [From exposures to a mosaic](concepts.md#from-exposures-to-a-mosaic).

## Stitch { #stitch }

The merge of tile cal files into one cal file ([`stitch`][selfcal.pipeline.tiled.stitch]): every
sky pixel becomes the mean of the tiles that cover it, weighted by their
[Fisher information](#fisher). Per-frame quantities are dropped. See
[Scaling up](concepts.md#scaling-up).

## Subchannel { #subchannel }

SPHEREx: a strip of the detector between two neighbouring lines of constant wavelength, an arc of
the [LVF](#lvf); `num_sub` per channel, with one padding subchannel at each end
(`num_sub · num_ch + 2` in all). It is the `subchannel` axis, the spectral axis, of the SPHEREx
chunk map; windows of subchannels are SPHEREx jobs (`windows`, `subch_window`).

## Task { #task }

What a run does (`task`): `reproject`, `cal` (solve, then mosaic; tiled with `[tiling]`), `mosaic`
(from an existing cal file), `npass`, or `precompute` (the instrument's geometry generator).
`task = "tiled"` is read as `cal` with `[tiling]`. See
[From exposures to a mosaic](concepts.md#from-exposures-to-a-mosaic) and
[Run configuration](configuration.md#schema).

## Tiling { #tiling }

Splitting a large field into tiles of the reference grid (`[tiling]`: a `grid` with `overlap_px`,
or explicit `tiles`), solving each tile on its own frames and [stitching](#stitch) the tile skies.
See [Scaling up](concepts.md#scaling-up).

## Valid weight { #valid-weight }

The instrument's weight of each detector pixel for a job
([`JobGeometry`][selfcal.instruments.base.JobGeometry]): `det_valid_weight` for the solve and
`grid_valid_weight`, on the oversampled detector grid, for the mosaic. SPHEREx: the job's
subchannels (in the solve, one more on each side for a channel job), tapered toward their edges in
the mosaic; Euclid: an optional taper at the detector edges. See
[Masks, outliers and weights](concepts.md#masks-outliers-and-weights).

## Warm start { #warm-start }

The vector the solver starts from. The run engine starts each frame's scalar at the weighted mean
of its data and every other unknown at zero; a model without the scalar starts every offset at its
own one-column least-squares estimate. With a limited number of iterations, it matters along
weakly constrained directions. See
[What the data cannot tell apart](concepts.md#what-the-data-cannot-tell-apart).

## Zodiacal-light anchor { #zodi-anchor }

A fit after the solve that sets the absolute level of SPHEREx maps, channel by channel: every
frame's mean level is fitted against a zodiacal-light prediction, and the intercept `C` is added to
the maps when they are read. The cal file and the mosaic are never rewritten; the fit is kept in
`<run>/zodi_anchor/anchor_D<N>.h5`. See [The absolute level](concepts.md#the-absolute-level) and
the [Zodiacal-light anchor](../tools/zodi-anchor.md) guide.
