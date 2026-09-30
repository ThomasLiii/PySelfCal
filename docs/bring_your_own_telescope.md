# Bring your own telescope

`selfcal` calibrates a set of overlapping exposures by sparse least squares. For every
*observation* — one value of one frame on one pixel `P` of a common reference grid, seen at a
detector position — it fits

```
data = Σ_j S_j[P] · c_j(v)  +  Σ_m Σ_k O_m[g_m(frame), chunk_m, k] · φ_mk(v)  +  s(frame)
```

* `S_j` — **sky maps** (one or more), each times a known **coefficient** `c_j`;
* `O_m` — **offsets** on a chunk partition of the detector, shared by the frames of a group
  `g_m` (every frame its own, all frames one, or frames with equal values of any per-frame
  quantity), optionally times known functions `φ_mk`;
* `s` — a per-frame **scalar**;
* `v` — **data variables**: named per-observation quantities (the detector position, the time,
  the wavelength, a polariser angle, a variance, a previous sky, anything) that every function of
  the model reads by name.

Priors (damping, smoothness, polynomial shape, anchors, or any linear rows you write) regularise
the solve; the corrected frames are then coadded into a mosaic. Nothing in the solver knows a
telescope. A telescope enters through an **instrument** (how raw exposures are read, how the
detector is chunked, which detector maps and per-frame values it defines) and the calibration is a
**model** (`[model]` in the run config). A new telescope, sky coefficient, offset set-up or prior
is a function you write — never an edit of `selfcal`.

## 1. No code: the `grid` instrument

Any imager whose exposures are FITS files with a science image + a celestial WCS in one
extension and (optionally) an integer mask in another.

```toml
# 1. reproject every exposure onto one reference grid
task = "reproject"
output_dir = "/data/runs/"
run_name = "mycam_field1"
resolution_arcsec = 1.0                # pixel scale of the reference grid
cache_dir = "/scratch/selfcal/"

[instrument]
name = "grid"
tag = "MyCam"                          # product-name tag
detector_shape = [2048, 2048]          # rows, cols of the science array
chunks = [8, 8]                        # offset chunk grid (rows, cols)
sci_ext = 1
dq_ext = 2                             # omit (or -1) if the exposures carry no mask

[reproject]
input_dirs = ["/data/mycam/field1/"]
file_pattern = "/*.fits"
reproj_func = "interp"                 # interp | exact | adaptive
padding_pixels = 100
max_workers = 16
```

```toml
# 2. calibrate + mosaic (same [instrument] table)
task = "cal"
mode = "continuum"                     # constant sky per pixel
output_dir = "/data/runs/"
run_name = "mycam_field1"
resolution_arcsec = 1.0
cache_dir = "/scratch/selfcal/"
suffix = "_v1"
apply_n_threads = 16

[instrument]
name = "grid"
tag = "MyCam"
detector_shape = [2048, 2048]
chunks = [8, 8]
sci_ext = 1
dq_ext = 2

[params]
reg_weight = 0.1                       # smoothness between neighbouring chunks
poly_weight = 0.5                      # optional: soft linear polynomial along the chunk columns

[calibration]
apply_mask = true
apply_weight = false
outlier_thresh = 5.0
ignore_list = []                       # mask bits to ignore
batch_size = 20
offset_regularization = true
weighted_damping = true
damp_weight = 0.1
max_workers = 16

[lsqr]
atol = 1e-6
btol = 1e-6
damp = 0
iter_lim = 100
precondition = true
solver = "lsqr"

[mosaic]
apply_mask = true
apply_weight = false
make_std_map = true
apply_sigma_clipping = true
sigma = 3.0
ignore_list = []
cache_intermediate = true
cache_batch_size = 20
coadd_batch_size = 20
max_workers = 16
```

```
python -m selfcal_scripts.run --config reproject.toml
python -m selfcal_scripts.run --config cal.toml
```

Products: `<output_dir>/<run_name>/reprojected/exp_*_det_00.h5`, `calibration/cal_MyCam_Chunks8x8_All_v1.h5`
(read it with `selfcal.io.calfile.CalFile`), `mosaic/mosaic_MyCam_Chunks8x8_All_v1.fits`
(`MEAN_MAP`, `STD_MAP`, `SC_MEAN_MAP` + weights). `tests/test_runner_e2e_toy.py` runs exactly this
on synthetic exposures (`tests/synthetic_exposures.py`) in a few seconds; copy it as a starting point.

## 2. The model is yours: `mode = "model"`

The named modes are presets of one thing, a `[model]`. Spell it out to change it:

```toml
mode = "model"

[model]
scalar = true                                    # per-frame scalar
weight = { variable = "variance", function = "mypkg.noise:inverse_sigma" }   # optional

[model.variables]                                # data variables beyond the instrument's
time     = { header = "MJD-AVG" }                # a keyword of each frame's header
variance = { layer = "variance" }                # a plane stored with each frame
psi      = { function = "mypkg.pol:angle", inputs = ["hwp", "pix_angle"] }

[[model.sky]]
name = "continuum"                               # no coefficient: a constant sky

[[model.sky]]
name = "annual"                                  # a map times ANY function of data variables
coefficient = { variable = "time", function = "mypkg.season:sine", params = { period = 365.25 } }
damp_weight = 1e-3

[[model.offset]]                                 # free per-frame offsets on the chunk grid
name = "chunks"
kind = "free"
reg_weight = 0.1
adjacency = ["row", "col"]                       # the grid instrument's chunk axes
mean_zero = true
poly = [ { axis = "col", degree = 1, weight = 0.5 } ]

[[model.offset]]                                 # a detector-fixed pattern times the temperature
kind = "fixed"
coefficient = { variable = "temperature", function = "mypkg.thermal:above", params = { t0 = 80.0 } }

[[model.offset]]                                 # a free 2-D gradient per frame
map = "detector"                                 # built-in: the whole detector as one chunk
kind = "free"
basis = { variable = ["det_x", "det_y"], function = "mypkg.shapes:plane", n = 2 }

[[model.prior]]                                  # any linear rows on the terms' unknowns
term = "chunks"
function = "frame_smoothness"                    # ready-made, or "mypkg.priors:mine"
variable = "time"
weight = 0.2
```

### Data variables

| source | `[model.variables]` form | one value per | examples |
| --- | --- | --- | --- |
| built-in | — | observation | `det_x`, `det_y` (detector position), `sky_x`, `sky_y` (reference pixel), `frame` |
| instrument detector map | — (`DetectorGeometry.aux`) | detector pixel | SPHEREx `BC` / `BW` (aliases `wavelength` / `bandwidth`), a pixel polariser angle |
| instrument frame value | — (`Instrument.frame_variables`) | frame | `exposure`, `detector` (built in), time, filter, HWP angle |
| header keyword | `{ header = "KEY", default = ... }` | frame | `MJD-AVG`, `FILTER`, `TEMP` |
| function of the frame list | `{ per_frame = "pkg.mod:fn" }` (or `inputs = [...]` of frame variables) | frame | a table lookup, `floor(time)` for the night |
| detector map | `{ detector = "pkg.mod:fn" \| "map.npy" \| "map.fits" }` | detector pixel | a QE map, pixel areas |
| sky map | `{ sky = "pkg.mod:fn" \| "map.npy" \| "map.fits" }` | reference pixel | ecliptic latitude, a dust template |
| solved sky | `{ sky_cal = "cal.h5", term = "continuum" }` | reference pixel | a previous sky, to linearise gains or flats |
| stored layer | `{ layer = "variance" }` | observation | variance, a per-frame wavelength map, per-pixel time |
| function of variables | `{ function = "pkg.mod:fn", inputs = [...] }` | observation | ψ = 2·HWP + pixel angle; λ shifted by temperature |
| function of the frame | `{ frame_function = "pkg.mod:fn" }` | observation | crosstalk from the frame's own data, persistence from the previous frame |

Every source takes `params = {...}`. A frame function receives a `FrameObservations`: the frame's
file, index, observation pixels, `sub_mapping` (detector coordinates), `raw()` (the stored values),
`header()` and the other variables.

### Terms, weights, priors

* **Sky term**: `name`, `coefficient` (`{variable, function, params}`; `function` is a built-in —
  `template` (tabulated, e.g. a spectral line template npz), `gaussian`, `linear` — or any
  `"pkg.mod:name"`; omitted = the variable itself; or `{catalog = ...}` for an instrument's named
  coefficient), `damp_weight`.
* **Offset term**: `map` (a chunk map of the instrument, or `"detector"`), `kind` (`free`,
  `fixed`, `grouped` + `groups = <any frame variable>`, `polybasis`), `coefficient` (one known
  function multiplying the offset at every observation) or `basis` (`n` functions: the unknowns are
  one coefficient per group × chunk × function), and the built-in priors `reg_weight` +
  `adjacency`, `poly`, `mean_zero` (per function for a basis), `damp`, `exact_group_rows`.
* **weight**: a function of data variables multiplying every observation's weight.
* **prior**: `term` (or `terms`, for a relation between terms: sky-term names, offset-term names,
  `scalar`), `function`, `weight`, and the function's parameters. A prior function receives one
  `TermInfo` per term (the unknowns' shape, their coverage, the frame → group map, the solve's
  variables, the chunk axes, `index(...)` for the global unknown ids) and returns
  `(rows, cols, vals, rhs)`. Ready-made (`selfcal.models.priors`): `frame_smoothness` (offsets of
  neighbouring frames in a frame variable), `sky_smoothness` (neighbouring sky pixels),
  `toward_variable` (unknowns toward a known map or per-frame value).
* **grouped clip**: `[calibration] outlier_group_variable = "<variable>"` with
  `outlier_group_edges = [...]` judges each observation against its own group's distribution.

Anchors need care when a term's groups observe different parts of its map (detectors on one
focal plane): `mean_zero` sums over every chunk of the map, observed or not, so anchor such a
term with `damp` (it acts on observed unknowns only) or a prior.

## 3. Examples: each is a config plus functions

Every row is calibrated end to end on synthetic data in `tests/test_any_telescope.py`, with
nothing but the functions and instruments defined in that file.

| telescope / model | what is written | test |
| --- | --- | --- |
| a sky modulated with the season | `time = {header}`; sky coefficient `sin 2π(t − t0)/P`; `frame_smoothness` prior on drifting offsets | `test_time_variable_sky` |
| imaging polarimeter (I, Q, U) | instrument subclass: detector map `pix_angle`, frame value `hwp`; `psi = {function, inputs}`; Q with `cos 2ψ`, U with `sin 2ψ` | `test_imaging_polarimeter` |
| thermal pattern + scattered-light gradients | fixed offset × `temperature`; free `basis` in `det_x`, `det_y` on `map = "detector"`; the mosaic subtracts both per observation | `test_thermal_pattern_and_gradients` |
| integral-field cube, slices as frames | a reader: one slice per `sci_ext`, `WAVE` keyword, variance layer; line map × Gaussian(`wave`); offsets grouped by `exposure`; `weight = 1/σ` | `test_data_cube_slices` |
| data that never were images | `write_frame` + the library API; a per-frame gain × a known sky (`sky` variable); a prior fixing the gains' scale | `test_frames_written_directly` |
| amplifier crosstalk | `frame_function` computing the mirror amplifier's mean from `raw()`; a fixed offset with that coefficient | `test_amplifier_ghost` |
| detectors of different sizes on one focal plane | a reader returning focal-plane `coords`; a chunk map with `-1` in the gaps; offsets grouped by `detector` | `test_heterogeneous_focal_plane` |

More set-ups the same machinery expresses (each a few lines of config and one function):

| set-up | how |
| --- | --- |
| filter-wheel imager, all filters in one solve | `filter = {header = "FILTER"}`; one sky term per filter with coefficient `filter == "J"`; or a reference map + a colour map × (λ_eff(filter) − λ0) |
| LVF whose wavelength shifts with temperature | sky coefficient of `["BC", "temperature"]`: `template(BC + k·(T − T0))` |
| drift-scan / rolling shutter | time of each row: `function` of `["time", "det_y"]` |
| emissivity relative to a known dust map | `dust = {sky = "dust.fits"}`; sky coefficient `dust` |
| stray light vs. Sun angle | `sun = {header = "SUNANG"}`; fixed offset with coefficient `h(sun)` |
| slowly drifting detector pattern | fixed offset, `basis` of `time`: `[1, t − t0, (t − t0)²]` |
| fringes / pickup of known shape | free offset, `basis` `[cos(kx), sin(kx)]` of `det_x` |
| per-frame gain or flat-field error | coefficient = a previous sky (`sky_cal`); free (gain) or fixed (flat) |
| persistence | `frame_function` reading the previous exposure's frame file |
| offsets shared per night / visit / filter | `kind = "grouped"`, `groups = "<frame variable>"` |
| offsets tied to a predicted value | prior `toward_variable` (or your own rows with a target) |
| sky smoothness or in-painting | prior `sky_smoothness` (`covered_only = false` smooths into gaps) |
| inverse-variance weighting | `variance = {layer}` + `weight = {variable = "variance", function = ...}` |
| windowed or binned read-outs | a reader returning the window's full-detector `coords` |

## 4. With code: an instrument

For a real geometry, subclass `Instrument` (five methods) and override only the hooks your
telescope needs:

```python
import numpy as np
from selfcal.instruments import (Instrument, register_instrument, Job, ChunkMap,
                                 DetectorGeometry, JobGeometry, ExposureLayout)
from selfcal.io.frames import ExposureData, frame_header_values
from selfcal.models.offset_structure import ChunkAxes
from selfcal.geometry.map_helper import make_grid_chunk_map


@register_instrument("mycam")                    # selects it: [instrument] name = "mycam"
class MyCam(Instrument):
    def jobs(self, cfg):                         # the run's job loop (one job here)
        return [Job("all")]

    def frame_tag(self, cfg):                    # product-name component
        return "MyCam"

    def exposure_layout(self, cfg):              # how a raw exposure is read
        return ExposureLayout(sci_ext=[1], dq_ext=[3], detector_ids=[0],
                              reader=None)       # or your reader (below)

    def detector_geometry(self, cfg, oversample):
        det = make_grid_chunk_map((2048, 2048), 8)              # any int map, -1 = no chunk
        grid = np.kron(det, np.ones((oversample, oversample), dtype=det.dtype))
        cm = ChunkMap("grid", det, grid,
                      axes=ChunkAxes.row_major(("row", "col"), (8, 8), ("y", "x")),
                      adjacency_axes=("row", "col"))
        return DetectorGeometry(shape=(2048, 2048), chunk_maps={"grid": cm}, primary="grid",
                                aux={"pix_angle": my_angle_map})   # detector variables

    def job_geometry(self, cfg, geom, job):     # per-job validity weights
        ones = np.ones(geom.shape, dtype=np.float32)
        return JobGeometry(det_valid_weight=ones,
                           grid_valid_weight=np.ones(geom.chunk_map.grid.shape, np.float32))

    # optional: per-frame values every model can read by name
    def frame_variable_names(self, cfg):
        return super().frame_variable_names(cfg) + ("hwp",)

    def frame_variables(self, frames, cfg=None):
        out = super().frame_variables(frames, cfg)             # exposure, detector
        out["hwp"] = frame_header_values(frames, ["HWPANG"])["HWPANG"]
        return out
```

**Raw data in any format** — a reader returns, for one detector frame of one file, the values, a
header carrying the celestial WCS (and any keywords you want as frame variables), an optional bit
mask, extra per-pixel planes (stored with the frame as layers) and optional detector coordinates
(a focal plane, the full detector of a window):

```python
def read_mycam(path, sci_ext, dq_ext=None, header_only=False):
    hdr = ...                                    # an astropy Header with the WCS (+ keywords)
    if header_only:
        return ExposureData(header=hdr, shape=(ny, nx))
    return ExposureData(header=hdr, data=values, mask=bits,
                        layers={"variance": var}, coords=(x_focal, y_focal))
```

`sci_ext` / `dq_ext` are whatever the layout lists — extension numbers, slice indices, detector
names; the reader interprets them. The reprojection, the reference-frame definition and everything
downstream use the reader; nothing else changes.

Optional hooks with defaults: `offset_renderer` (smooth chunk-to-pixel rendering for the mosaic),
`aux_coadds` / `finalize_mosaic` (extra per-pixel maps to coadd), `coefficient_catalog` (named
coefficients), `hooks` (per-frame hook factories selectable in `[hooks]`), `postcal_hooks`,
`data_unit`, `precompute`. `selfcal/instruments/spherex/adapter.py` (spectral, non-rectangular
chunks, wavelength maps), `selfcal/instruments/euclid/adapter.py` (16 detectors per exposure,
stripe and tilt maps) and the instruments of `tests/test_any_telescope.py` are reference
implementations; `selfcal/instruments/grid.py` is the minimal one.

A package of your own can register the instrument through the `selfcal.instruments` entry-point
group (`[project.entry-points."selfcal.instruments"] mycam = "mypkg.instrument:MyCam"`), so it is
selectable by name without touching `selfcal`.

## 5. Without images: write frames directly

The solver's input is a directory of frame files. Anything that can say, for each sample, which
reference pixel it falls on and where on the detector it was taken can be calibrated:

```python
from selfcal.io.frames import write_frame, standard_frame_path
write_frame(standard_frame_path(frame_dir, exposure, detector),
            values,                         # (H, W) on the box [y0, y1, x0, x1] of the reference grid
            [y0, y1, x0, x1],
            np.stack([det_x, det_y]),       # detector coordinates of every box pixel
            layers={"variance": var}, header=hdr)
```

then use the library directly (`selfcal.Calibrator(...).setup_lsqr(offset_model=..., sky_model=...,
variables=VariableSet(...), priors=[...])`) or point a run config's `reproj_override` at the
directory.

## 6. Limits

* Each observation samples one reference pixel per sky term. A forward model that mixes several
  sky pixels into one observation (slitless dispersion, a PSF deconvolution) needs a projection
  operator per observation in the row assembly — not available yet.
* The model is linear. Multiplicative effects (gains, flats, non-linearity) are linearised
  around a previous sky (a `sky_cal` variable) and iterated.
* The N-pass schedule (task `npass`) runs sky terms with coefficients of any variables; offset
  terms with a coefficient or basis run as task `cal`.
* A frame holds at most one observation per reference pixel. Samples that revisit a pixel within
  one exposure (a scanning instrument's time-ordered data) go into several frames, which share
  their offsets through a `grouped` term (e.g. `groups = "exposure"`).
* The sky is a 2-D grid (a WCS image). Another pixelisation (HEALPix) can be flattened to one row
  of `N` pixels; each frame then stores the range of pixel indices it touches.
* The mosaic coadds the data with the `[mosaic]` weights; the model's `weight` shapes the solve.

## 7. Where things are

| what | where |
| --- | --- |
| the model as data (variables, terms, priors) | `selfcal/models/spec.py` |
| data variables and their sources | `selfcal/models/variables.py` |
| ready-made priors | `selfcal/models/priors.py` |
| generic offset-structure builders on chunk axes | `selfcal/models/offset_structure.py` |
| the instrument contract + registry | `selfcal/instruments/base.py` |
| exposure readers, the frame file, header values | `selfcal/io/frames.py` |
| the run engine (config → tasks) | `selfcal_scripts/runner/` |
| calibration-file reader | `selfcal/io/calfile.py` |
| config schema | `selfcal_scripts/configs/README.md` |
| on-disk products, tuning knobs | `PIPELINE.md` |
