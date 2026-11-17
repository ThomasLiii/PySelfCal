# Bring your own telescope

`selfcal` calibrates a set of overlapping exposures by sparse least squares. For every
*observation* — one value of one frame on one pixel `P` of a common reference grid, seen at a
detector position — it fits

```
data = Σ_j S_j[P] · c_j(v)  +  Σ_m Σ_k O_m[g_m(frame), chunk_m, k] · φ_mk(v)
```

* `S_j` — **sky maps** (one or more), each times a known **coefficient** `c_j`;
* `O_m` — **offsets** on a chunk partition of the detector, shared by the frames of a group
  `g_m` (every frame its own, all frames one, or frames with equal values of any per-frame
  quantity), optionally times known functions `φ_mk`; on the map `"detector"`, whose one chunk is
  the whole detector, a per-frame **scalar**;
* `v` — **data variables**: named per-observation quantities (the detector position, the time,
  the wavelength, a polariser angle, a variance, a previous sky, anything) that every function of
  the model reads by name.

Priors (damping, smoothness, polynomial shape, or any linear rows you write) regularise the solve,
and hard constraints remove what an offset term must not carry (a constant, a polynomial along a
chunk axis); the corrected frames are then coadded into a mosaic. Nothing in the solver knows a
telescope. A telescope enters through an **instrument** (how raw exposures are read, its detector,
which detector maps and per-frame values it defines) and the calibration is a **model**
(`sc.Model`: its terms and the chunk maps they live on). A new telescope, sky coefficient, offset set-up or prior is a function you
write — never an edit of `selfcal`.

## 1. No code: `sc.Camera`

Any imager whose exposures are FITS files with a science image + a celestial WCS in one
extension and (optionally) an integer mask in another:

```python
import selfcal as sc

CAMERA = sc.Camera((2048, 2048), chunks=(8, 8), dq_ext=2, tag="MyCam")   # dq_ext=None: no mask
FIELD = sc.Field("/data/runs/mycam_field1", CAMERA, pixel_scale=1.0,
                 compute=sc.Compute("/scratch/selfcal", workers=16))
RECIPE = sc.Recipe(sc.continuum(smooth=0.1),                              # smoothness between chunks
                   fit=sc.Fit(100, clip=5.0),
                   coadd=sc.Coadd(clip=3.0),
                   name="v1")

if __name__ == "__main__":
    FIELD.reproject("/data/mycam/field1/*.fits", method="interp", padding=100)
    print(FIELD.calibrate(RECIPE, jobs=sc.Job("All")))      # one job: every chunk
```

Products: `/data/runs/mycam_field1/reprojected/exp_*_det_00.h5`,
`calibration/cal_MyCam_Chunks8x8_All_v1.h5` (read it with `result.cal()` or
`selfcal.io.calfile.CalFile`), `mosaic/mosaic_MyCam_Chunks8x8_All_v1.fits` (`MEAN_MAP`,
`STD_MAP`, `SC_MEAN_MAP` + weights). The [quickstart](getting-started/quickstart.md) runs exactly
this on simulated exposures.

A camera takes more without code:

* `reader=read_mycam` — exposures in any format (the reader contract is in
  [section 4](#4-with-code-an-instrument));
* `detector_maps={"pix_angle": angle}` — per-pixel data variables, arrays of the detector's shape;
* `headers={"hwp": "HWPANG"}` — per-frame data variables read from each frame's header;
* `unit="e-/s"` — the mosaic's `BUNIT`.

## 2. The model is yours

The presets (`sc.continuum`, `sc.spectral`, `sc.two_block`) are models; spell one out to change it.
The functions are yours, defined at the top level of a module the worker processes can import
(here `mycam.py`), and read the data variables their parameters name:

```python
import selfcal as sc
from mycam import above, inverse_sigma, plane, polariser_angle, season

MODEL = sc.Model(
    sky=[sc.Sky("continuum"),                                  # no coefficient: a constant sky
         sc.Sky("annual", times=sc.Function(season, of="time", period=365.25),   # a map times ANY
                damping=1e-3)],                                                  # function of variables
    offsets=[sc.Offsets("chunks", smooth=0.1,                 # free per-frame offsets on the chunk grid,
                        exclude=sc.Poly(0)),                   #   with no constant in any frame
             sc.Offsets("thermal", per="all",                  # a detector-fixed pattern times the temperature
                        times=sc.Function(above, of="temperature", t0=80.0)),
             sc.Offsets("gradient", on="detector",            # a free 2-D gradient per frame
                        basis=plane, n=2),                     # plane(det_x, det_y) returns 2 arrays
             sc.Offsets("scalar", on="detector")],             # the per-frame scalar: each frame's level
    variables={"time": sc.Header("MJD-AVG"),                   # a keyword of each frame's header
               "temperature": sc.Header("TEMP"),
               "variance": sc.Layer("variance"),               # a plane stored with each frame
               "psi": sc.Derived(polariser_angle, of=["hwp", "pix_angle"])},
    weight=sc.Function(inverse_sigma, of="variance"),          # optional: 1/σ observation weights
    priors=[sc.priors.frame_smoothness("chunks", "time", weight=0.2)])   # any linear rows
```

`print(MODEL)` shows the model; `field.plan(recipe, jobs=...)` checks it against the instrument
(every variable it reads, every chunk map and axis it names) and imports every function in a fresh worker
process, in seconds, before anything runs.

### Data variables

| source | Python | one value per | examples |
| --- | --- | --- | --- |
| built-in | — | observation | `det_x`, `det_y` (detector position), `sky_x`, `sky_y` (reference pixel), `frame`, `value` (the stored value), `flags` (its flag bits) |
| instrument detector map | `sc.Camera(detector_maps=...)`, an instrument's `sc.Geometry(aux=...)` | detector pixel | SPHEREx `BC` / `BW` (aliases `wavelength` / `bandwidth`), a pixel polariser angle |
| instrument frame value | `sc.Camera(headers=...)`, an instrument's `frame_variables()` | frame | `exposure`, `detector` (built in), time, filter, HWP angle |
| header keyword | `sc.Header("KEY", default=...)` | frame | `MJD-AVG`, `FILTER`, `TEMP` |
| function of the frame list | `sc.PerFrame(fn)` (or `of=[...]` frame variables) | frame | a table lookup, `floor(time)` for the night |
| detector map | `sc.DetectorMap(array \| "map.npy" \| "map.fits" \| fn)` | detector pixel | a QE map, pixel areas |
| sky map | `sc.SkyMap(array \| "map.npy" \| "map.fits" \| fn)` | reference pixel | ecliptic latitude, a dust template |
| solved sky | `sc.SolvedSky("cal.h5", term="continuum")` | reference pixel | a previous sky, to linearise gains or flats |
| stored layer | `sc.Layer("variance")` | observation | variance, a per-frame wavelength map, per-pixel time |
| function of variables | `sc.Derived(fn, of=[...])` | observation | ψ = 2·HWP + pixel angle; λ shifted by temperature |
| function of the frame | `sc.FrameFunction(fn)` | observation | crosstalk from the frame's own data, persistence from the previous frame |

Every source takes parameters as keywords (`sc.DetectorMap(qe_map, band="J")`). A frame function
receives a `FrameObservations`: the frame's file, index, observation pixels, `sub_mapping`
(detector coordinates), `raw()` (the stored values), `header()` and the other variables.

### Terms, weights, priors

* **Sky term**, `sc.Sky(name, times=c, damping=)`: `c` is a Python function of data variables
  (`sc.Function(fn, of=..., **params)` names them and binds parameters explicitly), a built-in
  shape — `sc.template(file)` (tabulated, e.g. a spectral line template npz), `sc.gaussian`,
  `sc.linear` —, or the name of a variable, which then multiplies the map itself; omitted, the sky is constant. `damping` pulls the
  map toward zero, weighted by coverage.
* **Chunk maps**, `sc.Model(chunk_maps={name: sc.ChunkMap(fn, tag=, **params)})`: the partitions of
  the detector the offset terms live on, in order. The first is the primary map: the default `on`,
  the map the jobs name their chunks on (`sc.Job(name, window={axis: range(lo, hi)})`), and the one
  whose `tag` joins the product names. `fn(geometry, **params)` returns an `sc.Chunks`
  ([section 4](#4-with-code-an-instrument)). A model that names none gets the instrument's default
  maps (`sc.Camera`: its `grid` of `chunks` rectangles).
* **Offset term**, `sc.Offsets(name, on=, per=, ...)`: on a chunk map of the model (`on`;
  default the primary map; `"detector"`: the whole detector as one chunk), shared by
  `per="frame"` (each frame its own), `per="all"` (all frames one) or `per=` any frame variable
  (frames with equal values); `times=φ` (one known function multiplying the offset at every
  observation) or `basis=fns, n=n` (`n` functions: the unknowns are one coefficient per group ×
  chunk × function; an offset that is a polynomial in a coordinate is a basis of that coordinate's
  polynomials); the built-in priors `smooth` + `smooth_along`, `damping`, `exact_group_rows`; and a
  hard constraint, `exclude=sc.Poly(k, along=)`: no polynomial of degree ≤ `k` along the axis (per
  line of the other axes, per group), imposed exactly by the solver with no row and no weight, for a
  term of one unknown per chunk (not `basis=`);
  `exclude=sc.Poly(0)` needs no `along` (no constant over all of a group's chunks). The term
  `sc.Offsets("scalar", on="detector")` is a per-frame scalar.
* **weight**, `sc.Model(weight=fn)`: a function of data variables multiplying every observation's
  row weight (`1/σ` for an inverse-variance fit); the mosaic coadds with its square, like the solve.
  `sc.Variance(fn, of=...)` gives the noise variance instead: the row weight is `1/sqrt` of it, the
  coadd's weight `1/` it. The function receives a whole frame's observations at once, so it can use
  per-frame statistics of the stored values (`value`) and flag bits (`flags`): `sc.shot_noise()` is
  the shot-noise variance `max(|value|, m)` floored at the frame's median `m`.
* **prior**, `sc.Prior(fn, terms, weight=, **params)`: `terms` names the terms the rows act on
  (sky-term names and offset-term names, such as `"scalar"` for a per-frame scalar term). A prior
  function receives one `TermInfo` per term (the unknowns' shape, their coverage, the frame → group
  map, the solve's variables, the chunk axes, `index(...)` for the global unknown ids) and returns
  `(rows, cols, vals, rhs)`.
  Ready-made (`sc.priors`): `frame_smoothness` (offsets of neighbouring frames in a frame
  variable), `sky_smoothness` (neighbouring sky pixels), `toward` (unknowns toward a known map or
  per-frame value).
* **grouped clip**: `sc.Fit(clip=sc.Clip(5.0, variable="<variable>", edges=[...]))` judges each
  observation against its own group's distribution; `per="chunk"` or
  `per=sc.ChunkGroups.along("<axis>")` groups by chunks instead.

Hard constraints need care when a term's groups observe different parts of its map (detectors on
one focal plane): `exclude=` acts on a group's chunks that are unknowns of the solve, those with an
observation or a prior row. A chunk that the group never observes drops out of it, unless a
smoothness row (`smooth`) makes that chunk an unknown; such a term can be settled with `damping` or
a prior instead.

A calibration variant is such a model, or a function that returns one, as the presets
`sc.continuum`, `sc.spectral` and `sc.two_block` are (`selfcal/models/model.py`). A recipe takes
the model itself: nothing is registered, and the engine needs no change.

## 3. Examples: each is a few lines of Python

Every row is calibrated end to end on synthetic data in `tests/test_any_telescope.py`, with
nothing but the functions and instruments defined in that file; copy the test that is closest to
your set-up.

| telescope / model | what is written | test |
| --- | --- | --- |
| a sky modulated with the season | `variables={"time": sc.Header("MJD-AVG")}`; a sky term `times=annual`, `sin 2π(t − t0)/P`; `sc.priors.frame_smoothness` on drifting offsets | `test_time_variable_sky` |
| imaging polarimeter (I, Q, U) | `sc.Camera(detector_maps={"pix_angle": ...}, headers={"hwp": "HWPANG"})`; `psi = sc.Derived(polariser_angle)`; Q with `times=cos2`, U with `times=sin2` | `test_imaging_polarimeter` |
| thermal pattern + scattered-light gradients | `sc.Offsets(per="all", times=centred)` of the temperature; `sc.Offsets(on="detector", basis=plane, n=2)` of `det_x`, `det_y`; the mosaic subtracts both per observation | `test_thermal_pattern_and_gradients` |
| integral-field cube, slices as frames | an `sc.Instrument` whose layout's reader returns one slice per `sci_ext`, its `WAVE` keyword and a variance layer; a line map `times=gaussian_line` of `wave`; offsets `per="exposure"`; `weight=inverse_sigma` | `test_data_cube_slices` |
| data that never were images | `write_frame`, then `field.calibrate(recipe, frames=<dir>)`; a per-frame gain on `on="detector"` times a known sky (`sc.SkyMap`); an `sc.Prior` fixing the gains' scale | `test_frames_written_directly` |
| amplifier crosstalk | `sc.FrameFunction(partner_mean)` computing the mirror amplifier's mean from `raw()`; `sc.Offsets(per="all", times="partner")` | `test_amplifier_ghost` |
| detectors of different sizes on one focal plane | an `sc.Instrument` whose reader returns focal-plane `coords` and whose chunk map has `-1` in the gaps; offsets `per="detector"`, anchored by `damping` | `test_heterogeneous_focal_plane` |

More set-ups the same machinery expresses (each a few lines and one function):

| set-up | how |
| --- | --- |
| filter-wheel imager, all filters in one solve | `"filter": sc.Header("FILTER")`; one sky term per filter, `times=` an indicator of its filter; or a reference map + a colour map times `λ_eff(filter) − λ0` |
| LVF whose wavelength shifts with temperature | a sky term `times=` a function of `BC` and `temperature`: `template(BC + k·(T − T0))` |
| drift-scan / rolling shutter | the time of each row: `sc.Derived(fn, of=["time", "det_y"])` |
| emissivity relative to a known dust map | `"dust": sc.SkyMap("dust.fits")`; a sky term `times="dust"` |
| stray light vs. Sun angle | `"sun": sc.Header("SUNANG")`; `sc.Offsets(per="all", times=h)` with `h(sun)` |
| slowly drifting detector pattern | `sc.Offsets(per="all", basis=drift, n=3)`, `drift(time)` returning `[1, t − t0, (t − t0)²]` |
| fringes / pickup of known shape | `sc.Offsets(basis=fringe, n=2)`, `fringe(det_x)` returning `[cos(kx), sin(kx)]` |
| per-frame gain or flat-field error | a previous sky as the coefficient (`sc.SolvedSky(cal)`); `per="frame"` (gain) or `per="all"` (flat) |
| persistence | `sc.FrameFunction(fn)` reading the previous exposure's frame file |
| offsets shared per night / visit / filter | `sc.Offsets(per="<frame variable>")` |
| offsets tied to a predicted value | `sc.priors.toward(term, variable)` (or your own rows with a target) |
| sky smoothness or in-painting | `sc.priors.sky_smoothness(term, covered_only=False)` smooths into gaps |
| inverse-variance weighting | `"variance": sc.Layer("variance")` + `weight=inverse_sigma` |
| shot-noise weighting, floored at each frame's median | `weight=sc.shot_noise()` (`sc.Variance(shot_noise_variance, of=("value", "flags"))`) |
| a per-frame gradient model's gauge (the sky plane the offsets absorbed) | `sc.Fit(gauge=sc.Gauge("slope"))` with `sc.Offsets("slope", times=sc.Function(xtilde, of="det_x"))` |
| windowed or binned read-outs | a reader returning the window's full-detector `coords` |

## 4. With code: an instrument

For another exposure layout or detector, subclass `sc.Instrument`: a frozen dataclass whose
fields are its settings, implementing `geometry(oversample)`, the detector; everything else has a
default. The chunk maps are functions of that geometry (`mycam.py`, importable by the worker
processes):

```python
from dataclasses import dataclass

import numpy as np
import selfcal as sc
from selfcal.instruments.camera import grid_chunks
from selfcal.io.frames import frame_header_values


def amplifiers(geometry, n):                          # a chunk map: n strips, one per amplifier
    return sc.Chunks.rectangles(geometry.shape, (1, n), axes=("all", "amp"))


@dataclass(frozen=True, kw_only=True)
class MyCam(sc.Instrument):
    chunks: int = 8
    tag: str = "MyCam"                                # product-name tag
    unit: str = "e-/s"                                # the mosaic's BUNIT

    def geometry(self, oversample):                   # the detector: its shape and per-pixel maps
        return sc.Geometry((2048, 2048), oversample,
                           aux={"pix_angle": np.load("pix_angle.npy")})   # detector variables

    # optional: the chunk maps of a model that names none, the first primary
    def default_chunk_maps(self):
        return {"grid": sc.ChunkMap(grid_chunks, tag=f"Chunks{self.chunks}x{self.chunks}",
                                    chunks=(self.chunks, self.chunks)),
                "amps": sc.ChunkMap(amplifiers, n=2)}

    def layout(self):                                 # how a raw exposure is read
        return sc.ExposureLayout(sci_ext=[1], dq_ext=[3], detector_ids=[0],
                                 reader=None)        # or your reader (below)

    # optional: per-frame values every model can read by name
    def frame_variable_names(self):
        return super().frame_variable_names() + ("hwp",)

    def frame_variables(self, frames):
        out = super().frame_variables(frames)         # exposure, detector
        out["hwp"] = frame_header_values(frames, ["HWPANG"])["HWPANG"]
        return out


FIELD = sc.Field("/data/runs/mycam_field1", MyCam(chunks=16), pixel_scale=1.0)
```

It may also override `job_geometry(geom, job)` (the valid pixels and weights of a job; default: 1
on the chunks of the primary map inside the job's window for the solve, inside its strict set for
the mosaic, 0 elsewhere) and `frame_groups(frames)` (the groupings an offset term can share an
offset over; default: the integer-valued frame variables). The engine receives the object itself
and calls it only through these methods; there is no registry. The jobs are named at each action
(`jobs=sc.Job("All")`, or jobs with a `window` on the primary map's axes).

Each action (`plan`, `calibrate`, `mosaic`) builds the geometry once and shares it between its
runs. A class whose geometry depends on nothing but its settings and the data files it names in
`geometry_files()` may say so with the class constant `geometry_is_pure = True` (as `sc.SPHEREx`,
`sc.Euclid` and `sc.Camera` do): the engine then keeps the last two geometries built for the
process, between actions, and builds one again when its settings or one of those files changed.
`MyCam` above reads `pix_angle.npy` without naming it, so it says nothing, and each action builds
its geometry (as it would if it read module state).

**Raw data in any format** — a reader returns, for one detector frame of one file, the values, a
header carrying the celestial WCS (and any keywords you want as frame variables), an optional bit
mask, extra per-pixel planes (stored with the frame as layers) and optional detector coordinates
(a focal plane, the full detector of a window):

```python
from selfcal.io.frames import ExposureData

def read_mycam(path, sci_ext, dq_ext=None, header_only=False):
    hdr = ...                                    # an astropy Header with the WCS (+ keywords)
    if header_only:
        return ExposureData(header=hdr, shape=(ny, nx))
    return ExposureData(header=hdr, data=values, mask=bits,
                        layers={"variance": var}, coords=(x_focal, y_focal))
```

`sci_ext` / `dq_ext` are the integers the layout lists — extension numbers, slice indices,
detector numbers; the reader interprets them. Give it to a camera (`sc.Camera(..., reader=read_mycam)`)
or to an instrument's layout. The reprojection, the reference-frame definition and everything
downstream use the reader; nothing else changes.

Optional hooks, with defaults: `aux_coadds(geom)` and
`finalize_mosaic(geom, mosaicker, maps, sigma)` (extra per-pixel maps to coadd).

**A chunk map of any shape** is a function `fn(geometry, **params)` that returns
`sc.Chunks(det, axes, ...)`:

* `det`: any integer map of the detector's shape, `-1` for no chunk;
* `axes`: the named coordinates of its chunk grid (`sc.ChunkAxes.row_major(names, sizes, scans)`),
  which the priors, the jobs' windows and the terms' `smooth` name; `adjacency_axes=`, the axes a
  term's `smooth` follows by default (without them, `smooth` needs `smooth_along`; the plan refuses
  smoothing with no axis); `spectral_axis=`, the axis along which the wavelength changes;
* `grid=`: the map sampled `geometry.oversample` times per pixel along each axis, for the mosaic
  (None: the engine replicates `det`); `extra=`: what the map's own functions read later;
* `renderers={name: make}` and `render=`: how the mosaic draws the map's offsets. `make(chunks,
  jobgeom)` returns the `renderer(chunk_map, offsets) -> image` of one job; `render` names the one
  used unless an offset term names another (`sc.Offsets(render=name)`); None, or the name
  `"constant"`: constant over each chunk. A renderer that fills the chunks the mosaic zeroed, those
  below `sc.Coadd(min_chunk_coverage=)`, from the frame's other chunks says so with a true attribute
  `fills_invalid`; the pixels of a zeroed chunk that the frame observed and that nothing fills are
  left out of the coadd.

A model names the function as `sc.ChunkMap(fn, tag=, **params)`, an instrument's
`default_chunk_maps()` likewise; `sc.Chunks.rectangles(shape, chunks, axes=)` builds a grid of
rectangles. The instruments of
`tests/test_any_telescope.py` are small reference implementations. The built-in instruments are
subclasses of `sc.Instrument` too: `selfcal/instruments/camera.py` (`sc.Camera`) is the minimal
one, and `selfcal/instruments/spherex/settings.py` (`sc.SPHEREx`: a reader of its own, wavelength
maps, the LVF arc fit its chunk maps read) and `selfcal/instruments/euclid/settings.py`
(`sc.Euclid`: 16 detectors per exposure, default grid, stripe and tilt maps with their renderers)
the full ones. The SPHEREx chunk maps on the arcs, with their renderer, are model-side functions
of the same kind, in `selfcal_scripts/spherex/chunks.py`.

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

then calibrate them in place, `field.calibrate(recipe, frames=frame_dir)`, on a field whose
directory holds the reference grid (`ref.fits`, written with `selfcal.geometry.wcs_helper.save_to_fits`).

## 6. Limits

* Each observation samples one reference pixel per sky term. A forward model that mixes several
  sky pixels into one observation (slitless dispersion, a PSF deconvolution) needs a projection
  operator per observation in the row assembly — not available yet.
* The model is linear. Multiplicative effects (gains, flats, non-linearity) are linearised
  around a previous sky (a `sc.SolvedSky` variable) and iterated.
* The N-pass solve (`calibrate(passes=...)`) runs sky terms with coefficients of any variables;
  offset terms with a coefficient or basis run in a plain calibration.
* A frame holds at most one observation per reference pixel. Samples that revisit a pixel within
  one exposure (a scanning instrument's time-ordered data) go into several frames, which share
  their offsets through a grouped term (e.g. `per="exposure"`).
* The sky is a 2-D grid (a WCS image). Another pixelisation (HEALPix) can be flattened to one row
  of `N` pixels; each frame then stores the range of pixel indices it touches.

## 7. Where things are

| what | where |
| --- | --- |
| the Python settings: model, recipe, machine | `selfcal/models/model.py`, `selfcal/run/recipe.py`, `selfcal/run/compute.py` |
| the instrument contract (`sc.Instrument`) and the built-in instruments | `selfcal/instruments/contract.py`, `selfcal/instruments/camera.py`, `selfcal/instruments/spherex/settings.py`, `selfcal/instruments/euclid/settings.py` |
| the model as the engine takes it (variables, terms, priors) | `selfcal/models/spec.py` |
| data variables and their sources | `selfcal/models/variables.py` |
| ready-made priors | `selfcal/priors.py` (`sc.priors`), over `selfcal/models/priors.py` |
| generic offset-structure builders on chunk axes | `selfcal/models/offset_structure.py` |
| the geometry types (`Chunks`, `DetectorGeometry`, `JobGeometry`, `ExposureLayout`) | `selfcal/instruments/base.py` |
| the model's chunk maps (`sc.ChunkMap`) and the run engine that evaluates them | `selfcal/models/model.py`, `selfcal/run/engine.py` |
| exposure readers, the frame file, header values | `selfcal/io/frames.py` |
| the actions and the run engine behind them | `selfcal/run/` |
| calibration-file reader | `selfcal/io/calfile.py` |
| the Python API | [The Python API](guide/python-api.md) |
| on-disk products, tuning knobs | [Pipeline runbook](guide/pipeline.md) |
