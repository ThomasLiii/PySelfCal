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
**model** (`sc.Model`). A new telescope, sky coefficient, offset set-up or prior is a function you
write — never an edit of `selfcal`.

## 1. No code: `sc.Camera`

Any imager whose exposures are FITS files with a science image + a celestial WCS in one
extension and (optionally) an integer mask in another:

```python
import selfcal as sc

CAMERA = sc.Camera((2048, 2048), chunks=(8, 8), dq_ext=2, tag="MyCam")   # dq_ext=None: no mask
FIELD = sc.Field("/data/runs/mycam_field1", CAMERA, pixel_scale=1.0,
                 compute=sc.Compute("/scratch/selfcal", workers=16))
RECIPE = sc.Recipe(sc.continuum(smooth=0.1,                               # smoothness between chunks
                                poly_prior=sc.Poly(1, along="col", weight=0.5)),   # optional
                   fit=sc.Fit(100, clip=5.0),
                   coadd=sc.Coadd(clip=3.0),
                   name="v1")

if __name__ == "__main__":
    FIELD.reproject("/data/mycam/field1/*.fits", method="interp", padding=100)
    print(FIELD.calibrate(RECIPE))
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

In a TOML config the same camera is the built-in `grid` instrument (`[instrument] name = "grid"`,
`detector_shape`, `chunks`, `sci_ext`, `dq_ext`, `tag`; [Run configuration](guide/configuration.md)).

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
    offsets=[sc.Offsets("chunks", smooth=0.1, mean_zero=True,  # free per-frame offsets on the chunk grid
                        poly_prior=sc.Poly(1, along="col", weight=0.5)),
             sc.Offsets("thermal", per="all",                  # a detector-fixed pattern times the temperature
                        times=sc.Function(above, of="temperature", t0=80.0)),
             sc.Offsets("gradient", on="detector",            # a free 2-D gradient per frame
                        basis=plane, n=2)],                    # plane(det_x, det_y) returns 2 arrays
    variables={"time": sc.Header("MJD-AVG"),                   # a keyword of each frame's header
               "temperature": sc.Header("TEMP"),
               "variance": sc.Layer("variance"),               # a plane stored with each frame
               "psi": sc.Derived(polariser_angle, of=["hwp", "pix_angle"])},
    weight=sc.Function(inverse_sigma, of="variance"),          # optional: 1/σ observation weights
    priors=[sc.priors.frame_smoothness("chunks", "time", weight=0.2)])   # any linear rows
```

`print(MODEL)` shows the model; `field.plan(recipe)` checks it against the instrument (every
variable it reads, every chunk map and axis it names) and imports every function in a fresh worker
process, in seconds, before anything runs.

### Data variables

| source | Python | one value per | examples |
| --- | --- | --- | --- |
| built-in | — | observation | `det_x`, `det_y` (detector position), `sky_x`, `sky_y` (reference pixel), `frame` |
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
  `sc.linear` —, `sc.catalog(name)` for an instrument's named coefficient, or the name of a
  variable, which then multiplies the map itself; omitted, the sky is constant. `damping` pulls the
  map toward zero, weighted by coverage.
* **Offset term**, `sc.Offsets(name, on=, per=, ...)`: on a chunk map of the instrument (`on`;
  default the primary map; `"detector"`: the whole detector as one chunk), shared by
  `per="frame"` (each frame its own), `per="all"` (all frames one) or `per=` any frame variable
  (frames with equal values); `times=φ` (one known function multiplying the offset at every
  observation) or `basis=fns, n=n` (`n` functions: the unknowns are one coefficient per group ×
  chunk × function), or `polynomial=sc.Poly(degree, along=, window=)` (an exact polynomial along a
  chunk axis); and the built-in priors `smooth` + `smooth_along`, `poly_prior`, `mean_zero` (per
  function for a basis), `damping`, `exact_group_rows`.
* **weight**, `sc.Model(weight=fn)`: a function of data variables multiplying every observation's
  row weight (`1/σ` for an inverse-variance fit); the mosaic coadds with its square, like the solve.
* **prior**, `sc.Prior(fn, terms, weight=, **params)`: `terms` names the terms the rows act on
  (sky-term names, offset-term names, `"scalar"`). A prior function receives one `TermInfo` per
  term (the unknowns' shape, their coverage, the frame → group map, the solve's variables, the
  chunk axes, `index(...)` for the global unknown ids) and returns `(rows, cols, vals, rhs)`.
  Ready-made (`sc.priors`): `frame_smoothness` (offsets of neighbouring frames in a frame
  variable), `sky_smoothness` (neighbouring sky pixels), `toward` (unknowns toward a known map or
  per-frame value).
* **grouped clip**: `sc.Fit(clip=sc.Clip(5.0, variable="<variable>", edges=[...]))` judges each
  observation against its own group's distribution; `per="chunk"` or
  `per=sc.ChunkGroups.along("<axis>")` groups by chunks instead.

Anchors need care when a term's groups observe different parts of its map (detectors on one
focal plane): `mean_zero` sums over every chunk of the map, observed or not, so anchor such a
term with `damping` (it acts on observed unknowns only) or a prior.

In a TOML config the model is a `[model]` table (`[[model.sky]]`, `[[model.offset]]`,
`[model.variables]`, `[[model.prior]]`, functions as `"package.module:name"` strings);
`selfcal convert` writes the Python form of one.

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
| windowed or binned read-outs | a reader returning the window's full-detector `coords` |

## 4. With code: an instrument

For a real geometry, subclass `sc.Instrument`: a frozen dataclass whose fields are its settings,
implementing `geometry(oversample)`; everything else has a default.

```python
from dataclasses import dataclass

import numpy as np
import selfcal as sc
from selfcal.io.frames import frame_header_values


@dataclass(frozen=True, kw_only=True)
class MyCam(sc.Instrument):
    chunks: int = 8
    tag: str = "MyCam"                                # product-name tag
    unit: str = "e-/s"                                # the mosaic's BUNIT

    def geometry(self, oversample):                   # chunk maps (any int map, -1 = no chunk)
        grid = sc.ChunkMap.rectangles("grid", (2048, 2048), (self.chunks, self.chunks))
        amps = sc.ChunkMap.rectangles("amps", (2048, 2048), (1, 2), axes=("all", "amp"))
        return sc.Geometry((2048, 2048), oversample, maps=[grid, amps],
                           aux={"pix_angle": np.load("pix_angle.npy")})   # detector variables

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

It may also override `default_jobs()` (the jobs a run makes) and `job_geometry(geom, job)` (the
valid pixels and weights of a job). The engine receives the object itself; no registry is needed.

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

Optional hooks, with defaults: `offset_renderer(geom, jobgeom, map_name=None, render=None)`
(a smooth chunk-to-pixel rendering for the mosaic), `aux_coadds(geom)` and
`finalize_mosaic(geom, mosaicker, maps, sigma)` (extra per-pixel maps to coadd), and
`coefficient_catalog()` (named coefficients, for `sc.catalog(name)`). A chunk map other than
rectangles is `sc.ChunkMap(name, ids, ids, axes=sc.ChunkAxes.row_major(...),
adjacency_axes=(...))`: any integer map, `-1` for no chunk, with named axes for the priors and the
axes a term's `smooth` follows by default (without them, `smooth` needs `smooth_along`; the plan
refuses smoothing with no axis). The instruments of
`tests/test_any_telescope.py` are small reference implementations, `selfcal/instruments/camera.py`
the minimal one, and `selfcal/instruments/spherex/settings.py` (spectral, non-rectangular chunks,
wavelength maps) and `selfcal/instruments/euclid/settings.py` (16 detectors per exposure, stripe
and tilt maps) the full ones.

TOML configs reach an instrument by name: there it is a subclass of the engine's
`selfcal.instruments.base.Instrument` registered with `@register_instrument("mycam")`, in-tree or
through the `selfcal.instruments` entry-point group
(`[project.entry-points."selfcal.instruments"] mycam = "mypkg.instrument:MyCam"`).

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
| instruments as settings: the contract, the camera | `selfcal/instruments/contract.py`, `selfcal/instruments/camera.py` |
| the model as the engine takes it (variables, terms, priors) | `selfcal/models/spec.py` |
| data variables and their sources | `selfcal/models/variables.py` |
| ready-made priors | `selfcal/priors.py` (`sc.priors`), over `selfcal/models/priors.py` |
| generic offset-structure builders on chunk axes | `selfcal/models/offset_structure.py` |
| the engine's instrument contract + registry | `selfcal/instruments/base.py` |
| exposure readers, the frame file, header values | `selfcal/io/frames.py` |
| the run engine (actions and TOML configs → tasks) | `selfcal/run/` |
| calibration-file reader | `selfcal/io/calfile.py` |
| the Python API, the TOML schema | [The Python API](guide/python-api.md), [Run configuration](guide/configuration.md) |
| on-disk products, tuning knobs | [Pipeline runbook](guide/pipeline.md) |
