# selfcal

selfcal self-calibrates and mosaics astronomical images. From many overlapping exposures of a field
it solves jointly for the sky and for the instrument's additive offsets, as one sparse linear
least-squares problem solved iteratively with LSQR, then coadds the calibrated frames into a mosaic.
It was built for SPHEREx, an all-sky spectral survey through a linear variable filter (LVF), and it
also calibrates Euclid NISP exposures (16 detectors each). Other imagers need no change to the
package: a single-detector FITS camera is one `sc.Camera(...)`, and an instrument with its own
geometry or raw format is a small `sc.Instrument` subclass.

## The model

An observation `i` is one value of one frame (one detector of one exposure) on one pixel `P` of a
common reference grid, seen at some position on the detector. For every observation the solver fits

```text
data_i = Σ_j S_j[P] · c_j(v_i)  +  Σ_m Σ_k O_m[g_m(frame), chunk_m(i), k] · φ_mk(v_i)  +  s(frame)
```

- **Sky terms** `S_j`: a map on the reference grid, shared by all frames, times a known coefficient
  `c_j(v)`. Without a coefficient (`c = 1`) the term is a plain sky map, one value per pixel; a line
  template of the wavelength or a sine of the time makes it a line map or a seasonal term.
- **Offset terms** `O_m`: an additive offset for each chunk of chunk map `m`, a partition of the
  detector into regions (a rectangular grid; for SPHEREx, arcs that follow the filter's lines of
  constant wavelength, cut into columns). The grouping `g_m` sets which frames share an offset:
  each frame its own (`free`), one for all frames (`fixed`), or one per group of frames with equal
  values of a per-frame quantity (`grouped`). With known functions `φ_mk(v)` the unknowns become
  their coefficients, for example a per-frame gradient in detector coordinates; the classic chunk
  offset has a single `φ = 1`.
- **Per-frame scalar** `s`: one additive constant per frame.
- **Data variables** `v`: named per-observation quantities that every coefficient and function
  reads by name: the detector position (`det_x`, `det_y`), the reference pixel (`sky_x`, `sky_y`),
  the instrument's detector maps (SPHEREx: the wavelength), header keywords such as the time,
  planes stored with each frame, and functions of these.

The data alone do not fix every unknown: adding a constant to the sky and subtracting it from every
per-frame scalar leaves every observation unchanged. Priors settle what the data leave free:
damping, smoothness between neighbouring chunks, a polynomial shape along a chunk axis, a mean-zero
anchor, or any linear rows you write. A model is a preset (`sc.continuum()`, `sc.spectral(lines)`)
or spelled out term by term (`sc.Model(sky=[...], offsets=[...], priors=[...])`); the run engine
receives it as a [`ModelSpec`][selfcal.models.spec.ModelSpec]. See
[How selfcal works](guide/concepts.md) for the terms and priors, and
[Bring your own telescope](bring_your_own_telescope.md) to write your own.

## What a run looks like

A run is a short Python script. This one calibrates a 2048 x 2048 camera with the continuum model:
one sky value per pixel, free offsets on an 8 x 8 grid of chunks with a mean-zero anchor per frame,
and a per-frame scalar.

```python
import selfcal as sc

camera = sc.Camera((2048, 2048), chunks=(8, 8), dq_ext=2, tag="MyCam")   # how to read an exposure
field = sc.Field("/data/runs/mycam_field1", camera, pixel_scale=1.0,         # the data set and its directory
                 compute=sc.Compute("/scratch/selfcal", workers=16))       # the machine
recipe = sc.Recipe(sc.continuum(smooth=0.1),     # sky + offsets smooth between chunks + per-frame scalar
                   fit=sc.Fit(100),              # at most 100 LSQR iterations
                   coadd=sc.Coadd(clip=3.0))     # mean, std and 3-sigma clipped mean maps

if __name__ == "__main__":
    field.reproject("/data/mycam/field1/*.fits")   # the reference grid; every exposure resampled onto it
    print(field.plan(recipe))                       # what will run, checked; nothing is computed
    result = field.calibrate(recipe)                # solve, then coadd the mosaic
```

Every setting is checked when it is built, so a misspelt keyword or an impossible value stops the
script at the line that made it. The field's actions:

| Action | What it does | Writes |
| --- | --- | --- |
| `field.reproject(exposures)` | Reads the raw exposures, defines the reference grid and resamples every frame onto it | `ref.fits`, `reprojected/exp_*_det_*.h5` |
| `field.calibrate(recipe)` | Builds and solves the system for each job, then coadds its mosaic. `tiles=` solves the field tile by tile and stitches the tiles; `passes=` runs the N-pass alternating solve | `calibration/cal_*.h5`, `mosaic/mosaic_*.fits` |
| `field.mosaic(recipe, cal=)` | Coadds the frames with an existing calibration | `mosaic/mosaic_*.fits` |
| `field.plan(recipe)` | Checks what `calibrate` would do without computing anything: the frames, the model against the instrument, each product (made, reused or refused) | nothing |
| `spherex.precompute_lvf(detectors)` | SPHEREx's rarely needed geometry generator | the LVF arc parameters, `lvf_params_D<n>.npy` |

A job is one unit of the instrument's loop (a SPHEREx channel or window; a camera has one, `All`).
Paths are relative to the field's directory. The run above writes
`calibration/cal_MyCam_Chunks8x8_All.h5` (the sky map, the offsets of every frame and chunk, the
per-frame scalars and the coverage; read it with `result.cal()` or
[`CalFile`][selfcal.io.calfile.CalFile]) and `mosaic/mosaic_MyCam_Chunks8x8_All.fits` (mean,
standard-deviation and sigma-clipped mean maps, each with a weight map; SPHEREx mosaics add
wavelength maps). A recipe's `name` ends the product names. Each action also writes a log to
`logs/`, a record to `records/` and, next to each product, `<product>.json` with what it was made
from: a product that exists is reused only when it was made by the same inputs.

Runs can also be written as TOML configs, the form of the shipped production configs, which keep
working ([Run configuration](guide/configuration.md)); `selfcal convert run.toml` writes the Python
form of a config and checks that it runs identically.

## Where to go next

- [Installation](getting-started/installation.md): requirements, installing, checking the install.
- [Quickstart](getting-started/quickstart.md): a first run on simulated exposures, from the raw
  FITS files to the mosaic, checked against the injected truth.
- [How selfcal works](guide/concepts.md): the method, the model, the priors and the mosaic.
- [The Python API](guide/python-api.md): the field, the instruments, the model, the recipe, the
  machine, big fields, products, records and reruns, the command line.
- [Run configuration](guide/configuration.md): the TOML form of a run: the config schema, the
  tasks, the modes and the `[model]` table.
- [Bring your own telescope](bring_your_own_telescope.md): a new instrument or model, from
  `sc.Camera` to an `sc.Instrument` subclass.
- [Pipeline runbook](guide/pipeline.md): tuning knobs, the N-pass solve, staging, and the on-disk
  formats of the frame, calibration and mosaic files.
- [API reference](reference/index.md): every module of `selfcal`, the run engine
  `selfcal.run` included, generated from the docstrings.

The [Glossary](guide/glossary.md) defines the terms used throughout.
