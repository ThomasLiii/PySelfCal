# selfcal

selfcal self-calibrates and mosaics astronomical images. From many overlapping exposures of a field
it solves jointly for the sky and for the instrument's additive offsets, as one sparse linear
least-squares problem solved iteratively with LSQR, then coadds the calibrated frames into a mosaic.
It was built for SPHEREx, an all-sky spectral survey through a linear variable filter (LVF), and it
also calibrates Euclid NISP exposures (16 detectors each). Other imagers need no change to the
package: a single-detector FITS camera is described entirely in the run config (the built-in `grid`
instrument), and an instrument with its own geometry or raw format is a small `Instrument` subclass
with five required methods.

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
anchor, or any linear rows you write. A run config picks a preset recipe (a *mode*) or spells the
terms out in a `[model]` table; both become a [`ModelSpec`][selfcal.models.spec.ModelSpec]. See
[How selfcal works](guide/concepts.md) for the terms and priors, and
[Bring your own telescope](bring_your_own_telescope.md) to write your own.

## What a run looks like

The library is used from Python (`import selfcal`); runs go through the run engine, which reads one
TOML file per run. This config calibrates a 2048 x 2048 camera with the `continuum` mode: one sky
value per pixel, free offsets on an 8 x 8 grid of chunks with a mean-zero anchor per frame, and a
per-frame scalar.

```toml
task = "cal"                       # solve, then coadd the mosaic
mode = "continuum"                 # the calibration recipe
output_dir = "/data/runs/"
run_name = "mycam_field1"          # products go to <output_dir>/<run_name>/
resolution_arcsec = 1.0            # pixel scale of the reference grid
cache_dir = "/scratch/selfcal/"    # staging area for the frames

[instrument]                       # the telescope: here the config-only grid imager
name = "grid"
tag = "MyCam"                      # product-name tag
detector_shape = [2048, 2048]      # rows, columns
chunks = [8, 8]                    # offsets on an 8 x 8 grid of detector chunks
dq_ext = 2                         # FITS extension of the bit mask

[params]                           # knobs of the mode
reg_weight = 0.1                   # smoothness between neighbouring chunks

[calibration]                      # keywords of the system build (setup_lsqr)
apply_weight = false
offset_regularization = true       # enables the reg_weight smoothness rows
weighted_damping = true            # enables the damp_weight rows
damp_weight = 0.1                  # coverage-weighted damping of the sky

[lsqr]                             # keywords of the solve (apply_lsqr)
solver = "lsqr"
damp = 0
iter_lim = 100

[mosaic]                           # keywords of the coadd (make_mosaic)
apply_weight = false
make_std_map = true
apply_sigma_clipping = true
sigma = 3.0
```

```bash
./selfcal_scripts/run.sh cal.toml --dry-run   # check the config: resolve the jobs and the mode
./selfcal_scripts/run.sh cal.toml             # run (from the repository root)
```

The `task` key selects the step; a job is one unit of the instrument's loop (a SPHEREx channel or
window; the `grid` instrument has one, `All`). Paths are relative to `<output_dir>/<run_name>/`.

| `task` | What it does | Writes |
| --- | --- | --- |
| `reproject` | Reads the raw exposures, defines the reference grid and resamples every frame onto it | `ref.fits`, `reprojected/exp_*_det_*.h5` |
| `cal` | Builds and solves the system for each job, then coadds the mosaic. With a `[tiling]` table it solves the field tile by tile and stitches the tile calibrations (no mosaic) | `calibration/cal_*.h5`, `mosaic/mosaic_*.fits` |
| `mosaic` | Coadds the frames with an existing calibration | `mosaic/mosaic_*.fits` |
| `npass` | The N-pass alternating solve: a `cal` run, then exact sky and offset passes in turn | a `calibration/cal_*.h5` file per pass |
| `precompute` | Runs the instrument's rarely needed geometry generator | SPHEREx: the LVF arc parameters, `lvf_params_D<n>.npy` in the package's data directory |

The config above reads the frames that an earlier `reproject` run with the same `output_dir`,
`run_name` and `[instrument]` table wrote, and writes `calibration/cal_MyCam_Chunks8x8_All.h5` (the
sky map, the offsets of every frame and chunk, the per-frame scalars and the coverage; read it with
[`CalFile`][selfcal.io.calfile.CalFile]) and `mosaic/mosaic_MyCam_Chunks8x8_All.fits` (mean,
standard-deviation and sigma-clipped mean maps, each with a weight map; SPHEREx mosaics add
wavelength maps). Each run also writes a log to `logs/`.

## Where to go next

- [Installation](getting-started/installation.md): requirements, installing, checking the install.
- [Quickstart](getting-started/quickstart.md): a first run on simulated exposures, from the raw
  FITS files to the mosaic, checked against the injected truth.
- [How selfcal works](guide/concepts.md): the method, the model, the priors and the mosaic.
- [Run configuration](guide/configuration.md): the config schema, the tasks, the modes and the
  `[model]` table.
- [Bring your own telescope](bring_your_own_telescope.md): a new instrument or model, from
  configuration alone to an `Instrument` subclass.
- [Pipeline runbook](guide/pipeline.md): tuning knobs, the N-pass solve, staging, and the on-disk
  formats of the frame, calibration and mosaic files.
- [API reference](reference/index.md): every module of `selfcal` and of the run engine
  `selfcal_scripts.runner`, generated from the docstrings.

The [Glossary](guide/glossary.md) defines the terms used throughout.
