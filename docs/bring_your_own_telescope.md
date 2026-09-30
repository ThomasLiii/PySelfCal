# Bring your own telescope

`selfcal` fits, for a set of overlapping exposures, the model

```
data(frame, pixel) = Σ_j S_j(pixel) · c_j(λ_pixel)  +  Σ_m O_m(frame, chunk_m(pixel))  +  s(frame)
```

a per-pixel **sky** (one or more terms), a per-frame **offset** on one or more chunk partitions
of the detector, and a per-frame **scalar**, by sparse least squares; then it coadds the corrected
frames into a mosaic. Nothing in the solver knows a telescope. A telescope enters through an
*instrument* (how exposures are read, how the detector is partitioned into chunks) and the
calibration is a *model* (which sky and offset terms, with which priors). This page shows both
paths: the built-in `grid` instrument, which needs no code, and a subclass.

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

The named modes are presets of one thing, a list of sky terms and offset terms. Spell it out to
change it — no Python:

```toml
mode = "model"

[model]
scalar = true                                    # per-frame scalar

[[model.sky]]
name = "continuum"                               # no coefficient: a constant sky

[[model.offset]]                                 # free per-frame offsets on the chunk grid
kind = "free"
reg_weight = 0.1
adjacency = ["row", "col"]                       # the grid instrument's chunk axes
mean_zero = true
poly = [ { axis = "col", degree = 1, weight = 0.5 } ]

[[model.offset]]                                 # plus a pattern shared by every frame (kind = "fixed"),
kind = "fixed"                                   # or one per detector: kind = "grouped", groups = "detector"
reg_weight = 0.0
mean_zero = true
damp = 0.3                                       # Tikhonov prior toward 0
```

Offset term kinds: `free` (one unknown per frame and chunk), `polybasis` (the offset IS a Chebyshev
polynomial along an axis, one per value of another axis), `fixed` (one vector for all frames),
`grouped` (one per frame group). Priors: `reg_weight` + `adjacency` (smoothness), `poly` (shape),
`mean_zero` (anchor), `damp` (shrinkage). Sky terms: a map times an optional coefficient, which is
any function of data variables the instrument provides for every observation:

```toml
[[model.sky]]
name = "mine"
coefficient = { variable = "u", function = "mypkg.shapes:my_shape", params = { power = 2.0 } }
```

where `my_shape(u, power)` is your own function of the per-observation values of `u` (built-in
shapes `template`, `gaussian`, `linear` need no code). The variables come from the instrument's
`DetectorGeometry.aux` maps; the `grid` instrument provides none, a subclass adds them (below).
`tests/test_sky_coefficients.py` recovers such a term end to end. Full vocabulary:
`selfcal_scripts/configs/README.md`.

## 3. With code: subclass `Instrument`

For a real geometry (non-rectangular chunks, several detectors per exposure, per-pixel wavelength
maps, a smooth offset renderer for the mosaic), implement five methods:

```python
import numpy as np
from selfcal.instruments import (Instrument, register_instrument, Job, ChunkMap,
                                 DetectorGeometry, JobGeometry, ExposureLayout)
from selfcal.models.offset_structure import ChunkAxes
from selfcal.geometry.map_helper import make_grid_chunk_map


@register_instrument("mycam")                    # selects it: [instrument] name = "mycam"
class MyCam(Instrument):
    capabilities = frozenset()                   # add "wavelength" if you provide a wavelength map

    def jobs(self, cfg):                         # the run's job loop (one job here)
        return [Job("all")]

    def frame_tag(self, cfg):                    # product-name component
        return f"MyCam{cfg['detector']}"

    def exposure_layout(self, cfg):              # how an exposure file is read
        return ExposureLayout(sci_ext=[1], dq_ext=[3], detector_ids=[0])

    def detector_geometry(self, cfg, oversample):
        det = make_grid_chunk_map((2048, 2048), 8)              # any int map, -1 = no chunk
        grid = np.kron(det, np.ones((oversample, oversample), dtype=det.dtype))
        cm = ChunkMap("grid", det, grid,
                      axes=ChunkAxes.row_major(("row", "col"), (8, 8), ("y", "x")),
                      adjacency_axes=("row", "col"))
        return DetectorGeometry(shape=(2048, 2048), chunk_maps={"grid": cm}, primary="grid",
                                aux={"u": my_variable_map})   # data variables sky coefficients may read

    def job_geometry(self, cfg, geom, job):     # per-job validity weights
        ones = np.ones(geom.shape, dtype=np.float32)
        return JobGeometry(det_valid_weight=ones,
                           grid_valid_weight=np.ones(geom.chunk_map.grid.shape, np.float32))
```

Optional hooks with defaults: `offset_renderer` (smooth chunk-to-pixel rendering for the mosaic),
`aux_coadds` / `finalize_mosaic` (extra per-pixel maps to coadd), `coefficient_catalog` (named coefficients),
`hooks` (per-frame hook factories selectable in `[hooks]`), `postcal_hooks`, `data_unit`,
`frame_groups` (groupings a `grouped` term can use; `"detector"` is provided), `precompute`.
`selfcal/instruments/spherex/adapter.py` (spectral, non-rectangular chunks, wavelength maps) and
`selfcal/instruments/euclid/adapter.py` (16 detectors per exposure, stripe and tilt maps) are the two
reference implementations; `selfcal/instruments/grid.py` the minimal one.

A package of your own can register the instrument through the `selfcal.instruments` entry-point
group (`[project.entry-points."selfcal.instruments"] mycam = "mypkg.instrument:MyCam"`), so it is
selectable by name without touching `selfcal`.

## 4. Where things are

| what | where |
| --- | --- |
| the model as data (terms + priors) | `selfcal/models/spec.py` |
| generic offset-structure builders on chunk axes | `selfcal/models/offset_structure.py` |
| the instrument contract + registry | `selfcal/instruments/base.py` |
| the run engine (config → tasks) | `selfcal_scripts/runner/` |
| calibration-file reader | `selfcal/io/calfile.py` |
| config schema | `selfcal_scripts/configs/README.md` |
| on-disk products, tuning knobs | `PIPELINE.md` |
