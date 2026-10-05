# The Python API

A run is configured in Python: settings objects that are checked when they are built, a field
that holds the data, and actions that run on it. Run configs written in TOML keep working
unchanged (see [Run configuration](configuration.md)); the Python objects lower onto the same
run engine, so a run configured either way makes the same bytes.

## In one screen

```python
import selfcal as sc

camera = sc.Camera((64, 64), chunks=(4, 4), dq_ext=2, tag="Sim")            # what took the data
field = sc.Field("quickstart_output/quickstart", camera, pixel_scale=10.0)   # one data set and its folder

if __name__ == "__main__":                 # worker processes import this file again: declarations only above
    field.reproject("quickstart_output/exposures/sim_*.fits", method="interp", padding=8)
    result = field.calibrate(sc.continuum(smooth=0.1))
    print(result)
```

- **Settings are checked when built.** A misspelt keyword names the closest setting
  (`sc.Fit(iteratons=100)` asks "did you mean 'iterations'?"), a value of the wrong type or
  outside its choices says what is expected, and rules across settings are checked
  (`sc.Coadd(clip=2.0, std=False)` is refused: the clip needs the std map). Objects are
  immutable: `recipe.replace(name="v2")` makes a checked copy. `repr` prints the Python that
  rebuilds an object.
- **Every action plans first.** `field.plan(recipe, jobs=...)` (and every action, before it
  starts) builds the instrument's geometry, checks the model against it, finds the frames and the
  existing products, and imports every function the run sends to the worker processes in a fresh
  worker. It takes seconds and computes nothing; `print(field.plan(...))` shows what would run.
- **Every action leaves a record and a log**: `<field>/records/<action>_<time>_<pid>.json` (the
  resolved settings, what they lowered to, the code version, the products, the outcome) and
  `<field>/logs/` (the console output, when run from a script).

## The pieces

| object | what it is |
| --- | --- |
| `sc.Camera`, `sc.SPHEREx`, `sc.Euclid`, an `sc.Instrument` subclass | the instrument |
| `sc.Field(path, instrument, pixel_scale, compute=)` | one data set: `path` holds `ref.fits`, `reprojected/`, `calibration/`, `mosaic/`, `logs/`, `records/` |
| `field.reproject`, `.calibrate`, `.mosaic`, `.plan`, `.result` | the actions |
| `sc.Model(sky=, offsets=, ...)`, `sc.continuum()`, `sc.spectral()`, `sc.two_block()` | the model |
| `sc.Recipe(model, fit=sc.Fit(), coadd=sc.Coadd(), numerics=sc.Numerics(), name=)` | everything that decides the numbers |
| `sc.Compute(scratch, workers=, ...)` | the machine (never changes a byte) |
| `sc.Tiles`, `sc.Passes` | big fields: tiling, the N-pass solve |
| `sc.Result` | the products, with readers |

### Instruments and jobs

```python
from selfcal.instruments import spherex

nep = sc.Field(f"{OUTPUTS}/SPHEREx_NEP_2026W17_D3_6p2arcsec", sc.SPHEREx(3, num_col=3), 6.2)
nep.calibrate(RECIPE, jobs=spherex.channel(17))                     # one map
nep.calibrate(RECIPE, jobs=spherex.channels(1, 34))                 # 34 maps, the frames staged once
nep.calibrate(RECIPE, jobs=spherex.group(17, 18))                   # one map over two channels
nep.calibrate(RECIPE, jobs=spherex.window("Multiline3", subchannels=range(200, 321)))
```

A `Camera` or `Euclid` field has one job and needs no `jobs=`. `sc.Camera` takes a custom
exposure `reader`, per-pixel `detector_maps` and per-frame `headers` as data variables; any
other chunk geometry is a subclass of `sc.Instrument` (see below).

### The model

```python
def above_80(temperature, t0=80.0):              # reads the data variable `temperature`
    return temperature - t0

def gradient(det_x, det_y):                      # two basis functions over the detector
    return [(det_x - 32) / 64, (det_y - 24) / 48]

MODEL = sc.Model(sky=[sc.Sky(damping=1e-4)],
                 offsets=[sc.Offsets("thermal", per="all", times=above_80, mean_zero=True),
                          sc.Offsets("gradient", on="detector", basis=gradient, n=2)],
                 variables={"temperature": sc.Header("TEMP")})
```

- `sc.Sky(name, times=, damping=)`: a sky map times a known function of data variables
  (`times`; none: a constant sky). `damping` pulls the map toward zero, weighted by coverage
  (default 0.1 for the first sky term, 0.3 for the others).
- `sc.Offsets(name, on=, per=, ...)`: an offset per chunk of the chunk map `on` (default the
  primary map; `"detector"`: the whole detector), shared by `per` (`"frame"`, `"all"`, or a
  frame variable such as `"detector"`), optionally times one function (`times=`) or `n` basis
  functions (`basis=`, `n=`), or a polynomial (`polynomial=sc.Poly(2, window=range(200, 321))`).
  Priors: `smooth` along `smooth_along` (default the map's own axes), `poly_prior` (soft
  `sc.Poly`), `mean_zero`, `damping`.
- **Functions** are Python functions of data variables: their parameters without defaults name
  the variables they read. `sc.Function(np.cos, of="phase")` says it explicitly, and binds
  parameters (`sc.Function(above_80, t0=70.0)`). Built-in shapes: `sc.template(file)`,
  `sc.gaussian(center, sigma=)`, `sc.linear(center, halfwidth)`, `sc.catalog(name)`.
- **Data variables** beyond the built-in coordinates and the instrument's maps:
  `sc.Header(key)`, `sc.PerFrame(fn)`, `sc.DetectorMap(array | file | fn)`, `sc.SkyMap(...)`,
  `sc.SolvedSky(cal)`, `sc.Layer()`, `sc.Derived(fn, of=)`, `sc.FrameFunction(fn)`.
- **Priors**: `sc.priors.frame_smoothness(term, variable)`, `sc.priors.sky_smoothness(term)`,
  `sc.priors.toward(term, variable)`, or your own function wrapped in `sc.Prior(fn, terms)`.
- **Presets**: `sc.continuum(smooth=0.1, poly_prior=None)`,
  `sc.spectral(lines, polynomial=None, smooth=None, poly_prior=None)`,
  `sc.two_block(second="readout")`; `spherex.line("aromatic", damping=5e-3)` is a sky term with
  a shipped line template.

### The recipe

```python
SUBCHANNEL = sc.ChunkGroups.along("subchannel")         # clip within each subchannel's chunks

NUMCOL3 = sc.Recipe(sc.continuum(smooth=0.1),
                    fit=sc.Fit(50, clip=sc.Clip(5.0, per=SUBCHANNEL)),
                    coadd=sc.Coadd(clip=2.0, oversample=2, ignore_flags=[21]),
                    name="damp0p1_reg0p1_outThresh5_sigma2")    # the products' suffix
```

`sc.Fit` is the solve (iterations, the outlier clip, masks, weights, tolerance, solver,
per-frame hooks); `sc.Coadd` the mosaic (`None`: no mosaic); `sc.Numerics` the summation
layout (threads and batch sizes). `Numerics` belongs to the recipe because it changes the last
bits of the products; its defaults are production's, so a recipe reproduces production on any
machine. A clip compares each observation with the robust spread of its group: its frame
(`per="frame"`, the default), its chunk (`per="chunk"`), or the chunk groups the run script
defines, `sc.ChunkGroups.along(axis)` (chunks with equal values of an axis of the primary chunk
map) or `sc.ChunkGroups.mapping(chunk_to_group)`. An observation belongs to the group of the chunk
that contributes most to it; along SPHEREx's spectral axis the groups are the subchannels, binned by
wavelength (the clip of the N-pass solve, whose passes group along that axis only).

### The machine

```python
ORCA = sc.Compute("/home/me/selfcal/cache", workers=48, coadd_workers=96)
field = sc.Field(path, sc.SPHEREx(3), 6.2, compute=ORCA)
```

`scratch` is fast local disk: frames are staged there (a verified copy into a directory the run
owns) and the solver and the coadd keep their intermediates there. Without it, frames are read
in place. Every action pins the BLAS and OpenMP threads to one per process before any worker
starts (the solver's thread pool is the parallelism).

### Worker processes and functions

The pipeline's worker processes start fresh and import the run script again, so:

- the script's work goes under `if __name__ == "__main__":` (an action started outside it is
  refused);
- every function a run uses must be importable: defined with `def` at the top level of a module
  or of the run script; lambdas, nested functions and functions defined in a notebook cell are
  refused when the setting is built, with the fix (in a notebook: write the functions to a
  module, `%%writefile myfuncs.py`, and import them). A function of the run script is recorded as
  `<script stem>:<name>`.

### Big fields

```python
LINES = [spherex.line(n, damping=5e-3) for n in ("aromatic", "aliphatic", "plateau")]
MULTILINE = sc.Recipe(sc.spectral(LINES, polynomial=sc.Poly(2, window=range(200, 321))),
                      fit=sc.Fit(300, ignore_flags=[21], shot_noise_weights=True), coadd=None,
                      name="multiline3_SEP6")
result = sep.calibrate(MULTILINE, jobs=spherex.window("Multiline3", subchannels=range(200, 321)),
                       tiles=sc.Tiles(boxes=SEP_BOXES),
                       passes=sc.Passes(5, sky_clip=sc.Clip(5.0, per=SUBCHANNEL),
                                        offset=sc.Refit(clip=sc.Clip(2.5, per=SUBCHANNEL))))
aromatic = result.sky("aromatic")               # from the last SKY pass
```

`sc.Tiles` solves the field tile by tile and stitches the tiles' skies; `sc.Passes` runs the
N-pass alternating solve (`order="offset_first"` by default; a schedule ending on an OFFSET pass
is refused unless `ends_on_offset=True`).

### Results

`field.calibrate(...)` returns a `sc.Result`: `result.cal_paths`, `result.mosaic_paths`,
`result.final` (the cal holding the field's final sky), `result.sky(name)`,
`result.offsets(name)`, `result.cal()` / `result.mosaic()` (readers, in a `with` block),
`result.show()` (a mosaic with a colour bar), `result[job]`. `field.result(recipe, jobs=...)`
finds the products of an earlier run without running anything.

## A new telescope

Most cameras need no code (`sc.Camera(..., reader=read_mycam, detector_maps={"pix_angle": a},
headers={"hwp": "HWPANG"})`). Another chunk geometry is a frozen-dataclass subclass of
`sc.Instrument` whose fields are its settings and which implements `geometry(oversample)`:

```python
from dataclasses import dataclass

@dataclass(frozen=True, kw_only=True)
class Owl(sc.Instrument):                         # two amplifier strips on a 2048 x 2048 detector
    chunks: int = 8
    tag: str = "Owl"
    unit: str = "e-/s"

    def geometry(self, oversample):
        grid = sc.ChunkMap.rectangles("grid", (2048, 2048), (self.chunks, self.chunks))
        amps = sc.ChunkMap.rectangles("amps", (2048, 2048), (1, 2), axes=("all", "amp"))
        return sc.Geometry((2048, 2048), oversample, maps=[grid, amps])
```

It may also override `layout()` (how raw exposure files are read), `default_jobs()`,
`job_geometry()` and `frame_variables()`. The engine receives the object itself; no registry
is needed. See [Bring your own telescope](../bring_your_own_telescope.md) for the instrument
contract in full.

## From a TOML config

`selfcal.run.convert.from_runconfig(load_config("x.toml"))` reads a TOML run config into these
objects (`.to_python()` writes the equivalent script); `selfcal_scripts/gates/config_equivalence.py
typed` checks, for every shipped config, that the converted objects make the run engine do
exactly what the TOML does. Where each TOML key goes:

| TOML | Python |
| --- | --- |
| `task` | the action: `field.reproject`, `field.calibrate` (`tiles=`, `passes=`), `field.mosaic`; `precompute`: `spherex.precompute_lvf(detectors)` |
| `mode` + `[params]`, `[model]` | the `Model` (a preset or spelled out) |
| `output_dir` + `run_name`, `resolution_arcsec` | `Field(path, ..., pixel_scale)` |
| `cache_dir`, `staging`, `keep_nvme`, `hdd_io_limit` | `Compute(scratch, stage=, keep_staged=, io_limit=)` |
| `suffix` | `Recipe(name=)` |
| `apply_n_threads`, `batch_size`, `cache_batch_size`, `coadd_batch_size` | `Numerics(threads, batch=, mosaic_batch=, coadd_batch=)` |
| `[calibration]`, `[lsqr]` | `Fit` (and `Sky(damping=)` per term) |
| `[mosaic]`, `oversample`, `wavelength_coadd`, `skip_mosaic` | `Coadd` (`None`: no mosaic) |
| `n_frames`, `reproj_override` | `calibrate(frames=n | directory | sc.frames_in(directory)[:n])` |
| `[instrument]` | `sc.SPHEREx` / `sc.Euclid` / `sc.Camera` and `jobs=` |
| `[tiling]`, `[passes]` | `sc.Tiles`, `sc.Passes` |
| `[hooks]`, `postprocess` | `Fit(frame_hook=, raw_frame_hook=)`, `Coadd(frame_hook=)` |
| `[zodi]` | `spherex.zodi_anchor(result, predictions=...)` after the calibration |
