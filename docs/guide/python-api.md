# The Python API

A run is configured in Python: settings objects that are checked when they are built, a field
that holds the data, and actions that run on it. A run is a script or a notebook; the actions
lower the objects onto the run engine. TOML run configs are no longer run
([Migrating from TOML](migrating-from-toml.md)).

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
- **Every action leaves a record**: `<field>/records/<action>_<time>_<pid>.json` (the resolved
  settings, what they lowered to, the code version, the products, the outcome); a script's console
  output goes to `<field>/logs/<script>_<time>_<pid>.log`.
- **A product is reused only when it was made by the same inputs**: every product carries a sidecar
  with what made it, and an existing product made by other inputs is refused (see
  [Products, records and reruns](#products-records-and-reruns)).

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
other chunk geometry is a subclass of `sc.Instrument` (see below). `instrument.geometry(oversample)`
is an instrument's detector geometry (its chunk maps, their axes, its per-pixel maps), without
any data: `sc.SPHEREx(4).geometry(1).chunk_maps["subchannel"]`.

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
  a shipped line template. A preset is a plain function that returns a `Model`; a calibration
  variant of your own is one too, or the `Model` itself.

### The recipe

```python
NUMCOL3 = sc.Recipe(sc.continuum(smooth=0.1),               # the NumCol3 channel maps' recipe
                    fit=sc.Fit(50, clip=5.0),              # 50 iterations; a 5-sigma clip in each frame
                    coadd=sc.Coadd(clip=2.0, oversample=2, ignore_flags=[21]),
                    name="damp0p1_reg0p1_outThresh5_sigma2")    # the products' suffix

SUBCHANNEL = sc.ChunkGroups.along("subchannel")              # chunks grouped by subchannel
BY_SUBCHANNEL = NUMCOL3.replace(fit=sc.Fit(50, clip=sc.Clip(5.0, per=SUBCHANNEL)), name="subch_clip")
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

**When the solve stops.** `sc.Fit(iterations=N, tolerance=t)` stops LSQR or LSMR at the first of
their tests: the residual or the least-squares gradient small enough for `t` (`atol = btol = t`;
`tolerance=(atol, btol)` sets them apart), the estimate of cond(A) above 1e8, or N iterations.
These tests read running estimates, and the weakest, largest-scale directions of a selfcal system
can keep converging long after they pass. **`sc.Fit(iterations=N, tolerance=0)` runs exactly N
iterations**: the tolerance tests and the condition-estimate stop are off (only machine precision
or an exact solution can end the solve earlier, and its record says so):

```python
FIXED = NUMCOL3.replace(fit=sc.Fit(700, tolerance=0), name="it700")   # 700 iterations, no stopping test
```

Every solve is recorded, whatever its tolerance: how it stopped and the solver's state at each
iteration (see [Records](#records)).

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

## Products, records and reruns

### A product is reused only when it was made by the same inputs

Every product an action writes (a cal file, a tile cal, a stitched cal, an N-pass product, a
mosaic) is written under a temporary name and renamed when complete, then gets a sidecar,
`<product>.json`, holding its *inputs*: the settings and files that decided its bytes.

| product | inputs |
| --- | --- |
| cal | the instrument, the reference grid (`ref.fits`, by content), the job, the model (template and map files by content), the fit (the N-pass first pass: with its clip), the solve's `Numerics`, the frames (by name) |
| tile cal | the same, plus the tile's box and how frames were assigned to it |
| stitched cal | its tile cals (by fingerprint) |
| N-pass product | the first pass (by fingerprint), the pass number and type, the pass settings |
| mosaic | its cal (by fingerprint), the reference grid, the model, the `Coadd`, the coadd's `Numerics`, the frames |

`Compute` never enters: it leaves the products byte-identical. Two things do not enter either: the
frames' contents (a frame reprojected again under the same name counts as the same frame) and the
code (a record names its version; a product made by older code is current if its inputs are). A
setting added in a later version enters only when it differs from its default, so the products
made before it stay current ([Settings and fingerprints](../developer/settings.md)). Before an
action starts, its plan compares each product that already exists with what it would make: a
product with the same inputs is reused (two recipes that share a first pass share its products); one made by other inputs is
refused, with the differences:

```text
ConfigError: cal_Detector3_..._Ch17_damp0p1.h5 exists but was made with different inputs
(fit.iterations: 50 -> 100). Give the recipe its own name (recipe.replace(name=...)), or pass
overwrite=True to make it again
```

A product written again after its sidecar (its size or modification time no longer the ones
recorded: another program wrote it, say) is refused, and so is a product without a sidecar (made
before records existed, by a TOML run of an earlier version, or interrupted before its sidecar).
`field.adopt(recipe, jobs=...)` checks such products against the recipe (their frames, sky terms
and offset maps, and that what each is made from is current or adopted too) and writes their
sidecars. What a product does not show, the fit's and the coadd's settings, is taken on trust:
adopt a product only with the recipe that made it, such as `NUMCOL3` (above) for the NumCol3 maps
its TOML production made:

```python
nep.adopt(NUMCOL3, jobs=spherex.channels(1, 34))   # the TOML production's maps: NUMCOL3 made them
```

`print(field.plan(...))` lists every product and whether it would be made or reused.

### Records

Each action writes `<field>/records/<action>_<time>_<pid>.json`: the settings with every default
resolved, the run specification they lowered to (`lowered`), the code version (commit, branch,
modified files), package versions, the environment knobs in effect, the products, the wall time and
the outcome.
A script's console output (its own and its workers') goes to one log per process,
`<field>/logs/<script>_<time>_<pid>.log`, which starts with the script's text; each action adds a
line naming its record.

A calibration's record also lists its `solves`, one entry per solve (a job, or a tile of a tiled
run): the cal it made (`cal`, `job`, `tile`) and how the solve ran, the same values as the cal
file's `solve` group (`selfcal.io.calfile.CalFile(cal).solve`; `result.cal()` opens a plain run's cal):

| key | content |
| --- | --- |
| `method`, `iterations`, `iterations_total`, `iteration_limit` | the solver; the iterations it ran, the cumulative count of the solution (the same until a solve continues another's), `Fit(iterations=)` |
| `istop`, `stop` | the solver's stop code and its meaning: 1-2 the tolerance tests, 3 the condition estimate, 4-6 machine precision, 7 the iteration limit |
| `r1norm`, `r2norm`, `arnorm`, `anorm`, `acond`, `xnorm` | the solver's final estimates of ‖b − A x‖ (without and with the damping term), ‖Aᵀ r‖, ‖A‖, cond(A) and ‖x‖ |
| `true_residual`, `bnorm` | ‖b − A x‖ and ‖b‖, computed once at the end with one product, in float64: an estimate drifting from the true residual shows here |
| `atol`, `btol`, `conlim`, `damp` | what the solver ran with (`conlim = 0`: `Fit(tolerance=0)`) |
| `rows`, `columns` | the system's shape (the active columns) |
| `wall_s` | the solver's wall time (in the record only: the cal file stays byte-identical from run to run) |
| `history_file` | `<field>/records/<cal stem>_history.npz`: the solver's state at every iteration |

The history file holds one array per column, row 0 the starting vector: `itn`, `r1norm`, `r2norm`,
`arnorm`, `anorm`, `acond`, `xnorm`, `test1` (‖r‖ / ‖b‖), `test2` (‖Aᵀ r‖ / (‖A‖ ‖r‖)) and
`elapsed_s`. The solver computes them anyway, so recording them costs nothing:

```python
import numpy as np, matplotlib.pyplot as plt
with result.cal() as cal:
    h = np.load(cal.solve["history_file"])
plt.semilogy(h["itn"], h["r1norm"]); plt.semilogy(h["itn"], h["arnorm"])
```

`sc.rerun("records/calibrate_....json")` (or `selfcal rerun RECORD`) runs the action again with the
recorded settings, warning when the code differs or when the frames it finds are not the ones the
record ran on; `overwrite=True` makes its products again (a reprojection: its frames). Its own
record names the original script.
`sc.compare(a, b)` (or `selfcal compare A B`) says whether two products are byte-identical, hold
equal values, or differ, by how much, and, from their sidecars, why.

### Running detached

`field.submit(recipe, jobs=...)` plans the run (all checks, seconds), then starts it in its own
session, so it survives the terminal; it returns the request file and the console log's path.
The run is the request's rerun: every function it uses must be importable.

### Functions from a notebook

A function defined in a notebook cell cannot be imported by the worker processes. Write it to a
module (`%%writefile myfuncs.py`, then `from myfuncs import ratio`), or wrap it:
`sc.Sky("line", times=sc.by_value(ratio))` sends the function by value (needs `cloudpickle`) and
keeps its source in the record; such a run cannot be submitted or rerun.

### Expert knobs

`sc.Compute(tuning=sc.Tuning(...))` sets the library's byte-neutral `SELFCAL_*` knobs for an
action (the worker start method, the parallel scatter, matrix block sizes, the vector threads, the
coadd's flush stripes, spill thresholds); each left at None defers to its environment variable.
The transpose product's thread count changes the last bits, so it lives in the recipe:
`sc.Numerics(rmatvec_threads=...)` (None: the solver's threads, capped by a 16 GB buffer).

### The command line

```bash
selfcal run my_run.py            # run a script with the BLAS/OpenMP threads pinned before numpy loads
selfcal plan my_run.py           # print its plan: each product made, reused or refused (nothing is run)
selfcal adopt my_run.py          # record products made without a sidecar as the script's (each is checked)
selfcal convert run.toml         # write run.py from an old TOML run config (unchecked: read its notes)
selfcal rerun RECORD.json        # run an action again from its record
selfcal compare A.h5 B.h5        # byte-identical, equal values, or different and why
```

`plan` and `adopt` read a run script's top level: `FIELD` (or `FIELDS`, a list, for a campaign),
`RECIPE`, and `RUN`, the keyword arguments of `calibrate` (`jobs=`, `tiles=`, `passes=`, ...);
the script runs `FIELD.calibrate(RECIPE, **RUN)` under its `__main__` guard. A reprojection script
defines `REPROJECT` (the arguments of `field.reproject`) instead of `RECIPE` and `RUN`.
`selfcal_scripts/runs/` holds the production run scripts, built from the shared recipes and fields
of `selfcal_scripts/recipes/`, and `selfcal_scripts/run.sh` runs one (`--dry-run`: the plan).

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
`job_geometry()` and `frame_variables()`, and define the optional hooks (`offset_renderer`,
`aux_coadds`, `finalize_mosaic`, `coefficient_catalog`). A chunk map other than rectangles is
`sc.ChunkMap(name, ids, ids, axes=sc.ChunkAxes.row_major(...), adjacency_axes=(...))` (the axes a
term's `smooth` follows unless it gives `smooth_along`). The engine receives the object itself and
calls it only through this contract, which the built-in instruments implement too
(`selfcal/instruments/camera.py`, `spherex/settings.py`, `euclid/settings.py`); there is no
registry. See [Bring your own telescope](../bring_your_own_telescope.md) for the instrument
contract in full.

## From a TOML config

TOML run configs are no longer run. `selfcal convert x.toml` writes `x.py`, a run script of the
same run, without checking it against the TOML; products a TOML run made are refused until they
are adopted (`selfcal adopt x.py`, or `field.adopt(recipe, ...)`).
[Migrating from TOML](migrating-from-toml.md) says what the converter writes, where each TOML key
and mode went, and which options were removed.
