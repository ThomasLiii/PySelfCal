# Quickstart

This page takes simulated exposures from raw FITS files to a calibrated mosaic in four steps that
together take under a minute. A run is a short Python script: it describes the camera, the field
its products belong to and the recipe to solve, then calls the field's actions. Because the data
are simulated, you can then compare what the solve recovered with what was injected.

Before you start, [install selfcal](installation.md). Run every command below from the root of the
repository checkout, in the environment where selfcal is installed. The outputs go to
`quickstart_output/` in that directory; delete it when you are done. The
[Glossary](../guide/glossary.md) defines the terms used here.

The files are in
[`examples/quickstart/`](https://github.com/ThomasLiii/PySelfCal/tree/main/examples/quickstart):

| File | What it does |
| --- | --- |
| `simulate.py` | writes the simulated exposures and the injected truth |
| `quickstart.py` | steps 2 and 3: the run (reprojection, then calibration and mosaic) |
| `inspect_results.py` | reads the products, compares them with the truth and saves a figure |
| `damping.py` | step 5: the same run with another damping |

## 1. Simulate the exposures

```bash
python examples/quickstart/simulate.py
```

```text
wrote 24 exposures of 64 x 64 pixels to quickstart_output/exposures/
wrote the injected offsets (24, 16) and scalars (24,) to quickstart_output/truth.npz
```

Each exposure is a 64 x 64-pixel image with 10 arcsec pixels:

```text
image = sky + offset[chunk] + scalar + noise
```

- `sky` is one fixed sky: a level of 5 with a gradient and a large-scale ripple, plus five compact
  Gaussian sources. Each exposure points at a random position up to 4 arcmin (24 pixels) east or
  west and north or south of the field centre, so the same patch of sky falls on different parts of
  the detector in different exposures. This dithering is what lets the solve tell the sky from the
  detector.
- `offset` is the detector's additive signature: one value for each chunk of a 4 x 4 grid of
  16 x 16-pixel chunks, drawn anew for every exposure (rms 0.3) and shifted to a zero mean in each
  exposure.
- `scalar` is one additive level per exposure (rms 1).
- `noise` is white, with an rms of 0.05 per pixel.

Extension 1 of each file holds the image and its celestial WCS (a tangent projection, north up, east
left), extension 2 an integer data-quality (DQ) mask with bit 3 set on about 1 % of the pixels. That
is the layout the camera of step 2 reads (`sc.Camera(..., dq_ext=2)`; the image in extension 1 is
the default). `truth.npz` keeps the
injected offsets and scalars for step 4.

??? example "simulate.py"

    ```python
    --8<-- "examples/quickstart/simulate.py"
    ```

## 2. Describe the run

```python
--8<-- "examples/quickstart/quickstart.py"
```

The script defines three objects, then acts on them:

- **`sc.Camera`** says how to read an exposure: the detector's shape, the grid of chunks that
  carry the offsets (4 x 4 chunks of 16 x 16 pixels), the FITS extensions of the image (1, the
  default) and of the DQ mask (2), and the tag that starts the product names. A camera whose
  exposures need more (a reader function, per-pixel detector maps, header values) takes them as
  further settings ([Bring your own telescope](../bring_your_own_telescope.md)); SPHEREx and Euclid
  have instruments of their own, `sc.SPHEREx(detector)` and `sc.Euclid(...)`.
- **`sc.Field`** is a directory of products for one instrument on one reference grid:
  `quickstart_output/quickstart/` here, holding the grid (`ref.fits`, at `pixel_scale` arcsec per
  pixel), the reprojected frames, the calibrations, the mosaics, and a log and a record of every
  action. `compute=` describes the machine: `sc.Compute` names the scratch space for staged frames
  and caches and the number of worker processes. Machine settings never change a product.
- **`sc.Recipe`** is what to solve and how. `sc.continuum(...)` is the model: a sky map, an offset
  for every frame and chunk with a smoothness prior between neighbouring chunks and a zero mean in
  each frame, a scalar for every frame, and a damping that pulls the sky toward zero. `sc.Fit` is
  the solve (at most 200 LSQR iterations, to a tolerance of 1e-8), `sc.Coadd` the mosaic (a
  sigma-clipped mean at 3 sigma; `coadd=None` makes none) and `sc.Numerics` the summation layout
  (threads and batch sizes, which change only the last bits of the products). The `name` ends the
  product names.

Every setting is checked when it is built, so a mistake stops the script at the line that made it:

```pycon
>>> sc.Fit(200, tolerence=1e-8)
TypeError: Fit() got an unexpected keyword argument 'tolerence'. Did you mean 'tolerance'?
>>> sc.Clip(2.0, per="chunks")
ConfigError: Clip(per=...): expected one of 'frame', 'chunk' or ChunkGroups, got 'chunks'
```

`print` shows any of these objects in the form that rebuilds it. The actions start worker
processes, which import the script; the `if __name__ == "__main__":` block keeps them from running
the actions again, and selfcal refuses to start an action from a script without one.

## 3. Reproject, calibrate and mosaic

```bash
python examples/quickstart/quickstart.py
```

```text
[selfcal] reproject quickstart: record quickstart_output/quickstart/records/reproject_20261002-152431_3042735.json
Globbed 24 candidate exposures
Reference WCS not found at quickstart_output/quickstart/ref.fits. Creating a new reference frame.
...
Reference WCS saved to quickstart_output/quickstart/ref.fits
Mosaic shape: (125, 126)
...
Batch reprojection completed. 24 frames successfully processed out of 24 (0 failed).
...
Plan: calibrate quickstart  (quickstart_output/quickstart)
  instrument  Camera((64, 64), dq_ext=2, tag='Sim')
  recipe      Recipe(Model(sky=(Sky(damping=0.001),), offsets=(Offsets(smooth=0.1, mean_zero=True),)), ...)
  sky term    continuum: damping 0.001
  offsets     Offsets(smooth=0.1, mean_zero=True)
  frames      24 in quickstart_output/quickstart/reprojected (staged (copy) to quickstart_output/cache/reproj_nvme_quickstart)
  cal All                cal_Sim_Chunks4x4_All_quickstart.h5  (made)
  mosaic All             mosaic_Sim_Chunks4x4_All_quickstart.fits  (made)
[selfcal] calibrate quickstart: record quickstart_output/quickstart/records/calibrate_20261002-152434_3042735.json
Copying 24 reproj files to NVMe (quickstart_output/cache/reproj_nvme_quickstart)...
Processing All (Sim_Chunks4x4)...
...
Applying target mean offset constraints for map 0 (24 frames)...
Applying Coverage-Weighted Damping (continuum, damp=0.001)...
Compacting zero-coverage columns inline (11769/16158 active).
...
Solving least squares for 11769 unknowns with 105803 equations (solver=lsqr).
...
istop =       2   r1norm = 1.1e+01   anorm = 1.8e+01   arnorm = 1.2e-06
itn   =     137   r2norm = 1.1e+01   acond = 9.3e+02   xnorm  = 1.6e+02
...
Calibration saved to quickstart_output/quickstart/calibration/cal_Sim_Chunks4x4_All_quickstart.h5
...
Mosaic saved to quickstart_output/quickstart/mosaic/mosaic_Sim_Chunks4x4_All_quickstart.fits
...
NVMe reproj cache cleaned up.
Result of quickstart: 1 job(s)
  All: cal quickstart_output/quickstart/calibration/cal_Sim_Chunks4x4_All_quickstart.h5
       mosaic quickstart_output/quickstart/mosaic/mosaic_Sim_Chunks4x4_All_quickstart.fits
  record: quickstart_output/quickstart/records/calibrate_20261002-152434_3042735.json
```

**The reprojection.** `FIELD.reproject` finds the exposures that match the glob, then defines the
reference grid: the smallest north-up, east-left grid with pixels of `pixel_scale` that contains
every exposure, widened by `padding` pixels on each side. It saves the grid as `ref.fits`
(125 x 126 pixels here) and resamples the image and the DQ mask of every exposure onto it
(`method="interp"` is bilinear; the default, `"exact"`, conserves flux), writing one frame file per
exposure and detector: `reprojected/exp_0000_det_00.h5` to `exp_0023_det_00.h5`. A frame file holds
the resampled data on a square cut-out of the reference grid, the cut-out's position in the grid and
the detector coordinates of every pixel, from which the solve finds each pixel's chunk ([frame file
format](../guide/pipeline.md#reprojected-h5-schema)). Frames that exist are kept, so a second run
reprojects nothing.

**The plan.** `FIELD.plan(RECIPE)` checks what `calibrate` would do without computing anything:
the frames, the model against the instrument's geometry, the functions the worker processes will
import, and each product, to be made, reused or refused. `calibrate` makes the same checks first,
so a run that cannot finish stops before it starts. From the shell,
`selfcal plan examples/quickstart/quickstart.py` prints the plan of a script's `FIELD` and `RECIPE`.

**The solve.** An observation is the value of one frame on one pixel of the reference grid. The
model of `sc.continuum` is the sky at that pixel, plus the offset of the chunk the pixel fell on,
plus the frame's scalar; `calibrate` writes one equation per valid observation, adds the priors
(smoothness between neighbouring chunks, a zero mean for each frame's offsets, the damping of the
sky) and solves for all the unknowns at once with LSQR, an iterative sparse least-squares solver.
It then subtracts the solved offsets and scalars from every frame and coadds the frames into the
mosaic.

- The action first copies the frames into the scratch space, the staging area (on a large run, a
  fast local disk: hence "NVMe"), and deletes the copy at the end.
- The 11,769 unknowns are the 11,361 pixels of the sky map that at least one frame observes, 384
  offsets (24 frames x 16 chunks) and 24 scalars. The other 4,389 pixels of the 125 x 126 grid have
  no data and are left out of the solve.
- `istop = 2` means that LSQR stopped because the solution met the tolerance, here after 137
  iterations; `istop = 7` would mean it ran out of iterations.

**The products** are named after the stem `<tag>_Chunks<rows>x<cols>_<job>_<recipe name>`. A job is
one solve of the instrument's loop (SPHEREx runs one per channel or subchannel window); a camera has
a single job, `All`:

- `calibration/cal_Sim_Chunks4x4_All_quickstart.h5`, the solution: the sky map on the reference grid
  with its coverage, the offset of every frame and chunk, the scalar of every frame and the list of
  frames. Read it with [`CalFile`][selfcal.io.calfile.CalFile], or with `result.cal()` from the
  result `calibrate` returns ([format](../guide/pipeline.md#cal_h5-schema-multi-chunk-map)).
- `mosaic/mosaic_Sim_Chunks4x4_All_quickstart.fits`, the coadd of the calibrated frames: the mean
  (`MEAN_MAP`), the standard deviation (`STD_MAP`) and the sigma-clipped mean (`SC_MEAN_MAP`), each
  followed by its summed weight (with unit pixel weights, the number of frames that contribute to
  the pixel) ([format](../guide/pipeline.md#mosaic-fits-schema)).

Next to each product, `<product>.json` records the inputs it was made from. `records/` holds a
record of every action (its settings with every default filled in, the code version, its products),
which `selfcal rerun RECORD` runs again, and `logs/` the console output of every run.

**Running again.** Run the script again and the plan shows both products as `(reused)`: they exist
and were made by the same inputs, so nothing is computed. Change the recipe but keep its name, and
the run stops before computing anything, naming the difference:

```text
ConfigError: mosaic_Sim_Chunks4x4_All_quickstart.fits exists but was made with different inputs
(coadd.clip: 3.0 -> 2.5). Give the recipe its own name (recipe.replace(name=...)), or pass
overwrite=True to make it again
```

## 4. Inspect the results

```bash
python examples/quickstart/inspect_results.py
```

```text
cal_Sim_Chunks4x4_All_quickstart.h5: schema v3
  sky blocks: ['continuum'] on (125, 126)
  offset maps: 1, frames: 24, frame scalar: yes
  attrs: num_maps=1, num_sky_blocks=1, schema_version=3
  offsets (24, 16), scalars (24,); the sky map is observed on 11361 pixels, up to 24 times
mosaic_Sim_Chunks4x4_All_quickstart.fits:
  MEAN_MAP            125 x 126
  MEAN_MAP_WEIGHT     125 x 126
  STD_MAP             125 x 126
  STD_MAP_WEIGHT      125 x 126
  SC_MEAN_MAP         125 x 126
  SC_MEAN_MAP_WEIGHT  125 x 126
Offsets, gauge removed: correlation 0.9999, rms difference 0.0036 (rms of the injected offsets 0.2954)
Scalars: recovered - injected = +5.0609 on average, rms about it 0.0045
Mosaic - injected sky: median -5.0609; rms about it 0.0206 on 11361 pixels
Saved quickstart_output/results_quickstart.png
```

**The offsets** are compared after removing their gauge, the part the data cannot determine: a
constant added to all of a frame's offsets is indistinguishable from a change of its scalar, and a
ramp fixed on the detector, with a change of every frame's scalar, can trade against a gradient of
the sky. So each frame's mean offset and each chunk's mean over all frames are removed from the
recovered and from the injected offsets before they are compared.
[How selfcal works](../guide/concepts.md) explains these degeneracies and the priors that settle
them.

**The zero point.** The scalars come out 5.06 above the injected ones and the mosaic 5.06 below the
injected sky. Adding a constant to the sky and subtracting it from every scalar changes no modelled
value, so the data leave the sky's zero point free. The damping fixes it by pulling the sky toward
zero, and the sky's mean level moves into the scalars. A survey sets its zero point from outside
information; SPHEREx fits it to a model of the zodiacal light
([zodiacal-light anchor](../tools/zodi-anchor.md)).

**The figure**, `quickstart_output/results_quickstart.png`, has four panels, each map with its
colour bar:

- the first exposure in detector pixels, with the chunk boundaries drawn: the offsets give each
  chunk its own level;
- the mosaic (`SC_MEAN_MAP`): the chunk pattern is gone, and the level sits near zero (the zero
  point above);
- the mosaic minus the injected sky minus the zero point, on a colour scale that is linear within
  ±0.05 (the noise of one pixel) and logarithmic beyond: noise, larger at the edges where fewer
  frames overlap, faint residuals at the sources, and a weak gradient: the small part of the sky's
  gradient that the solve moved into a ramp fixed on the detector (the gauge above);
- the recovered against the injected offsets, gauge removed, one point per frame and chunk.

??? example "inspect_results.py"

    ```python
    --8<-- "examples/quickstart/inspect_results.py"
    ```

## 5. Change a setting

The damping in `quickstart.py` is small, 0.001. `damping.py` runs the same recipe with the damping
of the production SPHEREx recipes, 0.1, under a name of its own, so that its products are new files
next to the first ones:

```python
--8<-- "examples/quickstart/damping.py"
```

`.replace(...)` returns a copy of the recipe with the given settings changed; every setting has
the same method. `damping.py` imports `FIELD` and the quickstart's `RECIPE` from `quickstart.py`: a
run script is a module, so a variant builds on it rather than copying it. Its own recipe is called
`RECIPE` too, so `selfcal plan examples/quickstart/damping.py` plans what it runs.

```bash
python examples/quickstart/damping.py
python examples/quickstart/inspect_results.py --suffix _damp0p1
```

```text
Offsets, gauge removed: correlation 0.9794, rms difference 0.0606 (rms of the injected offsets 0.2954)
Scalars: recovered - injected = +5.0607 on average, rms about it 0.0979
Mosaic - injected sky: median -5.0382; rms about it 0.2195 on 11361 pixels
```

The damping adds, for every pixel of the sky map, the damping times its number of observations
times the square of its value to the quantity the solve minimises. At 0.1 the solve lowers that
penalty by moving part of the sky into the offsets: most of the sky's gradient becomes a pattern
fixed on the detector, and the residual panel of `results_damp0p1.png` shows it as a large-scale
slope across the field.

## Next steps

- [The Python API](../guide/python-api.md): every setting, the instruments and their jobs, big
  fields (tiles and passes), products, records and reruns, and the command line.
- [How selfcal works](../guide/concepts.md): the model, the priors and the mosaic.
- [Bring your own telescope](../bring_your_own_telescope.md): your own exposures with `sc.Camera`,
  a model of your own, or an instrument class.
- [Tutorials and examples](tutorials.md): notebooks on SPHEREx and Euclid data, and end-to-end
  examples for other kinds of instrument.
- `tests/test_quickstart_example.py` runs these steps in temporary directories and checks that the
  solve recovers the injected offsets. <!-- check: the test after its TOML half is removed -->
