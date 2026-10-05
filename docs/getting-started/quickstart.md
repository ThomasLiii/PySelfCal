# Quickstart

This page takes simulated exposures from raw FITS files to a calibrated mosaic, in four steps that
together take well under a minute. The simulated camera is described by the built-in `grid`
instrument, so the run needs no code: two TOML configs and the runner. Because the data are
simulated, you can then compare what the solve recovered with what was injected.

Before you start, [install selfcal](installation.md). Run every command below from the root of the
repository checkout, in the environment where selfcal is installed: `run.sh` starts the `python` it
finds on your `PATH`. The outputs go to `quickstart_output/` in that directory; delete it when you
are done. The [Glossary](../guide/glossary.md) defines the terms used here.

The four files are in
[`examples/quickstart/`](https://github.com/ThomasLiii/PySelfCal/tree/main/examples/quickstart):

| File | What it does |
| --- | --- |
| `simulate.py` | writes the simulated exposures and the injected truth |
| `reproject.toml` | the config of step 2, the `reproject` task |
| `cal.toml` | the config of step 3, the `cal` task (calibration, then mosaic) |
| `inspect_results.py` | reads the products, compares them with the truth and saves a figure |

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
is the layout the `grid` instrument reads (`sci_ext = 1`, `dq_ext = 2`). `truth.npz` keeps the
injected offsets and scalars for step 4.

??? example "simulate.py"

    ```python
    --8<-- "examples/quickstart/simulate.py"
    ```

## 2. Reproject onto a common grid

```toml
--8<-- "examples/quickstart/reproject.toml"
```

```bash
./selfcal_scripts/run.sh examples/quickstart/reproject.toml
```

```text
[run] log: quickstart_output/quickstart/logs/reproject_20261001-135447_2096939.log
[run] task=reproject instrument=grid mode=None run_name=quickstart
Globbed 24 candidate exposures
Reference WCS not found at quickstart_output/quickstart/ref.fits. Creating a new reference frame.
...
Reference WCS saved to quickstart_output/quickstart/ref.fits
Mosaic shape: (125, 126)
...
Batch reprojection completed. 24 frames successfully processed out of 24 (0 failed).
...
[run] finished in 1.0 s
```

The `reproject` task finds the exposures (the files that match the glob `file_pattern` appended to
each entry of `input_dirs`), then defines the reference grid: the smallest north-up, east-left grid
with pixels of `resolution_arcsec` that contains every exposure, widened by `padding_pixels` on each
side. It saves the grid as `ref.fits` (125 x 126 pixels here) and resamples the image and the DQ
mask of every exposure onto it, writing one frame file per exposure and detector:
`reprojected/exp_0000_det_00.h5` to `exp_0023_det_00.h5`. A frame file holds the resampled data on a
square cut-out of the reference grid, the cut-out's position in the grid and the detector
coordinates of every pixel, from which the solve finds each pixel's chunk ([frame file
format](../guide/pipeline.md#reprojected-h5-schema)).

A run writes its products under `<output_dir>/<run_name>/`, here `quickstart_output/quickstart/`,
and a log of its console output to `logs/` there. The runner uses the paths in a config as written
and does not change directory, so relative paths such as `output_dir`, `cache_dir` and `input_dirs`
are relative to the directory you run from: from the repository root, everything lands in its
`quickstart_output/`.

## 3. Calibrate and mosaic

```toml
--8<-- "examples/quickstart/cal.toml"
```

```bash
./selfcal_scripts/run.sh examples/quickstart/cal.toml
```

An observation is the value of one frame on one pixel of the reference grid. The `continuum` mode
models it as the sky at that pixel, plus the offset of the chunk it fell on, plus the frame's
scalar; the `cal` task writes one equation per valid observation, adds the priors (smoothness
between neighbouring chunks, a zero mean for each frame's offsets, the damping of the sky) and
solves for all the unknowns at once with LSQR, an iterative sparse least-squares solver. It then
subtracts the solved offsets and scalars from every frame and coadds the frames into the mosaic.

```text
[run] log: quickstart_output/quickstart/logs/cal_20261001-135449_2096969.log
[run] task=cal instrument=grid mode=continuum run_name=quickstart
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
Finished All (Sim_Chunks4x4) in 3.11 seconds.
...
NVMe reproj cache cleaned up.
[run] finished in 3.1 s
```

- The task first copies the frames into `cache_dir`, the staging area (on a large run, a fast local
  disk: hence "NVMe"), and deletes the copy at the end.
- The 11,769 unknowns are the 11,361 pixels of the sky map that at least one frame observes, 384
  offsets (24 frames x 16 chunks) and 24 scalars. The other 4,389 pixels of the 125 x 126 grid have
  no data and are left out of the solve.
- `istop = 2` means that LSQR stopped because the solution met the `atol` tolerance, here after 137
  iterations; `istop = 7` would mean it ran out of iterations (`iter_lim`).

Both products are named after the stem `<tag>_Chunks<rows>x<cols>_<job><suffix>`. A job is one solve
of the instrument's loop (SPHEREx runs one per channel or subchannel window); the `grid` instrument
has a single job, `All`:

- `calibration/cal_Sim_Chunks4x4_All_quickstart.h5`, the solution: the sky map on the reference grid
  with its coverage, the offset of every frame and chunk, the scalar of every frame and the list of
  frames. Read it with [`CalFile`][selfcal.io.calfile.CalFile]
  ([format](../guide/pipeline.md#cal_h5-schema-multi-chunk-map)).
- `mosaic/mosaic_Sim_Chunks4x4_All_quickstart.fits`, the coadd of the calibrated frames: the mean
  (`MEAN_MAP`), the standard deviation (`STD_MAP`) and the sigma-clipped mean (`SC_MEAN_MAP`), each
  followed by its summed weight (with unit pixel weights, the number of frames that contribute to
  the pixel)
  ([format](../guide/pipeline.md#mosaic-fits-schema)).

If you run `cal.toml` again, the task finds the calibration file, prints `Calibration file ...
already exists. Skipping calibration.` and only remakes the mosaic. To solve again with other
settings, change `suffix` or delete the file.

!!! tip "Check a config without running it"

    `./selfcal_scripts/run.sh examples/quickstart/cal.toml --dry-run` reads the config, resolves the
    instrument's jobs and the mode, and stops:

    ```text
    [run] task=cal instrument=grid mode=continuum run_name=quickstart
    [dry-run] 1 job(s): ['All']
    [dry-run] mode=continuum mosaic_mode=full requires=() tiling=none
    [dry-run] config OK
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

The damping weight in `cal.toml` is small, 0.001. To see what it does, copy the config into the
output directory and give the copy the value of the production SPHEREx configs, 0.1, and a new
suffix, so that the solve runs again into new files:

```bash
cp examples/quickstart/cal.toml quickstart_output/cal_damp0p1.toml
```

```toml
suffix = "_damp0p1"        # in quickstart_output/cal_damp0p1.toml
damp_weight = 0.1
```

```bash
./selfcal_scripts/run.sh quickstart_output/cal_damp0p1.toml
python examples/quickstart/inspect_results.py --suffix _damp0p1
```

The paths in the copy still work: they are relative to the directory you run from, not to the
config file.

```text
Offsets, gauge removed: correlation 0.9794, rms difference 0.0606 (rms of the injected offsets 0.2954)
Scalars: recovered - injected = +5.0607 on average, rms about it 0.0979
Mosaic - injected sky: median -5.0382; rms about it 0.2195 on 11361 pixels
```

The damping adds, for every pixel of the sky map, `damp_weight` times its number of observations
times the square of its value to the quantity the solve minimises. At 0.1 the solve lowers that
penalty by moving part of the sky into the offsets: most of the sky's gradient becomes a pattern
fixed on the detector, and the residual panel of `results_damp0p1.png` shows it as a large-scale
slope across the field.

## Next steps

- [Run configuration](../guide/configuration.md): the config schema, the other tasks and modes,
  and the `[model]` table that spells out a model term by term.
- [How selfcal works](../guide/concepts.md): the model, the priors and the mosaic.
- [Bring your own telescope](../bring_your_own_telescope.md): your own exposures with the `grid`
  instrument, a model of your own, or an instrument class.
- [Tutorials and examples](tutorials.md): notebooks on SPHEREx and Euclid data, and end-to-end
  examples for other kinds of instrument.
- `tests/test_quickstart_example.py` runs these four steps in a temporary directory;
  `pytest -q tests/test_quickstart_example.py` takes 10 to 15 seconds.
