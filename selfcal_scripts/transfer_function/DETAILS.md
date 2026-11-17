# Transfer-function kit — details

Background and options behind [`README.md`](README.md).

## Why the injection is just a crop

The pipeline stores each reprojected frame's `sub_data` as the exposure **already
reprojected onto the reference grid**, bbox-cropped to `ref_coords = [y0,y1,x0,x1]`.
So `sub_data` has shape `(y1-y0, x1-x0)` and pixel `(i,j)` corresponds to
reference-grid pixel `(y0+i, x0+j)`. (Verified on a real D3 frame:
`ref_coords=[2422,5574,6421,9573]`, `sub_data` shape `(3152,3152)` — an exact bbox
crop, ~58% NaN = the detector footprint.)

Injecting a simulated sky `S` (defined on the detector's reference grid) is
therefore a plain per-frame crop:

```
sub_data_new = S[y0:y1, x0:x1]      # keep NaN wherever the real frame was NaN
```

Preserving the original NaN pattern keeps each frame's **observed footprint**
identical to the real-data run — only the sky values change, not the coverage.
`inject_simulated_sky.py` copies every frame whole (so `sub_mapping`, `sub_bitmask`,
`sub_foot`, `ref_coords`, WCS headers, and the **filename** are unchanged) and
overwrites only `sub_data`. Because the filenames are kept, the `exp_<n>_det_<n>.h5`
indices still parse, and the SPHEREx LVF geometry (the per-pixel wavelengths of
`sc.SPHEREx(N)`, the recipe's chunk maps on its arcs) still lines up.

The frames are **Zstd-compressed HDF5**, so any code that reads/writes them must
`import hdf5plugin` (a `selfcal` dependency). The kit scripts do.

## What the pipeline supplies automatically

You provide only the frames, the `ref.fits`, and (if you are injecting one) the
simulated sky. You do **not**
provide chunk maps, valid masks, or wavelengths — the recipe's chunk maps are
rebuilt from the detector number plus the LVF parameters shipped inside the package
(`selfcal/instruments/spherex/data/lvf_params/`), the job's valid pixels from its
channel, and the instrument, `sc.SPHEREx(N)`, reads the per-pixel wavelength maps (the `*BC_Band<N>.fits` / `*BW_Band<N>.fits`
band-centre / band-width files) from `sc.SPHEREx(calib_dir=)`, by default
`$SELFCAL_SPHEREX_CALIB_DIR`, else the SPHEREx calibration directory of the
processing host (`/data3/SPHEREx/SpecCal_202509/ParameterFiles`).

## The recipe

`RECIPE` in `transfer_function.py` is the production recipe of 2026-10, unchanged:
`SLOPE_GAUGE` of `selfcal_scripts/recipes/spherex.py`, on its chunk maps
`maps(num_ch=17, num_col=1)` (17 channels of 10 subchannels, one column) of `sc.SPHEREx(detector)`:

- the model: per frame, a constant per subchannel (mean zero over the frame) and a
  free gradient along detector x (`selfcal_scripts.spherex.terms.frame_slope()`: `(det_x - 1019.5) / 1020`
  times one number per frame and subchannel), the sky undamped; no adjacency,
  no polynomial, no prior
- the weights (`sc.shot_noise()`): every observation's variance is
  `max(|value|, the frame's median)`; the solve weights each row by its inverse
  square root, the coadd each pixel by its inverse
- the fit: deflated CGLS (`sc.Deflate(4)`) to the chi-squared stop
  (`sc.ChiSquaredRule(0.1, window=50, every=10)`: the chi-squared fell by less than
  0.1² over the last 50 iterations, checked every 10 — converged to a tenth of the
  error bars; at most 600 iterations), no clip, no flag ignored at the solve
- the post-solve gauge (`sc.Gauge("xs_term", degree=1)`): the per-frame gradients
  cannot tell a sky plane from its image in the offsets, so after the solve the
  survey-mean plane the offsets absorbed is fitted to them and moved into the sky:
  the cal's sky term and every mosaic of the cal gain it, the offsets stay as solved
- the coadd: the weighted mean, the std and the 2-sigma clipped mean, flag 21
  (the source mask) ignored, no instrument maps

You do not edit the script for normal use — the six run-specific inputs are its
flags. Copy it and change `RECIPE` only to deliberately deviate from the production
recipe. The machine settings (`--scratch`, `--workers`) never change a product's
bytes. The fiducial channel maps since 2026-10-14 (`selfcal_scripts/runs/channel_maps.py`)
are made with this recipe too, their mosaics also carrying the wavelength maps
(`CHANNEL_MAPS`: the same cal, the same mean, std and clipped maps). Until 2026-10 the kit
ran the NumCol 10 fiducial behind the D1–D6 channel mosaics of 2026-05 (`POLY_K1`, on
34 channels and 10 columns, with a mean-only coadd; removed on 2026-10-30 with the soft
polynomial prior it used); the README shows how to run another recipe.

Until October 2026 the kit also had a TOML form (`transfer_function.toml`, filled
by `run_transfer_function.sh`); the run engine was proven to be handed the same
work by both forms before the TOML form was removed with selfcal's TOML support.

## Channels

Each SPHEREx detector is a linear variable filter; the production recipe splits it
into 17 spectral channels (`num_ch=17` of its chunk maps, each the width of two of the 34 science
channels), and each channel is an independent calibration + mosaic at its
wavelength. The kit runs a **single** channel (default `Ch9`, the channel of the
transfer-function campaign of 2026-10; the job `jobs.channel(N)` of
`selfcal_scripts/spherex/jobs.py`) — enough to
characterize the transfer function without 17× the compute. Valid channel indices
are 1..17; pick with `--channel N`.

## The six inputs and three ways to set them

`transfer_function.py` exposes exactly six inputs and hides everything else
(the recipe, and pointing the run at your `ref.fits`), plus `--scratch`,
`--workers` and `--dry-run`:

| input | flag | env var |
| --- | --- | --- |
| detector (1..6) | `-d`, `--detector` | `DETECTOR` |
| channel (1..17) | `-c`, `--channel` | `CHANNEL` |
| simulated-sky frames dir | `-f`, `--frames` | `REPROJ_FRAME_DIR` |
| ref.fits | `-r`, `--ref` | `REF_FITS` |
| output dir | `-o`, `--output-dir` | `OUTPUT_DIR` |
| run name | `-n`, `--run-name` | `RUN_NAME` |

Three equivalent ways to pass them (**precedence: flag > env var > default**):

1. **Flags** (see README).
2. **Env vars**, e.g. a loop over detectors:
   ```bash
   for d in 1 2 3 4 5 6; do
     DETECTOR=$d REPROJ_FRAME_DIR=/scratch/tf/D${d}_simsky_frames \
     REF_FITS=/path/to/D${d}/ref.fits RUN_NAME=TF_D${d} \
     python selfcal_scripts/transfer_function/transfer_function.py
   done
   ```
3. **Edit the six defaults** (`INPUTS` in `transfer_function.py`), then run it
   with no arguments.

## How `ref.fits` is wired

The run reads the reference grid from `{output_dir}/{run_name}/ref.fits`.
`transfer_function.py` creates that path and symlinks your `--ref` file into it
(the file is not copied or modified), with an absolute target (a relative `--ref`
still resolves), and it never replaces a `ref.fits` that is a file of its own (a
real run's folder).

## Files

- `inject_simulated_sky.py` — **optional**: replace each frame's `sub_data` with
  the simulated-sky crop (footprint preserved); parallel over frames. Not needed
  if your frames already carry the simulated sky.
- `verify_frame.py` — schema + injection sanity check on one frame (with `--orig`,
  confirms only `sub_data` changed).
- `transfer_function.py` — the run: the six inputs, the `ref.fits` link, the
  production recipe `RECIPE` (and its chunk axes), then the plan and the run.
