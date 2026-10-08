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
identical to the fiducial run — only the sky values change, not the coverage.
`inject_simulated_sky.py` copies every frame whole (so `sub_mapping`, `sub_bitmask`,
`sub_foot`, `ref_coords`, WCS headers, and the **filename** are unchanged) and
overwrites only `sub_data`. Because the filenames are kept, the `exp_<n>_det_<n>.h5`
indices still parse, and the SPHEREx LVF geometry (chunk maps, per-pixel
wavelengths) that `sc.SPHEREx(N)` rebuilds still lines up.

The frames are **Zstd-compressed HDF5**, so any code that reads/writes them must
`import hdf5plugin` (a `selfcal` dependency). The kit scripts do.

## What the pipeline supplies automatically

You provide only the frames, the `ref.fits`, and (if you are injecting one) the
simulated sky. You do **not**
provide chunk maps, valid masks, or wavelengths — the instrument, `sc.SPHEREx(N)`,
rebuilds the chunk maps and valid masks from the detector number plus the LVF
parameters shipped inside the package (`selfcal/instruments/spherex/data/lvf_params/`),
and reads the per-pixel wavelength maps (the `*BC_Band<N>.fits` / `*BW_Band<N>.fits`
band-centre / band-width files) from `sc.SPHEREx(calib_dir=)`, by default
`$SELFCAL_SPHEREX_CALIB_DIR`, else the SPHEREx calibration directory of the
processing host (`/data3/SPHEREx/SpecCal_202509/ParameterFiles`).

## The fiducial recipe

`FIDUCIAL` in `transfer_function.py` is the frozen fiducial continuum recipe — the
same one behind the fiducial D1–D6 mosaics: the shared `POLY_K1` of
`selfcal_scripts/recipes/spherex.py` (the recipe of `selfcal_scripts/runs/d5.py`)
on `sc.SPHEREx(detector, num_col=10)`:

- the model `sc.continuum(smooth=0.1, poly_prior=sc.Poly(1, weight=0.5))`: adjacency
  smoothness 0.1, a linear column polynomial of weight 0.5, the sky damped at 0.1
- the fit `sc.Fit(50, clip=5.0)`: 50 LSQR iterations, preconditioned, tolerance
  1e-6, a 5-sigma clip
- the coadd: the mean map only (`clip=None, std=False, instrument_maps=False`,
  `oversample=2`), without a frame cache (`cache_frames=False` in `setup()`): the
  transfer function is read off `MEAN_MAP`, and the other maps cost roughly half
  the mosaic's wall time without being read. This is a mosaic-side choice only —
  the calibration is the fiducial one, and `MEAN_MAP` is the same map whether or
  not the others are built: the coadd accumulates in a fixed order (batch order),
  so the maps depend only on the frames and the batch sizes.

You do not edit the script for normal use — the six run-specific inputs are its
flags. Copy it and change `RECIPE` only to deliberately deviate from the fiducial.
The machine settings (`--scratch`, `--workers`) never change a product's bytes.

Until October 2026 the kit also had a TOML form (`transfer_function.toml`, filled
by `run_transfer_function.sh`); the run engine was proven to be handed the same
work by both forms before the TOML form was removed with selfcal's TOML support.

## Channels

Each SPHEREx detector is a linear variable filter split into 34 spectral channels
(`num_ch = 34`); each channel is an independent calibration + mosaic at its
wavelength. The kit runs a **single** channel (default the mid-band `Ch17`, the job
`spherex.channel(N)`) — enough to characterize the transfer function without 34× the
compute. Valid channel indices are 1..34; pick with `--channel N`.

## The six inputs and three ways to set them

`transfer_function.py` exposes exactly six inputs and hides everything else
(the recipe, and pointing the run at your `ref.fits`), plus `--scratch`,
`--workers` and `--dry-run`:

| input | flag | env var |
| --- | --- | --- |
| detector (1..6) | `-d`, `--detector` | `DETECTOR` |
| channel (1..34) | `-c`, `--channel` | `CHANNEL` |
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
  frozen recipe `FIDUCIAL`, then the plan and the run.
