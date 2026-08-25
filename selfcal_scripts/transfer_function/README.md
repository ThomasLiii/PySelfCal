# SelfCal transfer-function kit (SPHEREx D1–D6)

Measure the selfcal pipeline's transfer function by pushing a **known fake sky**
through the **fiducial** calibration + mosaic recipe and comparing the output
mosaic to what you put in. This kit makes that a config-based, few-lines-to-change
operation, and easy to swap between detectors.

Everything runs off the existing pipeline (`selfcal_scripts.run`) — no forked
code path. You reuse the real reprojected frames for each detector and only swap
their pixel values, so the geometry, WCS, footprints, and per-detector reference
grid stay exactly as in the fiducial run.

---

## The idea (why it's just a crop)

The pipeline stores each reprojected frame's `sub_data` as the exposure **already
reprojected onto the reference grid**, bbox-cropped to `ref_coords = [y0,y1,x0,x1]`.
So `sub_data` has shape `(y1-y0, x1-x0)` and pixel `(i,j)` is reference-grid pixel
`(y0+i, x0+j)`. (Verified on a real D3 frame: `ref_coords=[2422,5574,6421,9573]`,
`sub_data` shape `(3152,3152)` — an exact bbox crop, ~58% NaN = the detector
footprint.)

That means injecting a fake sky `S` (defined on the detector's ref grid) is a
plain crop, done per frame:

```
sub_data_new = S[y0:y1, x0:x1]      # keep NaN wherever the real frame was NaN
```

Preserving the original NaN pattern keeps each frame's **observed footprint**
identical to the fiducial run — so only the sky values change, not the coverage.
Everything else in the frame (`sub_mapping`, `sub_bitmask`, `sub_foot`,
`ref_coords`, WCS headers, and the **filename**) is copied through unchanged, so
the `exp_<n>_det_<n>.h5` indices still parse and the SPHEREx LVF geometry (chunk
maps, wavelengths) the runner rebuilds from `detector=N` still lines up.

> Note: the frames are **Zstd-compressed HDF5** — any script that reads/writes
> them must `import hdf5plugin` (a selfcal dependency). The kit scripts do.

---

## What you provide, per detector

1. **Real reprojected frames** — the detector's fiducial `reprojected/` dir.
2. **The detector's `ref.fits`** — the same reference grid used for its fiducial
   mosaic (defines the grid your fake sky lives on).
3. **A fake sky** on that ref grid (`.npy` or `.fits`, shape == ref.fits shape).

You do **not** provide chunk maps, valid masks, or wavelengths — the runner
rebuilds those from `detector=N` + the LVF params shipped in the package.

---

## Steps

### 1. Inject the fake sky into the frames
```
python inject_fake_sky.py \
    --frames-in  /mnt/md124/.../SPHEREx_..._D3_.../reprojected \
    --frames-out /scratch/tf/D3_fakesky_frames \
    --fake-sky   /scratch/tf/fake_sky_D3.npy \
    --ref-fits   /mnt/md124/.../SPHEREx_..._D3_.../ref.fits \
    --workers 16
```
Writes a full copy of every frame with `sub_data` replaced by the crop of your
fake sky (footprint preserved). Filenames are kept.

### 2. Sanity-check one frame (recommended before a 900-frame run)
```
python verify_frame.py /scratch/tf/D3_fakesky_frames/exp_000000_det_0.h5 \
    --orig /mnt/md124/.../reprojected/exp_000000_det_0.h5
```
Confirms the schema is intact and that only `sub_data` changed (footprint,
`sub_mapping`, `ref_coords`, etc. unchanged). Expect `RESULT: PASS`.

### 3. Run the fiducial cal + mosaic

`run_transfer_function.sh` is the whole interface — exactly **6 inputs**,
everything else (the frozen fiducial recipe, pointing the runner at your
`ref.fits`) is handled for you. Give the inputs as **command-line flags**:

```
SELFCAL_PY=~/anaconda3/envs/selfcal/bin/python \
./run_transfer_function.sh \
    --detector   3 \
    --channel    17 \
    --frames     /scratch/tf/D3_fakesky_frames \
    --ref        /mnt/md124/.../SPHEREx_..._D3_.../ref.fits \
    --output-dir /mnt/md124/thomasli/selfcal/outputs \
    --run-name   TF_D3
```
(`SELFCAL_PY` is only needed if your shell's `python` is not the selfcal env.
`--help` lists the flags; short forms `-d -c -f -r -o -n` also work; any flag
you omit falls back to the default at the top of the script.)

Two other ways to pass the same 6 inputs, if you prefer:
- **Env vars** (e.g. for a detector loop):
  ```
  for d in 1 2 3 4 5 6; do
    DETECTOR=$d REPROJ_FRAME_DIR=/scratch/tf/D${d}_fakesky_frames \
    REF_FITS=/mnt/.../D${d}/ref.fits RUN_NAME=TF_D${d} ./run_transfer_function.sh
  done
  ```
- **Edit the 6 defaults** at the top of the script, then just `./run_transfer_function.sh`.

Precedence: flag > env var > default.

Outputs land in `{OUTPUT_DIR}/{RUN_NAME}/{calibration,mosaic}/`:
`cal_*.h5` and `mosaic_*.fits`.

> Advanced: everything except the 6 inputs is the frozen fiducial recipe, held
> in `transfer_function.toml`. Only touch that file to deliberately deviate from
> the fiducial settings.

Outputs land in `{output_dir}/TF_D{detector}/{calibration,mosaic}/`:
`cal_*.h5` and `mosaic_*.fits`.

### 4. Compare → transfer function
Compare the output `mosaic_*.fits` to your injected fake sky on the ref grid
(they share the same WCS). Sweep different injected skies (a delta, a sinusoid
per spatial frequency, a flat, …) and repeat steps 1–3 per input to trace the
transfer function. Give each sweep a distinct `run_name`/`reproj_override` so
outputs don't collide (shared `/mnt` outputs).

---

A "channel" is one of the detector's 34 LVF wavelength slices; each is an
independent cal+mosaic. The kit runs a **single** channel (default the mid-band
`Ch17`) — enough to characterize the transfer function without 34× the compute.
Change it with the config's `channels = [[N]]` line, or the launcher's last arg.

## Swapping detectors

Per detector, three of the six inputs change — `DETECTOR`, `REPROJ_FRAME_DIR`,
`REF_FITS` (and usually `RUN_NAME`). Set them as env vars and the same launcher
sweeps all of D1–D6 in the loop shown in step 3.

---

## ⚠️ Confirm before use (fiducial recipe)

The config bakes in the recipe from `selfcal_scripts/configs/d5.toml`
(`continuum` mode; `damp_weight=0.1`, `reg_weight=0.1`, `outlier_thresh=5`,
`sigma=2`, `poly_degree=1`, `poly_weight=0.5`, `num_col=10`, `oversample=2`).
**Thomas: confirm** this is the exact recipe behind your fiducial D1–D6 mosaics.
(Channel is now just one representative slice — no per-detector range to pin down.)

---

## Optional cleanup for when this is promoted into the repo

The one awkward step is making `ref.fits` appear at `{output_dir}/{run_name}/`.
Today the kit handles it with a symlink (launcher does it automatically). A
cleaner fix is a **3-line `ref_override` config field**, mirroring the existing
`reproj_override`, so the collaborator points straight at the ref.fits with no
symlink:

- `selfcal_scripts/runner/config.py`: add `ref_override: str = None` to
  `RunConfig` (and to the `_SCALAR_KEYS`/parse list alongside `reproj_override`).
- `selfcal_scripts/runner/pipelines.py` `_make_config(cfg)`: pass
  `ref_path=cfg.ref_override or None` to `PipelineConfig(...)` (which already
  accepts a `ref_path` argument).

Then `ref_override = "/path/to/ref.fits"` replaces the symlink. Left out of the
kit for now since you wanted no tracked-code changes yet.

---

## Files
- `inject_fake_sky.py` — replace `sub_data` with the fake-sky crop (footprint preserved).
- `verify_frame.py` — schema/injection sanity check on one frame.
- `transfer_function.toml` — the fiducial config template (3 lines to edit).
- `run_transfer_function.sh` — one-command per-detector launcher (symlink + fill + run).
