# SelfCal transfer-function kit (SPHEREx D1–D6)

Measure the SelfCal pipeline's transfer function: inject a **known fake sky** into
reprojected frames, run the **fiducial** calibration + mosaic, and compare the
output mosaic to what you put in. It runs the standard pipeline — no special code
path. For the how and why (geometry, schema details, options), see
[`DETAILS.md`](DETAILS.md).

## 1. Install

Requires Python ≥ 3.11.

```bash
git clone git@github.com:ThomasLiii/PySelfCal.git
cd PySelfCal
pip install -e .          # installs the `selfcal` package + dependencies
```

## 2. What you need (per detector)

- **Reprojected frames** and the detector's **`ref.fits`** — provided by the
  pipeline owner (they define the exact fiducial geometry; too large for git).
- **A fake sky** to inject — a 2-D array on the **same grid as `ref.fits`**
  (`.npy` or `.fits`, shape == the `ref.fits` image shape).

### Reprojected-frame format

Each frame is one **Zstd-compressed HDF5** file named `exp_<e>_det_<d>.h5`
(e.g. `exp_000000_det_0.h5`). You inject a fake sky by replacing **only**
`sub_data` — the script below does it. Contents:

| key | type / shape | meaning |
| --- | --- | --- |
| `sub_data` | float32, `H×W` | the exposure on the reference grid, cropped to the bbox (this is what you replace) |
| `sub_mapping` | float32, `2×H×W` | detector↔reference coordinate map |
| `sub_bitmask` | int32, `H×W` | data-quality bits |
| `sub_foot` | float16, `H×W` | footprint / coverage |
| attr `ref_coords` | `[y0, y1, x0, x1]` | the frame's bbox in the reference grid, so `H = y1-y0`, `W = x1-x0` |
| attrs `sub_header`, `det_header` | str | WCS headers |

Because `sub_data` is already on the reference grid (cropped to `ref_coords`),
injecting a fake sky `S` is a plain crop: `sub_data = S[y0:y1, x0:x1]` (keeping
NaN where the frame was unobserved). See `DETAILS.md`. Reading/writing the frames
needs `import hdf5plugin` (a dependency) — the kit scripts handle this.

## 3. Run

All commands below are run from the repo root. Point `SELFCAL_PY` at the env's
Python if your shell's `python` is not the one you `pip install`ed into.

```bash
# a) Inject your fake sky into copies of the real frames.
python selfcal_scripts/transfer_function/inject_fake_sky.py \
    --frames-in  <real_reproj_dir> \
    --frames-out <fakesky_frames_dir> \
    --fake-sky   <fake_sky.npy> \
    --ref-fits   <ref.fits> \
    --workers 16

# b) (optional) Sanity-check one frame before the full run.
python selfcal_scripts/transfer_function/verify_frame.py \
    <fakesky_frames_dir>/exp_000000_det_0.h5 \
    --orig <real_reproj_dir>/exp_000000_det_0.h5

# c) Run the fiducial calibration + mosaic (a single channel).
SELFCAL_PY=<env python>  selfcal_scripts/transfer_function/run_transfer_function.sh \
    --detector   3 \
    --channel    17 \
    --frames     <fakesky_frames_dir> \
    --ref        <ref.fits> \
    --output-dir <output_dir> \
    --run-name   TF_D3
```

`run_transfer_function.sh --help` lists all flags (short forms `-d -c -f -r -o -n`).
Swap detector = change `--detector`, `--frames`, `--ref`, `--run-name`.

## 4. Output

Under `<output_dir>/<run-name>/`:

- **`mosaic/mosaic_*.fits`** — the recovered map, on the same WCS as `ref.fits`.
- `calibration/cal_*.h5` — the calibration solution.

Compare `mosaic_*.fits` to your injected fake sky (they share the reference WCS)
to read off the transfer function. Sweep different injected skies (a point
source, a sinusoid per spatial frequency, …), giving each a distinct `--run-name`
so outputs don't collide.
