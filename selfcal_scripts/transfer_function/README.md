# SelfCal transfer-function kit (SPHEREx D1–D6)

Measure the SelfCal pipeline's transfer function: run the **fiducial**
calibration + mosaic on frames carrying a **known simulated sky**, and compare
the output mosaic to what went in. It runs the standard pipeline — no special
code path. For the how and why (geometry, schema details, options), see
[`DETAILS.md`](DETAILS.md).

## 1. Install

Requires Python ≥ 3.11.

```bash
git clone git@github.com:ThomasLiii/PySelfCal.git
cd PySelfCal
pip install -e .          # installs the `selfcal` package + dependencies
```

## 2. What you need (per detector)

- **Reprojected frames carrying your simulated sky**, and the detector's
  **`ref.fits`** — `ref.fits` is provided by the pipeline owner (it defines the
  exact fiducial geometry; too large for git).
  - If you **already have simulated-sky frames** in the format below, you are
    ready: go straight to [step 3](#3-run).
  - If you have a simulated sky as a 2-D array and real reprojected frames to
    put it into, `inject_simulated_sky.py` does that for you — see
    [step 3a](#3-run).
- **A simulated sky**, only if you are injecting one: a 2-D array on the
  **same grid as `ref.fits`** (`.npy` or `.fits`, shape == the `ref.fits`
  image shape).

### Reprojected-frame format

Each frame is one **Zstd-compressed HDF5** file named `exp_<e>_det_<d>.h5`
(e.g. `exp_000000_det_0.h5`). The simulated sky lives in `sub_data`; every
other key must be the real frame's, untouched. Contents:

| key | type / shape | meaning |
| --- | --- | --- |
| `sub_data` | float32, `H×W` | the exposure on the reference grid, cropped to the bbox (this is what carries the simulated sky) |
| `sub_mapping` | float32, `2×H×W` | detector↔reference coordinate map |
| `sub_bitmask` | int32, `H×W` | data-quality bits |
| `sub_foot` | float16, `H×W` | footprint / coverage |
| attr `ref_coords` | `[y0, y1, x0, x1]` | the frame's bbox in the reference grid, so `H = y1-y0`, `W = x1-x0` |
| attrs `sub_header`, `det_header` | str | WCS headers |

Because `sub_data` is already on the reference grid (cropped to `ref_coords`),
injecting a simulated sky `S` is a plain crop: `sub_data = S[y0:y1, x0:x1]`
(keeping NaN where the frame was unobserved). See `DETAILS.md`. Reading/writing
the frames needs `import hdf5plugin` (a dependency) — the kit scripts handle this.

## 3. Run

All commands below are run from the repo root. Point `SELFCAL_PY` at the env's
Python if your shell's `python` is not the one you `pip install`ed into.

**a) (optional) Inject a simulated sky into copies of the real frames.**
Skip this if you already have frames carrying your simulated sky — go to (c)
and point `--frames` at them.

```bash
python selfcal_scripts/transfer_function/inject_simulated_sky.py \
    --frames-in      <real_reproj_dir> \
    --frames-out     <simsky_frames_dir> \
    --simulated-sky  <simulated_sky.npy> \
    --ref-fits       <ref.fits> \
    --workers 16
```

**b) (optional) Sanity-check one frame before the full run.** Worth doing on
frames from any source — it catches a schema or grid mismatch in seconds
instead of after a full run.

```bash
python selfcal_scripts/transfer_function/verify_frame.py \
    <simsky_frames_dir>/exp_000000_det_0.h5 \
    --orig <real_reproj_dir>/exp_000000_det_0.h5    # --orig is optional
```

**c) Run the fiducial calibration + mosaic** (a single channel).

```bash
SELFCAL_PY=<env python>  selfcal_scripts/transfer_function/run_transfer_function.sh \
    --detector   3 \
    --channel    17 \
    --frames     <simsky_frames_dir> \
    --ref        <ref.fits> \
    --output-dir <output_dir> \
    --run-name   TF_D3
```

`run_transfer_function.sh --help` lists all flags (short forms `-d -c -f -r -o -n`).
Swap detector = change `--detector`, `--frames`, `--ref`, `--run-name`.

## 4. Output

Under `<output_dir>/<run-name>/`:

- **`mosaic/mosaic_*.fits`** — the recovered map, on the same WCS as `ref.fits`.
  Extension `MEAN_MAP` is what you compare against; `MEAN_MAP_WEIGHT` is its
  coverage. (The fiducial science mosaics also carry std, sigma-clipped-mean
  and wavelength maps; the kit skips those — the transfer function does not
  read them, and building them costs ~30 % more mosaic wall time. The
  calibration is unaffected, and `MEAN_MAP` is the same map either way: the
  coadd accumulates in a fixed order, so the maps depend only on the frames
  and the batch sizes. To get the other maps, set `make_std_map`,
  `apply_sigma_clipping`, `cache_intermediate` and `wavelength_coadd` back
  to `true` in `transfer_function.toml`.)
- `calibration/cal_*.h5` — the calibration solution.

Compare `mosaic_*.fits` to the simulated sky you put in (they share the
reference WCS) to read off the transfer function. Sweep different simulated
skies (a point source, a sinusoid per spatial frequency, …), giving each a
distinct `--run-name` so outputs don't collide.
