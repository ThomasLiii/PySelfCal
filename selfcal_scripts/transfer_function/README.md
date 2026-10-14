# SelfCal transfer-function kit (SPHEREx D1–D6)

Measure the SelfCal pipeline's transfer function: run the **production**
calibration + mosaic (the 2026-10 recipe) on frames carrying a **known simulated
sky**, and compare the output mosaic to what went in. It runs the standard pipeline — no special
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
  exact geometry of the real-data run; too large for git).
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
(e.g. `exp_0000_det_00.h5`, as the pipeline names them). The simulated sky lives in `sub_data`; every
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

All commands below are run from the repo root, with the Python you `pip install`ed
into.

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
    <simsky_frames_dir>/exp_0000_det_00.h5 \
    --orig <real_reproj_dir>/exp_0000_det_00.h5    # --orig is optional
```

**c) Run the production calibration + mosaic** (a single channel).

```bash
python selfcal_scripts/transfer_function/transfer_function.py \
    --detector   3 \
    --channel    9 \
    --frames     <simsky_frames_dir> \
    --ref        <ref.fits> \
    --output-dir <output_dir> \
    --run-name   TF_D3
```

It links `ref.fits` into the run's folder, prints the plan (the frames it found,
the products it will make), then runs; `--dry-run` stops after the plan. `--help`
lists all flags (short forms `-d -c -f -r -o -n`). Each flag also reads an
environment variable (`DETECTOR`, `CHANNEL`, `REPROJ_FRAME_DIR`, `REF_FITS`,
`OUTPUT_DIR`, `RUN_NAME`; the flag wins). `--scratch` and `--workers` set the
machine (fast local disk, processes), which never changes the products.
Swap detector = change `--detector`, `--frames`, `--ref`, `--run-name`.

**Another recipe.** The recipe is Python: `RECIPE` in
[`transfer_function.py`](transfer_function.py) is the production recipe
(`SLOPE_GAUGE` in [`recipes/spherex.py`](../recipes/spherex.py)), solved on 17
channels and one column (`NUM_CH, NUM_COL = 17, 1` next to it). Copy the script
and change `RECIPE` under a name of its own (the products carry it, so they do
not collide with the production recipe's):

```python
RECIPE = SLOPE_GAUGE.replace(fit=SLOPE_GAUGE.fit.replace(gauge=None), name="nogauge")       # no gauge
RECIPE = SLOPE_GAUGE.replace(fit=SLOPE_GAUGE.fit.replace(gauge=sc.Gauge("xs_term", degree=2)),
                             name="quadgauge")                                              # the quadratic gauge
RECIPE, NUM_CH, NUM_COL = POLY_K1, 34, 10       # the NumCol 10 fiducial of 2026-05 (from recipes/spherex.py)
```

Settings are checked as they are built; the [Python API guide](../../docs/guide/python-api.md)
lists them. (The chunk axes belong to the instrument: `NUM_CH` and `NUM_COL`
build `sc.SPHEREx(detector, num_col=NUM_COL, num_ch=NUM_CH)` in `setup()`.)

The kit's TOML form (`run_transfer_function.sh` filling `transfer_function.toml`)
was removed with selfcal's TOML support; the script above makes the same products.

## 4. Output

Under `<output_dir>/<run-name>/`:

- **`mosaic/mosaic_*.fits`** — the recovered map, on the same WCS as `ref.fits`.
  Extension `MEAN_MAP` is what you compare against: the weighted mean, with the
  post-solve plane gauge added; `MEAN_MAP_WEIGHT` is its coverage. The mosaic
  also carries what the production mosaics do: the std map and the 2-sigma
  clipped mean (`SC_MEAN_MAP`, the gauge added too).
- `calibration/cal_*.h5` — the calibration solution. Its sky term carries the
  plane gauge, recorded in the group `gauge/`; the offsets stay as solved.
- `logs/transfer_function_<timestamp>_<pid>.log` — everything the run printed
  (including worker processes and any error traceback), headed by the command,
  the code version and the script used. One file per run, so reruns never
  overwrite an earlier log.
- `records/calibrate_<timestamp>_<pid>.json` (the settings with every default
  resolved, and what they ran as) and, next to each product, `<product>.json`:
  what made it.

Compare `mosaic_*.fits` to the simulated sky you put in (they share the
reference WCS) to read off the transfer function. Sweep different simulated
skies (a point source, a sinusoid per spatial frequency, …), giving each a
distinct `--run-name` so outputs don't collide: a run into a folder that already
holds its cal reuses that cal (when the same recipe made it), and frames count by
file name, not content.
