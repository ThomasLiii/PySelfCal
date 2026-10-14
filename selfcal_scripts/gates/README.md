# Byte-equality gates

Every structural change to the pipeline is gated on reproducing fixed products byte for byte.
The gates are Python: each is a function of `python_gates.py`, a calibration written with the
[Python API](../../docs/guide/python-api.md) on a fixed set of frames
(`python -m selfcal_scripts.gates.python_gates continuum | spectral | e2e | npass3 | euclid | m13`).
Their products carry a `py` suffix. `run_gates.sh <tag> [gate ...]` runs, on this box (the paths
to `/mnt/md124` and the staged fixtures are in `python_gates.py`), the steps named, compares each
product with its golden and writes `workspace/unify/logs/gates_<tag>.log`; `SELFCAL_REPO` picks
the tree. The steps are `pytest` (the test suite, first), then the gates below; the default is
every step but `m13`, which `run_m13_gate.sh <tag>` runs (it calls `run_gates.sh <tag> m13` and
logs to `m13_gate_<tag>.log`).

| gate | what | golden |
| --- | --- | --- |
| continuum | D3 Ch17 NumCol3, 300 frames, iter 50 (51 MB cal) | `cal_..._Ch17_gate_golden_f64.h5` |
| spectral | D4 AromaticPAHfit NumCol5, 150 frames, iter 20 (73 MB cal) | `cal_..._AromaticPAHfit_gate_golden_f64.h5` |
| e2e | D3 Ch17 cal + FULL mosaic (std, sigma-clip, wavelength maps; a 4.8 GB mosaic) | `*_unify_e2e_golden_f64.{h5,fits}` |
| npass3 | multiline J=4, n=3: INIT + closed-form SKY + per-frame OFFSET refit | none yet (see below) |
| euclid | the EDFN recipe as an `sc.Model` on `sc.Euclid`, 3 exposures x 16 detectors | `cal_EDFN_Y_golden_f64.h5`, `mosaic_EDFN_Y_golden_f64.fits` |
| m13 | the npass n=1 gate on the NEP M13 tile (1,101 frames, iter 300, ~50 min on a quiet box, ~90 GB of memory and ~35 GB of scratch) | `*_UNIFYNPASS1GOLDEN_F64_*.h5` |
| rerun | the continuum gate's record run again from the record alone (`selfcal rerun --overwrite RECORD`) | the continuum golden |
| transfer | D3 Ch9 (NumCh17, NumCol1) on the 300-frame qr2 set: the 2026-10 production recipe -- the weighted solve (`sc.shot_noise()`), the post-solve plane gauge on the map (`sc.Gauge('xs_term', degree=1)`), the weighted coadd (mean, std, 2-sigma clip) -- against what the archived workspace implementations made | `*_transfer_golden_f64.h5` (the cal's offsets), `*_transfer_plane_golden_f64.h5` (its sky), `*_transfer_planemap_golden_f64.fits` (the mosaic) |
| transfer_quad | the quadratic gauge (`sc.Gauge(degree=2)`) of the library on the golden ungauged cal and map of `transfer`, against the workspace's (`transfer/quad_check.py`; no solve) | `*_transfer_gq2_golden_f64.h5` (the sky), `*_transfer_gq2map_golden_f64.fits` (the maps) |

`python_gates.py` imports numpy before selfcal on purpose: the actions pin the threads
themselves. `h5_diff.py` / `fits_diff.py` compare every dataset / extension exactly.

Goldens are regenerated only when a numerical change is intended, from the committed tree, and
the commit says so: `make_goldens.sh <tag> [gate ...]` runs the Python gates on a clean tree
(`SELFCAL_REPO`) and keeps each product as its golden (an existing golden is replaced only with
`FORCE=1`; the log records the commit). The goldens are float64-norm (`*golden_f64*`): since
2026-10-05 (e8781bb) LSQR accumulates the norms of long float32 vectors in float64, which changed
the bytes of production-size solves, and every run uses those norms
(`SELFCAL_LSQR_FLOAT32_NORMS=1`, the old norms, is refused by `make_goldens.sh`). The npass3 gate
has no float64 golden yet: its only goldens (`*_unify_npass3_golden*.h5`, 2026-09-28) were made
with the float32 norms, so the gate fails with a missing file until `make_goldens.sh <tag> npass3`
makes one. The earlier goldens (`*_golden_stat*`, `*_golden.*`, `*_UNIFYNPASS1GOLDEN_iter300*`)
are kept as made, by the float32 norms. The assembly folds its per-pixel moments
deterministically (batch-id order), so two runs of the same tree are byte-identical on any box
load.

## The transfer gate

`transfer/` holds the workspace implementations of the pixel weights and the post-solve gauge as they ran in
production before they entered the library -- `before_solve.py` (the weighted solve:
the assembly's subframe preparation wrapped to weight every row by `1/sqrt(max(|value|, the frame's window
median))`), `before_mosaic.py` (the weighted coadd: `1/max(|raw value|, that median)`) and the gauge scripts
-- and `make_transfer_goldens.sh <tag>`, which runs them on the 300-frame qr2 D3 fixture (Ch9) with the
library of a committed tree (`SELFCAL_REPO`, a tree whose library predates the features) and keeps every
product as a golden. The `transfer` gate runs the same recipe on the library's own features
(`sc.Model(weight=sc.shot_noise())`, `sc.Fit(gauge=sc.Gauge('xs_term'))`; `python_gates.py transfer`) and must
reproduce them byte for byte: `transfer_check.py` compares the gate cal's offsets, coverage and frame scalars
with the ungauged golden and its sky term with the plane-gauged golden's (the library gauges the sky term and
the maps, the archived script also moved the offsets), `fits_diff.py` the mosaic with the plane-gauged map.
The production gauge is the plane (`apply_gauge.py --no-quadratic`, since 2026-10-14); the quadratic gauge
(`apply_gauge_q.py`, `sc.Gauge(degree=2)`) stays in the library as an option, and `transfer_quad` checks it on
the golden ungauged cal and map without a solve. `GAUGE_ONLY=1 make_transfer_goldens.sh <tag>` makes gauged
goldens that are missing from the existing ungauged ones (the plane's were added that way, 2026-10-14).
`before_solve.py` evaluates its weights in float64, as the library does; the production patch computed
them in float32 (`TRANSFER_FLOAT32_PATCH=1` reproduces it), which changes every weight by up to 9e-8
relative. On the fixture that moves the chi-squared stop from 520 to 540 iterations and the two cals
differ by the solve's own convergence tolerance: rms 6.2e-4 MJy/sr over the sky (2.1 % of the sky's
rms), 3-5 % of the offsets' rms (measured 2026-10-14; the products made in production with the float32
patch differ from the library's by that much, not by the weights' rounding).

## The engine views

`config_equivalence.py` checks what the run engine does with each run without a solve, to show
that a change to the engine leaves it unchanged:

```bash
python selfcal_scripts/gates/config_equivalence.py views <dir> [name ...]   # on the tree before the change, then after
python selfcal_scripts/gates/config_equivalence.py compare-views <dir_before> <dir_after>
```

`views` writes the engine view (`selfcal.run.equivalence.engine_view`) of every engine run of
each producer, one JSON file per producer: the library calls' keywords with their defaults filled,
the per-term damping, the offset rows and the sky coefficients on the instrument's real geometry,
the jobs, the product paths, the frames, the staging, the tiles and the passes. The producers and
their files:

| file | producer |
| --- | --- |
| `script__<name>.json` | the run script `selfcal_scripts/runs/<name>.py` (a campaign's views by field) |
| `example__quickstart.json` | `examples/quickstart/quickstart.py` |
| `tool__transfer_function.json` | `selfcal_scripts/transfer_function/transfer_function.py` with its default inputs |
| `gate__<name>.json` | each gate of `python_gates.py` (`continuum`, `spectral`, `e2e`, `npass3`, `euclid`, `m13`), its `Field.calibrate` call captured |

The views read this machine's files (frame lists, reference grids, calibration data) and its CPU
count (the default number of workers): compare views made on one machine.

The gates were TOML configs (`configs/*.toml`) before October 2026; they and the configs that
recorded how each golden was made are in the git history
([Migrating from TOML](../../docs/guide/migrating-from-toml.md)).
