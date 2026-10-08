# Byte-equality gates

Every structural change to the pipeline is gated on reproducing fixed products byte for byte.
The gates are Python: each is a function of `python_gates.py`, a calibration written with the
[Python API](../../docs/guide/python-api.md) on a fixed set of frames
(`python -m selfcal_scripts.gates.python_gates continuum | spectral | e2e | npass3 | euclid | m13`).
Their products carry a `py` suffix. `run_gates.sh <tag> [gate ...]` runs, on this box (the paths
to `/mnt/md124` and the staged fixtures are in `python_gates.py`), the gates named (default: all),
compares each product with its golden and writes `workspace/unify/logs/gates_<tag>.log`;
`SELFCAL_REPO` picks the tree. <!-- check: run_gates.sh after S2 (gate list, pytest step, whether run_python_gates.sh remains) -->

| gate | what | golden |
| --- | --- | --- |
| continuum | D3 Ch17 NumCol3, 300 frames, iter 50 (51 MB cal) | `cal_..._Ch17_gate_golden_f64.h5` |
| spectral | D4 AromaticPAHfit NumCol5, 150 frames, iter 20 (73 MB cal) | `cal_..._AromaticPAHfit_gate_golden_f64.h5` |
| e2e | D3 Ch17 cal + FULL mosaic (std, sigma-clip, wavelength maps; a 4.8 GB mosaic) | `*_unify_e2e_golden_f64.{h5,fits}` |
| npass3 | multiline J=4, n=3: INIT + closed-form SKY + per-frame OFFSET refit | none yet (see below) |
| euclid | the EDFN recipe as an `sc.Model` on `sc.Euclid`, 3 exposures x 16 detectors | `cal_EDFN_Y_golden_f64.h5`, `mosaic_EDFN_Y_golden_f64.fits` |
| m13 | the npass n=1 gate on the NEP M13 tile (1,101 frames, iter 300, ~50 min on a quiet box, ~90 GB of memory and ~35 GB of scratch) | `*_UNIFYNPASS1GOLDEN_F64_*.h5` |
| rerun | the continuum gate's record run again from the record alone (`selfcal rerun --overwrite RECORD`) | the continuum golden |

`run_m13_gate.sh <tag>` runs the m13 gate on its own. The script imports numpy before selfcal on
purpose: the actions pin the threads themselves. `h5_diff.py` / `fits_diff.py` compare every
dataset / extension exactly.

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

## The engine views

`config_equivalence.py` checks what the run engine does with each run without a solve, to show
that a change to the engine leaves it unchanged:

```bash
python selfcal_scripts/gates/config_equivalence.py views <dir> [name ...]   # on the tree before the change, then after
python selfcal_scripts/gates/config_equivalence.py compare-views <dir_before> <dir_after>
```

`views` writes the engine view (`selfcal.run.equivalence.engine_view`) of every run, one JSON
file per producer: the library calls' keywords with their defaults filled, the per-term damping,
the offset rows and the sky coefficients on the instrument's real geometry, the jobs, the product
paths, the frames, the staging, the tiles and the passes. The producers are the run scripts of
`selfcal_scripts/runs/` (`script__<name>.json`), the quickstart, the transfer-function kit and
the gates of `python_gates.py` (their `Field.calibrate` call captured). The views read this
machine's files (frame lists, reference grids, calibration data) and its CPU count (the default
number of workers): compare views made on one machine. <!-- check: the producers and file names of views after S2 -->

The gates were TOML configs (`configs/*.toml`) before October 2026; they and the configs that
recorded how each golden was made are in the git history
([Migrating from TOML](../../docs/guide/migrating-from-toml.md)).
