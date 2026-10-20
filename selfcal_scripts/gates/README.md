# Byte-equality gates

Every structural change to the pipeline is gated on reproducing fixed products byte for byte.
The gates are Python: each is a function of `python_gates.py`, a calibration written with the
[Python API](../../docs/guide/python-api.md) on a fixed set of frames
(`python -m selfcal_scripts.gates.python_gates continuum | spectral | e2e | npass3 | euclid`).
Their products carry a `py` suffix. `run_gates.sh <tag> [gate ...]` runs, on this box (the paths
to `/mnt/md124` and the fixtures are in `python_gates.py`), the steps named, compares each
product with its golden and writes `gates_<tag>.log` to `$GATES_LOGDIR` (default
`/mnt/md124/thomasli/selfcal/outputs/_gates/logs`, next to the goldens; `make_goldens.sh` logs there
too). The scripts name no worktree: `SELFCAL_REPO` picks the tree (default: the one the script is
in; its `cache/` is the scratch), `SELFCAL_PYTHON` the interpreter (default: `python` on `PATH`; the
log records it and its numpy) and `SELFCAL_GATE_FIXTURES` the frame fixtures (default
`/home/thomasli/selfcal-project/gate_fixtures`: `qr2_d3_links400` for continuum, `qr2_d3_frames300`
for e2e, `pahfit_sanity` for spectral and npass3). The gates read
and write nothing in `workspace/`: their inputs are the repository, the fixtures and the goldens. The
steps are `pytest` (the test suite, first), then the gates below; the default is every step.

| gate | what | golden |
| --- | --- | --- |
| continuum | D3 Ch17 NumCol3, 300 frames, iter 50 (51 MB cal) | `cal_..._Ch17_gate_golden_f64.h5` |
| spectral | D4 AromaticPAHfit NumCol5, 150 frames, iter 20 (73 MB cal) | `cal_..._AromaticPAHfit_gate_golden_f64.h5` |
| e2e | D3 Ch17 cal + FULL mosaic (std, sigma-clip, wavelength maps; a 4.8 GB mosaic) | `*_unify_e2e_golden_f64.{h5,fits}` |
| npass3 | multiline J=4, n=3: INIT + closed-form SKY + per-frame direct OFFSET solve | `*_unify_npass3_golden_f64{,_pass2sky,_pass3off}.h5` |
| euclid | the EDFN recipe as an `sc.Model` on `sc.Euclid`, 3 exposures x 16 detectors | `cal_EDFN_Y_golden_f64.h5`, `mosaic_EDFN_Y_golden_f64.fits` |
| rerun | the continuum gate's record run again from the record alone (`selfcal rerun --overwrite RECORD`) | the continuum golden |

`python_gates.py` imports numpy before selfcal on purpose: the actions pin the threads
themselves. `h5_diff.py` / `fits_diff.py` compare every dataset / extension exactly;
`diff_cal_h5.py` compares two cal files dataset by dataset across the cal schemas (a legacy
single-offset file against a multi-map one), for the continuum, spectral, e2e and rerun gates.

Goldens are regenerated only when a numerical change is intended, from the committed tree, and
the commit says so: `make_goldens.sh <tag> [gate ...]` runs the Python gates on a clean tree
(`SELFCAL_REPO`) and keeps each product as its golden (an existing golden is replaced only with
`FORCE=1`; the log records the commit). The goldens are float64-norm (`*golden_f64*`): since
2026-10-05 (e8781bb) LSQR accumulates the norms of long float32 vectors in float64, which changed
the bytes of production-size solves, and every run uses those norms
(`SELFCAL_LSQR_FLOAT32_NORMS=1`, the old norms, is refused by `make_goldens.sh`). The npass3
gate's float64 goldens were made 2026-10-17 from 1ef83ae, when the N-pass passes stopped building the
joint matrix (the OFFSET pass solves each frame directly from its normal equations, the SKY pass sums
the per-pixel moments alone); its earlier goldens (`*_unify_npass3_golden.h5` etc., 2026-09-28) were
made with the float32 norms and the per-frame polynomial refit. The earlier goldens (`*_golden_stat*`, `*_golden.*`)
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
| `gate__<name>.json` | each gate of `python_gates.py` (`continuum`, `spectral`, `e2e`, `npass3`, `euclid`), its `Field.calibrate` call captured |

The views read this machine's files (frame lists, reference grids, calibration data) and its CPU
count (the default number of workers): compare views made on one machine.

The gates were TOML configs (`configs/*.toml`) before October 2026; they and the configs that
recorded how each golden was made are in the git history
([Migrating from TOML](../../docs/guide/migrating-from-toml.md)).
