# Byte-equality gates

Every structural change to the pipeline is gated on reproducing fixed products byte for byte.
`run_gates.sh <tag> [gate ...]` runs, on this box (paths to `/mnt/md124` and the staged fixtures
are inside the configs), the gates named (default: all), and writes
`workspace/unify/logs/gates_<tag>.log`; `SELFCAL_REPO` picks the tree:

| step | config | what | reference |
| --- | --- | --- | --- |
| pytest | — | the test suite | — |
| continuum | `configs/gate_continuum_unify.toml` | D3 Ch17 NumCol3, 300 frames, iter 50 (51 MB cal) | `cal_..._Ch17_gate_golden_f64.h5` |
| spectral | `configs/gate_spectral_unify.toml` | D4 AromaticPAHfit NumCol5, 150 frames, iter 20 (73 MB cal) | `cal_..._AromaticPAHfit_gate_golden_f64.h5` |
| e2e | `configs/gate_e2e.toml` | D3 Ch17 cal + FULL mosaic (std, sigma-clip, wavelength maps; a 4.8 GB mosaic) | `*_unify_e2e_golden_f64.{h5,fits}` |
| npass probe | `configs/gate_npass3_unify.toml` | multiline J=4, n=3: INIT + closed-form SKY + per-frame OFFSET refit | `*_unify_npass3_golden_f64{,_pass2sky,_pass3off}.h5` |
| euclid | `configs/gate_euclid_unify.toml` | the EDFN recipe as a `[model]` table on the `euclid` instrument, 3 exposures x 16 detectors | `cal_EDFN_Y_golden_f64.h5`, `mosaic_EDFN_Y_golden_f64.fits` |

`run_m13_gate.sh <tag>` runs the npass n=1 gate on the NEP M13 tile (1,101 frames, iter 300,
~50 min on a quiet box, ~90 GB of memory and ~35 GB of scratch) against
`*_UNIFYNPASS1GOLDEN_F64_*.h5`. `mode_lowering_snapshot.py dump|compare` snapshots
what every mode lowers to (offset model, sky model, aux maps, mosaic geometry, x0, N-pass hooks)
for a config, so a refactor of the modes/instruments is checked without a solve
(`configs/snap_*.toml` cover the modes without a shipped config). `h5_diff.py` / `fits_diff.py`
compare every dataset / extension exactly.

`python_gates.py` is the same gate set written with the [Python API](../../docs/guide/python-api.md)
(each gate a function: `python -m selfcal_scripts.gates.python_gates continuum | spectral | e2e |
npass3 | euclid | m13`); its products carry a `py` suffix and `run_python_gates.sh <tag> [gate ...]`
compares them with the same goldens (`SELFCAL_REPO` picks the tree). The script imports numpy
before selfcal on purpose: the actions pin the threads themselves. `run_python_gates.sh` has one
gate more, `rerun`: it runs the continuum gate's record again from the record alone
(`selfcal rerun --overwrite RECORD`) and compares the product with the same golden.

`config_equivalence.py` checks the configuration layer without a solve: `baseline` / `compare`
record what the engine reads from every shipped config and what its mode lowers to on the real
geometry (before and after an engine change); `typed` converts each config to the Python API's
objects, lowers them again and compares what the engine does with each (the library calls with
defaults filled, per-term damping, offset rows, sky coefficients, jobs, product paths, frames,
staging, tiles, passes); `runs` does the same for the hand-written run scripts,
`selfcal_scripts/runs/<name>.py` against `configs/<name>.toml` (a script without a config, such as a
campaign's, is listed and skipped).

Goldens are regenerated only when a numerical change is intended, from the committed tree, and
the commit says so: `make_goldens.sh <tag> [gate ...]` runs the gate configs on a clean tree
(`SELFCAL_REPO`) and keeps each product as its golden (an existing golden is replaced only with
`FORCE=1`; the log records the commit). The goldens are float64-norm (`*golden_f64*`): since
2026-10-05 (e8781bb) LSQR accumulates the norms of long float32 vectors in float64, which changed the
bytes of production-size solves, and every run uses those norms (`SELFCAL_LSQR_FLOAT32_NORMS=1`, the
old norms, is refused by `make_goldens.sh`). The earlier goldens (`*_golden_stat*`, `*_golden.*`,
`*_UNIFYNPASS1GOLDEN_iter300*`) are kept as made, by the float32 norms. A gate whose float64 golden
has not been made yet fails with a missing file. The assembly folds its per-pixel moments
deterministically (batch-id order), so two runs of the same tree are byte-identical on any box
load.

The `configs/*_golden.toml` files are records of how each golden was produced by the tree that
made it (some on the frozen baseline worktree); they keep that tree's config spelling
(`mode = "multiline"`, `[tiled]`, `subch_poly_*`) and are not run by the gate scripts. The gate
configs themselves use the current spelling.
