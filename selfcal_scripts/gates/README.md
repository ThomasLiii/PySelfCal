# Byte-equality gates

Every structural change to the pipeline is gated on reproducing fixed products byte for byte.
`run_gates.sh <tag>` runs, on this box (paths to `/mnt/md124` and the staged fixtures are inside
the configs), and writes `workspace/unify/logs/gates_<tag>.log`:

| step | config | what | reference |
| --- | --- | --- | --- |
| pytest | — | the test suite | — |
| continuum | `configs/gate_continuum_unify.toml` | D3 Ch17 NumCol3, 300 frames, iter 50 | `cal_..._Ch17_gate_golden_stat.h5` |
| spectral | `configs/gate_spectral_unify.toml` | D4 AromaticPAHfit NumCol5, 150 frames, iter 20 | `cal_..._AromaticPAHfit_gate_golden_stat.h5` |
| e2e | `configs/gate_e2e.toml` | D3 Ch17 cal + FULL mosaic (std, sigma-clip, wavelength maps) | `*_unify_e2e_golden.{h5,fits}` |
| npass probe | `configs/gate_npass3_unify.toml` | multiline J=4, n=3: INIT + closed-form SKY + per-frame OFFSET refit | `*_unify_npass3_golden{,_pass2sky,_pass3off}.h5` |
| euclid | `configs/gate_euclid_unify.toml` | the EDFN recipe as a `[model]` table on the `euclid` instrument, 3 exposures x 16 detectors | products of `run_euclid_golden.sh` (the original script) |

`run_m13_gate.sh <tag>` runs the npass n=1 gate on the NEP M13 tile (1,101 frames, iter 300,
~50 min) against `*_UNIFYNPASS1GOLDEN_*.h5`. `mode_lowering_snapshot.py dump|compare` snapshots
what every mode lowers to (offset model, sky model, aux maps, mosaic geometry, x0, N-pass hooks)
for a config, so a refactor of the modes/instruments is checked without a solve
(`configs/snap_*.toml` cover the modes without a shipped config). `h5_diff.py` / `fits_diff.py`
compare every dataset / extension exactly.

`python_gates.py` is the same gate set written with the [Python API](../../docs/guide/python-api.md)
(each gate a function: `python -m selfcal_scripts.gates.python_gates continuum | spectral | e2e |
npass3 | euclid | m13`); its products carry a `py` suffix and `run_python_gates.sh <tag> [gate ...]`
compares them with the same goldens (`SELFCAL_REPO` picks the tree). The script imports numpy
before selfcal on purpose: the actions pin the threads themselves.

`config_equivalence.py` checks the configuration layer without a solve: `baseline` / `compare`
record what the engine reads from every shipped config and what its mode lowers to on the real
geometry (before and after an engine change); `typed` converts each config to the Python API's
objects, lowers them again and compares what the engine does with each (the library calls with
defaults filled, per-term damping, offset rows, sky coefficients, jobs, product paths, frames,
staging, tiles, passes).

Goldens are regenerated only when a numerical change is intended, from the committed tree, and
the commit says so. The assembly folds its per-pixel moments deterministically (batch-id order),
so two runs of the same tree are byte-identical on any box load.

The `configs/*_golden.toml` files are records of how each golden was produced by the tree that
made it (some on the frozen baseline worktree); they keep that tree's config spelling
(`mode = "multiline"`, `[tiled]`, `subch_poly_*`) and are not run by the gate scripts. The gate
configs themselves use the current spelling.
