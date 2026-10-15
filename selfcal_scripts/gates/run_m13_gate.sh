#!/usr/bin/env bash
# The npass n=1 M13 tile gate (python_gates.py m13: the NEP multi-line INIT on the adaptive tile M13) on the CURRENT
# tree vs its float64-norm golden (make_goldens.sh m13). Usage: run_m13_gate.sh <tag>
#   -> $GATES_LOGDIR/m13_gate_<tag>.log. SELFCAL_REPO picks the tree (default: this worktree).
set -u
R=${SELFCAL_REPO:-/home/thomasli/selfcal-project/selfcal-memopt}
L=${GATES_LOGDIR:-/mnt/md124/thomasli/selfcal/outputs/_gates/logs}   # the logs, next to the goldens
TAG=${1:?tag}
mkdir -p $L
GATES_LOG=$L/m13_gate_${TAG}.log SELFCAL_REPO=$R exec "$R/selfcal_scripts/gates/run_gates.sh" "$TAG" m13
