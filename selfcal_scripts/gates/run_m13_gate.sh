#!/usr/bin/env bash
# The npass n=1 M13 tile gate (python_gates.py m13: the NEP multi-line INIT on the adaptive tile M13) on the CURRENT
# tree vs its float64-norm golden (make_goldens.sh m13). Usage: run_m13_gate.sh <tag>
#   -> workspace/unify/logs/m13_gate_<tag>.log. SELFCAL_REPO picks the tree (default: this worktree).
set -u
R=${SELFCAL_REPO:-/home/thomasli/selfcal-project/selfcal-memopt}
W=/home/thomasli/selfcal-project/selfcal-memopt/workspace/unify            # logs (gitignored)
TAG=${1:?tag}
GATES_LOG=$W/logs/m13_gate_${TAG}.log SELFCAL_REPO=$R exec "$R/selfcal_scripts/gates/run_gates.sh" "$TAG" m13
