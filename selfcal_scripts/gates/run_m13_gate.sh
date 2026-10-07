#!/usr/bin/env bash
# npass n=1 M13 tile gate on the CURRENT tree vs its float64-norm golden (make_goldens.sh m13). Usage: run_m13_gate.sh <tag>
# SELFCAL_REPO picks the tree (default: this worktree).
set -u
R=${SELFCAL_REPO:-/home/thomasli/selfcal-project/selfcal-memopt}; PY=/home/thomasli/anaconda3/envs/selfcal-memopt/bin/python; G=$R/selfcal_scripts/gates
W=/home/thomasli/selfcal-project/selfcal-memopt/workspace/unify            # logs (gitignored)
TAG=${1:?tag}; LOG=$W/logs/m13_gate_${TAG}.log
CAL=/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration
P=$CAL/cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_multiline3_NEPovlp_M13
cd $R
{
echo "=== m13 gate [$TAG] start $(date) tree=$(git rev-parse --short HEAD) dirty=$(git status --short | grep -v '^??' | wc -l)"
rm -f ${P}_UNIFYNPASS1GATE_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2.h5
rm -f $CAL/cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_multiline3_NEPovlp_UNIFYNPASS1GATE_STITCHED_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2_npass_monitor.json
PYTHONUNBUFFERED=1 $PY -m selfcal_scripts.run --no-log --config $G/configs/gate_npass1_M13_unify_gate.toml > $W/logs/m13_gate_${TAG}.console 2>&1
echo "run rc=$? $(date +%T)"
echo "--- h5_diff (every dataset + attrs, exact)"; $PY $G/h5_diff.py ${P}_UNIFYNPASS1GATE_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2.h5 ${P}_UNIFYNPASS1GOLDEN_F64_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2.h5 2>&1 | tail -12
echo "--- diff_cal_h5 (schema-aware)"; $PY selfcal_scripts/drivers/diff_cal_h5.py ${P}_UNIFYNPASS1GATE_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2.h5 ${P}_UNIFYNPASS1GOLDEN_F64_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2.h5 2>&1 | tail -3
echo "=== m13 gate [$TAG] end $(date)"
} > $LOG 2>&1
