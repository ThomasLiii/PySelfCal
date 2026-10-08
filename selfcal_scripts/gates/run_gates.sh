#!/usr/bin/env bash
# Byte-equality gates: pytest, then the gates of python_gates.py (runs of the Python API), each product compared
# with its float64-norm golden (*golden_f64*, made by make_goldens.sh from a committed tree; a missing golden fails
# the gate). Usage: run_gates.sh <tag> [gate ...]   -> workspace/unify/logs/gates_<tag>.log
# Gates: pytest continuum spectral e2e npass3 euclid m13 rerun (default: all but m13, which run_m13_gate.sh runs;
# rerun runs the continuum gate's record again, `selfcal rerun --overwrite`, so it follows continuum).
# 1. pytest; 2. continuum + spectral cal gates; 3. the e2e cal + mosaic gate (D3 Ch17, 300 frames, full mosaic incl.
# wavelength maps); 4. the npass n=3 probe (INIT+SKY+OFFSET); 5. the Euclid EDFN recipe (3 exposures x 16 detectors);
# 6. the NEP multi-line INIT on the tile M13. SELFCAL_REPO picks the tree (default: this worktree).
set -u
R=${SELFCAL_REPO:-/home/thomasli/selfcal-project/selfcal-memopt}
PY=/home/thomasli/anaconda3/envs/selfcal-memopt/bin/python
G=$R/selfcal_scripts/gates
W=/home/thomasli/selfcal-project/selfcal-memopt/workspace/unify            # logs (gitignored)
TAG=${1:?tag}; shift
GATES=${*:-pytest continuum spectral e2e npass3 euclid rerun}
LOG=${GATES_LOG:-$W/logs/gates_${TAG}.log}
O=/mnt/md124/thomasli/selfcal/outputs
D3=$O/SPHEREx_nep_qr2_det3_6p2arcsec/calibration/cal_Detector3_NumSub10_NumCh34_NumCol3_Ch17
D4=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol5_AromaticPAHfit
E2E_MOS=$O/SPHEREx_nep_qr2_det3_6p2arcsec/mosaic/mosaic_Detector3_NumSub10_NumCh34_NumCol3_Ch17
NP=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3
EU=$O/EDFN_Y_1p5arcsec_unifygolden
M13=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_multiline3_NEPovlp
M13_TAIL=iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2
cd $R
FILTER="=== python gate|Setup LSQR finished|Mosaic saved|\[npass\] pass|Error|Traceback|ConfigError"
run() { PYTHONPATH=$R PYTHONUNBUFFERED=1 $PY -m selfcal_scripts.gates.python_gates $1 2>&1 | grep -E "$FILTER" ; }
# a product and its sidecar (<product>.json)
rmp() { for f in "$@"; do rm -f "$f" "$f.json"; done; }
{
echo "=== gates [$TAG] start $(date) tree=$(git rev-parse --short HEAD) dirty=$(git status --short | grep -v '^??' | wc -l) repo=$R gates=[$GATES]"
for g in $GATES; do
  case $g in
  pytest)
    echo "--- pytest"; $PY -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -2 ;;
  continuum)
    rmp ${D3}_unify_gate_py.h5; run continuum
    echo "--- continuum: byte-diff vs golden_f64"; $PY selfcal_scripts/drivers/diff_cal_h5.py ${D3}_unify_gate_py.h5 ${D3}_gate_golden_f64.h5 2>&1 | tail -1 ;;
  spectral)
    rmp ${D4}_unify_gate_py.h5; run spectral
    echo "--- spectral: byte-diff vs golden_f64"; $PY selfcal_scripts/drivers/diff_cal_h5.py ${D4}_unify_gate_py.h5 ${D4}_gate_golden_f64.h5 2>&1 | tail -1 ;;
  e2e)
    rmp ${D3}_unify_e2e_gate_py.h5 ${E2E_MOS}_unify_e2e_gate_py.fits; run e2e
    echo "--- e2e cal: byte-diff vs unify_e2e_golden_f64"; $PY selfcal_scripts/drivers/diff_cal_h5.py ${D3}_unify_e2e_gate_py.h5 ${D3}_unify_e2e_golden_f64.h5 2>&1 | tail -1
    echo "--- e2e mosaic: byte-diff vs unify_e2e_golden_f64"; $PY $G/fits_diff.py ${E2E_MOS}_unify_e2e_gate_py.fits ${E2E_MOS}_unify_e2e_golden_f64.fits 2>&1 | tail -1 ;;
  npass3)
    rmp ${NP}_unify_npass3_gate_py.h5 ${NP}_unify_npass3_gate_py_pass2sky.h5 ${NP}_unify_npass3_gate_py_pass3off.h5
    rm -f ${NP}_unify_npass3_gate_py_npass_monitor.json
    rm -rf $R/cache/npass_cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_unify_npass3_gate_py
    run npass3
    for prod in "" _pass2sky _pass3off; do
      echo "--- npass3 '${prod:-init cal}': byte-diff vs golden_f64"; $PY $G/h5_diff.py ${NP}_unify_npass3_gate_py${prod}.h5 ${NP}_unify_npass3_golden_f64${prod}.h5 2>&1 | tail -1
    done ;;
  euclid)
    rmp $EU/calibration/cal_EDFN_Y_unify_gate_py.h5 $EU/mosaic/mosaic_EDFN_Y_unify_gate_py.fits; run euclid
    echo "--- euclid cal: byte-diff vs golden_f64"; $PY $G/h5_diff.py $EU/calibration/cal_EDFN_Y_unify_gate_py.h5 $EU/calibration/cal_EDFN_Y_golden_f64.h5 2>&1 | tail -1
    echo "--- euclid mosaic: byte-diff vs golden_f64"; $PY $G/fits_diff.py $EU/mosaic/mosaic_EDFN_Y_unify_gate_py.fits $EU/mosaic/mosaic_EDFN_Y_golden_f64.fits 2>&1 | tail -1 ;;
  m13)
    rmp ${M13}_M13_UNIFYNPASS1GATEPY_${M13_TAIL}.h5
    rm -f ${M13}_UNIFYNPASS1GATEPY_STITCHED_${M13_TAIL}_npass_monitor.json
    run m13
    echo "--- m13: byte-diff vs golden_f64"; $PY $G/h5_diff.py ${M13}_M13_UNIFYNPASS1GATEPY_${M13_TAIL}.h5 ${M13}_M13_UNIFYNPASS1GOLDEN_F64_${M13_TAIL}.h5 2>&1 | tail -1 ;;
  rerun)
    # the record the continuum gate wrote, run again from the record alone (made again: --overwrite)
    REC=$(ls -t $O/SPHEREx_nep_qr2_det3_6p2arcsec/records/calibrate_*.json | xargs grep -l "_unify_gate_py.h5" | head -1)
    echo "=== python gate rerun: $REC"
    PYTHONPATH=$R PYTHONUNBUFFERED=1 $PY -m selfcal rerun --overwrite "$REC" 2>&1 | grep -E "$FILTER"
    echo "--- rerun of the continuum record: byte-diff vs golden_f64"; $PY selfcal_scripts/drivers/diff_cal_h5.py ${D3}_unify_gate_py.h5 ${D3}_gate_golden_f64.h5 2>&1 | tail -1 ;;
  *) echo "unknown gate $g" ;;
  esac
done
echo "=== gates [$TAG] end $(date)"
} > $LOG 2>&1
grep -E "^=== |passed|BYTE-EQUAL|DIFFER|Error|Traceback|No such file|unknown gate" $LOG
