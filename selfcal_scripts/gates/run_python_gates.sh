#!/usr/bin/env bash
# The byte-equality gates run from Python (python_gates.py, the Python API) instead of the TOML configs:
# the same goldens as run_gates.sh / run_m13_gate.sh. Usage: run_python_gates.sh <tag> [gate ...]
#   -> workspace/unify/logs/python_gates_<tag>.log. Gates: continuum spectral e2e npass3 euclid m13 (default: all).
set -u
R=${SELFCAL_REPO:-/home/thomasli/selfcal-project/selfcal-memopt}
PY=/home/thomasli/anaconda3/envs/selfcal-memopt/bin/python
G=$R/selfcal_scripts/gates
W=/home/thomasli/selfcal-project/selfcal-memopt/workspace/unify            # logs (gitignored)
TAG=${1:?tag}; shift
GATES=${*:-continuum spectral e2e npass3 euclid m13}
LOG=$W/logs/python_gates_${TAG}.log
O=/mnt/md124/thomasli/selfcal/outputs
D3=$O/SPHEREx_nep_qr2_det3_6p2arcsec/calibration/cal_Detector3_NumSub10_NumCh34_NumCol3_Ch17
D4=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol5_AromaticPAHfit
E2E_MOS=$O/SPHEREx_nep_qr2_det3_6p2arcsec/mosaic/mosaic_Detector3_NumSub10_NumCh34_NumCol3_Ch17
NP=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3
EU=$O/EDFN_Y_1p5arcsec_unifygolden
M13=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_multiline3_NEPovlp_M13
cd $R
run() { PYTHONPATH=$R PYTHONUNBUFFERED=1 $PY -m selfcal_scripts.gates.python_gates $1 2>&1 | grep -E "=== python gate|Setup LSQR finished|Mosaic saved|\[npass\] pass|Error|Traceback|ConfigError" ; }
{
echo "=== python gates [$TAG] start $(date) tree=$(git rev-parse --short HEAD) dirty=$(git status --short | grep -v '^??' | wc -l) repo=$R"
for g in $GATES; do
  case $g in
  continuum)
    rm -f ${D3}_unify_gate_py.h5; run continuum
    echo "--- continuum (python): byte-diff vs golden_stat"; $PY selfcal_scripts/drivers/diff_cal_h5.py ${D3}_unify_gate_py.h5 ${D3}_gate_golden_stat.h5 2>&1 | tail -1 ;;
  spectral)
    rm -f ${D4}_unify_gate_py.h5; run spectral
    echo "--- spectral (python): byte-diff vs golden_stat"; $PY selfcal_scripts/drivers/diff_cal_h5.py ${D4}_unify_gate_py.h5 ${D4}_gate_golden_stat.h5 2>&1 | tail -1 ;;
  e2e)
    rm -f ${D3}_unify_e2e_gate_py.h5 ${E2E_MOS}_unify_e2e_gate_py.fits; run e2e
    echo "--- e2e cal (python): byte-diff vs unify_e2e_golden"; $PY selfcal_scripts/drivers/diff_cal_h5.py ${D3}_unify_e2e_gate_py.h5 ${D3}_unify_e2e_golden.h5 2>&1 | tail -1
    echo "--- e2e mosaic (python): byte-diff vs unify_e2e_golden"; $PY $G/fits_diff.py ${E2E_MOS}_unify_e2e_gate_py.fits ${E2E_MOS}_unify_e2e_golden.fits 2>&1 | tail -1 ;;
  npass3)
    rm -f ${NP}_unify_npass3_gate_py.h5 ${NP}_unify_npass3_gate_py_pass2sky.h5 ${NP}_unify_npass3_gate_py_pass3off.h5 ${NP}_unify_npass3_gate_py_npass_monitor.json
    rm -rf /home/thomasli/selfcal-project/selfcal-memopt/cache/npass_cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_unify_npass3_gate_py
    run npass3
    for prod in "" _pass2sky _pass3off; do
      echo "--- npass3 (python) '${prod:-init cal}': byte-diff vs golden"; $PY $G/h5_diff.py ${NP}_unify_npass3_gate_py${prod}.h5 ${NP}_unify_npass3_golden${prod}.h5 2>&1 | tail -1
    done ;;
  euclid)
    rm -f $EU/calibration/cal_EDFN_Y_unify_gate_py.h5 $EU/mosaic/mosaic_EDFN_Y_unify_gate_py.fits; run euclid
    echo "--- euclid cal (python): byte-diff vs golden"; $PY $G/h5_diff.py $EU/calibration/cal_EDFN_Y_unify_gate_py.h5 $EU/calibration/cal_EDFN_Y_golden.h5 2>&1 | tail -1
    echo "--- euclid mosaic (python): byte-diff vs golden"; $PY $G/fits_diff.py $EU/mosaic/mosaic_EDFN_Y_unify_gate_py.fits $EU/mosaic/mosaic_EDFN_Y_golden.fits 2>&1 | tail -1 ;;
  m13)
    rm -f ${M13}_UNIFYNPASS1GATEPY_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2.h5
    run m13
    echo "--- m13 (python): byte-diff vs golden"; $PY $G/h5_diff.py ${M13}_UNIFYNPASS1GATEPY_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2.h5 ${M13}_UNIFYNPASS1GOLDEN_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2.h5 2>&1 | tail -1 ;;
  esac
done
echo "=== python gates [$TAG] end $(date)"
} > $LOG 2>&1
grep -E "^=== |BYTE-EQUAL|DIFFER|Error|Traceback" $LOG
