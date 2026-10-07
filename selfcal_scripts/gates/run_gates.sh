#!/usr/bin/env bash
# Byte-equality gates (the TOML configs). Usage: run_gates.sh <tag> [gate ...]
#   -> workspace/unify/logs/gates_<tag>.log. Gates: pytest continuum spectral e2e npass3 euclid (default: all).
# 1. pytest; 2. continuum + spectral cal gates; 3. the e2e cal + mosaic gate (D3 Ch17, 300 frames, full mosaic incl.
# wavelength maps; own suffix: gate suffixes must be unique per config, else a cal from another gate is found and the
# solve is skipped); 4. the npass n=3 probe (INIT+SKY+OFFSET); 5. the Euclid EDFN recipe (3 exposures x 16 detectors).
# Each product is compared with its float64-norm golden (*golden_f64*, made by make_goldens.sh from a committed tree);
# a missing golden fails the gate. SELFCAL_REPO picks the tree (default: this worktree).
set -u
R=${SELFCAL_REPO:-/home/thomasli/selfcal-project/selfcal-memopt}
PY=/home/thomasli/anaconda3/envs/selfcal-memopt/bin/python
G=$R/selfcal_scripts/gates
W=/home/thomasli/selfcal-project/selfcal-memopt/workspace/unify            # logs (gitignored)
TAG=${1:?tag}; shift
GATES=${*:-pytest continuum spectral e2e npass3 euclid}
LOG=$W/logs/gates_${TAG}.log
O=/mnt/md124/thomasli/selfcal/outputs
D3=$O/SPHEREx_nep_qr2_det3_6p2arcsec/calibration/cal_Detector3_NumSub10_NumCh34_NumCol3_Ch17
D4=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol5_AromaticPAHfit
E2E_MOS=$O/SPHEREx_nep_qr2_det3_6p2arcsec/mosaic/mosaic_Detector3_NumSub10_NumCh34_NumCol3_Ch17
NP=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3
EU=$O/EDFN_Y_1p5arcsec_unifygolden
run() { PYTHONUNBUFFERED=1 $PY -m selfcal_scripts.run --no-log --config $G/configs/$1 2>&1 | grep -E "$2" ; }
cd $R
{
echo "=== unify gates [$TAG] start $(date) tree=$(git rev-parse --short HEAD) dirty=$(git status --short | grep -v '^??' | wc -l) repo=$R gates=[$GATES]"
for g in $GATES; do
  case $g in
  pytest)
    echo "--- pytest"; $PY -m pytest tests/ -q -p no:cacheprovider 2>&1 | tail -2 ;;
  continuum|spectral)
    [ $g = continuum ] && P=$D3 || P=$D4
    rm -f ${P}_unify_gate.h5
    echo "--- $g cal gate $(date +%T)"; run gate_${g}_unify.toml "Setup LSQR finished|Error|Traceback|saved"
    echo "--- $g: byte-diff vs golden_f64"; $PY selfcal_scripts/drivers/diff_cal_h5.py ${P}_unify_gate.h5 ${P}_gate_golden_f64.h5 2>&1 | tail -1 ;;
  e2e)
    rm -f ${D3}_unify_e2e_gate.h5 ${E2E_MOS}_unify_e2e_gate.fits
    echo "--- e2e cal+mosaic gate $(date +%T)"; run gate_e2e.toml "Setup LSQR finished|passes finished|Mosaic saved|Error|Traceback|saved"
    echo "--- e2e cal: byte-diff vs unify_e2e_golden_f64"; $PY selfcal_scripts/drivers/diff_cal_h5.py ${D3}_unify_e2e_gate.h5 ${D3}_unify_e2e_golden_f64.h5 2>&1 | tail -1
    echo "--- e2e mosaic: byte-diff vs unify_e2e_golden_f64"; $PY $G/fits_diff.py ${E2E_MOS}_unify_e2e_gate.fits ${E2E_MOS}_unify_e2e_golden_f64.fits 2>&1 | tail -1 ;;
  npass3)
    rm -f ${NP}_unify_npass3_gate.h5 ${NP}_unify_npass3_gate_pass2sky.h5 ${NP}_unify_npass3_gate_pass3off.h5 ${NP}_unify_npass3_gate_npass_monitor.json
    rm -rf /home/thomasli/selfcal-project/selfcal-memopt/cache/npass_cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_unify_npass3_gate
    echo "--- npass n=3 probe gate (INIT + SKY + OFFSET) $(date +%T)"; run gate_npass3_unify.toml "\[npass\] pass|\[npass\] DONE|Error|Traceback"
    for prod in "" _pass2sky _pass3off; do
      echo "--- npass3 product '${prod:-init cal}': byte-diff vs golden_f64"; $PY $G/h5_diff.py ${NP}_unify_npass3_gate${prod}.h5 ${NP}_unify_npass3_golden_f64${prod}.h5 2>&1 | tail -1
    done ;;
  euclid)
    rm -f $EU/calibration/cal_EDFN_Y_unify_gate.h5 $EU/mosaic/mosaic_EDFN_Y_unify_gate.fits
    echo "--- euclid gate (EDFN recipe as a [model] table on the euclid instrument) $(date +%T)"; run gate_euclid_unify.toml "Setup LSQR finished|Mosaic saved|Error|Traceback|saved"
    echo "--- euclid cal: byte-diff vs golden_f64"; $PY $G/h5_diff.py $EU/calibration/cal_EDFN_Y_unify_gate.h5 $EU/calibration/cal_EDFN_Y_golden_f64.h5 2>&1 | tail -1
    echo "--- euclid mosaic: byte-diff vs golden_f64"; $PY $G/fits_diff.py $EU/mosaic/mosaic_EDFN_Y_unify_gate.fits $EU/mosaic/mosaic_EDFN_Y_golden_f64.fits 2>&1 | tail -1 ;;
  *) echo "unknown gate $g" ;;
  esac
done
echo "=== unify gates [$TAG] end $(date)"
} > $LOG 2>&1
grep -E "^=== |passed|BYTE-EQUAL|DIFFER|Error|Traceback|No such file|unknown gate" $LOG
