#!/usr/bin/env bash
# Byte-equality gates for the unify effort. Usage: run_gates.sh <tag>  -> workspace/unify/logs/gates_<tag>.log
# 1. pytest; 2. continuum + spectral cal gates vs *_gate_golden_stat.h5; 3. the e2e cal+mosaic gate; 4. the npass n=3 probe (INIT+SKY+OFFSET) vs *_unify_npass3_golden*; 5. the Euclid EDFN recipe (3 exposures x 16 detectors) vs the script-path golden
#    (D3 Ch17, 300 frames, full mosaic incl. wavelength maps) vs the *_unify_e2e_golden* products (own suffix: gate suffixes must be unique per config, else a cal from another gate is found and the solve is skipped).
set -u
R=/home/thomasli/selfcal-project/selfcal-memopt
PY=/home/thomasli/anaconda3/envs/selfcal-memopt/bin/python
G=$R/selfcal_scripts/gates
W=$R/workspace/unify            # logs (gitignored)
TAG=${1:?tag}
LOG=$W/logs/gates_${TAG}.log
O=/mnt/md124/thomasli/selfcal/outputs
D3=$O/SPHEREx_nep_qr2_det3_6p2arcsec/calibration/cal_Detector3_NumSub10_NumCh34_NumCol3_Ch17
D4=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol5_AromaticPAHfit
E2E_CAL=$O/SPHEREx_nep_qr2_det3_6p2arcsec/calibration/cal_Detector3_NumSub10_NumCh34_NumCol3_Ch17
E2E_MOS=$O/SPHEREx_nep_qr2_det3_6p2arcsec/mosaic/mosaic_Detector3_NumSub10_NumCh34_NumCol3_Ch17
cd $R
{
echo "=== unify gates [$TAG] start $(date) tree=$(git rev-parse --short HEAD) dirty=$(git status --short | grep -v '^??' | wc -l)"
echo "--- pytest"; $PY -m pytest tests/ -q 2>&1 | tail -2
rm -f ${D3}_unify_gate.h5 ${D4}_unify_gate.h5 ${E2E_CAL}_unify_e2e_gate.h5 ${E2E_MOS}_unify_e2e_gate.fits
for g in continuum spectral; do
  echo "--- $g cal gate $(date +%T)"
  PYTHONUNBUFFERED=1 $PY -m selfcal_scripts.run --no-log --config $G/configs/gate_${g}_unify.toml 2>&1 | grep -E "Setup LSQR finished|Error|Traceback|saved"
done
echo "--- continuum: byte-diff vs golden_stat"; $PY selfcal_scripts/drivers/diff_cal_h5.py ${D3}_unify_gate.h5 ${D3}_gate_golden_stat.h5 2>&1 | tail -1
echo "--- spectral: byte-diff vs golden_stat";  $PY selfcal_scripts/drivers/diff_cal_h5.py ${D4}_unify_gate.h5 ${D4}_gate_golden_stat.h5 2>&1 | tail -1
echo "--- e2e cal+mosaic gate $(date +%T)"
PYTHONUNBUFFERED=1 $PY -m selfcal_scripts.run --no-log --config $G/configs/gate_e2e.toml 2>&1 | grep -E "Setup LSQR finished|passes finished|Mosaic saved|Error|Traceback|saved"
echo "--- e2e cal: byte-diff vs unify_e2e_golden"; $PY selfcal_scripts/drivers/diff_cal_h5.py ${E2E_CAL}_unify_e2e_gate.h5 ${E2E_CAL}_unify_e2e_golden.h5 2>&1 | tail -1
echo "--- e2e mosaic: byte-diff vs unify_e2e_golden"; $PY $G/fits_diff.py ${E2E_MOS}_unify_e2e_gate.fits ${E2E_MOS}_unify_e2e_golden.fits 2>&1 | tail -1
NP=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3
rm -f ${NP}_unify_npass3_gate.h5 ${NP}_unify_npass3_gate_pass2sky.h5 ${NP}_unify_npass3_gate_pass3off.h5 ${NP}_unify_npass3_gate_npass_monitor.json
rm -rf $R/cache/npass_cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_unify_npass3_gate
echo "--- npass n=3 probe gate (INIT + SKY + OFFSET) $(date +%T)"
PYTHONUNBUFFERED=1 $PY -m selfcal_scripts.run --no-log --config $G/configs/gate_npass3_unify.toml 2>&1 | grep -E "\[npass\] pass|\[npass\] DONE|Error|Traceback"
# Since the deterministic-fold commit (391cf9f) the goldens and the gate runs are byte-reproducible:
# every dataset must match exactly (no rounding allowance).
for prod in "" _pass2sky _pass3off; do
  echo "--- npass3 product '${prod:-init cal}': byte-diff vs golden"; $PY $G/h5_diff.py ${NP}_unify_npass3_gate${prod}.h5 ${NP}_unify_npass3_golden${prod}.h5 2>&1 | tail -1
done
EU=$O/EDFN_Y_1p5arcsec_unifygolden
rm -f $EU/calibration/cal_EDFN_Y_unify_gate.h5 $EU/mosaic/mosaic_EDFN_Y_unify_gate.fits
echo "--- euclid gate (EDFN recipe as a [model] table on the euclid instrument, vs the script-path golden) $(date +%T)"
PYTHONUNBUFFERED=1 $PY -m selfcal_scripts.run --no-log --config $G/configs/gate_euclid_unify.toml 2>&1 | grep -E "Setup LSQR finished|Mosaic saved|Error|Traceback|saved"
echo "--- euclid cal: byte-diff vs golden"; $PY $G/h5_diff.py $EU/calibration/cal_EDFN_Y_unify_gate.h5 $EU/calibration/cal_EDFN_Y_golden.h5 2>&1 | tail -1
echo "--- euclid mosaic: byte-diff vs golden"; $PY $G/fits_diff.py $EU/mosaic/mosaic_EDFN_Y_unify_gate.fits $EU/mosaic/mosaic_EDFN_Y_golden.fits 2>&1 | tail -1
echo "=== unify gates [$TAG] end $(date)"
} > $LOG 2>&1
grep -E "^=== |passed|BYTE-EQUAL|DIFFER|Error|Traceback" $LOG
