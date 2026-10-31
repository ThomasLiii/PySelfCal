#!/usr/bin/env bash
# Make the float64-norm goldens (*golden_f64*) the gates compare with, by running the gates of python_gates.py on a
# COMMITTED tree. Usage: make_goldens.sh <tag> [gate ...]   (gates: continuum spectral e2e npass3 euclid production
# compact compact_recurrence tiles; default: all) -> $GATES_LOGDIR/make_goldens_<tag>.log. SELFCAL_REPO picks the tree (default: the one this
# script is in; SELFCAL_PYTHON the interpreter, default python on PATH); it must be clean, so that each golden belongs to a commit (the log records it). An existing golden is
# never replaced unless FORCE=1. The goldens are made with the float64 LSQR norms (selfcal/core/lsqr_inplace.py,
# since 2026-10-05): SELFCAL_LSQR_FLOAT32_NORMS=1 is refused. The e2e and production gates write the large goldens
# (mosaics of 4.8 GB and 3.8 GB).
set -u
R=${SELFCAL_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}   # default: the tree this script is in
PY=${SELFCAL_PYTHON:-$(command -v python)}                          # the active env unless set
L=${GATES_LOGDIR:-/mnt/md124/thomasli/selfcal/outputs/_gates/logs}   # the logs, next to the goldens
TAG=${1:?tag}; shift
GATES=${*:-continuum spectral e2e npass3 euclid production compact compact_recurrence tiles}
mkdir -p $L
LOG=$L/make_goldens_${TAG}.log
O=/mnt/md124/thomasli/selfcal/outputs
D3=$O/SPHEREx_nep_qr2_det3_6p2arcsec/calibration/cal_Detector3_NumSub10_NumCh34_NumCol3_Ch17
D4=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol5_AromaticPAHfit
E2E_MOS=$O/SPHEREx_nep_qr2_det3_6p2arcsec/mosaic/mosaic_Detector3_NumSub10_NumCh34_NumCol3_Ch17
NP=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3
EU=$O/EDFN_Y_1p5arcsec_unifygolden
D3C9=$O/SPHEREx_nep_qr2_det3_6p2arcsec/calibration/cal_Detector3_NumSub10_NumCh17_NumCol1_Ch9
D3C9_MOS=$O/SPHEREx_nep_qr2_det3_6p2arcsec/mosaic/mosaic_Detector3_NumSub10_NumCh17_NumCol1_Ch9
D3C9_SNAP=$O/SPHEREx_nep_qr2_det3_6p2arcsec/calibration/snapshots/cal_Detector3_NumSub10_NumCh17_NumCol1_Ch9
cd $R
if [ -n "$(git status --short | grep -v '^??')" ]; then
  echo "make_goldens: $R has uncommitted changes; a golden is made from a committed tree" >&2; exit 1
fi
if [ "${SELFCAL_LSQR_FLOAT32_NORMS:-}" = 1 ]; then
  echo "make_goldens: SELFCAL_LSQR_FLOAT32_NORMS=1 is set; the goldens are float64-norm" >&2; exit 1
fi
run() { PYTHONPATH=$R PYTHONUNBUFFERED=1 $PY -m selfcal_scripts.gates.python_gates $1 2>&1 |
        grep -E "Setup LSQR finished|Mosaic saved|\[npass\] DONE|Error|Traceback|ConfigError|saved" ; }
# a product and its sidecar (<product>.json)
rmp() { for f in "$@"; do rm -f "$f" "$f.json"; done; }
# keep <product> <golden>: the gate product becomes the golden (written under a temporary name, then renamed)
keep() {
  if [ ! -e "$1" ]; then echo "  MISSING product $1"; return; fi
  if [ -e "$2" ] && [ "${FORCE:-0}" != 1 ]; then echo "  kept the existing $2 (FORCE=1 replaces it)"; return; fi
  cp -p "$1" "$2.part.$$" && mv "$2.part.$$" "$2" && echo "  golden: $2"
}
{
echo "=== make goldens [$TAG] start $(date) tree=$(git rev-parse HEAD) repo=$R gates=[$GATES]"
echo "    python=$PY numpy=$($PY -c 'import numpy; print(numpy.__version__)')"
for g in $GATES; do
  echo "--- $g $(date +%T)"
  case $g in
  continuum|spectral)
    [ $g = continuum ] && P=$D3 || P=$D4
    rmp ${P}_unify_gate_py.h5; run $g; keep ${P}_unify_gate_py.h5 ${P}_gate_golden_f64.h5 ;;
  e2e)
    rmp ${D3}_unify_e2e_gate_py.h5 ${E2E_MOS}_unify_e2e_gate_py.fits; run e2e
    keep ${D3}_unify_e2e_gate_py.h5 ${D3}_unify_e2e_golden_f64.h5
    keep ${E2E_MOS}_unify_e2e_gate_py.fits ${E2E_MOS}_unify_e2e_golden_f64.fits ;;
  npass3)
    rmp ${NP}_unify_npass3_gate_py.h5 ${NP}_unify_npass3_gate_py_pass2sky.h5 ${NP}_unify_npass3_gate_py_pass3off.h5
    rm -f ${NP}_unify_npass3_gate_py_npass_monitor.json
    rm -rf $R/cache/npass_cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_unify_npass3_gate_py
    run npass3
    for prod in "" _pass2sky _pass3off; do keep ${NP}_unify_npass3_gate_py${prod}.h5 ${NP}_unify_npass3_golden_f64${prod}.h5; done ;;
  euclid)
    rmp $EU/calibration/cal_EDFN_Y_unify_gate_py.h5 $EU/mosaic/mosaic_EDFN_Y_unify_gate_py.fits; run euclid
    keep $EU/calibration/cal_EDFN_Y_unify_gate_py.h5 $EU/calibration/cal_EDFN_Y_golden_f64.h5
    keep $EU/mosaic/mosaic_EDFN_Y_unify_gate_py.fits $EU/mosaic/mosaic_EDFN_Y_golden_f64.fits ;;
  production)
    rmp ${D3C9}_production_gate_py.h5 ${D3C9_MOS}_production_gate_py.fits; rm -f ${D3C9_SNAP}_production_gate_py_it*.h5
    rm -rf $R/cache/production_gate
    run production
    keep ${D3C9}_production_gate_py.h5 ${D3C9}_production_golden_f64.h5
    keep ${D3C9_MOS}_production_gate_py.fits ${D3C9_MOS}_production_golden_f64.fits ;;
  compact)
    rmp ${D3C9}_compact_gate_py.h5 ${D3C9_MOS}_compact_gate_py.fits; rm -f ${D3C9_SNAP}_compact_gate_py_it*.h5
    rm -rf $R/cache/compact_gate
    run compact
    keep ${D3C9}_compact_gate_py.h5 ${D3C9}_compact_golden_f64.h5
    keep ${D3C9_MOS}_compact_gate_py.fits ${D3C9_MOS}_compact_golden_f64.fits ;;
  compact_recurrence)
    rmp ${D3C9}_compact_recurrence_gate_py.h5 ${D3C9_MOS}_compact_recurrence_gate_py.fits
    rm -f ${D3C9_SNAP}_compact_recurrence_gate_py_it*.h5; rm -rf $R/cache/compact_recurrence_gate
    run compact_recurrence
    keep ${D3C9}_compact_recurrence_gate_py.h5 ${D3C9}_compact_recurrence_golden_f64.h5
    keep ${D3C9_MOS}_compact_recurrence_gate_py.fits ${D3C9_MOS}_compact_recurrence_golden_f64.fits ;;
  tiles)
    rmp ${NP}_tiles_T1_gate_py.h5 ${NP}_tiles_T2_gate_py.h5 ${NP}_tiles_stitched_gate_py.h5
    rm -f ${NP}_tiles_stitched_gate_py_npass_monitor.json; rm -rf $R/cache/tiles_gate
    run tiles
    for t in T1 T2 stitched; do keep ${NP}_tiles_${t}_gate_py.h5 ${NP}_tiles_${t}_golden_f64.h5; done ;;
  *) echo "unknown gate $g" ;;
  esac
done
echo "=== make goldens [$TAG] end $(date)"
} > $LOG 2>&1
grep -E "^=== |^    python=|golden:|kept|MISSING|Error|Traceback|No module named|unknown gate" $LOG
