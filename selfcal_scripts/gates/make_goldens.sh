#!/usr/bin/env bash
# Make the float64-norm goldens (*golden_f64*) the gates compare with, by running the gates of python_gates.py on a
# COMMITTED tree. Usage: make_goldens.sh <tag> [gate ...]   (gates: continuum spectral e2e npass3 euclid m13;
# default: all) -> workspace/unify/logs/make_goldens_<tag>.log. SELFCAL_REPO picks the tree (default: this
# worktree); it must be clean, so that each golden belongs to a commit (the log records it). An existing golden is
# never replaced unless FORCE=1. The goldens are made with the float64 LSQR norms (selfcal/core/lsqr_inplace.py,
# since 2026-10-05): SELFCAL_LSQR_FLOAT32_NORMS=1 is refused. The e2e and m13 gates are the large ones (a 4.8 GB
# mosaic; ~90 GB of memory and ~35 GB of scratch).
set -u
R=${SELFCAL_REPO:-/home/thomasli/selfcal-project/selfcal-memopt}
PY=/home/thomasli/anaconda3/envs/selfcal-memopt/bin/python
W=/home/thomasli/selfcal-project/selfcal-memopt/workspace/unify            # logs (gitignored)
TAG=${1:?tag}; shift
GATES=${*:-continuum spectral e2e npass3 euclid m13}
LOG=$W/logs/make_goldens_${TAG}.log
O=/mnt/md124/thomasli/selfcal/outputs
D3=$O/SPHEREx_nep_qr2_det3_6p2arcsec/calibration/cal_Detector3_NumSub10_NumCh34_NumCol3_Ch17
D4=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol5_AromaticPAHfit
E2E_MOS=$O/SPHEREx_nep_qr2_det3_6p2arcsec/mosaic/mosaic_Detector3_NumSub10_NumCh34_NumCol3_Ch17
NP=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3
EU=$O/EDFN_Y_1p5arcsec_unifygolden
M13=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_multiline3_NEPovlp
M13_TAIL=iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2
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
  m13)
    rmp ${M13}_M13_UNIFYNPASS1GATEPY_${M13_TAIL}.h5
    rm -f ${M13}_UNIFYNPASS1GATEPY_STITCHED_${M13_TAIL}_npass_monitor.json
    run m13
    keep ${M13}_M13_UNIFYNPASS1GATEPY_${M13_TAIL}.h5 ${M13}_M13_UNIFYNPASS1GOLDEN_F64_${M13_TAIL}.h5 ;;
  *) echo "unknown gate $g" ;;
  esac
done
echo "=== make goldens [$TAG] end $(date)"
} > $LOG 2>&1
grep -E "^=== |golden:|kept|MISSING|Error|Traceback|unknown gate" $LOG
