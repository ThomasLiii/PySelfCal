#!/usr/bin/env bash
# Make the float64-norm goldens (*golden_f64*) the gates compare with, by running the TOML gate configs on a
# COMMITTED tree. Usage: make_goldens.sh <tag> [gate ...]   (gates: continuum spectral e2e npass3 euclid m13;
# default: all) -> workspace/unify/logs/make_goldens_<tag>.log. SELFCAL_REPO picks the tree (default: this
# worktree); it must be clean, so that each golden belongs to a commit (the log records it). An existing golden is
# never replaced unless FORCE=1. The goldens are made with the float64 LSQR norms (selfcal/core/lsqr_inplace.py,
# since 2026-10-05): SELFCAL_LSQR_FLOAT32_NORMS=1 is refused. The e2e and m13 gates are the large ones (a 4.8 GB
# mosaic; ~90 GB of memory and ~35 GB of scratch).
set -u
R=${SELFCAL_REPO:-/home/thomasli/selfcal-project/selfcal-memopt}
PY=/home/thomasli/anaconda3/envs/selfcal-memopt/bin/python
G=$R/selfcal_scripts/gates
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
M13=$O/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration/cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_multiline3_NEPovlp_M13
M13_TAIL=iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2.h5
cd $R
if [ -n "$(git status --short | grep -v '^??')" ]; then
  echo "make_goldens: $R has uncommitted changes; a golden is made from a committed tree" >&2; exit 1
fi
if [ "${SELFCAL_LSQR_FLOAT32_NORMS:-}" = 1 ]; then
  echo "make_goldens: SELFCAL_LSQR_FLOAT32_NORMS=1 is set; the goldens are float64-norm" >&2; exit 1
fi
run() { PYTHONUNBUFFERED=1 $PY -m selfcal_scripts.run --no-log --config $G/configs/$1 2>&1 |
        grep -E "Setup LSQR finished|passes finished|Mosaic saved|\[npass\] DONE|Error|Traceback|saved" ; }
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
    rm -f ${P}_unify_gate.h5; run gate_${g}_unify.toml; keep ${P}_unify_gate.h5 ${P}_gate_golden_f64.h5 ;;
  e2e)
    rm -f ${D3}_unify_e2e_gate.h5 ${E2E_MOS}_unify_e2e_gate.fits; run gate_e2e.toml
    keep ${D3}_unify_e2e_gate.h5 ${D3}_unify_e2e_golden_f64.h5
    keep ${E2E_MOS}_unify_e2e_gate.fits ${E2E_MOS}_unify_e2e_golden_f64.fits ;;
  npass3)
    rm -f ${NP}_unify_npass3_gate.h5 ${NP}_unify_npass3_gate_pass2sky.h5 ${NP}_unify_npass3_gate_pass3off.h5 ${NP}_unify_npass3_gate_npass_monitor.json
    rm -rf /home/thomasli/selfcal-project/selfcal-memopt/cache/npass_cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_unify_npass3_gate
    run gate_npass3_unify.toml
    for prod in "" _pass2sky _pass3off; do keep ${NP}_unify_npass3_gate${prod}.h5 ${NP}_unify_npass3_golden_f64${prod}.h5; done ;;
  euclid)
    rm -f $EU/calibration/cal_EDFN_Y_unify_gate.h5 $EU/mosaic/mosaic_EDFN_Y_unify_gate.fits; run gate_euclid_unify.toml
    keep $EU/calibration/cal_EDFN_Y_unify_gate.h5 $EU/calibration/cal_EDFN_Y_golden_f64.h5
    keep $EU/mosaic/mosaic_EDFN_Y_unify_gate.fits $EU/mosaic/mosaic_EDFN_Y_golden_f64.fits ;;
  m13)
    rm -f ${M13}_UNIFYNPASS1GATE_${M13_TAIL}; run gate_npass1_M13_unify_gate.toml
    keep ${M13}_UNIFYNPASS1GATE_${M13_TAIL} ${M13}_UNIFYNPASS1GOLDEN_F64_${M13_TAIL} ;;
  *) echo "unknown gate $g" ;;
  esac
done
echo "=== make goldens [$TAG] end $(date)"
} > $LOG 2>&1
grep -E "^=== |golden:|kept|MISSING|Error|Traceback|unknown gate" $LOG
