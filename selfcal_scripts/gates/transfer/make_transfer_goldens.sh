#!/usr/bin/env bash
# Make the transfer gate's goldens: the products of the WORKSPACE implementations (the "before" of moving the pixel
# weights and the post-solve gauge into the library) on the gate fixture -- D3 Ch9 (NumCh17, NumCol1) on the 300-frame
# qr2 set, the field SPHEREx_nep_qr2_det3_6p2arcsec. Steps: before_solve.py (weighted solve, no clip) -> apply_gauge_q.py
# (HDEG=2, the exact geometry) -> before_mosaic.py (weighted coadd of the UNGAUGED cal: mean, std, 2-sigma clipped mean)
# -> gauge_map.py (the gauge's polynomial on the map). Each product is kept as a golden (*_transfer*_golden_f64*); the
# library's run of the same recipe (python_gates.py: transfer) must reproduce every one byte for byte. Run from a
# COMMITTED tree (the log records the commit); an existing golden is replaced only with FORCE=1.
#   make_transfer_goldens.sh <tag>
set -u
T=$(cd "$(dirname "$0")" && pwd)                     # this directory: the archived implementations
R=${SELFCAL_REPO:-$(cd "$T/../../.." && pwd)}
PY=/home/thomasli/anaconda3/envs/selfcal-memopt/bin/python
TAG=${1:?tag}
O=/mnt/md124/thomasli/selfcal/outputs
FIELD=$O/SPHEREx_nep_qr2_det3_6p2arcsec
FX=${TRANSFER_FIXTURE:-/home/thomasli/selfcal-project/selfcal-memopt/cache/reproj_nvme_SPHEREx_nep_qr2_det3_6p2arcsec}
SCRATCH=${TRANSFER_SCRATCH:-$R/cache/transfer_goldens_$TAG}
LOGDIR=$R/workspace/unify/logs; mkdir -p $LOGDIR $SCRATCH; LOG=$LOGDIR/make_transfer_goldens_${TAG}.log
C=$FIELD/calibration/cal_Detector3_NumSub10_NumCh17_NumCol1_Ch9
M=$FIELD/mosaic/mosaic_Detector3_NumSub10_NumCh17_NumCol1_Ch9
cd $R
if [ -n "$(git status --short | grep -v '^??')" ]; then echo "make_transfer_goldens: $R has uncommitted changes" >&2; exit 1; fi
if [ "${SELFCAL_LSQR_FLOAT32_NORMS:-}" = 1 ]; then echo "make_transfer_goldens: float32 norms refused" >&2; exit 1; fi
for g in ${C}_transfer_golden_f64.h5 ${C}_transfer_gq2_golden_f64.h5 ${M}_transfer_golden_f64.fits ${M}_transfer_gq2map_golden_f64.fits; do
  if [ -e $g ] && [ "${FORCE:-}" != 1 ]; then echo "make_transfer_goldens: $g exists (FORCE=1 to replace)" >&2; exit 1; fi
done
{
  echo "[goldens] tree $R @ $(git rev-parse HEAD) ($(git branch --show-current)); fixture $FX; $(date)"
  export PYTHONPATH=$R NUMPY_MADVISE_HUGEPAGE=0 DET=3 SCENE_FIELD=$FIELD SCENE_SCRATCH=$SCRATCH GAUGE_REF=$FIELD/ref.fits
  rm -f ${C}_transfer_before_py.h5 ${C}_transfer_before_py.h5.json ${C}_transfer_before_py_gq2.h5 ${M}_transfer_before_py.fits ${M}_transfer_before_py.fits.json ${M}_transfer_before_py_gq2map.fits
  $PY $T/before_solve.py $FX transfer_before_py; echo "[goldens] solve exit $? $(date)"
  HDEG=2 $PY $T/apply_gauge_q.py ${C}_transfer_before_py.h5 --frames-dir $FX --out ${C}_transfer_before_py_gq2.h5; echo "[goldens] quad gauge exit $? $(date)"
  $PY $T/before_mosaic.py $FX ${C}_transfer_before_py.h5 transfer_before_py; echo "[goldens] coadd exit $? $(date)"
  $PY $T/gauge_map.py ${C}_transfer_before_py.h5 ${C}_transfer_before_py_gq2.h5 ${M}_transfer_before_py.fits ${M}_transfer_before_py_gq2map.fits; echo "[goldens] gauge on the map exit $? $(date)"
  cp ${C}_transfer_before_py.h5 ${C}_transfer_golden_f64.h5 && cp ${C}_transfer_before_py.h5.json ${C}_transfer_golden_f64.h5.json
  cp ${C}_transfer_before_py_gq2.h5 ${C}_transfer_gq2_golden_f64.h5
  cp ${M}_transfer_before_py.fits ${M}_transfer_golden_f64.fits && cp ${M}_transfer_before_py.fits.json ${M}_transfer_golden_f64.fits.json
  cp ${M}_transfer_before_py_gq2map.fits ${M}_transfer_gq2map_golden_f64.fits
  echo "[goldens] done $(date)"; ls -la ${C}_transfer_golden_f64.h5 ${C}_transfer_gq2_golden_f64.h5 ${M}_transfer_golden_f64.fits ${M}_transfer_gq2map_golden_f64.fits
} 2>&1 | tee $LOG
