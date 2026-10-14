#!/usr/bin/env bash
# Make the transfer gate's goldens: the products of the WORKSPACE implementations (the "before" of moving the pixel
# weights and the post-solve gauge into the library) on the gate fixture -- D3 Ch9 (NumCh17, NumCol1) on the 300-frame
# qr2 set, the field SPHEREx_nep_qr2_det3_6p2arcsec. Steps: before_solve.py (weighted solve, no clip) -> before_mosaic.py
# (weighted coadd of the UNGAUGED cal: mean, std, 2-sigma clipped mean) -> the two gauges, each fitted to the ungauged
# cal with the exact geometry and put on the map by gauge_map.py: apply_gauge.py --no-quadratic (the plane, the
# production gauge since 2026-10-14) and apply_gauge_q.py HDEG=2 (plane + quadratic). Each product is kept as a golden
# (*_transfer*_golden_f64*); the library must reproduce every one byte for byte (python_gates.py transfer: the
# production recipe; transfer/quad_check.py: the quadratic, sc.Gauge(degree=2), on the golden ungauged cal and map).
# Run from a COMMITTED tree (the log records the commit); an existing golden is replaced only with FORCE=1.
# GAUGE_ONLY=1 makes only the gauged goldens still missing (all of them with FORCE=1) from the existing ungauged ones:
# no solve, no coadd (the gauge scripts use no selfcal code, so the tree's library does not matter there).
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
missing() { [ ! -e $1 ] || [ "${FORCE:-}" = 1 ]; }
plane() {   # the plane gauge of the ungauged cal $1 -> gauged cal $3; on the map: mosaic $2 -> $4
  $PY $T/apply_gauge.py $1 --frames-dir $FX --no-quadratic --out $3; echo "[goldens] plane gauge exit $? $(date)"
  $PY $T/gauge_map.py $1 $3 $2 $4; echo "[goldens] plane on the map exit $? $(date)"
}
quad() {    # the same with the quadratic gauge
  HDEG=2 $PY $T/apply_gauge_q.py $1 --frames-dir $FX --out $3; echo "[goldens] quad gauge exit $? $(date)"
  $PY $T/gauge_map.py $1 $3 $2 $4; echo "[goldens] quad on the map exit $? $(date)"
}
if [ "${GAUGE_ONLY:-}" = 1 ]; then
  for g in ${C}_transfer_golden_f64.h5 ${M}_transfer_golden_f64.fits; do
    [ -e $g ] || { echo "make_transfer_goldens: GAUGE_ONLY needs the ungauged golden $g" >&2; exit 1; }
  done
else
  for g in ${C}_transfer_golden_f64.h5 ${C}_transfer_plane_golden_f64.h5 ${C}_transfer_gq2_golden_f64.h5 \
           ${M}_transfer_golden_f64.fits ${M}_transfer_planemap_golden_f64.fits ${M}_transfer_gq2map_golden_f64.fits; do
    if [ -e $g ] && [ "${FORCE:-}" != 1 ]; then echo "make_transfer_goldens: $g exists (FORCE=1 to replace)" >&2; exit 1; fi
  done
fi
{
  echo "[goldens] tree $R @ $(git rev-parse HEAD) ($(git branch --show-current)); fixture $FX; GAUGE_ONLY=${GAUGE_ONLY:-0}; $(date)"
  export PYTHONPATH=$R NUMPY_MADVISE_HUGEPAGE=0 DET=3 SCENE_FIELD=$FIELD SCENE_SCRATCH=$SCRATCH GAUGE_REF=$FIELD/ref.fits
  if [ "${GAUGE_ONLY:-}" = 1 ]; then
    U=${C}_transfer_golden_f64.h5 UM=${M}_transfer_golden_f64.fits
    if missing ${C}_transfer_plane_golden_f64.h5 || missing ${M}_transfer_planemap_golden_f64.fits; then
      plane $U $UM ${C}_transfer_plane_golden_f64.h5 ${M}_transfer_planemap_golden_f64.fits
    fi
    if missing ${C}_transfer_gq2_golden_f64.h5 || missing ${M}_transfer_gq2map_golden_f64.fits; then
      quad $U $UM ${C}_transfer_gq2_golden_f64.h5 ${M}_transfer_gq2map_golden_f64.fits
    fi
  else
    B=${C}_transfer_before_py BM=${M}_transfer_before_py
    rm -f $B.h5 $B.h5.json ${B}_plane.h5 ${B}_gq2.h5 $BM.fits $BM.fits.json ${BM}_planemap.fits ${BM}_gq2map.fits
    $PY $T/before_solve.py $FX transfer_before_py; echo "[goldens] solve exit $? $(date)"
    $PY $T/before_mosaic.py $FX $B.h5 transfer_before_py; echo "[goldens] coadd exit $? $(date)"
    plane $B.h5 $BM.fits ${B}_plane.h5 ${BM}_planemap.fits
    quad $B.h5 $BM.fits ${B}_gq2.h5 ${BM}_gq2map.fits
    cp $B.h5 ${C}_transfer_golden_f64.h5 && cp $B.h5.json ${C}_transfer_golden_f64.h5.json
    cp $BM.fits ${M}_transfer_golden_f64.fits && cp $BM.fits.json ${M}_transfer_golden_f64.fits.json
    cp ${B}_plane.h5 ${C}_transfer_plane_golden_f64.h5 && cp ${BM}_planemap.fits ${M}_transfer_planemap_golden_f64.fits
    cp ${B}_gq2.h5 ${C}_transfer_gq2_golden_f64.h5 && cp ${BM}_gq2map.fits ${M}_transfer_gq2map_golden_f64.fits
  fi
  echo "[goldens] done $(date)"
  ls -la ${C}_transfer_golden_f64.h5 ${C}_transfer_plane_golden_f64.h5 ${C}_transfer_gq2_golden_f64.h5 \
         ${M}_transfer_golden_f64.fits ${M}_transfer_planemap_golden_f64.fits ${M}_transfer_gq2map_golden_f64.fits
} 2>&1 | tee $LOG
