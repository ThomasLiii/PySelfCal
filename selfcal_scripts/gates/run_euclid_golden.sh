#!/usr/bin/env bash
# Euclid golden from the SCRIPT path (euclid_pipeline.py of the selfcal-euclid workspace) run against
# THIS tree's library: 3 Y-band exposures x 16 detectors at 1.5", the frozen recipe's model at small
# scale (detector-fixed N=40 grid map + per-frame column/row stripes damped 0.3 + scalar), float64
# solve, iter 30, full mosaic. Products: /mnt/md124/thomasli/selfcal/outputs/EDFN_Y_1p5arcsec_unifygolden/
set -u
R=/home/thomasli/selfcal-project/selfcal-memopt
PY=/home/thomasli/anaconda3/envs/selfcal-memopt/bin/python
E=/home/thomasli/selfcal-project/selfcal-euclid/workspace/euclid-nep-mosaic
cd $E
echo "[euclid golden] tree=$(git -C $R rev-parse --short HEAD) start $(date)"
PYTHONPATH=$R PYTHONUNBUFFERED=1 $PY euclid_pipeline.py --band Y --res 1.5 --subset 3 --n-side 40 \
  --perframe-stripes --perframe-damp 0.3 --reg-weight 0.1 --damp-weight 0.1 --outlier-thresh 5.0 \
  --iter-lim 30 --run-name EDFN_Y_1p5arcsec_unifygolden --tag golden --steps reproject,cal,mosaic \
  --max-workers-reproject 16 --max-workers-cal 8 --max-workers-mosaic 8 --n-threads 8 --coadd-sigma 2.0 \
  --batch-size 8 --cache-batch-size 8 --coadd-batch-size 8
echo "[euclid golden] rc=$? $(date)"
