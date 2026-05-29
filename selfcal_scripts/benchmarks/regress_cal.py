"""Fixed-subset cal generator for the perf/algo-optimizations regression gate.

Runs the real Calibrator.setup_lsqr + apply_lsqr + save_calibration on a fixed,
sorted subset of the D3 Ch17 NumCol=3 NVMe-staged reproj files, with the exact
benchmark_d3_ch17_numcol3 config. Save BEFORE a change as the baseline, then
after each change diff with diff_cal_h5.py — opt A/B MUST be byte-equal.

Usage:
    python regress_cal.py --suffix _gate_baseline --n-frames 300
    python regress_cal.py --suffix _gate_optA     --n-frames 300
    python diff_cal_h5.py <cal_dir>/cal_..._gate_baseline.h5 \
                          <cal_dir>/cal_..._gate_optA.h5
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import glob as glob_module
import time

import numpy as np

from SelfCal import PipelineWrapper
from SelfCal.MakeMap import set_hdd_io_limit
from SelfCal.solution import compute_x0_scalar_only
from run_cal_baseline_test import prepare_detector_inputs, prepare_channel_inputs

FRAME_SETTING = {'Detector': 3, 'NumSub': 10, 'NumCh': 34, 'NumCol': 3}
MOSAIC_OVERSAMPLE = 2
NVME_DIR = '/home/thomasli/selfcal-project/selfcal/cache/reproj_nvme_SPHEREx_nep_qr2_det3_6p2arcsec'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', required=True)
    ap.add_argument('--n-frames', type=int, default=300)
    ap.add_argument('--max-workers', type=int, default=48)
    ap.add_argument('--batch-size', type=int, default=50)
    args = ap.parse_args()

    cfg = PipelineWrapper.PipelineConfig(
        output_dir='/mnt/md124/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_nep_qr2_det{FRAME_SETTING["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2,
    )

    set_hdd_io_limit(None)
    det_inputs = prepare_detector_inputs(FRAME_SETTING, MOSAIC_OVERSAMPLE)
    ch_inputs = prepare_channel_inputs(
        [17], FRAME_SETTING, det_inputs['det_chunk_map'], det_inputs['grid_chunk_map'])

    cc = PipelineWrapper.Calibrator(cfg, reproj_dir=NVME_DIR)
    files = sorted(glob_module.glob(os.path.join(NVME_DIR, '*.h5')))[:args.n_frames]
    cc.reproj_list = files
    num_frames = len(cc.reproj_list)
    print(f"n_frames={num_frames}  num_chunks={int(det_inputs['det_chunk_map'].max())+1}")

    frame_str = '_'.join(f'{k}{v}' for k, v in FRAME_SETTING.items())
    cal_file = f'cal_{frame_str}_Ch17{args.suffix}.h5'

    t0 = time.time()
    cc.setup_lsqr(
        chunk_maps=[det_inputs['det_chunk_map']],
        grid_valid_weight=ch_inputs['det_valid_mask_padded'],
        oversample_factor=1,
        adj_infos=[det_inputs['adj_info']],
        mean_offsets_list=[np.zeros(num_frames)],
        use_per_frame_scalar=True,
        apply_mask=True, apply_weight=False, outlier_thresh=5.0, ignore_list=[],
        batch_size=args.batch_size, offset_regularization=True, reg_weights=[0.1],
        weighted_damping=True, damp_weight=0.1, max_workers=args.max_workers,
    )
    t_setup = time.time() - t0
    print(f"setup_lsqr: {t_setup:.2f} s")

    x0 = compute_x0_scalar_only(
        cc.A, cc.b, cc.ref_shape,
        scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
    )
    cc.apply_lsqr(x0=x0, atol=1e-06, btol=1e-06, damp=0, iter_lim=50,
                  precondition=True, solver='lsqr', use_float32=True, n_threads=args.max_workers)

    # Save with original HDD paths (keeps cal valid; reproj_list dataset identical
    # across baseline/candidate so diff_cal_h5 compares apples to apples).
    nvme_list = cc.reproj_list
    cc.reproj_list = [os.path.join(cfg.reproj_dir, os.path.basename(f)) for f in nvme_list]
    cal_path = cc.save_calibration(cal_file=cal_file)
    print(f"wrote {cal_path}  (total {time.time()-t0:.2f} s)")


if __name__ == '__main__':
    main()
