"""Fixed-subset mosaic generator for the perf/algo-optimizations regression gate.

Builds the D3 Ch17 NumCol=3 mosaic (cache + mean + std + sigma-clip + wav) on the
same fixed 300-frame subset used by regress_cal.py, against a pre-generated cal.
Used to gate the coadd-fusion opts (C/D): generate a baseline mosaic on the
pre-change code, then a candidate on the changed code, and compare the
STD_MAP / SC_MEAN_MAP / MEAN_MAP / WAV_* HDUs within float32 ε with
compare_mosaic.py.

Usage:
    python regress_mosaic.py --suffix _mos_baseline --n-frames 300
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import glob as glob_module
import shutil
import time
from functools import partial

import numpy as np

from SelfCal import PipelineWrapper
from SelfCal.MakeMap import set_hdd_io_limit
from SelfCal.SPHERExUtility import make_spherex_stripped_offset_map
from SelfCal.SPHERExAppendWav import wav_coadd
from run_cal_baseline_test import prepare_detector_inputs, prepare_channel_inputs

FRAME_SETTING = {'Detector': 3, 'NumSub': 10, 'NumCh': 34, 'NumCol': 3}
MOSAIC_OVERSAMPLE = 2
NVME_DIR = '/home/thomasli/selfcal-project/selfcal/cache/reproj_nvme_SPHEREx_nep_qr2_det3_6p2arcsec'
CACHE_ROOT = '/home/thomasli/selfcal-project/selfcal/cache'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--suffix', required=True)
    ap.add_argument('--cal-suffix', default='_gate_baseline')
    ap.add_argument('--n-frames', type=int, default=300)
    ap.add_argument('--max-workers', type=int, default=48)
    ap.add_argument('--no-wav', action='store_true')
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

    frame_str = '_'.join(f'{k}{v}' for k, v in FRAME_SETTING.items())
    cal_file = f'cal_{frame_str}_Ch17{args.cal_suffix}.h5'
    cal_path = os.path.join(cfg.cal_dir, cal_file)
    assert os.path.exists(cal_path), f"missing cal: {cal_path} (run regress_cal.py first)"

    mm = PipelineWrapper.Mosaicker(cfg, reproj_dir=NVME_DIR)
    mm.load_calibration(cal_path=cal_path)
    files = sorted(glob_module.glob(os.path.join(NVME_DIR, '*.h5')))[:args.n_frames]
    mm.reproj_list = files
    print(f"n_frames={len(mm.reproj_list)}  n_offsets={mm.offsets[0].shape}")

    cache_dir = os.path.join(CACHE_ROOT, f'cache_mosregress{args.suffix}')
    partial_make_offset_map = partial(
        make_spherex_stripped_offset_map,
        chunk_valid_mask=ch_inputs['chunk_valid_mask'],
        lvf_params=det_inputs['lvf_params'],
        r_edges=det_inputs['r_edges'],
        x_edges=det_inputs['x_edges'],
        tot_subchannels=FRAME_SETTING['NumSub'] * FRAME_SETTING['NumCh'] + 2,
        num_columns=FRAME_SETTING['NumCol'],
        fill_invalid=True,
    )

    t0 = time.time()
    maps = mm.make_mosaic(
        chunk_maps=[det_inputs['grid_chunk_map']],
        grid_valid_weight=ch_inputs['grid_valid_weight'],
        oversample_factor=MOSAIC_OVERSAMPLE,
        det_offset_funcs=[partial_make_offset_map],
        cache_dir=cache_dir,
        apply_mask=True, apply_weight=False,
        make_std_map=True, apply_sigma_clipping=True, sigma=2.0,
        ignore_list=[21],
        cache_batch_size=50, coadd_batch_size=50,
        cache_intermediate=True, max_workers=args.max_workers,
    )
    print(f"make_mosaic: {time.time()-t0:.2f} s")

    if not args.no_wav:
        wav_mean, wav_std = wav_coadd(
            det_inputs['det_BC'], det_inputs['det_BW'],
            mean_map=maps['mean_map']['data'], std_map=maps['std_map']['data'],
            reproj_list=mm.reproj_list, cache_list=mm.cached_list,
            ref_shape=maps['mean_map']['data'].shape, sigma=2.0,
            batch_size=50, max_workers=args.max_workers,
        )
        mm.append_maps({
            'wav_mean_map': {'data': wav_mean, 'unit': 'um'},
            'wav_std_map': {'data': wav_std, 'unit': 'um'},
        })

    mos_file = f'mosaic_{frame_str}_Ch17{args.suffix}.fits'
    mos_path = mm.save_mosaic(mos_file=mos_file, overwrite=True)
    print(f"wrote {mos_path}  (total {time.time()-t0:.2f} s)")

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


if __name__ == '__main__':
    main()
