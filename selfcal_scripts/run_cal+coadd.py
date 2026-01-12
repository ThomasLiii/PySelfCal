import sys
import os
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
import time

from SelfCal import PipelineWrapper
from SelfCal.SPHERExUtility import make_fiducial_chunk_map, make_fiducial_chunk_mask, \
load_calibration, make_spherex_offset_map, compute_offsets_guess
from SelfCal.SPHERExAppendWav import wav_coadd

from astropy.io import fits
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
# Import LogNorm
from tqdm import tqdm
import gc
from functools import partial

DETECTOR = 1
OVERSAMPLE_FACTOR = 2
NUM_SUBCHANNELS = 10
NUM_CHANNELS = 17
FILE_SUFFIX = ''
FILE_PREFIX = f''

config = {}
config['output_dir'] = '/mnt/md124/thomasli/selfcal/outputs/'
config['run_name'] = f'nep_det{DETECTOR}_6p2arcsec'
config['resolution_arcsec'] = 6.2

# chs = [[19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32]]
chs = [[i] for i in range(1, 18)]

det_BC, det_BW = load_calibration(band=DETECTOR, calibration_dir='/home/thomasli/spherex/SPHEREx_Spectral_Calibration')
chunk_map, lvf_params = make_fiducial_chunk_map(DETECTOR, det_BC, num_subchannels=NUM_SUBCHANNELS, num_channels=NUM_CHANNELS, 
                                                oversample_factor=OVERSAMPLE_FACTOR)
det_chunk_map, _ = make_fiducial_chunk_map(DETECTOR, det_BC, num_subchannels=NUM_SUBCHANNELS, num_channels=NUM_CHANNELS, 
                                           oversample_factor=1, lvf_params=lvf_params)

for ch in chs:
    print(f"Processing channel {ch} for detector {DETECTOR}")
    t0 = time.time()
    chunk_valid_mask = make_fiducial_chunk_mask(ch, num_subchannels=NUM_SUBCHANNELS, num_channels=NUM_CHANNELS)

    chunk_valid_mask_1subpad = chunk_valid_mask.copy()
    where_valid = np.where(chunk_valid_mask_1subpad)
    chunk_valid_mask_1subpad[np.min(where_valid)-1:np.max(where_valid)+2] = 1
    # Erode back to original
    det_valid_mask = chunk_valid_mask[chunk_map]
    det_valid_mask_1subpad = chunk_valid_mask_1subpad[chunk_map]
    cc = PipelineWrapper.Calibrator(config)
    cc.setup_lsqr(
        apply_mask=True, 
        apply_weight=True, 
        chunk_map=chunk_map, 
        det_valid_mask=det_valid_mask_1subpad, 
        max_workers=40, 
        outlier_thresh=10.0,
        ignore_list=[],
        oversample_factor=OVERSAMPLE_FACTOR,
        batch_size=20
        )

    offset_guess = compute_offsets_guess(cc.reproj_list, det_chunk_map)
    x0 = np.hstack([np.zeros(np.prod(cc.ref_shape)), offset_guess.flatten()])
    cc.apply_lsqr(x0=x0, atol=1e-06, btol=1e-06, damp=1e-3, iter_lim=200)
    cal_path = cc.save_calibration(cal_file=f'cal{FILE_PREFIX}_D{DETECTOR}_Ch{"-".join(map(str, ch))}{FILE_SUFFIX}.h5')

    del cc
    gc.collect()

    mm = PipelineWrapper.Mosaicker(config)
    mm.load_calibration(cal_path=cal_path)
    partial_make_offset_map = partial(make_spherex_offset_map, chunk_valid_mask=chunk_valid_mask, lvf_params=lvf_params)
    sc_sigma = 1.0
    maps = mm.make_mosaic(
        apply_mask=True, 
        apply_weight=True, 
        chunk_map=chunk_map, 
        det_valid_mask=det_valid_mask, 
        max_workers=40,
        make_std_map=True, 
        apply_sigma_clipping=True,  
        sigma=sc_sigma,
        ignore_list=[21],
        oversample_factor=OVERSAMPLE_FACTOR,
        det_offset_func=partial_make_offset_map,
        cache_batch_size=20,
        coadd_batch_size=100,
        cache_dir='/home/thomasli/spherex/selfcal/cache',
        cache_intermediate=True,
        det_aux=None
    )

    wav_mean, wav_std = wav_coadd(det_BC, det_BW, mean_map=maps['mean_map']['data'], std_map=maps['std_map']['data'], 
                                  reproj_list=mm.reproj_list, cache_list=mm.cached_list, ref_shape=maps['mean_map']['data'].shape, 
                                  sigma=sc_sigma, batch_size=40, max_workers=40)    

    wav_mean_maps = {'data': wav_mean, 'unit': 'um'}
    wav_std_maps = {'data': wav_std, 'unit': 'um'}
    mm.append_maps({'wav_mean_map': wav_mean_maps, 'wav_std_map': wav_std_maps})

    mm.save_mosaic(mos_file=f'mosaic{FILE_PREFIX}_D{DETECTOR}_Ch{"-".join(map(str, ch))}{FILE_SUFFIX}.fits', overwrite=True)

    # Clear memory
    del mm, maps
    gc.collect()
    t1 = time.time()
    print(f"Finished channel {ch} for detector {DETECTOR} in {t1 - t0:.2f} seconds")
