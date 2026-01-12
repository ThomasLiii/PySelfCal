import sys
import os
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
import time
from concurrent.futures import ProcessPoolExecutor

from SelfCal import PipelineWrapper
from SelfCal.SPHERExUtility import make_fiducial_chunk_map, make_fiducial_chunk_mask, \
load_calibration, make_spherex_offset_map, compute_offsets_guess
from SelfCal.SPHERExAppendWav import wav_coadd
from SelfCal.MapHelper import bit_to_bool
from SelfCal.MakeMap import load_reproj_file

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
FILE_SUFFIX = 'naiveoffset'
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

def process_exp(exp_path):
    with fits.open(exp_path) as hdul:
        sci_data = hdul[1].data
        mask_data = hdul[2].data
    bool_mask = bit_to_bool(mask_data, ignore_list=[], invert=True)
    # Note: chunk_map must be accessible within this scope
    chunk_cube = np.array([det_chunk_map == i for i in range(1, 171)])
    mask_cube = chunk_cube & bool_mask[None, :, :]
    return np.array([sci_data[mask].mean() for mask in mask_cube])

reproj_list = sorted(glob.glob(f'/data3/thomasli/selfcal/outputs/nep_det{DETECTOR}_6p2arcsec/reprojected/*'))
exp_list = []
for reproj_file in tqdm(reproj_list):
    exp_list.append(load_reproj_file(reproj_file, fields=['file_path'])['file_path'])
with ProcessPoolExecutor(max_workers=20) as executor:
    full_mean_vals = list(tqdm(executor.map(process_exp, exp_list), total=len(exp_list)))
full_mean_vals = np.array(full_mean_vals)
full_mean_vals_padded = np.pad(full_mean_vals, ((0, 0), (1, 1)), mode='constant', constant_values=0)

for ch in chs:
    print(f"Processing channel {ch} for detector {DETECTOR}")
    t0 = time.time()
    chunk_valid_mask = make_fiducial_chunk_mask(ch, num_subchannels=NUM_SUBCHANNELS, num_channels=NUM_CHANNELS)
    det_valid_mask = chunk_valid_mask[chunk_map]


    mm = PipelineWrapper.Mosaicker(config)
    mm.O = full_mean_vals_padded

    partial_make_offset_map = partial(make_spherex_offset_map, chunk_valid_mask=chunk_valid_mask, lvf_params=lvf_params)
    sc_sigma = 1.0
    maps = mm.make_mosaic(
        apply_mask=True, 
        apply_weight=False, 
        chunk_map=chunk_map, 
        det_valid_mask=det_valid_mask, 
        max_workers=80,
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
