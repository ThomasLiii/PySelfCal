import sys
import os
import shutil
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

DETECTOR = 4
OVERSAMPLE_FACTOR = 2
NUM_SUBCHANNELS = 10
NUM_CHANNELS = 34
FILE_SUFFIX = f'PAH_subchannel249-260_unnormoffsets'
FILE_PREFIX = f'_34channels'

config = {}
config['output_dir'] = '/mnt/md124/thomasli/selfcal/outputs/'
config['run_name'] = f'nep_det{DETECTOR}_6p2arcsec'
config['resolution_arcsec'] = 6.2


det_BC, det_BW = load_calibration(band=DETECTOR, calibration_dir='/home/thomasli/spherex/SPHEREx_Spectral_Calibration')
chunk_map, lvf_params = make_fiducial_chunk_map(DETECTOR, det_BC, num_subchannels=NUM_SUBCHANNELS, num_channels=NUM_CHANNELS, 
                                                oversample_factor=OVERSAMPLE_FACTOR)
det_chunk_map, _ = make_fiducial_chunk_map(DETECTOR, det_BC, num_subchannels=NUM_SUBCHANNELS, num_channels=NUM_CHANNELS, 
                                           oversample_factor=1, lvf_params=lvf_params)

chunk_valid_mask = np.zeros(NUM_CHANNELS * NUM_SUBCHANNELS + 2)
chunk_valid_mask[249:260] = 1

t0 = time.time()

chunk_valid_mask_1subpad = chunk_valid_mask.copy()
where_valid = np.where(chunk_valid_mask_1subpad)
chunk_valid_mask_1subpad[np.min(where_valid)-1:np.max(where_valid)+2] = 1
# Erode back to original
det_valid_mask = chunk_valid_mask[chunk_map]
det_valid_mask_1subpad = chunk_valid_mask_1subpad[chunk_map]

cal_path = f'/mnt/md124/thomasli/selfcal/outputs/nep_det4_6p2arcsec/calibration/cal_34channels_D4_3p4um_subchannel249-260.h5'

mm = PipelineWrapper.Mosaicker(config)
mm.load_calibration(cal_path=cal_path)
partial_make_offset_map = partial(make_spherex_offset_map, chunk_valid_mask=chunk_valid_mask, lvf_params=lvf_params)
sc_sigma = 1.0
cache_dir = '/home/thomasli/spherex/selfcal/cache/' + f'cache{FILE_PREFIX}_D{DETECTOR}_{FILE_SUFFIX}/'
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
    cache_dir=cache_dir,
    cache_intermediate=True,
    det_aux=None,
    normalize_offset=False
)

wav_mean, wav_std = wav_coadd(det_BC, det_BW, mean_map=maps['mean_map']['data'], std_map=maps['std_map']['data'], 
                                reproj_list=mm.reproj_list, cache_list=mm.cached_list, ref_shape=maps['mean_map']['data'].shape, 
                                sigma=sc_sigma, batch_size=40, max_workers=40)    

wav_mean_maps = {'data': wav_mean, 'unit': 'um'}
wav_std_maps = {'data': wav_std, 'unit': 'um'}
mm.append_maps({'wav_mean_map': wav_mean_maps, 'wav_std_map': wav_std_maps})

mm.save_mosaic(mos_file=f'mosaic{FILE_PREFIX}_D{DETECTOR}_{FILE_SUFFIX}.fits', overwrite=True)

# Clear memory
del mm, maps
gc.collect()

if os.path.isdir(cache_dir):
    try:
        shutil.rmtree(cache_dir)
        print(f"Directory '{cache_dir}' and all its contents have been removed.")
    except OSError as e:
        print(f"Error: {cache_dir} : {e.strerror}")

t1 = time.time()
print(f"Finished in {t1 - t0:.2f} seconds")
