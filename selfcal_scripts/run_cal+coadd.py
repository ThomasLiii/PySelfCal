import sys
import os
import time

from SelfCal import PipelineWrapper
from SelfCal.SPHERExUtility import make_fiducial_chunk_map, make_fiducial_chunk_mask, \
load_calibration, make_spherex_offset_map, compute_offsets_guess, load_lvf_params, compute_vertical_strip_adjacency, \
make_stripped_chunk_map, make_stripped_chunk_valid_mask
from SelfCal.SPHERExAppendWav import wav_coadd

from astropy.io import fits
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LogNorm
mpl.rcParams['figure.dpi'] = 200
# Import LogNorm
from tqdm import tqdm
import gc
from functools import partial

if __name__ == "__main__":
    DETECTOR = 4
    OVERSAMPLE_FACTOR = 2
    NUM_SUBCHANNELS = 10
    NUM_CHANNELS = 17
    NUM_COLUMNS = 5

    setting_tags = {
                'NumSub': NUM_SUBCHANNELS,
                'NumCh': NUM_CHANNELS,
                'NumCol': NUM_COLUMNS,
    }
    setting_str = '_'.join([f'{key}{value}' for key, value in setting_tags.items()])

    FILE_SUFFIX = f''
    FILE_PREFIX = f''

    config = {}
    config['output_dir'] = '/mnt/md124/thomasli/selfcal/outputs/'
    config['run_name'] = f'SPHEREx_nep_qr2_det{DETECTOR}_6p2arcsec'
    config['resolution_arcsec'] = 6.2

    lvf_filename = f'lvf_params_D{DETECTOR}.npy'
    lvf_params = load_lvf_params(lvf_filename)

    grid_chunk_map, _ = make_stripped_chunk_map(DETECTOR, num_subchannels=NUM_SUBCHANNELS, num_channels=NUM_CHANNELS, num_columns=NUM_COLUMNS,
                                                    oversample_factor=OVERSAMPLE_FACTOR, lvf_params=lvf_params)
    det_chunk_map, _ = make_stripped_chunk_map(DETECTOR, num_subchannels=NUM_SUBCHANNELS, num_channels=NUM_CHANNELS, num_columns=NUM_COLUMNS,
                                            oversample_factor=1, lvf_params=lvf_params)
    
    ch = [12]

    chunk_valid_mask_padded = make_stripped_chunk_valid_mask(ch, num_subchannels=NUM_SUBCHANNELS, num_channels=NUM_CHANNELS, 
                                    num_columns=NUM_COLUMNS, subchannel_padding=1)
    chunk_valid_mask = make_stripped_chunk_valid_mask(ch, num_subchannels=NUM_SUBCHANNELS, num_channels=NUM_CHANNELS, 
                                    num_columns=NUM_COLUMNS, subchannel_padding=0)

    adj_info = compute_vertical_strip_adjacency(det_chunk_map, NUM_COLUMNS)

    cc = PipelineWrapper.Calibrator(config)
    cc.setup_lsqr(
        apply_mask=True, 
        apply_weight=False,
        chunk_map=det_chunk_map, 
        det_valid_mask=chunk_valid_mask_padded[det_chunk_map], 
        max_workers=50, 
        outlier_thresh=2.0,
        ignore_list=[],
        oversample_factor=1,
        batch_size=30,
        reg_weight=10.0,
        adj_info=adj_info,
        weighted_damping=True,
        damp_weight=1
        )
    
    cal_path = cc.save_calibration(cal_file=f'cal{FILE_PREFIX}_D{DETECTOR}_Ch{"-".join(map(str, ch))}_{setting_str}{FILE_SUFFIX}.h5')