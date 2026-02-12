import sys
import os
import shutil
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
import time
import time
from astropy.io import fits
import numpy as np
import glob
import gc
from functools import partial
import matplotlib.pyplot as plt

import sys
import os
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
from matplotlib.colors import LogNorm
mpl.rcParams['figure.dpi'] = 200
# Import LogNorm
from tqdm import tqdm
import gc
from functools import partial

def make_vertical_band_maps(sub_channel_map, num_vertical_bands):
    vertchunk_map = np.zeros_like(sub_channel_map)
    for band in range(num_vertical_bands):
        width = vertchunk_map.shape[1] // num_vertical_bands
        vertchunk_map[:, band*width:(band+1)*width] = band
    return vertchunk_map

def make_chunk_valid_mask(subchannel_valid_mask, num_vertical_bands):
    chunk_valid_mask = np.zeros(len(subchannel_valid_mask)*num_vertical_bands, dtype=subchannel_valid_mask.dtype)
    for band in range(num_vertical_bands):
        chunk_valid_mask[band::num_vertical_bands] = subchannel_valid_mask
    return chunk_valid_mask

def compute_chunk_adjacency(chunk_map, reg_axis='both'):
    """
    Computes the adjacency list for a given chunk map.
    
    Parameters
    ----------
    chunk_map : np.ndarray
        2D array where each pixel value is the chunk ID. -1 indicates ignored pixels.
    reg_axis : str, optional
        'both': Horizontal and Vertical neighbors.
        'x': Horizontal only.
        'y': Vertical only.
        
    Returns
    -------
    tuple or None
        (neighbors_i, neighbors_j) arrays of shape (N_pairs,), or None if no pairs found.
    """
    if chunk_map is None:
        return None

    print(f"Pre-computing adjacency matrix (Axis: {reg_axis})...")
    
    all_i_list = []
    all_j_list = []

    # 1. Horizontal neighbors (x-axis)
    if reg_axis in ['both', 'x', 'horizontal']:
        # Compare [:, :-1] with [:, 1:]
        h_diff = (chunk_map[:, :-1] != -1) & \
                 (chunk_map[:, 1:] != -1) & \
                 (chunk_map[:, :-1] != chunk_map[:, 1:])
                 
        h_idx_i = chunk_map[:, :-1][h_diff]
        h_idx_j = chunk_map[:, 1:][h_diff]
        all_i_list.append(h_idx_i)
        all_j_list.append(h_idx_j)
    
    # 2. Vertical neighbors (y-axis)
    if reg_axis in ['both', 'y', 'vertical']:
        # Compare [:-1, :] with [1:, :]
        v_diff = (chunk_map[:-1, :] != -1) & \
                 (chunk_map[1:, :] != -1) & \
                 (chunk_map[:-1, :] != chunk_map[1:, :])
                 
        v_idx_i = chunk_map[:-1, :][v_diff]
        v_idx_j = chunk_map[1:, :][v_diff]
        all_i_list.append(v_idx_i)
        all_j_list.append(v_idx_j)
    
    # Combine and remove duplicates
    if all_i_list:
        all_i = np.concatenate(all_i_list)
        all_j = np.concatenate(all_j_list)
        
        # Ensure i < j to avoid double counting
        mask = all_i < all_j
        unique_pairs = np.unique(np.stack([all_i[mask], all_j[mask]], axis=1), axis=0)
        return (unique_pairs[:, 0], unique_pairs[:, 1])
    else:
        print("Warning: No adjacency pairs found.")
        return None
    
def compute_vertical_strip_adjacency(chunk_map, num_bands):
    """
    Generates adjacency pairs ONLY for vertical strip transitions, 
    ignoring spectral arc transitions.
    
    Parameters
    ----------
    chunk_map : np.ndarray
        The full ID map (Subchannel * N + Band)
    num_bands : int
        The NUM_VERTICAL_BANDS constant used to build the map.
    """
    print("Computing Vertical Strip Adjacency (Filtering Arcs)...")
    
    # 1. Get ALL horizontal transitions (Arc + Strip boundaries)
    # Compare pixel i with i+1
    mask = (chunk_map[:, :-1] != -1) & \
           (chunk_map[:, 1:] != -1) & \
           (chunk_map[:, :-1] != chunk_map[:, 1:])
           
    u = chunk_map[:, :-1][mask]
    v = chunk_map[:, 1:][mask]
    
    # 2. Decompose IDs back into (Subchannel, Band)
    # Formula: ID = Sub * N + Band
    sub_u = u // num_bands
    sub_v = v // num_bands
    
    # 3. FILTER: Only keep pairs that are in the SAME Subchannel
    # This rejects the boundaries where the arc changes.
    valid_pair_mask = (sub_u == sub_v)
    
    u_filtered = u[valid_pair_mask]
    v_filtered = v[valid_pair_mask]
    
    # 4. Remove duplicates
    # Sort pairs so (u,v) is same as (v,u) for unique checking
    pairs = np.sort(np.stack([u_filtered, v_filtered], axis=1), axis=1)
    unique_pairs = np.unique(pairs, axis=0)
    
    print(f"Found {len(unique_pairs)} vertical strip boundaries.")
    return unique_pairs[:, 0], unique_pairs[:, 1]

if __name__ == "__main__":
    DETECTOR = 1
    OVERSAMPLE_FACTOR = 2
    NUM_SUBCHANNELS = 10
    NUM_CHANNELS = 17
    NUM_VERTICAL_BANDS = 5
    FILE_SUFFIX = f'_{NUM_CHANNELS}channels_{NUM_VERTICAL_BANDS}verticalbands'
    FILE_PREFIX = f''

    config = {}
    config['output_dir'] = '/mnt/md124/thomasli/selfcal/outputs/'
    config['run_name'] = f'SPHEREx_nep_qr2_det{DETECTOR}_6p2arcsec'
    config['resolution_arcsec'] = 6.2

    det_BC, det_BW = load_calibration(band=DETECTOR, calibration_dir='/home/thomasli/spherex/SPHEREx_Spectral_Calibration')
    grid_subchannel_map, lvf_params = make_fiducial_chunk_map(DETECTOR, det_BC, num_subchannels=NUM_SUBCHANNELS, num_channels=NUM_CHANNELS, 
                                                    oversample_factor=OVERSAMPLE_FACTOR)
    det_subchannel_map, _ = make_fiducial_chunk_map(DETECTOR, det_BC, num_subchannels=NUM_SUBCHANNELS, num_channels=NUM_CHANNELS, 
                                            oversample_factor=1, lvf_params=lvf_params)
    
    ch = [10]

    grid_vertchunk_map = make_vertical_band_maps(grid_subchannel_map, NUM_VERTICAL_BANDS)
    det_vertchunk_map = make_vertical_band_maps(det_subchannel_map, NUM_VERTICAL_BANDS)

    grid_chunk_map = grid_subchannel_map * NUM_VERTICAL_BANDS + grid_vertchunk_map
    det_chunk_map = det_subchannel_map * NUM_VERTICAL_BANDS + det_vertchunk_map

    subchannel_valid_mask = make_fiducial_chunk_mask(ch, num_subchannels=NUM_SUBCHANNELS, num_channels=NUM_CHANNELS, padding=0)
    subchannel_valid_mask_padded = make_fiducial_chunk_mask(ch, num_subchannels=NUM_SUBCHANNELS, num_channels=NUM_CHANNELS, padding=1)

    chunk_valid_mask = make_chunk_valid_mask(subchannel_valid_mask, NUM_VERTICAL_BANDS)
    chunk_valid_mask_padded = make_chunk_valid_mask(subchannel_valid_mask_padded, NUM_VERTICAL_BANDS)

    det_valid_mask = chunk_valid_mask[det_chunk_map]
    grid_valid_mask = chunk_valid_mask[grid_chunk_map]
    det_valid_mask_padded = chunk_valid_mask_padded[det_chunk_map]
    grid_valid_mask_padded = chunk_valid_mask_padded[grid_chunk_map]

    # adj_info = compute_vertical_strip_adjacency(det_chunk_map, NUM_VERTICAL_BANDS)

    # cc = PipelineWrapper.Calibrator(config)
    # cc.setup_lsqr(
    #     apply_mask=True, 
    #     apply_weight=False,
    #     chunk_map=det_chunk_map, 
    #     det_valid_mask=det_valid_mask_padded, 
    #     max_workers=50, 
    #     outlier_thresh=10.0,
    #     ignore_list=[],
    #     oversample_factor=1,
    #     batch_size=30,
    #     reg_weight=1.0,
    #     adj_info=adj_info
    #     )
    
    # cc.apply_lsqr(x0=None, atol=1e-06, btol=1e-06, damp=1e-3, iter_lim=100, precondition=False)

    # cal_path = cc.save_calibration(cal_file=f'cal{FILE_PREFIX}_D{DETECTOR}_Ch{"-".join(map(str, ch))}{FILE_SUFFIX}.h5')

    cal_path = '/mnt/md124/thomasli/selfcal/outputs/SPHEREx_nep_qr2_det1_6p2arcsec/calibration/cal_D1_Ch10_17channels_5verticalbands.h5'
    mm = PipelineWrapper.Mosaicker(config)
    mm.load_calibration(cal_path=cal_path)
    # partial_make_offset_map = partial(make_spherex_offset_map, chunk_valid_mask=chunk_valid_mask, lvf_params=lvf_params)
    sc_sigma = 1.0
    maps = mm.make_mosaic(
        apply_mask=True, 
        apply_weight=False, 
        chunk_map=grid_chunk_map, 
        det_valid_mask=grid_valid_mask, 
        max_workers=20,
        make_std_map=True, 
        apply_sigma_clipping=True,  
        sigma=sc_sigma,
        ignore_list=[21],
        oversample_factor=OVERSAMPLE_FACTOR,
        det_offset_func=None,#partial_make_offset_map,
        cache_batch_size=20,
        coadd_batch_size=100,
        cache_dir='/home/thomasli/spherex/selfcal/cache',
        cache_intermediate=True,
        det_aux=None
    )

    # wav_mean, wav_std = wav_coadd(det_BC, det_BW, mean_map=maps['mean_map']['data'], std_map=maps['std_map']['data'], 
    #                               reproj_list=mm.reproj_list, cache_list=mm.cached_list, ref_shape=maps['mean_map']['data'].shape, 
    #                               sigma=sc_sigma, batch_size=40, max_workers=40)    

    # wav_mean_maps = {'data': wav_mean, 'unit': 'um'}
    # wav_std_maps = {'data': wav_std, 'unit': 'um'}
    # mm.append_maps({'wav_mean_map': wav_mean_maps, 'wav_std_map': wav_std_maps})

    mm.save_mosaic(mos_file=f'mosaic{FILE_PREFIX}_D{DETECTOR}_Ch{"-".join(map(str, ch))}{FILE_SUFFIX}.fits', overwrite=True)

    # # Clear memory
    # del mm, maps
    # gc.collect()
    # t1 = time.time()
    # print(f"Finished channel {ch} for detector {DETECTOR} in {t1 - t0:.2f} seconds")

