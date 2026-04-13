import sys
import os
import shutil
import time
import gc
from functools import partial
import numpy as np
import argparse # Added argparse for command line arguments

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from SelfCal import PipelineWrapper
from SelfCal.MakeMap import encode_x
from SelfCal.SPHERExUtility import load_calibration, load_lvf_params, compute_column_adjacency, \
compute_subchannel_adjacency, compute_offsets_guess, \
make_stripped_chunk_map, make_stripped_chunk_valid_mask, make_spherex_stripped_offset_map, fast_vertical_dist


def prepare_detector_inputs(frame_setting, mosaic_setting_oversample):
    detector = frame_setting['Detector']
    num_subchannels = frame_setting['NumSub']
    num_channels = frame_setting['NumCh']
    num_columns = frame_setting['NumCol']
    
    lvf_filename = f'lvf_params_D{detector}.npy'
    lvf_params = load_lvf_params(lvf_filename)

    det_BC, det_BW = load_calibration(band=detector, calibration_dir='/home/thomasli/spherex/SPHEREx_Spectral_Calibration')
    grid_chunk_map, _, _, _ = make_stripped_chunk_map(detector, num_subchannels=num_subchannels, num_channels=num_channels, num_columns=num_columns,
                                                    oversample_factor=mosaic_setting_oversample, lvf_params=lvf_params)
    det_chunk_map, _, r_edges, x_edges = make_stripped_chunk_map(detector, num_subchannels=num_subchannels, num_channels=num_channels, num_columns=num_columns,
                                            oversample_factor=1, lvf_params=lvf_params)
    
    adj_info_column = compute_column_adjacency(det_chunk_map, num_columns)
    adj_info = adj_info_column
        
    return {
        'lvf_params': lvf_params,
        'det_BC': det_BC,
        'det_BW': det_BW,
        'grid_chunk_map': grid_chunk_map,
        'det_chunk_map': det_chunk_map,
        'r_edges': r_edges,
        'x_edges': x_edges,
        'adj_info': adj_info
    }


def prepare_channel_inputs(ch, frame_setting, det_chunk_map, grid_chunk_map):
    num_subchannels = frame_setting['NumSub']
    num_channels = frame_setting['NumCh']
    num_columns = frame_setting['NumCol']
    
    chunk_valid_mask_padded = make_stripped_chunk_valid_mask(ch=ch, num_subchannels=num_subchannels, num_channels=num_channels, 
                                    num_columns=num_columns, subchannel_padding=1)
    chunk_valid_mask = make_stripped_chunk_valid_mask(ch=ch, num_subchannels=num_subchannels, num_channels=num_channels, 
                                    num_columns=num_columns, subchannel_padding=0)

    # Pre-calculate weights safely
    det_valid_mask = chunk_valid_mask_padded[det_chunk_map]
    det_valid_weight = fast_vertical_dist(det_valid_mask)
    if np.max(det_valid_weight) > 0:
        det_valid_weight /= np.max(det_valid_weight) 

    grid_valid_mask = chunk_valid_mask_padded[grid_chunk_map]
    grid_valid_weight = fast_vertical_dist(grid_valid_mask)
    if np.max(grid_valid_weight) > 0:
        grid_valid_weight /= np.max(grid_valid_weight) 

    return {
        'chunk_valid_mask_padded': chunk_valid_mask_padded,
        'chunk_valid_mask': chunk_valid_mask,
        'det_valid_mask': det_valid_mask,
        'grid_valid_mask': grid_valid_mask,
        'det_valid_weight': det_valid_weight,
        'grid_valid_weight': grid_valid_weight
    }

if __name__ == "__main__":
    # ----------------------------- Argument Parsing -----------------------------
    parser = argparse.ArgumentParser(description="Run SelfCal Synthetic Pipeline")
    parser.add_argument('--output_dir', type=str, default='/data3/caoye/selfcal/outputs', help='Path to output directory')
    parser.add_argument('--run_name', type=str, default='nep_det2_6p2arcsec', help='Name of the run')
    args = parser.parse_args()

    # ----------------------------- Start of Settings -----------------------------
    frame_setting = {
        'Detector': 2,
        'NumSub': 10,
        'NumCh': 17,
        'NumCol': 1,
    }

    selfcal_config = PipelineWrapper.PipelineConfig(
        output_dir=args.output_dir, # Updated to use arg
        run_name=args.run_name,     # Updated to use arg
        resolution_arcsec=6.2
    )

    calibration_kwargs = {
        'apply_mask': True,
        'apply_weight': False,
        'outlier_thresh': 2.0,
        'ignore_list': [],
        'batch_size': 40,
        'offset_regularization': False,
        'reg_weight': 10.0,
        'weighted_damping': True,
        'damp_weight': 0.1,
        'max_workers': 30,
        'postprocess_func': None,
    }

    lsqr_kwargs = {
        'atol': 1e-06,
        'btol': 1e-06,
        'damp': 1e-3,
        'iter_lim': 20,
        'precondition': True
    }

    mosaic_kwargs = {
        'apply_mask': True,
        'apply_weight': False,
        'make_std_map': True,
        'apply_sigma_clipping': True,
        'sigma': 1.0,
        'ignore_list': [21],
        'cache_batch_size': 40,
        'coadd_batch_size': 100,
        'cache_intermediate': True,
        'max_workers': 30
    }
    
    mosaic_oversample_factor = 2

    CACHE_DIR = '/home/thomasli/spherex/selfcal/cache/'
    FILE_SUFFIX = f'fast'

    # Channels to process
    ch = [10]
    # ----------------------------- End of Settings -----------------------------

    frame_setting_str = '_'.join([f'{key}{value}' for key, value in frame_setting.items()])
    
    # 1. Prepare overarching detector inputs
    detector_inputs = prepare_detector_inputs(frame_setting, mosaic_oversample_factor)
    
    # 2. Iterate through channels
    job_name = f'Ch{ch[0]}'
    t0 = time.time()
    print(f"Processing channel {job_name} for detector {frame_setting['Detector']}...")

    job_tag = f'{frame_setting_str}_{job_name}{FILE_SUFFIX}'
    cal_file = f'cal_{job_tag}.h5'
    mos_file = f"mosaic_{job_tag}.fits"
    cache_dir = f'{CACHE_DIR}cache_{job_tag}'

    # Prepare specific inputs for this channel
    channel_inputs = prepare_channel_inputs(ch, frame_setting, detector_inputs['det_chunk_map'], detector_inputs['grid_chunk_map'])
        
    # ----------------------------- Calibration -----------------------------
    cc = PipelineWrapper.Calibrator(selfcal_config)
    cc.setup_lsqr(
        chunk_map=detector_inputs['det_chunk_map'],
        grid_valid_weight=channel_inputs['det_valid_mask'],
        oversample_factor=1,
        adj_info=detector_inputs['adj_info'],
        **calibration_kwargs
    )
    offset = compute_offsets_guess(reproj_list=cc.reproj_list, det_chunk_map=detector_inputs['det_chunk_map'])
    skymap = np.zeros(cc.ref_shape)
    x0 = encode_x(skymap, offset)
    cc.apply_lsqr(x0=x0, **lsqr_kwargs)
    cal_path = cc.save_calibration(cal_dir=, cal_file=cal_file)

    # ----------------------------- Mosaicking -----------------------------
    partial_make_offset_map = partial(make_spherex_stripped_offset_map,
                                    chunk_valid_mask=channel_inputs['chunk_valid_mask'], 
                                    lvf_params=detector_inputs['lvf_params'], 
                                    r_edges=detector_inputs['r_edges'], 
                                    x_edges=detector_inputs['x_edges'], 
                                    tot_subchannels=frame_setting['NumSub']*frame_setting['NumCh']+2, 
                                    num_columns=frame_setting['NumCol'],
                                    fill_invalid=True)
    
    mm = PipelineWrapper.Mosaicker(selfcal_config)
    mm.load_calibration(cal_path=cal_path)

    maps = mm.make_mosaic(
        chunk_map=detector_inputs['grid_chunk_map'],
        grid_valid_weight=channel_inputs['grid_valid_weight'],
        oversample_factor=mosaic_oversample_factor,
        det_offset_func=partial_make_offset_map,
        cache_dir=cache_dir,
        **mosaic_kwargs
    )

    mm.save_mosaic(mos_file=mos_file, overwrite=True)
         
    # Clean up
    del cc, mm, maps
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    gc.collect()
        
    print(f"Finished channel {job_name} for detector {frame_setting['Detector']} in {time.time() - t0:.2f} seconds.")
    print("-" * 50 + "\n")