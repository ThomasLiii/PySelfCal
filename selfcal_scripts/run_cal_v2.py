import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import shutil
import time
import gc
from functools import partial
import numpy as np
from threadpoolctl import threadpool_limits

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from SelfCal import PipelineWrapper
from SelfCal.MakeMap import encode_x
from SelfCal.SPHERExUtility import load_calibration, load_lvf_params, compute_column_adjacency, \
compute_subchannel_adjacency, compute_offsets_guess, \
make_stripped_chunk_map, make_stripped_chunk_valid_mask, make_spherex_stripped_offset_map, fast_vertical_dist
from SelfCal.SPHERExAppendWav import wav_coadd


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
    # adj_info_subchannel = compute_subchannel_adjacency(det_chunk_map, num_columns)
    
    # adj_info = (
    #     np.concatenate([adj_info_column[0], adj_info_subchannel[0]]),
    #     np.concatenate([adj_info_column[1], adj_info_subchannel[1]])
    # )
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
    
    if isinstance(ch, list) or isinstance(ch, np.ndarray):
        chunk_valid_mask_padded = make_stripped_chunk_valid_mask(ch=ch, num_subchannels=num_subchannels, num_channels=num_channels, 
                                        num_columns=num_columns, subchannel_padding=1)
        chunk_valid_mask = make_stripped_chunk_valid_mask(ch=ch, num_subchannels=num_subchannels, num_channels=num_channels, 
                                        num_columns=num_columns, subchannel_padding=0)
    elif isinstance(ch, str):
        if ch == 'Aromatic':
            subch = np.arange(225, 236)
        elif ch == 'Aliphatic':
            subch = np.arange(249, 260)
        chunk_valid_mask_padded = make_stripped_chunk_valid_mask(subch=subch, num_subchannels=num_subchannels, num_channels=num_channels, 
                                        num_columns=num_columns, subchannel_padding=1)
        chunk_valid_mask = make_stripped_chunk_valid_mask(subch=subch, num_subchannels=num_subchannels, num_channels=num_channels, 
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

def mask_bright_pixels(local_vars):
    sub_data = local_vars['sub_data']
    sub_weight = local_vars['sub_weight']
    
    valid_mask = sub_weight > 0
    if np.sum(valid_mask) > 0:
        threshold = np.nanpercentile(sub_data[valid_mask], 25)
        sub_data[sub_data > threshold] = np.nan
        
    return sub_data

if __name__ == "__main__":
    # ----------------------------- Start of Settings -----------------------------
    frame_setting = {
        'Detector': 1,
        'NumSub': 10,
        'NumCh': 34,
        'NumCol': 3,
    }

    selfcal_config = PipelineWrapper.PipelineConfig(
        output_dir='/mnt/md124/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_nep_qr2_det{frame_setting["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2
    )

    calibration_kwargs = {
        'apply_mask': True,
        'apply_weight': False,
        'outlier_thresh': 5.0,
        'ignore_list': [],
        'batch_size': 100,
        'offset_regularization': True,
        'reg_weight': 0.1,
        'weighted_damping': True,
        'damp_weight': 0.1,
        'max_workers': 30,
        'postprocess_func': None, #mask_bright_pixels,
    }

    lsqr_kwargs = {
        'atol': 1e-06,
        'btol': 1e-06,
        'damp': 0,
        'iter_lim': 10,
        'precondition': True,
        'solver': 'lsqr',
    }

    mosaic_kwargs = {
        'apply_mask': True,
        'apply_weight': False,
        'make_std_map': True,
        'apply_sigma_clipping': True,
        'sigma': 2.0,
        'ignore_list': [21],
        'cache_batch_size': 100,
        'coadd_batch_size': 200,
        'cache_intermediate': True,
        'max_workers': 30
    }
    
    mosaic_oversample_factor = 2

    CACHE_DIR = '/home/thomasli/spherex/selfcal/cache/'
    FILE_SUFFIX = f'_damp0p1_reg0p1_outThresh5_sigma2'

    # Channels to process
    chs = [[26], [27], [28], [29], [30], [31], [32], [33], [34]]
    # chs = [[29], [30], [31], [32], [33], [34]]
    # chs = [[1], [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17], [18], [19], [20], [21], [22], [23], [24], [25], [26], [27], [28], [29], [30], [31], [32], [33], [34]]
    # chs = ['Aliphatic', 'Aromatic']
    # ----------------------------- End of Settings -----------------------------

    frame_setting_str = '_'.join([f'{key}{value}' for key, value in frame_setting.items()])
    
    # 1. Prepare overarching detector inputs
    detector_inputs = prepare_detector_inputs(frame_setting, mosaic_oversample_factor)
    
    # 2. Iterate through channels
    for ch in chs:
        if isinstance(ch, list):
            job_name = f'Ch{"-".join(map(str, ch))}'
        else:
            job_name = ch
        t0 = time.time()
        print(f"Processing channel {job_name} for detector {frame_setting['Detector']}...")

        job_tag = f'{frame_setting_str}_{job_name}{FILE_SUFFIX}'
        cal_file = f'cal_{job_tag}.h5'
        mos_file = f"mosaic_{job_tag}.fits"
        cache_dir = f'{CACHE_DIR}cache_{job_tag}'

        # Prepare specific inputs for this channel
        channel_inputs = prepare_channel_inputs(ch, frame_setting, detector_inputs['det_chunk_map'], detector_inputs['grid_chunk_map'])
        
        # ----------------------------- Calibration -----------------------------
        cal_path = os.path.join(selfcal_config.cal_dir, cal_file)
        cc = PipelineWrapper.Calibrator(selfcal_config)
        if os.path.exists(cal_path):
            print(f"Calibration file {cal_path} already exists. Skipping calibration.")
        else:
            cc.setup_lsqr(
                chunk_map=detector_inputs['det_chunk_map'],
                grid_valid_weight=channel_inputs['det_valid_mask'],
                oversample_factor=1,
                adj_info=detector_inputs['adj_info'],
                **calibration_kwargs
            )
            
            print('Computing initial guess offsets...')
            t00 = time.time()
            offset = compute_offsets_guess(reproj_list=cc.reproj_list, det_chunk_map=detector_inputs['det_chunk_map'])
            skymap = np.zeros(cc.ref_shape)
            x0 = encode_x(skymap, offset)
            print(f"Initial guess offsets computed in {time.time() - t00:.2f} seconds.")
            
            with threadpool_limits(limits=8, user_api='blas'):
                cc.apply_lsqr(x0=x0, **lsqr_kwargs)
            cal_path = cc.save_calibration(cal_file=cal_file)

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

        # Append wavelength maps
        print("Coadding wavelength maps...")
        t00 = time.time()
        wav_mean, wav_std = wav_coadd(detector_inputs['det_BC'], detector_inputs['det_BW'], 
                                      mean_map=maps['mean_map']['data'], 
                                      std_map=maps['std_map']['data'], 
                                      reproj_list=mm.reproj_list, 
                                      cache_list=mm.cached_list,
                                      ref_shape=maps['mean_map']['data'].shape, 
                                      sigma=mosaic_kwargs['sigma'], 
                                      batch_size=40, max_workers=30)    
        print(f"Wavelength coaddition finished in {time.time() - t00:.2f} seconds.")

        mm.append_maps({
            'wav_mean_map': {'data': wav_mean, 'unit': 'um'},
            'wav_std_map': {'data': wav_std, 'unit': 'um'}
        })

        mm.save_mosaic(mos_file=mos_file, overwrite=True)
         
        # Clean up
        del cc, mm, maps
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        gc.collect()
        
        print(f"Finished channel {job_name} for detector {frame_setting['Detector']} in {time.time() - t0:.2f} seconds.")
        print("-" * 50 + "\n")
