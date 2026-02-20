import sys
import os
import shutil
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
import time
from astropy.io import fits
import numpy as np
import gc
from functools import partial
import glob
from tqdm import tqdm

from SelfCal import PipelineWrapper
from SelfCal.SPHERExUtility import load_calibration, load_lvf_params, compute_vertical_strip_adjacency, \
make_stripped_chunk_map, make_stripped_chunk_valid_mask, make_spherex_stripped_offset_map, fast_vertical_dist
from SelfCal.SPHERExAppendWav import wav_coadd


def prepare_detector_inputs(frame_setting, calibration_setting, mosaic_setting):
    detector = frame_setting['Detector']
    num_subchannels = frame_setting['NumSub']
    num_channels = frame_setting['NumCh']
    num_columns = frame_setting['NumCol']
    oversample_factor = mosaic_setting['OversampleFactor']

    lvf_filename = f'lvf_params_D{detector}.npy'
    lvf_params = load_lvf_params(lvf_filename)

    det_BC, det_BW = load_calibration(band=detector, calibration_dir='/home/thomasli/spherex/SPHEREx_Spectral_Calibration')
    grid_chunk_map, _, _, _ = make_stripped_chunk_map(detector, num_subchannels=num_subchannels, num_channels=num_channels, num_columns=num_columns,
                                                    oversample_factor=oversample_factor, lvf_params=lvf_params)
    det_chunk_map, _, r_edges, x_edges = make_stripped_chunk_map(detector, num_subchannels=num_subchannels, num_channels=num_channels, num_columns=num_columns,
                                            oversample_factor=1, lvf_params=lvf_params)
    detector_inputs = {
        'lvf_params': lvf_params,
        'det_BC': det_BC,
        'det_BW': det_BW,
        'grid_chunk_map': grid_chunk_map,
        'det_chunk_map': det_chunk_map,
        'r_edges': r_edges,
        'x_edges': x_edges
    }
    return detector_inputs

def prepare_channel_inputs(ch, frame_setting):
    num_subchannels = frame_setting['NumSub']
    num_channels = frame_setting['NumCh']
    num_columns = frame_setting['NumCol']
     # Prepare masks
    chunk_valid_mask_padded = make_stripped_chunk_valid_mask(ch, num_subchannels=num_subchannels, num_channels=num_channels, 
                                        num_columns=num_columns, subchannel_padding=1)
    chunk_valid_mask = make_stripped_chunk_valid_mask(ch, num_subchannels=num_subchannels, num_channels=num_channels, 
                                        num_columns=num_columns, subchannel_padding=0)

    channel_inputs = {
        'chunk_valid_mask_padded': chunk_valid_mask_padded,
        'chunk_valid_mask': chunk_valid_mask
    }
    return channel_inputs

if __name__ == "__main__":
    # ----------------------------- Start of Settings -----------------------------
    frame_setting = {
        'Detector': 4,
        'NumSub': 10,
        'NumCh': 34,
        'NumCol': 5,
    }

    calibration_setting = {
        'ApplyMask': True,
        'ApplyWeight': False,
        'OutlierThresh': 2.0,
        'IgnoreList': [21],
        'OffsetRegularization': True,
        'RegWeight': 10.0,
        'WeightedDamping': True,
        'DampWeight': 100.0,
    }

    mosaic_setting = {
        'ApplyMask': True,
        'ApplyWeight': False,
        'MakeStdMap': True,
        'ApplySigmaClipping': True,
        'Sigma': 1.0,
        'IgnoreList': [21],
        'OversampleFactor': 2,
    }

    CACHE_DIR = '/home/thomasli/spherex/selfcal/cache/'
    FILE_SUFFIX = f'_Control'

    selfcal_config = {}
    selfcal_config['output_dir'] = '/mnt/md124/thomasli/selfcal/outputs/'
    selfcal_config['run_name'] = f'SPHEREx_nep_qr2_det{frame_setting["Detector"]}_6p2arcsec'
    selfcal_config['resolution_arcsec'] = 6.2

    # ----------------------------- End of Settings -----------------------------

    frame_setting_str = '_'.join([f'{key}{value}' for key, value in frame_setting.items()])
    detector_inputs = prepare_detector_inputs(frame_setting, calibration_setting, mosaic_setting)
    # chs = [[8], [9], [10], [11], [12], [13], [14], [15], [16]]
    # chs = [[17], [18], [19], [20], [21], [22], [23], [24], [25]]
    chs = [[8]]
    channel_jobs = {
        f'Ch{"-".join(map(str, ch))}': prepare_channel_inputs(ch, frame_setting) for ch in chs
    }

    for job_name, channel_inputs in channel_jobs.items():
        t0 = time.time()
        print(f"Processing channel {job_name} for detector {frame_setting['Detector']}...")

        # Prepare inputs
        chunk_valid_mask_padded = channel_inputs['chunk_valid_mask_padded']
        chunk_valid_mask = channel_inputs['chunk_valid_mask']
        adj_info = compute_vertical_strip_adjacency(detector_inputs['det_chunk_map'], frame_setting['NumCol'])
        job_tag = f'{frame_setting_str}_{job_name}{FILE_SUFFIX}'
        cal_file = f'cal_{job_tag}.h5'
        mos_file = f"mosaic_{job_tag}.fits"

        # Calibration
        cc = PipelineWrapper.Calibrator(selfcal_config)
        cal_path = os.path.join(cc.config['cal_dir'], cal_file)
        # if os.path.exists(cal_path):
        #     print(f"Calibration file {cal_path} already exists. Skipping calibration and mosaicking for channel {job_name}.")
        # else:
        det_valid_mask = chunk_valid_mask_padded[detector_inputs['det_chunk_map']]
        det_valid_weight = fast_vertical_dist(det_valid_mask)
        det_valid_weight /= np.max(det_valid_weight)  # Normalize weights to [0, 1]
        # det_valid_weight[det_valid_mask>0] = 0.5 + 0.5 * det_valid_weight[det_valid_mask>0]  # Scale to [0.5, 1]
        cc.setup_lsqr(
            apply_mask=calibration_setting['ApplyMask'], 
            apply_weight=calibration_setting['ApplyWeight'],
            chunk_map=detector_inputs['det_chunk_map'], 
            grid_valid_weight=det_valid_mask, 
            max_workers=50, 
            outlier_thresh=calibration_setting['OutlierThresh'],
            ignore_list=calibration_setting['IgnoreList'],
            oversample_factor=1,
            batch_size=40,
            offset_regularization=calibration_setting['OffsetRegularization'],
            reg_weight=calibration_setting['RegWeight'],
            adj_info=adj_info,
            weighted_damping=calibration_setting['WeightedDamping'],
            damp_weight=calibration_setting['DampWeight'],
            )
        
        cc.apply_lsqr(x0=None, atol=1e-06, btol=1e-06, damp=1e-3, iter_lim=50, precondition=False)
        cal_path = cc.save_calibration(cal_file=cal_file)

        # Mosaicking
        partial_make_offset_map = partial(make_spherex_stripped_offset_map,
                                    chunk_valid_mask=chunk_valid_mask, 
                                    lvf_params=detector_inputs['lvf_params'], 
                                    r_edges=detector_inputs['r_edges'], 
                                    x_edges=detector_inputs['x_edges'], 
                                    tot_subchannels=frame_setting['NumSub']*frame_setting['NumCh']+2, 
                                    num_columns=frame_setting['NumCol'])
        
        mm = PipelineWrapper.Mosaicker(selfcal_config)
        mm.load_calibration(cal_path=cal_path)
        cache_dir = f'{CACHE_DIR}cache_{job_tag}'
        grid_valid_mask = chunk_valid_mask_padded[detector_inputs['grid_chunk_map']]
        grid_valid_weight = fast_vertical_dist(grid_valid_mask)
        grid_valid_weight /= np.max(grid_valid_weight)  # Normalize weights to [0, 1]
        # grid_valid_weight[grid_valid_mask>0] = 0.5 + 0.5 * grid_valid_weight[grid_valid_mask>0]  # Scale to [0.5, 1]
        maps = mm.make_mosaic(
            apply_mask=mosaic_setting['ApplyMask'], 
            apply_weight=mosaic_setting['ApplyWeight'], 
            chunk_map=detector_inputs['grid_chunk_map'], 
            grid_valid_weight=grid_valid_weight, 
            max_workers=50,
            make_std_map=mosaic_setting['MakeStdMap'], 
            apply_sigma_clipping=mosaic_setting['ApplySigmaClipping'],  
            sigma=mosaic_setting['Sigma'],
            ignore_list=mosaic_setting['IgnoreList'],
            oversample_factor=mosaic_setting['OversampleFactor'],
            det_offset_func=partial_make_offset_map,#partial_make_offset_map,
            cache_batch_size=40,
            coadd_batch_size=100,
            cache_dir=cache_dir,
            cache_intermediate=True,
            det_aux=None
        )

        # Append wavelength maps
        wav_mean, wav_std = wav_coadd(detector_inputs['det_BC'], detector_inputs['det_BW'], 
                                      mean_map=maps['mean_map']['data'], 
                                      std_map=maps['std_map']['data'], 
                                      reproj_list=mm.reproj_list, 
                                      cache_list=mm.cached_list,
                                      ref_shape=maps['mean_map']['data'].shape, 
                                      sigma=mosaic_setting['Sigma'], batch_size=40, max_workers=50)    

        wav_mean_maps = {'data': wav_mean, 'unit': 'um'}
        wav_std_maps = {'data': wav_std, 'unit': 'um'}
        mm.append_maps({'wav_mean_map': wav_mean_maps, 'wav_std_map': wav_std_maps})

        mm.save_mosaic(mos_file=mos_file, overwrite=True)
         
        # Clean up
        del mm, maps
        shutil.rmtree(cache_dir)
        gc.collect()
        t1 = time.time()
        print(f"Finished channel {job_name} for detector {frame_setting['Detector']} in {t1 - t0:.2f} seconds.")
        print("-" * 50)
        print('')