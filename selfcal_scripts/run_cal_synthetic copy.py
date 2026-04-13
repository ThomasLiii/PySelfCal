import sys
import os
import shutil
import time
import gc
from functools import partial
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from SelfCal import PipelineWrapper
from SelfCal.MakeMap import encode_x
from SelfCal.SPHERExUtility import load_calibration, load_lvf_params, compute_column_adjacency, \
compute_subchannel_adjacency, compute_offsets_guess, \
make_stripped_chunk_map, make_stripped_chunk_valid_mask, make_spherex_stripped_offset_map, fast_vertical_dist
from SelfCal.SPHERExAppendWav import wav_coadd


@dataclass
class FrameSetting:
    def __init__(self, detector: int = 2, num_sub: int = 10, num_ch: int = 17, num_col: int = 1):
        self.detector = detector
        self.num_sub = num_sub
        self.num_ch = num_ch
        self.num_col = num_col

    @property
    def tag(self) -> str:
        return f"Detector{self.detector}_NumSub{self.num_sub}_NumCh{self.num_ch}_NumCol{self.num_col}"


@dataclass
class RunConfig:
    frame: FrameSetting = field(default_factory=FrameSetting)
    
    mosaic_oversample_factor: int = 2
    cache_parent_dir: str = 'cache/'
    file_suffix: str = ''

    calibration_kwargs: Dict[str, Any] = field(default_factory=lambda: {
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
    })

    lsqr_kwargs: Dict[str, Any] = field(default_factory=lambda: {
        'atol': 1e-06,
        'btol': 1e-06,
        'damp': 1e-3,
        'iter_lim': 20,
        'precondition': True
    })

    mosaic_kwargs: Dict[str, Any] = field(default_factory=lambda: {
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
    })

    def __init__(self, frame: FrameSetting, output_dir: str = 'outputs', run_name: str = 'run', resolution_arcsec: float = 1.0, channels: List[int] = None, **kwargs):
        self.frame = frame
        self.output_dir = output_dir
        self.run_name = run_name
        self.resolution_arcsec = resolution_arcsec
        self.channels = channels
        super().__init__(**kwargs)

    def __post_init__(self):
        if self.run_name is None:
            self.run_name = f'nep_det{self.frame.detector}_6p2arcsec'

    def get_pipeline_config(self) -> PipelineWrapper.PipelineConfig:
        return PipelineWrapper.PipelineConfig(
            output_dir=self.output_dir,
            run_name=self.run_name,
            resolution_arcsec=self.resolution_arcsec
        )

    def get_job_tag(self, channel_list: List[int]) -> str:
        ch_str = '_'.join([f'Ch{c}' for c in channel_list])
        return f"{self.frame.tag}_{ch_str}{self.file_suffix}"
    
    def get_cal_file(self, channel_list: List[int]) -> str:
        return f"cal_{self.get_job_tag(channel_list)}.h5"

    def get_mos_file(self, channel_list: List[int]) -> str:
        return f"mosaic_{self.get_job_tag(channel_list)}.fits"

    def get_cache_dir(self, channel_list: List[int]) -> str:
        return os.path.join(self.cache_parent_dir, f"cache_{self.get_job_tag(channel_list)}")


def prepare_detector_inputs(frame_setting: FrameSetting, mosaic_setting_oversample: int):
    detector = frame_setting.detector
    num_subchannels = frame_setting.num_sub
    num_channels = frame_setting.num_ch
    num_columns = frame_setting.num_col
    
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


def prepare_channel_inputs(ch, frame_setting: FrameSetting, det_chunk_map, grid_chunk_map):
    num_subchannels = frame_setting.num_sub
    num_channels = frame_setting.num_ch
    num_columns = frame_setting.num_col
    
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

    output_dir: str = '/data3/caoye/selfcal/outputs'
    run_name: Optional[str] = None
    resolution_arcsec: float = 6.2
if __name__ == "__main__":
    # ----------------------------- Start of Settings -----------------------------
    frame = FrameSetting(detector=2, num_sub=10, num_ch=17, num_col=1)
    config = RunConfig(
        frame=frame,
        output_dir='/data3/caoye/selfcal/outputs',
        run_name='nep_det2_6p2arcsec',
        resolution_arcsec=6.2,
        channels=[10]
    )

    # ----------------------------- End of Settings -----------------------------

    # 1. Prepare overarching detector inputs
    detector_inputs = prepare_detector_inputs(config.frame, config.mosaic_oversample_factor)
    
    # 2. Iterate through channels
    job_name = f'Ch{config.channels[0]}'
    t0 = time.time()
    print(f"Processing channel {job_name} for detector {config.frame.detector}...")

    cal_file = config.get_cal_file(config.channels)
    mos_file = config.get_mos_file(config.channels)
    cache_dir = config.get_cache_dir(config.channels)

    # Prepare specific inputs for this channel
    channel_inputs = prepare_channel_inputs(config.channels, config.frame, detector_inputs['det_chunk_map'], detector_inputs['grid_chunk_map'])
        
    # ----------------------------- Calibration -----------------------------
    pipeline_config = config.get_pipeline_config()
    cc = PipelineWrapper.Calibrator(pipeline_config)
    cc.setup_lsqr(
        chunk_map=detector_inputs['det_chunk_map'],
        grid_valid_weight=channel_inputs['det_valid_mask'],
        oversample_factor=1,
        adj_info=detector_inputs['adj_info'],
        **config.calibration_kwargs
    )
    offset = compute_offsets_guess(reproj_list=cc.reproj_list, det_chunk_map=detector_inputs['det_chunk_map'])
    skymap = np.zeros(cc.ref_shape)
    x0 = encode_x(skymap, offset)
    cc.apply_lsqr(x0=x0, **config.lsqr_kwargs)
    cal_path = cc.save_calibration(cal_file=cal_file)

    # ----------------------------- Mosaicking -----------------------------
    partial_make_offset_map = partial(make_spherex_stripped_offset_map,
                                    chunk_valid_mask=channel_inputs['chunk_valid_mask'], 
                                    lvf_params=detector_inputs['lvf_params'], 
                                    r_edges=detector_inputs['r_edges'], 
                                    x_edges=detector_inputs['x_edges'], 
                                    tot_subchannels=config.frame.num_sub * config.frame.num_ch + 2, 
                                    num_columns=config.frame.num_col,
                                    fill_invalid=True)
    
    mm = PipelineWrapper.Mosaicker(pipeline_config)
    mm.load_calibration(cal_path=cal_path)

    maps = mm.make_mosaic(
        chunk_map=detector_inputs['grid_chunk_map'],
        grid_valid_weight=channel_inputs['grid_valid_weight'],
        oversample_factor=config.mosaic_oversample_factor,
        det_offset_func=partial_make_offset_map,
        cache_dir=cache_dir,
        **config.mosaic_kwargs
    )

    mm.save_mosaic(mos_file=mos_file, overwrite=True)
         
    # Clean up
    del cc, mm, maps
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    gc.collect()
        
    print(f"Finished channel {job_name} for detector {config.frame.detector} in {time.time() - t0:.2f} seconds.")
    print("-" * 50 + "\n")
