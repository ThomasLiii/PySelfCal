import sys
import os
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from SelfCal import PipelineWrapper
from SelfCal.SPHERExUtility import interpolate_array, make_fiducial_chunk_map, make_fiducial_chunk_mask,\
visualize_chunk_map, load_calibration, make_spherex_offset_map
from SelfCal import MakeMap

from astropy.io import fits
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
# Import LogNorm
from tqdm import tqdm
import gc
from functools import partial

detector = 4
config = {}
config['output_dir'] = '/mnt/md124/thomasli/selfcal/outputs/'
config['run_name'] = f'nep_det{detector}_3p1arcsec'
config['resolution_arcsec'] = 3.1


det_BC, det_BW = load_calibration(band=detector, calibration_dir='/home/thomasli/spherex/SPHEREx_Spectral_Calibration')

oversample_factor = 4
chunk_map, lvf_params = make_fiducial_chunk_map(detector, det_BC, num_subchannels=10, num_channels=17, oversample_factor=oversample_factor)

chs = [[10], [12], [14]]

for ch in chs:
    print(f"Processing channel {ch} for detector {detector}")
    chunk_valid_mask = make_fiducial_chunk_mask(ch,  num_subchannels=10, num_channels=17)
    det_valid_mask = chunk_valid_mask[chunk_map]
    cc = PipelineWrapper.Calibrator(config)
    cc.setup_lsqr(
        apply_mask=True, 
        apply_weight=True, 
        chunk_map=chunk_map, 
        det_valid_mask=det_valid_mask, 
        max_workers=40, 
        outlier_thresh=10.0,
        ignore_list=[],
        oversample_factor=oversample_factor,
        batch_size=10
        )

    cc.apply_lsqr(x0=None, atol=1e-06, btol=1e-06, damp=1e-0, iter_lim=500)

    cal_path = cc.save_calibration(cal_file=f'cal_det{detector}_ch{'-'.join(map(str, ch))}.h5')

    mm = PipelineWrapper.Mosaicker(config)
    
    mm.load_calibration(cal_path=cal_path)

    partial_make_offset_map = partial(make_spherex_offset_map, chunk_valid_mask=chunk_valid_mask, lvf_params=lvf_params)
    maps = mm.make_mosaic(
        apply_mask=True, 
        apply_weight=True, 
        chunk_map=chunk_map, 
        det_valid_mask=det_valid_mask, 
        max_workers=40,
        make_std_map=True, 
        apply_sigma_clipping=True, 
        sigma=1.0,
        ignore_list=[21],
        oversample_factor=oversample_factor,
        det_offset_func=partial_make_offset_map,
        batch_size=20
    )

    mm.save_mosaic(mos_file=f'mosaic_det{detector}_ch{"-".join(map(str, ch))}_narrow.fits', overwrite=True)

    # Clear memory
    del cc, mm, maps
    gc.collect()
