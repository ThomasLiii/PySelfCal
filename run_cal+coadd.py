from SelfCal import PipelineWrapper
from astropy.io import fits
import numpy as np
import glob
from SelfCal.SPHERExUtility import interpolate_array, make_chunk_map, make_chunk_mask, visualize_chunk_map, interp_2d_vertical
import matplotlib.pyplot as plt
import matplotlib as mpl
# Import LogNorm
from tqdm import tqdm
import os
from SelfCal import MakeMap
import gc

detector = 1
config = {}
config['output_dir'] = '/mnt/md127/thomasli/selfcal/outputs/'
config['run_name'] = f'nep_det{detector}_3p1arcsec'
config['resolution_arcsec'] = 3.1

chunk_map = make_chunk_map(detector, interp_factor=10,
                          channel_file='/home/thomasli/spherex/spherex_channels.csv')

chs = [[16]]#, [2], [3], [4], [5], [6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17]]

for ch in chs:
    print(f"Processing channel {ch} for detector {detector}")
    chunk_valid_mask = make_chunk_mask(ch, interp_factor=10)
    det_valid_mask = chunk_valid_mask[chunk_map]
    cc = PipelineWrapper.Calibrator(config)
    cc.setup_lsqr(
        apply_mask=True, 
        apply_weight=False, 
        chunk_map=chunk_map, 
        det_valid_mask=det_valid_mask, 
        max_workers=50, 
        outlier_thresh=20.0,
        ignore_list=[],
        )

    cc.apply_lsqr(x0=None, atol=1e-06, btol=1e-06, damp=1e-1, iter_lim=500)

    cal_path = cc.save_calibration(cal_file=f'cal_det{detector}_ch{'-'.join(map(str, ch))}.h5')

    mm = PipelineWrapper.Mosaicker(config)
    
    mm.load_calibration(cal_path=cal_path)

    maps = mm.make_mosaic(
        apply_mask=True, 
        apply_weight=False, 
        chunk_map=chunk_map, 
        det_valid_mask=det_valid_mask, 
        max_workers=50,
        make_std_map=True, 
        apply_sigma_clipping=True, 
        sigma=1.0,
        ignore_list=[21],
        interp_func=interp_2d_vertical
    )

    mm.save_mosaic(mos_file=f'mosaic_det{detector}_ch{"-".join(map(str, ch))}.fits', overwrite=True)

    # Clear memory
    del cc, mm, maps
    gc.collect()
