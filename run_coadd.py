from py import PipelineWrapper
from astropy.io import fits
import numpy as np
import glob
from py.SPHERExUtility import interpolate_array, make_chunk_map, make_chunk_mask, visualize_chunk_map

detector = 6

config = {}
config['output_dir'] = '/home/thomasli/spherex/selfcal/outputs'
config['run_name'] = f'nep_det{detector}_6p2arcsec'
config['resolution_arcsec'] = 6.2

chunk_map = make_chunk_map(detector, interp_factor=5)
chs = [[17]]

for ch in chs:
    ch_name = '-'.join([str(i) for i in ch])
    print(f"Processing channels {ch_name}")
    
    chunk_valid_mask = make_chunk_mask(ch, interp_factor=5)

    cal_path = f'/home/thomasli/spherex/selfcal/outputs/nep_det{detector}_6p2arcsec/calibration/cal_det{detector}_ch{ch_name}.h5'

    mm = PipelineWrapper.Mosaicker(config)
    mm.load_calibration(cal_path=cal_path)

    mean, std, sc = mm.make_mosaic(
        apply_mask=True, 
        apply_weight=True, 
        chunk_map=chunk_map, 
        chunk_valid_mask=chunk_valid_mask, 
        max_workers=40,
        make_std_map=True, 
        apply_sigma_clipping=True, 
        sigma=2.0
    )

    mm.save_mosaic(mos_file=f'nep_6p2arcsec_det{detector}_ch{ch_name}.fits', overwrite=True)