from py import PipelineWrapper
from astropy.io import fits
import numpy as np
import glob
from py.SPHERExUtility import interpolate_array, make_chunk_map, make_chunk_mask, visualize_chunk_map

detector = 1
exposure_list = glob.glob(f'/data1/SPHEREx/deep_north/*/*/*/*D{detector}*.fits')
for exp_file in exposure_list:
    hdul = fits.open(exp_file)
    header = hdul[1].header
    good_astrometry = header.get('FINAST', 2)
    if good_astrometry != 0:
        print(f"Skipping {exp_file} due to poor astrometry (FINAST={good_astrometry})")
        exposure_list.remove(exp_file)
print(f"Found {len(exposure_list)} exposures")

config = {}
config['output_dir'] = '/home/thomasli/spherex/selfcal/outputs'
config['run_name'] = f'nep_det{detector}_6p2arcsec'
config['resolution_arcsec'] = 6.2

rr = PipelineWrapper.Reprojector(config, exposure_list=exposure_list)

rr.define_reference(padding_pixels=100, use_ext=[1])

rr.run_reproject(max_workers=40, method='exact', padding_percentage=0.05, oversample_factor=2, 
                      sci_ext_list=[1], 
                      dq_ext_list=[2], 
                      exp_idx_list=np.arange(0, len(exposure_list)), 
                      det_idx_list=[0]*len(exposure_list),
                      replace_existing=False)