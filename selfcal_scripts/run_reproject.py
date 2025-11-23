import sys
import os
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from SelfCal import PipelineWrapper
from astropy.io import fits
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400 # User can set this outside the class if needed
# Import LogNorm
from matplotlib.colors import LogNorm
from tqdm import tqdm
from SelfCal import MakeMap

detector = 1
# exposure_list = glob.glob(f'/data1/SPHEREx/reproc_data/deep_north/*/*/*/*D{detector}*.fits')
exposure_list = glob.glob(f'/mnt/md127/SPHEREx/reproc_data/deep_north/*/*/*/*D{detector}*.fits')
for exp_file in exposure_list:
    hdul = fits.open(exp_file)
    header = hdul[1].header
    good_astrometry = header.get('FINAST', 2)
    if good_astrometry != 0:
        print(f"Skipping {exp_file} due to poor astrometry (FINAST={good_astrometry})")
        exposure_list.remove(exp_file)
print(f"Found {len(exposure_list)} exposures")

config = {}
config['output_dir'] = '/mnt/md124/thomasli/selfcal/outputs/'
config['run_name'] = f'nep_det{detector}_3p1arcsec'
config['resolution_arcsec'] = 3.1
# config['ref_path'] = '/home/thomasli/spherex/selfcal/outputs/common_ref.fits'

rr = PipelineWrapper.Reprojector(config, exposure_list=exposure_list)

rr.define_reference(padding_pixels=100, use_ext=[1])

rr.run_reproject(max_workers=50, reproj_func='exact', padding_percentage=0.05, 
                    sci_ext_list=[1], 
                    dq_ext_list=[2], 
                    exp_idx_list=np.arange(0, len(exposure_list)), 
                    det_idx_list=[0]*len(exposure_list),
                    replace_existing=False,
                 reproject_kwargs={'parallel': 4}
                )