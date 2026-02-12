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

from SelfCal import PipelineWrapper
from SelfCal.SPHERExUtility import make_fiducial_chunk_map, make_fiducial_chunk_mask, \
load_calibration, make_spherex_offset_map, compute_offsets_guess
from SelfCal.SPHERExAppendWav import wav_coadd

DETECTOR = 5
config = {}
config['output_dir'] = '/mnt/md124/thomasli/selfcal/outputs/'
config['run_name'] = f'SPHEREx_nep_qr2_det{DETECTOR}_6p2arcsec'
config['resolution_arcsec'] = 6.2

qr1_dir = '/mnt/md124/SPHEREx/SPHEREx_nep_data/qr1_newgain'
qr2_dir = '/mnt/md124/SPHEREx/SPHEREx_nep_data/qr2'
file_pattern = f'/*/*/*/*D{DETECTOR}*.fits'

exposure_list = []
exposure_list += glob.glob(qr1_dir+file_pattern)
exposure_list += glob.glob(qr2_dir+file_pattern)
exposure_list = sorted(exposure_list)

remove_list = []
for exp_file in exposure_list:
    hdul = fits.open(exp_file)
    header = hdul[1].header
    # Check for good astrometry
    good_astrometry = header.get('FINAST', 2)
    if good_astrometry != 0:
        print(f"Skipping {exp_file} due to poor astrometry (FINAST={good_astrometry})")
        exposure_list.remove(exp_file)
        remove_list.append(exp_file)
print(f"Removed {len(remove_list)} exposures with poor astrometry")
print(f"Found {len(exposure_list)} exposures")

# Initialize Reprojector and run reprojection
rr = PipelineWrapper.Reprojector(config, exposure_list=exposure_list)
# Define reference frame with padding
rr.define_reference(padding_pixels=100, use_ext=[1])
# Run reprojection
rr.run_reproject(max_workers=100, # number of parallel workers for reprojection
                 reproj_func='exact', # reprojection function, can be 'exact', 'interp', or 'adaptive' (use 'exact' for best accuracy, 'interp' for speed)
                 padding_percentage=0.05, # percentage of padding around the footprint
                 sci_ext_list=[1], # list of science extensions in the input fits files
                 dq_ext_list=[2], # list of data quality extensions in the input fits files
                 exp_idx_list=np.arange(0, len(exposure_list)), # list of exposure indices
                 det_idx_list=[0]*len(exposure_list), # list of detector indices (0-indexed) corresponding to each exposure
                 replace_existing=False, # whether to replace existing reprojected files
                 reproject_kwargs={'parallel': 4} # additional kwargs for reprojection
                )

print("Reprojection complete")