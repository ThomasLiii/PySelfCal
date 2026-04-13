

import os
import sys
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)
from SelfCal.SPHERExUtility import make_fiducial_chunk_map, make_fiducial_chunk_mask, interpolate_array, load_calibration, interp_2d_vertical, interp_1d
import numpy as np

def load_lvf_params(filename, input_dir='/home/thomasli/spherex/selfcal/selfcal_scripts/lvf_params'):
    input_path = os.path.join(input_dir, filename)
    if not os.path.exists(input_path):
        print(f"LVF parameters file {input_path} not found. Returning None.")
        return None
    lvf_params = np.load(input_path, allow_pickle=True).item()
    print(f"Loaded LVF parameters from {input_path}")
    return lvf_params

def save_lvf_params(lvf_params, output_dir='/home/thomasli/spherex/selfcal/selfcal_scripts/lvf_params'):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, lvf_params['filename'])
    np.save(output_path, lvf_params)
    print(f"Saved LVF parameters to {output_path}")

if __name__ == "__main__":
    detector_list = [1, 2, 3, 4, 5, 6]
    for DETECTOR in detector_list:
        det_BC, det_BW = load_calibration(band=DETECTOR, calibration_dir='/home/thomasli/spherex/SPHEREx_Spectral_Calibration')
        _, lvf_params = make_fiducial_chunk_map(DETECTOR, det_BC, num_subchannels=10, num_channels=34, 
                                                        oversample_factor=1)
        lvf_params['filename'] = f'lvf_params_D{DETECTOR}.npy'
        save_lvf_params(lvf_params)