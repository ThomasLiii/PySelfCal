import h5py
import numpy as np
import os
import glob
from multiprocessing import Pool
import multiprocessing
from tqdm import tqdm

CONVERSION_RULES = {
    'sub_data': (np.float32, None),  # float64 -> float32
    'det_data': (np.float32, None),  # float64 -> float32
    'ref_coords': (np.int32, -9999), # float64 -> int32, nan -> -9999
    'grid_bitmask': (np.int32, -9999), # float64 -> int32, nan -> -9999
    'grid_mapping': (np.int16, -9999), # float64 -> int16, nan -> -9999
    'sub_foot': (np.int16, 0),      # float64 -> int16, nan -> 0
}
COPY_DATASETS = ['sub_header', 'det_header', 'file_path']

def convert_h5_file(filepath):
    """
    Converts a single HDF5 file to the new, optimized format.
    Writes to a .tmp file first, then replaces the original on success.
    """
    temp_filepath = filepath + ".tmp"
    
    try:
        with h5py.File(filepath, 'r') as hf_in, h5py.File(temp_filepath, 'w') as hf_out:
            
            # Iterate over all datasets in the source file
            for dset_name in hf_in:
                if dset_name in CONVERSION_RULES:
                    # This dataset needs conversion
                    target_dtype, nan_val = CONVERSION_RULES[dset_name]
                    
                    # Read all data from the source dataset
                    data = hf_in[dset_name][()] 
                    
                    if nan_val is not None:
                        # This is an int conversion, handle NaNs first
                        # np.nan_to_num replaces np.nan with nan_val
                        data = np.nan_to_num(data, nan=nan_val)
                    
                    # Cast to target type
                    data_converted = data.astype(target_dtype)
                    
                    # Create the new dataset with compression
                    hf_out.create_dataset(dset_name, 
                                        data=data_converted, 
                                        compression='gzip', 
                                        dtype=target_dtype)
                    
                elif dset_name in COPY_DATASETS:
                    # This is a string/other dataset, just copy it
                    hf_in.copy(dset_name, hf_out)
                    
                else:
                    # Found a dataset not in our rules, copy it as-is to be safe
                    print(f"\nWarning: Unknown dataset '{dset_name}' in {filepath}.")
                    print("Copying as-is.")
                    hf_in.copy(dset_name, hf_out)

        # If the 'with' block succeeded, replace the original file
        os.replace(temp_filepath, filepath)

    except Exception as e:
        print(f"\nERROR: Failed to convert {filepath}: {e}")
        print("Original file was NOT modified.")
        # Clean up the failed temp file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

if __name__ == "__main__":
    h5_dir_list = ['/data3/thomasli/selfcal/outputs/nep_det4_3p1arcsec/reprojected',
                   '/data3/thomasli/selfcal/outputs/nep_det1_6p2arcsec/reprojected',
                   '/data3/thomasli/selfcal/outputs/nep_det2_6p2arcsec/reprojected',
                   '/data3/thomasli/selfcal/outputs/nep_det3_6p2arcsec/reprojected',
                   '/data3/thomasli/selfcal/outputs/nep_det4_6p2arcsec/reprojected',
                   '/data3/thomasli/selfcal/outputs/nep_det5_6p2arcsec/reprojected',
                   '/data3/thomasli/selfcal/outputs/nep_det6_6p2arcsec/reprojected',
                   ]
    
    # Use all available CPU cores
    num_processes = 40

    for h5_dir in h5_dir_list:
        print(f"Processing directory: {h5_dir}")
        reproj_list = glob.glob(f'{h5_dir}/*.h5')
        with Pool(processes=num_processes) as pool:
            list(tqdm(pool.imap(convert_h5_file, reproj_list), total=len(reproj_list)))