import os
import h5py
from tqdm import tqdm
from multiprocessing import Pool 
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u

from reproject import reproject_interp
from reproject import reproject_exact
from reproject import reproject_adaptive

from scipy.sparse import coo_matrix, vstack
from scipy.sparse.linalg import lsqr
import sys 
import gc 
from functools import partial

from MapUtility import bit_to_bool, make_weight, find_outliers, map_pixels, det_to_sub, compute_chunk_contrib, compute_crop, bin2d
from WCSUtility import load_from_fits, save_to_fits, find_optimal_frame, upscale_wcs
import traceback
import warnings

'''
Naming convention:
sub = subframe, reprojected exposure inside the bounding box inside the reference frame
ref = reference frame, the mosaic
det = detector, the original detector frame
grid = grid, the oversampled grid of the subframe
off = offset, the exposure or detector offsets
{frame}_{name}: frame is the type of frame (sub, ref, det, grid), name describes the content

#TODO:
- Modify _prep_lsqr to handle out of bounds subframes
- Modify apply_lsqr for multiple detectors
- Better error handling in all parallel functions
- Add more documentation and comments
- Accelerate _prep_subframe
- Adapt Mosaicker class to use the new functions
- Add functionality to handle multiple chunk maps and valid masks
'''

def _reproject_task(task_args):
    """Individual tasks called by batch_reproject's multiprocessing instances"""
    # Unpacked arguments for clarity
    method, file_path, exp_idx, det_idx, sci_ext, dq_ext, ref_wcs, sub_width, \
    output_dir, oversample_factor, replace_existing = task_args

    # Save to HDF5
    # Filename uses overall exposure index (file_idx) and detector index within that exposure (det_idx)
    output_file = os.path.join(output_dir, f'exp_{exp_idx:04d}_det_{det_idx:02d}.h5')
    if not replace_existing and os.path.exists(output_file):
        return output_file # Skip if file already exists and replace_existing is False

    reproj_funcs = {'exact': reproject_exact, 'interp': reproject_interp, 'adaptive': reproject_adaptive}
    kwargs = {}
    if method == 'adaptive':
        kwargs = {'bad_value_mode': 'ignore', 'boundary_mode': 'ignore', 'conserve_flux': True}

    try:
        with fits.open(file_path) as hdul:
            det_data = hdul[sci_ext].data.astype(np.float32)
            det_width = np.shape(det_data)[-1]
            det_header = hdul[sci_ext].header
            det_header_str = det_header.tostring().encode('utf-8')
            det_wcs = WCS(det_header)

            # Map detector center to world, then to reference frame pixels 
            det_center = [det_data.shape[0] / 2.0, det_data.shape[1] / 2.0]           
            skycoord = det_wcs.pixel_to_world(det_center[1], det_center[0])
            ref_det_center = np.array(ref_wcs.world_to_pixel(skycoord))
            
            # Define sub-frame boundaries in the reference frame
            ref_x_min = int(ref_det_center[0] - sub_width // 2)
            ref_x_max = ref_x_min + sub_width
            ref_y_min = int(ref_det_center[1] - sub_width // 2)
            ref_y_max = ref_y_min + sub_width

            # Create WCS for the sub-frame
            sub_wcs = ref_wcs.deepcopy()
            sub_wcs.wcs.crpix[0] -= ref_x_min # Adjust CRPIX for the sub-frame origin
            sub_wcs.wcs.crpix[1] -= ref_y_min
            sub_header_str = sub_wcs.to_header().tostring().encode('utf-8') 
            
            # Perform reprojection
            sub_data, sub_foot = reproj_funcs[method](
                (det_data, det_wcs), 
                sub_wcs, 
                shape_out=(sub_width, sub_width), 
                **kwargs
            )
            
            # Process detector auxiliary data
            bitmask_data = hdul[dq_ext].data
            # bitmask_header = hdul[dq_ext].header
            # valid_mask = bit_to_bool(bitmask_data, ignore_flags, bitmask_header, invert=True)

            det_x, det_y = np.meshgrid(np.arange(det_width), np.arange(det_width))

            det_aux = np.stack((bitmask_data, det_x, det_y), axis=0) # Stack valid mask and pixel coordinates
            grid_wcs = upscale_wcs(sub_wcs, oversample_factor)
            grid_width = sub_width * oversample_factor

            grid_aux, _ = reproj_funcs['interp'](
                (det_aux, det_wcs), 
                grid_wcs, 
                shape_out=(grid_width, grid_width), 
                order='nearest-neighbor'
            )

            grid_bitmask = grid_aux[0] # Valid mask in the grid
            grid_mapping = (grid_aux[1],grid_aux[2]) # X coordinates in the grid
        
        with h5py.File(output_file, 'w') as hf:
            hf.create_dataset('sub_data', data=sub_data, compression='gzip')
            hf.create_dataset('det_data', data=det_data, compression='gzip') # Original detector data
            hf.create_dataset('sub_header', data=sub_header_str) # WCS of sub_data
            hf.create_dataset('det_header', data=det_header_str) # WCS of det_data
            hf.create_dataset('ref_coords', data=np.array([ref_y_min, ref_y_max, ref_x_min, ref_x_max], dtype=np.int32)) # Sub-frame location in full reference
            hf.create_dataset('sub_foot', data=sub_foot, compression='gzip') # Footprint of sub_data
            hf.create_dataset('file_path', data=file_path)
            hf.create_dataset('grid_bitmask', data=grid_bitmask, compression='gzip') # Validity mask for grid
            hf.create_dataset('grid_mapping', data=grid_mapping, compression='gzip') # Pixel mapping in the grid
        return output_file # Return path on success
    except Exception as e:
        print(f'Error processing detector {det_idx} from exposure file index {exp_idx} ({file_path}): {e}')
        # import traceback; traceback.print_exc() # Uncomment for detailed debugging
        return None # Return None on failure

def batch_reproject(exposure_list, ref_wcs, ref_shape,
                    output_dir='output/', padding_percentage=0.05, oversample_factor=2, num_processes=1,
                    sci_ext_list=[], dq_ext_list=[], method='interp', exp_idx_list=None, det_idx_list=None, replace_existing=True):
    """Reproject individual exposures to bounding boxes in reference frame, output sored in HDF5 files.

    Parameters
    ----------
    exposure_list : list or tup
        List of strings describing path to fits files containing the exposures
    ref : object
        An astropy.wcs.WCS object that defines the WCS of the reference (mosaic) frame
    ref_shape : list or tup
        List of shape (2, ) defining the size of the mosaic
    output_dir : str, optional
        Directory for all selfcal outputs
    padding_percentage: float, optional
        Fraction of the mosaic width to pad
    num_processes : int, optional
        Number of parallel processes to use
    ignore_flags : list or tup
        List of strings describing the header keywords in data quality extension, flags corresponding listed will be ignored
    sci_ext_list : list or tup
        List of integers defining the extension in the fits files containing the science data
    dq_ext_list : list or tup
        List of integers defining the extension in the fits files containing the data quality bitmask
    method : str
        Method for reprojecting the science extensions
        - 'Exact': Slowest, conserves flux
        - 'Interp': Fastest, alter PSF profile, does not conserves flux
        - 'Adaptive': Faster then 'Exact', conserves flux

    Returns
    -------
    success_file : list
        List of path to the HDF5 files containing the reprojected data
    """
    assert oversample_factor >= 1, "Oversample factor must be >= 1"
    assert type(oversample_factor) is int, "Oversample factor must be an integer"
    if not exposure_list: raise ValueError('No exposure files loaded.')
    if ref_wcs is None or ref_shape is None: raise ValueError('Reference WCS and shape not defined.')

    os.makedirs(output_dir, exist_ok=True)
    print(f'Starting batch reprojection. Output will be saved to: {output_dir}')
    # Determine sub-frame width based on a sample detector frame
    try:
        with fits.open(exposure_list[0]) as hdul_sample:
            # Assuming first science extension is representative
            sci_ext_0 = 1 # Example: first detector's science HDU
            if sci_ext_0 >= len(hdul_sample):
                raise ValueError(f'Sample FITS {exposure_list[0]} does not have extension {sci_ext_0}')
            det_data_0 = hdul_sample[sci_ext_0].data
            det_wcs_0 = WCS(hdul_sample[sci_ext_0].header)
    except Exception as e:
        raise ValueError(f'Could not read sample FITS file {exposure_list[0]} to determine detector properties: {e}')

    ref_reso = np.abs(proj_plane_pixel_scales(ref_wcs)[0]) # Assuming square pixels
    det_reso = np.abs(proj_plane_pixel_scales(det_wcs_0)[0])
    
    reso_ratio = ref_reso / det_reso 
    # Calculate sub_width needed to contain the diagonal of the detector frame after reprojection, plus padding
    sub_width = int(np.ceil(np.sqrt(2) * np.max(det_data_0.shape) / reso_ratio * (1 + 2 * padding_percentage)))

    tasks = []
    for i, file_path in enumerate(exposure_list): # file_idx is the overall exposure index
        for j, (sci_ext, dq_ext) in enumerate(zip(sci_ext_list, dq_ext_list)):
            exp_idx = exp_idx_list[i] if exp_idx_list is not None else i
            det_idx = det_idx_list[j] if det_idx_list is not None else j
            tasks.append((
                method, file_path, exp_idx, det_idx, sci_ext, dq_ext, 
                ref_wcs, sub_width, 
                output_dir, oversample_factor, replace_existing
            ))
    
    results = []
    if num_processes > 1 and len(tasks) > 0 : # Ensure there are tasks for multiprocessing
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap_unordered(_reproject_task, tasks), total=len(tasks), desc='Reprojecting frames'))
    elif len(tasks) > 0: # Sequential execution
        for task in tqdm(tasks, desc='Reprojecting frames (sequentially)'):
            results.append(_reproject_task(task))
    
    success_file = [r for r in results if r is not None]
    print(f'Batch reprojection completed. {len(success_file)} frames successfully processed out of {len(tasks)}.')
    return success_file


def load_reproj_file(file_path, fields):
    """Helper to load selected fields from a single HDF5 file.
    Parameters
    ----------
    file_path : str
        Path to a reprojected HDF5 file
    fields: tup
        List of strings corresponding to name of dataset to extract from the HDF5 file

    Returns
    -------
    data : dict
        Dictionary containing the extracted data, key is the fields and value is the corresponding datas
    """

    data = {}
    is_file_missing = False
    try:
        with h5py.File(file_path, 'r', libver='latest', swmr=True) as file:
            for key in fields:
                if key in ('sub_wcs', 'det_wcs'):
                    header_key = 'sub_header' if key == 'sub_wcs' else 'det_header'
                    header_str = file[header_key][()].decode('utf-8')
                    data[key] = WCS(fits.Header.fromstring(header_str))
                else:
                    data[key] = file[key][()]  # Efficient read
    except Exception as e:
        print(f"Error loading {file_path}: {e}. Will use placeholders.")
        is_file_missing = True
        for key in fields:
            data[key] = None
    data['_is_missing_'] = is_file_missing
    return data

def grid_bitmask_to_sub_mask(bitmask, oversample_factor, ignore_list=[], valid_threshold=0.99):
    valid_bit = ~np.isnan(bitmask)
    grid_mask = np.zeros_like(bitmask, dtype=bool)
    grid_mask[valid_bit] = bit_to_bool(bitmask[valid_bit].astype(np.uint32), ignore_list, invert=True) # 1 = Good pixel, 0 = Bad pixel
    sub_mask_float = bin2d(grid_mask, oversample_factor) # Downscale to sub-frame size
    sub_mask = np.where(sub_mask_float > valid_threshold, 1, 0).astype(bool)
    return sub_mask # Boolean mask for the sub-frame, True = Good pixel, False = Bad pixel

def _prep_subframe(file, exp_offset, det_offset, exp_idx, det_idx, 
                apply_weight, apply_mask, chunk_map, chunk_valid_mask):
    """Prepares data from a single file for co-addition or lsqr."""
    fields=['sub_data', 'ref_coords', 'grid_mapping']
    if apply_mask:
        fields.append('grid_bitmask') # More like grid validity weight
    result = load_reproj_file(file, fields=fields)

    data = result['sub_data']
    coords = result['ref_coords']
    grid = result['grid_mapping']
    # Ensure grid is not None before trying to access its shape
    oversample_factor = int(grid.shape[-1] / data.shape[-1]) if grid is not None and data.shape[-1] > 0 else 1

    # Apply mask
    if apply_mask:
        bitmask = result['grid_bitmask']
        sub_mask = grid_bitmask_to_sub_mask(bitmask, oversample_factor, ignore_list=[17, 21], valid_threshold=0.99)
        data[~sub_mask] = np.nan

    # Apply exposure offset if provided
    if exp_offset is not None:
        data -= exp_offset[exp_idx]

    # Compute chunk contribution to the sub-frame
    chunk_contrib = None # Initialize chunk_contrib
    if chunk_map is not None and grid is not None: # Check grid is not None
        chunk_contrib = compute_chunk_contrib(
            grid_mapping=grid,
            chunk_map=chunk_map,
            oversample_factor=oversample_factor
        )

    # Apply detector offset ONLY IF det_offset AND chunk_contrib are available
    if det_offset is not None and chunk_contrib is not None:
        sub_offset_flat = chunk_contrib.T @ det_offset[det_idx].flatten()
        sub_offset = sub_offset_flat.reshape(data.shape) # Use data.shape for consistency
        data -= sub_offset

    # If chunk_valid_mask is provided, compute the contribution weight
    chunk_weight = 1.0
    if chunk_valid_mask is not None and chunk_contrib is not None:
        chunk_weight_flat = chunk_contrib.T @ chunk_valid_mask.flatten()
        chunk_weight = chunk_weight_flat.reshape(data.shape) # Use data.shape
    
    weight = make_weight(data) if apply_weight else np.ones_like(data, dtype=np.float32)
    weight *= chunk_weight # chunk_weight is 1.0 if not modified
    
    return coords, data, weight, chunk_contrib

def _compute_chunk_worker(args):
    """Worker function for Pool processing that computes partial sums for a chunk of files"""
    chunk_files, chunk_indices, exp_offset, det_offset, exp_idx_list, det_idx_list, \
    apply_weight, apply_mask, chunk_map, chunk_valid_mask, ref_shape, operation_type, \
    mean_map, std_map, sigma = args
    
    data_sum = np.zeros(ref_shape, dtype=np.float32)
    weight_sum = np.zeros(ref_shape, dtype=np.float32)
    
    for i, file_path in enumerate(chunk_files):
        idx = chunk_indices[i]
        coords, data, weight, _ = _prep_subframe(
            file_path, exp_offset, det_offset, 
            exp_idx_list[idx], det_idx_list[idx], 
            apply_weight, apply_mask, 
            chunk_map, chunk_valid_mask
        )
        
        if coords is None:
            continue
            
        sub_crop, ref_crop = compute_crop(ref_shape, coords)
        data_crop = data[sub_crop]
        weight_crop = weight[sub_crop]
        valid = ~np.isnan(data_crop)
        
        if operation_type == 'mean':
            data_sum[ref_crop] += np.where(valid, data_crop * weight_crop, 0.0)
            weight_sum[ref_crop] += np.where(valid, weight_crop, 0.0)
        elif operation_type == 'std':
            mean_crop = mean_map[ref_crop]
            data_sum[ref_crop] += np.where(valid, (data_crop - mean_crop)**2 * weight_crop, 0.0)
            weight_sum[ref_crop] += np.where(valid, weight_crop, 0.0)
        elif operation_type == 'sigma_clip':
            mean_crop = mean_map[ref_crop]
            std_crop = std_map[ref_crop]
            clip_mask = np.abs(data_crop - mean_crop) <= sigma * std_crop
            valid_clipped = valid & clip_mask
            data_sum[ref_crop] += np.where(valid_clipped, data_crop * weight_crop, 0.0)
            weight_sum[ref_crop] += np.where(valid_clipped, weight_crop, 0.0)
    
    return data_sum, weight_sum

def compute_mean_map(ref_shape, reproj_file_list, exp_offset=None, det_offset=None, 
                     exp_idx_list=None, det_idx_list=None, apply_weight=True, apply_mask=True, 
                     chunk_map=None, chunk_valid_mask=None, max_workers=10):
    
    total = len(reproj_file_list)
    
    # Split files equally among workers
    chunk_size = (total + max_workers - 1) // max_workers  # Ceiling division
    chunks = []
    
    for i in range(max_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total)
        if start_idx < total:
            chunk_files = reproj_file_list[start_idx:end_idx]
            chunk_indices = list(range(start_idx, end_idx))
            chunks.append((chunk_files, chunk_indices, exp_offset, det_offset, 
                          exp_idx_list, det_idx_list, apply_weight, apply_mask, 
                          chunk_map, chunk_valid_mask, ref_shape, 'mean', None, None, None))
    
    # Process chunks in parallel using Pool
    with Pool(processes=max_workers) as pool:
        results = list(tqdm(pool.imap(_compute_chunk_worker, chunks), 
                           total=len(chunks), desc="Computing mean map"))
    
    # Combine results from all workers
    data_sum = np.zeros(ref_shape, dtype=np.float32)
    weight_sum = np.zeros(ref_shape, dtype=np.float32)
    
    for chunk_data_sum, chunk_weight_sum in results:
        data_sum += chunk_data_sum
        weight_sum += chunk_weight_sum

    mean_map = np.divide(data_sum, weight_sum, out=np.zeros_like(data_sum), where=weight_sum != 0)
    return mean_map, weight_sum

def compute_std_map(mean_map, ref_shape, reproj_file_list, exp_offset=None, det_offset=None, 
                    exp_idx_list=None, det_idx_list=None, apply_weight=True, apply_mask=True, 
                    chunk_map=None, chunk_valid_mask=None, max_workers=10):
    
    total = len(reproj_file_list)
    
    # Split files equally among workers
    chunk_size = (total + max_workers - 1) // max_workers  # Ceiling division
    chunks = []
    
    for i in range(max_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total)
        if start_idx < total:
            chunk_files = reproj_file_list[start_idx:end_idx]
            chunk_indices = list(range(start_idx, end_idx))
            chunks.append((chunk_files, chunk_indices, exp_offset, det_offset, 
                          exp_idx_list, det_idx_list, apply_weight, apply_mask, 
                          chunk_map, chunk_valid_mask, ref_shape, 'std', mean_map, None, None))
    
    # Process chunks in parallel using Pool
    with Pool(processes=max_workers) as pool:
        results = list(tqdm(pool.imap(_compute_chunk_worker, chunks), 
                           total=len(chunks), desc="Computing std map"))
    
    # Combine results from all workers
    sq_diff_sum = np.zeros(ref_shape, dtype=np.float32)
    weight_sum = np.zeros(ref_shape, dtype=np.float32)
    
    for chunk_sq_diff_sum, chunk_weight_sum in results:
        sq_diff_sum += chunk_sq_diff_sum
        weight_sum += chunk_weight_sum

    variance = np.divide(sq_diff_sum, weight_sum, out=np.zeros_like(sq_diff_sum), where=weight_sum > 0)
    return np.sqrt(variance), weight_sum

def compute_sc_mean(ref_shape, reproj_file_list, mean_map, std_map, sigma=3.0, 
                    exp_offset=None, det_offset=None, exp_idx_list=None, det_idx_list=None, 
                    apply_weight=True, apply_mask=True, chunk_map=None, chunk_valid_mask=None, max_workers=10):
    
    total = len(reproj_file_list)
    
    # Split files equally among workers
    chunk_size = (total + max_workers - 1) // max_workers  # Ceiling division
    chunks = []
    
    for i in range(max_workers):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, total)
        if start_idx < total:
            chunk_files = reproj_file_list[start_idx:end_idx]
            chunk_indices = list(range(start_idx, end_idx))
            chunks.append((chunk_files, chunk_indices, exp_offset, det_offset, 
                          exp_idx_list, det_idx_list, apply_weight, apply_mask, 
                          chunk_map, chunk_valid_mask, ref_shape, 'sigma_clip', mean_map, std_map, sigma))
    
    # Process chunks in parallel using Pool
    with Pool(processes=max_workers) as pool:
        results = list(tqdm(pool.imap(_compute_chunk_worker, chunks), 
                           total=len(chunks), desc="Sigma-clipped coadd"))
    
    # Combine results from all workers
    data_sum = np.zeros(ref_shape, dtype=np.float32)
    weight_sum = np.zeros(ref_shape, dtype=np.float32)
    
    for chunk_data_sum, chunk_weight_sum in results:
        data_sum += chunk_data_sum
        weight_sum += chunk_weight_sum

    clipped_map = np.divide(data_sum, weight_sum, out=np.zeros_like(data_sum), where=weight_sum != 0)
    return clipped_map, weight_sum

def _prep_coverage(file, exp_offset, det_offset, exp_idx, det_idx, 
                chunk_map, chunk_valid_mask):
    fields = ['sub_data', 'ref_coords', 'grid_mapping']
    result = load_reproj_file(file, fields=fields)

    data = result['sub_data']
    coords = result['ref_coords']
    grid = result['grid_mapping']
    oversample_factor = int(grid.shape[-1] / data.shape[-1])

    chunk_contrib = compute_chunk_contrib(
        grid_mapping=grid,
        chunk_map=chunk_map,
        oversample_factor=oversample_factor
    )

    chunk_weight = 1.0
    if chunk_valid_mask is not None:
        chunk_weight_flat = chunk_contrib.T @ chunk_valid_mask.flatten()
        chunk_weight = chunk_weight_flat.reshape(data.shape) # Use data.shape

    if exp_offset is not None or det_offset is not None:
        cover = np.zeros_like(data)
        if exp_offset is not None:
            cover -= exp_offset[exp_idx]
        if det_offset is not None:
            sub_offset_flat = chunk_contrib.T @ det_offset[det_idx].flatten()
            sub_offset = sub_offset_flat.reshape(data.shape) # Use data.shape for consistency
            cover -= sub_offset
    else:
        cover = np.ones_like(data)

    cover *= chunk_weight # chunk_weight is 1.0 if not modified
    
    return coords, cover

def compute_coverage_map(ref_shape, reproj_file_list, exp_offset=None, det_offset=None, 
                     exp_idx_list=None, det_idx_list=None, 
                     chunk_map=None, chunk_valid_mask=None, max_workers=20):
    coverage_map = np.zeros(ref_shape, dtype=np.float32)
    batch_size = max_workers * 10
    total = len(reproj_file_list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            futures = {
                executor.submit(_prep_coverage, reproj_file_list[i], exp_offset, det_offset, exp_idx_list[i], det_idx_list[i], 
                chunk_map, chunk_valid_mask): i 
                for i in range(batch_start, batch_end)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Computing mean map [{batch_start}/{total}]"):
                coords, cover = future.result() # Ignore chunk_contrib
                if coords is None: continue # Skip if _prep_subframe failed
    
                sub_crop, ref_crop = compute_crop(ref_shape, coords)
                coverage_map[ref_crop] += cover[sub_crop]

                del coords, cover, sub_crop, ref_crop
                gc.collect()

    return coverage_map

def _prep_lsqr(i, reproj_file, ref_shape, exp_idx, det_idx, num_exp, num_det, num_chunks, 
               apply_mask, apply_weight, chunk_map, chunk_valid_mask, outlier_thresh):
    '''Compute the components of the LSQR matrix A and vector b for a single subframe.
    A.shape = (subframe_pixels, num_sky_pixels + num_det + num_chunks * num_det)
    b.shape = (subframe_pixels,)
    Solve for x which has x.shape = (num_sky_pixels + num_exp + num_det * num_chunks)
    Assumptions:
    - Each pixel value in sub_data corresponds to a single sky pixel in the reference frame.
    - Each subframe comes from a single exposure and a single detector.
    '''
    try:
        ref_coords, sub_data, sub_weight, chunk_contrib = _prep_subframe(
            file=reproj_file,
            exp_offset=None,  # These are being solved for
            det_offset=None,
            exp_idx=None,
            det_idx=None,
            apply_weight=apply_weight,
            apply_mask=apply_mask,
            chunk_map=chunk_map,
            chunk_valid_mask=chunk_valid_mask
        )

        chunk_contrib = chunk_contrib.tocsr()

        ref_h, ref_w = ref_shape
        sub_h, sub_w = sub_data.shape
        num_sky = ref_h * ref_w

        # Identify valid pixels after _prep_subframe has applied its masking
        sub_valid = ~np.isnan(sub_data) # Boolean mask for valid pixels in sub_data
        if isinstance(outlier_thresh, (int, float)) and outlier_thresh > 0:
            sub_out = find_outliers(sub_data, threshold=outlier_thresh) # Find outliers in the sub_data
            sub_valid &= ~sub_out # Combine valid mask with outlier mask
        valid_sub_coords = np.nonzero(sub_valid) # Coordinates within sub_data, flat
        sub_pix_indices = valid_sub_coords[0] * sub_w + valid_sub_coords[1]
        valid_vals = sub_data[valid_sub_coords] # Values at valid coordinates, flat
        valid_weight = sub_weight[valid_sub_coords] # Weights at valid coordinates, flat
        num_valid_pixels = valid_vals.shape[0]

        if num_valid_pixels == 0:
            print(f"No valid pixels found in subframe {i} from file {reproj_file}. Skipping.")
            return np.array([]), np.array([]), np.array([]), np.array([])

        ref_pix_indices = (valid_sub_coords[0] + ref_coords[0]) * ref_w + (valid_sub_coords[1] + ref_coords[2]) # Convert to flat indices in the reference frame    

        # Component S: Sky pixel indices
        S_rows = np.arange(num_valid_pixels)  # Rows for this frame's equations
        S_cols = ref_pix_indices  # Sky pixel indices
        S_data = valid_weight  # Weights for sky pixels

        # Component O: Frame offsets
        O_rows = np.arange(num_valid_pixels)  # Rows for this frame's equations
        O_cols = np.full(num_valid_pixels, exp_idx) + num_sky # Frame offset index
        O_data = valid_weight  # Weights for frame offsets

        # Component D: Detector chunks
        chunk_idx, sub_idx = chunk_contrib[:, sub_pix_indices].nonzero() # Get chunk indices and subframe indices
        chunk_vals = chunk_contrib[(chunk_idx, sub_idx)].A[0]
        D_rows = sub_idx  # Local rows for this frame's equations
        D_cols = chunk_idx + num_sky + num_exp + det_idx * num_chunks # Global column indices for detector chunks
        D_data = valid_weight[sub_idx] * chunk_vals  # Weights for detector chunks

        sub_b = valid_vals * valid_weight  # Vector b for this subframe

        # Concatenate datapoints for S, O, D
        sub_rows = np.concatenate([S_rows, O_rows, D_rows])
        sub_cols = np.concatenate([S_cols, O_cols, D_cols])
        sub_data = np.concatenate([S_data, O_data, D_data])

        # Remove rows where sub_b is NaN or both sub_data and sub_b are zero, and reindex rows and sub_b accordingly
        # First, get the mask for valid rows: not nan, and not both zero
        valid_mask = ~np.isnan(sub_b[sub_rows]) & ~((sub_data == 0) & (sub_b[sub_rows] == 0))
        sub_rows = sub_rows[valid_mask]
        sub_cols = sub_cols[valid_mask]
        sub_data = sub_data[valid_mask]

        # Find which unique rows remain and create a mapping to new row indices
        unique_rows, new_row_indices = np.unique(sub_rows, return_inverse=True)
        sub_rows = new_row_indices

        # Filter sub_b to only the kept rows, in the order of unique_rows
        sub_b = sub_b[unique_rows]
        num_rows = len(sub_b)
        return sub_rows, sub_cols, sub_data, sub_b, num_rows
    except Exception as e:
        print(f"Error processing file {reproj_file} for exp_idx={exp_idx}, det_idx={det_idx}: {e}")
        traceback.print_exc()
        return None


def setup_lsqr(reproj_file_list, ref_shape, exp_idx_list, det_idx_list,
               apply_mask, apply_weight, chunk_map, chunk_valid_mask, outlier_thresh=3,
               max_workers=20):

    if not reproj_file_list:
        raise ValueError("reproj_file_list must be provided.")
    if not (len(reproj_file_list) == len(exp_idx_list) == len(det_idx_list)):
        raise ValueError("reproj_file_list, exp_idx_list, and det_idx_list must have the same length.")
    
    num_chunks = len(np.unique(chunk_map))
    num_exp = len(np.unique(exp_idx_list))
    num_det = len(np.unique(det_idx_list))
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    total_cols = num_sky + num_exp + num_det * num_chunks

    # Prepare lists to collect all data for COO matrix
    all_rows = []
    all_cols = []
    all_data = []
    all_b = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_prep_lsqr, i, 
                            reproj_file, ref_shape, exp_idx, det_idx, num_exp, num_det, num_chunks, 
                            apply_mask, apply_weight, chunk_map, chunk_valid_mask, outlier_thresh
                            ): i
            for i, (reproj_file, exp_idx, det_idx) in enumerate(zip(reproj_file_list, exp_idx_list, det_idx_list))
        }
        row_offset = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building A, b matrix"):
            result = future.result()
            if result is None:
                # End all parallel processing if any task fails
                print("A task failed. Ending all parallel processing.")
                # Explicityly end all workers
                executor.shutdown(wait=True)
                return None, None
            sub_rows, sub_cols, sub_data, sub_b, num_rows = result
            if len(sub_b) == 0:
                continue  # Skip frames with no valid data
            # Concatenate rows, cols, data directly
            all_rows.append(sub_rows + row_offset)
            all_cols.append(sub_cols)
            all_data.append(sub_data)
            # b vector
            all_b.append(sub_b)
            row_offset += num_rows

    if len(all_b) == 0:
        print("No valid data found in any subframe.")
        return None, None

    # Concatenate all arrays
    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    data = np.concatenate(all_data)
    b = np.concatenate(all_b)

    full_A = coo_matrix((data, (rows, cols)), shape=(row_offset, total_cols))
    full_b = b
    return full_A, full_b


def apply_lsqr(A, b, ref_shape, exp_idx_list, det_idx_list, x0=None, 
                atol=1e-05, btol=1e-05, damp=1e-2, iter_lim=100):
    num_exp = len(np.unique(exp_idx_list))
    num_det = len(np.unique(det_idx_list))
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w

    print(f"Solving least squares for {A.shape[1]} unknowns with {A.shape[0]} equations.")
    result = lsqr(A, b, x0=x0, show=True, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim)
    x = result[0]

    S = x[:num_sky].reshape(ref_shape)
    O = x[num_sky : num_sky + num_exp]
    D = x[num_sky + num_exp:]

    return O, S, D
