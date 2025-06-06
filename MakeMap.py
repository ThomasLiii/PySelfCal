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

'''
Naming convention:
sub = subframe, reprojected exposure inside the bounding box inside the reference frame
ref = reference frame, the mosaic
det = detector, the original detector frame
grid = grid, the oversampled grid of the subframe
off = offset, the exposure or detector offsets
{frame}_{name}: frame is the type of frame (sub, ref, det, grid), name describes the content
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
        with h5py.File(file_path, 'r') as file:
            for key in fields:
                if key == 'sub_wcs' or key == 'det_wcs': # WCS objects
                    header_key = 'sub_header' if key == 'sub_wcs' else 'det_header'
                    header_str = file[header_key][()].decode('utf-8')
                    data[key] = WCS(fits.Header.fromstring(header_str))
                else: # Numpy arrays
                    data[key] = np.array(file[key])
    except Exception as e:
        print(f"Error loading {file_path}: {e}. Will use placeholders.")
        is_file_missing = True # Treat as missing if other error occurs
        for key in fields:
            data[key] = None
    data['_is_missing_'] = is_file_missing # Add a flag
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

def compute_mean_map(ref_shape, reproj_file_list, exp_offset=None, det_offset=None, 
                     exp_idx_list=None, det_idx_list=None, apply_weight=True, apply_mask=True, 
                     chunk_map=None, chunk_valid_mask=None, max_workers=20):
    data_sum = np.zeros(ref_shape, dtype=np.float32)
    weight_sum = np.zeros(ref_shape, dtype=np.float32)
    batch_size = max_workers * 10
    total = len(reproj_file_list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            futures = {
                executor.submit(_prep_subframe, reproj_file_list[i], exp_offset, det_offset, 
                                exp_idx_list[i], det_idx_list[i], apply_weight, apply_mask, 
                                chunk_map, chunk_valid_mask): i 
                for i in range(batch_start, batch_end)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Computing mean map [{batch_start}/{total}]"):
                coords, data, weight, _ = future.result() # Ignore chunk_contrib
                if coords is None: continue # Skip if _prep_subframe failed
                
                sub_crop, ref_crop = compute_crop(ref_shape, coords)
                data_crop = data[sub_crop]
                weight_crop = weight[sub_crop]
                valid = ~np.isnan(data_crop)
                
                data_sum[ref_crop] += np.where(valid, data_crop * weight_crop, 0.0)
                weight_sum[ref_crop] += np.where(valid, weight_crop, 0.0)

                del coords, data, weight, _, sub_crop, ref_crop, data_crop, weight_crop, valid
                gc.collect()

    mean_map = np.divide(data_sum, weight_sum, out=np.zeros_like(data_sum), where=weight_sum != 0)
    return mean_map, weight_sum

def compute_std_map(mean_map, ref_shape, reproj_file_list, exp_offset=None, det_offset=None, 
                    exp_idx_list=None, det_idx_list=None, apply_weight=True, apply_mask=True, 
                    chunk_map=None, chunk_valid_mask=None, max_workers=20):
    sq_diff_sum = np.zeros(ref_shape, dtype=np.float32)
    weight_sum = np.zeros(ref_shape, dtype=np.float32)
    batch_size = max_workers * 10
    total = len(reproj_file_list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            futures = {
                executor.submit(_prep_subframe, reproj_file_list[i], exp_offset, det_offset,
                                exp_idx_list[i], det_idx_list[i], apply_weight, apply_mask,
                                chunk_map, chunk_valid_mask): i
                for i in range(batch_start, batch_end)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Computing Std [{batch_start}/{total}]"):
                coords, data, weight, _ = future.result() # Ignore chunk_contrib
                if coords is None: continue
                
                sub_crop, ref_crop = compute_crop(ref_shape, coords)
                
                valid = ~np.isnan(data[sub_crop])
                mean_crop = mean_map[ref_crop]
                
                sq_diff_sum[ref_crop] += np.where(valid, (data[sub_crop] - mean_crop)**2 * weight[sub_crop], 0.0)
                weight_sum[ref_crop] += np.where(valid, weight[sub_crop], 0.0) # Use actual weights sum

                del coords, data, weight, _, sub_crop, ref_crop, valid, mean_crop
                gc.collect()

    variance = np.divide(sq_diff_sum, weight_sum, out=np.zeros_like(sq_diff_sum), where=weight_sum > 0) # Prevent division by zero for variance
    return np.sqrt(variance), weight_sum


def compute_sc_mean(ref_shape, reproj_file_list, mean_map, std_map, sigma=3.0, 
                                exp_offset=None, det_offset=None, exp_idx_list=None, det_idx_list=None, 
                                apply_weight=True, apply_mask=True, chunk_map=None, chunk_valid_mask=None, max_workers=20):
    data_sum = np.zeros(ref_shape, dtype=np.float32)
    weight_sum = np.zeros(ref_shape, dtype=np.float32)
    batch_size = max_workers * 10
    total = len(reproj_file_list)

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        for batch_start in range(0, total, batch_size):
            batch_end = min(batch_start + batch_size, total)
            futures = {
                 executor.submit(_prep_subframe, reproj_file_list[i], exp_offset, det_offset,
                                exp_idx_list[i], det_idx_list[i], apply_weight, apply_mask,
                                chunk_map, chunk_valid_mask): i
                for i in range(batch_start, batch_end)
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Sigma-Clipped Coadd [{batch_start}/{total}]"):
                coords, data, weight, _ = future.result() # Ignore chunk_contrib
                if coords is None: continue

                sub_crop, ref_crop = compute_crop(ref_shape, coords)
                
                data_crop = data[sub_crop]
                mean_crop = mean_map[ref_crop]
                std_crop = std_map[ref_crop]
                
                clip_mask = np.abs(data_crop - mean_crop) <= sigma * std_crop # Use <= for inclusive range
                valid = (~np.isnan(data_crop)) & clip_mask
                
                data_sum[ref_crop] += np.where(valid, data_crop * weight[sub_crop], 0.0)
                weight_sum[ref_crop] += np.where(valid, weight[sub_crop], 0.0)

                del coords, data, weight, _, sub_crop, ref_crop, data_crop, mean_crop, std_crop, clip_mask, valid
                gc.collect()

    clipped_map = np.divide(data_sum, weight_sum, out=np.zeros_like(data_sum), where=weight_sum != 0)
    return clipped_map, weight_sum

def _prep_lsqr(i, reproj_file, ref_shape, exp_idx, det_idx, num_exp, num_det, num_chunks, 
               apply_mask, apply_weight, chunk_map, chunk_valid_mask):
    '''Compute the components of the LSQR matrix A and vector b for a single subframe.
    A.shape = (subframe_pixels, num_sky_pixels + num_det + num_chunks * num_det)
    b.shape = (subframe_pixels,)
    Solve for x which has x.shape = (num_sky_pixels + num_exp + num_det * num_chunks)
    Assumtions:
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
        sub_valid = np.ones_like(sub_data)#~np.isnan(sub_data)
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

        keep_data = ~np.isnan(sub_b[sub_rows])
        sub_rows = sub_rows[keep_data]
        sub_cols = sub_cols[keep_data]
        sub_data = sub_data[keep_data]
        sub_b[np.isnan(sub_b)] = 0
        num_rows = sub_h * sub_w
        return sub_rows, sub_cols, sub_data, sub_b, num_rows
    except Exception as e:
        print(f"Error processing file {reproj_file} for exp_idx={exp_idx}, det_idx={det_idx}: {e}")
        traceback.print_exc()
        return None


def setup_lsqr(reproj_file_list, ref_shape, exp_idx_list, det_idx_list,
               apply_mask, apply_weight, chunk_map, chunk_valid_mask,
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
                            apply_mask, apply_weight, chunk_map, chunk_valid_mask
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


def apply_lsqr(A, b, ref_shape, exp_idx_list, det_idx_list, chunk_map, x0=None, 
                atol=1e-05, btol=1e-05, damp=1e-2, iter_lim=100):
    num_chunks = len(np.unique(chunk_map))
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

class Mosaicker:
    def __init__(self,  exposure_list=[], reproj_list=[], config=None, data_dict=None):
        self.exposure_list = exposure_list
        self.reproj_list = reproj_list
        self.config = {
            'band': None,
            'n_chunk': None, 
            'det_width': None, 
            'num_detectors': None,
            'ref_reso': None,
            'grid_reso': None,
            'ref_shape': None,
            'ref_wcs': None,
            'sci_ext_list': None,
            'dq_ext_list': None,
            'out_dir': None,
            'run_name': None
            }
        if config is not None:
            self.config.update(config)
        # Convert padding pad_fraction to width in pixels
        self.config['ref_padding'] = int((1+config['pad_fraction']) * (np.sqrt(2)-1) * \
                                         config['det_width'] / (config['ref_reso']/config['det_reso']))
        # Define path for reference wcs and reprojected files
        self.config['wcs_path'] = os.path.join(config['out_dir'], config['run_name'], 'ref_frame.fits')
        self.config['reproj_path'] = os.path.join(config['out_dir'], config['run_name'], 'reprojected', config['band'])
        self.config['cal_path'] = os.path.join(config['out_dir'], config['run_name'], f'{config['band']}_cal.h5')
    
    def define_reference(self, new=False, use_ext=[1, 10, 37, 46]):
        if os.path.exists(self.config['wcs_path']) and not new:
            ref_wcs, ref_shape = load_from_fits(self.config['wcs_path'])
        else:
            ref_wcs, ref_shape = find_optimal_frame(
                exposure_list=self.exposure_list,
                resolution_arcsec=self.config['ref_reso'],
                padding_pixels=self.config['ref_padding'],
                use_ext=use_ext)
            save_to_fits(ref_wcs, ref_shape, self.config['wcs_path'])
        self.config['ref_wcs'] = ref_wcs
        self.config['ref_shape'] = ref_shape
        print(f'Reference frame loaded with shape {ref_shape}')
        print(ref_wcs)


    def run_reproject(self, apply_mask=True, ignore_flags = ['HIERARCH MSK_FLAG_DARKNODET', 'HIERARCH MSK_FLAG_NLINEAR'], 
                      method = 'adaptive', num_processes = 100):
        assert self.exposure_list, 'No exposure files loaded. Please load exposure files first.'
        assert self.config['ref_wcs'] is not None, 'Reference WCS not defined. Please define reference frame first.'
        reproj_list = batch_reproject(
            # Can edit
            apply_mask = apply_mask, 
            num_processes = num_processes,
            ignore_flags = ignore_flags,
            method = method,  # interp: fastest, adaptive: conserves flux

            # Porbably don't want to edit
            exposure_list = self.exposure_list,
            ref_wcs = self.config['ref_wcs'], 
            ref_shape = self.config['ref_shape'],
            output_dir = self.config['reproj_path'], 
            padding_percentage = self.config['pad_fraction'],
            grid_reso_arcsec = self.config['grid_reso'], 
            sci_ext_list = self.config['sci_ext_list'], 
            dq_ext_list = self.config['dq_ext_list'],
            )
        self.reproj_list = sorted(reproj_list)

    def setup_lsqr(self, reproj_list=[], clip_outlier=True, apply_weight=True, max_workers=20):
        if reproj_list:
            self.reproj_list = reproj_list
        A, b = setup_lsqr(
            ref_shape=self.config['ref_shape'], 
            clip_outlier=clip_outlier, 
            apply_weight=apply_weight, 
            n_chunk=self.config['n_chunk'],
            num_detectors=self.config['num_detectors'],
            reproj_file_list=self.reproj_list,
            max_workers=max_workers,
            )
        self.A = A
        self.b = b
        print(f'Setup LSQR completed')
        print(f'A has shape {A.shape} and {A.nnz} non-zero elements')
        print(f'b has shape {b.shape}')
    
    def solve_lsqr(self, x0=None, atol=1e-05, btol=1e-05, damp=1e-2, iter_lim=3000, save=True):
        assert self.A is not None, 'LSQR matrix A not defined. Please run setup_lsqr first.'
        O, S, D = apply_lsqr(
            A=self.A, 
            b=self.b, 
            ref_shape=self.config['ref_shape'], 
            num_frames=len(self.reproj_list), 
            n_chunk=self.config['n_chunk'], 
            x0=x0, 
            num_detectors=self.config['num_detectors'], 
            atol=atol,
            btol=btol, 
            damp=damp,
            iter_lim=iter_lim
            )
        self.O = O
        self.S = S
        self.D = D
        print(f'LSQR solved. O, S, D have shapes {O.shape}, {S.shape}, {D.shape}')
        print(f'O has mean {np.nanmean(O):.2f} and std {np.nanstd(O):.2f}')
        print(f'S has mean {np.nanmean(S):.2f} and std {np.nanstd(S):.2f}')
        print(f'D has mean {np.nanmean(D):.2f} and std {np.nanstd(D):.2f}')
        if save:
            with h5py.File(self.config['cal_path'], 'w') as f:
                f.create_dataset('O', data=O)
                f.create_dataset('S', data=S)
                f.create_dataset('D', data=D)
            print(f'LSQR results saved to {self.config['cal_path']}')

    def load_cal(self, cal_path=None):
        if cal_path is None:
            cal_path = self.config['cal_path']
        with h5py.File(cal_path, 'r') as f:
            self.O = f['O'][()]
            self.S = f['S'][()]
            self.D = f['D'][()]
        print(f'LSQR results loaded from {cal_path}')
        print(f'O has shape {self.O.shape}')
        print(f'S has shape {self.S.shape}')
        print(f'D has shape {self.D.shape}')
    
    def make_mosaic(self, apply_sigma_clip=False, apply_cal=False, norm_cal=True, sigma=3.0, apply_weight=True, max_workers=20):
        assert self.reproj_list, 'No reprojection files loaded. Please run run_reproject first.'
        assert self.config['ref_shape'] is not None, 'Reference shape not defined. Please define reference shape first.'
        O = None
        D = None
        if apply_cal:
            assert self.O is not None, 'Offsets not defined. Please run solve_lsqr first.'
            assert self.D is not None, 'Detector chunk offsets not defined. Please run solve_lsqr first.'
            O = self.O.copy()
            D = self.D.copy()
            if norm_cal:
                O -= np.mean(self.O)
                D -= np.mean(self.D, axis=0)
        mean_map, mean_weight = compute_mean_map(
            ref_shape=self.config['ref_shape'], 
            reproj_file_list=self.reproj_list, 
            det_width=self.config['det_width'], 
            exp_offset=O,
            det_offset=D,
            apply_weight=apply_weight, 
            max_workers=max_workers
        )
        if apply_sigma_clip:
            std_map, std_weight = compute_std_map(
                ref_shape=self.config['ref_shape'], 
                reproj_file_list=self.reproj_list, 
                det_width=self.config['det_width'], 
                mean_map=mean_map, 
                exp_offset=O, 
                det_offset=D, 
                apply_weight=apply_weight, 
                max_workers=max_workers
            )
            coadd, coadd_weight = sigma_clip_coadd(
                ref_shape=self.config['ref_shape'], 
                reproj_file_list=self.reproj_list, 
                det_width=self.config['det_width'], 
                mean_map=mean_map, 
                std_map=std_map, 
                sigma=sigma, 
                exp_offset=O, 
                det_offset=D, 
                apply_weight=apply_weight, 
                max_workers=max_workers
            )
        else:
            coadd = mean_map
            coadd_weight = mean_weight   
        return coadd, coadd_weight 

"""------------------------------------------------------------Legacy Code---------------------------------------------------------"""
# def estimate_memory_requirement(reproj_file_list, fields):
#     """Estimate memroy requirement to load a list of HDF5 files into memory with the given fields
#     Parameters
#     ----------
#     reproj_file_list : list or tup
#         List of path to HDF5 files
#     fields: tup
#         List of strings corresponding to name of dataset to extract from the HDF5 file

#     Returns
#     -------
#     mem_gb : float
#         Memory requirement in GB
#     """
#     sample_data = None
#     for file_path in reproj_file_list:
#         if file_path and os.path.exists(file_path): # Check if path is not None and exists
#             sample_data = load_reproj_file(file_path, fields)
#             if not sample_data.get('_is_missing_'): break # Found a valid sample
#         sample_data = None # Reset if loop finishes without break or if file_path was None

#     if sample_data is None or sample_data.get('_is_missing_'):
#         print('Could not load a sample file for memory estimation. Assuming 0.')
#         return 0 
    
#     size_per_file = 0
#     for key in fields: # Iterate through requested fields
#         item = sample_data.get(key) # Get item from sample
#         if item is not None:
#             if isinstance(item, np.ndarray): size_per_file += item.nbytes
#             elif isinstance(item, WCS): size_per_file += sys.getsizeof(str(item.to_header())) 
#             else: size_per_file += sys.getsizeof(item)
    
#     mem_gb = len(reproj_file_list) * size_per_file / (1024**3)
#     print(f'Estimated memory for {len(reproj_file_list)} files ({fields}): {mem_gb:.2f} GB')
#     return mem_gb

# def batch_load_reproj_data(reproj_file_list, 
#                                 fields=('sub_data', 'sub_wcs', 'det_wcs', 'ref_coords', 'grid_mapping'), 
#                                 max_workers=1, memory_limit=256):

#     print(f"Attempting to load {len(reproj_file_list)} HDF5 files with {max_workers} workers...")
#     memory_requ = estimate_memory_requirement(reproj_file_list, fields)
#     if memory_requ > memory_limit:
#         print(f'Warning: memory requirement ({memory_requ:.2f}GB) > memory limit ({memory_limit}GB) exceeded. Exiting.')
#         return
    
#     data_dict = {key: [None] * len(reproj_file_list) for key in fields}
    
#     # Try to load the first existing file to get shapes for pre-allocation
#     sample_data = None
#     sample_path = None
#     for file_path in reproj_file_list:
#         if os.path.exists(file_path):
#             sample_data = load_reproj_file(file_path, fields)
#             if not sample_data.get('_is_missing_'):
#                 sample_path = file_path
#                 break # Found a valid sample
#         sample_data = None # Reset if loop finishes

#     if sample_data:
#         for key in fields:
#             item = sample_data.get(key)
#             if isinstance(item, np.ndarray):
#                 shape = item.shape
#                 dtype = item.dtype
#                 try:
#                     data_dict[key] = np.empty((len(reproj_file_list), *shape), dtype=dtype)
#                 except MemoryError:
#                     print(f"MemoryError pre-allocating for {key}. Exiting.")
#                     return
#     else: 
#         print("Warning: Could not infer shapes from any HDF5 file. Exiting.")
#         return
    
#     num_success = 0
    
#     if max_workers > 1:
#         load_func = partial(load_reproj_file, fields=fields)
#         try:
#             with ProcessPoolExecutor(max_workers=max_workers) as executor:
#                 for i, data_item in enumerate(tqdm(executor.map(load_func, reproj_file_list), total=len(reproj_file_list), desc='Parallel loading')):
#                     if not data_item.get('_is_missing_'):
#                         num_success += 1
#                     for key in fields:
#                         data_dict[key][i] = data_item[key]
#                     del data_item  # Free memory immediately
#         except KeyboardInterrupt:
#             print("KeyboardInterrupt: cleaning memory")
#             gc.collect()
#             raise
#         except Exception as e:
#             print(f"An error occurred during parallel HDF5 loading: {e}")
#     else:
#         for i, file_path in enumerate(tqdm(reproj_file_list, desc='Sequential loading')):
#             data_item = load_reproj_file(file_path, fields=fields)
#             for key in fields:
#                 num_success += 1
#                 data_dict[key][i] = data_item[key]
    
#     print(f"Finished loading reprojected data. Successfully loaded {num_success} out of {len(reproj_file_list)} expected files.")
#     return data_dict

