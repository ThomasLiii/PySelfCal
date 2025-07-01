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

from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
import sys 
import gc 
from functools import partial

from MapUtility import bit_to_bool, make_weight, find_outliers, map_pixels, det_to_sub, compute_chunk_contrib, compute_crop
from WCSUtility import load_from_fits, save_to_fits, find_optimal_frame

def _reproject_task(task_args):
    """Individual tasks called by batch_reproject's multiprocessing instances"""
    # Unpacked arguments for clarity
    method, file_idx, file_path, det_idx, sci_ext, dq_ext, ref_wcs, sub_width, \
    det_center, apply_mask, output_dir, upgrade_factor, ignore_flags = task_args
    
    reproj_funcs = {'exact': reproject_exact, 'interp': reproject_interp, 'adaptive': reproject_adaptive}
    kwargs = {}
    if method == 'adaptive':
        kwargs = {'bad_value_mode': 'ignore', 'boundary_mode': 'ignore', 'conserve_flux': True}
    # Science extension index in the FITS file (0-indexed)
    # Assuming 3 HDUs per detector: SCI, ERR, DQ (or similar pattern)
    try:
        with fits.open(file_path) as hdul:
            det_data = hdul[sci_ext].data.astype(np.float32)
            det_header = hdul[sci_ext].header
            det_header_str = det_header.tostring().encode('utf-8') # For saving
            det_wcs = WCS(det_header)

            if apply_mask:
                bitmask_data = hdul[dq_ext].data
                bitmask_header = hdul[dq_ext].header
                # Define flags to ignore (i.e., pixels with these flags are BAD)
                valid_mask = bit_to_bool(bitmask_data, ignore_flags, bitmask_header, invert=True)
                # det_data[~valid_mask] = np.nan

            # Map detector center to world, then to reference frame pixels            
            det_center_coords = det_wcs.pixel_to_world_values(det_center[1], det_center[0])
            ref_det_center = np.array(ref_wcs.world_to_pixel_values(det_center_coords[0], det_center_coords[1]))
            
            # Define sub-frame boundaries in the reference frame
            x_min_ref = int(ref_det_center[0] - sub_width // 2)
            x_max_ref = x_min_ref + sub_width
            y_min_ref = int(ref_det_center[1] - sub_width // 2)
            y_max_ref = y_min_ref + sub_width

            # Create WCS for the sub-frame
            sub_wcs = ref_wcs.deepcopy()
            sub_wcs.wcs.crpix[0] -= x_min_ref # Adjust CRPIX for the sub-frame origin
            sub_wcs.wcs.crpix[1] -= y_min_ref
            sub_header_str = sub_wcs.to_header().tostring().encode('utf-8') # For saving
            
            # Perform reprojection
            sub_data, sub_foot = reproj_funcs[method](
                (det_data, det_wcs), 
                sub_wcs, 
                shape_out=(sub_width, sub_width), 
                **kwargs
            )

            sub_valid_mask, _ = reproj_funcs['interp'](
                (valid_mask, det_wcs), 
                sub_wcs, 
                shape_out=(sub_width, sub_width), 
                #order="nearest-neighbor"
            )
            sub_valid_mask = np.nan_to_num(np.where(sub_valid_mask<0.9, 0, 1), nan=0).astype(bool)
            # sub_data[~sub_valid_mask] = np.nan

            # Generate fine grid mapping if requested
            det_grid_x, det_grid_y = None, None
            if upgrade_factor is not None and upgrade_factor > 0:
                # Sample grid of index position in the subframe coordinate, sample per pixel width = upgrade_factor
                sub_grid_idx = np.arange(sub_width*upgrade_factor) / upgrade_factor
                sub_grid_x, sub_grid_y = np.meshgrid(sub_grid_idx, sub_grid_idx)
                # Compute index of grid in detector coordinate
                det_grid_x, det_grid_y = map_pixels(sub_wcs, det_wcs, sub_grid_x, sub_grid_y)

        # Save to HDF5
        # Filename uses overall exposure index (file_idx) and detector index within that exposure (det_idx)
        output_file = os.path.join(output_dir, f'exp_{file_idx:04d}_det_{det_idx:02d}.h5')
        
        with h5py.File(output_file, 'w') as hf:
            hf.create_dataset('sub_data', data=sub_data, compression='gzip')
            hf.create_dataset('det_data', data=det_data, compression='gzip') # Original detector data
            hf.create_dataset('sub_header', data=sub_header_str) # WCS of sub_data
            hf.create_dataset('det_header', data=det_header_str) # WCS of det_data
            hf.create_dataset('ref_coords', data=np.array([y_min_ref, y_max_ref, x_min_ref, x_max_ref], dtype=np.int32)) # Sub-frame location in full reference
            hf.create_dataset('sub_foot', data=sub_foot, compression='gzip') # Footprint of sub_data
            # hf.create_dataset('sub_valid_mask', data=sub_valid_mask, compression='gzip') # Footprint of sub_data
            if det_grid_x is not None and det_grid_y is not None:
                hf.create_dataset('det_mapping', data=np.stack((det_grid_x, det_grid_y), axis=0), compression='gzip')
            hf.create_dataset('file_path', data=file_path)
            hf.create_dataset('sub_valid_mask', data=sub_valid_mask, compression='gzip') # Valid mask for sub_data
        return output_file # Return path on success
    except Exception as e:
        print(f'Error processing detector {det_idx} from exposure file index {file_idx} ({file_path}): {e}')
        # import traceback; traceback.print_exc() # Uncomment for detailed debugging
        return None # Return None on failure

def batch_reproject(exposure_list, ref_wcs, ref_shape,
                    output_dir='output/', padding_percentage=0.05,
                    apply_mask=True, mapping_reso_arcsec=None, num_processes=1,
                    ignore_flags = ['HIERARCH MSK_FLAG_DARKNODET', 'HIERARCH MSK_FLAG_NLINEAR'],
                    sci_ext_list=[], dq_ext_list=[], method='interp'):
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
    apply_mask : bool, optional
        Extract bitmask from data quality extension defined by dq_ext_list and mask science data defined by sci_ext_list
    mapping_reso_arcsec: float, optional
        Define the resolution to map detector offset contribution to reference frame
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
    # Center of the original detector frame (in its own pixel coordinates)
    det_center = [det_data_0.shape[0] / 2.0, det_data_0.shape[1] / 2.0] 

    upgrade_factor = None
    if mapping_reso_arcsec is not None: # mapping_reso_arcsec is in arcsec/pixel
        ref_reso_arcsec = (ref_reso * u.degree).to(u.arcsec).value
        if abs(ref_reso_arcsec) <= abs(mapping_reso_arcsec): 
                print(f'Warning: Reference resolution ({ref_reso_arcsec:.2f}\') is not significantly larger than mapping resolution ({mapping_reso_arcsec:.2f}\'). Disabling fine mapping.')
        else:
            upgrade_factor = abs(ref_reso_arcsec / mapping_reso_arcsec) # Ratio of resolutions
            print(f'Pixel mapping upgrade factor: {upgrade_factor:.2f} (Ref reso: {ref_reso_arcsec:.2f}\', Map reso: {mapping_reso_arcsec:.2f}\')')

    tasks = []
    for file_idx, file_path in enumerate(exposure_list): # file_idx is the overall exposure index
        for det_idx, (sci_ext, dq_ext) in enumerate(zip(sci_ext_list, dq_ext_list)):
            tasks.append((
                method, file_idx, file_path, det_idx, sci_ext, dq_ext, 
                ref_wcs, sub_width, 
                det_center, apply_mask, output_dir, upgrade_factor, ignore_flags
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


def estimate_memory_requirement(reproj_file_list, fields):
    """Estimate memroy requirement to load a list of HDF5 files into memory with the given fields
    Parameters
    ----------
    reproj_file_list : list or tup
        List of path to HDF5 files
    fields: tup
        List of strings corresponding to name of dataset to extract from the HDF5 file

    Returns
    -------
    mem_gb : float
        Memory requirement in GB
    """
    sample_data = None
    for file_path in reproj_file_list:
        if file_path and os.path.exists(file_path): # Check if path is not None and exists
            sample_data = load_reproj_file(file_path, fields)
            if not sample_data.get('_is_missing_'): break # Found a valid sample
        sample_data = None # Reset if loop finishes without break or if file_path was None

    if sample_data is None or sample_data.get('_is_missing_'):
        print('Could not load a sample file for memory estimation. Assuming 0.')
        return 0 
    
    size_per_file = 0
    for key in fields: # Iterate through requested fields
        item = sample_data.get(key) # Get item from sample
        if item is not None:
            if isinstance(item, np.ndarray): size_per_file += item.nbytes
            elif isinstance(item, WCS): size_per_file += sys.getsizeof(str(item.to_header())) 
            else: size_per_file += sys.getsizeof(item)
    
    mem_gb = len(reproj_file_list) * size_per_file / (1024**3)
    print(f'Estimated memory for {len(reproj_file_list)} files ({fields}): {mem_gb:.2f} GB')
    return mem_gb

def batch_load_reproj_data(reproj_file_list, 
                                fields=('sub_data', 'sub_wcs', 'det_wcs', 'ref_coords', 'det_mapping'), 
                                max_workers=1, memory_limit=256):

    print(f"Attempting to load {len(reproj_file_list)} HDF5 files with {max_workers} workers...")
    memory_requ = estimate_memory_requirement(reproj_file_list, fields)
    if memory_requ > memory_limit:
        print(f'Warning: memory requirement ({memory_requ:.2f}GB) > memory limit ({memory_limit}GB) exceeded. Exiting.')
        return
    
    data_dict = {key: [None] * len(reproj_file_list) for key in fields}
    
    # Try to load the first existing file to get shapes for pre-allocation
    sample_data = None
    sample_path = None
    for file_path in reproj_file_list:
        if os.path.exists(file_path):
            sample_data = load_reproj_file(file_path, fields)
            if not sample_data.get('_is_missing_'):
                sample_path = file_path
                break # Found a valid sample
        sample_data = None # Reset if loop finishes

    if sample_data:
        for key in fields:
            item = sample_data.get(key)
            if isinstance(item, np.ndarray):
                shape = item.shape
                dtype = item.dtype
                try:
                    data_dict[key] = np.empty((len(reproj_file_list), *shape), dtype=dtype)
                except MemoryError:
                    print(f"MemoryError pre-allocating for {key}. Exiting.")
                    return
    else: 
        print("Warning: Could not infer shapes from any HDF5 file. Exiting.")
        return
    
    num_success = 0
    
    if max_workers > 1:
        load_func = partial(load_reproj_file, fields=fields)
        try:
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                for i, data_item in enumerate(tqdm(executor.map(load_func, reproj_file_list), total=len(reproj_file_list), desc='Parallel loading')):
                    if not data_item.get('_is_missing_'):
                        num_success += 1
                    for key in fields:
                        data_dict[key][i] = data_item[key]
                    del data_item  # Free memory immediately
        except KeyboardInterrupt:
            print("KeyboardInterrupt: cleaning memory")
            gc.collect()
            raise
        except Exception as e:
            print(f"An error occurred during parallel HDF5 loading: {e}")
    else:
        for i, file_path in enumerate(tqdm(reproj_file_list, desc='Sequential loading')):
            data_item = load_reproj_file(file_path, fields=fields)
            for key in fields:
                num_success += 1
                data_dict[key][i] = data_item[key]
    
    print(f"Finished loading reprojected data. Successfully loaded {num_success} out of {len(reproj_file_list)} expected files.")
    return data_dict

def _prep_coadd(file, det_width, exp_offset, det_offset, exp_idx, det_idx, 
                apply_weight=False, apply_mask=True, chunk_map=None, chunk_valid_mask=None):
    if apply_mask:
        fields=['sub_data', 'ref_coords', 'det_mapping', 'sub_valid_mask']
    else:
        fields=['sub_data', 'ref_coords', 'det_mapping']
    result = load_reproj_file(file, fields=fields)
    data = result['sub_data']
    coords = result['ref_coords']
    mapping = result['det_mapping']
    chunk_weight = 1
    if apply_mask:
        mask = result['sub_valid_mask']
        data[~mask] = np.nan

    if exp_offset is not None:
        data -= exp_offset[exp_idx]

    if chunk_map is not None:
        chunk_contrib = compute_chunk_contrib(
            det_mapping=mapping,
            chunk_map=chunk_map,
            sub_width=data.shape[-1],
            det_width=det_width,
            upgrade_factor= int(mapping.shape[-1] / data.shape[-1])
            )
        if det_offset is not None:
            sub_offset_flat = chunk_contrib.T @ det_offset[det_idx].flatten()  # shape: (sub_width**2,)
            sub_offset = sub_offset_flat.reshape(data.shape[-1], data.shape[-1])
            data -= sub_offset
        if chunk_valid_mask is not None:
            chunk_weight_flat = chunk_contrib.T @ chunk_valid_mask.flatten()  # shape: (sub_width**2,)
            chunk_weight = chunk_weight_flat.reshape(data.shape[-1], data.shape[-1])
            
    mask = ~np.isnan(data)
    weight = make_weight(data) if apply_weight else np.ones_like(data)
    weight *= chunk_weight
    data = np.nan_to_num(data, nan=0.0)
    return coords, data, mask, weight

def compute_mean_map(ref_shape, file_list, det_width, exp_offset=None, det_offset=None, exp_idx_list=None, det_idx_list=None, 
                     apply_weight=True, apply_mask=True, chunk_map=None, chunk_valid_mask=None, max_workers=20):
    data_sum = np.zeros(ref_shape, dtype=np.float32)
    weight_sum = np.zeros(ref_shape, dtype=np.float32)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_prep_coadd, file, det_width, exp_offset, det_offset, exp_idx, det_idx, 
                            apply_weight, apply_mask, chunk_map, chunk_valid_mask): 
            i for i, (file, exp_idx, det_idx) in enumerate(zip(file_list, exp_idx_list, det_idx_list))
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Mean"):
            coords, data, mask, weight = future.result()
            sub_crop, ref_crop = compute_crop(ref_shape, coords)
            data_sum[ref_crop] += data[sub_crop] * weight[sub_crop]
            weight_sum[ref_crop] += mask[sub_crop] * weight[sub_crop]

    mean = np.divide(data_sum, weight_sum, out=np.zeros_like(data_sum), where=weight_sum != 0)
    return mean, weight_sum

def compute_std_map(ref_shape, file_list, det_width, mean_map, exp_offset=None, det_offset=None, apply_weight=True, max_workers=20):
    var_sum = np.zeros(ref_shape, dtype=np.float32)
    weight_sum = np.zeros(ref_shape, dtype=np.float32)
    num_detectors = det_offset.shape[0] if det_offset is not None else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_prep_coadd, file, det_width, exp_offset, det_offset, i, num_detectors, apply_weight): i
            for i, file in enumerate(file_list)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Std"):
            coords, data, mask, weight = future.result()
            y_min, y_max, x_min, x_max = coords
            diff = data - mean_map[y_min:y_max, x_min:x_max]
            var_sum[y_min:y_max, x_min:x_max] += (diff**2) * weight
            weight_sum[y_min:y_max, x_min:x_max] += mask * weight

    std = np.sqrt(np.divide(var_sum, weight_sum, out=np.zeros_like(var_sum), where=weight_sum != 0))
    return std, weight_sum

def sigma_clip_coadd(ref_shape, file_list, det_width, mean_map, std_map, sigma=3.0, exp_offset=None, det_offset=None, apply_weight=True, max_workers=20):
    data_sum = np.zeros(ref_shape, dtype=np.float32)
    weight_sum = np.zeros(ref_shape, dtype=np.float32)
    num_detectors = det_offset.shape[0] if det_offset is not None else 1

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_prep_coadd, file, det_width, exp_offset, det_offset, i, num_detectors, apply_weight): i
            for i, file in enumerate(file_list)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Sigma-Clipped Coadd"):
            coords, data, mask, weight = future.result()
            y_min, y_max, x_min, x_max = coords
            mean = mean_map[y_min:y_max, x_min:x_max]
            std = std_map[y_min:y_max, x_min:x_max]
            diff = np.abs(data - mean)
            clip_mask = (diff <= sigma * std) & mask
            data_sum[y_min:y_max, x_min:x_max] += data * weight * clip_mask
            weight_sum[y_min:y_max, x_min:x_max] += weight * clip_mask

    coadd = np.divide(data_sum, weight_sum, out=np.zeros_like(data_sum), where=weight_sum != 0)
    return coadd, weight_sum

def _prep_lsqr(i, from_memory, sub_data, ref_coords, det_mapping, file_list, num_detectors,
               clip_outlier, apply_weight, footprint, ref_w, ref_h, sub_w, sub_h,
               num_pixels, num_frames, chunks_per_det, n_chunk, det_width):
    '''Prepare row, col, data, b, and count for one frame'''
    if from_memory:
        frame = sub_data[i]
        y_min, y_max, x_min, x_max = ref_coords[i]
        mapping = det_mapping[i]
    else:
        result = load_reproj_file(file_list[i], fields=('sub_data', 'ref_coords', 'det_mapping'))
        frame = result['sub_data']
        y_min, y_max, x_min, x_max = result['ref_coords']
        mapping = result['det_mapping']

    detector_idx = i % num_detectors

    if clip_outlier:
        submask = ~(np.isnan(frame) | find_outliers(frame, threshold=3))
    else:
        submask = ~np.isnan(frame)
    y_valid, x_valid = np.nonzero(submask)
    vals = frame[y_valid, x_valid]
    n = len(vals)
    if n == 0:
        return None

    if apply_weight:
        weight = make_weight(frame)
        sqrt_w = np.sqrt(weight[y_valid, x_valid])
    else:
        sqrt_w = np.ones_like(vals)

    if footprint is not None:
        footprint_values = footprint[y_min:y_max, x_min:x_max]
        inv_footprint = 1.0 / np.clip(footprint_values, 1e-3, None)
        sqrt_w *= inv_footprint[y_valid, x_valid]

    y_valid_global = y_valid + y_min
    x_valid_global = x_valid + x_min
    pix_idx = y_valid_global * ref_w + x_valid_global
    rows = np.arange(n)

    chunk_contrib_map = det_to_sub(
        det_mapping=mapping,
        n_chunk=n_chunk[0],
        det_width=det_width,
        sub_width=sub_w
    ).toarray()

    chunk_contrib_valid = chunk_contrib_map[:, y_valid * sub_w + x_valid]
    chunk_idx, pix_idx_valid = np.nonzero(chunk_contrib_valid)
    contrib_vals = chunk_contrib_valid[chunk_idx, pix_idx_valid]
    rows_full = pix_idx_valid
    global_chunk_idx = num_pixels + num_frames + detector_idx * chunks_per_det + chunk_idx

    return rows, pix_idx, sqrt_w, vals, i, rows_full, global_chunk_idx, contrib_vals

def setup_lsqr(ref_shape, sub_data=None, ref_coords=None, det_mapping=None, file_list=[], 
               num_detectors=16, det_width=2040, 
               footprint=None, clip_outlier=True, apply_weight=True,
               n_chunk=(10,10), max_workers=1):

    if (sub_data is not None) and (ref_coords is not None):
        from_memory = True
    elif file_list:
        from_memory = False
    else:
        raise ValueError("Provide either (sub_data and ref_coords) or file_list.")

    ref_h, ref_w = ref_shape
    num_pixels = ref_h * ref_w
    chunks_per_det = n_chunk[0] * n_chunk[1]
    total_chunks = num_detectors * chunks_per_det

    if from_memory:
        num_frames, sub_h, sub_w = sub_data.shape
    else:
        num_frames = len(file_list)
        sample = load_reproj_file(file_list[0], fields=('sub_data',))
        sub_h, sub_w = sample['sub_data'].shape

    total_unknowns = num_pixels + num_frames + total_chunks
    estimated_nonzero = num_frames * sub_h * sub_w * 3 + num_frames * 2 * 2 * sub_w * sub_h // num_detectors

    A_data = np.empty(estimated_nonzero, dtype=np.float32)
    A_rows = np.empty(estimated_nonzero, dtype=np.int64)
    A_cols = np.empty(estimated_nonzero, dtype=np.int64)
    b_vals = np.empty(estimated_nonzero // 3, dtype=np.float32)

    data_ptr = 0
    row_idx = 0

    if not from_memory and max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_prep_lsqr, i, from_memory, sub_data, ref_coords, det_mapping,
                                file_list, num_detectors, clip_outlier, apply_weight, footprint,
                                ref_w, ref_h, sub_w, sub_h, num_pixels, num_frames, chunks_per_det,
                                n_chunk, det_width): i
                for i in range(num_frames)
            }
            for future in tqdm(as_completed(futures), total=num_frames, desc="Building A, b (parallel)"):
                result = future.result()
                if result is None:
                    continue
                rows, pix_idx, sqrt_w, vals, frame_idx, rows_full, global_chunk_idx, contrib_vals = result
                n = len(vals)

                A_rows[data_ptr:data_ptr+n] = row_idx + rows
                A_cols[data_ptr:data_ptr+n] = pix_idx
                A_data[data_ptr:data_ptr+n] = sqrt_w
                data_ptr += n

                A_rows[data_ptr:data_ptr+n] = row_idx + rows
                A_cols[data_ptr:data_ptr+n] = num_pixels + frame_idx
                A_data[data_ptr:data_ptr+n] = sqrt_w
                data_ptr += n

                A_rows[data_ptr:data_ptr+len(contrib_vals)] = row_idx + rows_full
                A_cols[data_ptr:data_ptr+len(contrib_vals)] = global_chunk_idx
                A_data[data_ptr:data_ptr+len(contrib_vals)] = sqrt_w[rows_full] * contrib_vals
                data_ptr += len(contrib_vals)

                b_vals[row_idx:row_idx+n] = sqrt_w * vals
                row_idx += n
    else:
        for i in tqdm(range(num_frames), desc="Building A, b"):
            result = _prep_lsqr(i, from_memory, sub_data, ref_coords, det_mapping, file_list, num_detectors,
                                clip_outlier, apply_weight, footprint, ref_w, ref_h, sub_w, sub_h,
                                num_pixels, num_frames, chunks_per_det, n_chunk, det_width)
            if result is None:
                continue

            rows, pix_idx, sqrt_w, vals, frame_idx, rows_full, global_chunk_idx, contrib_vals = result
            n = len(vals)

            A_rows[data_ptr:data_ptr+n] = row_idx + rows
            A_cols[data_ptr:data_ptr+n] = pix_idx
            A_data[data_ptr:data_ptr+n] = sqrt_w
            data_ptr += n

            A_rows[data_ptr:data_ptr+n] = row_idx + rows
            A_cols[data_ptr:data_ptr+n] = num_pixels + frame_idx
            A_data[data_ptr:data_ptr+n] = sqrt_w
            data_ptr += n

            A_rows[data_ptr:data_ptr+len(contrib_vals)] = row_idx + rows_full
            A_cols[data_ptr:data_ptr+len(contrib_vals)] = global_chunk_idx
            A_data[data_ptr:data_ptr+len(contrib_vals)] = sqrt_w[rows_full] * contrib_vals
            data_ptr += len(contrib_vals)

            b_vals[row_idx:row_idx+n] = sqrt_w * vals
            row_idx += n

    A = coo_matrix((A_data[:data_ptr], (A_rows[:data_ptr], A_cols[:data_ptr])),
                   shape=(row_idx, total_unknowns))
    b = b_vals[:row_idx]
    return A, b

def apply_lsqr(A, b, ref_shape, num_frames, n_chunk=(10, 10), x0=None, 
               num_detectors=16, atol=1e-05, btol=1e-05, damp=1e-2, iter_lim=3000):
    '''
    Solve Ax = b where x = [S (sky), O (offsets), D (detector chunk offsets)].
    '''
    ref_h, ref_w = ref_shape
    num_pixels = ref_h * ref_w
    chunks_per_det = n_chunk[0] * n_chunk[1]
    
    total_chunks = num_detectors * chunks_per_det
    total_unknowns = num_pixels + num_frames + total_chunks

    print(f"Solving least squares for {total_unknowns} unknowns...")

    # Solve sparse least squares
    result = lsqr(A, b, x0=x0, show=True, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim)
    x = result[0]

    # Extract components
    S = x[:num_pixels].reshape(ref_shape)
    O = x[num_pixels:num_pixels + num_frames]
    D = x[num_pixels + num_frames:].reshape((num_detectors, n_chunk[0], n_chunk[1]))

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
            'mapping_reso': None,
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
            mapping_reso_arcsec = self.config['mapping_reso'], 
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
            file_list=self.reproj_list,
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
            file_list=self.reproj_list, 
            det_width=self.config['det_width'], 
            exp_offset=O,
            det_offset=D,
            apply_weight=apply_weight, 
            max_workers=max_workers
        )
        if apply_sigma_clip:
            std_map, std_weight = compute_std_map(
                ref_shape=self.config['ref_shape'], 
                file_list=self.reproj_list, 
                det_width=self.config['det_width'], 
                mean_map=mean_map, 
                exp_offset=O, 
                det_offset=D, 
                apply_weight=apply_weight, 
                max_workers=max_workers
            )
            coadd, coadd_weight = sigma_clip_coadd(
                ref_shape=self.config['ref_shape'], 
                file_list=self.reproj_list, 
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

# def _coadd_task(i, from_memory, sub_data, ref_coords, det_mapping, file_list,
#                  exp_offset, det_offset, det_width, num_detectors, apply_weight, extract_inter):
#     if from_memory:
#         data = sub_data[i]
#         coords = ref_coords[i]
#         mapping = det_mapping[i] if det_offset is not None else None
#     else:
#         file = file_list[i]
#         fields = ['sub_data', 'ref_coords']
#         mapping = None
#         if det_offset is not None and not extract_inter:
#             fields.append('det_mapping')
#         result = load_reproj_file(file, fields=fields)
#         data = result['sub_data']
#         coords = result['ref_coords']
#         if det_offset is not None:
#             if extract_inter:
#                 npy_file = file.replace('.h5', '_mapping.npy')
#                 mapping = np.load(npy_file, mmap_mode='r')
#             else:
#                 mapping = result['det_mapping']

#     y_min, y_max, x_min, x_max = coords
#     if y_max <= y_min or x_max <= x_min:
#         return None  # skip invalid

#     weight = np.ones_like(data, dtype=np.float32)
#     if apply_weight:
#         weight = make_weight(data)
#     if exp_offset is not None:
#         data -= exp_offset[i]
#     if det_offset is not None:
#         sub_off = det_to_sub(mapping, n_chunk=det_offset.shape[-1],
#                              sub_width=data.shape[-1], det_width=det_width,
#                              det_off=det_offset[i % num_detectors])
#         data -= sub_off

#     mask = ~np.isnan(data)
#     data = np.nan_to_num(data, nan=0.0)
#     weighted_data = data * weight
#     weighted_mask = mask * weight

#     return y_min, y_max, x_min, x_max, weighted_data, weighted_mask

# def coadd_sub_data(ref_shape, sub_data=None, ref_coords=None, det_mapping=None, file_list=None, det_width=2040, num_detectors=16,
#                    exp_offset=None, det_offset=None,
#                    apply_weight=False, extract_inter=False, max_workers=20):
#     """
#     Coadd reprojected subframes into a common reference frame.

#     Parameters
#     ----------
#     ref_shape : tuple
#         Shape of the reference frame (height, width).
#     sub_data : np.ndarray, optional
#         Array of shape (N, H, W) of reprojected subframes.
#     ref_coords : np.ndarray, optional
#         Array of shape (N, 4) of bounding boxes: [y_min, y_max, x_min, x_max].
#     file_list : list of str, optional
#         List of HDF5 files containing subframe data.
#     exp_offset : np.ndarray, optional
#         Exposure-level offsets to subtract per frame.
#     det_offset : np.ndarray, optional
#         Detector chunk offsets used with det_to_sub().
#     apply_weight : bool, optional
#         Whether to apply weights computed from data.
#     extract_inter : bool, optional
#         Whether to pre-extract det_mapping to .npy before coadding.
    
#     Returns
#     -------
#     tuple of np.ndarray
#         (coadded image, accumulated weight map)
#     """
#     data_sum = np.zeros(ref_shape, dtype=np.float32)
#     weight_sum = np.zeros(ref_shape, dtype=np.float32)

#     if (sub_data is not None) and (ref_coords is not None):
#         N_frames = len(sub_data)
#         from_memory = True
#     elif file_list is not None:
#         N_frames = len(file_list)
#         from_memory = False
#     else:
#         raise ValueError("Provide either (sub_data and ref_coords) or file_list.")

#     if extract_inter and not from_memory:
#         batch_extract_mappings_to_npy(file_list, max_workers=max_workers)
#     if exp_offset is not None:
#         exp_offset -= np.nanmean(exp_offset)

#     if isinstance(max_workers, int):
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             futures = {executor.submit(_coadd_task, i, from_memory, sub_data, ref_coords, det_mapping,
#                                     file_list, exp_offset, det_offset, det_width, num_detectors,
#                                     apply_weight, extract_inter): i for i in range(N_frames)}
#             for future in tqdm(as_completed(futures), total=N_frames, desc="Coadding reprojected data (parallel)"):
#                 result = future.result()
#                 if result is not None:
#                     y_min, y_max, x_min, x_max, weighted_data, weighted_mask = result
#                     data_sum[y_min:y_max, x_min:x_max] += weighted_data
#                     weight_sum[y_min:y_max, x_min:x_max] += weighted_mask
#     else:
#         for i in tqdm(range(N_frames), desc="Coadding reprojected data"):
#             y_min, y_max, x_min, x_max, weighted_data, weighted_mask = _coadd_task(
#                 i, from_memory, sub_data, ref_coords, det_mapping, file_list, exp_offset, det_offset, det_width, num_detectors, 
#                 apply_weight, extract_inter)

#             data_sum[y_min:y_max, x_min:x_max] += weighted_data
#             weight_sum[y_min:y_max, x_min:x_max] += weighted_mask

#     coadd_data = np.divide(data_sum, weight_sum, out=np.zeros_like(data_sum), where=weight_sum != 0)
#     return coadd_data, weight_sum

# def _extract_task(file_path):
#     npy_file_path = file_path.replace('.h5', '_mapping.npy')
#     if os.path.exists(npy_file_path):
#         return 
#     with h5py.File(file_path, 'r') as f:
#         if 'det_mapping' in f:
#             np.save(npy_file_path, np.array(f['det_mapping']))
#         else:
#             print(f"Warning: 'det_mapping' not found in {file_path}")

# def batch_extract_mappings_to_npy(reproj_file_list=None, max_workers=20):
#     files_to_use = reproj_file_list

#     # Only process if the HDF5 file exists AND the NPY doesn't
#     files_to_process = [f for f in files_to_use if os.path.exists(f) and not os.path.exists(f.replace('.h5', '_mapping.npy'))]
        
#     print(f"Extracting 'det_mapping' for {len(files_to_process)} files to .npy format...")
#     try:
#         with ProcessPoolExecutor(max_workers=max_workers) as executor:
#             futures = [executor.submit(_extract_task, file_path) for file_path in files_to_process]
#             for _ in tqdm(as_completed(futures), total=len(futures), desc="Extracting mappings to .npy"):
#                 pass
#     except KeyboardInterrupt:
#         print("KeyboardInterrupt: cleaning memory")
#         gc.collect()
#         raise
#     print("Finished extracting mappings.")