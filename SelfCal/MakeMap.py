import os
import h5py
import hdf5plugin
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing.shared_memory import SharedMemory
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

from .MapHelper import bit_to_bool, bool_to_bit, make_weight, find_outliers, map_pixels, compute_chunk_contrib, compute_crop, \
    bin2d_cv, chunk_to_det, check_invalid, det_to_sub, make_linear_interp_matrix
from .WCSHelper import load_from_fits, save_to_fits, find_optimal_frame, upscale_wcs
import traceback
import warnings

'''
Naming convention:
sub = subframe, reprojected exposure inside the bounding box inside the reference frame
ref = reference frame, the mosaic
det = detector, the original detector frame
off = offset, the exposure or detector offsets
{frame}_{name}: frame is the type of frame (sub, ref, det), name describes the content

#TODO:
- Better error handling in all parallel functions
- Add functionality to handle multiple chunk maps and valid masks
'''

def _reproject_worker(task_params):
    """Individual tasks called by batch_reproject's multiprocessing instances"""
    # Unpacked arguments for clarity
    reproj_func = task_params['reproj_func']
    file_path = task_params['file_path']
    exp_idx = task_params['exp_idx']
    det_idx = task_params['det_idx']
    sci_ext = task_params['sci_ext']
    dq_ext = task_params['dq_ext']
    ref_wcs = task_params['ref_wcs']
    sub_width = task_params['sub_width']
    output_dir = task_params['output_dir']
    replace_existing = task_params['replace_existing']
    reproject_kwargs = task_params['reproject_kwargs']

    # Save to HDF5
    output_file = os.path.join(output_dir, f'exp_{exp_idx:04d}_det_{det_idx:02d}.h5')
    if not replace_existing and os.path.exists(output_file):
        return output_file # Skip if file already exists and replace_existing is False

    reproj_func_dict = {'exact': reproject_exact, 'interp': reproject_interp, 'adaptive': reproject_adaptive}

    try:
        with fits.open(file_path) as hdul:
            det_data = hdul[sci_ext].data
            det_header = hdul[sci_ext].header
            det_bitmask = hdul[dq_ext].data
        det_width = np.shape(det_data)[-1]
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
        sub_data, sub_foot = reproj_func_dict[reproj_func](
            (det_data, det_wcs), 
            sub_wcs, 
            shape_out=(sub_width, sub_width), 
            **reproject_kwargs
        )
        
        # Process detector auxiliary data
        det_expanded_mask = bit_to_bool(det_bitmask, expand_bits=True)
        det_xmesh, det_ymesh = np.meshgrid(np.arange(det_width), np.arange(det_width))
        det_aux = np.stack((det_xmesh, det_ymesh, *det_expanded_mask), axis=0)
        sub_aux, _ = reproject_interp(
            (det_aux, det_wcs), 
            sub_wcs, 
            shape_out=(sub_width, sub_width), 
            order='bilinear',
        )
        sub_mapping = sub_aux[0:2] # x, y
        sub_expanded_mask_float = sub_aux[2:]
        sub_expanded_mask_bool = sub_expanded_mask_float > 0.01
        sub_bitmask = bool_to_bit(sub_expanded_mask_bool)
        
        with h5py.File(output_file, 'w', libver='latest') as hf:
    
            # CONFIG: Zstd + Shuffle
            # Zstd creates smaller files than Gzip, relieving your I/O bottleneck.
            # **hdf5plugin.Zstd() automatically handles the filter setup.
            comp_args = {
                **hdf5plugin.Zstd(clevel=5), 
                'shuffle': True,
                'track_times': False
            }

            # 1. Save Data
            hf.create_dataset('sub_data', data=sub_data, dtype=np.float32, chunks=sub_data.shape, **comp_args)
            hf.create_dataset('sub_foot', data=sub_foot, dtype=np.float16, chunks=sub_foot.shape, **comp_args)
            hf.create_dataset('sub_bitmask', data=sub_bitmask, dtype=np.int32, chunks=sub_bitmask.shape, **comp_args)
            hf.create_dataset('sub_mapping', data=sub_mapping, dtype=np.float32, chunks=sub_mapping.shape, **comp_args)
            
            # 2. Save Metadata as Attributes
            hf.attrs['sub_header'] = sub_header_str
            hf.attrs['det_header'] = det_header_str
            hf.attrs['file_path'] = file_path
            hf.attrs['ref_coords'] = np.array([ref_y_min, ref_y_max, ref_x_min, ref_x_max], dtype=np.int32)

        return output_file # Return path on success
    except Exception as e:
        print(f'Error processing detector {det_idx} from exposure file index {exp_idx} ({file_path}): {e}')
        # import traceback; traceback.print_exc() # Uncomment for detailed debugging
        return None # Return None on failure

def batch_reproject(exposure_list, ref_wcs, ref_shape,
                    output_dir='output/', padding_percentage=0.05, num_processes=1,
                    sci_ext_list=[], dq_ext_list=[], reproj_func='interp', exp_idx_list=None, det_idx_list=None, 
                    replace_existing=False, reproject_kwargs={}):
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
    reproj_func : str
        Reproject function for reprojecting the science extensions
        - 'Exact': Slowest, conserves flux
        - 'Interp': Fastest, alter PSF profile, does not conserves flux
        - 'Adaptive': Faster then 'Exact', conserves flux
    exp_idx_list : list or tup, optional
        List of integers defining the exposure index in the exposure_list, if None, will use the index of the exposure_list
    det_idx_list : list or tup, optional
        List of integers defining the detector index in the exposure_list, if None, will use the index of the in fits file
    replace_existing : bool, optional
        If True, will overwrite existing files in the output directory, default is False

    Returns
    -------
    success_file : list
        List of path to the HDF5 files containing the reprojected data
    """

    assert isinstance(exposure_list, (list, tuple)) and exposure_list, "exposure_list must be a non-empty list or tuple"
    assert isinstance(ref_wcs, WCS), "Reference WCS must be an astropy.wcs.WCS object"
    assert isinstance(ref_shape, (list, tuple, np.ndarray)) and len(ref_shape) == 2, "ref_shape must be a list or tuple of length 2"
    assert isinstance(padding_percentage, float) and 0 <= padding_percentage, "padding_percentage must be a float larger than 0"
    assert isinstance(num_processes, int) and num_processes > 0, "num_processes must be a positive integer"
    assert sci_ext_list is None or isinstance(sci_ext_list, (list, tuple, np.ndarray)), "sci_ext_list must be a list or tuple"
    assert dq_ext_list is None or isinstance(dq_ext_list, (list, tuple, np.ndarray)), "dq_ext_list must be a list or tuple"
    assert reproj_func in ['exact', 'interp', 'adaptive'], "reproj_func must be one of 'exact', 'interp', or 'adaptive'"
    assert exp_idx_list is None or isinstance(exp_idx_list, (list, tuple, np.ndarray)), "exp_idx_list must be a list or tuple"
    assert det_idx_list is None or isinstance(det_idx_list, (list, tuple, np.ndarray)), "det_idx_list must be a list or tuple"
    assert type(replace_existing) is bool, "replace_existing must be a boolean"  
    
    os.makedirs(output_dir, exist_ok=True)
    print(f'Starting batch reprojection. Output will be saved to: {output_dir}')
    # Determine sub-frame width based on a sample detector frame
    try:
        with fits.open(exposure_list[0]) as hdul_sample:
            # Assuming first science extension is representative
            sci_ext_0 = sci_ext_list[0] if sci_ext_list else 1
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
            
            # Create a dictionary of parameters for each task
            task_params = {
                'reproj_func': reproj_func,
                'file_path': file_path,
                'exp_idx': exp_idx,
                'det_idx': det_idx,
                'sci_ext': sci_ext,
                'dq_ext': dq_ext,
                'ref_wcs': ref_wcs,
                'sub_width': sub_width,
                'output_dir': output_dir,
                'replace_existing': replace_existing,
                'reproject_kwargs': reproject_kwargs
            }
            tasks.append(task_params)
    
    results = []
    if num_processes > 1 and len(tasks) > 0 : # Ensure there are tasks for multiprocessing
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap_unordered(_reproject_worker, tasks), total=len(tasks), desc='Reprojecting frames'))
    elif len(tasks) > 0: # Sequential execution
        for task in tqdm(tasks, desc='Reprojecting frames (sequentially)'):
            results.append(_reproject_worker(task))
    
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
        Available fields: ['sub_data', 'sub_header', 'det_header', 'ref_coords', 'sub_foot', 'file_path', 
        'sub_bitmask', 'sub_mapping']

    Returns
    -------
    data : dict
        Dictionary containing the extracted data, key is the fields and value is the corresponding datas
    """

    assert isinstance(file_path, str) and os.path.isfile(file_path), "file_path must be a valid file path"
    assert isinstance(fields, (list, tuple)), "fields must be a list or tuple of strings"

    data = {}
    is_file_missing = False
    
    try:
        # swmr=True allows reading while the file is being written (if supported), 
        # libver='latest' supports the newer layout used in creation.
        with h5py.File(file_path, 'r', libver='latest', swmr=True) as file:
            
            for key in fields:
                # --- CASE 1: WCS Objects (Derived from Header Attributes) ---
                if key in ('sub_wcs', 'det_wcs'):
                    attr_key = 'sub_header' if key == 'sub_wcs' else 'det_header'
                    # Retrieve from attributes
                    if attr_key in file.attrs:
                        header_val = file.attrs[attr_key]
                        # Attributes often come out as bytes if encoded during write
                        if isinstance(header_val, bytes):
                            header_val = header_val.decode('utf-8')
                        data[key] = WCS(fits.Header.fromstring(header_val))
                    else:
                        data[key] = None # Handle missing header gracefully

                # --- CASE 2: Attributes (Metadata: headers, coords, paths) ---
                elif key in file.attrs:
                    val = file.attrs[key]
                    # Decode bytes to string if necessary (e.g., for file_path or headers)
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    data[key] = val

                # --- CASE 3: Datasets (Heavy Data: sub_data, sub_bitmask, etc.) ---
                elif key in file:
                    data[key] = file[key][()] # Load dataset into memory

                # --- CASE 4: Key not found ---
                else:
                    # Fallback for backward compatibility or missing keys
                    data[key] = None

    except Exception as e:
        print(f"Error loading {file_path}: {e}. Will use placeholders.")
        is_file_missing = True
        for key in fields:
            data[key] = None
            
    data['_is_missing_'] = is_file_missing
    return data

def _prep_subframe(file, chunk_map, apply_weight=False, apply_mask=False, 
                   chunk_offset=None, det_offset_func=None, ignore_list=None, 
                   det_valid_mask=None, valid_threshold=0.99, 
                   for_lsqr=False, oversample_factor=1, 
                   # These arguments are accepted for compatibility/internal logic 
                   # but might not be used depending on logic path
                   exp_idx=None, det_idx=None):
    """
    Prepares data from a single file for co-addition or lsqr.
    
    Refactored to use explicit arguments instead of **kwargs.
    """
    if ignore_list is None: ignore_list = []

    fields = ['sub_data', 'ref_coords', 'sub_mapping']
    if apply_mask:
        fields.append('sub_bitmask')
    result = load_reproj_file(file, fields=fields)

    sub_data = result['sub_data']
    ref_coords = result['ref_coords']
    sub_fullmask = np.ones_like(sub_data, dtype=bool)
    sub_mapping = result['sub_mapping']
    
    if 'sub_bitmask' in result:
        # invert=True: 1 = Good pixel, 0 = Bad pixel
        sub_boolmask = bit_to_bool(result['sub_bitmask'], ignore_list, invert=True)
        sub_fullmask &= sub_boolmask

    interp_matrix = None
    if (chunk_map is not None) or (chunk_offset is not None) or for_lsqr:
        sub_mapping_flat = sub_mapping.reshape(2, np.prod(sub_mapping.shape[1:]))
        sub_mapping_flat_scaled = sub_mapping_flat * oversample_factor
        interp_matrix = make_linear_interp_matrix(sub_mapping_flat_scaled[::-1], input_shape=np.shape(chunk_map))

    if chunk_offset is not None:
        if det_offset_func is not None:
            det_offset = det_offset_func(chunk_map, chunk_offset)
        else:
            det_offset = chunk_to_det(chunk_map, chunk_data=chunk_offset)
        sub_offset = det_to_sub(det_offset, interp_matrix=interp_matrix)
        sub_data -= sub_offset
   
    if det_valid_mask is not None:
        sub_valid_frac = det_to_sub(det_valid_mask, interp_matrix=interp_matrix)
        sub_valid_mask = sub_valid_frac > valid_threshold
        sub_fullmask &= sub_valid_mask

    sub_data[~sub_fullmask] = np.nan
    
    sub_weight = make_weight(sub_data) if apply_weight else np.ones_like(sub_data, dtype=np.float32)

    chunk_contrib = None
    if for_lsqr:
        # Note: If compute_chunk_contrib needs exp_idx/det_idx, ensure they are passed here.
        chunk_contrib = compute_chunk_contrib(chunk_map, interp_matrix)

    return ref_coords, sub_data, sub_weight, chunk_contrib

def _coadd_batch_worker(params):
    """
    Worker function that processes a batch. 
    Arguments are unpacked explicitly from the params dictionary.
    """
    # 1. Unpack Batch-specific data
    batch_files = params['batch_files']
    batch_indices = params['batch_indices']
    
    # 2. Unpack Global Configuration for Processing
    ref_shape = params['ref_shape']
    mode = params['mode']
    
    # 3. Unpack Iterables (to be indexed by 'i')
    exp_idx_list = params['exp_idx_list']
    det_idx_list = params['det_idx_list']
    offset_list = params['offset_list']
    
    # 4. Unpack Configuration for _prep_subframe explicitly
    # This dictionary isolates the kwargs needed for _prep_subframe
    prep_config = {
        'chunk_map': params['chunk_map'],
        'apply_weight': params['apply_weight'],
        'apply_mask': params['apply_mask'],
        'ignore_list': params['ignore_list'],
        'det_valid_mask': params['det_valid_mask'],
        'det_offset_func': params['det_offset_func'],
        'oversample_factor': params['oversample_factor'],
        'valid_threshold': params['valid_threshold'], # Ensure defaults match upper level
        'for_lsqr': False
    }

    # --- Shared Memory Reconstruction Start ---
    shm_handles = []
    mean_map = None
    std_map = None
    if 'mean_map_name' in params:
        shm_mean = SharedMemory(name=params['mean_map_name'])
        mean_map = np.ndarray(ref_shape, dtype=params['mean_map_dtype'], buffer=shm_mean.buf)
        shm_handles.append(shm_mean)
        
    if 'std_map_name' in params:
        shm_std = SharedMemory(name=params['std_map_name'])
        std_map = np.ndarray(ref_shape, dtype=params['std_map_dtype'], buffer=shm_std.buf)
        shm_handles.append(shm_std)
    # --- Shared Memory Reconstruction End ---

    data_sum = np.zeros(ref_shape, dtype=np.float32)
    weight_sum = np.zeros(ref_shape, dtype=np.float32)
    
    for i, file_path in enumerate(batch_files):
        idx = batch_indices[i]
        
        # Calculate dynamic arguments for this specific file
        current_exp = exp_idx_list[idx] if exp_idx_list is not None else None
        current_det = det_idx_list[idx] if det_idx_list is not None else None
        current_offset = offset_list[idx] if offset_list is not None else None

        coords, data, weight, _ = _prep_subframe(
            file=file_path,
            exp_idx=current_exp,
            det_idx=current_det,
            chunk_offset=current_offset,
            **prep_config 
        )

        if coords is None:
            continue
            
        sub_crop, ref_crop = compute_crop(ref_shape, coords)
        data_crop = data[sub_crop]
        weight_crop = weight[sub_crop]
        valid = ~check_invalid(data_crop)

        if mode == 'mean':
            data_sum[ref_crop] += np.where(valid, data_crop * weight_crop, 0.0)
            weight_sum[ref_crop] += np.where(valid, weight_crop, 0.0)

        elif mode == 'std':
            mean_crop = mean_map[ref_crop]
            data_sum[ref_crop] += np.where(valid, (data_crop - mean_crop)**2 * weight_crop, 0.0)
            weight_sum[ref_crop] += np.where(valid, weight_crop, 0.0)
        
        elif mode == 'sigma_clip':
            mean_crop = mean_map[ref_crop]
            std_crop = std_map[ref_crop]
            sigma = params['sigma']
            clip_mask = np.abs(data_crop - mean_crop) <= sigma * std_crop
            valid_clipped = valid & clip_mask
            
            data_sum[ref_crop] += np.where(valid_clipped, data_crop * weight_crop, 0.0)
            weight_sum[ref_crop] += np.where(valid_clipped, weight_crop, 0.0)

        elif mode == 'custom':
            pass 
    
    # Cleanup shared memory handles for this worker
    for shm in shm_handles:
        shm.close()

    return data_sum, weight_sum

def _parallel_coadd(mode, params):
    """
    Central function to manage parallel processing.
    Accepts all processing parameters in a single dictionary.
    """
    reproj_file_list = params['reproj_file_list']
    ref_shape = params['ref_shape']
    max_workers = params.get('max_workers', 10)
    batch_size = params.get('batch_size', 20) 
    
    total_files = len(reproj_file_list)
    if total_files == 0:
        return np.zeros(ref_shape, dtype=np.float32), np.zeros(ref_shape, dtype=np.float32)
        
    tasks = []
    # Create tasks based on fixed batch_size instead of splitting by max_workers
    for start_idx in range(0, total_files, batch_size):
        end_idx = min(start_idx + batch_size, total_files)
        
        # Create the parameter dictionary for this specific batch/worker
        task_params = params.copy()
        task_params.update({
            'batch_files': reproj_file_list[start_idx:end_idx],
            'batch_indices': list(range(start_idx, end_idx)),
            'mode': mode,
        })
        tasks.append(task_params)

    print(f"Processing {total_files} files in {len(tasks)} batches (Batch Size: {batch_size})...")

    total_data_sum = np.zeros(ref_shape, dtype=np.float32)
    total_weight_sum = np.zeros(ref_shape, dtype=np.float32)

    with Pool(processes=max_workers) as pool:
        for data_part, weight_part in tqdm(pool.imap_unordered(_coadd_batch_worker, tasks), total=len(tasks)):
            total_data_sum += data_part
            total_weight_sum += weight_part
            del data_part, weight_part
        
    return total_data_sum, total_weight_sum

def compute_coadd_map(mode, ref_shape, reproj_file_list, mean_map=None, std_map=None, sigma=3.0, 
                      offset_list=None, exp_idx_list=None, det_idx_list=None, apply_weight=True, 
                      apply_mask=True, chunk_map=None, det_valid_mask=None, 
                      max_workers=10, ignore_list=[], det_offset_func=None, oversample_factor=1,
                      batch_size=10, valid_threshold=0.99):
    """
    This function serves as a unified interface for creating mean maps, standard
    deviation maps, and sigma-clipped mean maps in parallel.

    Parameters
    ----------
    mode : {'mean', 'std', 'sigma_clip'}
        The type of computation to perform.
        - 'mean': Computes the weighted mean map.
        - 'std': Computes the weighted standard deviation map. Requires `mean_map`.
        - 'sigma_clip': Computes a sigma-clipped weighted mean. Requires `mean_map` and `std_map`.
    ref_shape : tuple, list
        Shape of the reference frame (height, width).
    reproj_file_list : list
        List of paths to the reprojected HDF5 files.
    mean_map : np.ndarray, optional
        The pre-computed mean map. Required when `mode` is 'std' or 'sigma_clip'.
    std_map : np.ndarray, optional
        The pre-computed standard deviation map. Required when `mode` is 'sigma_clip'.
    sigma : float, optional
        The number of standard deviations for sigma clipping. Used only when `mode` is 'sigma_clip'. Default is 3.0.
    offset_list : list, optional
        List of offsets for each exposure, shape (num_reproj_file, num_chunks). Default is None.
    exp_idx_list : list, optional
        List of exposure indices. Default is None.
    det_idx_list : list, optional
        List of detector indices. Default is None.
    apply_weight : bool, optional
        Whether to apply weights to the data. Default is True.
    apply_mask : bool, optional
        Whether to apply masks to the data. Default is True.
    chunk_map : dict, optional
        Mapping of chunk indices to their corresponding pixel indices. Default is None.
    det_valid_mask : np.ndarray, optional
        Mask indicating valid pixels for each detector. Default is None.
    max_workers : int, optional
        Maximum number of worker processes for parallel processing. Default is 10.
    ignore_list : list, optional
        List of data quality flags to ignore. Default is an empty list.
    det_offset_func : callable, optional
        Function to compute detector offsets. Default is None.
    oversample_factor : int, optional
        Factor by which the chunk map is oversampled. Default is 1.
    batch_size : int, optional
        Number of files to process per worker task. Default is 20.
    Returns
    -------
    result_map : np.ndarray
        The computed map (mean, std, or sigma-clipped mean).
    weight_sum : np.ndarray
        The sum of weights used in the calculation.
    """
    # --- Common Assertions for All Modes ---
    assert mode in ['mean', 'std', 'sigma_clip'], "mode must be one of 'mean', 'std', or 'sigma_clip'"
    assert isinstance(ref_shape, (list, np.ndarray, tuple)) and len(ref_shape) == 2, "ref_shape must be a list or tuple of length 2"
    assert isinstance(reproj_file_list, (list, np.ndarray)) and reproj_file_list, "reproj_file_list must be a non-empty list"
    assert offset_list is None or (isinstance(offset_list, (list, np.ndarray)) and np.shape(offset_list) == (len(reproj_file_list), len(np.unique(chunk_map)))), \
        "offset_list must be a list or array of shape (num_reproj_file, num_chunks)"
    assert exp_idx_list is None or isinstance(exp_idx_list, (list, np.ndarray)), "exp_idx_list must be a list or array"
    assert det_idx_list is None or isinstance(det_idx_list, (list, np.ndarray)), "det_idx_list must be a list or array"
    assert isinstance(apply_weight, bool), "apply_weight must be a boolean"
    assert isinstance(apply_mask, bool), "apply_mask must be a boolean"
    assert chunk_map is None or isinstance(chunk_map, (list, np.ndarray)), "chunk_map must be a list or array"
    assert det_valid_mask is None or isinstance(det_valid_mask, np.ndarray), "det_valid_mask must be a numpy array"
    assert isinstance(max_workers, int) and max_workers > 0, "max_workers must be a positive integer"
    assert isinstance(ignore_list, (list, np.ndarray)), "ignore_list must be a list or array of data quality flags to ignore"
    assert det_offset_func is None or callable(det_offset_func), "det_offset_func must be a callable function or None"
    assert isinstance(oversample_factor, int) and oversample_factor > 0, "oversample_factor must be a positive integer"
    assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"

    # Explicitly build the params dictionary. 
    # Only keys defined here are accessible in _coadd_batch_worker.
    params = {
        # Execution control
        'mode': mode,
        'ref_shape': ref_shape,
        'reproj_file_list': reproj_file_list, # Used by _parallel_coadd for batch slicing
        'max_workers': max_workers,
        'batch_size': batch_size,
        
        # Iterables
        'exp_idx_list': exp_idx_list,
        'det_idx_list': det_idx_list,
        'offset_list': offset_list,
        
        # Configuration for _prep_subframe
        'chunk_map': chunk_map,
        'det_valid_mask': det_valid_mask,
        'apply_weight': apply_weight,
        'apply_mask': apply_mask,
        'ignore_list': ignore_list,
        'det_offset_func': det_offset_func,
        'oversample_factor': oversample_factor,
        'valid_threshold': valid_threshold,
        
        # Mode-specific
        'sigma': sigma,
    }

    try:
        # --- Shared Memory Setup ---
        shm_objects = []
        if mean_map is not None and mode in ['std', 'sigma_clip']:
            shm_mean = SharedMemory(create=True, size=mean_map.nbytes)
            shared_mean_arr = np.ndarray(mean_map.shape, dtype=mean_map.dtype, buffer=shm_mean.buf)
            shared_mean_arr[:] = mean_map[:]
            shm_objects.append(shm_mean)
            
            params['mean_map_name'] = shm_mean.name
            params['mean_map_dtype'] = mean_map.dtype 

        if std_map is not None and mode in ['sigma_clip']:
            shm_std = SharedMemory(create=True, size=std_map.nbytes)
            shared_std_arr = np.ndarray(std_map.shape, dtype=std_map.dtype, buffer=shm_std.buf)
            shared_std_arr[:] = std_map[:]
            shm_objects.append(shm_std)
            
            params['std_map_name'] = shm_std.name
            params['std_map_dtype'] = std_map.dtype
    
        # --- Mode-Specific Logic ---
        if mode == 'mean':
            data_sum, weight_sum = _parallel_coadd(mode=mode, params=params)
            result_map = np.divide(data_sum, weight_sum, out=np.zeros_like(data_sum), where=weight_sum != 0)
            return result_map, weight_sum

        elif mode == 'std':
            sq_diff_sum, weight_sum = _parallel_coadd(mode=mode, params=params)
            variance = np.divide(sq_diff_sum, weight_sum, out=np.zeros_like(sq_diff_sum), where=weight_sum > 0)
            result_map = np.sqrt(variance)
            return result_map, weight_sum

        elif mode == 'sigma_clip':
            data_sum, weight_sum = _parallel_coadd(mode=mode, params=params)
            result_map = np.divide(data_sum, weight_sum, out=np.zeros_like(data_sum), where=weight_sum != 0)
            return result_map, weight_sum

    finally:
        for shm in shm_objects:
            shm.close()
            shm.unlink()

def _prep_lsqr(task_params):
    '''Compute the components of the LSQR matrix A and vector b for a single subframe.'''
    # 1. Unpack task specific
    i = task_params['i']
    reproj_file = task_params['reproj_file']
    exp_idx = task_params['exp_idx']
    det_idx = task_params['det_idx']
    
    # 2. Unpack Config for Logic
    ref_shape = task_params['ref_shape']
    num_exp = task_params['num_exp']
    num_chunks = task_params['num_chunks']
    outlier_thresh = task_params['outlier_thresh']
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w

    try:
        # 3. Explicit Call to _prep_subframe
        # We explicitly retrieve args from task_params. 
        # Using .get() allows for safe defaults if the setup function forgot to pack them.
        ref_coords, sub_data, sub_weight, chunk_contrib = _prep_subframe(
            file=reproj_file,
            exp_idx=exp_idx,
            det_idx=det_idx,
            chunk_offset=None, # LSQR usually solves for this, so we don't apply it yet
            for_lsqr=True,
            det_offset_func=None,
            
            # Explicit Parameters passed down
            chunk_map=task_params['chunk_map'],
            apply_weight=task_params['apply_weight'],
            apply_mask=task_params['apply_mask'],
            ignore_list=task_params['ignore_list'],
            det_valid_mask=task_params['det_valid_mask'],
            oversample_factor=task_params['oversample_factor'],
            valid_threshold=task_params['valid_threshold']
        )

        sub_h, sub_w = sub_data.shape
        
        # ... [Rest of logic remains identical] ...
        sub_valid = ~check_invalid(sub_data) 
        if isinstance(outlier_thresh, (int, float)) and outlier_thresh > 0:
            sub_out = find_outliers(sub_data, threshold=outlier_thresh) 
            sub_valid &= ~sub_out 
        valid_sub_coords = np.nonzero(sub_valid) 
        sub_pix_indices = valid_sub_coords[0] * sub_w + valid_sub_coords[1]
        valid_vals = sub_data[valid_sub_coords] 
        valid_weight = sub_weight[valid_sub_coords] 
        num_valid_pixels = valid_vals.shape[0]

        if num_valid_pixels == 0:
            return np.array([]), np.array([]), np.array([]), np.array([]), 0

        ref_pix_indices = (valid_sub_coords[0] + ref_coords[0]) * ref_w + (valid_sub_coords[1] + ref_coords[2]) 

        S_rows = np.arange(num_valid_pixels) 
        S_cols = ref_pix_indices 
        S_data = valid_weight 

        chunk_idx, sub_idx = chunk_contrib[:, sub_pix_indices].nonzero() 
        chunk_vals = chunk_contrib[:, sub_pix_indices][(chunk_idx, sub_idx)].A[0]
        O_rows = sub_idx 
        O_cols = chunk_idx + exp_idx*num_chunks + (num_sky) 
        O_data = valid_weight[sub_idx] * chunk_vals 

        sub_b = valid_vals * valid_weight 

        sub_rows = np.concatenate([S_rows, O_rows])
        sub_cols = np.concatenate([S_cols, O_cols])
        sub_data_vec = np.concatenate([S_data, O_data]) 

        valid_mask = ~check_invalid(sub_b[sub_rows]) & ~((sub_data_vec == 0) & (sub_b[sub_rows] == 0))
        sub_rows = sub_rows[valid_mask]
        sub_cols = sub_cols[valid_mask]
        sub_data_vec = sub_data_vec[valid_mask]

        unique_rows, new_row_indices = np.unique(sub_rows, return_inverse=True)
        sub_rows = new_row_indices
        sub_b = sub_b[unique_rows]
        
        return sub_rows, sub_cols, sub_data_vec, sub_b, len(sub_b)

    except Exception as e:
        print(f"Error processing file {reproj_file} for exp_idx={exp_idx}: {e}")
        traceback.print_exc()
        return None

def _prep_lsqr_batch_worker(batch_params):
    """Wrapper to process a list (batch) of subframes in a single worker process."""
    sub_tasks = batch_params['sub_tasks']
    
    batch_rows = []
    batch_cols = []
    batch_data = []
    batch_b = []
    batch_row_offset = 0

    for task_params in sub_tasks:
        result = _prep_lsqr(task_params)
        
        if result is None:
            continue
            
        sub_rows, sub_cols, sub_data, sub_b, num_rows = result
        if len(sub_b) == 0:
            continue
            
        batch_rows.append(sub_rows + batch_row_offset)
        batch_cols.append(sub_cols)
        batch_data.append(sub_data)
        batch_b.append(sub_b)
        batch_row_offset += num_rows

    if len(batch_b) == 0:
        return None

    return (
        np.concatenate(batch_rows),
        np.concatenate(batch_cols),
        np.concatenate(batch_data),
        np.concatenate(batch_b),
        batch_row_offset
    )

def setup_lsqr(reproj_file_list, ref_shape, exp_idx_list, det_idx_list, 
               chunk_map=None, det_valid_mask=None, apply_mask=True, apply_weight=False, 
               # Exposed new arguments here to be explicit
               valid_threshold=0.99,
               outlier_thresh=3, max_workers=20, ignore_list=[], oversample_factor=1, batch_size=10):
    """Prepares the LSQR matrix A and vector b for all subframes in parallel.
    Parameters
    ----------
    reproj_file_list : list
        List of paths to the reprojected HDF5 files
    ref_shape : tuple, list
        Shape of the reference frame (height, width)
    exp_idx_list : list, optional
        List of exposure indices corresponding to each reprojection file.
    det_idx_list : list, optional
        List of detector indices corresponding to each reprojection file.
    chunk_map : np.ndarray, optional
        Mapping of chunk indices to their corresponding pixel indices.
    det_valid_mask : np.ndarray, optional
        Mask indicating valid pixels for each detector.
    apply_mask : bool, optional
        Whether to apply masks to the data. Default is True.
    apply_weight : bool, optional
        Whether to apply weights to the data. Default is True.
    outlier_thresh : float, optional
        z-value threshold for outlier detection, default is 3.0.
    max_workers : int, optional
        Maximum number of worker processes to use for parallel processing, default is 20.
    ignore_list : list, optional
        List of data quality flags to ignore, default is an empty list.
    oversample_factor : int, optional
        Factor by which the chunk map is oversampled. Default is 1.
    batch_size : int, optional
        Number of files to process per worker task. Default is 10.
    Returns
    -------
    full_A : scipy.sparse.coo_matrix
        The sparse matrix A in COO format, shape is (num_equations, num_unknowns)
    full_b : np.ndarray
        The vector b, shape is (num_equations,)
    """
    assert isinstance(reproj_file_list, (list, np.ndarray)) and reproj_file_list, "reproj_file_list must be a non-empty list"
    assert len(reproj_file_list) == len(exp_idx_list) == len(det_idx_list), \
        "reproj_file_list, exp_idx_list, and det_idx_list must have the same length"
    assert isinstance(ref_shape, (list, np.ndarray, tuple)) and len(ref_shape) == 2, "ref_shape must be a list of length 2"
    assert chunk_map is None or isinstance(chunk_map, np.ndarray), "chunk_map must be a numpy array"
    assert det_valid_mask is None or isinstance(det_valid_mask, np.ndarray), "det_valid_mask must be a numpy array"
    assert isinstance(apply_mask, bool), "apply_mask must be a boolean"
    assert isinstance(apply_weight, bool), "apply_weight must be a boolean"
    assert isinstance(outlier_thresh, (int, float)) and outlier_thresh > 0, "outlier_thresh must be a positive number"
    assert isinstance(max_workers, int) and max_workers > 0, "max_workers must be a positive integer"
    assert isinstance(ignore_list, (list, np.ndarray)), "ignore_list must be a list or array of data quality flags to ignore"
    assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"

    num_chunks = len(np.unique(chunk_map)) if chunk_map is not None else 0
    unique_exps, reindexed_exp_idx_list = np.unique(exp_idx_list, return_inverse=True)
    num_exp = len(unique_exps)
    num_det = len(np.unique(det_idx_list))
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    total_cols = num_sky + num_exp * num_chunks

    # Explicitly gather the common configuration for the workers
    common_params = {
        # Arguments for _prep_subframe
        'chunk_map': chunk_map,
        'det_valid_mask': det_valid_mask,
        'apply_mask': apply_mask,
        'apply_weight': apply_weight,
        'ignore_list': ignore_list,
        'oversample_factor': oversample_factor,
        'valid_threshold': valid_threshold,
        
        # Arguments for _prep_lsqr logic
        'outlier_thresh': outlier_thresh,
        'num_exp': num_exp,
        'num_chunks': num_chunks,
        'ref_shape': ref_shape,
    }

    # 1. Create all individual task definitions
    all_individual_tasks = []
    for i, (reproj_file, exp_idx, det_idx) in enumerate(zip(reproj_file_list, reindexed_exp_idx_list, det_idx_list)):
        task_params = {
            'i': i,
            'reproj_file': reproj_file,
            'exp_idx': exp_idx,
            'det_idx': det_idx,
        }
        # Combine the dynamic task params with the static common params
        task_params.update(common_params)
        all_individual_tasks.append(task_params)

    # 2. Group tasks into batches
    batched_tasks = []
    for i in range(0, len(all_individual_tasks), batch_size):
        batch = {
            'sub_tasks': all_individual_tasks[i : i + batch_size]
        }
        batched_tasks.append(batch)

    print(f"Processing {len(all_individual_tasks)} items in {len(batched_tasks)} batches (Batch Size: {batch_size})...")

    all_rows = []
    all_cols = []
    all_data = []
    all_b = []

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_prep_lsqr_batch_worker, batch): i for i, batch in enumerate(batched_tasks)}
        
        total_rows = 0
        for future in tqdm(as_completed(futures), total=len(futures), desc="Building A, b matrix"):
            result = future.result()
            
            if result is None:
                continue
                
            b_rows, b_cols, b_data, b_b, b_num_rows = result
            
            all_rows.append(b_rows + total_rows)
            all_cols.append(b_cols)
            all_data.append(b_data)
            all_b.append(b_b)
            total_rows += b_num_rows

    if len(all_b) == 0:
        print("No valid data found in any subframe.")
        return None, None

    rows = np.concatenate(all_rows)
    cols = np.concatenate(all_cols)
    data = np.concatenate(all_data)
    b = np.concatenate(all_b)

    full_A = coo_matrix((data, (rows, cols)), shape=(total_rows, total_cols))
    full_b = b
    return full_A, full_b

def apply_lsqr(A, b, ref_shape, exp_idx_list, det_idx_list, x0=None, 
                atol=1e-05, btol=1e-05, damp=1e-2, iter_lim=100):
    """Applies the LSQR algorithm to solve for the sky and detector offsets.
    Parameters
    ----------
    A : scipy.sparse.coo_matrix
        The sparse matrix A in COO format, shape is (num_equations, num_unknowns)
    b : np.ndarray
        The vector b, shape is (num_equations,)
    ref_shape : tuple, list
        Shape of the reference frame (height, width)
    exp_idx_list : list, optional
        List of exposure indices corresponding to each reprojection file.
    det_idx_list : list, optional
        List of detector indices corresponding to each reprojection file.
    x0 : np.ndarray, optional
        Initial guess for the solution, shape is (num_unknowns,). If None, will use a zero vector.
    atol : float, optional
        Absolute tolerance for convergence, default is 1e-05.
    btol : float, optional
        Relative tolerance for convergence, default is 1e-05. 
    damp : float, optional
        Damping factor for the LSQR algorithm, default is 1e-2.
    iter_lim : int, optional
        Maximum number of iterations for the LSQR algorithm, default is 100.

    Returns
    -------
    O : np.ndarray
        The detector offsets, shape is (num_exp, num_det, num_chunks)
    S : np.ndarray
        The sky offsets, shape is (height, width)   
    """
    num_exp = len(np.unique(exp_idx_list))
    num_det = len(np.unique(det_idx_list))
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w

    print(f"Solving least squares for {A.shape[1]} unknowns with {A.shape[0]} equations.")
    result = lsqr(A, b, x0=x0, show=True, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim)
    x = result[0]

    S = x[:num_sky].reshape(ref_shape)
    O = x[num_sky:].reshape(num_exp, (x.shape[0]-num_sky) // num_exp)
    return O, S
