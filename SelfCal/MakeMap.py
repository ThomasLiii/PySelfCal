import os
import h5py
import hdf5plugin
import threading
from tqdm import tqdm
from multiprocessing import Pool, Lock as _MPLock, BoundedSemaphore as _MPSemaphore
from multiprocessing.shared_memory import SharedMemory

# Semaphore to limit concurrent HDD reads. With many workers doing random reads
# on a RAID array, seek thrashing kills throughput. Uses multiprocessing.BoundedSemaphore
# so it works across both threads (ThreadPoolExecutor) and forked processes (Pool).
_hdd_io_semaphore = None
_coadd_flush_lock = None

def _init_coadd_worker(lock):
    """Pool initializer: store the multiprocessing Lock as a module global."""
    global _coadd_flush_lock
    _coadd_flush_lock = lock

def set_hdd_io_limit(n):
    """Set the max number of concurrent file reads from slow storage.
    Call before any parallel processing starts. Works across both threads and processes.
    """
    global _hdd_io_semaphore
    _hdd_io_semaphore = _MPSemaphore(n) if n and n > 0 else None
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
import numpy as np

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
import astropy.units as u

from reproject import reproject_interp
from reproject import reproject_exact
from reproject import reproject_adaptive

from scipy.sparse import coo_matrix, csr_matrix, vstack, diags
from scipy.sparse.linalg import lsqr, lsmr, LinearOperator
from threadpoolctl import threadpool_limits
import sys 
import gc 
import traceback
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
            sci_ext_0 = sci_ext_list[0] if len(sci_ext_list) > 0 else 1
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

    sem = _hdd_io_semaphore
    if sem is not None:
        sem.acquire()
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

        # Parse indices from filename
        det_idx = int(os.path.basename(file_path).replace('.h5', '').split('_')[-1])
        exp_idx = int(os.path.basename(file_path).replace('.h5', '').split('_')[-3])
        data['det_idx'] = det_idx
        data['exp_idx'] = exp_idx
    except Exception as e:
        print(f"Error loading {file_path}: {e}. Will use placeholders.")
        is_file_missing = True
        for key in fields:
            data[key] = None
        det_idx = None
        exp_idx = None
    finally:
        if sem is not None:
            sem.release()

    data['_is_missing_'] = is_file_missing
    return data

def _prep_subframe(file, chunk_map, apply_weight=False, apply_mask=False, 
                   chunk_offset=None, det_offset_func=None, ignore_list=None, 
                   grid_valid_weight=None, valid_threshold=0.99, 
                   for_lsqr=False, oversample_factor=1, 
                   # These arguments are accepted for compatibility/internal logic 
                   # but might not be used depending on logic path
                   det_aux=None, postprocess_func=None, preprocess_func=None):
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
    sub_weight = np.ones_like(sub_data, dtype=np.float32)
    sub_mapping = result['sub_mapping']
    exp_idx = result['exp_idx']
    det_idx = result['det_idx']
    
    if preprocess_func is not None:
        sub_data = preprocess_func(locals())

    # Apply bitmask
    if 'sub_bitmask' in result:
        # invert=True: 1 = Good pixel, 0 = Bad pixel
        sub_boolmask = bit_to_bool(result['sub_bitmask'], ignore_list, invert=True)
        sub_weight *= sub_boolmask

    # Compute bilinear interpolation matrix for mapping between chunk and subframe
    interp_matrix = None
    if (chunk_map is not None) or (chunk_offset is not None) or (for_lsqr) or (det_aux is not None) or (grid_valid_weight is not None):
        sub_mapping_flat = sub_mapping.reshape(2, np.prod(sub_mapping.shape[1:]))
        sub_mapping_flat_scaled = sub_mapping_flat * oversample_factor
        interp_matrix = make_linear_interp_matrix(sub_mapping_flat_scaled[::-1], input_shape=np.shape(chunk_map))

    # Apply chunk offset if provided
    if chunk_offset is not None:
        if det_offset_func is not None:
            grid_offset = det_offset_func(chunk_map, chunk_offset)
        else:
            grid_offset = chunk_to_det(chunk_map, chunk_data=chunk_offset)
        sub_offset = det_to_sub(grid_offset, interp_matrix=interp_matrix)
        sub_data -= sub_offset
    
    # Apply valid weight
    if grid_valid_weight is not None:
        sub_valid_weight = det_to_sub(grid_valid_weight, interp_matrix=interp_matrix)
        sub_weight *= sub_valid_weight

    sub_aux = None
    if det_aux is not None:
        sub_aux = np.array([det_to_sub(det_aux_data, interp_matrix=interp_matrix) for det_aux_data in det_aux])

    if apply_weight:
        sub_weight *= make_weight(sub_data)

    chunk_contrib = None
    if for_lsqr:
        chunk_contrib = compute_chunk_contrib(chunk_map, interp_matrix)

    if postprocess_func is not None:
        sub_data = postprocess_func(locals())

    # Check for NaNs and set corresponding weights to 0
    nan_mask = np.isnan(sub_data)
    sub_data[nan_mask] = 0.0
    sub_weight[nan_mask] = 0.0
    
    return ref_coords, sub_data, sub_weight, chunk_contrib, sub_aux

def _coadd_batch_worker(params):
    """
    Unified worker function for parallel processing.
    
    Modes:
    1. Caching Mode:
       - Runs _prep_subframe -> Saves to HDF5 -> Returns list of filenames.
    2. Coadd Mode:
       - Runs _prep_subframe (or loads cache) -> Stacks data -> Returns (sum, weights).
    """
    # 1. Unpack Common Parameters
    mode = params['mode']
    batch_files = params['batch_files']
    batch_indices = params['batch_indices']
    batch_offsets = params['batch_offsets']
    ref_shape = params['ref_shape']
    
    # Execution Flags
    use_cached = params['use_cached']

    shm_handles = []
    det_aux = params.get('det_aux')
    if det_aux is None and 'det_aux_name' in params:
        shm_aux = SharedMemory(name=params['det_aux_name'])
        det_aux = np.ndarray(params['det_aux_shape'], dtype=params['det_aux_dtype'], buffer=shm_aux.buf)
        shm_handles.append(shm_aux)

    if 'chunk_map_shm_name' in params:
        shm_cm = SharedMemory(name=params['chunk_map_shm_name'])
        params['chunk_map'] = np.ndarray(params['chunk_map_shape'], dtype=params['chunk_map_dtype'], buffer=shm_cm.buf)
        shm_handles.append(shm_cm)

    if 'gvw_shm_name' in params:
        shm_gvw = SharedMemory(name=params['gvw_shm_name'])
        params['grid_valid_weight'] = np.ndarray(params['gvw_shape'], dtype=params['gvw_dtype'], buffer=shm_gvw.buf)
        shm_handles.append(shm_gvw)

    # 2. Initialize Mode-Specific Containers
    if mode == 'cache':
        cache_dir = params['cache_dir']
        cached_list = []
    else:
        # Accumulators — direct arrays (threads) or SharedMemory (processes)
        data_sum_arr = params.get('total_data_sum')
        if data_sum_arr is None:
            shm_data_sum = SharedMemory(name=params['total_data_sum_name'])
            data_sum_arr = np.ndarray(ref_shape, dtype=np.float32, buffer=shm_data_sum.buf)
            shm_handles.append(shm_data_sum)

        weight_sum_arr = params.get('total_weight_sum')
        if weight_sum_arr is None:
            shm_weight_sum = SharedMemory(name=params['total_weight_sum_name'])
            weight_sum_arr = np.ndarray(ref_shape, dtype=np.float32, buffer=shm_weight_sum.buf)
            shm_handles.append(shm_weight_sum)

        aux_sum_arr = params.get('total_aux_sum')
        if aux_sum_arr is None and 'total_aux_sum_name' in params:
            shm_aux_sum = SharedMemory(name=params['total_aux_sum_name'])
            aux_sum_arr_shape = (params['det_aux_shape'][0],) + ref_shape
            aux_sum_arr = np.ndarray(aux_sum_arr_shape, dtype=np.float32, buffer=shm_aux_sum.buf)
            shm_handles.append(shm_aux_sum)

        # Local accumulators — each worker accumulates independently, then flushes
        # to shared arrays once at the end (single lock acquisition per batch).
        local_data_sum = np.zeros(ref_shape, dtype=np.float32)
        local_weight_sum = np.zeros(ref_shape, dtype=np.float32)
        local_aux_sum = np.zeros_like(aux_sum_arr) if aux_sum_arr is not None else None

        # Read-only maps — direct arrays (threads) or SharedMemory (processes)
        mean_map = params.get('mean_map')
        if mean_map is None and 'mean_map_name' in params:
            shm_mean = SharedMemory(name=params['mean_map_name'])
            mean_map = np.ndarray(ref_shape, dtype=params['mean_map_dtype'], buffer=shm_mean.buf)
            shm_handles.append(shm_mean)

        std_map = params.get('std_map')
        if std_map is None and 'std_map_name' in params:
            shm_std = SharedMemory(name=params['std_map_name'])
            std_map = np.ndarray(ref_shape, dtype=params['std_map_dtype'], buffer=shm_std.buf)
            shm_handles.append(shm_std)

    # 3. Configuration for _prep_subframe
    if not use_cached:
        prep_config = {
            'chunk_map': params['chunk_map'],
            'apply_weight': params['apply_weight'],
            'apply_mask': params['apply_mask'],
            'ignore_list': params['ignore_list'],
            'grid_valid_weight': params['grid_valid_weight'],
            'det_offset_func': params['det_offset_func'],
            'oversample_factor': params['oversample_factor'],
            'valid_threshold': params['valid_threshold'],
            'for_lsqr': False,
            'preprocess_func': params['preprocess_func'],
            'postprocess_func': params['postprocess_func'],
        }

    # 4. Processing Loop
    for in_index, file_path in enumerate(batch_files):
        index = batch_indices[in_index]
        
        ref_coords, sub_data, sub_weight, sub_aux = None, None, None, None

        # --- A. GET DATA ---
        if use_cached:
            # Load from existing cache
            try:
                with h5py.File(file_path, 'r') as hf:
                    ref_coords = hf['ref_coords'][:]
                    sub_data = hf['sub_data'][:]
                    sub_weight = hf['sub_weight'][:]
                    if 'sub_aux' in hf:
                        sub_aux = hf['sub_aux'][:]

            except Exception as e:
                print(f"Error loading cached file {file_path}: {e}")
                continue
        else:
            # Compute on the fly
            current_offset = batch_offsets[in_index] if batch_offsets is not None else None

            ref_coords, sub_data, sub_weight, _, sub_aux = _prep_subframe(
                file=file_path,
                chunk_offset=current_offset,
                det_aux=det_aux,
                **prep_config 
            )

        # --- B. PROCESS DATA ---
        
        # Path 1: Save to Cache (and stop there)
        if mode == 'cache':
            org_name = os.path.basename(file_path)
            cache_name = f"cached_{org_name}"
            cache_path = os.path.join(cache_dir, cache_name)

            # Crop to the tight bbox of nonzero weight: typically only a small
            # fraction of the subframe is valid (e.g., a single channel within
            # a multi-channel detector), so this dramatically reduces I/O.
            nz_rows = np.any(sub_weight, axis=1)
            nz_cols = np.any(sub_weight, axis=0)
            row_idx = np.where(nz_rows)[0]
            col_idx = np.where(nz_cols)[0]
            if row_idx.size > 0 and col_idx.size > 0:
                rmin, rmax = int(row_idx[0]), int(row_idx[-1]) + 1
                cmin, cmax = int(col_idx[0]), int(col_idx[-1]) + 1
            else:
                rmin = cmin = 0
                rmax = cmax = 0

            sub_data_c = sub_data[rmin:rmax, cmin:cmax]
            sub_weight_c = sub_weight[rmin:rmax, cmin:cmax]
            sub_aux_c = sub_aux[:, rmin:rmax, cmin:cmax] if sub_aux is not None else None

            # Update ref_coords to reflect the crop in ref-frame coordinates,
            # so compute_crop(ref_shape, ref_coords) yields the correct slices
            # for the cropped sub_data on the consumer side.
            y_min, y_max, x_min, x_max = ref_coords
            new_ref_coords = np.array(
                [y_min + rmin, y_min + rmax, x_min + cmin, x_min + cmax],
                dtype=ref_coords.dtype,
            )

            with h5py.File(cache_path, 'w') as hf:
                hf.create_dataset('ref_coords', data=new_ref_coords, dtype=new_ref_coords.dtype, track_times=False)
                hf.create_dataset('sub_data', data=sub_data_c, dtype=sub_data_c.dtype, track_times=False)
                hf.create_dataset('sub_weight', data=sub_weight_c, dtype=sub_weight_c.dtype, track_times=False)
                if sub_aux_c is not None:
                    hf.create_dataset('sub_aux', data=sub_aux_c, dtype=sub_aux_c.dtype, track_times=False)
                # Bbox in the original (full) sub frame coordinates, so consumers
                # that compute auxiliary arrays at full sub shape (e.g. wav_coadd
                # building sub_BC/sub_BW from sub_mapping) can match the crop.
                hf.create_dataset('sub_bbox', data=np.array([rmin, rmax, cmin, cmax], dtype=np.int32), track_times=False)

            cached_list.append(cache_path)
        
        # Path 2: Accumulate to Mosaic
        else:
            sub_crop, ref_crop = compute_crop(ref_shape, ref_coords)
            data_crop = sub_data[sub_crop]
            weight_crop = sub_weight[sub_crop]

            if mode == 'mean':
                data_val = data_crop * weight_crop
                aux_val = sub_aux[:, *sub_crop] * weight_crop if (local_aux_sum is not None and sub_aux is not None) else None
                local_data_sum[ref_crop] += data_val
                local_weight_sum[ref_crop] += weight_crop
                if aux_val is not None:
                    local_aux_sum[:, *ref_crop] += aux_val

            elif mode == 'std':
                mean_crop = mean_map[ref_crop]
                data_val = (data_crop - mean_crop)**2 * weight_crop
                aux_val = (sub_aux[:, *sub_crop] - mean_crop)**2 * weight_crop if (local_aux_sum is not None and sub_aux is not None) else None
                local_data_sum[ref_crop] += data_val
                local_weight_sum[ref_crop] += weight_crop
                if aux_val is not None:
                    local_aux_sum[:, *ref_crop] += aux_val

            elif mode == 'sigma_clip':
                mean_crop = mean_map[ref_crop]
                std_crop = std_map[ref_crop]
                sigma = params['sigma']
                clip_mask = np.abs(data_crop - mean_crop) <= sigma * std_crop

                valid_data = data_crop * weight_crop * clip_mask
                valid_weight = weight_crop * clip_mask
                aux_val = sub_aux[:, *sub_crop] * valid_weight if (local_aux_sum is not None and sub_aux is not None) else None
                local_data_sum[ref_crop] += valid_data
                local_weight_sum[ref_crop] += valid_weight
                if aux_val is not None:
                    local_aux_sum[:, *ref_crop] += aux_val

    # Flush local accumulators to shared arrays (one lock per batch instead of per file)
    if mode != 'cache':
        _lock = params.get('_lock') or _coadd_flush_lock
        with _lock:
            data_sum_arr += local_data_sum
            weight_sum_arr += local_weight_sum
            if local_aux_sum is not None:
                aux_sum_arr += local_aux_sum

    # 5. Cleanup & Return
    for shm in shm_handles:
        shm.close()
    if mode == 'cache':
        return cached_list
    else:
        return None, None, None

def _coadd_batch_manager(params):
    """
    Unified manager that handles Shared Memory setup and Worker Pool execution.
    """
    mode = params['mode']
    file_list = params['file_list']
    offset_list = params['offset_list']
    max_workers = params['max_workers']
    batch_size = params['batch_size']
    mean_map = params['mean_map']
    std_map = params['std_map']
    det_aux = params['det_aux']
        
    total_files = len(file_list)
    if total_files == 0:
        if mode == 'cache': return []
        else: return np.zeros(params['ref_shape']), np.zeros(params['ref_shape'])

    ref_shape = params['ref_shape']
    shm_objects = []

    if mode == 'cache':
        # Cache mode: use Pool (real processes) for true parallelism — GIL prevents
        # threads from scaling. Move large read-only arrays to SharedMemory.
        if det_aux is not None:
            shm = SharedMemory(create=True, size=det_aux.nbytes)
            np.ndarray(det_aux.shape, dtype=det_aux.dtype, buffer=shm.buf)[:] = det_aux
            shm_objects.append(shm)
            params['det_aux_name'] = shm.name
            params['det_aux_dtype'] = det_aux.dtype
            params['det_aux_shape'] = det_aux.shape
            params['det_aux'] = None

        chunk_map = params.get('chunk_map')
        if chunk_map is not None:
            shm = SharedMemory(create=True, size=chunk_map.nbytes)
            np.ndarray(chunk_map.shape, dtype=chunk_map.dtype, buffer=shm.buf)[:] = chunk_map
            shm_objects.append(shm)
            params['chunk_map_shm_name'] = shm.name
            params['chunk_map_shape'] = chunk_map.shape
            params['chunk_map_dtype'] = chunk_map.dtype
            params['chunk_map'] = None

        gvw = params.get('grid_valid_weight')
        if gvw is not None:
            shm = SharedMemory(create=True, size=gvw.nbytes)
            np.ndarray(gvw.shape, dtype=gvw.dtype, buffer=shm.buf)[:] = gvw
            shm_objects.append(shm)
            params['gvw_shm_name'] = shm.name
            params['gvw_shape'] = gvw.shape
            params['gvw_dtype'] = gvw.dtype
            params['grid_valid_weight'] = None
    else:
        # Coadd modes: Pool (real processes) for true parallelism — same as cache mode.
        # Accumulators and read-only maps go into SharedMemory.
        use_cached = params.get('use_cached', False)

        # --- Accumulator arrays in SharedMemory ---
        data_nbytes = int(np.prod(ref_shape)) * np.dtype(np.float32).itemsize

        shm_data = SharedMemory(create=True, size=data_nbytes)
        data_arr = np.ndarray(ref_shape, dtype=np.float32, buffer=shm_data.buf)
        data_arr.fill(0)
        shm_objects.append(shm_data)
        params['total_data_sum_name'] = shm_data.name
        params.pop('total_data_sum', None)

        shm_weight = SharedMemory(create=True, size=data_nbytes)
        weight_arr = np.ndarray(ref_shape, dtype=np.float32, buffer=shm_weight.buf)
        weight_arr.fill(0)
        shm_objects.append(shm_weight)
        params['total_weight_sum_name'] = shm_weight.name
        params.pop('total_weight_sum', None)

        total_aux_shape = None
        if det_aux is not None:
            total_aux_shape = (det_aux.shape[0],) + ref_shape
            aux_nbytes = int(np.prod(total_aux_shape)) * np.dtype(np.float32).itemsize
            shm_aux_sum = SharedMemory(create=True, size=aux_nbytes)
            aux_arr_shm = np.ndarray(total_aux_shape, dtype=np.float32, buffer=shm_aux_sum.buf)
            aux_arr_shm.fill(0)
            shm_objects.append(shm_aux_sum)
            params['total_aux_sum_name'] = shm_aux_sum.name
            params['det_aux_shape'] = det_aux.shape
            params.pop('total_aux_sum', None)

        # --- Read-only maps (mean_map, std_map) in SharedMemory ---
        if mean_map is not None:
            shm = SharedMemory(create=True, size=mean_map.nbytes)
            np.ndarray(ref_shape, dtype=mean_map.dtype, buffer=shm.buf)[:] = mean_map
            shm_objects.append(shm)
            params['mean_map_name'] = shm.name
            params['mean_map_dtype'] = mean_map.dtype
            params['mean_map'] = None

        if std_map is not None:
            shm = SharedMemory(create=True, size=std_map.nbytes)
            np.ndarray(ref_shape, dtype=std_map.dtype, buffer=shm.buf)[:] = std_map
            shm_objects.append(shm)
            params['std_map_name'] = shm.name
            params['std_map_dtype'] = std_map.dtype
            params['std_map'] = None

        # --- Read-only prep arrays in SharedMemory (only needed when not using cache) ---
        if not use_cached:
            if det_aux is not None:
                shm = SharedMemory(create=True, size=det_aux.nbytes)
                np.ndarray(det_aux.shape, dtype=det_aux.dtype, buffer=shm.buf)[:] = det_aux
                shm_objects.append(shm)
                params['det_aux_name'] = shm.name
                params['det_aux_dtype'] = det_aux.dtype
                params['det_aux_shape'] = det_aux.shape

            chunk_map = params.get('chunk_map')
            if chunk_map is not None:
                shm = SharedMemory(create=True, size=chunk_map.nbytes)
                np.ndarray(chunk_map.shape, dtype=chunk_map.dtype, buffer=shm.buf)[:] = chunk_map
                shm_objects.append(shm)
                params['chunk_map_shm_name'] = shm.name
                params['chunk_map_shape'] = chunk_map.shape
                params['chunk_map_dtype'] = chunk_map.dtype
                params['chunk_map'] = None

            gvw = params.get('grid_valid_weight')
            if gvw is not None:
                shm = SharedMemory(create=True, size=gvw.nbytes)
                np.ndarray(gvw.shape, dtype=gvw.dtype, buffer=shm.buf)[:] = gvw
                shm_objects.append(shm)
                params['gvw_shm_name'] = shm.name
                params['gvw_shape'] = gvw.shape
                params['gvw_dtype'] = gvw.dtype
                params['grid_valid_weight'] = None
        else:
            # Strip large objects not needed when reading from cache
            params['det_aux'] = None
            params['chunk_map'] = None
            params['grid_valid_weight'] = None
            params['det_offset_func'] = None
            params['preprocess_func'] = None
            params['postprocess_func'] = None

    # Remove full lists — workers only need their batch slice
    params.pop('file_list', None)
    params.pop('offset_list', None)

    tasks = []
    for start_idx in range(0, total_files, batch_size):
        end_idx = min(start_idx + batch_size, total_files)
        task_params = params.copy()
        task_params.update({
            'batch_files': file_list[start_idx:end_idx],
            'batch_indices': list(range(start_idx, end_idx)),
            'batch_offsets': offset_list[start_idx:end_idx] if offset_list is not None else None,
        })
        tasks.append(task_params)

    print(f"Processing {total_files} files in {len(tasks)} batches...")

    # --- Execution ---
    try:
        if mode == 'cache':
            # Pool: real processes bypass the GIL for full CPU parallelism
            total_result = []
            with Pool(processes=max_workers) as pool:
                for res in tqdm(pool.imap_unordered(_coadd_batch_worker, tasks), total=len(tasks)):
                    total_result.extend(res)
            total_result.sort()
            return total_result
        else:
            # Pool: real processes bypass the GIL for full CPU parallelism
            mp_lock = _MPLock()
            with Pool(processes=max_workers, initializer=_init_coadd_worker, initargs=(mp_lock,)) as pool:
                for _ in tqdm(pool.imap_unordered(_coadd_batch_worker, tasks), total=len(tasks)):
                    pass  # Workers flush to SharedMemory accumulators

            # Read final results from SharedMemory (copy before cleanup)
            final_data_sum = np.ndarray(ref_shape, dtype=np.float32, buffer=shm_data.buf).copy()
            final_weight_sum = np.ndarray(ref_shape, dtype=np.float32, buffer=shm_weight.buf).copy()
            final_aux_sum = None
            if total_aux_shape is not None:
                final_aux_sum = np.ndarray(total_aux_shape, dtype=np.float32, buffer=shm_aux_sum.buf).copy()
            return final_data_sum, final_weight_sum, final_aux_sum
    finally:
        for shm in shm_objects:
            shm.close()
            shm.unlink()

def compute_coadd_map(mode, ref_shape, file_list, mean_map=None, std_map=None, sigma=3.0, 
                      offset_list=None, apply_weight=True, 
                      apply_mask=True, chunk_map=None, grid_valid_weight=None, 
                      max_workers=10, ignore_list=[], det_offset_func=None, oversample_factor=1,
                      batch_size=10, valid_threshold=0.99,
                      cache_dir='cache/', use_cached=False, det_aux=None,
                      preprocess_func=None, postprocess_func=None):
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
    file_list : list
        List of paths to the reprojected HDF5 files.
    mean_map : np.ndarray, optional
        The pre-computed mean map. Required when `mode` is 'std' or 'sigma_clip'.
    std_map : np.ndarray, optional
        The pre-computed standard deviation map. Required when `mode` is 'sigma_clip'.
    sigma : float, optional
        The number of standard deviations for sigma clipping. Used only when `mode` is 'sigma_clip'. Default is 3.0.
    offset_list : list, optional
        List of offsets for each exposure, shape (num_reproj_file, num_chunks). Default is None.
    apply_weight : bool, optional
        Whether to apply weights to the data. Default is True.
    apply_mask : bool, optional
        Whether to apply masks to the data. Default is True.
    chunk_map : dict, optional
        Mapping of chunk indices to their corresponding pixel indices. Default is None.
    grid_valid_weight : np.ndarray, optional
        Weight indicating valid pixels for each grid. Default is None.
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
    valid_threshold : float, optional
        Threshold for valid pixel fraction when applying detector valid mask. Default is 0.99.
    cache_dir : str, optional
        Directory to store or load cached intermediate results. Default is 'cache/'.
    use_cached : bool, optional
        If True, assume file_list contains cached intermediate results and load them instead of recomputing. Default is False.
    det_aux : list, optional
        Additional data that may be required for specific computations. Default is None.
    Returns
    -------
    result_map : np.ndarray
        The computed map (mean, std, or sigma-clipped mean).
    weight_sum : np.ndarray
        The sum of weights used in the calculation.
    """
    # --- Common Assertions for All Modes ---
    assert mode in ['mean', 'std', 'sigma_clip', 'cache'], "mode must be one of 'mean', 'std', 'sigma_clip', or 'cache'"
    if mode == 'cache':
        assert cache_dir is not None, "cache_dir must be provided if cache_intermediate is True"
        os.makedirs(cache_dir, exist_ok=True)
    if mode == 'std':
        assert mean_map is not None, "mean_map must be provided for 'std' mode"
    if mode == 'sigma_clip':
        assert mean_map is not None, "mean_map must be provided for 'sigma_clip' mode"
        assert std_map is not None, "std_map must be provided for 'sigma_clip' mode"
        assert isinstance(sigma, (int, float)) and sigma > 0, "sigma must be a positive number"
    assert isinstance(ref_shape, (list, np.ndarray, tuple)) and len(ref_shape) == 2, "ref_shape must be a list or tuple of length 2"
    assert isinstance(file_list, (list, np.ndarray)) and file_list, "file_list must be a non-empty list"
    # assert offset_list is None or (isinstance(offset_list, (list, np.ndarray)) and np.shape(offset_list) == (len(file_list), len(np.unique(chunk_map)))), \
    #     "offset_list must be a list or array of shape (num_reproj_file, num_chunks)"
    assert isinstance(apply_weight, bool), "apply_weight must be a boolean"
    assert isinstance(apply_mask, bool), "apply_mask must be a boolean"
    assert chunk_map is None or isinstance(chunk_map, (list, np.ndarray)), "chunk_map must be a list or array"
    assert grid_valid_weight is None or isinstance(grid_valid_weight, np.ndarray), "grid_valid_weight must be a numpy array"
    assert isinstance(max_workers, int) and max_workers > 0, "max_workers must be a positive integer"
    assert isinstance(ignore_list, (list, np.ndarray)), "ignore_list must be a list or array of data quality flags to ignore"
    assert det_offset_func is None or callable(det_offset_func), "det_offset_func must be a callable function or None"
    assert isinstance(oversample_factor, int) and oversample_factor > 0, "oversample_factor must be a positive integer"
    assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"
    assert use_cached & (mode == 'cache') == False, "use_cached and mode='cache' cannot both be True"
    if use_cached:
        assert os.path.isdir(cache_dir), "cache_dir must be a valid directory when use_cached is True"

    # Pack parameters
    params = {
        'mode': mode,
        'ref_shape': ref_shape,
        'file_list': file_list, 
        'max_workers': max_workers,
        'batch_size': batch_size,
        'mean_map': mean_map,
        'std_map': std_map,
        'det_aux': det_aux,
        
        'offset_list': offset_list,
        
        'chunk_map': chunk_map,
        'grid_valid_weight': grid_valid_weight,
        'apply_weight': apply_weight,
        'apply_mask': apply_mask,
        'ignore_list': ignore_list,
        'det_offset_func': det_offset_func,
        'oversample_factor': oversample_factor,
        'valid_threshold': valid_threshold,
        'sigma': sigma,
        'preprocess_func': preprocess_func,
        'postprocess_func': postprocess_func,
        
        'cache_dir': cache_dir,
        'use_cached': use_cached
    }
    
    # --- Branch 1: Caching Only ---
    if mode == 'cache':
        cached_list = _coadd_batch_manager(params)
        return cached_list

    # --- Branch 2: Co-addition (Standard or from Cache) ---
    else:
        data_sum, weight_sum, aux_sum = _coadd_batch_manager(params)

        if mode == 'mean':
            result_map = np.divide(data_sum, weight_sum, out=np.zeros_like(data_sum), where=weight_sum != 0)
    
        elif mode == 'std':
            sq_diff_sum = data_sum # For std mode, data_sum contains squared differences
            variance = np.divide(sq_diff_sum, weight_sum, out=np.zeros_like(sq_diff_sum), where=weight_sum > 0)
            result_map = np.sqrt(variance)

        elif mode == 'sigma_clip':
            result_map = np.divide(data_sum, weight_sum, out=np.zeros_like(data_sum), where=weight_sum != 0)
        
        aux_map = np.divide(aux_sum, weight_sum, out=np.zeros_like(aux_sum), where=weight_sum != 0) if aux_sum is not None else None

        return result_map, weight_sum, aux_map

def _prep_lsqr(task_params):
    '''Compute the components of the LSQR matrix A and vector b for a single subframe.'''
    # 1. Unpack task specific
    index = task_params['index']
    reproj_file = task_params['reproj_file']

    # 2. Unpack Config for Logic
    ref_shape = task_params['ref_shape']
    num_frames = task_params['num_frames']
    num_chunks = task_params['num_chunks']
    outlier_thresh = task_params['outlier_thresh']
    reg_weight = task_params['reg_weight']
    offset_regularization = task_params['offset_regularization']
    adj_info = task_params['adj_info'] # Pre-computed adjacency (row, col) pairs
    frame_to_group = task_params['frame_to_group']
    scalar_col_start = task_params['scalar_col_start']
    num_scalar_cols = task_params['num_scalar_cols']
    det_template = task_params.get('det_template')
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    group_idx = frame_to_group[index]

    try:
        # 3. Explicit Call to _prep_subframe
        ref_coords, sub_data, sub_weight, chunk_contrib, _ = _prep_subframe(
            file=reproj_file,
            chunk_offset=None,
            for_lsqr=True,
            det_offset_func=None,
            det_aux=None,
            chunk_map=task_params['chunk_map'],
            apply_weight=task_params['apply_weight'],
            apply_mask=task_params['apply_mask'],
            ignore_list=task_params['ignore_list'],
            grid_valid_weight=task_params['grid_valid_weight'],
            oversample_factor=task_params['oversample_factor'],
            valid_threshold=task_params['valid_threshold'],
            postprocess_func=task_params['postprocess_func'],
            preprocess_func=task_params['preprocess_func']
        )

        sub_h, sub_w = sub_data.shape
        
        sub_valid = sub_weight > 0
        if isinstance(outlier_thresh, (int, float)) and outlier_thresh > 0:
            sub_out = find_outliers(np.where(sub_valid, sub_data, np.nan), threshold=outlier_thresh)
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

        if det_template is not None:
            # Template mode: single alpha column per frame, weighted by template
            O_cols = np.full(len(chunk_idx), num_sky + index, dtype=np.int64)
            O_data = valid_weight[sub_idx] * chunk_vals * det_template[group_idx, chunk_idx]
        else:
            O_cols = num_sky + (group_idx * num_chunks) + chunk_idx
            O_data = valid_weight[sub_idx] * chunk_vals

        sub_b = valid_vals * valid_weight

        # --- Spatial Regularization (Adjacency) ---
        # Skip in template mode (single scalar per frame, no spatial structure to regularize)
        reg_rows, reg_cols, reg_data, reg_b = [], [], [], []
        if reg_weight > 0 and adj_info is not None and offset_regularization and det_template is None:
            # adj_info is expected to be a tuple of (chunk_i, chunk_j) indices that are neighbors
            chunk_i, chunk_j = adj_info
            num_constraints = len(chunk_i)
            offset_base = num_sky + (group_idx * num_chunks)
            
            # Constraint: reg_weight * (O_i - O_j) = 0
            # Equations start after the data equations (num_valid_pixels)
            reg_rows = np.repeat(np.arange(num_constraints) + num_valid_pixels, 2)
            reg_cols = np.stack([offset_base + chunk_i, offset_base + chunk_j], axis=1).flatten()
            reg_data = np.tile([reg_weight, -reg_weight], num_constraints)
            reg_b = np.zeros(num_constraints)

        # Per-frame scalar term (one column per frame, applied to every valid pixel)
        Sc_rows, Sc_cols, Sc_data = [], [], []
        if num_scalar_cols > 0:
            scalar_col = scalar_col_start + index
            Sc_rows = np.arange(num_valid_pixels)
            Sc_cols = np.full(num_valid_pixels, scalar_col, dtype=np.int64)
            Sc_data = valid_weight

        # Concatenate Data, Offset, Scalar, and Regularization
        parts_rows = [S_rows, O_rows]
        parts_cols = [S_cols, O_cols]
        parts_data = [S_data, O_data]
        if len(Sc_rows) > 0:
            parts_rows.append(Sc_rows)
            parts_cols.append(Sc_cols)
            parts_data.append(Sc_data)
        if len(reg_rows) > 0:
            parts_rows.append(reg_rows)
            parts_cols.append(reg_cols)
            parts_data.append(reg_data)
        sub_rows = np.concatenate(parts_rows)
        sub_cols = np.concatenate(parts_cols)
        sub_data_vec = np.concatenate(parts_data)
        sub_b = np.concatenate([sub_b, reg_b]) if len(reg_b) > 0 else sub_b

        valid_mask = ~check_invalid(sub_b[sub_rows]) & ~((sub_data_vec == 0) & (sub_b[sub_rows] == 0))
        sub_rows = sub_rows[valid_mask]
        sub_cols = sub_cols[valid_mask]
        sub_data_vec = sub_data_vec[valid_mask]

        unique_rows, new_row_indices = np.unique(sub_rows, return_inverse=True)
        sub_rows = new_row_indices.astype(np.int32)
        sub_cols = sub_cols.astype(np.int32)
        sub_data_vec = sub_data_vec.astype(np.float32)
        sub_b = sub_b[unique_rows]

        return sub_rows, sub_cols, sub_data_vec, sub_b, len(sub_b)

    except Exception as e:
        print(f"Error processing file {reproj_file}: {e}")
        traceback.print_exc()
        return None

def _prep_lsqr_batch_worker(batch_params):
    """Wrapper to process a list (batch) of subframes in a single worker process."""
    sub_tasks = batch_params['sub_tasks']

    # Reconstruct shared memory arrays once per batch (avoids per-file overhead)
    shm_handles = []
    shm_arrays = {}

    if 'chunk_map_shm_name' in sub_tasks[0]:
        shm_cm = SharedMemory(name=sub_tasks[0]['chunk_map_shm_name'])
        shm_arrays['chunk_map'] = np.ndarray(sub_tasks[0]['chunk_map_shape'], dtype=sub_tasks[0]['chunk_map_dtype'], buffer=shm_cm.buf)
        shm_handles.append(shm_cm)

    if 'gvw_shm_name' in sub_tasks[0]:
        shm_gvw = SharedMemory(name=sub_tasks[0]['gvw_shm_name'])
        shm_arrays['grid_valid_weight'] = np.ndarray(sub_tasks[0]['gvw_shape'], dtype=sub_tasks[0]['gvw_dtype'], buffer=shm_gvw.buf)
        shm_handles.append(shm_gvw)

    if 'adj_shm_name_0' in sub_tasks[0]:
        adj_parts = []
        for idx in range(2):
            shm = SharedMemory(name=sub_tasks[0][f'adj_shm_name_{idx}'])
            adj_parts.append(np.ndarray(sub_tasks[0][f'adj_shape_{idx}'], dtype=sub_tasks[0][f'adj_dtype_{idx}'], buffer=shm.buf))
            shm_handles.append(shm)
        shm_arrays['adj_info'] = tuple(adj_parts)

    try:
        batch_rows = []
        batch_cols = []
        batch_data = []
        batch_b = []
        batch_row_offset = 0

        for task_params in sub_tasks:
            # Inject reconstructed shared memory arrays
            task_params.update(shm_arrays)

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

        # Write results to shared memory to avoid pickle/pipe IPC overhead
        cat_rows = np.concatenate(batch_rows)
        cat_cols = np.concatenate(batch_cols)
        cat_data = np.concatenate(batch_data)
        cat_b = np.concatenate(batch_b)

        result_shm = []
        for arr in (cat_rows, cat_cols, cat_data, cat_b):
            shm = SharedMemory(create=True, size=max(arr.nbytes, 1))
            np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
            result_shm.append((shm.name, arr.shape, arr.dtype.str))
            shm.close()

        return {'shm': result_shm, 'num_rows': batch_row_offset}
    finally:
        for shm in shm_handles:
            shm.close()

def setup_lsqr(file_list, ref_shape, 
               chunk_map=None, grid_valid_weight=None, apply_mask=True, apply_weight=False, 
               valid_threshold=0.99,
               outlier_thresh=3, max_workers=20, ignore_list=[], oversample_factor=1, batch_size=10, offset_regularization=False,
               reg_weight=0.0, adj_info=None, mean_offsets=None, det_groups=None, det_template=None, postprocess_func=None, preprocess_func=None,
               weighted_damping=False, damp_weight=0.1):
    """Prepares the LSQR matrix A and vector b for all subframes in parallel.
    Parameters
    ----------
    file_list : list
        List of paths to the reprojected HDF5 files
    ref_shape : tuple, list
        Shape of the reference frame (height, width)
    chunk_map : np.ndarray, optional
        Mapping of chunk indices to their corresponding pixel indices.
        Must be 0 indexed and continuous!!
    grid_valid_weight : np.ndarray, optional
        Weight indicating valid pixels for each grid pixel.
    apply_mask : bool, optional
        Whether to apply masks to the data.
        Default is True.
    apply_weight : bool, optional
        Whether to apply weights to the data.
        Default is True.
    outlier_thresh : float, optional
        z-value threshold for outlier detection, default is 3.0.
    max_workers : int, optional
        Maximum number of worker processes to use for parallel processing, default is 20.
    ignore_list : list, optional
        List of data quality flags to ignore, default is an empty list.
    oversample_factor : int, optional
        Factor by which the chunk map is oversampled.
        Default is 1.
    batch_size : int, optional
        Number of files to process per worker task.
        Default is 10.
    reg_weight : float, optional
        Weight for spatial regularization between adjacent detector chunks.
    adj_info : tuple or None, optional
        Precomputed adjacency information for regularization.
    mean_offsets : list or np.ndarray, optional
        A list of target mean offset values for each frame (length must equal num_frames).
        This forces the average of a frame's chunk offsets to equal the given value.
    Returns
    -------
    full_A : scipy.sparse.coo_matrix
        The sparse matrix A in COO format, shape is (num_equations, num_unknowns)
    full_b : np.ndarray
        The vector b, shape is (num_equations,)
    """
    assert isinstance(file_list, (list, np.ndarray)) and file_list, "file_list must be a non-empty list"
    assert isinstance(ref_shape, (list, np.ndarray, tuple)) and len(ref_shape) == 2, "ref_shape must be a list of length 2"
    assert chunk_map is None or isinstance(chunk_map, np.ndarray), "chunk_map must be a numpy array"
    assert grid_valid_weight is None or isinstance(grid_valid_weight, np.ndarray), "grid_valid_weight must be a numpy array"
    assert isinstance(apply_mask, bool), "apply_mask must be a boolean"
    assert isinstance(apply_weight, bool), "apply_weight must be a boolean"
    assert isinstance(outlier_thresh, (int, float, type(None))) and (outlier_thresh is None or outlier_thresh > 0), "outlier_thresh must be a positive number or None"
    assert isinstance(max_workers, int) and max_workers > 0, "max_workers must be a positive integer"
    assert isinstance(ignore_list, (list, np.ndarray)), "ignore_list must be a list or array of data quality flags to ignore"
    assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"
    # if postprocess_func is not None:
    #     assert callable(postprocess_func), "postprocess_func must be a callable function"
    #     test_data = np.random.rand(100, 100).astype(np.float32)
    #     assert test_data.shape == postprocess_func(test_data, np.ones_like(test_data))[0].shape, \
    #         "postprocess_func must return data and weight arrays of the same shape as input"

    # if preprocess_func is not None:
    #     assert callable(preprocess_func), "preprocess_func must be a callable function"
    #     test_data = np.random.rand(100, 100).astype(np.float32)
    #     assert test_data.shape == preprocess_func(test_data, np.ones_like(test_data))[0].shape, \
    #         "preprocess_func must return data and weight arrays of the same shape as input"

    num_chunks = int(chunk_map.max()) + 1 if chunk_map is not None else 0
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    num_frames = len(file_list)

    # Build group mapping for detector offset locking
    if det_groups is not None:
        det_groups_arr = np.asarray(det_groups)
        unique_groups, frame_to_group = np.unique(det_groups_arr, return_inverse=True)
        num_offset_groups = len(unique_groups)
        num_scalar_cols = num_frames
        print(f"Locking detector offsets: {num_frames} frames -> {num_offset_groups} groups + {num_frames} frame scalars")
    else:
        frame_to_group = np.arange(num_frames)
        num_offset_groups = num_frames
        num_scalar_cols = 0

    # Template mode: fix spatial pattern, solve for per-frame amplitude
    if det_template is not None:
        assert det_groups is not None, "det_template requires det_groups"
        det_template = np.asarray(det_template, dtype=np.float32)
        total_cols = num_sky + num_frames + num_scalar_cols
        scalar_col_start = num_sky + num_frames
        print(f"Template mode: {num_frames} alpha unknowns (pattern fixed from template)")
    else:
        total_cols = num_sky + num_chunks * num_offset_groups + num_scalar_cols
        scalar_col_start = num_sky + num_chunks * num_offset_groups

    common_params = {
        'chunk_map': chunk_map,
        'grid_valid_weight': grid_valid_weight,
        'apply_mask': apply_mask,
        'apply_weight': apply_weight,
        'ignore_list': ignore_list,
        'oversample_factor': oversample_factor,
        'valid_threshold': valid_threshold,
        'outlier_thresh': outlier_thresh,
        'num_chunks': num_chunks,
        'num_frames': num_frames,
        'ref_shape': ref_shape,
        'offset_regularization': offset_regularization,
        'reg_weight': reg_weight,
        'adj_info': adj_info,
        'postprocess_func': postprocess_func,
        'preprocess_func': preprocess_func,
        'frame_to_group': frame_to_group,
        'scalar_col_start': scalar_col_start,
        'num_scalar_cols': num_scalar_cols,
        'det_template': det_template,
    }

    # Move large arrays to shared memory so forked processes can access them
    # without pickling. Each process reconstructs numpy views in the worker.
    shm_objects = []

    if chunk_map is not None:
        shm_cm = SharedMemory(create=True, size=chunk_map.nbytes)
        np.ndarray(chunk_map.shape, dtype=chunk_map.dtype, buffer=shm_cm.buf)[:] = chunk_map
        shm_objects.append(shm_cm)
        common_params['chunk_map_shm_name'] = shm_cm.name
        common_params['chunk_map_shape'] = chunk_map.shape
        common_params['chunk_map_dtype'] = chunk_map.dtype
        common_params['chunk_map'] = None

    if grid_valid_weight is not None:
        shm_gvw = SharedMemory(create=True, size=grid_valid_weight.nbytes)
        np.ndarray(grid_valid_weight.shape, dtype=grid_valid_weight.dtype, buffer=shm_gvw.buf)[:] = grid_valid_weight
        shm_objects.append(shm_gvw)
        common_params['gvw_shm_name'] = shm_gvw.name
        common_params['gvw_shape'] = grid_valid_weight.shape
        common_params['gvw_dtype'] = grid_valid_weight.dtype
        common_params['grid_valid_weight'] = None

    if adj_info is not None:
        for idx, arr in enumerate(adj_info):
            shm = SharedMemory(create=True, size=arr.nbytes)
            np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
            shm_objects.append(shm)
            common_params[f'adj_shm_name_{idx}'] = shm.name
            common_params[f'adj_shape_{idx}'] = arr.shape
            common_params[f'adj_dtype_{idx}'] = arr.dtype
        common_params['adj_info'] = None

    all_individual_tasks = []
    for index, reproj_file in enumerate(file_list):
        task_params = {'index': index, 'reproj_file': reproj_file}
        task_params.update(common_params)
        all_individual_tasks.append(task_params)

    batched_tasks = []
    for i in range(0, len(all_individual_tasks), batch_size):
        batch = {'sub_tasks': all_individual_tasks[i : i + batch_size]}
        batched_tasks.append(batch)

    print(f"Processing {len(all_individual_tasks)} items in {len(batched_tasks)} batches...")

    all_rows, all_cols, all_data, all_b = [], [], [], []

    def _read_shm(info):
        """Read array from shared memory and clean up the segment."""
        name, shape, dtype = info
        shm = SharedMemory(name=name)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
        shm.close()
        shm.unlink()
        return arr

    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_prep_lsqr_batch_worker, batch): i for i, batch in enumerate(batched_tasks)}

            total_rows = 0
            n_collected = 0
            for future in tqdm(as_completed(futures), total=len(futures), desc="Building A, b matrix"):
                result = future.result()
                if result is None: continue

                shm_infos = result['shm']
                b_rows = _read_shm(shm_infos[0]).astype(np.int64) + total_rows
                b_cols = _read_shm(shm_infos[1])
                b_data = _read_shm(shm_infos[2])
                b_b = _read_shm(shm_infos[3])
                all_rows.append(b_rows)
                all_cols.append(b_cols)
                all_data.append(b_data)
                all_b.append(b_b)
                total_rows += result['num_rows']

                # Consolidate periodically to avoid memory fragmentation
                n_collected += 1
                if n_collected % 100 == 0:
                    all_rows = [np.concatenate(all_rows)]
                    all_cols = [np.concatenate(all_cols)]
                    all_data = [np.concatenate(all_data)]
                    all_b = [np.concatenate(all_b)]
    finally:
        for shm in shm_objects:
            shm.close()
            shm.unlink()

    if len(all_b) == 0:
        print("No valid data found in any subframe.")
        return None, None

    # Overwrite the lists with a single concatenated array 
    # This prepares them for appending and lets us access the combined cols safely
    all_rows = [np.concatenate(all_rows)]
    all_cols = [np.concatenate(all_cols)]
    all_data = [np.concatenate(all_data)]
    all_b = [np.concatenate(all_b)]

    # Ensure total_rows is a Python int to avoid numpy int32 overflow in
    # subsequent constraint blocks that may push row counts beyond 2^31.
    total_rows = int(total_rows)

    # --- MEMORY OPTIMIZED: Calculate valid pixel fractions AND coverage map ---
    # We access the combined columns via all_cols[0]
    pixel_counts = np.bincount(all_cols[0], minlength=total_cols)
    sky_pixel_counts = pixel_counts[:num_sky]
    offset_pixel_counts = pixel_counts[num_sky:]

    # --- ADD PER-FRAME TARGET MEAN CONSTRAINT ---
    #TODO: Pass weight from higher level instead of hardcoding here
    if mean_offsets is not None:
        print(f"Applying target mean offset constraints for {num_frames} frames...")
        mean_offsets_arr = np.asarray(mean_offsets)
        
        constraint_weight = 10.0 
        
        constr_rows = total_rows + np.repeat(np.arange(num_frames), num_chunks)
        
        constr_cols = []
        for i in range(num_frames):
            offset_start = num_sky + (frame_to_group[i] * num_chunks)
            constr_cols.extend(np.arange(offset_start, offset_start + num_chunks))
        
        constr_data = np.ones(len(constr_cols), dtype=np.float32) * constraint_weight
        b_constr = mean_offsets_arr.flatten() * num_chunks * constraint_weight
        
        # Append directly to our existing lists
        all_rows.append(constr_rows)
        all_cols.append(np.array(constr_cols))
        all_data.append(constr_data)
        all_b.append(b_constr)
        
        total_rows += num_frames

    # --- COVERAGE-WEIGHTED DAMPING ---
    if weighted_damping and damp_weight > 0:
        print("Applying Coverage-Weighted Damping...")
        
        valid_pixel_indices = np.nonzero(sky_pixel_counts)[0]
        
        if len(valid_pixel_indices) > 0:
            damp_values = np.sqrt(damp_weight * sky_pixel_counts[valid_pixel_indices])
            num_damp_constraints = len(valid_pixel_indices)
            
            damp_rows = total_rows + np.arange(num_damp_constraints)
            damp_cols = valid_pixel_indices
            b_damp = np.zeros(num_damp_constraints)
            
            # Append directly to our existing lists
            all_rows.append(damp_rows)
            all_cols.append(damp_cols)
            all_data.append(damp_values)
            all_b.append(b_damp)
            
            total_rows += num_damp_constraints

    # --- FINAL SPARSE MATRIX CONSTRUCTION ---
    # One final concatenation of the main data + new constraints
    full_A = coo_matrix((np.concatenate(all_data), 
                        (np.concatenate(all_rows), np.concatenate(all_cols))), 
                        shape=(total_rows, total_cols))
    
    full_b = np.concatenate(all_b)

    return full_A, full_b, pixel_counts

def parse_pixel_counts(pixel_counts, ref_shape, num_offset_groups, chunk_map):
    num_sky = ref_shape[0] * ref_shape[1]
    num_chunks = int(np.max(chunk_map)) + 1
    num_offset = num_offset_groups * num_chunks
    skymap_coverage = pixel_counts[:num_sky].reshape(ref_shape)
    offset_coverage = pixel_counts[num_sky:num_sky + num_offset].reshape(num_offset_groups, num_chunks)
    chunk_sizes = np.bincount(chunk_map[chunk_map >= 0].ravel(), minlength=num_chunks)
    offset_valid_frac = (offset_coverage / np.maximum(chunk_sizes, 1))
    return skymap_coverage, offset_coverage, offset_valid_frac

def _partition_csr(A, n_blocks):
    """Split CSR matrix into row-blocks sharing data/indices arrays (zero-copy)."""
    n_rows = A.shape[0]
    boundaries = np.linspace(0, n_rows, n_blocks + 1, dtype=int)
    blocks = []
    for i in range(n_blocks):
        sr, er = int(boundaries[i]), int(boundaries[i + 1])
        nnz_s, nnz_e = A.indptr[sr], A.indptr[er]
        blk = csr_matrix(
            (A.data[nnz_s:nnz_e], A.indices[nnz_s:nnz_e], A.indptr[sr:er+1] - nnz_s),
            shape=(er - sr, A.shape[1]), copy=False
        )
        blocks.append(blk)
    return blocks, boundaries

def _make_parallel_operator(A_csr, n_threads):
    """Build a LinearOperator with thread-parallel matvec/rmatvec.

    Pre-computes A^T as CSR and partitions both into row-blocks.
    GIL is released during scipy CSR SpMV, enabling true thread parallelism.
    """
    m, n = A_csr.shape

    print(f"Building parallel SpMV operator ({n_threads} threads)...")
    AT_csr = A_csr.T.tocsr()

    A_blocks, A_bounds = _partition_csr(A_csr, n_threads)
    AT_blocks, AT_bounds = _partition_csr(AT_csr, n_threads)

    executor = ThreadPoolExecutor(max_workers=n_threads)
    mv_out = np.empty(m, dtype=A_csr.dtype)
    rmv_out = np.empty(n, dtype=A_csr.dtype)

    def _matvec(x):
        def _work(i):
            mv_out[A_bounds[i]:A_bounds[i+1]] = A_blocks[i] @ x
        list(executor.map(_work, range(n_threads)))
        return mv_out.copy()

    def _rmatvec(y):
        def _work(i):
            rmv_out[AT_bounds[i]:AT_bounds[i+1]] = AT_blocks[i] @ y
        list(executor.map(_work, range(n_threads)))
        return rmv_out.copy()

    op = LinearOperator((m, n), matvec=_matvec, rmatvec=_rmatvec, dtype=A_csr.dtype)
    op._executor = executor
    op._AT_csr = AT_csr  # prevent GC
    return op

def apply_lsqr(A, b, ref_shape, num_offset_groups, x0=None,
                atol=1e-05, btol=1e-05, damp=1e-2, iter_lim=100, precondition=True,
                solver='lsmr', use_float32=False, n_threads=32):
    """Applies LSQR or LSMR to solve for the sky and detector offsets.

    Parameters
    ----------
    solver : str, optional
        Solver to use: 'lsmr' (default, faster convergence) or 'lsqr'.
    use_float32 : bool, optional
        If True, cast matrix data and b to float32 before solving.
        Reduces memory bandwidth (~2x faster SpMV) at the cost of precision.
    """
    assert isinstance(A, coo_matrix), "A must be a scipy.sparse.coo_matrix"
    assert isinstance(b, np.ndarray), "b must be a numpy array"
    assert isinstance(ref_shape, (list, np.ndarray, tuple)) and len(ref_shape) == 2, "ref_shape must be a list or tuple of length 2"

    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    num_cols = A.shape[1]

    # --- Fused preprocessing: column elimination + float32 + preconditioning + CSR ---
    col_nnz = np.bincount(A.col, minlength=num_cols)
    active_mask = col_nnz > 0
    num_active = int(np.sum(active_mask))

    if num_active < num_cols:
        print(f"Eliminating {num_cols - num_active} zero columns ({num_active}/{num_cols} active)...")
        col_map = np.full(num_cols, -1, dtype=A.col.dtype)
        col_map[active_mask] = np.arange(num_active, dtype=A.col.dtype)
        new_col = col_map[A.col]
        x0_compressed = x0[active_mask] if x0 is not None else None
    else:
        new_col = A.col
        x0_compressed = x0
        active_mask = None

    n_active = num_active if active_mask is not None else num_cols

    if use_float32:
        print("Downcasting to float32 for faster SpMV...")
        data = A.data.astype(np.float32)
        b = b.astype(np.float32)
        if x0_compressed is not None:
            x0_compressed = x0_compressed.astype(np.float32)
    else:
        data = A.data

    if precondition:
        print("Applying column-norm preconditioning...")
        col_sq_norm = np.bincount(new_col, weights=data.astype(np.float64)**2, minlength=n_active)
        col_norms = np.sqrt(col_sq_norm)
        col_norms[col_norms == 0] = 1.0
        M_inv = col_norms
        M = 1.0 / M_inv
        data = data * M[new_col].astype(data.dtype)
        x0_solver = x0_compressed * M_inv.astype(x0_compressed.dtype) if x0_compressed is not None else None
    else:
        M = None
        x0_solver = x0_compressed

    print(f"Solving least squares for {n_active} unknowns with {A.shape[0]} equations (solver={solver}).")
    A_csr = coo_matrix((data, (A.row, new_col)), shape=(A.shape[0], n_active)).tocsr()
    del data, new_col

    # --- Build parallel operator or use CSR directly ---
    if n_threads > 1:
        op = _make_parallel_operator(A_csr, n_threads)
        try:
            with threadpool_limits(limits=1, user_api='blas'):
                if solver == 'lsmr':
                    result = lsmr(op, b, x0=x0_solver, show=True, atol=atol, btol=btol, damp=damp, maxiter=iter_lim)
                elif solver == 'lsqr':
                    result = lsqr(op, b, x0=x0_solver, show=True, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim)
                else:
                    raise ValueError(f"Unknown solver: {solver}. Use 'lsqr' or 'lsmr'.")
        finally:
            op._executor.shutdown(wait=False)
    else:
        if solver == 'lsmr':
            result = lsmr(A_csr, b, x0=x0_solver, show=True, atol=atol, btol=btol, damp=damp, maxiter=iter_lim)
        elif solver == 'lsqr':
            result = lsqr(A_csr, b, x0=x0_solver, show=True, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim)
        else:
            raise ValueError(f"Unknown solver: {solver}. Use 'lsqr' or 'lsmr'.")
    x_solver = result[0]
    del A_csr

    # --- Undo preconditioning ---
    if precondition:
        x_solver = x_solver * M

    # --- Expand back to full column space ---
    if active_mask is not None:
        x = np.zeros(num_cols, dtype=x_solver.dtype)
        x[active_mask] = x_solver
    else:
        x = x_solver

    return x

def parse_x(x, ref_shape, num_offset_groups, num_chunks, num_frames=None):
    """Parse the LSQR solution vector x into sky, detector offset, and frame scalar components.

    Parameters
    ----------
    num_offset_groups : int
        Number of offset groups (= num_frames when det_groups is not used).
    num_chunks : int
        Number of chunks per offset group.
    num_frames : int or None
        If not None, the last num_frames entries are per-frame scalars.
    """
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    num_offset = num_offset_groups * num_chunks
    skymap = x[:num_sky].reshape(ref_shape)
    det_offset = x[num_sky:num_sky + num_offset].reshape(num_offset_groups, num_chunks)
    frame_scalar = x[num_sky + num_offset:] if num_frames else np.array([])
    return skymap, det_offset, frame_scalar

def encode_x(skymap, offset):
    """Utility function to encode the sky and offset components back into a single vector x."""
    return np.concatenate([skymap.flatten(), offset.flatten()])

def compute_x0_from_Ab(A, b, ref_shape, num_offset_groups=None):
    """Compute initial guess x0 assuming sky=0, solving offset = A_off^T b / A_off^T A_off diag.

    This avoids re-reading all FITS files to estimate offsets — the information
    is already encoded in the sparse matrix A and vector b from setup_lsqr.
    """
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    num_cols = A.shape[1]

    # Extract offset portion of A (columns num_sky onwards)
    offset_mask = A.col >= num_sky
    off_row = A.row[offset_mask]
    off_col = A.col[offset_mask] - num_sky
    off_data = A.data[offset_mask]

    num_offset_cols = num_cols - num_sky

    # offset_j = (A_off[:, j]^T @ b) / (A_off[:, j]^T @ A_off[:, j])
    AtA_diag = np.bincount(off_col, weights=off_data ** 2, minlength=num_offset_cols)
    Atb = np.bincount(off_col, weights=off_data * b[off_row], minlength=num_offset_cols)

    offsets = np.where(AtA_diag > 0, Atb / AtA_diag, 0.0)

    x0 = np.zeros(num_cols)
    x0[num_sky:] = offsets
    return x0
