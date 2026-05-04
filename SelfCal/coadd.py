"""Co-addition pipeline: mean, std, sigma-clipped maps, and intermediate caching."""

import os
import h5py
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, Lock as _MPLock
from multiprocessing.shared_memory import SharedMemory

from . import _state
from .subframe import _prep_subframe
from .MapHelper import compute_crop


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
        _lock = params.get('_lock') or _state._coadd_flush_lock
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
            with Pool(processes=max_workers, initializer=_state._init_coadd_worker, initargs=(mp_lock,)) as pool:
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
