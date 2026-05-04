"""LSQR/LSMR matrix construction, parallel solving, and pixel count parsing."""

import os
import traceback
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor
from multiprocessing.shared_memory import SharedMemory
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import lsqr, lsmr, LinearOperator
from threadpoolctl import threadpool_limits

from .subframe import _prep_subframe
from .MapHelper import find_outliers, check_invalid


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
