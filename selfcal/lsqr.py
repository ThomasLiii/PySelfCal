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
from .layout import SystemLayout
from .sky_model import SkyModel
from .constraint_builders import (mean_offset_block, sky_damping_block,
                                  offset_damping_block)


def _prep_lsqr(task_params):
    '''Compute the components of the LSQR matrix A and vector b for a single subframe.'''
    # 1. Unpack task specific
    index = task_params['index']
    reproj_file = task_params['reproj_file']

    # 2. Unpack Config for Logic (per-map lists are length K; col_bases is length K+1)
    ref_shape = task_params['ref_shape']
    num_frames = task_params['num_frames']
    num_chunks_list = task_params['num_chunks_list']
    outlier_thresh = task_params['outlier_thresh']
    reg_weight_list = task_params['reg_weight_list']
    offset_regularization = task_params['offset_regularization']
    adj_info_list = task_params['adj_info_list']
    poly_constraint_list = task_params['poly_constraint_list']
    frame_to_group_list = task_params['frame_to_group_list']
    col_bases = task_params['col_bases']
    scalar_col_start = task_params['scalar_col_start']
    num_scalar_cols = task_params['num_scalar_cols']
    det_template_list = task_params['det_template_list']
    chunk_maps = task_params['chunk_maps']
    K = len(chunk_maps)

    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    group_idx_list = [frame_to_group_list[m][index] for m in range(K)]

    try:
        # 3. Explicit Call to _prep_subframe — returns one chunk_contrib per map
        ref_coords, sub_data, sub_weight, chunk_contribs, sub_aux = _prep_subframe(
            file=reproj_file,
            chunk_offsets=None,
            for_lsqr=True,
            det_offset_funcs=None,
            det_aux=task_params.get('det_aux'),
            chunk_maps=chunk_maps,
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

        # --- Sky rows: one nnz per sky component per data row ---
        # SkyModel generalizes the legacy num_sky_blocks {1,2} cases. Each
        # component j contributes a coefficient over the valid pixels:
        #   - None  -> identity (continuum): store valid_weight directly.
        #   - array -> e.g. line profile G(λ) (LineComponent), store w_i * coeff.
        # J==1 with an identity coefficient takes the fast path (no interleave,
        # no multiply). J>=2 interleaves S_cols[j::J] = j*num_sky + P and
        # S_data[j::J]. With components [continuum] or [continuum, pah_3p29
        # Gaussian] this reproduces the old single/two-block emission byte-for-
        # byte (same order, same float ops, same dtypes). aux maps (BC/BW) are
        # sampled to the valid pixels and passed by name to each component.
        sky_components = task_params.get('sky_components')
        if sky_components is None:
            J = 1
            sky_coeffs = [None]
        else:
            J = len(sky_components)
            aux = {}
            if sub_aux is not None:
                aux_keys = task_params.get('aux_keys') or []
                for i, k in enumerate(aux_keys):
                    aux[k] = sub_aux[i][valid_sub_coords]
            sky_coeffs = [c.coefficients(aux) for c in sky_components]

        if J == 1 and sky_coeffs[0] is None:
            S_rows = np.arange(num_valid_pixels)
            S_cols = ref_pix_indices
            S_data = valid_weight
        else:
            S_rows = np.repeat(np.arange(num_valid_pixels, dtype=np.int32), J)
            S_cols = np.empty(J * num_valid_pixels, dtype=np.int64)
            S_data = np.empty(J * num_valid_pixels, dtype=np.float32)
            for j in range(J):
                S_cols[j::J] = j * num_sky + ref_pix_indices
                cj = sky_coeffs[j]
                S_data[j::J] = valid_weight if cj is None else valid_weight * cj

        # --- Offset rows: one block per chunk map ---
        O_rows_parts, O_cols_parts, O_data_parts = [], [], []
        for m in range(K):
            cc_m = chunk_contribs[m]
            # Slice the chunk-contrib columns for the valid pixels ONCE, then
            # derive (chunk_idx, sub_idx, vals) from that single COO object.
            # The previous code built cc_m[:, sub_pix_indices] twice (once for
            # .nonzero(), once for .A[0] value extraction). Filtering tocoo() to
            # numerically-nonzero entries reproduces .nonzero()'s exact set, so
            # the assembled matrix is bit-identical.
            sliced_m = cc_m[:, sub_pix_indices].tocoo()
            nz_m = sliced_m.data != 0
            chunk_idx_m = sliced_m.row[nz_m]
            sub_idx_m = sliced_m.col[nz_m]
            chunk_vals_m = sliced_m.data[nz_m]
            O_rows_parts.append(sub_idx_m)
            if det_template_list[m] is not None:
                # Template mode: one alpha column per frame for this map
                O_cols_parts.append(np.full(len(chunk_idx_m), col_bases[m] + index, dtype=np.int64))
                O_data_parts.append(valid_weight[sub_idx_m] * chunk_vals_m
                                    * det_template_list[m][group_idx_list[m], chunk_idx_m])
            else:
                O_cols_parts.append(col_bases[m]
                                    + (group_idx_list[m] * num_chunks_list[m])
                                    + chunk_idx_m)
                O_data_parts.append(valid_weight[sub_idx_m] * chunk_vals_m)
        O_rows = np.concatenate(O_rows_parts) if O_rows_parts else np.empty(0, dtype=np.int64)
        O_cols = np.concatenate(O_cols_parts) if O_cols_parts else np.empty(0, dtype=np.int64)
        O_data = np.concatenate(O_data_parts) if O_data_parts else np.empty(0, dtype=np.float64)

        sub_b = valid_vals * valid_weight

        # --- Spatial Regularization (Adjacency + polynomial-order constraints) ---
        # One block per map; skipped for any map in template mode.
        reg_rows, reg_cols, reg_data, reg_b = [], [], [], []
        if offset_regularization:
            reg_rows_parts, reg_cols_parts, reg_data_parts, reg_b_parts = [], [], [], []
            reg_row_offset = num_valid_pixels
            for m in range(K):
                if det_template_list[m] is not None:
                    continue
                offset_base_m = col_bases[m] + (group_idx_list[m] * num_chunks_list[m])

                rw_m = reg_weight_list[m]
                if rw_m > 0 and adj_info_list[m] is not None:
                    chunk_i, chunk_j = adj_info_list[m]
                    num_constraints = len(chunk_i)
                    # Constraint: rw_m * (O_i - O_j) = 0; rows start after data equations
                    # plus any earlier maps' constraint rows.
                    reg_rows_parts.append(np.repeat(np.arange(num_constraints) + reg_row_offset, 2))
                    reg_cols_parts.append(np.stack([offset_base_m + chunk_i,
                                                    offset_base_m + chunk_j], axis=1).flatten())
                    reg_data_parts.append(np.tile([rw_m, -rw_m], num_constraints))
                    reg_b_parts.append(np.zeros(num_constraints))
                    reg_row_offset += num_constraints

                # Polynomial-order constraints: λ · Σ_ℓ stencil[ℓ] · O[chains[r, ℓ]] = 0
                # per chain r. Generalizes the [1,-1] adjacency stencil to arbitrary
                # length-L stencils on user-supplied chunk-id chains.
                groups_m = poly_constraint_list[m]
                if groups_m:
                    for grp in groups_m:
                        chains = grp['chains']
                        stencil = grp['stencil']
                        weight = grp['weight']
                        if weight == 0 or chains.shape[0] == 0:
                            continue
                        num_chains, L = chains.shape
                        reg_rows_parts.append(
                            np.repeat(np.arange(num_chains) + reg_row_offset, L))
                        reg_cols_parts.append((offset_base_m + chains).reshape(-1))
                        reg_data_parts.append(
                            np.tile(weight * stencil.astype(np.float64), num_chains))
                        reg_b_parts.append(np.zeros(num_chains))
                        reg_row_offset += num_chains
            if reg_rows_parts:
                reg_rows = np.concatenate(reg_rows_parts)
                reg_cols = np.concatenate(reg_cols_parts)
                reg_data = np.concatenate(reg_data_parts)
                reg_b = np.concatenate(reg_b_parts)

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

    chunk_maps_meta = sub_tasks[0].get('chunk_maps_meta')
    if chunk_maps_meta is not None:
        chunk_maps = []
        for meta in chunk_maps_meta:
            if meta is None:
                chunk_maps.append(None)
            else:
                name, shape, dtype = meta
                shm = SharedMemory(name=name)
                chunk_maps.append(np.ndarray(shape, dtype=dtype, buffer=shm.buf))
                shm_handles.append(shm)
        shm_arrays['chunk_maps'] = chunk_maps

    # det_aux: list of detector-grid float arrays (e.g. [BC_map, BW_map] for
    # spectral-fit mode). Reconstructed from SHM mirroring chunk_maps_meta.
    det_aux_metas = sub_tasks[0].get('det_aux_metas')
    if det_aux_metas is not None:
        det_aux = []
        for meta in det_aux_metas:
            name, shape, dtype = meta
            shm = SharedMemory(name=name)
            det_aux.append(np.ndarray(shape, dtype=dtype, buffer=shm.buf))
            shm_handles.append(shm)
        shm_arrays['det_aux'] = det_aux

    if 'gvw_shm_name' in sub_tasks[0]:
        shm_gvw = SharedMemory(name=sub_tasks[0]['gvw_shm_name'])
        shm_arrays['grid_valid_weight'] = np.ndarray(sub_tasks[0]['gvw_shape'], dtype=sub_tasks[0]['gvw_dtype'], buffer=shm_gvw.buf)
        shm_handles.append(shm_gvw)

    adj_metas = sub_tasks[0].get('adj_metas')
    if adj_metas is not None:
        adj_info_list = []
        for per_map_meta in adj_metas:
            if per_map_meta is None:
                adj_info_list.append(None)
                continue
            adj_parts = []
            for name, shape, dtype in per_map_meta:
                shm = SharedMemory(name=name)
                adj_parts.append(np.ndarray(shape, dtype=dtype, buffer=shm.buf))
                shm_handles.append(shm)
            adj_info_list.append(tuple(adj_parts))
        shm_arrays['adj_info_list'] = adj_info_list

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
               chunk_maps=None, grid_valid_weight=None, apply_mask=True, apply_weight=False,
               valid_threshold=0.99,
               outlier_thresh=3, max_workers=20, ignore_list=[], oversample_factor=1, batch_size=10, offset_regularization=False,
               reg_weights=None, adj_infos=None, poly_constraints_list=None,
               mean_offsets_list=None, det_groups_list=None, det_templates=None,
               use_per_frame_scalar=False,
               postprocess_func=None, preprocess_func=None,
               weighted_damping=False, damp_weight=0.1, damp_offset=0.0,
               det_aux=None,
               spectral_fit=False, line_center=None, line_sigma=None,
               damp_weight_line=None,
               sky_model=None,
               top2_compaction_enabled=True):
    """Prepares the LSQR matrix A and vector b for all subframes in parallel.

    The model is ``d_i = s(p_i) + Σ_m o^(m)[g_m(k), c_m(i)] + ε``: K independent
    additive offset blocks, each with its own chunk map, frame-to-group mapping,
    template, regularization, and mean-offset constraint. The K=1 case mirrors
    the original single-chunk-map solver bit-for-bit.

    Parameters
    ----------
    file_list : list
        List of paths to the reprojected HDF5 files.
    ref_shape : tuple, list
        Shape of the reference frame (height, width).
    chunk_maps : list of np.ndarray or None
        K chunk maps (each must be 0-indexed and contiguous). All maps must
        share the same shape. ``None`` or an empty list disables the offset
        block entirely.
    grid_valid_weight : np.ndarray, optional
        Weight indicating valid pixels for each grid pixel.
    reg_weights : list of float or None
        Per-map adjacency regularization weights (length K). Defaults to all 0.
    adj_infos : list of tuple or None
        Per-map precomputed adjacency information (length K). Each entry is a
        ``(chunk_i, chunk_j)`` tuple or ``None``.
    poly_constraints_list : list or None
        Per-map polynomial-order constraint groups (length K). Each entry is
        ``None`` (no poly constraints on this map) or a list of dicts, each
        ``{'chains': (num_chains, L) int ndarray, 'stencil': (L,) float
        ndarray, 'weight': float}``. The constraint is
        ``λ · Σ_ℓ stencil[ℓ] · o[chains[r, ℓ]] = 0`` per chain r per frame.
        Generalizes the constant-prior ``adj_infos`` (stencil=[1,-1]) to
        arbitrary finite-difference operators. Skipped in template mode.
    mean_offsets_list : list or None
        Per-map mean-offset constraint targets (length K). Each entry is a
        length-num_frames array or ``None`` to disable for that map.
    det_groups_list : list or None
        Per-map frame→group labels (length K). Each entry is ``None`` (one
        group per frame) or an array of length num_frames. A single per-frame
        scalar bias column is added if any map uses det_groups (or if
        ``use_per_frame_scalar`` is True).
    use_per_frame_scalar : bool, optional
        When True, add a per-frame scalar column (one per frame) even when
        no map uses ``det_groups`` — useful for absorbing per-frame DC into
        an explicit column so the per-frame chunk offsets only carry
        within-frame structure. Pair with ``mean_offsets_list=[zeros]`` and
        a zero-chunk x0 init to push DC into the scalar.
    det_templates : list or None
        Per-map fixed spatial templates (length K). When set for map m, that
        map solves only for a per-frame amplitude α[k] (block size = num_frames).
    """
    assert isinstance(file_list, (list, np.ndarray)) and file_list, "file_list must be a non-empty list"
    assert isinstance(ref_shape, (list, np.ndarray, tuple)) and len(ref_shape) == 2, "ref_shape must be a list of length 2"
    assert grid_valid_weight is None or isinstance(grid_valid_weight, np.ndarray), "grid_valid_weight must be a numpy array"
    assert isinstance(apply_mask, bool), "apply_mask must be a boolean"
    assert isinstance(apply_weight, bool), "apply_weight must be a boolean"
    assert isinstance(outlier_thresh, (int, float, type(None))) and (outlier_thresh is None or outlier_thresh > 0), "outlier_thresh must be a positive number or None"
    assert isinstance(max_workers, int) and max_workers > 0, "max_workers must be a positive integer"
    assert isinstance(ignore_list, (list, np.ndarray)), "ignore_list must be a list or array of data quality flags to ignore"
    assert isinstance(batch_size, int) and batch_size > 0, "batch_size must be a positive integer"

    # Normalize chunk_maps and per-map arguments to length-K lists.
    if chunk_maps is None:
        chunk_maps = []
    assert isinstance(chunk_maps, list), "chunk_maps must be a list"
    for cm in chunk_maps:
        assert isinstance(cm, np.ndarray), "every chunk_maps entry must be a numpy array"
    K = len(chunk_maps)

    def _default(x, fill):
        return [fill] * K if x is None else x

    reg_weights = _default(reg_weights, 0.0)
    adj_infos = _default(adj_infos, None)
    poly_constraints_list = _default(poly_constraints_list, None)
    mean_offsets_list = _default(mean_offsets_list, None)
    det_groups_list = _default(det_groups_list, None)
    det_templates = _default(det_templates, None)

    # Normalize and validate poly-constraint groups: each entry is None or a
    # non-empty list of dicts; each dict has matching chains.shape[1] == len(stencil).
    # Cast chains to int64 and stencil to float64 once here so workers don't
    # re-cast per call.
    normalized_poly = []
    for m, groups in enumerate(poly_constraints_list):
        if groups is None:
            normalized_poly.append(None)
            continue
        norm_groups = []
        for g_idx, grp in enumerate(groups):
            chains = np.asarray(grp['chains'], dtype=np.int64)
            stencil = np.asarray(grp['stencil'], dtype=np.float64)
            weight = float(grp['weight'])
            assert chains.ndim == 2, \
                f"poly_constraints_list[{m}][{g_idx}]['chains'] must be 2-D"
            assert stencil.ndim == 1, \
                f"poly_constraints_list[{m}][{g_idx}]['stencil'] must be 1-D"
            assert chains.shape[1] == stencil.shape[0], \
                (f"poly_constraints_list[{m}][{g_idx}]: chains.shape[1]="
                 f"{chains.shape[1]} != len(stencil)={stencil.shape[0]}")
            norm_groups.append({'chains': chains, 'stencil': stencil, 'weight': weight})
        normalized_poly.append(norm_groups if norm_groups else None)
    poly_constraints_list = normalized_poly

    # Adjacency tuples with all-empty arrays (e.g. NumCol=1 from
    # compute_column_adjacency) produce zero adjacency constraints anyway —
    # demote to None so the SHM packing below doesn't try to create a
    # zero-byte segment, which raises ValueError.
    adj_infos = [
        None if (adj is not None and all(np.asarray(a).size == 0 for a in adj)) else adj
        for adj in adj_infos
    ]
    for name, arr in (('reg_weights', reg_weights), ('adj_infos', adj_infos),
                      ('poly_constraints_list', poly_constraints_list),
                      ('mean_offsets_list', mean_offsets_list),
                      ('det_groups_list', det_groups_list),
                      ('det_templates', det_templates)):
        assert len(arr) == K, f"{name} must have length {K} (got {len(arr)})"

    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    num_frames = len(file_list)

    # --- Spectral-fit mode: 2-block sky (continuum + line amplitude per pixel) ---
    # When spectral_fit is True, the sky block grows from num_sky to 2*num_sky:
    # x[:num_sky] is the continuum sky map, x[num_sky:2*num_sky] is the line
    # amplitude map (PAH 3.29 μm by default). The data row for one observation
    # of ref pixel P at LVF wavelength λ_i gains a second sky nnz:
    #
    #   data_i = w_i * (sky_cont[P] + G(λ_i) * sky_line[P]) + offsets + scalar
    #
    # where G(λ) is the Gaussian line profile (peak = 1 at line_center,
    # sigma = line_sigma). λ_i is sampled per (frame, sub-pixel) via the
    # det_aux plumbing: BC_map must be passed as det_aux[0]. Optionally
    # det_aux[1] = BW_map gives per-pixel σ (mixed with PAH intrinsic).
    # --- Sky model resolution ---
    # sky_model= is the forward-looking API; the legacy spectral_fit flag (+
    # line_center / line_sigma) is a deprecated shim that builds the equivalent
    # SkyModel. The model's components drive the per-pixel sky row emission in the
    # worker (continuum -> J=1 identity fast path; +line -> interleave with the
    # profile coefficient). For sky_model=None this reproduces the old
    # num_sky_blocks {1,2} behavior byte-for-byte.
    if sky_model is None:
        if spectral_fit:
            sky_model = SkyModel.continuum_plus_pah_gaussian(line_center, line_sigma)
        else:
            sky_model = SkyModel.continuum_only()
    num_sky_blocks = sky_model.n_blocks
    if num_sky_blocks > 1:
        # A spectral SkyModel (>=1 non-continuum block) needs the wavelength aux
        # map(s) and gets decoupled line-block damping by default (the line
        # columns have smaller average coefficients than continuum, so ~3x more
        # Tikhonov shrinkage at the same data S/N).
        if damp_weight_line is None:
            damp_weight_line = 3.0 * damp_weight
        if det_aux is None or len(det_aux) < 1:
            raise ValueError(
                "A spectral SkyModel (>1 sky block) requires det_aux=[BC_map] "
                "(or [BC_map, BW_map] for per-pixel σ). Pass BC_map from "
                "selfcal.SPHERExUtility.load_calibration(band=detector).")
        print(f"Spectral mode ON: {num_sky_blocks} sky blocks {sky_model.names}, "
              f"{num_sky_blocks * num_sky} sky cols, damp_weight_line={damp_weight_line}.")
    # Positional det_aux -> named aux dict (SPHEREx convention: [BC, BW]).
    aux_keys = ['BC', 'BW'][:len(det_aux)] if det_aux is not None else []

    # --- Column layout (single source of truth: selfcal.layout.SystemLayout) ---
    # SystemLayout computes the per-map group mapping, template normalization,
    # col_bases, the per-frame scalar block, and the total column count. The
    # Calibrator builds the same layout from the same inputs (see
    # PipelineWrapper.Calibrator.setup_lsqr) so the parent-side and parse-side
    # column arithmetic can never drift.
    any_det_groups = any(g is not None for g in det_groups_list)
    layout = SystemLayout.build(
        ref_shape, chunk_maps, num_sky_blocks=num_sky_blocks, num_frames=num_frames,
        det_groups_list=det_groups_list, det_templates=det_templates,
        use_per_frame_scalar=use_per_frame_scalar)
    frame_to_group_list = layout.frame_to_group_list
    num_offset_groups_list = layout.num_offset_groups_list
    num_chunks_list = layout.num_chunks_list
    det_template_arr_list = layout.det_template_arr_list
    num_scalar_cols = layout.num_scalar_cols
    col_bases = layout.col_bases
    scalar_col_start = layout.scalar_col_start
    total_cols = layout.total_cols

    if any_det_groups or use_per_frame_scalar:
        print(f"Locking detector offsets: {num_frames} frames -> "
              f"groups {num_offset_groups_list} + {num_frames} frame scalars")
    if any(t is not None for t in det_template_arr_list):
        tmpl_indices = [m for m, t in enumerate(det_template_arr_list) if t is not None]
        print(f"Template mode for maps {tmpl_indices}: {num_frames} alpha unknowns each")

    common_params = {
        'chunk_maps': chunk_maps,
        'grid_valid_weight': grid_valid_weight,
        'apply_mask': apply_mask,
        'apply_weight': apply_weight,
        'ignore_list': ignore_list,
        'oversample_factor': oversample_factor,
        'valid_threshold': valid_threshold,
        'outlier_thresh': outlier_thresh,
        'num_chunks_list': num_chunks_list,
        'num_frames': num_frames,
        'ref_shape': ref_shape,
        'offset_regularization': offset_regularization,
        'reg_weight_list': reg_weights,
        'adj_info_list': adj_infos,
        'poly_constraint_list': poly_constraints_list,
        'postprocess_func': postprocess_func,
        'preprocess_func': preprocess_func,
        'frame_to_group_list': frame_to_group_list,
        'col_bases': col_bases,
        'scalar_col_start': scalar_col_start,
        'num_scalar_cols': num_scalar_cols,
        'det_template_list': det_template_arr_list,
        'num_sky_blocks': num_sky_blocks,
        'sky_components': sky_model.components,
        'aux_keys': aux_keys,
        'line_center': line_center,
        'line_sigma': line_sigma,
    }

    # Move large arrays to shared memory so forked processes can access them
    # without pickling. Each process reconstructs numpy views in the worker.
    shm_objects = []

    if K > 0:
        chunk_maps_meta = []
        for cm in chunk_maps:
            shm_cm = SharedMemory(create=True, size=cm.nbytes)
            np.ndarray(cm.shape, dtype=cm.dtype, buffer=shm_cm.buf)[:] = cm
            shm_objects.append(shm_cm)
            chunk_maps_meta.append((shm_cm.name, cm.shape, cm.dtype))
        common_params['chunk_maps_meta'] = chunk_maps_meta
        common_params['chunk_maps'] = None  # populated by worker from SHM

    if det_aux is not None:
        # Pack each detector-grid array into SHM. Workers reconstruct via
        # det_aux_metas. Mirrors the chunk_maps_meta pattern above. Used by
        # spectral-fit mode to expose BC_map (and optionally BW_map) at row
        # assembly time so the line-amplitude column coefficient can be
        # evaluated per sub-pixel.
        det_aux_metas = []
        for arr in det_aux:
            arr_f32 = np.ascontiguousarray(arr, dtype=np.float32)
            shm_da = SharedMemory(create=True, size=arr_f32.nbytes)
            np.ndarray(arr_f32.shape, dtype=arr_f32.dtype, buffer=shm_da.buf)[:] = arr_f32
            shm_objects.append(shm_da)
            det_aux_metas.append((shm_da.name, arr_f32.shape, arr_f32.dtype))
        common_params['det_aux_metas'] = det_aux_metas
        common_params['det_aux'] = None  # populated by worker from SHM

    if grid_valid_weight is not None:
        shm_gvw = SharedMemory(create=True, size=grid_valid_weight.nbytes)
        np.ndarray(grid_valid_weight.shape, dtype=grid_valid_weight.dtype, buffer=shm_gvw.buf)[:] = grid_valid_weight
        shm_objects.append(shm_gvw)
        common_params['gvw_shm_name'] = shm_gvw.name
        common_params['gvw_shape'] = grid_valid_weight.shape
        common_params['gvw_dtype'] = grid_valid_weight.dtype
        common_params['grid_valid_weight'] = None

    if any(a is not None for a in adj_infos):
        adj_metas = []
        for adj in adj_infos:
            if adj is None:
                adj_metas.append(None)
                continue
            per_map = []
            for arr in adj:
                shm = SharedMemory(create=True, size=arr.nbytes)
                np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
                shm_objects.append(shm)
                per_map.append((shm.name, arr.shape, arr.dtype))
            adj_metas.append(per_map)
        common_params['adj_metas'] = adj_metas
        common_params['adj_info_list'] = None  # populated by worker from SHM

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

    # Per-batch streaming accumulators for pixel_counts and pixel_fisher.
    # Allocated lazily on the first batch result. This replaces the
    # post-loop full-nnz bincount (which materialized a ~50 GB float64
    # squared-data temp at no-srcmask region-10k scale).
    pixel_counts = np.zeros(total_cols, dtype=np.int64)
    pixel_fisher = np.zeros(total_cols, dtype=np.float64)

    def _read_shm(info):
        """Read array from shared memory and clean up the segment."""
        name, shape, dtype = info
        shm = SharedMemory(name=name)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
        shm.close()
        shm.unlink()
        return arr

    # ----------------------------------------------------------------
    # Phase 1: collate worker results.
    #
    # We retain each batch's (rows, cols, data, b) so Phase 4 can scatter
    # them directly into the final CSR buffers in batch-id order. Per-batch
    # we also accumulate row_nnz_per_batch[batch_id] = bincount(local_rows)
    # so the CSR indptr can be built without re-touching every batch's row
    # array. pixel_counts / pixel_fisher are accumulated as before (Top 3).
    # ----------------------------------------------------------------
    batch_results = [None] * len(batched_tasks)
    row_nnz_per_batch = [None] * len(batched_tasks)
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_prep_lsqr_batch_worker, batch): i
                       for i, batch in enumerate(batched_tasks)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Building A, b matrix"):
                batch_id = futures[future]
                result = future.result()
                if result is None:
                    continue
                shm_infos = result['shm']
                batch_results[batch_id] = {
                    'rows': _read_shm(shm_infos[0]),         # local int32 row ids
                    'cols': _read_shm(shm_infos[1]),
                    'data': _read_shm(shm_infos[2]),
                    'b':    _read_shm(shm_infos[3]),
                    'num_rows': result['num_rows'],
                }
                _b_cols = batch_results[batch_id]['cols']
                _b_data = batch_results[batch_id]['data']
                _b_rows = batch_results[batch_id]['rows']
                # Per-batch streaming accumulation. Avoids holding a
                # full-nnz float64 squared-data temp later.
                pixel_counts += np.bincount(_b_cols, minlength=total_cols)
                pixel_fisher += np.bincount(
                    _b_cols,
                    weights=_b_data.astype(np.float64) ** 2,
                    minlength=total_cols,
                )
                # Per-batch row nnz over LOCAL row ids (0..num_rows-1). We
                # add the global row offset in Phase 3 (cumulative across
                # batches). Keeping this batch-local is what lets Phase 4
                # stream-scatter without revisiting all batches twice.
                row_nnz_per_batch[batch_id] = np.bincount(
                    _b_rows, minlength=result['num_rows']
                ).astype(np.int32, copy=False)
    finally:
        for shm in shm_objects:
            shm.close()
            shm.unlink()

    # ----------------------------------------------------------------
    # Phase 2: compute per-batch row offsets, total_rows_data, and the
    # data-row half of row_nnz.
    # ----------------------------------------------------------------
    batch_row_starts = [0] * len(batched_tasks)
    total_rows_data = 0
    any_kept = False
    for batch_id in range(len(batched_tasks)):
        batch_row_starts[batch_id] = total_rows_data
        r = batch_results[batch_id]
        if r is None:
            continue
        any_kept = True
        total_rows_data += r['num_rows']

    if not any_kept:
        print("No valid data found in any subframe.")
        return None, None

    # Ensure Python int to avoid numpy int32 overflow on big runs.
    total_rows_data = int(total_rows_data)

    # ----------------------------------------------------------------
    # Phase 2b: build constraint blocks (rows-local-to-block, cols, data,
    # b). We assemble them up front so we can pre-count their per-row nnz,
    # finalize total_rows, and allocate the CSR buffers once.
    # ----------------------------------------------------------------
    num_sky_eff = num_sky_blocks * num_sky
    sky_pixel_counts = pixel_counts[:num_sky]                       # continuum coverage
    if num_sky_blocks == 2:
        line_pixel_counts = pixel_counts[num_sky:2*num_sky]         # line amplitude coverage
    else:
        line_pixel_counts = None
    offset_pixel_counts = pixel_counts[num_sky_eff:]

    # Global constraint blocks (see selfcal.constraint_builders). Emission order
    # is load-bearing for the CSR scatter: mean-offset anchors (per map) ->
    # sky damping (continuum, then line blocks) -> offset damping.
    constraint_blocks = []

    # --- Per-frame mean-offset constraints (one block per chunk map) ---
    #TODO: Pass weight from higher level instead of hardcoding here
    constraint_weight = 10.0
    for m in range(K):
        mean_off = mean_offsets_list[m]
        if mean_off is None:
            continue
        if det_template_arr_list[m] is not None:
            print(f"Skipping mean-offset constraint for map {m}: template mode does not have per-chunk offsets")
            continue
        print(f"Applying target mean offset constraints for map {m} ({num_frames} frames)...")
        constraint_blocks.append(mean_offset_block(
            m, mean_off, num_frames, num_chunks_list[m], frame_to_group_list[m],
            col_bases, weight=constraint_weight).as_dict())

    # --- Coverage-weighted sky damping (continuum, then each line block) ---
    if weighted_damping and damp_weight > 0:
        print("Applying Coverage-Weighted Damping (continuum)...")
        blk = sky_damping_block(0, damp_weight, sky_pixel_counts, num_sky)
        if blk is not None:
            constraint_blocks.append(blk.as_dict())

        # --- LINE-AMPLITUDE DAMPING (spectral, block 1) ---
        if num_sky_blocks == 2 and damp_weight_line is not None and damp_weight_line > 0:
            print(f"Applying Coverage-Weighted Damping (line, damp={damp_weight_line})...")
            blk = sky_damping_block(1, damp_weight_line, line_pixel_counts, num_sky)
            if blk is not None:
                constraint_blocks.append(blk.as_dict())

    # --- Coverage-weighted offset damping ---
    if damp_offset > 0:
        print(f"Applying Coverage-Weighted Offset Damping (damp_offset={damp_offset})...")
        n_offset_cols = scalar_col_start - num_sky_eff
        blk = offset_damping_block(damp_offset, offset_pixel_counts[:n_offset_cols], num_sky_eff)
        if blk is not None:
            constraint_blocks.append(blk.as_dict())


    # ----------------------------------------------------------------
    # Phase 2c: finalize total_rows + build row_nnz over the entire row
    # space (data rows first, then each constraint block).
    # ----------------------------------------------------------------
    constraint_row_total = sum(blk['num_rows'] for blk in constraint_blocks)
    total_rows = total_rows_data + constraint_row_total

    row_nnz = np.empty(total_rows, dtype=np.int32)
    # Data-row half: concat per-batch row_nnz in batch-id order. None entries
    # contribute zero rows so they simply don't appear.
    cursor = 0
    for batch_id in range(len(batched_tasks)):
        rn = row_nnz_per_batch[batch_id]
        if rn is None:
            continue
        row_nnz[cursor:cursor + rn.size] = rn
        cursor += rn.size
    assert cursor == total_rows_data, (cursor, total_rows_data)
    # Constraint-row half: each block is uniform nnz_per_row.
    for blk in constraint_blocks:
        row_nnz[cursor:cursor + blk['num_rows']] = blk['nnz_per_row']
        cursor += blk['num_rows']
    assert cursor == total_rows
    # Free per-batch row_nnz; no longer needed.
    row_nnz_per_batch = None

    # ----------------------------------------------------------------
    # Phase 2d: build full_b (vector of right-hand-sides). Concatenated in
    # the same order as the row scatter: per-batch b in batch-id order,
    # then each constraint block's b.
    # ----------------------------------------------------------------
    b_pieces = []
    for batch_id in range(len(batched_tasks)):
        r = batch_results[batch_id]
        if r is None:
            continue
        b_pieces.append(r['b'])
    for blk in constraint_blocks:
        b_pieces.append(blk['b'])
    full_b = np.concatenate(b_pieces) if b_pieces else np.zeros(0, dtype=np.float64)
    # Drop per-batch b refs so the streaming scatter loop holds the
    # smallest possible footprint.
    for batch_id in range(len(batched_tasks)):
        r = batch_results[batch_id]
        if r is not None:
            r['b'] = None
    del b_pieces
    for blk in constraint_blocks:
        blk['b'] = None

    # ----------------------------------------------------------------
    # Phase 2e: build CSR indptr (int64) + total_nnz.
    # ----------------------------------------------------------------
    indptr = np.zeros(total_rows + 1, dtype=np.int64)
    np.cumsum(row_nnz, out=indptr[1:])
    total_nnz = int(indptr[-1])
    # row_nnz no longer needed; we will use indptr+write_cursor for scatter.
    del row_nnz

    # ----------------------------------------------------------------
    # Top 2: gated early column compaction. If no template-mode map is in
    # use, every "active" column (pixel_counts > 0) gets compacted now so
    # apply_lsqr can skip its full-nnz col_map gather. Template-mode runs
    # keep the legacy uncompacted layout (apply_lsqr handles compaction).
    # ----------------------------------------------------------------
    top2_active = not any(t is not None for t in det_template_arr_list)
    # Bisect/debug knob: caller can force the gated path off to compare against
    # the legacy uncompacted-CSR layout (used when bisecting which optimization
    # phase introduced a regression). Default True keeps prod behavior.
    if not top2_compaction_enabled:
        top2_active = False
    if top2_active:
        # Some non-template runs still want the legacy path — e.g. when the
        # caller passes mean_offsets_list (per-frame chunk constraint rows
        # write into specific column indices that must not be dropped).
        # mean_offsets / damping rows already register coverage via the
        # data rows, so columns they touch will have pixel_counts > 0.
        # Constraint rows pointing at otherwise-uncovered cols would still
        # need those slots; guard with a coverage check.
        active_mask = pixel_counts > 0
        # Verify every constraint col is "active" — otherwise gating must
        # bail out and let apply_lsqr handle compaction.
        all_constraint_cols_active = True
        for blk in constraint_blocks:
            cols_b = blk['cols']
            if cols_b.size and not active_mask[cols_b].all():
                all_constraint_cols_active = False
                break
        if not all_constraint_cols_active:
            top2_active = False

    if top2_active:
        col_map = np.cumsum(active_mask, dtype=np.int64) - 1  # int64; mapped col fits in int32
        n_active = int(active_mask.sum())
        print(f"Top 2: compacting columns inline ({n_active}/{total_cols} active).")
    else:
        active_mask = None
        col_map = None
        n_active = total_cols

    # ----------------------------------------------------------------
    # Phase 3: allocate CSR buffers.
    # ----------------------------------------------------------------
    csr_data = np.empty(total_nnz, dtype=np.float32)
    csr_indices = np.empty(total_nnz, dtype=np.int32)
    write_cursor = np.zeros(total_rows, dtype=np.int32)

    # ----------------------------------------------------------------
    # Phase 4a: streaming scatter of data rows (one batch at a time;
    # release each batch's arrays immediately after scatter so the peak
    # footprint stays small).
    #
    # Within-batch row collisions are common (e.g. spectral_fit mode places
    # 2 sky nnz + 1 offset + 1 scalar in the SAME row). A naive
    #   slots = indptr[rows_b] + write_cursor[rows_b]
    # would have all duplicates read the same write_cursor value and
    # overwrite the same slot. We instead stable-sort by row and use a
    # within-row cumcount so every entry lands in its own slot. Stable
    # sort preserves the original col-within-row order across the batch,
    # which matches what coo_matrix(...).tocsr() does for stable sort_by_row.
    # ----------------------------------------------------------------
    for batch_id in range(len(batched_tasks)):
        batch = batch_results[batch_id]
        if batch is None:
            continue
        row_offset = batch_row_starts[batch_id]
        rows_b = batch['rows'].astype(np.int64, copy=False) + row_offset
        cols_b = batch['cols']
        data_b = batch['data']
        if top2_active:
            # int32 compact col indices (safe since n_active < 2^31).
            cols_b = col_map[cols_b].astype(np.int32, copy=False)
        else:
            cols_b = cols_b.astype(np.int32, copy=False)

        n_b = rows_b.shape[0]
        if n_b > 0:
            # Stable sort by row so duplicates are contiguous; preserves
            # within-row col order from the worker output.
            order = np.argsort(rows_b, kind="stable")
            rows_s = rows_b[order]
            cols_s = cols_b[order]
            data_s = data_b[order]

            # Within-row cumcount in sorted order.
            is_new_group = np.empty(n_b, dtype=bool)
            is_new_group[0] = True
            is_new_group[1:] = rows_s[1:] != rows_s[:-1]
            group_starts = np.flatnonzero(is_new_group)
            group_idx = np.cumsum(is_new_group, dtype=np.int64) - 1
            within_row = np.arange(n_b, dtype=np.int64) - group_starts[group_idx]

            slots = indptr[rows_s] + write_cursor[rows_s] + within_row
            csr_data[slots] = data_s
            csr_indices[slots] = cols_s

            # Update write_cursor: add the per-row count contributed by
            # this batch.
            unique_rows = rows_s[group_starts]
            counts = np.diff(np.append(group_starts, n_b)).astype(np.int32, copy=False)
            write_cursor[unique_rows] += counts
        # Free the batch refs.
        batch_results[batch_id] = None
        del batch, rows_b, cols_b, data_b
    batch_results = None

    # ----------------------------------------------------------------
    # Phase 4b: scatter constraint blocks at their reserved row ranges.
    #
    # Same row-collision concern as Phase 4a: e.g. the per-frame
    # mean-offset block has nc_m entries per row. Use the same
    # stable-sort + within-row cumcount pattern. write_cursor for
    # constraint rows starts at 0 (no prior batches touched them) but
    # we still write through write_cursor for uniformity.
    # ----------------------------------------------------------------
    cb_cursor = total_rows_data
    for blk in constraint_blocks:
        nrows = blk['num_rows']
        nnz_per_row = blk['nnz_per_row']
        rows_b = blk['rows_local'] + cb_cursor
        cols_b = blk['cols']
        data_b = blk['data']
        if top2_active:
            cols_b = col_map[cols_b].astype(np.int32, copy=False)
        else:
            cols_b = cols_b.astype(np.int32, copy=False)

        n_b = rows_b.shape[0]
        if n_b > 0:
            # nnz_per_row can be a scalar (uniform) or a numpy array (varying per-row).
            is_uniform_one = (np.ndim(nnz_per_row) == 0 and int(nnz_per_row) == 1)
            if is_uniform_one:
                # Fast path: every constraint row contributes exactly one
                # entry, so no duplicates exist and no sort is required.
                slots = indptr[rows_b] + write_cursor[rows_b]
                csr_data[slots] = data_b
                csr_indices[slots] = cols_b
                write_cursor[rows_b] += 1
            else:
                order = np.argsort(rows_b, kind="stable")
                rows_s = rows_b[order]
                cols_s = cols_b[order]
                data_s = data_b[order]

                is_new_group = np.empty(n_b, dtype=bool)
                is_new_group[0] = True
                is_new_group[1:] = rows_s[1:] != rows_s[:-1]
                group_starts = np.flatnonzero(is_new_group)
                group_idx = np.cumsum(is_new_group, dtype=np.int64) - 1
                within_row = np.arange(n_b, dtype=np.int64) - group_starts[group_idx]

                slots = indptr[rows_s] + write_cursor[rows_s] + within_row
                csr_data[slots] = data_s
                csr_indices[slots] = cols_s

                unique_rows = rows_s[group_starts]
                counts = np.diff(np.append(group_starts, n_b)).astype(np.int32, copy=False)
                write_cursor[unique_rows] += counts
        cb_cursor += nrows
    del write_cursor

    # ----------------------------------------------------------------
    # Phase 6: build CSR. Indices are row-grouped but NOT col-sorted within
    # row (sufficient for LSQR/LSMR matvec/rmatvec — convergence depends
    # only on A as a linear map). We mark has_sorted_indices=False so
    # scipy callers that depend on it (e.g. .T → CSC view) can either
    # tolerate or trigger sort themselves.
    # ----------------------------------------------------------------
    full_A = csr_matrix(
        (csr_data, csr_indices, indptr),
        shape=(total_rows, n_active),
        copy=False,
    )
    full_A.has_sorted_indices = False
    # Sort indices within each row in place. This is per-row qsort with no
    # allocation peak and gives downstream code (CSC views, indices-binary
    # search, etc.) the canonical layout. We also call sum_duplicates() as a
    # defensive no-op (the bucket-sort build guarantees no duplicate (row,
    # col) entries) so that downstream consumers can rely on canonical CSR.
    full_A.sort_indices()
    full_A.sum_duplicates()

    if top2_active:
        return full_A, full_b, pixel_counts, pixel_fisher, active_mask
    return full_A, full_b, pixel_counts, pixel_fisher


def parse_pixel_counts(pixel_counts, ref_shape, num_offset_groups_list, chunk_maps,
                       num_sky_blocks=1):
    """Slice ``pixel_counts`` into per-block coverage arrays.

    When ``num_sky_blocks=2`` (spectral_fit mode), the returned
    ``skymap_coverage`` is the *continuum* block's coverage and an additional
    ``line_coverage`` ndarray is returned as the second element. The slicing
    boundary between sky and offset blocks shifts to ``num_sky_blocks*num_sky``.

    Returns
    -------
    skymap_coverage : np.ndarray
        Continuum sky-block coverage, shape ref_shape.
    line_coverage : np.ndarray or None
        Line-amplitude block coverage when num_sky_blocks==2, else None.
        Same value as skymap_coverage in expectation (one row → one count per
        block), but kept separate for symmetry with the parse_x API.
    offset_coverages : list of np.ndarray
        One ``(num_offset_groups[m], num_chunks[m])`` array per chunk map.
    offset_valid_fracs : list of np.ndarray
        Each block's coverage normalized by the chunk pixel-count.
    """
    num_sky = ref_shape[0] * ref_shape[1]
    skymap_coverage = pixel_counts[:num_sky].reshape(ref_shape)
    if num_sky_blocks == 2:
        line_coverage = pixel_counts[num_sky:2*num_sky].reshape(ref_shape)
        cursor = 2 * num_sky
    else:
        line_coverage = None
        cursor = num_sky

    offset_coverages = []
    offset_valid_fracs = []
    for ng, cm in zip(num_offset_groups_list, chunk_maps):
        num_chunks = int(np.max(cm)) + 1
        block = ng * num_chunks
        offset_coverage = pixel_counts[cursor:cursor + block].reshape(ng, num_chunks)
        chunk_sizes = np.bincount(cm[cm >= 0].ravel(), minlength=num_chunks)
        offset_valid_frac = (offset_coverage / np.maximum(chunk_sizes, 1))
        offset_coverages.append(offset_coverage)
        offset_valid_fracs.append(offset_valid_frac)
        cursor += block
    if num_sky_blocks == 2:
        return skymap_coverage, line_coverage, offset_coverages, offset_valid_fracs
    return skymap_coverage, offset_coverages, offset_valid_fracs


def parse_pixel_fisher(pixel_fisher, ref_shape, num_sky_blocks=1):
    """Slice pixel_fisher into per-block Fisher arrays.

    pixel_fisher is the sum-of-squared sparse-matrix coefficients per
    column, computed over data rows only. For the line block the
    coefficient is w_i * G(λ_i), so this is the correct per-pixel line
    constraint metric. For the continuum block it equals sum(w_i^2).

    Returns
    -------
    skymap_fisher : np.ndarray
        Continuum sky-block Fisher info, shape ref_shape.
    line_fisher : np.ndarray or None
        Line-amplitude block Fisher info when num_sky_blocks==2, else None.
    """
    num_sky = ref_shape[0] * ref_shape[1]
    skymap_fisher = pixel_fisher[:num_sky].reshape(ref_shape)
    if num_sky_blocks == 2:
        line_fisher = pixel_fisher[num_sky:2*num_sky].reshape(ref_shape)
    else:
        line_fisher = None
    return skymap_fisher, line_fisher


def apply_line_fisher_mask(sky_line, line_fisher, threshold):
    """Apply Phase 6 Fisher mask at READ time: zero sky_line where Fisher < threshold.

    Cals saved by Calibrator.save_calibration contain RAW sky_line (no
    destructive mask) plus the per-pixel line Fisher info. The
    line_fisher_threshold attribute (if present) is the recommended
    threshold, but the analyst is free to pick their own.

    Parameters
    ----------
    sky_line : np.ndarray
        Raw skymap_line as saved (ref grid).
    line_fisher : np.ndarray
        Raw skymap_line_fisher as saved (same shape).
    threshold : float
        Fisher threshold; pixels with line_fisher < threshold are zeroed.

    Returns
    -------
    sky_line_masked : np.ndarray
        Copy of sky_line with low-Fisher pixels set to 0.
    mask : np.ndarray of bool
        Boolean mask (True where masked-out).
    """
    mask = line_fisher < float(threshold)
    out = sky_line.copy()
    out[mask] = 0.0
    return out, mask


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
    """Build a LinearOperator with thread-parallel matvec, scipy-native rmatvec.

    matvec: per-thread row-block partition of A_csr (GIL released during scipy CSR SpMV).
    rmatvec: A_csr.T as a zero-copy CSC view; scipy CSC SpMV handles A.T @ y directly.

    The previous version materialized AT_csr = A_csr.T.tocsr() as an explicit copy, which
    at no-srcmask region-10k scale was ~88 GB. The CSC view shares A_csr's storage and
    costs O(1) — scipy's CSC @ vec is fast and releases the GIL.
    """
    m, n = A_csr.shape

    print(f"Building parallel SpMV operator ({n_threads} threads)...")
    AT_view = A_csr.T  # zero-copy CSC view sharing storage with A_csr

    A_blocks, A_bounds = _partition_csr(A_csr, n_threads)

    executor = ThreadPoolExecutor(max_workers=n_threads)
    dtype = A_csr.dtype

    def _matvec(x):
        out = np.empty(m, dtype=dtype)
        def _work(i):
            out[A_bounds[i]:A_bounds[i+1]] = A_blocks[i] @ x
        list(executor.map(_work, range(n_threads)))
        return out

    def _rmatvec(y):
        # Single scipy CSC SpMV — no per-thread slicing needed; the kernel is already
        # vectorized and releases the GIL.
        return AT_view @ y

    op = LinearOperator((m, n), matvec=_matvec, rmatvec=_rmatvec, dtype=A_csr.dtype)
    op._executor = executor
    op._AT_view = AT_view  # prevent GC
    return op

def apply_lsqr(A, b, ref_shape, x0=None,
                atol=1e-05, btol=1e-05, damp=1e-2, iter_lim=100, precondition=True,
                solver='lsmr', use_float32=False, n_threads=32,
                active_mask=None, num_cols_full=None):
    """Applies LSQR or LSMR to solve for the sky and detector offsets.

    Parameters
    ----------
    A : coo_matrix or csr_matrix
        Sparse system matrix. When ``csr_matrix`` is passed together with an
        ``active_mask``, setup_lsqr is assumed to have already compacted the
        zero columns (Top 2 path); ``apply_lsqr`` skips its own column
        elimination and uses ``active_mask`` only to expand the solution
        back to the full column space at the end.
    solver : str, optional
        Solver to use: 'lsmr' (default, faster convergence) or 'lsqr'.
    use_float32 : bool, optional
        If True, cast matrix data and b to float32 before solving.
        Reduces memory bandwidth (~2x faster SpMV) at the cost of precision.
    active_mask : np.ndarray of bool, optional
        When set, marks the columns of the original (uncompacted) layout
        that are present in the supplied compact CSR. Used to expand the
        compact solution back to the full column space on return.
    num_cols_full : int, optional
        Original (uncompacted) column count. Required when ``active_mask``
        is given. Equals ``A.shape[1]`` when no compaction happened upstream.
    """
    assert isinstance(A, (coo_matrix, csr_matrix)), \
        "A must be a scipy.sparse.coo_matrix or csr_matrix"
    assert isinstance(b, np.ndarray), "b must be a numpy array"
    assert isinstance(ref_shape, (list, np.ndarray, tuple)) and len(ref_shape) == 2, "ref_shape must be a list or tuple of length 2"

    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w

    # ---- Top 2 fast path: A is already compact CSR ------------------
    if isinstance(A, csr_matrix):
        if active_mask is not None:
            assert num_cols_full is not None, \
                "num_cols_full must be supplied alongside active_mask"
            num_cols = int(num_cols_full)
            n_active = int(active_mask.sum())
            assert A.shape[1] == n_active, (
                f"A.shape[1]={A.shape[1]} != active_mask.sum()={n_active}")
            x0_compressed = x0[active_mask] if x0 is not None else None
        else:
            num_cols = A.shape[1]
            n_active = num_cols
            x0_compressed = x0

        A_shape = A.shape
        if use_float32:
            print("Downcasting to float32 for faster SpMV...")
            if A.data.dtype != np.float32:
                A.data = A.data.astype(np.float32)
            _b_in = b
            b = _b_in.astype(np.float32)
            del _b_in
            if x0_compressed is not None:
                x0_compressed = x0_compressed.astype(np.float32)

        data = A.data
        new_col = A.indices

        if precondition:
            print("Applying column-norm preconditioning...")
            chunk_size = 64_000_000  # ~256 MB per chunk at f32
            col_sq_norm = np.zeros(n_active, dtype=np.float32)
            for start in range(0, data.size, chunk_size):
                stop = min(start + chunk_size, data.size)
                d_chunk = data[start:stop]
                c_chunk = new_col[start:stop]
                col_sq_norm += np.bincount(c_chunk, weights=d_chunk * d_chunk, minlength=n_active).astype(np.float32)
            col_norms = np.sqrt(col_sq_norm)
            col_norms[col_norms == 0] = 1.0
            M_inv = col_norms
            M = 1.0 / M_inv
            for start in range(0, data.size, chunk_size):
                stop = min(start + chunk_size, data.size)
                data[start:stop] *= M[new_col[start:stop]].astype(data.dtype, copy=False)
            x0_solver = x0_compressed * M_inv.astype(x0_compressed.dtype) if x0_compressed is not None else None
        else:
            M = None
            x0_solver = x0_compressed

        print(f"Solving least squares for {n_active} unknowns with {A_shape[0]} equations (solver={solver}).")
        A_csr = A
        del A

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
        if precondition:
            x_solver = x_solver * M
        if active_mask is not None:
            x = np.zeros(num_cols, dtype=x_solver.dtype)
            x[active_mask] = x_solver
        else:
            x = x_solver
        return x

    # ---- Legacy path: A is a COO. apply_lsqr does the compaction itself.
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
        # setup_lsqr workers always emit float32 for sub_data_vec, so A.data
        # is already f32 in production; skip the redundant ~25 GB copy.
        if A.data.dtype == np.float32:
            data = A.data
        else:
            data = A.data.astype(np.float32)
        # Drop the f64 b reference once the f32 cast exists (caller already released self.b;
        # this releases the local f64 reference, saving ~80 GB at no-srcmask region-10k scale).
        _b_in = b
        b = _b_in.astype(np.float32)
        del _b_in
        if x0_compressed is not None:
            x0_compressed = x0_compressed.astype(np.float32)
    else:
        data = A.data

    if precondition:
        print("Applying column-norm preconditioning...")
        # Chunked float32 accumulation of column-squared-norms.
        # Avoids materializing the full nnz-sized f64 (data**2) temp (~56 GB at no-srcmask
        # region-10k scale). f32 sum is safe: max per-column sum is bounded
        # (max data ~10 from apply_weight * max ~17k contributors ~1.7M, << f32 max 3.4e38).
        chunk_size = 64_000_000  # ~256 MB per chunk at f32
        col_sq_norm = np.zeros(n_active, dtype=np.float32)
        for start in range(0, data.size, chunk_size):
            stop = min(start + chunk_size, data.size)
            d_chunk = data[start:stop]
            c_chunk = new_col[start:stop]
            col_sq_norm += np.bincount(c_chunk, weights=d_chunk * d_chunk, minlength=n_active).astype(np.float32)
        col_norms = np.sqrt(col_sq_norm)
        col_norms[col_norms == 0] = 1.0
        M_inv = col_norms
        M = 1.0 / M_inv
        # Chunked in-place gather-multiply: avoids two full-nnz transients
        # (the M[new_col] gather and the .astype(data.dtype) copy together
        # held ~52 GB at no-srcmask region-10k scale). M values are tiny
        # (n_active entries), so per-chunk gather is cheap.
        chunk_size = 64_000_000  # ~256 MB per chunk at f32
        for start in range(0, data.size, chunk_size):
            stop = min(start + chunk_size, data.size)
            data[start:stop] *= M[new_col[start:stop]].astype(data.dtype, copy=False)
        x0_solver = x0_compressed * M_inv.astype(x0_compressed.dtype) if x0_compressed is not None else None
    else:
        M = None
        x0_solver = x0_compressed

    # Bind the row array + shape locally so we can drop the COO container immediately
    # after CSR build. Caller already released its reference (see Calibrator.apply_lsqr);
    # this lets the (~140 GB at no-srcmask region-10k scale) COO arrays be freed as soon
    # as CSR construction finishes. row/col are scipy properties without deleters, so we
    # drop A itself after rebinding row locally.
    A_row = A.row
    A_shape = A.shape
    del A
    print(f"Solving least squares for {n_active} unknowns with {A_shape[0]} equations (solver={solver}).")
    A_csr = coo_matrix((data, (A_row, new_col)), shape=(A_shape[0], n_active)).tocsr()
    del data, new_col, A_row

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
