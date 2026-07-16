"""Per-subframe LSQR row assembly (runs in the worker processes).

Split out of the former monolithic lsqr.py. ``_prep_lsqr`` emits one subframe's
sparse rows (sky components, offsets, per-frame scalar, per-frame constraints);
``_prep_lsqr_batch_worker`` reconstructs the shared-memory arrays once per batch
and assembles that batch's rows. ``selfcal.core.system.setup_lsqr`` dispatches
``_prep_lsqr_batch_worker`` to a ProcessPoolExecutor.
"""
import traceback

import numpy as np
from multiprocessing.shared_memory import SharedMemory

from .subframe import _prep_subframe
from ..geometry.map_helper import find_outliers, check_invalid
from ..models.offset_basis import eval_offset_basis, n_coef


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
    poly_basis_list = task_params.get('poly_basis_list') or [None] * len(task_params['chunk_maps'])
    offset_line_downweight = float(task_params.get('offset_line_downweight', 0.0) or 0.0)
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

        # --- Offset line-emission downweight (approach #2) ---
        # Fit the per-frame offset from LINE-FREE observations: scale each offset
        # row's contribution by (1 - rho*G), where G = the (peak-normalized) line
        # coefficient of the last spectral sky block at that observation. Near the
        # line peak (G~1) the offset is ~ignored and the polynomial interpolates
        # through from the wings (G~0), so the offset stays a clean zodi estimate
        # and cannot absorb the PAH. rho=0 (default) => factor 1 => byte-identical.
        _off_dw = None
        if offset_line_downweight > 0.0 and J >= 2 and sky_coeffs[-1] is not None:
            _off_dw = 1.0 - offset_line_downweight * np.clip(
                np.asarray(sky_coeffs[-1], dtype=np.float64), 0.0, None)
            np.clip(_off_dw, 0.0, 1.0, out=_off_dw)

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
            if det_template_list[m] is not None:
                # Template mode: one alpha column per frame for this map
                O_rows_parts.append(sub_idx_m)
                O_cols_parts.append(np.full(len(chunk_idx_m), col_bases[m] + index, dtype=np.int64))
                O_data_parts.append(valid_weight[sub_idx_m] * chunk_vals_m
                                    * det_template_list[m][group_idx_list[m], chunk_idx_m])
            elif poly_basis_list[m] is not None:
                # Hard poly-basis: the offset is a polynomial in an abstract
                # per-chunk COORDINATE (``chunk_coord``), independent per per-chunk
                # GROUP (``chunk_group``, ``num_groups`` of them) — the instrument
                # supplies both, so this core is encoding-agnostic. Each
                # (chunk-contrib, obs) entry emits n_coef nnz into coeff columns
                # a[frame, group, k], coefficient w * chunk_val * B_k(coord).
                pb = poly_basis_list[m]
                ng = int(pb['num_groups']); ncf = n_coef(pb)
                coord = np.asarray(pb['chunk_coord'])[chunk_idx_m]
                grp = np.asarray(pb['chunk_group'])[chunk_idx_m]
                B = eval_offset_basis(coord, pb)                                 # (n, ncf)
                w_cv = valid_weight[sub_idx_m] * chunk_vals_m                    # (n,)
                if _off_dw is not None:
                    w_cv = w_cv * _off_dw[sub_idx_m]
                base = col_bases[m] + (group_idx_list[m] * (ng * ncf))
                coeff_base = grp * ncf                                           # + k below
                for k in range(ncf):
                    O_rows_parts.append(sub_idx_m)
                    O_cols_parts.append(base + coeff_base + k)
                    O_data_parts.append(w_cv * B[:, k])
            else:
                _fw = valid_weight[sub_idx_m] * chunk_vals_m
                if _off_dw is not None:
                    _fw = _fw * _off_dw[sub_idx_m]
                O_rows_parts.append(sub_idx_m)
                O_cols_parts.append(col_bases[m]
                                    + (group_idx_list[m] * num_chunks_list[m])
                                    + chunk_idx_m)
                O_data_parts.append(_fw)
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
                if poly_basis_list[m] is not None:
                    # Hard poly-basis carries no per-chunk adjacency/penalty rows —
                    # the polynomial is exact (no weight knob).
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

