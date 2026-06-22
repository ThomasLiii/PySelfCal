"""LSQR system assembly: parent-side orchestration.

Split out of the former monolithic lsqr.py. ``setup_lsqr`` builds the sparse
design matrix + RHS for K offset blocks and N sky components: it resolves the
sky model + column layout, stages shared memory, dispatches the per-batch row
assembly (selfcal.core.assembly) across a process pool, appends the global
constraint blocks (selfcal.core.constraint_builders), assembles the CSR, and
applies the Top-2 column compaction. Also the post-solve coverage/Fisher parsers.
"""
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.shared_memory import SharedMemory
from scipy.sparse import coo_matrix, csr_matrix

from .layout import SystemLayout
from ..models.sky_model import SkyModel
from .constraint_builders import (mean_offset_block, sky_damping_block,
                                  offset_damping_block)
from .assembly import _prep_lsqr_batch_worker


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
                "selfcal.instruments.spherex.spherex_utility.load_calibration(band=detector).")
        print(f"Spectral mode ON: {num_sky_blocks} sky blocks {sky_model.names}, "
              f"{num_sky_blocks * num_sky} sky cols, damp_weight_line={damp_weight_line}.")
    # Positional det_aux -> named aux dict (SPHEREx convention: [BC, BW]).
    aux_keys = ['BC', 'BW'][:len(det_aux)] if det_aux is not None else []

    # --- Column layout (single source of truth: selfcal.layout.SystemLayout) ---
    # SystemLayout computes the per-map group mapping, template normalization,
    # col_bases, the per-frame scalar block, and the total column count. The
    # Calibrator builds the same layout from the same inputs (see
    # pipeline_wrapper.Calibrator.setup_lsqr) so the parent-side and parse-side
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


def parse_pixel_counts_sky(pixel_counts, ref_shape, num_offset_groups_list, chunk_maps,
                           num_sky_blocks=1):
    """Generic coverage slicing for any number of sky blocks.

    Returns ``(sky_coverages, offset_coverages, offset_valid_fracs)`` where
    ``sky_coverages`` is a length-``num_sky_blocks`` list of ``ref_shape`` arrays
    (block 0 = continuum, 1.. = line blocks). :func:`parse_pixel_counts` is the
    back-compat fixed-tuple wrapper for <=2 blocks.
    """
    num_sky = ref_shape[0] * ref_shape[1]
    sky_coverages = [pixel_counts[j * num_sky:(j + 1) * num_sky].reshape(ref_shape)
                     for j in range(num_sky_blocks)]
    cursor = num_sky_blocks * num_sky

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
    return sky_coverages, offset_coverages, offset_valid_fracs


def parse_pixel_counts(pixel_counts, ref_shape, num_offset_groups_list, chunk_maps,
                       num_sky_blocks=1):
    """Back-compat fixed-tuple coverage slicing for <=2 sky blocks.

    Returns ``(skymap_coverage, offset_coverages, offset_valid_fracs)`` for 1
    block and ``(skymap_coverage, line_coverage, offset_coverages,
    offset_valid_fracs)`` for 2. Use :func:`parse_pixel_counts_sky` for N>2.
    """
    sky_coverages, offset_coverages, offset_valid_fracs = parse_pixel_counts_sky(
        pixel_counts, ref_shape, num_offset_groups_list, chunk_maps,
        num_sky_blocks=num_sky_blocks)
    if num_sky_blocks == 2:
        return sky_coverages[0], sky_coverages[1], offset_coverages, offset_valid_fracs
    return sky_coverages[0], offset_coverages, offset_valid_fracs


def parse_pixel_fisher_sky(pixel_fisher, ref_shape, num_sky_blocks=1):
    """Generic Fisher slicing: returns a length-``num_sky_blocks`` list of
    ``ref_shape`` arrays (block 0 = continuum). See :func:`parse_pixel_counts_sky`.
    """
    num_sky = ref_shape[0] * ref_shape[1]
    return [pixel_fisher[j * num_sky:(j + 1) * num_sky].reshape(ref_shape)
            for j in range(num_sky_blocks)]


def parse_pixel_fisher(pixel_fisher, ref_shape, num_sky_blocks=1):
    """Back-compat fixed-tuple Fisher slicing for <=2 sky blocks.

    Returns ``(skymap_fisher, line_fisher)`` where ``line_fisher`` is None for
    1 block. Use :func:`parse_pixel_fisher_sky` for N>2.
    """
    sky_fishers = parse_pixel_fisher_sky(pixel_fisher, ref_shape, num_sky_blocks=num_sky_blocks)
    if num_sky_blocks == 2:
        return sky_fishers[0], sky_fishers[1]
    return sky_fishers[0], None


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


