"""Shared subframe preparation logic used by both coadd and LSQR pipelines."""

import numpy as np

from .io import load_reproj_file
from .MapHelper import (bit_to_bool, make_weight, make_linear_interp_matrix,
                        chunk_to_det, det_to_sub, compute_chunk_contrib)


def _prep_subframe(file, chunk_maps=None, apply_weight=False, apply_mask=False,
                   chunk_offsets=None, det_offset_funcs=None, ignore_list=None,
                   grid_valid_weight=None, valid_threshold=0.99,
                   for_lsqr=False, oversample_factor=1,
                   # These arguments are accepted for compatibility/internal logic
                   # but might not be used depending on logic path
                   det_aux=None, postprocess_func=None, preprocess_func=None):
    """
    Prepares data from a single file for co-addition or lsqr.

    Parameters
    ----------
    chunk_maps : list of np.ndarray or None
        K chunk maps. All must share the same shape so a single
        interpolation matrix can be reused. None or an empty list disables
        chunk-based logic.
    chunk_offsets : list of np.ndarray or None
        Per-map per-chunk offsets to subtract from this frame (mosaic path
        only). Length-K list aligned with ``chunk_maps``; the per-map grid
        offsets are accumulated into a single ``total_grid_offset`` and
        subtracted via one ``det_to_sub`` call. ``None`` skips offset
        subtraction entirely.
    det_offset_funcs : list of callable or None
        Per-map ``(chunk_map, chunk_offset) -> grid_offset`` callables.
        ``None`` (or per-map ``None``) falls back to the standard
        ``chunk_to_det`` for that map.

    Returns
    -------
    chunk_contribs : list of scipy.sparse matrices
        One per input chunk map (empty list when ``for_lsqr`` is False or
        ``chunk_maps`` is empty).
    """
    if ignore_list is None: ignore_list = []
    if chunk_maps is None: chunk_maps = []

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
    interp_input_shape = None
    if chunk_maps:
        # All maps must share shape (single interp matrix is reused across maps)
        shape0 = chunk_maps[0].shape
        for cm in chunk_maps[1:]:
            assert cm.shape == shape0, "all chunk_maps must share the same shape"
        interp_input_shape = shape0
    if (chunk_maps) or (chunk_offsets is not None) or (for_lsqr) or (det_aux is not None) or (grid_valid_weight is not None):
        sub_mapping_flat = sub_mapping.reshape(2, np.prod(sub_mapping.shape[1:]))
        sub_mapping_flat_scaled = sub_mapping_flat * oversample_factor
        interp_matrix = make_linear_interp_matrix(sub_mapping_flat_scaled[::-1], input_shape=interp_input_shape)

    # Apply per-map chunk offsets (mosaic path).
    # Per-map grid offsets are accumulated, then a single det_to_sub call
    # bilinear-interpolates the total once regardless of K.
    if chunk_offsets is not None:
        assert len(chunk_offsets) == len(chunk_maps), \
            "chunk_offsets length must match chunk_maps"
        total_grid_offset = None
        for m, off_m in enumerate(chunk_offsets):
            if off_m is None:
                continue
            cm = chunk_maps[m]
            func_m = det_offset_funcs[m] if det_offset_funcs is not None else None
            if func_m is not None:
                grid_offset_m = func_m(cm, off_m)
            else:
                grid_offset_m = chunk_to_det(cm, chunk_data=off_m)
            if total_grid_offset is None:
                total_grid_offset = grid_offset_m
            else:
                total_grid_offset = total_grid_offset + grid_offset_m
        if total_grid_offset is not None:
            sub_offset = det_to_sub(total_grid_offset, interp_matrix=interp_matrix)
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

    chunk_contribs = []
    if for_lsqr:
        chunk_contribs = [compute_chunk_contrib(cm, interp_matrix) for cm in chunk_maps]

    if postprocess_func is not None:
        sub_data = postprocess_func(locals())

    # Check for NaNs and set corresponding weights to 0
    nan_mask = np.isnan(sub_data)
    sub_data[nan_mask] = 0.0
    sub_weight[nan_mask] = 0.0

    return ref_coords, sub_data, sub_weight, chunk_contribs, sub_aux
