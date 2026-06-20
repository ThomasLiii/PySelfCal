"""Shared subframe preparation logic used by both coadd and LSQR pipelines."""

import numpy as np
from scipy.ndimage import map_coordinates

from ..io.reproj import load_reproj_file
from ..geometry.MapHelper import (bit_to_bool, make_weight, make_linear_interp_matrix,
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

    # Compute bilinear interpolation matrix for mapping between chunk and subframe.
    # Infer the detector-grid shape from whichever detector-space input is
    # provided (chunk_maps[0], grid_valid_weight, det_aux[0]). All such inputs
    # live on the same grid, so we cross-check that any provided shapes agree.
    interp_matrix = None
    interp_input_shape = None
    shape_sources = []
    if chunk_maps:
        s0 = chunk_maps[0].shape
        for cm in chunk_maps[1:]:
            assert cm.shape == s0, "all chunk_maps must share the same shape"
        shape_sources.append(('chunk_maps[0]', s0))
    if grid_valid_weight is not None:
        shape_sources.append(('grid_valid_weight', grid_valid_weight.shape))
    if det_aux is not None:
        shape_sources.append(('det_aux[0]', np.shape(det_aux[0])))
    if shape_sources:
        interp_input_shape = shape_sources[0][1]
        for name, s in shape_sources[1:]:
            assert s == interp_input_shape, (
                f"_prep_subframe detector-space shape mismatch: "
                f"{shape_sources[0][0]}={interp_input_shape} vs {name}={s}")

    # Build interp_matrix iff a downstream step actually needs it.
    # for_lsqr alone with empty chunk_maps is a no-op (no chunk_contribs to
    # build), so we don't trigger on it directly — chunk_maps non-empty is
    # the real trigger for the LSQR path.
    need_interp = (
        bool(chunk_maps)
        or chunk_offsets is not None
        or det_aux is not None
        or grid_valid_weight is not None
    )
    if need_interp:
        if interp_input_shape is None:
            raise ValueError(
                "_prep_subframe needs to build an interpolation matrix but "
                "none of chunk_maps / grid_valid_weight / det_aux was given "
                "to infer the detector-grid shape from.")
        sub_mapping_flat = sub_mapping.reshape(2, np.prod(sub_mapping.shape[1:]))
        sub_mapping_flat_scaled = sub_mapping_flat * oversample_factor
        # Row-slice-first: when grid_valid_weight is available, pre-filter
        # the rows whose bilinear sample of grid_valid_weight is zero (i.e.
        # whose downstream sub_weight will be zero anyway). For narrow
        # channel masks this drops ~70-90 % of rows that the current code
        # builds and then multiplies by zero. map_coordinates with
        # order=1, mode='constant', cval=0.0 is the exact bilinear sampler
        # that the interp matrix implements, so dropped rows are guaranteed
        # to be zero-contribution. coords are (row_coords, col_coords);
        # map_coordinates wants (row_coords, col_coords) too — we pass them
        # directly. Out-of-bounds and NaN coords both map to 0 (NaN comes
        # out as NaN, but isfinite check guards that).
        valid_row_mask = None
        if grid_valid_weight is not None:
            coords_for_filter = sub_mapping_flat_scaled[::-1]
            sample = map_coordinates(
                grid_valid_weight,
                coords_for_filter,
                order=1, mode='constant', cval=0.0,
            )
            valid_row_mask = np.isfinite(sample) & (sample > 0)
        interp_matrix = make_linear_interp_matrix(
            sub_mapping_flat_scaled[::-1],
            input_shape=interp_input_shape,
            valid_row_mask=valid_row_mask,
        )

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
