"""Shared subframe preparation logic used by both coadd and LSQR pipelines."""

import inspect

import numpy as np
from scipy.ndimage import map_coordinates

from ..io.reproj import load_reproj_file
from ..geometry.map_helper import (bit_to_bool, make_weight, make_linear_interp_matrix,
                        chunk_to_det, det_to_sub, compute_chunk_contrib)


def _valid_row_mask(coords_yx, grid_valid_weight):
    """Rows whose bilinear sample of ``grid_valid_weight`` is finite and > 0.

    Identical to ``isfinite(sample) & (sample > 0)`` with
    ``sample = map_coordinates(grid_valid_weight, coords, order=1,
    mode='constant', cval=0)`` over every row, but only evaluates the sampler
    where the answer can be True: a sample is > 0 only if one of its four
    bilinear corners lies inside the bounding box of ``grid_valid_weight > 0``,
    i.e. ``floor(y)`` in ``[r_lo - 1, r_hi]`` and ``floor(x)`` in
    ``[c_lo - 1, c_hi]`` (non-finite coordinates fail both comparisons).  For
    a single-channel frame that is a few per cent of the rows.
    """
    y, x = coords_yx
    out = np.zeros(y.shape[0], dtype=bool)
    nz = grid_valid_weight > 0
    rows = np.flatnonzero(nz.any(axis=1))
    if rows.size == 0:
        return out
    cols = np.flatnonzero(nz.any(axis=0))
    idx = np.flatnonzero((y >= rows[0] - 1) & (y < rows[-1] + 1))
    xs = x[idx]
    idx = idx[(xs >= cols[0] - 1) & (xs < cols[-1] + 1)]
    if idx.size == 0:
        return out
    sample = map_coordinates(grid_valid_weight, np.stack([y[idx], x[idx]]),
                             order=1, mode='constant', cval=0.0)
    out[idx] = np.isfinite(sample) & (sample > 0)
    return out


def _mask_bbox(mask2d):
    """``(r0, r1, c0, c1)`` bounding box of True in a 2-D mask (exclusive ends); None if empty."""
    rows = np.flatnonzero(mask2d.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(mask2d.any(axis=0))
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


_needed_support = {}


def _accepts_needed(func):
    """Whether an offset renderer takes the ``needed`` keyword (cached per callable)."""
    key = id(func)
    hit = _needed_support.get(key)
    if hit is not None and hit[0] is func:
        return hit[1]
    try:
        params = inspect.signature(func).parameters
        ok = 'needed' in params or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    except (TypeError, ValueError):
        ok = False
    if len(_needed_support) > 64:
        _needed_support.clear()
    _needed_support[key] = (func, ok)
    return ok


def _render_offset(func, chunk_map, chunk_offset, needed):
    """Per-frame grid offset of one map at the flat grid indices ``needed`` (1-D)."""
    if func is None:
        return chunk_to_det(chunk_map, chunk_data=chunk_offset, needed=needed)
    if _accepts_needed(func):
        return func(chunk_map, chunk_offset, needed=needed)
    return np.asarray(func(chunk_map, chunk_offset)).ravel()[needed]


def _prep_subframe(file, chunk_maps=None, apply_weight=False, apply_mask=False,
                   chunk_offsets=None, det_offset_funcs=None, ignore_list=None,
                   grid_valid_weight=None, valid_threshold=0.99,
                   for_lsqr=False, oversample_factor=1,
                   # These arguments are accepted for compatibility/internal logic
                   # but might not be used depending on logic path
                   det_aux=None, postprocess_func=None, preprocess_func=None,
                   extras=None):
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
        ``chunk_to_det`` for that map.  A callable that also accepts
        ``needed=`` (flat grid indices) is asked for just those pixels.
    extras : dict or None
        If given, receives ``extras['sub_mapping']`` (the raw detector
        coordinate map of the frame) for callers that need it after the
        call, e.g. the coadd's band-centre/width sampling.

    Returns
    -------
    chunk_contribs : list of scipy.sparse matrices
        One per input chunk map (empty list when ``for_lsqr`` is False or
        ``chunk_maps`` is empty).

    Notes
    -----
    On the mosaic path (``for_lsqr=False`` with a ``grid_valid_weight``) the
    interpolation matrix, offsets, valid weight and auxiliary maps are built
    only over the bounding box of the rows that can carry nonzero weight;
    outside it the weight is exactly zero and the data untouched, so the
    returned full-frame arrays are identical to the full-frame computation.
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
    if extras is not None:
        extras['sub_mapping'] = sub_mapping

    if preprocess_func is not None:
        sub_data = preprocess_func(locals())

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
            if cm.shape != s0:
                raise ValueError("all chunk_maps must share the same shape")
        shape_sources.append(('chunk_maps[0]', s0))
    if grid_valid_weight is not None:
        shape_sources.append(('grid_valid_weight', grid_valid_weight.shape))
    if det_aux is not None:
        shape_sources.append(('det_aux[0]', np.shape(det_aux[0])))
    if shape_sources:
        interp_input_shape = shape_sources[0][1]
        for name, s in shape_sources[1:]:
            if s != interp_input_shape:
                raise ValueError(
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
    box = None          # (r0, r1, c0, c1) of the rows that can carry weight (mosaic path)
    if need_interp:
        if interp_input_shape is None:
            raise ValueError(
                "_prep_subframe needs to build an interpolation matrix but "
                "none of chunk_maps / grid_valid_weight / det_aux was given "
                "to infer the detector-grid shape from.")
        H, W = sub_data.shape
        sub_mapping_flat = sub_mapping.reshape(2, np.prod(sub_mapping.shape[1:]))
        sub_mapping_flat_scaled = sub_mapping_flat * oversample_factor
        coords_yx = sub_mapping_flat_scaled[::-1]
        # Pre-filter zero-weight rows before building the interp matrix:
        # when grid_valid_weight is available, drop rows whose bilinear
        # sample of grid_valid_weight is zero (their downstream sub_weight
        # would be zero anyway). For narrow channel masks this avoids
        # building ~70-90 % of rows whose contribution would otherwise be
        # multiplied by zero downstream. map_coordinates with
        # order=1, mode='constant', cval=0.0 is the exact bilinear sampler
        # that the interp matrix implements, so dropped rows are guaranteed
        # to be zero-contribution.
        valid_row_mask = None
        if grid_valid_weight is not None:
            valid_row_mask = _valid_row_mask(coords_yx, grid_valid_weight)
        if not for_lsqr and valid_row_mask is not None:
            # Mosaic path: everything downstream is zero outside the bbox of
            # the kept rows, so the matrix (and the products below) cover the
            # bbox only. Row entries do not depend on the row's neighbours, so
            # they are the ones the full-frame build would produce.
            box = _mask_bbox(valid_row_mask.reshape(H, W))
            if box is not None:
                r0, r1, c0, c1 = box
                interp_matrix = make_linear_interp_matrix(
                    coords_yx.reshape(2, H, W)[:, r0:r1, c0:c1].reshape(2, -1),
                    input_shape=interp_input_shape,
                    valid_row_mask=valid_row_mask.reshape(H, W)[r0:r1, c0:c1].ravel(),
                )
        else:
            interp_matrix = make_linear_interp_matrix(
                coords_yx,
                input_shape=interp_input_shape,
                valid_row_mask=valid_row_mask,
            )

    if box is not None:
        # ---- mosaic path over the bbox (identical values, ~3 % of the work) ----
        r0, r1, c0, c1 = box
        bh, bw = r1 - r0, c1 - c0
        w_box = np.ones((bh, bw), dtype=np.float32)
        if 'sub_bitmask' in result:
            # invert=True: 1 = Good pixel, 0 = Bad pixel
            w_box *= bit_to_bool(result['sub_bitmask'][r0:r1, c0:c1], ignore_list, invert=True)
        # Grid pixels the matrix reads: the only ones any renderer must produce.
        touched = np.zeros(int(np.prod(interp_input_shape)), dtype=bool)
        touched[interp_matrix.indices] = True
        needed = np.flatnonzero(touched)
        del touched
        if chunk_offsets is not None:
            if len(chunk_offsets) != len(chunk_maps):
                raise ValueError("chunk_offsets length must match chunk_maps")
            total_vals = None
            for m, off_m in enumerate(chunk_offsets):
                if off_m is None:
                    continue
                func_m = det_offset_funcs[m] if det_offset_funcs is not None else None
                vals = _render_offset(func_m, chunk_maps[m], off_m, needed)
                total_vals = vals if total_vals is None else total_vals + vals
            if total_vals is not None:
                grid_vec = np.zeros(int(np.prod(interp_input_shape)), dtype=total_vals.dtype)
                grid_vec[needed] = total_vals
                sub_data[r0:r1, c0:c1] -= (interp_matrix @ grid_vec).reshape(bh, bw)
        w_box *= (interp_matrix @ grid_valid_weight.ravel()).reshape(bh, bw)
        sub_aux = None
        if det_aux is not None:
            aux_box = [(interp_matrix @ np.asarray(a).ravel()).reshape(bh, bw) for a in det_aux]
            sub_aux = np.zeros((len(aux_box), H, W), dtype=np.result_type(*aux_box))
            for k, a in enumerate(aux_box):
                sub_aux[k, r0:r1, c0:c1] = a
        if apply_weight:
            w_box *= make_weight(sub_data[r0:r1, c0:c1])
        sub_weight[...] = 0.0
        sub_weight[r0:r1, c0:c1] = w_box
        chunk_contribs = []
    else:
        # Apply bitmask
        if 'sub_bitmask' in result:
            # invert=True: 1 = Good pixel, 0 = Bad pixel
            sub_boolmask = bit_to_bool(result['sub_bitmask'], ignore_list, invert=True)
            sub_weight *= sub_boolmask

        # Apply per-map chunk offsets (mosaic path).
        # Per-map grid offsets are accumulated, then a single det_to_sub call
        # bilinear-interpolates the total once regardless of K.
        if chunk_offsets is not None:
            if len(chunk_offsets) != len(chunk_maps):
                raise ValueError("chunk_offsets length must match chunk_maps")
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
            if interp_matrix is not None:
                sub_valid_weight = det_to_sub(grid_valid_weight, interp_matrix=interp_matrix)
                sub_weight *= sub_valid_weight
            else:
                # no row can carry weight (empty bbox on the mosaic path)
                sub_weight[...] = 0.0

        sub_aux = None
        if det_aux is not None:
            if interp_matrix is not None:
                sub_aux = np.array([det_to_sub(det_aux_data, interp_matrix=interp_matrix) for det_aux_data in det_aux])
            else:
                sub_aux = np.zeros((len(det_aux),) + sub_data.shape, dtype=np.float32)

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
