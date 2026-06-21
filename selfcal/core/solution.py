"""Solution vector parsing, encoding, and initial-guess computation."""

import numpy as np
from scipy.sparse import csr_matrix


def parse_x_sky(x, ref_shape, num_offset_groups_list, num_chunks_list, num_frames=None,
                num_sky_blocks=1):
    """Generic parse of the LSQR solution vector for any number of sky blocks.

    Returns
    -------
    sky_maps : list of np.ndarray
        Length ``num_sky_blocks``; each a ``ref_shape`` sky block (block 0 is the
        continuum, blocks 1.. are line-amplitude maps).
    det_offsets : list of np.ndarray
        One ``(num_offset_groups[m], num_chunks[m])`` array per chunk map.
    frame_scalar : np.ndarray
        Per-frame scalars (empty when num_frames is None/0).

    This is the N-component generalization; :func:`parse_x` is the back-compat
    fixed-tuple wrapper for the 1- and 2-block cases.
    """
    assert len(num_offset_groups_list) == len(num_chunks_list), (
        "num_offset_groups_list and num_chunks_list must have the same length")
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    sky_maps = [x[j * num_sky:(j + 1) * num_sky].reshape(ref_shape)
                for j in range(num_sky_blocks)]
    cursor = num_sky_blocks * num_sky

    det_offsets = []
    for ng, nc in zip(num_offset_groups_list, num_chunks_list):
        block = ng * nc
        det_offsets.append(x[cursor:cursor + block].reshape(ng, nc))
        cursor += block

    frame_scalar = x[cursor:cursor + num_frames] if num_frames else np.array([])
    return sky_maps, det_offsets, frame_scalar


def parse_x(x, ref_shape, num_offset_groups_list, num_chunks_list, num_frames=None,
            num_sky_blocks=1):
    """Back-compat fixed-tuple parse for <=2 sky blocks (use parse_x_sky for N>2).

    Returns ``(skymap, det_offsets, frame_scalar)`` for ``num_sky_blocks==1`` and
    ``(skymap, skymap_line, det_offsets, frame_scalar)`` for ``num_sky_blocks==2``.
    """
    sky_maps, det_offsets, frame_scalar = parse_x_sky(
        x, ref_shape, num_offset_groups_list, num_chunks_list,
        num_frames=num_frames, num_sky_blocks=num_sky_blocks)
    if num_sky_blocks == 1:
        return sky_maps[0], det_offsets, frame_scalar
    if num_sky_blocks == 2:
        return sky_maps[0], sky_maps[1], det_offsets, frame_scalar
    raise ValueError(
        f"parse_x returns fixed tuples for <=2 sky blocks (got {num_sky_blocks}); "
        "use parse_x_sky for the N-component list form")


def encode_x(skymap, offsets):
    """Concatenate sky + per-map offsets back into the solution vector.

    ``offsets`` may be a single ndarray (K=1 convenience) or a list of ndarrays.
    """
    parts = [skymap.flatten()]
    if isinstance(offsets, np.ndarray):
        parts.append(offsets.flatten())
    else:
        parts.extend(o.flatten() for o in offsets)
    return np.concatenate(parts)


def compute_x0_from_Ab(A, b, ref_shape, num_sky_blocks=1, active_mask=None):
    """Compute initial guess x0 assuming sky=0, solving offset = A_off^T b / A_off^T A_off diag.

    This avoids re-reading all FITS files to estimate offsets — the information
    is already encoded in the sparse matrix A and vector b from setup_lsqr. The
    diagonal LS treats every non-sky column independently, so it is agnostic to
    the per-map column layout.

    When ``num_sky_blocks=2`` (spectral_fit), the full sky block has
    ``2*num_sky`` columns and the offset block starts at ``2*num_sky``.

    Parameters
    ----------
    active_mask : np.ndarray of bool, optional
        When supplied, ``A`` is the COMPACT matrix produced by the Top 2
        column-elimination path; ``active_mask`` (length = original
        ``num_cols_full``) marks which original columns survived. The
        per-column diag-LS still runs on the compact ``A``, but the
        sky/offset boundary is computed in the compact column space
        (``n_active_sky = active_mask[:num_sky_eff].sum()``) and the
        returned ``x0`` is expanded back to the FULL (uncompacted) layout
        via the active_mask. Returns a vector of length ``num_cols_full``
        when active_mask is supplied (so callers downstream can pass it
        through ``apply_lsqr``, which expects full-layout x0 and does its
        own compression).

        When ``active_mask is None`` (default), behavior is unchanged:
        the returned ``x0`` has length ``A.shape[1]`` and the sky/offset
        boundary is ``num_sky_eff`` on the supplied ``A``.
    """
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    num_sky_eff = num_sky_blocks * num_sky
    num_cols = A.shape[1]

    # setup_lsqr now returns a csr_matrix (post Top 1 refactor); the original
    # COO path is preserved for callers that still pass a coo_matrix. We need
    # row/col/data triples to do the per-column diag-LS below, so convert CSR
    # to COO here. tocoo(copy=False) shares data/indices but does allocate a
    # transient nnz-sized row array via np.repeat — acceptable since this is
    # a one-time setup call (not in the LSQR hot loop).
    if isinstance(A, csr_matrix):
        A_coo = A.tocoo(copy=False)
    else:
        A_coo = A

    # Boundary between sky and offset blocks IN THE COMPACT COLUMN SPACE.
    # When active_mask is supplied, A is already compacted (Top 2), so the
    # original sky boundary num_sky_eff may correspond to a smaller compact
    # column index. Otherwise the matrix is in its original layout.
    if active_mask is not None:
        if active_mask.size < num_sky_eff:
            raise ValueError(
                f"active_mask.size={active_mask.size} smaller than "
                f"num_sky_eff={num_sky_eff}")
        n_active_sky = int(active_mask[:num_sky_eff].sum())
        if num_cols < n_active_sky:
            raise ValueError(
                f"A.shape[1]={num_cols} < n_active_sky={n_active_sky}; "
                "active_mask does not match A")
        boundary = n_active_sky
    else:
        boundary = num_sky_eff

    # Extract offset portion of A (columns boundary onwards)
    offset_mask = A_coo.col >= boundary
    off_row = A_coo.row[offset_mask]
    off_col = A_coo.col[offset_mask] - boundary
    off_data = A_coo.data[offset_mask]

    num_offset_cols = num_cols - boundary

    # offset_j = (A_off[:, j]^T @ b) / (A_off[:, j]^T @ A_off[:, j])
    AtA_diag = np.bincount(off_col, weights=off_data ** 2, minlength=num_offset_cols)
    Atb = np.bincount(off_col, weights=off_data * b[off_row], minlength=num_offset_cols)

    offsets = np.where(AtA_diag > 0, Atb / AtA_diag, 0.0)

    # Build the compact x0 first.
    x0_compact = np.zeros(num_cols)
    x0_compact[boundary:] = offsets

    if active_mask is None:
        return x0_compact

    # Scatter back to the full (uncompacted) column space so downstream
    # apply_lsqr can do its `x0[active_mask]` compression.
    num_cols_full = int(active_mask.size)
    x0_full = np.zeros(num_cols_full, dtype=x0_compact.dtype)
    x0_full[active_mask] = x0_compact
    return x0_full


def compute_x0_scalar_only(A, b, ref_shape, scalar_col_start, num_sky_blocks=1,
                           active_mask=None):
    """x0 for the ``use_per_frame_scalar`` setup: scalar gets diag-LS, chunks
    start at 0, sky starts at 0.

    The diag-LS for a scalar column degenerates to the weighted mean of valid
    pixel values in that frame's data (since the scalar column has constant
    ``valid_weight`` for every valid pixel in the frame). With chunk and sky
    init at 0, all per-frame DC is concentrated in the scalar at iter 0, and
    LSQR's first few iterations refine the chunk offsets around it.

    Parameters
    ----------
    scalar_col_start : int
        Column index where the per-frame scalar block begins, in the ORIGINAL
        (uncompacted) column space. For K maps, this is the value Calibrator
        stores as ``col_bases[K]``. When ``active_mask`` is supplied, this is
        still the original-layout value; the function internally derives the
        compact equivalent.
    num_sky_blocks : int
        1 for the legacy single-sky-block layout. 2 for spectral_fit mode.
    active_mask : np.ndarray of bool, optional
        When supplied, ``A`` is the COMPACT matrix produced by Top 2 column
        elimination. The returned ``x0`` is expanded back to the FULL
        (uncompacted) layout — same convention as ``compute_x0_from_Ab``.
    """
    x0 = compute_x0_from_Ab(A, b, ref_shape, num_sky_blocks=num_sky_blocks,
                            active_mask=active_mask)
    ref_h, ref_w = ref_shape
    num_sky_eff = num_sky_blocks * ref_h * ref_w
    # x0 is in the FULL layout iff active_mask was supplied; either way the
    # chunk-block slice [num_sky_eff:scalar_col_start] addresses the same
    # original columns. The scatter through active_mask leaves zero-coverage
    # original columns at 0, so this zeroing is still correct.
    x0[num_sky_eff:scalar_col_start] = 0.0
    return x0
