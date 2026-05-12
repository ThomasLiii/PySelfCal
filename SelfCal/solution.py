"""Solution vector parsing, encoding, and initial-guess computation."""

import numpy as np


def parse_x(x, ref_shape, num_offset_groups_list, num_chunks_list, num_frames=None):
    """Parse the LSQR solution vector x into sky, per-map detector offsets, and frame scalar.

    Parameters
    ----------
    num_offset_groups_list : list of int
        Number of offset groups for each chunk map (= num_frames when no det_groups
        is set, or len(unique groups), or num_frames in template mode).
    num_chunks_list : list of int
        Number of chunks per offset group for each chunk map (1 in template mode).
    num_frames : int or None
        If not None, the last num_frames entries of x are per-frame scalars.

    Returns
    -------
    skymap : np.ndarray
        Reshaped sky map block.
    det_offsets : list of np.ndarray
        One ``(num_offset_groups[m], num_chunks[m])`` array per chunk map.
    frame_scalar : np.ndarray
        Per-frame scalars (empty when num_frames is None).
    """
    assert len(num_offset_groups_list) == len(num_chunks_list), (
        "num_offset_groups_list and num_chunks_list must have the same length")
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    skymap = x[:num_sky].reshape(ref_shape)

    det_offsets = []
    cursor = num_sky
    for ng, nc in zip(num_offset_groups_list, num_chunks_list):
        block = ng * nc
        det_offsets.append(x[cursor:cursor + block].reshape(ng, nc))
        cursor += block

    frame_scalar = x[cursor:cursor + num_frames] if num_frames else np.array([])
    return skymap, det_offsets, frame_scalar


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


def compute_x0_from_Ab(A, b, ref_shape):
    """Compute initial guess x0 assuming sky=0, solving offset = A_off^T b / A_off^T A_off diag.

    This avoids re-reading all FITS files to estimate offsets — the information
    is already encoded in the sparse matrix A and vector b from setup_lsqr. The
    diagonal LS treats every non-sky column independently, so it is agnostic to
    the per-map column layout.
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


def compute_x0_scalar_only(A, b, ref_shape, scalar_col_start):
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
        Column index where the per-frame scalar block begins. For K maps,
        this is the value Calibrator stores as ``col_bases[K]``.
    """
    x0 = compute_x0_from_Ab(A, b, ref_shape)
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    x0[num_sky:scalar_col_start] = 0.0
    return x0
