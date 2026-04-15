"""Solution vector parsing, encoding, and initial-guess computation."""

import numpy as np


def parse_x(x, ref_shape, num_offset_groups, num_chunks, num_frames=None):
    """Parse the LSQR solution vector x into sky, detector offset, and frame scalar components.

    Parameters
    ----------
    num_offset_groups : int
        Number of offset groups (= num_frames when det_groups is not used).
    num_chunks : int
        Number of chunks per offset group.
    num_frames : int or None
        If not None, the last num_frames entries are per-frame scalars.
    """
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    num_offset = num_offset_groups * num_chunks
    skymap = x[:num_sky].reshape(ref_shape)
    det_offset = x[num_sky:num_sky + num_offset].reshape(num_offset_groups, num_chunks)
    frame_scalar = x[num_sky + num_offset:] if num_frames else np.array([])
    return skymap, det_offset, frame_scalar

def encode_x(skymap, offset):
    """Utility function to encode the sky and offset components back into a single vector x."""
    return np.concatenate([skymap.flatten(), offset.flatten()])

def compute_x0_from_Ab(A, b, ref_shape, num_offset_groups=None):
    """Compute initial guess x0 assuming sky=0, solving offset = A_off^T b / A_off^T A_off diag.

    This avoids re-reading all FITS files to estimate offsets — the information
    is already encoded in the sparse matrix A and vector b from setup_lsqr.
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
