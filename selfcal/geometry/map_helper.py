import logging

import numpy as np

from scipy.sparse import coo_matrix, csr_matrix

from scipy.interpolate import PchipInterpolator, CubicSpline, Akima1DInterpolator, RectBivariateSpline, griddata
from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)

def bit_to_bool(bitmask_array, ignore_list=None, bitmask_header=None, invert=False, expand_bits=False):
    # By default, 1 indicates bad pixels and 0 indicates good pixels.
    # If invert=True, this is flipped.
    if ignore_list is None:
        ignore_list = []
    ignore_mask_val = np.uint32(0)
    for item in ignore_list:
        bit = bitmask_header[item] if bitmask_header is not None else item
        ignore_mask_val |= np.uint32(1 << bit)
    
    relevant_mask = np.invert(ignore_mask_val)

    if expand_bits:
        if bitmask_header is not None:
            return {
                name: (~((bitmask_array & (1 << bit)) != 0) if invert else ((bitmask_array & (1 << bit)) != 0))
                for name, bit in bitmask_header.items()
                if not (ignore_mask_val & (1 << bit))
            }
        else:
            # Return (32, ...) boolean array
            bits = np.arange(32, dtype=np.uint32)
            
            # Broadcast: (32, 1) & (1, N) -> (32, N)
            # Ensure bitmask_array is at least 1D for broadcasting or expand dims appropriately
            # Using bitmask_array & (1 << bits)[:, None] works if bitmask_array is (N,)
            expanded_mask = (bitmask_array & (1 << bits)[:, None, None]) != 0
            
            # Apply ignore mask (32, 1) against (32, N)
            keep_bits = (relevant_mask & (1 << bits)) != 0
            expanded_mask &= keep_bits[:, None, None]
            
            return ~expanded_mask if invert else expanded_mask

    mask = (bitmask_array & relevant_mask) != 0
    return ~mask if invert else mask

def bool_to_bit(expanded_mask, dtype=np.uint32):
    """
    Converts an expanded boolean mask (32, N, ...) back into a 
    compact integer bitmask (N, ...).
    """
    
    # Create bit values [1, 2, 4, 8...] as a (32, 1) column vector
    bit_values = (1 << np.arange(32, dtype=dtype))[:, None, None]
    
    # Multiply the (32, N) boolean mask by the (32, 1) bit values
    # This uses broadcasting, resulting in a (32, N) array
    # Then, sum along the bit-axis (axis=0) to collapse into (N,)
    bitmask = np.sum(expanded_mask * bit_values, axis=0, dtype=dtype)
    
    return bitmask

def make_weight(frame, sigma=1.4, floor=1e-4):
    '''Make weight used for weighted mean / weighted-LSQR.

    Poisson-optimal inverse-variance weight: w = 1/sqrt(|frame| + floor).
    Bright pixels get down-weighted ∝ 1/sqrt(brightness), so the per-row L2
    contribution (∝ w²) is ∝ 1/brightness — matching shot-noise variance.

    Do not steepen this to 1/frame² (that assumes variance ∝ brightness⁴):
    the ~10000x bright-vs-dim suppression ill-conditions the solve. The
    Poisson form keeps ~10x dynamic range between dim and bright pixels —
    enough to break the sky→offset leakage that produces bowls around bright
    cirrus, while keeping the solve stable.
    '''
    abs_frame = np.abs(frame) + floor
    weight = 1.0 / np.sqrt(abs_frame)
    return np.nan_to_num(weight, nan=0)

def find_outliers(data, threshold=3):
    '''Return 1 where outlier is detected, else return 0'''
    if np.all(np.isnan(data)):
        return np.zeros_like(data, dtype=bool)
    median = np.nanmedian(data)
    nmad = 1.4826 * np.nanmedian(np.abs(data - median))
    if nmad == 0:
        return np.zeros_like(data, dtype=bool)
    z_score = (data - median)/nmad
    return np.abs(z_score) > threshold


def find_outliers_grouped(data, group_ids, threshold=3):
    '''Robust outlier flag computed WITHIN each group separately.

    Like :func:`find_outliers` but the median + NMAD (hence the z-score) are
    computed within each ``group_ids`` value rather than over the whole array.
    For SPHEREx the group is the SUBCHANNEL, so a bright star pixel is judged
    against its own subchannel's sky (not the frame-wide distribution that mixes
    subchannels of very different brightness). NaN data are ignored; groups with
    < 3 finite pixels or zero NMAD flag nothing. Same (H, W) bool return.
    '''
    out = np.zeros(np.shape(data), dtype=bool)
    fd = np.asarray(data).ravel()
    fg = np.asarray(group_ids).ravel()
    fo = out.ravel()
    finite = np.isfinite(fd)
    if not finite.any():
        return out
    for g in np.unique(fg[finite]):
        m = finite & (fg == g)
        if m.sum() < 3:
            continue
        d = fd[m]
        med = np.median(d)
        nmad = 1.4826 * np.median(np.abs(d - med))
        if nmad == 0:
            continue
        sel = np.nonzero(m)[0]
        fo[sel[np.abs((d - med) / nmad) > threshold]] = True
    return out

def compute_chunk_edges(det_shape, chunk_size):
    '''Split detector into chunks and return edge of chunks'''
    det_h, det_w = det_shape
    chunk_h, chunk_w = chunk_size
    y_edges = np.arange(0, det_h + 1, chunk_h)
    x_edges = np.arange(0, det_w + 1, chunk_w)
    
    return x_edges, y_edges

def bin2d(arr, bin_factor, bin_func=np.mean):
    """
    Bins a 2D array or a stack of 2D arrays (3D) by a given factor.
    """
    if arr.ndim == 2:
        h, w = arr.shape
        if not (h % bin_factor == 0 and w % bin_factor == 0):
            raise ValueError("h and w must be divisible by bin_factor")
        h_bins = h // bin_factor
        w_bins = w // bin_factor

        # Reshape and apply function for a single 2D array
        reshaped = arr.reshape(h_bins, bin_factor, w_bins, bin_factor)
        binned = bin_func(reshaped, axis=(1, 3))

    elif arr.ndim == 3:
        num_layers, h, w = arr.shape
        if not (h % bin_factor == 0 and w % bin_factor == 0):
            raise ValueError("h and w must be divisible by bin_factor")
        h_bins = h // bin_factor
        w_bins = w // bin_factor
        
        # Reshape and apply function for a stack of 2D arrays
        reshaped = arr.reshape(num_layers, h_bins, bin_factor, w_bins, bin_factor)
        binned = bin_func(reshaped, axis=(2, 4)) # Bin along the new height and width factor axes
        
    else:
        raise ValueError(f"bin2d supports 2D or 3D arrays, but got {arr.ndim} dimensions.")
        
    return binned

def compute_crop(ref_shape, coords):
    y_min, y_max, x_min, x_max = coords
    H, W = ref_shape

    y0, y1 = max(y_min, 0), min(y_max, H)
    x0, x1 = max(x_min, 0), min(x_max, W)
    dy0, dx0 = y0 - y_min, x0 - x_min
    dy1, dx1 = y1 - y_min, x1 - x_min

    sub_crop = np.s_[dy0:dy1, dx0:dx1]
    ref_crop = np.s_[y0:y1, x0:x1]
    return sub_crop, ref_crop

def chunk_to_det(chunk_map, chunk_data, needed=None):
    """Render per-chunk values onto the detector grid.

    ``needed`` (flat grid indices) restricts the render to those pixels and
    returns a 1-D array in that order; it is the same gather as
    ``chunk_data[chunk_map].ravel()[needed]`` without the full-grid render.
    """
    if needed is not None:
        ids = chunk_map.ravel()[needed]
        vals = chunk_data[ids]
        if ids.size and ids.min() < 0:                # outside every chunk: no offset
            vals = np.where(ids >= 0, vals, 0)
        return vals
    det_offset = chunk_data[chunk_map]
    if chunk_map.size and chunk_map.min() < 0:
        det_offset = np.where(chunk_map >= 0, det_offset, 0)
    return det_offset

def make_linear_interp_matrix(coords, input_shape, valid_row_mask=None):
    """
    Build a sparse bilinear-interpolation matrix (CSR, shape (N_total, H*W))
    mapping a flattened (H, W) source grid to N_total fractional sample
    coordinates; rows with non-finite coordinates are left structurally empty.

    Parameters
    ----------
    coords : ndarray, shape (2, N_total)
        Stacked (row_coords, col_coords) into the input grid.
    input_shape : tuple (H, W)
        Shape of the source detector grid.
    valid_row_mask : ndarray of bool, shape (N_total,), optional
        If provided, rows where this mask is False are skipped entirely:
        no (row, col, data) entries are added to the COO output for them.
        The output matrix still has shape (N_total, H*W) so callers see the
        same indexing — only the structural nonzero rows differ. For rows
        the mask keeps, the matrix entries are byte-identical to the
        ``valid_row_mask=None`` build. Intended use is callers (e.g.
        ``_prep_subframe``) that know a priori certain rows have zero
        downstream weight (``sub_weight=0``) and can be omitted without
        changing any final result.
    """
    # Coords = (y_coords, x_coords)
    H, W = input_shape
    N_total = coords.shape[1]

    # 1. Identify valid inputs (removing NaNs)
    # np.isfinite is generally slightly faster than ~np.isnan
    valid_mask = np.isfinite(coords[0]) & np.isfinite(coords[1])
    if valid_row_mask is not None:
        # Caller-supplied row filter: drop rows whose downstream sub_weight
        # will be zero anyway. The kept set is a (strict) subset of the
        # finite-coord set; for kept rows the produced matrix entries are
        # bit-identical to the unfiltered build.
        valid_mask &= valid_row_mask
    valid_idxs = np.where(valid_mask)[0] # Indices in original array

    # Filter coordinates immediately
    row_coords = coords[0][valid_mask]
    col_coords = coords[1][valid_mask]
    
    n_valid = len(row_coords)
    if n_valid == 0:
        return coo_matrix((0, 0), shape=(N_total, H * W)).tocsr()

    # 2. Integer floors and fractional parts
    # Floor returns float, safe to cast to int32 for indexing
    r0 = np.floor(row_coords).astype(np.int32)
    c0 = np.floor(col_coords).astype(np.int32)
    
    r_frac = row_coords - r0
    c_frac = col_coords - c0
    rf_inv = 1.0 - r_frac
    cf_inv = 1.0 - c_frac

    # 3. Bounds check. Test r0/c0 (size-N arrays) and derive the four
    # per-corner in-bounds masks from them, rather than materializing
    # 4N-sized expanded corner-coordinate arrays just to range-test them.
    in_r0 = (r0 >= 0) & (r0 < H)
    in_c0 = (c0 >= 0) & (c0 < W)
    in_r1 = (r0 + 1 >= 0) & (r0 + 1 < H)
    in_c1 = (c0 + 1 >= 0) & (c0 + 1 < W)

    # 4. Allocation
    total_entries = n_valid * 4
    
    # Use float32 for weights to save memory (sufficient precision for interp)
    data = np.empty(total_entries, dtype=np.float32)
    cols = np.empty(total_entries, dtype=np.int32)
    
    # Create the row indices repeated 4 times
    rows = np.repeat(valid_idxs, 4).astype(np.int32)

    # 5. Fill Weights and Indices (Strided assignment)
    base_idx = r0 * W + c0
    
    # We construct the bounds_mask directly using the pre-calced booleans
    bounds_mask = np.empty(total_entries, dtype=bool)

    # Top-Left (r0, c0)
    data[0::4] = rf_inv * cf_inv
    cols[0::4] = base_idx
    bounds_mask[0::4] = in_r0 & in_c0

    # Top-Right (r0, c0+1)
    data[1::4] = rf_inv * c_frac
    cols[1::4] = base_idx + 1
    bounds_mask[1::4] = in_r0 & in_c1

    # Bottom-Left (r0+1, c0)
    data[2::4] = r_frac * cf_inv
    cols[2::4] = base_idx + W
    bounds_mask[2::4] = in_r1 & in_c0

    # Bottom-Right (r0+1, c0+1)
    data[3::4] = r_frac * c_frac
    cols[3::4] = base_idx + W + 1
    bounds_mask[3::4] = in_r1 & in_c1

    # 6. Final Construction
    # Filter using the boolean mask
    keep_rows = rows[bounds_mask]
    keep_cols = cols[bounds_mask]
    keep_data = data[bounds_mask]

    interp_matrix = coo_matrix(
        (keep_data, (keep_rows, keep_cols)), 
        shape=(N_total, H * W),
        dtype=np.float32
    )
    
    return interp_matrix.tocsr()

def det_to_sub(det_data, sub_mapping=None, interp_matrix=None, sub_shape=None):
    """A detector-grid map resampled onto a subframe, through the bilinear
    ``interp_matrix`` (rows = subframe pixels, in row-major order; ``sub_shape``
    gives the subframe shape, default: a square) or by direct interpolation at
    the ``sub_mapping`` detector coordinates."""
    if interp_matrix is not None:
        if sub_shape is None:
            sub_width = np.sqrt(interp_matrix.shape[0]).astype(np.int32)
            sub_shape = (sub_width, sub_width)
        det_data_flat = det_data.ravel()
        sub_data_flat = interp_matrix @ det_data_flat
        sub_data = sub_data_flat.reshape(sub_shape)
    elif sub_mapping is not None:
        sub_data = map_coordinates(det_data, sub_mapping[::-1], order=1, output=np.float32)
    else:
        raise ValueError("Either sub_mapping or interp_matrix must be provided.")
    return sub_data

_chunk_map_parsed_cache = {}


def _parse_chunk_map(chunk_map):
    """One-hot CSR ``(n_pixels, n_chunks)`` for a chunk map.

    Depends only on ``chunk_map`` (constant across every frame in a batch), so
    it is memoized by object identity. The cached entry also holds a reference
    to the source array: that keeps its ``id`` valid (so a different array can
    never alias a live cache key) and the ``is`` re-check makes a stale hit
    impossible. The cache is bounded since only a handful of distinct maps
    ever appear. Result is a pure function of ``chunk_map``, so reuse is
    bit-identical to rebuilding.
    """
    key = id(chunk_map)
    entry = _chunk_map_parsed_cache.get(key)
    if entry is not None and entry[0] is chunk_map:
        return entry[1]

    chunk_map_flat = chunk_map.ravel()
    total_rows = chunk_map_flat.size
    total_cols = chunk_map_flat.max() + 1
    inside = chunk_map_flat >= 0
    if inside.all():
        indptr = np.arange(total_rows + 1)
        indices = chunk_map_flat
        data = np.ones(total_rows, dtype=np.float32)
    else:
        # -1 marks pixels outside every chunk (a gap between detectors, a
        # masked region): their rows stay empty.
        indptr = np.concatenate([[0], np.cumsum(inside)])
        indices = chunk_map_flat[inside]
        data = np.ones(indices.size, dtype=np.float32)
    chunk_map_parsed = csr_matrix((data, indices, indptr), shape=(total_rows, total_cols))

    if len(_chunk_map_parsed_cache) > 8:
        _chunk_map_parsed_cache.clear()
    _chunk_map_parsed_cache[key] = (chunk_map, chunk_map_parsed)
    return chunk_map_parsed


def compute_chunk_contrib(chunk_map, interp_matrix=None):
    """Computes the sparse matrix contribution for LSQR."""
    chunk_map_parsed = _parse_chunk_map(chunk_map)
    if interp_matrix is not None:
        chunk_contrib = (interp_matrix @ chunk_map_parsed).T
        return chunk_contrib
    else:
        return chunk_map_parsed

def check_invalid(arr):
    if np.issubdtype(arr.dtype, np.integer):
        invalid = arr == -9999
    elif np.issubdtype(arr.dtype, np.floating):
        invalid = np.isnan(arr)
    else:
        raise ValueError("Unsupported array data type for invalid check.")
    return invalid

def linear_spline(x_sample, y_sample):
    def interpolator(x):
        return np.interp(x, x_sample, y_sample)
    return interpolator

def mean_preserving_spline(x_edge, y_mean, method='cubic'):
    """
    Generates a mean-preserving spline function f(x) based on edge
    positions x_edge and the average value y_mean in each interval.

    The function f(x) is constructed as the derivative of a monotonic
    cubic spline F(x), where F(x) is the integral of f(x).
    """
    if len(x_edge) != len(y_mean) + 1:
        raise ValueError(
            "Length of x_edge must be 1 more than the length of y_mean.")

    x_edge = np.asarray(x_edge, dtype=float)
    y_mean = np.asarray(y_mean, dtype=float)
    dx = np.diff(x_edge)
    interval_integrals = y_mean * dx
    integral_values = np.concatenate(([0], np.cumsum(interval_integrals)))

    if method == 'pchip':
        # Pchip (monotonic C1 for F, C0 for f)
        # Guarantees f(x) >= 0 if all y_mean >= 0
        F_spline = PchipInterpolator(x_edge, integral_values)
    elif method == 'akima':
        # Akima (local C1 for F, C0 for f)
        # Avoids ringing and often looks more natural than PCHIP.
        F_spline = Akima1DInterpolator(x_edge, integral_values)
    elif method == 'cubic':
        # Standard C^2 spline (C1 for f)
        # "Smoother" (f(x) will be C^1), but F(x) is not guaranteed
        # to be monotonic, so f(x) may go < 0 ("ringing").
        F_spline = CubicSpline(x_edge, integral_values, bc_type='not-a-knot')
    else:
        raise ValueError("method must be one of 'pchip', 'akima', or 'cubic'")

    f_spline = F_spline.derivative()

    return f_spline


def compute_chunk_adjacency(chunk_map, reg_axis='both'):
    """
    Computes the adjacency list for a given chunk map.
    
    Parameters
    ----------
    chunk_map : np.ndarray
        2D array where each pixel value is the chunk ID. -1 indicates ignored pixels.
    reg_axis : str, optional
        'both': Horizontal and Vertical neighbors.
        'x': Horizontal only.
        'y': Vertical only.
        
    Returns
    -------
    tuple or None
        (neighbors_i, neighbors_j) arrays of shape (N_pairs,), or None if no pairs found.
    """
    if chunk_map is None:
        return None

    logger.info(f"Pre-computing adjacency matrix (Axis: {reg_axis})...")

    all_i_list = []
    all_j_list = []

    # 1. Horizontal neighbors (x-axis)
    if reg_axis in ['both', 'x', 'horizontal']:
        # Compare [:, :-1] with [:, 1:]
        h_diff = (chunk_map[:, :-1] != -1) & \
                 (chunk_map[:, 1:] != -1) & \
                 (chunk_map[:, :-1] != chunk_map[:, 1:])
                 
        h_idx_i = chunk_map[:, :-1][h_diff]
        h_idx_j = chunk_map[:, 1:][h_diff]
        all_i_list.append(h_idx_i)
        all_j_list.append(h_idx_j)
    
    # 2. Vertical neighbors (y-axis)
    if reg_axis in ['both', 'y', 'vertical']:
        # Compare [:-1, :] with [1:, :]
        v_diff = (chunk_map[:-1, :] != -1) & \
                 (chunk_map[1:, :] != -1) & \
                 (chunk_map[:-1, :] != chunk_map[1:, :])
                 
        v_idx_i = chunk_map[:-1, :][v_diff]
        v_idx_j = chunk_map[1:, :][v_diff]
        all_i_list.append(v_idx_i)
        all_j_list.append(v_idx_j)
    
    # Combine and remove duplicates
    if all_i_list:
        all_i = np.concatenate(all_i_list)
        all_j = np.concatenate(all_j_list)
        
        # Ensure i < j to avoid double counting
        mask = all_i < all_j
        unique_pairs = np.unique(np.stack([all_i[mask], all_j[mask]], axis=1), axis=0)
        return (unique_pairs[:, 0], unique_pairs[:, 1])
    else:
        logger.warning("Warning: No adjacency pairs found.")
        return None
    
def mean_preserving_spline_2d(y_edges, x_edges, means, x_degree=3, y_degree=3):
    """
    Generates a 2D mean-preserving spline surface f(y, x).
    
    Parameters
    ----------
    y_edges : array-like
        The edges of the y bins (length N+1).
    x_edges : array-like
        The edges of the x bins (length M+1).
    means : array-like
        The 2D array of mean offsets in each bin (shape N, M).
        means[i, j] corresponds to interval (y[i]~y[i+1], x[j]~x[j+1]).
    x_degree, y_degree : int
        Degrees of the bivariate spline (3=cubic).
        
    Returns
    -------
    evaluator : function
        A function `func(y, x)` that takes coordinates and returns 
        the interpolated continuous offset values.
    """
    y_edges = np.asarray(y_edges, dtype=float)
    x_edges = np.asarray(x_edges, dtype=float)
    means = np.asarray(means, dtype=float)
    
    single_ybin = (len(y_edges) == 2)
    single_xbin = (len(x_edges) == 2)

    volume = means
    if not single_ybin:
        dy = np.diff(y_edges)[:, None]
        volume = volume * dy
        
    if not single_xbin:
        dx = np.diff(x_edges)[None, :]
        volume = volume * dx

    if not single_ybin:
        # Pad Y axis with 0
        temp = np.zeros((len(y_edges), volume.shape[1]))
        temp[1:, :] = np.cumsum(volume, axis=0)
        volume = temp
    else:
        volume = np.vstack([volume, volume])

    if not single_xbin:
        final_grid = np.zeros((volume.shape[0], len(x_edges)))
        final_grid[:, 1:] = np.cumsum(volume, axis=1)
        integral_surface = final_grid
    else:
        integral_surface = np.hstack([volume, volume])

    kx_fit = 1 if single_ybin else min(y_degree, len(y_edges)-1)
    ky_fit = 1 if single_xbin else min(x_degree, len(x_edges)-1)
    # Note: kx parameter controls Y-axis (axis 0), ky parameter controls X-axis (axis 1)
    F_spline = RectBivariateSpline(y_edges, x_edges, integral_surface, 
                                   kx=kx_fit, ky=ky_fit, s=0)
    
    d_order_y = 0 if single_ybin else 1
    d_order_x = 0 if single_xbin else 1

    def spl(y, x):
        y = np.atleast_1d(y)
        x = np.atleast_1d(x)
        
        return F_spline(y, x, dx=d_order_y, dy=d_order_x, grid=False)
        
    return spl

def get_valid_bounds(mask):
    """Finds the bounding box of valid (False) data in a boolean mask."""
    # Rows where at least one pixel is valid
    valid_rows = np.any(~mask, axis=1)
    # Cols where at least one pixel is valid
    valid_cols = np.any(~mask, axis=0)

    # Find indices
    y_min, y_max = np.where(valid_rows)[0][[0, -1]]
    x_min, x_max = np.where(valid_cols)[0][[0, -1]]

    # Return slices (add +1 to max for python slicing)
    return slice(y_min, y_max + 1), slice(x_min, x_max + 1)

def make_grid_chunk_map(det_shape, n_chunks_per_side):
    """Regular square grid chunk map: ``n_chunks_per_side`` x ``n_chunks_per_side``
    equal cells over ``det_shape`` (row-major chunk ids, 0..n^2-1).

    Generic detector geometry (e.g. broadband imagers such as Euclid NISP). Each
    cell is ``det_h // n`` x ``det_w // n`` px; any remainder rows/cols on the
    high edge fall in the last cell. This is the square-chunk layout used by
    ``notebooks/euclid_mosaic.ipynb``.
    """
    det_h, det_w = det_shape
    chunk_h = det_h // n_chunks_per_side
    chunk_w = det_w // n_chunks_per_side
    y_edges = np.arange(0, det_h + 1, chunk_h)
    x_edges = np.arange(0, det_w + 1, chunk_w)
    chunk_map = np.zeros(det_shape, dtype=int)
    chunk_id = 0
    for j in range(len(y_edges) - 1):
        for i in range(len(x_edges) - 1):
            chunk_map[y_edges[j]:y_edges[j + 1], x_edges[i]:x_edges[i + 1]] = chunk_id
            chunk_id += 1
    return chunk_map


def fill_invalid_offsets(data):
    """
    Fills zeros in a 2D array using linear interpolation for the interior
    and nearest-neighbor for extrapolation at the edges.
    """
    h, w = data.shape
    y, x = np.mgrid[0:h, 0:w]
    
    # 1. Mask the zeros (the "bad" data)
    mask = (data != 0)
    
    # If the whole thing is zeros or there are no zeros, return as is
    if not np.any(mask) or np.all(mask):
        return data

    # 2. Extract valid points
    points = np.array((y[mask], x[mask])).T
    values = data[mask]
    
    # 3. Interpolate the entire grid
    # 'linear' handles the interior bilinear logic
    # We use 'nearest' for the points griddata can't reach (extrapolation)
    # If points are collinear (e.g. valid data in only one column), Delaunay triangulation fails.
    # In that case, we catch the Qhull precision error and fallback to 'nearest' immediately.
    from scipy.spatial.qhull import QhullError
    try:
        filled = griddata(points, values, (y, x), method='linear')
    except QhullError:
        filled = griddata(points, values, (y, x), method='nearest')
    
    # 4. Fill remaining NaNs (edges/corners) with nearest neighbor extrapolation
    nan_mask = np.isnan(filled)
    if np.any(nan_mask):
        filled[nan_mask] = griddata(points, values, (y[nan_mask], x[nan_mask]), method='nearest')
        
    return filled
