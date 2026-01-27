from xml.parsers.expat import errors
from tqdm import tqdm
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix, csr_matrix
import cv2

from mpsplines import MeanPreservingInterpolation as MPI
from scipy.interpolate import PchipInterpolator, CubicSpline, Akima1DInterpolator
from scipy.optimize import minimize
from scipy.ndimage import map_coordinates

def bit_to_bool(bitmask_array, ignore_list=[], bitmask_header=None, invert=False, expand_bits=False):
    # By default, 1 indicates bad pixels and 0 indicates good pixels.
    # If invert=True, this is flipped.
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

def make_weight(frame, sigma=1.4):
    '''Make weight used for weighted mean'''
    # inverse = 1/frame
    # inverse[check_invalid(inverse)] = 0
    # weight = gaussian_filter(inverse, sigma=sigma)
    filled_frame = np.nan_to_num(frame, nan=np.nanmedian(frame))
    convolved_frame = gaussian_filter(filled_frame, sigma=sigma) - frame
    weight = 1.0 / (np.abs(frame))# + np.abs(convolved_frame * 4) + 0.1) ** 2
    # abs_frame = np.abs(filled_frame)
    # filling_value = np.nanpercentile(abs_frame[abs_frame>0], 1)
    # abs_frame[abs_frame < filling_value] = filling_value
    # weight = 1.0 / (abs_frame) ** 2
    return np.nan_to_num(weight, nan=0)

def find_outliers(data, threshold=3):
    '''Return 1 where outlier is detected, else return 0'''
    median = np.nanmedian(data)
    nmad = 1.4826 * np.nanmedian(np.abs(data - median))
    z_score = (data - median)/nmad
    return np.abs(z_score) > threshold

def map_pixels(wcs_in, wcs_out, x_in, y_in):
    ra, dec = wcs_in.pixel_to_world_values(x_in, y_in)
    x_out, y_out = wcs_out.world_to_pixel_values(ra, dec)
    return x_out, y_out

def compute_chunk_edges(det_shape, chunk_size):
    '''Split detector into chunks and return edge of chunks'''
    det_h, det_w = det_shape
    chunk_h, chunk_w = chunk_size
    y_edges = np.arange(0, det_h + 1, chunk_h)
    x_edges = np.arange(0, det_w + 1, chunk_w)
    
    return x_edges, y_edges

# def bin2d_last_axes(arr, bin_factor):
#     d, h, w = arr.shape
#     assert h % bin_factor == 0 and w % bin_factor == 0, "h and w must be divisible by bin_factor"

#     h_bins = h // bin_factor
#     w_bins = w // bin_factor

#     # Reshape and average
#     reshaped = arr.reshape(d, h_bins, bin_factor, w_bins, bin_factor)
#     binned = reshaped.mean(axis=(2, 4))

#     return binned

def bin2d(arr, bin_factor, bin_func=np.mean):
    """
    Bins a 2D array or a stack of 2D arrays (3D) by a given factor.
    """
    if arr.ndim == 2:
        h, w = arr.shape
        assert h % bin_factor == 0 and w % bin_factor == 0, "h and w must be divisible by bin_factor"
        h_bins = h // bin_factor
        w_bins = w // bin_factor
        
        # Reshape and apply function for a single 2D array
        reshaped = arr.reshape(h_bins, bin_factor, w_bins, bin_factor)
        binned = bin_func(reshaped, axis=(1, 3))
        
    elif arr.ndim == 3:
        num_layers, h, w = arr.shape
        assert h % bin_factor == 0 and w % bin_factor == 0, "h and w must be divisible by bin_factor"
        h_bins = h // bin_factor
        w_bins = w // bin_factor
        
        # Reshape and apply function for a stack of 2D arrays
        reshaped = arr.reshape(num_layers, h_bins, bin_factor, w_bins, bin_factor)
        binned = bin_func(reshaped, axis=(2, 4)) # Bin along the new height and width factor axes
        
    else:
        raise ValueError(f"bin2d supports 2D or 3D arrays, but got {arr.ndim} dimensions.")
        
    return binned

def bin2d_cv(arr, bin_factor):
    """
    Bins a 2D array using cv2.resize.
    This is only appropriate for mean-binning.
    """
    if arr.ndim != 2:
        raise ValueError("OpenCV resize only suitable for 2D arrays in this context.")
        
    h, w = arr.shape
    if not (h % bin_factor == 0 and w % bin_factor == 0):
        # cv2.resize can handle this, but it's not a true 'binning'
        # if the dimensions are not multiples.
        print("Warning: Dimensions not divisible by bin_factor. Result is a resize, not a clean bin.")

    h_new, w_new = h // bin_factor, w // bin_factor
    
    # INTER_AREA is the key for averaging-like downsampling
    binned = cv2.resize(arr.astype(np.float32), (w_new, h_new), interpolation=cv2.INTER_AREA)
    
    return binned

def bin2d_coo_matrix(mat, height, width, bin_factor):
    """
    Fast binning of a COO sparse matrix shaped (N, height * width), where the second axis is a flattened 2D grid.
    Performs 2D average pooling with bin_factor.
    """
    assert isinstance(mat, coo_matrix), "Input must be COO format"
    assert height % bin_factor == 0 and width % bin_factor == 0

    row, flat_col, data = mat.row, mat.col, mat.data

    # Vectorized 2D binning map (fast index transform)
    new_width = width // bin_factor
    i_bin = (flat_col // width) // bin_factor
    j_bin = (flat_col % width) // bin_factor
    new_col = i_bin * new_width + j_bin

    # Compute binned matrix with summed data
    binned = coo_matrix((data, (row, new_col)), shape=(mat.shape[0], (height // bin_factor) * (width // bin_factor)))

    # Normalize to get average
    binned.data /= (bin_factor * bin_factor)
    # binned.sum_duplicates()
    return binned

def make_footprint(sub_data, ref_coords, ref_shape, exp_offset=None):
    '''Compute footprint, weighted by exp_offset if given'''
    N = len(sub_data)
    footprint = np.zeros(ref_shape)

    for i in range(N):
        (y_min, y_max, x_min, x_max) = ref_coords[i]
        p = ~check_invalid(sub_data[i])
        if exp_offset is not None:
            p = p.astype(np.float32) * exp_offset[i]
        footprint[y_min:y_max, x_min:x_max] += p
    
    return footprint

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

def chunk_to_det(chunk_map, chunk_data):
    det_offset = chunk_data[chunk_map]
    return det_offset

def make_linear_interp_matrix(coords, input_shape):
    """
    Optimized generation of sparse interpolation matrix.
    """
    # Coords = (y_coords, x_coords)
    H, W = input_shape
    N_total = coords.shape[1] 

    # 1. Identify valid inputs (removing NaNs)
    # np.isfinite is generally slightly faster than ~np.isnan
    valid_mask = np.isfinite(coords[0]) & np.isfinite(coords[1])
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

    # 3. Optimized Bounds Check (The bottleneck fix)
    # Instead of creating 'all_r' (size 4N), check bounds on 'r0' (size N)
    # Pre-calculate boolean masks for rows/cols being inside image
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

def det_to_sub(det_data, sub_mapping=None, interp_matrix=None):
    if interp_matrix is not None:
        sub_width = np.sqrt(interp_matrix.shape[0]).astype(np.int32)
        det_data_flat = det_data.ravel()
        sub_data_flat = interp_matrix @ det_data_flat
        sub_data = sub_data_flat.reshape(sub_width, sub_width)
    elif sub_mapping is not None:
        sub_data = map_coordinates(det_data, sub_mapping[::-1], order=1, output=np.float32)
    else:
        raise ValueError("Either sub_mapping or interp_matrix must be provided.")
    return sub_data

def compute_chunk_contrib(chunk_map, interp_matrix=None):
    """Computes the sparse matrix contribution for LSQR."""
    chunk_map_flat = chunk_map.ravel()
    total_rows = chunk_map_flat.size
    total_cols = chunk_map_flat.max() + 1

    indptr = np.arange(total_rows + 1)
    indices = chunk_map_flat
    data = np.ones(total_rows, dtype=np.float32)

    chunk_map_parsed = csr_matrix((data, indices, indptr), shape=(total_rows, total_cols))
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
    assert len(x_edge) == len(y_mean) + 1, \
        "Length of x_edge must be 1 more than the length of y_mean."

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


def arc_spline(x_sample, y_sample, return_params=False):
    assert len(x_sample) == len(y_sample), "x and y must be the same length."

    def _arc_cost(params, x, y):
        xc, yc, R = params
        distances = np.sqrt((x - xc)**2 + (y - yc)**2)
        errors = distances - R
        return np.sum(errors**2)

    yc_guess = 10000#np.mean(y)
    xc_guess = np.median(x_sample)
    R_guess = np.mean(np.sqrt((x_sample - xc_guess)**2 + (y_sample - yc_guess)**2))

    initial_guess = [xc_guess, yc_guess, R_guess]

    # Run the optimization (non-linear least squares)
    result = minimize(
        _arc_cost,
        initial_guess,
        args=(x_sample, y_sample),
        method='Nelder-Mead' # A robust method for this type of problem
    )

    if not result.success:
        raise RuntimeError(f"Arc fitting optimization failed: {result.message}")

    # Extract the fitted parameters
    xc_fit, yc_fit, R_fit = result.x

    spl = lambda x: -np.sqrt(R_fit**2 - (x - xc_fit)**2) + yc_fit
    if return_params:
        return spl, (xc_fit, yc_fit, R_fit)
    return spl

def upscale2d(array, upscale_factor):
    # Ensure input is a numpy array
    array = np.array(array, dtype=float)
    h, w = array.shape
    
    # Calculate new dimensions
    new_h, new_w = h * upscale_factor, w * upscale_factor
    
    # generate grid coordinates for the new array
    # We want to map indices 0 to new_h-1 back to 0 to h-1
    # aligning the centers of the corner pixels
    r_idx = np.linspace(0, h - 1, new_h)
    c_idx = np.linspace(0, w - 1, new_w)
    
    # Get vertical and horizontal indices for the grid
    # r and c are float indices in the original array space
    r, c = np.meshgrid(r_idx, c_idx, indexing='ij')
    
    # Get the integer parts (top-left neighbor)
    r0 = np.floor(r).astype(int)
    c0 = np.floor(c).astype(int)
    
    # Get the neighbor to the right/bottom, clamping to edge
    r1 = np.minimum(r0 + 1, h - 1)
    c1 = np.minimum(c0 + 1, w - 1)
    
    # Calculate weights (fractional parts)
    dr = r - r0
    dc = c - c0
    
    # Get the values of the four neighbors
    # Ia: Top-left, Ib: Top-right
    # Ic: Bottom-left, Id: Bottom-right
    Ia = array[r0, c0]
    Ib = array[r0, c1]
    Ic = array[r1, c0]
    Id = array[r1, c1]
    
    # Perform bilinear interpolation
    # Interpolate top row (wa) and bottom row (wb)
    wa = (1 - dc) * Ia + dc * Ib
    wb = (1 - dc) * Ic + dc * Id
    
    # Interpolate vertically between top and bottom
    result = (1 - dr) * wa + dr * wb
    
    return result