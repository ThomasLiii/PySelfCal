from tqdm import tqdm
import numpy as np

from scipy.ndimage import gaussian_filter
from scipy.sparse import coo_matrix


def bit_to_bool(bitmask_array, ignore_list, bitmask_header=None, invert=False):
    # bitmask_array = bitmask_array ^ np.uint32(1 << bitmask_header['MP_PERSIST'])
    ignore_mask_val = np.uint32(0)
    for item in ignore_list:
        bit = bitmask_header[item] if bitmask_header is not None else item
        ignore_mask_val |= np.uint32(1 << bit)
    relevant_mask = np.invert(ignore_mask_val)
    mask = (bitmask_array & relevant_mask) != 0
    return ~mask if invert else mask

def make_weight(frame, sigma=1.4):
    '''Make weight used for weighted mean'''
    # inverse = 1/frame
    # inverse[np.isnan(inverse)] = 0
    # weight = gaussian_filter(inverse, sigma=sigma)
    filled_frame = np.nan_to_num(frame, nan=np.nanmedian(frame))
    convolved_frame = gaussian_filter(filled_frame, sigma=sigma) - frame
    weight = 1.0 / (np.abs(frame) + np.abs(convolved_frame * 4) + 0.1) ** 2
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
        p = ~np.isnan(sub_data[i])
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

def interp_1d(arr, method='mp', edge='extend'):
    idx = np.arange(len(arr))
    start = np.where(arr[:-1] != arr[1:])[0]+1
    mean_idx = (start[:-1] + (start[1:] - 1))/2
    mean_val = arr[start[:-1]]
    if method == 'mp':
        from mpsplines import MeanPreservingInterpolation as MPI
        # https://github.com/jararias/mpsplines
        mpi = MPI(yi=mean_val, xi=mean_idx)
        smooth_arr = mpi(idx)
    elif method == 'linear':
        smooth_arr = np.interp(idx, mean_idx, mean_val)
        
    return smooth_arr

def interp_2d_vertical(arr, method='mp'):
    return np.apply_along_axis(interp_1d, axis=0, arr=arr, method=method)

def compute_offset_map(chunk_offset, chunk_map, interp_method=''):
    offset_map = chunk_offset[chunk_map]
    if interp_method != '':
        offset_map = interp_2d_vertical(offset_map, method=interp_method)
    return offset_map

def parse_grid_mapping(grid_mapping):
    """
    Parses a grid mapping to find valid pixels and their original detector coordinates.
    This is the core reusable logic.
    """
    valid_pix = np.all(~np.isnan(grid_mapping), axis=0)
    grid_x_flat = np.rint(grid_mapping[0][valid_pix]).astype(np.int32)
    grid_y_flat = np.rint(grid_mapping[1][valid_pix]).astype(np.int32)
    return valid_pix, grid_x_flat, grid_y_flat

def det_to_grid(grid_mapping, det_data):
    """
    Maps detector data onto the grid using the grid_mapping.
    This function can handle a single 2D array or a stack of 2D arrays (3D).
    """
    valid_pix, grid_x_flat, grid_y_flat = parse_grid_mapping(grid_mapping)

    if det_data.ndim == 2:
        # Handle a single 2D array
        grid_data = np.zeros_like(grid_mapping[0])
        grid_data[valid_pix] = det_data[grid_y_flat, grid_x_flat]
        return grid_data
    elif det_data.ndim == 3:
        # Handle a stack of 2D arrays (3D)
        num_layers = det_data.shape[0]
        grid_shape = (num_layers, grid_mapping.shape[1], grid_mapping.shape[2])
        grid_data = np.zeros(grid_shape, dtype=det_data.dtype)
        # Use advanced indexing to map all layers at once
        grid_data[:, valid_pix] = det_data[:, grid_y_flat, grid_x_flat]
        return grid_data
    else:
        raise ValueError(f"det_data must be 2D or 3D, but got {det_data.ndim} dimensions.")

def compute_chunk_contrib(grid_mapping, chunk_map, oversample_factor):
    """Computes the sparse matrix contribution for LSQR."""
    num_chunks = len(np.unique(chunk_map))
    map_width = grid_mapping.shape[-1]
    
    valid_pix, grid_x_flat, grid_y_flat = parse_grid_mapping(grid_mapping)
    
    chunk_idx_flat = chunk_map[grid_y_flat, grid_x_flat]
    cols = np.flatnonzero(valid_pix)
    data = np.ones_like(cols, dtype=np.float32)

    chunk_idx_parsed = coo_matrix(
        (data, (chunk_idx_flat, cols)), 
        shape=(num_chunks, map_width * map_width)
    )
    chunk_contrib = bin2d_coo_matrix(chunk_idx_parsed, map_width, map_width, oversample_factor)
    return chunk_contrib
