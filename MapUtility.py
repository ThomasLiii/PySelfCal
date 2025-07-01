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
    h, w = arr.shape
    assert h % bin_factor == 0 and w % bin_factor == 0, "h and w must be divisible by bin_factor"

    h_bins = h // bin_factor
    w_bins = w // bin_factor

    # Reshape and average
    reshaped = arr.reshape(h_bins, bin_factor, w_bins, bin_factor)
    binned = bin_func(reshaped, axis=(1, 3))

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

def det_to_sub(grid_mapping=None, n_chunk=10, sub_width = 157, det_size = 2040, det_off=None):
    det_grid_x, det_grid_y = grid_mapping
    oversample_factor = int(np.shape(det_grid_x)[-1]/sub_width)
    
    det_flat_x, det_flat_y = (det_grid_x.flatten(), det_grid_y.flatten())
    
    chunk_size = int(det_size/n_chunk)
    chunk_edges_x, chunk_edges_y = compute_chunk_edges((det_size,det_size), (chunk_size, chunk_size))

    chunk_idx_x = np.searchsorted(chunk_edges_x, det_flat_x)
    chunk_mask_x = np.any([chunk_idx_x==len(chunk_edges_x), chunk_idx_x==0], axis=0)
    chunk_idx_y = np.searchsorted(chunk_edges_y, det_flat_y)
    chunk_mask_y = np.any([chunk_idx_y==len(chunk_edges_y), chunk_idx_y==0], axis=0)
    valid_mask = ~np.any([chunk_mask_x, chunk_mask_y], axis=0)
    
    chunk_idx = ((chunk_idx_y-1)*(len(chunk_edges_x)-1) + (chunk_idx_x-1))

    valid_idx = np.where(valid_mask)[0]
    rows = chunk_idx[valid_mask]
    cols = valid_idx
    data = np.ones_like(rows, dtype=np.uint8)
    chunk_contrib = coo_matrix((data, (rows, cols)), shape=(n_chunk**2, (sub_width*oversample_factor)**2))
    chunk_contrib_bin = bin2d_coo_matrix(chunk_contrib, sub_width*oversample_factor, sub_width*oversample_factor, oversample_factor)

    if det_off is not None:
        # Map arbitrary offset_map (shape: n_chunk x n_chunk) to sub-frame
        flat_offset = det_off.flatten()
        sub_frame_offset = chunk_contrib_bin.T @ flat_offset  # shape: (sub_width**2,)
        return sub_frame_offset.reshape(sub_width, sub_width)
    
    return chunk_contrib_bin

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


def compute_chunk_contrib(grid_mapping, chunk_map, oversample_factor):
    num_chunks = len(np.unique(chunk_map))
    sub_width = grid_mapping.shape[-1] // oversample_factor
    valid_pix = np.all(~np.isnan(grid_mapping), axis=0)

    # Extract valid pixel indices directly
    flat_pix_idx0 = np.rint(grid_mapping[0][valid_pix]).astype(np.int32)
    flat_pix_idx1 = np.rint(grid_mapping[1][valid_pix]).astype(np.int32)

    # Map chunk indices and construct sparse matrix
    chunk_idx_flat = chunk_map[flat_pix_idx1, flat_pix_idx0]
    cols = np.flatnonzero(valid_pix)
    data = np.ones_like(cols, dtype=np.float32)
    map_width = sub_width * oversample_factor

    chunk_idx_parsed = coo_matrix((data, (chunk_idx_flat, cols)), shape=(num_chunks, map_width * map_width))
    chunk_contrib = bin2d_coo_matrix(chunk_idx_parsed, map_width, map_width, oversample_factor)
    return chunk_contrib


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