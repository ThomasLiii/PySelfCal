import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
import scipy.ndimage as nd
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from skimage import measure
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import least_squares
from SelfCal.MapHelper import arc_spline, linear_spline, mean_preserving_spline, bit_to_bool
from SelfCal.MakeMap import load_reproj_file


def load_calibration(band, calibration_dir='/home/thomasli/spherex/spherex_calibration'):
    BC_files = glob.glob(os.path.join(calibration_dir, f'*BC_Band{band}.fits'))
    BW_files = glob.glob(os.path.join(calibration_dir, f'*BW_Band{band}.fits'))
    if len(BC_files) != 1 or len(BW_files) != 1:
        raise ValueError(f"Expected one BC and one BW file for band {band}, found {len(BC_files)} BC files and {len(BW_files)} BW files.")
    BC_map = fits.getdata(BC_files[0])
    BW_map = fits.getdata(BW_files[0])
    return BC_map, BW_map

def extract_spherex_channel_edges(band, channel_file='/home/thomasli/spherex/spherex_channels.csv'):
    tbl = Table.read(channel_file)
    sub_tbl = tbl[tbl['band'] == band]
    channel_edges = np.hstack([sub_tbl['lmin'].data, sub_tbl['lmax'].data[-1:]])
    return channel_edges

def interpolate_array(data_arr, interp_factor=5):
    interp_arr = np.hstack([
        np.linspace(data_arr[i], data_arr[i + 1], interp_factor, endpoint=False) 
        for i in range(len(data_arr) - 1)
    ] + [data_arr[-1]])  # Append the last element
    return interp_arr

def extract_edge_samples(BC_map, channel_edges):
    edge_x_list = []
    edge_y_list = []
    for i, lam in tqdm(enumerate(channel_edges), total=len(channel_edges)):
        edge_y = np.argmin(np.abs(BC_map - lam), axis=0).astype(np.float32)
        edge_x = np.arange(len(edge_y)).astype(np.float32)

        if i == len(channel_edges)-1:
            edge_mask = (edge_x > 650) & (edge_x < BC_map.shape[0]-650)
            edge_y[edge_mask] = np.nan
            edge_x[edge_mask] = np.nan
        elif i == 0:
            edge_mask = (edge_x < 50) & (edge_x > BC_map.shape[0]-50)
            edge_y[edge_mask] = np.nan
            edge_x[edge_mask] = np.nan

        edge_x_list.append(edge_x)
        edge_y_list.append(edge_y)

    return np.array(edge_x_list), np.array(edge_y_list)
    
def fit_lvf_arcs(edge_x_list, edge_y_list):
    assert edge_x_list.shape == edge_y_list.shape, "x and y must be the same shape."

    def _arc_residuals(params, edge_x_list, edge_y_list):
        xc, yc = params[0], params[1]
        R_list = params[2:]
        distances = np.sqrt((edge_x_list - xc)**2 + (edge_y_list - yc)**2)
        R_list_expanded = R_list[:, np.newaxis]
        errors = distances - R_list_expanded
        return np.nan_to_num(errors.ravel())

    arc_x_means = np.nanmean(edge_x_list, axis=1)
    arc_y_means = np.nanmean(edge_y_list, axis=1)
    xc_guess = 1020
    yc_guess = 9632.4376
    R_guess_list = np.sqrt((arc_x_means - xc_guess)**2 + (arc_y_means - yc_guess)**2)
    initial_params = np.concatenate(([xc_guess, yc_guess], R_guess_list))

    result = least_squares(
        _arc_residuals,
        initial_params,
        args=(edge_x_list, edge_y_list),
        method='lm'
    )

    if not result.success:
        # result.status is more informative than just result.message
        raise RuntimeError(f"Arc fitting optimization failed: {result.status} ({result.message})")

    xc_fit, yc_fit = result.x[0], result.x[1]
    R_fit = result.x[2:]
    lvf_params = {'xc': xc_fit, 'yc': yc_fit, 'R': R_fit}
    return lvf_params

def make_arc_spline(xc, yc, R):
    def arc_spline(x):
        return -np.sqrt(R**2 - (x - xc)**2) + yc
    return arc_spline

def fit_lvf_params(BC_map, channel_edges):
    edge_x_list, edge_y_list = extract_edge_samples(BC_map, channel_edges)
    lvf_params = fit_lvf_arcs(edge_x_list, edge_y_list)
    lvf_params['wave_edges'] = channel_edges
    return lvf_params

def make_spherex_chunk_map(BC_map, channel_edges, oversample_factor=1, lvf_params=None):
    out_shape = (BC_map.shape[0]*oversample_factor, BC_map.shape[1]*oversample_factor)
    chunk_map = np.zeros(out_shape, dtype=np.int32)
    x_mesh, y_mesh = np.meshgrid(np.arange(out_shape[1]), np.arange(out_shape[0]))
    if lvf_params is None:
        print("Fitting LVF parameters...")
        lvf_params = fit_lvf_params(BC_map, channel_edges)

    print("Making chunk map...")
    y_bound = np.full(out_shape[1], out_shape[0]-1)
    for i, lam in tqdm(enumerate(channel_edges), total=len(channel_edges)):
        prev_y_bound = y_bound

        xc = lvf_params['xc']
        yc = lvf_params['yc']
        if lam not in lvf_params['wave_edges']:
            R = np.interp(lam, lvf_params['wave_edges'], lvf_params['R'])
        else:
            R = lvf_params['R'][np.where(lvf_params['wave_edges'] == lam)[0][0]]
        spl = make_arc_spline(xc, yc, R)

        x_bound = np.arange(out_shape[1])
        y_bound = spl(x_bound/oversample_factor) * oversample_factor
        y_bound = np.clip(y_bound, 0, out_shape[1])
        chunk_map[(y_mesh >= y_bound) & (y_mesh < prev_y_bound)] = i
    else:
        prev_y_bound = y_bound
        y_bound = np.zeros_like(y_bound)
        chunk_map[(y_mesh >= y_bound) & (y_mesh < prev_y_bound)] = i+1
    return chunk_map, lvf_params

def make_fiducial_chunk_map(band, BC_map, num_channels=17, num_subchannels=10, channel_file='/home/thomasli/spherex/spherex_channels.csv', 
                            oversample_factor=1, lvf_params=None):
    if num_channels%17 != 0:
        raise ValueError("num_channels must be a multiple of 17.")
    interp_factor = num_subchannels * num_channels//17
    channel_edges = extract_spherex_channel_edges(band, channel_file=channel_file)
    fine_edges = interpolate_array(channel_edges, interp_factor=interp_factor)
    chunk_map, lvf_params = make_spherex_chunk_map(BC_map, fine_edges, oversample_factor=oversample_factor, lvf_params=lvf_params)
    return chunk_map, lvf_params

def make_fiducial_chunk_mask(valid_channels, num_channels=17, num_subchannels=10, padding=0):
    chunk_valid_mask = np.zeros(num_channels*num_subchannels + 2)
    valid_subchannels = np.hstack(((np.array(valid_channels)-1)*num_subchannels)[:, None] + \
                                  np.arange(0-padding,num_subchannels+padding)) + 1
    chunk_valid_mask[valid_subchannels] = 1
    return chunk_valid_mask

def visualize_chunk_map(chunk_map, chunk_valid_mask):
    masked_chunk_map = np.where(chunk_valid_mask[chunk_map], chunk_map, np.nan)
    plt.imshow(masked_chunk_map, cmap='viridis', interpolation='none')

# https://github.com/jararias/mpsplines
from mpsplines import MeanPreservingInterpolation as MPI
def interp_1d(arr, method='mp', edge='extend'):
    idx = np.arange(len(arr))
    mean_idx, mean_val, edge_idx = parse_bin(arr)
    if method == 'mp_external':
        interpolator = MPI(yi=mean_val, xi=mean_idx)
    elif method == 'mp':
        interpolator = mean_preserving_spline(edge_idx, mean_val, method='cubic')
    elif method == 'linear':
        interpolator = linear_spline(mean_idx, mean_val)
    smooth_arr = interpolator(idx)
    return smooth_arr

def interp_2d_vertical(arr, method='mp'):
    return np.apply_along_axis(interp_1d, axis=0, arr=arr, method=method)

def parse_bin(arr):
    start = np.where(arr[:-1] != arr[1:])[0]+1
    edge = start - 1/2
    mean_idx = (start[:-1] + (start[1:] - 1))/2
    mean_val = arr[start[:-1]]
    return mean_idx, mean_val, edge

def make_spherex_offset_map(chunk_map, chunk_offset, chunk_valid_mask, lvf_params):
    R = lvf_params['R']
    xc, yc = lvf_params['xc'], lvf_params['yc']

    edge_valid_mask = chunk_valid_mask[1:].astype(bool) | chunk_valid_mask[:-1].astype(bool)
    valid_R = R[edge_valid_mask]
    spl = mean_preserving_spline(x_edge=valid_R, y_mean=chunk_offset[chunk_valid_mask.astype(bool)])

    h, w = np.shape(chunk_map)
    oversample_factor = h // 2040
    
    x_vec = (np.arange(w) / oversample_factor) - xc
    y_vec = (np.arange(h) / oversample_factor) - yc
    r_mesh = np.sqrt(x_vec**2 + y_vec[:, None]**2)
    
    offset_map = spl(r_mesh)
    return offset_map

def compute_mean_pixval(exp_file, chunk_map):
    with fits.open(exp_file) as hdul:
        data = hdul[1].data
        bitmask = hdul[2].data

    valid_mask = bit_to_bool(bitmask, ignore_list=[7], invert=True)
    max_chunk_id = np.max(chunk_map)
    index_range = np.arange(max_chunk_id + 1)

    labels = chunk_map.copy()
    labels[~valid_mask] = -1
    mean = nd.mean(data, labels=labels, index=index_range)
    if np.isnan(mean).any():
        print("Warning: NaN values found in mean pixel value computation, filling with 0.")
        mean = np.nan_to_num(mean, nan=0.0)
    return mean


def compute_offsets_guess(reproj_list, det_chunk_map):
    def _extract_file_path(reproj_file):
        reproj_data = load_reproj_file(reproj_file, fields=['file_path'])
        return reproj_data['file_path']
    exp_files = [_extract_file_path(reproj_file) for reproj_file in reproj_list]
    compute_mean_pixval_partial = partial(compute_mean_pixval, chunk_map=det_chunk_map)

    with ProcessPoolExecutor(max_workers=20) as executor:
        results = list(tqdm(executor.map(compute_mean_pixval_partial, exp_files), 
                            total=len(exp_files), 
                            desc="Calculating initial guess offsets"))
    offset_guess = np.array(results)
    return offset_guess


def load_lvf_params(filename, input_dir='/home/thomasli/spherex/selfcal/selfcal_scripts/lvf_params'):
    input_path = os.path.join(input_dir, filename)
    if not os.path.exists(input_path):
        print(f"LVF parameters file {input_path} not found. Returning None.")
        return None
    lvf_params = np.load(input_path, allow_pickle=True).item()
    print(f"Loaded LVF parameters from {input_path}")
    return lvf_params

def save_lvf_params(lvf_params, output_dir='/home/thomasli/spherex/selfcal/selfcal_scripts/lvf_params'):
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, lvf_params['filename'])
    np.save(output_path, lvf_params)
    print(f"Saved LVF parameters to {output_path}")

def compute_vertical_strip_adjacency(chunk_map, num_vertical_bands):
    """
    Generates adjacency pairs ONLY for vertical strip transitions, 
    ignoring spectral arc transitions.
    
    Parameters
    ----------
    chunk_map : np.ndarray
        The full ID map (Subchannel * N + Band)
    num_vertical_bands : int
        The NUM_VERTICAL_BANDS constant used to build the map.
    """
    print("Computing Vertical Strip Adjacency (Filtering Arcs)...")
    
    # 1. Get ALL horizontal transitions (Arc + Strip boundaries)
    # Compare pixel i with i+1
    mask = (chunk_map[:, :-1] != -1) & \
           (chunk_map[:, 1:] != -1) & \
           (chunk_map[:, :-1] != chunk_map[:, 1:])
           
    u = chunk_map[:, :-1][mask]
    v = chunk_map[:, 1:][mask]
    
    # 2. Decompose IDs back into (Subchannel, Band)
    # Formula: ID = Sub * N + Band
    sub_u = u // num_vertical_bands
    sub_v = v // num_vertical_bands
    
    # 3. FILTER: Only keep pairs that are in the SAME Subchannel
    # This rejects the boundaries where the arc changes.
    valid_pair_mask = (sub_u == sub_v)
    
    u_filtered = u[valid_pair_mask]
    v_filtered = v[valid_pair_mask]
    
    # 4. Remove duplicates
    # Sort pairs so (u,v) is same as (v,u) for unique checking
    pairs = np.sort(np.stack([u_filtered, v_filtered], axis=1), axis=1)
    unique_pairs = np.unique(pairs, axis=0)
    
    print(f"Found {len(unique_pairs)} vertical strip boundaries.")
    return unique_pairs[:, 0], unique_pairs[:, 1]

def make_stripped_chunk_map(detector, num_subchannels=10, num_channels=17, 
                            oversample_factor=1, num_vertical_bands=1, lvf_params=None, calibration_dir='/home/thomasli/spherex/SPHEREx_Spectral_Calibration'):
    det_BC, det_BW = load_calibration(band=detector, calibration_dir=calibration_dir)
    def make_vertical_band_maps(sub_channel_map, num_vertical_bands):
        vertchunk_map = np.zeros_like(sub_channel_map)
        for band in range(num_vertical_bands):
            width = vertchunk_map.shape[1] // num_vertical_bands
            vertchunk_map[:, band*width:(band+1)*width] = band
        return vertchunk_map
    subchannel_map, lvf_params = make_fiducial_chunk_map(detector, det_BC, num_subchannels=num_subchannels, num_channels=num_channels, oversample_factor=oversample_factor, lvf_params=lvf_params)
    verticalchunk_map = make_vertical_band_maps(subchannel_map, num_vertical_bands)
    chunk_map = subchannel_map * num_vertical_bands + verticalchunk_map
    return chunk_map, lvf_params

def make_stripped_chunk_valid_mask(ch, num_subchannels=10, num_channels=17, 
                                   num_vertical_bands=1, subchannel_padding=0):
    def make_chunk_valid_mask(subchannel_valid_mask, num_vertical_bands):
        chunk_valid_mask = np.zeros(len(subchannel_valid_mask)*num_vertical_bands, dtype=subchannel_valid_mask.dtype)
        for band in range(num_vertical_bands):
            chunk_valid_mask[band::num_vertical_bands] = subchannel_valid_mask
        return chunk_valid_mask
    subchannel_valid_mask = make_fiducial_chunk_mask(ch, num_subchannels=num_subchannels, num_channels=num_channels, padding=subchannel_padding)
    chunk_valid_mask = make_chunk_valid_mask(subchannel_valid_mask, num_vertical_bands=num_vertical_bands)
    return chunk_valid_mask