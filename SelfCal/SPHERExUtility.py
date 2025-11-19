import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from skimage import measure
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import least_squares
from SelfCal.MapHelper import arc_spline, linear_spline, mean_preserving_spline


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

def interpolate_array(data_arr, interp_factor=5):
    interp_arr = np.hstack([
        np.linspace(data_arr[i], data_arr[i + 1], interp_factor, endpoint=False) 
        for i in range(len(data_arr) - 1)
    ] + [data_arr[-1]])  # Append the last element
    return interp_arr

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

def make_spherex_chunk_map(BC_map, channel_edges, oversample_factor=1):
    out_shape = (BC_map.shape[0]*oversample_factor, BC_map.shape[1]*oversample_factor)
    chunk_map = np.zeros(out_shape, dtype=np.int16)
    y_bound = np.full(out_shape[1], out_shape[0]-1)
    x_mesh, y_mesh = np.meshgrid(np.arange(out_shape[1]), np.arange(out_shape[0]))

    print("Fitting LVF contours...")
    edge_x_list, edge_y_list = extract_edge_samples(BC_map, channel_edges)
    lvf_params = fit_lvf_arcs(edge_x_list, edge_y_list)
    lvf_params['wave_edges'] = channel_edges

    print("Making chunk map...")
    for i, lam in tqdm(enumerate(channel_edges), total=len(channel_edges)):
        prev_y_bound = y_bound

        spl = make_arc_spline(lvf_params['xc'], lvf_params['yc'], lvf_params['R'][i])
        x_bound = np.arange(out_shape[1])
        y_bound = spl(x_bound)
        y_bound = np.clip(y_bound, 0, out_shape[1])
        chunk_map[(y_mesh >= y_bound) & (y_mesh < prev_y_bound)] = i
    else:
        prev_y_bound = y_bound
        y_bound = np.zeros_like(y_bound)
        chunk_map[(y_mesh >= y_bound) & (y_mesh < prev_y_bound)] = i+1
    return chunk_map, lvf_params

def make_fiducial_chunk_map(band, BC_map, num_channels=17, num_subchannels=10, channel_file='/home/thomasli/spherex/spherex_channels.csv', 
                            oversample_factor=1):
    if num_channels%17 != 0:
        raise ValueError("num_channels must be a multiple of 17.")
    interp_factor = num_subchannels * num_channels//17
    channel_edges = extract_spherex_channel_edges(band, channel_file=channel_file)
    fine_edges = interpolate_array(channel_edges, interp_factor=interp_factor)
    chunk_map, lvf_params = make_spherex_chunk_map(BC_map, fine_edges, oversample_factor=oversample_factor)
    return chunk_map, lvf_params

def make_fiducial_chunk_mask(valid_channels, num_channels=17, num_subchannels=10):
    chunk_valid_mask = np.zeros(num_channels*num_subchannels + 2)
    chunk_valid_mask[np.hstack(((np.array(valid_channels)-1)*num_subchannels)[:, None] + np.arange(num_subchannels)) + 1] = 1
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