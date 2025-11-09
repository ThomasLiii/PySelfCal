import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from skimage import measure
from scipy.interpolate import make_smoothing_spline


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

def make_spherex_chunk_map(BC_map, channel_edges):

    channel_idx = np.searchsorted(channel_edges, BC_map, side='right')

    mask_smooth = np.zeros_like(channel_idx)

    for i in tqdm(np.unique(channel_idx)[:]):
        mask = channel_idx >= i
        contours = measure.find_contours(mask, level=0.1, positive_orientation='low')
        if len(contours) == 0:
            continue
        contour = []
        for c in contours:
            if len(c) > 500:  # Filter out small contours
                contour.append(c)
        contour = np.concatenate(contour)
        ys, xs = contour[:, 0], contour[:, 1]

        # sort
        sorted_indices = np.argsort(xs)
        xs = xs[sorted_indices]
        ys = ys[sorted_indices]
        # remove duplicates
        unique_indices = np.unique(xs, return_index=True)[1]
        xs = xs[unique_indices]
        ys = ys[unique_indices]

        spl = make_smoothing_spline(xs, ys, lam=1e7)
        x_idx = np.arange(mask_smooth.shape[1])
        y_lim = spl(x_idx)
        y_lim = np.clip(y_lim, 0, mask_smooth.shape[0]-1)

        for x, y in zip(x_idx, y_lim):
            mask_smooth[:int(y+0.5), int(x)] = i

    return mask_smooth

def make_fiducial_chunk_map(band, BC_map, num_channels=17, num_subchannels=10, channel_file='/home/thomasli/spherex/spherex_channels.csv'):
    if num_channels%17 != 0:
        raise ValueError("num_channels must be a multiple of 17.")
    interp_factor = num_subchannels * num_channels//17
    channel_edges = extract_spherex_channel_edges(band, channel_file=channel_file)
    fine_edges = interpolate_array(channel_edges, interp_factor=interp_factor)
    chunk_map = make_spherex_chunk_map(BC_map, fine_edges)
    return chunk_map

def load_calibration(band, calibration_dir='/home/thomasli/spherex/spherex_calibration'):
    BC_files = glob.glob(os.path.join(calibration_dir, f'*BC_Band{band}.fits'))
    BW_files = glob.glob(os.path.join(calibration_dir, f'*BW_Band{band}.fits'))
    if len(BC_files) != 1 or len(BW_files) != 1:
        raise ValueError(f"Expected one BC and one BW file for band {band}, found {len(BC_files)} BC files and {len(BW_files)} BW files.")
    BC_map = fits.getdata(BC_files[0])
    BW_map = fits.getdata(BW_files[0])
    return BC_map, BW_map

def make_fiducial_chunk_mask(valid_channels, num_channels=17, num_subchannels=10):
    chunk_valid_mask = np.zeros(num_channels*num_subchannels + 2)
    chunk_valid_mask[np.hstack(((np.array(valid_channels)-1)*num_subchannels)[:, None] + np.arange(num_subchannels)) + 1] = 1
    return chunk_valid_mask

def visualize_chunk_map(chunk_map, chunk_valid_mask):
    masked_chunk_map = np.where(chunk_valid_mask[chunk_map], chunk_map, np.nan)
    plt.imshow(masked_chunk_map, cmap='viridis', interpolation='none')

def interp_1d(arr, method='mp', edge='extend'):
    idx = np.arange(len(arr))
    mean_idx, mean_val, edge_idx = parse_bin(arr)
    if method == 'mp_external':
        from mpsplines import MeanPreservingInterpolation as MPI
        # https://github.com/jararias/mpsplines
        mpi = MPI(yi=mean_val, xi=mean_idx)
        smooth_arr = mpi(idx)
    elif method == 'mp':
        mps_interp = mean_preserving_spline(edge_idx, mean_val, method='cubic')
        smooth_arr = mps_interp(idx)
    elif method == 'linear':
        smooth_arr = np.interp(idx, mean_idx, mean_val)
    return smooth_arr

def interp_2d_vertical(arr, method='mp'):
    return np.apply_along_axis(interp_1d, axis=0, arr=arr, method=method)

def parse_bin(arr):
    start = np.where(arr[:-1] != arr[1:])[0]+1
    edge = start - 1/2
    mean_idx = (start[:-1] + (start[1:] - 1))/2
    mean_val = arr[start[:-1]]
    return mean_idx, mean_val, edge

from scipy.interpolate import PchipInterpolator, CubicSpline, Akima1DInterpolator

def mean_preserving_spline(x_edges, avg_values, method='cubic'):
    """
    Generates a mean-preserving spline function f(x) based on edge
    positions x_edges and the average value avg_values in each interval.

    The function f(x) is constructed as the derivative of a monotonic
    cubic spline F(x), where F(x) is the integral of f(x).
    """
    assert len(x_edges) == len(avg_values) + 1, \
        "Length of x_edges must be 1 more than the length of avg_values."

    x_edges = np.asarray(x_edges, dtype=float)
    avg_values = np.asarray(avg_values, dtype=float)
    dx = np.diff(x_edges)
    interval_integrals = avg_values * dx
    integral_values = np.concatenate(([0], np.cumsum(interval_integrals)))

    if method == 'pchip':
        # Pchip (monotonic C1 for F, C0 for f)
        # Guarantees f(x) >= 0 if all avg_values >= 0
        F_spline = PchipInterpolator(x_edges, integral_values)
    elif method == 'akima':
        # Akima (local C1 for F, C0 for f)
        # Avoids ringing and often looks more natural than PCHIP.
        F_spline = Akima1DInterpolator(x_edges, integral_values)
    elif method == 'cubic':
        # Standard C^2 spline (C1 for f)
        # "Smoother" (f(x) will be C^1), but F(x) is not guaranteed
        # to be monotonic, so f(x) may go < 0 ("ringing").
        F_spline = CubicSpline(x_edges, integral_values, bc_type='not-a-knot')
    else:
        raise ValueError("method must be one of 'pchip', 'akima', or 'cubic'")

    f_spline = F_spline.derivative()

    return f_spline
