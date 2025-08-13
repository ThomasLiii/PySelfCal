import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm

from skimage import measure
from scipy.interpolate import make_smoothing_spline

def interpolate_array(data_arr, interp_factor=5):
    interp_arr = np.hstack([
        np.linspace(data_arr[i], data_arr[i + 1], interp_factor, endpoint=False) 
        for i in range(len(data_arr) - 1)
    ] + [data_arr[-1]])  # Append the last element
    return interp_arr

def make_chunk_map(band, interp_factor=5,
                   calibration_file='/data1/SPHEREx/Data/Survey_3/calibs/SSDC_SpecCal_v2025Feb/SSDC_CENTER_all_v20250202.fits'):
    hdul = fits.open(calibration_file)
    wav_map = hdul[0].data[band-1]
    tbl = Table.read('/home/thomasli/spherex/spherex_nep_catalogues/spherex_channels.csv')
    sub_tbl = tbl[tbl['band'] == band]
    channel_edges = np.hstack([sub_tbl['lmin'].data, sub_tbl['lmax'].data[-1:]])
    fine_edges = interpolate_array(channel_edges, interp_factor=interp_factor)
    channel_idx = np.searchsorted(fine_edges, wav_map, side='right')

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

def make_chunk_mask(valid_channels, interp_factor=5):
    chunk_valid_mask = np.zeros(17*interp_factor + 2)
    chunk_valid_mask[np.hstack(((np.array(valid_channels)-1)*interp_factor)[:, None] + np.arange(interp_factor)) + 1] = 1
    return chunk_valid_mask

def visualize_chunk_map(chunk_map, chunk_valid_mask):
    masked_chunk_map = np.where(chunk_valid_mask[chunk_map], chunk_map, np.nan)
    plt.imshow(masked_chunk_map, cmap='viridis', interpolation='none')

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