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
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Pool

from skimage import measure
from scipy.interpolate import make_smoothing_spline, griddata
from scipy.optimize import least_squares
from ...geometry.MapHelper import arc_spline, linear_spline, mean_preserving_spline, bit_to_bool, mean_preserving_spline_2d, get_valid_bounds
from ...MakeMap import load_reproj_file
from ...config import (resolve_path, ENV_SPHEREX_CALIB_DIR,
                       ENV_SPHEREX_CHANNEL_FILE, ENV_LVF_PARAMS_DIR)


# Canonical on-host paths for the SPHEREx spectral-calibration products. These
# are fallback defaults only: external users set $SELFCAL_SPHEREX_CALIB_DIR /
# $SELFCAL_SPHEREX_CHANNEL_FILE or pass explicit paths (see selfcal.config).
DEFAULT_CALIBRATION_DIR = '/home/thomasli/spherex/SPHEREx_Spectral_Calibration'
DEFAULT_CHANNEL_FILE = '/home/thomasli/spherex/spherex_channels.csv'


def load_calibration(band, calibration_dir=None):
    calibration_dir = resolve_path(
        calibration_dir, env_var=ENV_SPHEREX_CALIB_DIR,
        default=DEFAULT_CALIBRATION_DIR, what='SPHEREx calibration dir')
    BC_files = glob.glob(os.path.join(calibration_dir, f'*BC_Band{band}.fits'))
    BW_files = glob.glob(os.path.join(calibration_dir, f'*BW_Band{band}.fits'))
    if len(BC_files) != 1 or len(BW_files) != 1:
        raise ValueError(f"Expected one BC and one BW file for band {band}, found {len(BC_files)} BC files and {len(BW_files)} BW files.")
    BC_map = fits.getdata(BC_files[0])
    BW_map = fits.getdata(BW_files[0])
    return BC_map, BW_map


# --- PAH 3.29 μm aromatic emission feature defaults ------------------------
# Used by the per-pixel spectral-fit mode in setup_lsqr (sky block split into
# continuum + line amplitude per ref pixel; line column coefficient is the
# Gaussian profile evaluated at each observation's LVF wavelength).
#
# Line center: PAH 3.29 μm C-H stretch (Tokunaga 1991; Draine & Li 2007). Fixed
# at 3.290 μm — appropriate for galactic cirrus, biased for extragalactic.
# Intrinsic FWHM: ~30-40 nm from literature; we use 40 nm conservatively.
# LVF FWHM: ~94 nm median at the 3.29 arc in Band 4 BW_map. The combined
# observed sigma uses Gaussian-convolution-of-Gaussians: sigma_obs = sqrt(
#   (LVF_FWHM/2.355)^2 + (intrinsic_FWHM/2.355)^2 ) ≈ 0.0434 μm.
# For best fidelity callers should sample per-pixel BW_map and recompute
# sigma_per_pixel = sqrt((sub_BW/2.355)^2 + PAH_INTRINSIC_SIGMA_UM^2); the
# fixed defaults below are the fallback when BW_map is not threaded.
PAH_LINE_CENTER_UM = 3.290
PAH_INTRINSIC_FWHM_UM = 0.040
PAH_INTRINSIC_SIGMA_UM = PAH_INTRINSIC_FWHM_UM / 2.355  # ≈ 0.0170 μm
LVF_FWHM_AT_PAH_UM = 0.0942  # Band 4 BW_map median where BC ∈ [3.27, 3.31]
LINE_FWHM_UM = float(np.sqrt(LVF_FWHM_AT_PAH_UM**2 + PAH_INTRINSIC_FWHM_UM**2))  # ≈ 0.1023
LINE_SIGMA_UM = LINE_FWHM_UM / 2.355  # ≈ 0.0434 μm


def gaussian_line_profile(wave_um, center_um=PAH_LINE_CENTER_UM, sigma_um=LINE_SIGMA_UM):
    """Gaussian line profile, peak = 1 at wave_um = center_um.

    Used as the LSQR sky_line column coefficient in spectral-fit mode:
    each data row's sky_line entry is `valid_weight * gaussian_line_profile(λ_i)`,
    where λ_i is the LVF band-center wavelength at the sub-pixel that frame k
    sees ref pixel P through (sampled via BC_map[sub_mapping]).

    Parameters
    ----------
    wave_um : np.ndarray
        Per-pixel wavelengths in micrometers. May be scalar or any shape.
    center_um : float
        Line peak wavelength. Default PAH_LINE_CENTER_UM = 3.290 μm.
    sigma_um : float | np.ndarray
        Gaussian σ in μm. Default LINE_SIGMA_UM ≈ 0.0434 (LVF ⊕ PAH intrinsic).
        Pass an array (same shape as wave_um) when using per-pixel σ from
        BW_map for higher fidelity.

    Returns
    -------
    np.ndarray of float32, same shape as wave_um.
    """
    return np.exp(-0.5 * ((wave_um - center_um) / sigma_um)**2).astype(np.float32)


def extract_spherex_channel_edges(band, channel_file=None):
    channel_file = resolve_path(
        channel_file, env_var=ENV_SPHEREX_CHANNEL_FILE,
        default=DEFAULT_CHANNEL_FILE, what='SPHEREx channel file')
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
        lvf_params = fit_lvf_params(BC_map, channel_edges)

    r_edges = []
    y_bound = np.full(out_shape[1], out_shape[0]-1)
    
    for i, lam in tqdm(enumerate(channel_edges), total=len(channel_edges)):
        prev_y_bound = y_bound
        xc = lvf_params['xc']
        yc = lvf_params['yc']
        
        if lam not in lvf_params['wave_edges']:
            R = np.interp(lam, lvf_params['wave_edges'], lvf_params['R'])
        else:
            R = lvf_params['R'][np.where(lvf_params['wave_edges'] == lam)[0][0]]
        
        r_edges.append(R)

        spl = make_arc_spline(xc, yc, R)
        x_bound = np.arange(out_shape[1])
        y_bound = spl(x_bound/oversample_factor) * oversample_factor
        y_bound = np.clip(y_bound, 0, out_shape[1])
        chunk_map[(y_mesh >= y_bound) & (y_mesh < prev_y_bound)] = i
    else:
        prev_y_bound = y_bound
        y_bound = np.zeros_like(y_bound)
        chunk_map[(y_mesh >= y_bound) & (y_mesh < prev_y_bound)] = i + 1
    
    return chunk_map, lvf_params, np.array(r_edges)

def make_fiducial_chunk_map(band, BC_map, num_channels=17, num_subchannels=10,
                            channel_file=None,
                            oversample_factor=1, lvf_params=None):
    if num_channels%17 != 0:
        raise ValueError("num_channels must be a multiple of 17.")
    interp_factor = num_subchannels * num_channels//17
    channel_edges = extract_spherex_channel_edges(band, channel_file=channel_file)
    fine_edges = interpolate_array(channel_edges, interp_factor=interp_factor)
    
    chunk_map, lvf_params, r_edges = make_spherex_chunk_map(
        BC_map, fine_edges, oversample_factor=oversample_factor, lvf_params=lvf_params
    )
    return chunk_map, lvf_params, r_edges

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

_offset_worker_ctx = {}

def _offset_worker_init(shm_name, shm_shape, shm_dtype, max_chunk_id):
    """Attach shared memory chunk_map once per worker process."""
    shm = SharedMemory(name=shm_name)
    _offset_worker_ctx['chunk_map'] = np.ndarray(shm_shape, dtype=shm_dtype, buffer=shm.buf)
    _offset_worker_ctx['shm'] = shm
    _offset_worker_ctx['max_chunk_id'] = max_chunk_id

def _offset_worker_func(reproj_file):
    """Combined worker: HDF5 attr read -> FITS read -> bincount mean."""
    file_path = load_reproj_file(reproj_file, fields=['file_path'])['file_path']

    with fits.open(file_path) as hdul:
        data = hdul[1].data
        bitmask = hdul[2].data

    chunk_map = _offset_worker_ctx['chunk_map']
    max_id = _offset_worker_ctx['max_chunk_id']

    valid = bit_to_bool(bitmask, ignore_list=[], invert=True)
    flat_cm = chunk_map.ravel()
    flat_data = data.ravel().astype(np.float64)
    flat_valid = valid.ravel() & (flat_cm >= 0)

    sums = np.bincount(flat_cm[flat_valid], weights=flat_data[flat_valid], minlength=max_id + 1)
    counts = np.bincount(flat_cm[flat_valid], minlength=max_id + 1)
    mean = np.where(counts > 0, sums / counts, 0.0)
    return mean

def compute_offsets_guess(reproj_list, det_chunk_map, max_workers=16):
    max_chunk_id = int(np.max(det_chunk_map))

    shm = SharedMemory(create=True, size=det_chunk_map.nbytes)
    np.ndarray(det_chunk_map.shape, dtype=det_chunk_map.dtype, buffer=shm.buf)[:] = det_chunk_map

    try:
        with Pool(
            processes=max_workers,
            initializer=_offset_worker_init,
            initargs=(shm.name, det_chunk_map.shape, det_chunk_map.dtype, max_chunk_id)
        ) as pool:
            results = list(tqdm(
                pool.imap(_offset_worker_func, reproj_list, chunksize=20),
                total=len(reproj_list),
                desc="Calculating initial guess offsets"
            ))
    finally:
        shm.close()
        shm.unlink()

    return np.array(results)


# lvf_params ship with the package under instruments/spherex/data/lvf_params/.
# Resolved relative to this module so it is correct in every worktree and in an
# installed wheel; overridable via $SELFCAL_LVF_PARAMS_DIR or an explicit
# input_dir/output_dir.
_LVF_PARAMS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               'data', 'lvf_params')


def load_lvf_params(filename, input_dir=None):
    input_dir = resolve_path(input_dir, env_var=ENV_LVF_PARAMS_DIR,
                             default=_LVF_PARAMS_DIR, what='LVF params dir')
    input_path = os.path.join(input_dir, filename)
    if not os.path.exists(input_path):
        print(f"LVF parameters file {input_path} not found. Returning None.")
        return None
    lvf_params = np.load(input_path, allow_pickle=True).item()
    print(f"Loaded LVF parameters from {input_path}")
    return lvf_params

def save_lvf_params(lvf_params, output_dir=None):
    output_dir = resolve_path(output_dir, env_var=ENV_LVF_PARAMS_DIR,
                              default=_LVF_PARAMS_DIR, what='LVF params dir',
                              must_exist=False)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, lvf_params['filename'])
    np.save(output_path, lvf_params)
    print(f"Saved LVF parameters to {output_path}")

def compute_column_adjacency(chunk_map, num_columns):
    """
    Generates adjacency pairs ONLY for vertical strip transitions, 
    ignoring spectral arc transitions.
    
    Parameters
    ----------
    chunk_map : np.ndarray
        The full ID map (Subchannel * N + Band)
    num_columns : int
        The NUM_COLUMNS constant used to build the map.
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
    sub_u = u // num_columns
    sub_v = v // num_columns
    
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

def compute_subchannel_adjacency(chunk_map, num_columns):
    """
    Generates adjacency pairs for vertical subchannel transitions.
    This links chunk IDs across the boundaries of subchannels, keeping within the same column.
    """
    print("Computing Vertical Subchannel Adjacency...")
    
    # Compare pixel i with pixel i+1 vertically
    mask = (chunk_map[:-1, :] != -1) & \
           (chunk_map[1:, :] != -1) & \
           (chunk_map[:-1, :] != chunk_map[1:, :])
           
    u = chunk_map[:-1, :][mask]
    v = chunk_map[1:, :][mask]
    
    # Check that they represent different subchannels in the same column
    sub_u = u // num_columns
    sub_v = v // num_columns
    col_u = u % num_columns
    col_v = v % num_columns
    
    # Keep pairs that are adjacent vertically AND in the same column
    valid_pair_mask = (np.abs(sub_u - sub_v) == 1) & (col_u == col_v)
    
    u_filtered = u[valid_pair_mask]
    v_filtered = v[valid_pair_mask]
    
    if len(u_filtered) == 0:
        print("Found 0 vertical subchannel boundaries.")
        return np.array([]), np.array([])
        
    pairs = np.sort(np.stack([u_filtered, v_filtered], axis=1), axis=1)
    unique_pairs = np.unique(pairs, axis=0)
    
    print(f"Found {len(unique_pairs)} vertical subchannel boundaries.")
    return unique_pairs[:, 0], unique_pairs[:, 1]

def compute_column_polynomial_chains(chunk_map, num_columns, degree=1):
    """Build chains and stencil for a polynomial-degree constraint along
    columns within each subchannel of a SPHEREx stripped chunk map.

    A polynomial of degree ``degree`` is annihilated by the
    ``(degree + 1)``-th finite-difference operator, which has
    ``L = degree + 2`` coefficients ``(-1)^k * C(degree + 1, k)`` for
    ``k = 0..degree + 1``. Examples:

    ====== === =================
    degree  L  stencil
    ====== === =================
    0       2  ``[1, -1]`` (the existing constant-prior adjacency)
    1       3  ``[1, -2, 1]`` (linear)
    2       4  ``[1, -3, 3, -1]`` (quadratic)
    ====== === =================

    For each subchannel ``s``, sliding windows of length ``L`` over the
    columns ``[chunk(s, 0), chunk(s, 1), ...]`` form the chains; there are
    ``num_columns - L + 1 = num_columns - degree - 1`` windows per
    subchannel.

    Parameters
    ----------
    chunk_map : (det_h, det_w) int ndarray
        Stripped chunk map produced by ``make_stripped_chunk_map``. Chunk
        IDs are assumed to be ``subchannel * num_columns + column``;
        ``num_subchannels`` is inferred as ``(chunk_map.max() + 1) // num_columns``.
    num_columns : int
        Number of column subdivisions per subchannel.
    degree : int
        Polynomial degree to enforce (1 = linear, 2 = quadratic, ...).

    Returns
    -------
    chains : (num_subchannels * (num_columns - degree - 1), degree + 2) int64 ndarray
    stencil : (degree + 2,) float64 ndarray

    Raises
    ------
    ValueError
        If ``degree < 0`` or ``num_columns < degree + 2`` (no length-L window fits).
    """
    from math import comb

    if degree < 0:
        raise ValueError(f"degree must be >= 0 (got {degree})")
    L = degree + 2
    if num_columns < L:
        raise ValueError(
            f"num_columns={num_columns} too small for degree={degree}: need "
            f">= {L} columns per subchannel for a length-{L} chain")

    num_chunks = int(chunk_map.max()) + 1
    if num_chunks % num_columns != 0:
        raise ValueError(
            f"chunk_map.max()+1={num_chunks} is not divisible by "
            f"num_columns={num_columns}; cannot infer num_subchannels")
    num_subchannels = num_chunks // num_columns
    num_windows = num_columns - L + 1  # = num_columns - degree - 1

    sub_idx = np.arange(num_subchannels)[:, None, None]
    win_start = np.arange(num_windows)[None, :, None]
    offset = np.arange(L)[None, None, :]
    chunk_ids = sub_idx * num_columns + win_start + offset
    chains = chunk_ids.reshape(num_subchannels * num_windows, L).astype(np.int64)

    stencil = np.array(
        [(-1) ** k * comb(degree + 1, k) for k in range(L)],
        dtype=np.float64,
    )
    return chains, stencil


def compute_subchannel_polynomial_chains(num_subchannels, num_columns,
                                         degree=1, subch_lo=None, subch_hi=None):
    """Subchannel-direction analog of ``compute_column_polynomial_chains``.

    For each column ``c``, sliding windows of length ``L = degree + 2`` over
    consecutive subchannels ``s, s+1, ..., s+L-1`` form the chains, optionally
    restricted to a window ``[subch_lo, subch_hi]`` on ``s`` (inclusive).

    The chunk-id convention matches the rest of the codebase:
    ``chunk(s, c) = s * num_columns + c``.

    Together with the FD stencil ``(-1)^k * C(degree+1, k)``, the constraint
    ``λ · Σ_ℓ stencil[ℓ] · o[k, chains[r, ℓ]] = 0`` annihilates any polynomial
    of degree ``≤ degree`` in ``s``, per frame ``k`` and per chain ``r``. Use
    this to force the per-frame offset to be a low-order polynomial along the
    subchannel axis within a spectral window — e.g. degree=3 over the PAH
    window so anything Gaussian-shaped is pushed onto the sky_line column
    instead of being absorbed by the offset.

    Parameters
    ----------
    num_subchannels : int
        Total number of subchannels in the chunk map (``TOT_SUB``).
    num_columns : int
        Number of columns per subchannel (``NumCol``).
    degree : int
        Polynomial degree to enforce (1 = linear, 2 = quadratic, ...).
    subch_lo, subch_hi : int or None
        Inclusive lower/upper bounds on the chain's starting subchannel ``s``
        — i.e. the chain spans ``[s, s+L-1]``. When ``None``, defaults to the
        full range ``[0, num_subchannels-L]``.

    Returns
    -------
    chains : (num_chains, L) int64 ndarray
    stencil : (L,) float64 ndarray
    """
    from math import comb

    if degree < 0:
        raise ValueError(f"degree must be >= 0 (got {degree})")
    L = degree + 2
    if num_subchannels < L:
        raise ValueError(
            f"num_subchannels={num_subchannels} too small for degree={degree}: "
            f"need >= {L}")
    s_lo = 0 if subch_lo is None else int(subch_lo)
    s_hi_chain_start = (num_subchannels - L) if subch_hi is None else int(subch_hi) - L + 1
    if s_lo < 0 or s_hi_chain_start > num_subchannels - L:
        raise ValueError(
            f"window subch_lo={subch_lo}, subch_hi={subch_hi} (chain start "
            f"range [{s_lo}, {s_hi_chain_start}]) outside valid "
            f"[0, {num_subchannels - L}]")
    if s_hi_chain_start < s_lo:
        raise ValueError(
            f"window [{subch_lo}, {subch_hi}] yields no length-{L} chains")

    s_starts = np.arange(s_lo, s_hi_chain_start + 1, dtype=np.int64)  # (n_starts,)
    n_starts = s_starts.size
    cols = np.arange(num_columns, dtype=np.int64)  # (num_columns,)
    offsets = np.arange(L, dtype=np.int64)  # (L,)
    # chains shape: (n_starts, num_columns, L)
    # chain[i, c, l] = (s_starts[i] + l) * num_columns + cols[c]
    chunk_ids = (s_starts[:, None, None] + offsets[None, None, :]) * num_columns \
                + cols[None, :, None]
    chains = chunk_ids.reshape(n_starts * num_columns, L)

    stencil = np.array(
        [(-1) ** k * comb(degree + 1, k) for k in range(L)],
        dtype=np.float64,
    )
    return chains, stencil


def make_stripped_chunk_map(detector, num_subchannels=10, num_channels=17,
                            oversample_factor=1, num_columns=1, lvf_params=None,
                            calibration_dir=None):
    det_BC, det_BW = load_calibration(band=detector, calibration_dir=calibration_dir)
    
    subchannel_map, lvf_params, r_edges = make_fiducial_chunk_map(
        detector, det_BC, num_subchannels=num_subchannels, num_channels=num_channels, 
        oversample_factor=oversample_factor, lvf_params=lvf_params
    )
    
    vertchunk_map = np.zeros_like(subchannel_map)
    width = vertchunk_map.shape[1]
    x_edges = np.linspace(0, width, num_columns + 1)
    
    for band in range(num_columns):
        start = int(x_edges[band])
        end = int(x_edges[band+1])
        vertchunk_map[:, start:end] = band

    chunk_map = subchannel_map * num_columns + vertchunk_map
    
    return chunk_map, lvf_params, r_edges, x_edges

def make_stripped_chunk_valid_mask(ch=None, subch=None, num_subchannels=10, num_channels=17, 
                                   num_columns=1, subchannel_padding=0):
    def make_chunk_valid_mask(subchannel_valid_mask, num_columns):
        chunk_valid_mask = np.zeros(len(subchannel_valid_mask)*num_columns, dtype=subchannel_valid_mask.dtype)
        for band in range(num_columns):
            chunk_valid_mask[band::num_columns] = subchannel_valid_mask
        return chunk_valid_mask
    if ch is not None:
        subchannel_valid_mask = make_fiducial_chunk_mask(ch, num_subchannels=num_subchannels, num_channels=num_channels, padding=subchannel_padding)
    elif subch is not None:
        subchannel_valid_mask = np.zeros(num_subchannels*num_channels+2, dtype=bool)
        subchannel_valid_mask[subch] = 1
    else:
        raise ValueError("Either ch or subch must be provided.")
    chunk_valid_mask = make_chunk_valid_mask(subchannel_valid_mask, num_columns=num_columns)
    return chunk_valid_mask

def make_spherex_stripped_offset_map(chunk_map, chunk_offset, chunk_valid_mask, lvf_params, r_edges, x_edges, tot_subchannels, num_columns, fill_invalid=False):
    reshaped_offset = chunk_offset.reshape(tot_subchannels, num_columns)[1:-1]
    reshaped_valid_mask = chunk_valid_mask.reshape(tot_subchannels, num_columns)[1:-1]

    y_slice, x_slice = get_valid_bounds(~reshaped_valid_mask.astype(bool))

    trimmed_offset = reshaped_offset[y_slice, x_slice]
    if fill_invalid:
        trimmed_offset = fill_invalid_offsets(trimmed_offset)
    trimmed_r_edges = r_edges[y_slice.start : y_slice.stop + 1]
    trimmed_x_edges = x_edges[x_slice.start : x_slice.stop + 1]

    spl = mean_preserving_spline_2d(trimmed_r_edges, trimmed_x_edges, trimmed_offset, x_degree=3, y_degree=3)

    xc, yc = lvf_params['xc'], lvf_params['yc']

    h, w = np.shape(chunk_map)
    oversample_factor = h // 2040
    subpixel_shift = 0.5 / oversample_factor
    det_size = 2040
    increment = 1 / oversample_factor
    x_mesh, y_mesh = np.meshgrid(np.arange(subpixel_shift, det_size+subpixel_shift, increment), np.arange(subpixel_shift, det_size+subpixel_shift, increment))
    r_mesh = np.sqrt((y_mesh - yc)**2 + (x_mesh - xc)**2)
    
    offset_map = spl(r_mesh, x_mesh)
    return offset_map

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

def fast_vertical_dist(arr):
    rows, cols = arr.shape
    # Result arrays
    dist_up = np.zeros((rows, cols), dtype=np.int32)
    dist_down = np.zeros((rows, cols), dtype=np.int32)

    # We use a running count that resets at every 0
    # 1. Distance to zero ABOVE
    for r in range(1, rows):
        # If current is 1, distance is (dist of row above) + 1
        # If current is 0, distance is 0
        dist_up[r] = (dist_up[r-1] + 1) * (arr[r] != 0)

    # 2. Distance to zero BELOW
    for r in range(rows - 2, -1, -1):
        dist_down[r] = (dist_down[r+1] + 1) * (arr[r] != 0)

    return np.minimum(dist_up, dist_down).astype(np.float32)