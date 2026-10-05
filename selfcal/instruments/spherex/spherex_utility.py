"""SPHEREx linear variable filter (LVF) geometry: arc fits, chunk maps and offset maps.

A SPHEREx pixel's band centre ``BC`` is nearly constant along concentric circular arcs.
:func:`load_calibration` reads a detector's ``BC`` / ``BW`` maps; :func:`fit_lvf_params`
fits the arcs, which :func:`load_lvf_params` and :func:`save_lvf_params` read and write.
:func:`make_stripped_chunk_map` cuts the detector into subchannels between neighbouring
arcs and those into columns, :func:`make_stripped_chunk_valid_mask` selects a job's chunks
and :func:`make_spherex_stripped_offset_map` renders chunk offsets as a smooth map;
:class:`~selfcal.instruments.spherex.adapter.SPHERExInstrument` builds a run's geometry
with them. :func:`gaussian_line_profile` is a Gaussian line, by default the PAH 3.29 μm
feature. The adjacency and chain builders here were replaced by
:mod:`selfcal.models.offset_structure`.
"""
import os
import glob
import logging
import numpy as np
from astropy.io import fits
from astropy.table import Table
from tqdm import tqdm
from multiprocessing.shared_memory import SharedMemory
from multiprocessing import Pool

from scipy.interpolate import griddata
from scipy.optimize import least_squares
from ... import _state
from ...geometry.map_helper import (linear_spline, mean_preserving_spline, bit_to_bool, mean_preserving_spline_2d, get_valid_bounds, fill_invalid_offsets)
from ...io.reproj import load_reproj_file
from ...config import (resolve_path, ENV_SPHEREX_CALIB_DIR,
                       ENV_SPHEREX_CHANNEL_FILE, ENV_LVF_PARAMS_DIR)

logger = logging.getLogger(__name__)


# The SPHEREx spectral-calibration maps (BC / BW) live on the processing host; this
# is a fallback default only: external users set $SELFCAL_SPHEREX_CALIB_DIR or pass
# an explicit path (see selfcal.config). The channel table (102 channels, 17 per
# band: lmin / lmean / lmax ...) ships with the package; $SELFCAL_SPHEREX_CHANNEL_FILE
# or an explicit path overrides it.
DEFAULT_CALIBRATION_DIR = '/data3/SPHEREx/SpecCal_202509/ParameterFiles'
DEFAULT_CHANNEL_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'data', 'spherex_channels.csv')


def load_calibration(band, calibration_dir=None):
    """Read the LVF band-centre and band-width maps of one SPHEREx detector.

    Returns ``(BC_map, BW_map)``, the data of the single ``*BC_Band<band>.fits`` and
    ``*BW_Band<band>.fits`` files in the calibration directory: detector-sized maps in μm,
    where ``band`` is the detector number (1 to 6). The directory resolves from
    ``calibration_dir``, then ``$SELFCAL_SPHEREX_CALIB_DIR``, then
    ``DEFAULT_CALIBRATION_DIR`` (:class:`~selfcal.config.SelfCalConfigError` if it does not
    exist). Raises ``ValueError`` unless exactly one file of each kind matches.
    """
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

    A standalone helper (nothing in the package calls it): the solve's line
    coefficients come from the model's sky terms (``gaussian`` or ``template``
    functions of the band-centre wavelength ``BC``, see
    :mod:`selfcal.models.sky_model`).

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
    """Return the wavelength edges (μm) of one band's channels from the SPHEREx channel table.

    Reads the table with astropy (columns ``band``, ``lmin``, ``lmax``) and returns the
    ``lmin`` of each of the band's channels, in table order, followed by the last channel's
    ``lmax``: 18 edges for the 17 channels of a band. The table resolves from
    ``channel_file``, then ``$SELFCAL_SPHEREX_CHANNEL_FILE``, then ``DEFAULT_CHANNEL_FILE``,
    the table shipped with the package (``data/spherex_channels.csv``).
    """
    channel_file = resolve_path(
        channel_file, env_var=ENV_SPHEREX_CHANNEL_FILE,
        default=DEFAULT_CHANNEL_FILE, what='SPHEREx channel file')
    tbl = Table.read(channel_file)
    sub_tbl = tbl[tbl['band'] == band]
    channel_edges = np.hstack([sub_tbl['lmin'].data, sub_tbl['lmax'].data[-1:]])
    return channel_edges

def interpolate_array(data_arr, interp_factor=5):
    """Subdivide each interval of a 1-D array into ``interp_factor`` equal steps.

    Inserts ``interp_factor - 1`` linearly spaced values between neighbouring elements and
    returns the ``(len(data_arr) - 1) * interp_factor + 1`` values, every original one
    included. :func:`make_fiducial_chunk_map` uses it to split channel edges into
    subchannel edges.
    """
    interp_arr = np.hstack([
        np.linspace(data_arr[i], data_arr[i + 1], interp_factor, endpoint=False) 
        for i in range(len(data_arr) - 1)
    ] + [data_arr[-1]])  # Append the last element
    return interp_arr

def extract_edge_samples(BC_map, channel_edges):
    """Trace each wavelength edge across the detector as the row where ``BC`` is closest.

    For every wavelength in ``channel_edges`` and every column ``x``, the sample is the row
    that minimises ``|BC_map[:, x] - wavelength|``. Returns ``(edge_x, edge_y)``, two float32
    arrays of shape ``(len(channel_edges), BC_map.shape[1])``. For the last edge the central
    columns, ``650 < x < BC_map.shape[0] - 650``, where that arc can run below row 0, are set
    to NaN in both; for the first edge, the outer columns ``x < 50`` and
    ``x > BC_map.shape[0] - 50``, where that arc can run past the last row.
    """
    edge_x_list = []
    edge_y_list = []
    for i, lam in tqdm(enumerate(channel_edges), total=len(channel_edges),
                       disable=not _state.progress_enabled):
        edge_y = np.argmin(np.abs(BC_map - lam), axis=0).astype(np.float32)
        edge_x = np.arange(len(edge_y)).astype(np.float32)

        if i == len(channel_edges)-1:
            edge_mask = (edge_x > 650) & (edge_x < BC_map.shape[0]-650)
            edge_y[edge_mask] = np.nan
            edge_x[edge_mask] = np.nan
        elif i == 0:
            # Known limitation (2026-10, left as is): the first arc runs past the last row
            # on more columns than this window covers (D4: 129 clipped columns, 99 masked),
            # and argmin returns the clipped last row there. Masking every sample that sits
            # on the first or last row would be stricter. It only matters when the LVF
            # params are regenerated (task `precompute`): the shipped fits used the old,
            # never-firing `&` mask, and the `|` refit moves D4's arcs by <= 0.29 px.
            edge_mask = (edge_x < 50) | (edge_x > BC_map.shape[0]-50)
            edge_y[edge_mask] = np.nan
            edge_x[edge_mask] = np.nan

        edge_x_list.append(edge_x)
        edge_y_list.append(edge_y)

    return np.array(edge_x_list), np.array(edge_y_list)
    
def fit_lvf_arcs(edge_x_list, edge_y_list):
    """Fit concentric circles, one radius per wavelength edge, to traced edge samples.

    The ``(n_edges, n_samples)`` arrays come from :func:`extract_edge_samples`. A
    Levenberg-Marquardt least-squares fit finds a shared centre ``(xc, yc)`` and one radius
    per edge that minimise each sample's distance from the centre minus its edge's radius;
    NaN samples contribute nothing. It starts from ``xc = 1020``, ``yc = 9632.4376`` and,
    per edge, the distance of the mean sample from that centre. Returns
    ``{'xc': xc, 'yc': yc, 'R': R}`` in detector pixels, ``R`` of shape ``(n_edges,)``.
    Raises ``ValueError`` when the two arrays differ in shape and ``RuntimeError`` when the
    optimiser reports failure.
    """
    if edge_x_list.shape != edge_y_list.shape:
        raise ValueError("x and y must be the same shape.")

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
    """Return the function ``y(x) = yc - sqrt(R**2 - (x - xc)**2)`` of one LVF arc.

    ``y(x)`` is the row, at column ``x``, of the half of the circle of radius ``R`` about
    ``(xc, yc)`` that lies below the centre (rows less than ``yc``), in detector pixels; it
    is NaN where ``|x - xc| > R``.
    """
    def arc_spline(x):
        return -np.sqrt(R**2 - (x - xc)**2) + yc
    return arc_spline

def fit_lvf_params(BC_map, channel_edges):
    """Fit the LVF arc model to a band-centre map, giving the detector's ``lvf_params``.

    Traces the ``channel_edges`` wavelengths across ``BC_map`` (:func:`extract_edge_samples`),
    fits concentric arcs to them (:func:`fit_lvf_arcs`) and returns
    ``{'xc', 'yc', 'R', 'wave_edges'}``, where ``wave_edges`` is ``channel_edges`` and
    ``R[i]`` is the radius, in detector pixels, of the arc at ``wave_edges[i]``.
    """
    edge_x_list, edge_y_list = extract_edge_samples(BC_map, channel_edges)
    lvf_params = fit_lvf_arcs(edge_x_list, edge_y_list)
    lvf_params['wave_edges'] = channel_edges
    return lvf_params

def make_spherex_chunk_map(BC_map, channel_edges, oversample_factor=1, lvf_params=None):
    """Label each pixel with its subchannel, the strip between two neighbouring arcs.

    Edge ``i`` (wavelength ``channel_edges[i]``) is the arc of :func:`make_arc_spline` whose
    radius comes from ``lvf_params``: ``R`` at that wavelength of ``wave_edges`` when it is
    listed there, otherwise interpolated linearly in wavelength; ``lvf_params=None`` fits
    the arcs to ``BC_map`` first (:func:`fit_lvf_params`). A pixel gets id ``i`` between
    arcs ``i - 1`` and ``i``, id ``0`` between the first arc and the last row, and id
    ``len(channel_edges)`` between the last arc and row 0 (radii growing along
    ``channel_edges``, as the fitted ones do). Apart from that fit, ``BC_map`` only sets the
    shape: the map is ``oversample_factor`` times its size on each axis, pixel ``(i, j)``
    lying at detector position ``(i, j) / oversample_factor``.

    Returns ``(chunk_map, lvf_params, r_edges)``: the int32 id map, the arc parameters used
    and the radii of the ``len(channel_edges)`` arcs, in detector pixels.
    """
    out_shape = (BC_map.shape[0]*oversample_factor, BC_map.shape[1]*oversample_factor)
    chunk_map = np.zeros(out_shape, dtype=np.int32)
    x_mesh, y_mesh = np.meshgrid(np.arange(out_shape[1]), np.arange(out_shape[0]))
    
    if lvf_params is None:
        lvf_params = fit_lvf_params(BC_map, channel_edges)

    r_edges = []
    y_bound = np.full(out_shape[1], out_shape[0]-1)
    
    for i, lam in tqdm(enumerate(channel_edges), total=len(channel_edges),
                       disable=not _state.progress_enabled):
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

    # Region below the last (largest-R) arc gets the final chunk id (i + 1 after the loop).
    prev_y_bound = y_bound
    y_bound = np.zeros_like(y_bound)
    chunk_map[(y_mesh >= y_bound) & (y_mesh < prev_y_bound)] = i + 1

    return chunk_map, lvf_params, np.array(r_edges)

def make_fiducial_chunk_map(band, BC_map, num_channels=17, num_subchannels=10,
                            channel_file=None,
                            oversample_factor=1, lvf_params=None):
    """Build a band's subchannel map: ``num_channels`` channels of ``num_subchannels`` each.

    Splits each channel of the SPHEREx channel table (:func:`extract_spherex_channel_edges`,
    17 per band) into ``num_subchannels * num_channels // 17`` equal wavelength steps
    (:func:`interpolate_array`), so that each of the ``num_channels`` channels holds
    ``num_subchannels`` subchannels, and labels the pixels with
    :func:`make_spherex_chunk_map`. Ids run from ``0`` to ``num_channels * num_subchannels + 1``;
    the first and the last are the regions beyond the band's first and last edges. Returns
    ``(chunk_map, lvf_params, r_edges)`` as :func:`make_spherex_chunk_map` does. Raises
    ``ValueError`` unless ``num_channels`` is a multiple of 17.
    """
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
    """Return a 0/1 mask over subchannel ids that selects the subchannels of given channels.

    The float mask has ``num_channels * num_subchannels + 2`` entries, one per id of
    :func:`make_fiducial_chunk_map`. It is 1 on the ``num_subchannels`` subchannels of each
    channel in ``valid_channels`` (a sequence of channel numbers counted from 1), widened by
    ``padding`` subchannels on each side: channel ``c`` covers ids
    ``(c - 1) * num_subchannels + 1 - padding`` to ``c * num_subchannels + padding``.
    """
    chunk_valid_mask = np.zeros(num_channels*num_subchannels + 2)
    valid_subchannels = np.hstack(((np.array(valid_channels)-1)*num_subchannels)[:, None] + \
                                  np.arange(0-padding,num_subchannels+padding)) + 1
    chunk_valid_mask[valid_subchannels] = 1
    return chunk_valid_mask

def visualize_chunk_map(chunk_map, chunk_valid_mask):
    """Plot a chunk map with ``plt.imshow``, blanking the pixels of invalid chunks.

    Pixels whose chunk has ``chunk_valid_mask[id] == 0`` are drawn as NaN. The image goes to
    pyplot's current axes; nothing is returned.
    """
    # Lazy import: matplotlib is only needed for this plotting helper, so the
    # module (and the pipeline that imports it) does not depend on it at import.
    import matplotlib.pyplot as plt
    masked_chunk_map = np.where(chunk_valid_mask[chunk_map], chunk_map, np.nan)
    plt.imshow(masked_chunk_map, cmap='viridis', interpolation='none')

def interp_1d(arr, method='mp', edge='extend'):
    """Smooth a piecewise-constant 1-D profile by interpolating between its runs of values.

    :func:`parse_bin` splits ``arr`` into runs of equal values; the first and last runs,
    which the array's ends may cut, do not constrain the result. ``method='mp'`` (the
    default) fits a mean-preserving cubic spline over the interior runs' extents
    (:func:`~selfcal.geometry.map_helper.mean_preserving_spline`) and extrapolates it over
    the end runs; ``'linear'`` interpolates linearly between the interior runs' centres
    (:func:`~selfcal.geometry.map_helper.linear_spline`), holding the end values beyond them;
    ``'mp_external'`` uses ``MeanPreservingInterpolation`` of the optional ``mpsplines``
    package (the ``mpsplines`` extra). Any other method raises ``UnboundLocalError``.
    ``edge`` is not used. Returns the interpolant at every index, a float array of
    ``len(arr)``.
    """
    idx = np.arange(len(arr))
    mean_idx, mean_val, edge_idx = parse_bin(arr)
    if method == 'mp_external':
        # Optional external mean-preserving interpolator. Imported lazily so the
        # package installs from PyPI without the git-only mpsplines dependency;
        # the default 'mp' method below uses the in-tree scipy implementation.
        # Install the [mpsplines] extra to use method='mp_external'.
        from mpsplines import MeanPreservingInterpolation as MPI
        interpolator = MPI(yi=mean_val, xi=mean_idx)
    elif method == 'mp':
        interpolator = mean_preserving_spline(edge_idx, mean_val, method='cubic')
    elif method == 'linear':
        interpolator = linear_spline(mean_idx, mean_val)
    smooth_arr = interpolator(idx)
    return smooth_arr

def interp_2d_vertical(arr, method='mp'):
    """Apply :func:`interp_1d` to every column of a 2-D array, returning the same shape."""
    return np.apply_along_axis(interp_1d, axis=0, arr=arr, method=method)

def parse_bin(arr):
    """Find the runs of equal values in a 1-D array: interior-run centres, values and edges.

    Returns ``(mean_idx, mean_val, edge)``. ``edge`` holds the position ``i - 0.5`` of every
    change of value (``arr[i] != arr[i - 1]``); ``mean_idx`` and ``mean_val`` hold the
    centre index and the value of each run between two changes, so
    ``len(edge) == len(mean_val) + 1``. The first and last runs, bounded by the array's
    ends, are left out.
    """
    start = np.where(arr[:-1] != arr[1:])[0]+1
    edge = start - 1/2
    mean_idx = (start[:-1] + (start[1:] - 1))/2
    mean_val = arr[start[:-1]]
    return mean_idx, mean_val, edge

def make_spherex_offset_map(chunk_map, chunk_offset, chunk_valid_mask, lvf_params):
    """Render one frame's per-subchannel offsets as a smooth map in arc radius.

    The single-column counterpart of :func:`make_spherex_stripped_offset_map`. It fits a
    mean-preserving cubic spline in radius
    (:func:`~selfcal.geometry.map_helper.mean_preserving_spline`) whose average between the
    two arcs of each valid subchannel is that subchannel's ``chunk_offset``, and evaluates it
    at every pixel's distance from the arc centre ``(xc, yc)``. ``chunk_offset`` and
    ``chunk_valid_mask`` are indexed by the subchannel ids of the arcs ``lvf_params['R']``
    (``len(R) + 1`` entries); the valid ids must be contiguous and exclude ``0`` and
    ``len(R)``. Only the shape of ``chunk_map`` is used: the oversampling is
    ``shape[0] // 2040``, pixel ``(i, j)`` lying at detector position
    ``(i, j) / oversample``. Returns a float array of that shape.
    """
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
    """Average each frame's raw pixels per chunk, a quick first guess of the chunk offsets.

    For every reprojected file in ``reproj_list``, reads the exposure named by its
    ``file_path`` attribute (data in FITS extension 1, bit mask in extension 2) and averages
    the data over the pixels with no mask bit set in each chunk of ``det_chunk_map``, which
    must have the exposure's shape; negative ids are skipped. Returns a float64 array of
    shape ``(len(reproj_list), det_chunk_map.max() + 1)``, 0 for a chunk without a valid
    pixel. A pool of ``max_workers`` processes reads the frames and shares the chunk map
    through shared memory.
    """
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
                desc="Calculating initial guess offsets",
                disable=not _state.progress_enabled
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
    """Load a saved LVF arc fit (``lvf_params``), or return ``None`` if the file is missing.

    Reads ``filename`` (a dict saved by :func:`save_lvf_params`, with keys ``xc``, ``yc``,
    ``R``, ``wave_edges`` and ``filename``) from the directory ``input_dir``, else
    ``$SELFCAL_LVF_PARAMS_DIR``, else the package's ``data/lvf_params/``. The package ships
    ``lvf_params_D1.npy`` to ``lvf_params_D6.npy``, fitted at the 341 subchannel edges of 34
    channels of 10 subchannels. A missing directory raises
    :class:`~selfcal.config.SelfCalConfigError`; a missing file logs a warning.
    """
    input_dir = resolve_path(input_dir, env_var=ENV_LVF_PARAMS_DIR,
                             default=_LVF_PARAMS_DIR, what='LVF params dir')
    input_path = os.path.join(input_dir, filename)
    if not os.path.exists(input_path):
        logger.warning(f"LVF parameters file {input_path} not found. Returning None.")
        return None
    lvf_params = np.load(input_path, allow_pickle=True).item()
    logger.info(f"Loaded LVF parameters from {input_path}")
    return lvf_params

def save_lvf_params(lvf_params, output_dir=None):
    """Save an LVF arc fit with ``np.save`` as ``lvf_params['filename']``.

    The directory resolves from ``output_dir``, then ``$SELFCAL_LVF_PARAMS_DIR``, then the
    package's ``data/lvf_params/``, and is created if missing. The ``filename`` key must be
    set, as :meth:`~selfcal.instruments.spherex.adapter.SPHERExInstrument.precompute` sets it
    to ``lvf_params_D<n>.npy``.
    """
    output_dir = resolve_path(output_dir, env_var=ENV_LVF_PARAMS_DIR,
                              default=_LVF_PARAMS_DIR, what='LVF params dir',
                              must_exist=False)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, lvf_params['filename'])
    np.save(output_path, lvf_params)
    logger.info(f"Saved LVF parameters to {output_path}")

def compute_column_adjacency(chunk_map, num_columns):
    """
    Generates adjacency pairs ONLY for vertical strip transitions, 
    ignoring spectral arc transitions.
    
    Parameters
    ----------
    chunk_map : np.ndarray
        The stripped chunk-ID map. IDs follow chunk = subchannel * num_columns
        + column ("column" = vertical strip index within a subchannel, NOT the
        detector band).
    num_columns : int
        Number of column subdivisions per subchannel, i.e. the num_columns
        value chunk_map was built with (see make_stripped_chunk_map).
    """
    logger.info("Computing Vertical Strip Adjacency (Filtering Arcs)...")

    # 1. Get ALL horizontal transitions (Arc + Strip boundaries)
    # Compare pixel i with i+1
    mask = (chunk_map[:, :-1] != -1) & \
           (chunk_map[:, 1:] != -1) & \
           (chunk_map[:, :-1] != chunk_map[:, 1:])
           
    u = chunk_map[:, :-1][mask]
    v = chunk_map[:, 1:][mask]
    
    # 2. Decompose: chunk_id = subchannel * num_columns + column
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
    
    logger.info(f"Found {len(unique_pairs)} vertical strip boundaries.")
    return unique_pairs[:, 0], unique_pairs[:, 1]

def compute_subchannel_adjacency(chunk_map, num_columns):
    """
    Generates adjacency pairs for vertical subchannel transitions.
    This links chunk IDs across the boundaries of subchannels, keeping within the same column.
    """
    logger.info("Computing Vertical Subchannel Adjacency...")

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
        logger.info("Found 0 vertical subchannel boundaries.")
        return np.array([]), np.array([])
        
    pairs = np.sort(np.stack([u_filtered, v_filtered], axis=1), axis=1)
    unique_pairs = np.unique(pairs, axis=0)
    
    logger.info(f"Found {len(unique_pairs)} vertical subchannel boundaries.")
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
    0       2  ``[1, -1]`` (the pairwise equal-offset constraint, equivalent to the adjacency pairs produced by ``compute_column_adjacency``)
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
    """Build a SPHEREx detector's stripped chunk map: LVF subchannels cut into columns.

    Reads the detector's ``BC`` map (:func:`load_calibration`), labels its subchannels with
    :func:`make_fiducial_chunk_map` (channel table resolved as in
    :func:`extract_spherex_channel_edges`) and cuts the image into
    ``num_columns`` vertical strips at ``x_edges``. Chunk ``subchannel * num_columns + column``
    holds the pixels of that subchannel and column; subchannels ``0`` and
    ``num_channels * num_subchannels + 1`` lie beyond the band's first and last edges, and
    column ``0`` starts at ``x = 0``.

    Parameters
    ----------
    detector : int
        SPHEREx detector, which is also its band number (1 to 6).
    num_subchannels : int
        Subchannels per channel.
    num_channels : int
        Channels per band, a multiple of 17.
    oversample_factor : int
        Pixels of the map per detector pixel, on each axis.
    num_columns : int
        Vertical strips per subchannel.
    lvf_params : dict or None
        Arc fit (:func:`load_lvf_params`); ``None`` fits it to the ``BC`` map.
    calibration_dir : str or None
        Directory of the ``BC`` / ``BW`` files (see :func:`load_calibration`).

    Returns
    -------
    chunk_map : ndarray of int32
        The chunk ids, ``oversample_factor`` times the detector's shape.
    lvf_params : dict
        The arc fit used.
    r_edges : ndarray
        Radii, in detector pixels, of the ``num_channels * num_subchannels + 1`` subchannel
        edges.
    x_edges : ndarray
        The ``num_columns + 1`` column edges, ``linspace(0, width, num_columns + 1)``, in
        pixels of ``chunk_map``.
    """
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
    """Return a mask over a stripped chunk map's ids that selects whole subchannels.

    ``ch`` (a sequence of channel numbers counted from 1) selects their subchannels, widened
    by ``subchannel_padding`` on each side (:func:`make_fiducial_chunk_mask`), as a float 0/1
    mask. Otherwise ``subch`` (indices over all ``num_subchannels * num_channels + 2``
    subchannels) selects those subchannels as a boolean mask, ignoring
    ``subchannel_padding``. Every column of a selected subchannel is selected: entry
    ``subchannel * num_columns + column`` takes the subchannel's value, for
    ``(num_subchannels * num_channels + 2) * num_columns`` entries. Raises ``ValueError``
    when both ``ch`` and ``subch`` are ``None``.
    """
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

def _stripped_offset_spline(chunk_offset, chunk_valid_mask, r_edges, x_edges, tot_subchannels,
                            num_columns, fill_invalid):
    """The mean-preserving (arc radius, x) spline of one frame's chunk offsets."""
    reshaped_offset = chunk_offset.reshape(tot_subchannels, num_columns)[1:-1]
    reshaped_valid_mask = chunk_valid_mask.reshape(tot_subchannels, num_columns)[1:-1]

    y_slice, x_slice = get_valid_bounds(~reshaped_valid_mask.astype(bool))

    trimmed_offset = reshaped_offset[y_slice, x_slice]
    if fill_invalid:
        trimmed_offset = fill_invalid_offsets(trimmed_offset)
    trimmed_r_edges = r_edges[y_slice.start : y_slice.stop + 1]
    trimmed_x_edges = x_edges[x_slice.start : x_slice.stop + 1]

    return mean_preserving_spline_2d(trimmed_r_edges, trimmed_x_edges, trimmed_offset, x_degree=3, y_degree=3)


def make_spherex_stripped_offset_map(chunk_map, chunk_offset, chunk_valid_mask, lvf_params, r_edges, x_edges, tot_subchannels, num_columns, fill_invalid=False, needed=None):
    """Render one frame's chunk offsets as a smooth detector-grid offset map.

    ``needed`` (flat indices into the grid) evaluates the spline only at those
    pixels and returns them as a 1-D array in that order.  The spline is
    evaluated pointwise, and the pixel coordinates are the same ``arange``
    values the full mesh is built from, so the subset is bit-identical to
    ``full.ravel()[needed]`` — the mosaic uses it because a single-channel
    frame samples ~3 % of the grid.
    """
    spl = _stripped_offset_spline(chunk_offset, chunk_valid_mask, r_edges, x_edges,
                                  tot_subchannels, num_columns, fill_invalid)

    xc, yc = lvf_params['xc'], lvf_params['yc']

    h, w = np.shape(chunk_map)
    oversample_factor = h // 2040
    subpixel_shift = 0.5 / oversample_factor
    det_size = 2040
    increment = 1 / oversample_factor
    axis = np.arange(subpixel_shift, det_size+subpixel_shift, increment)
    if needed is not None:
        ii, jj = np.divmod(np.asarray(needed), w)
        x_pts = axis[jj]
        y_pts = axis[ii]
        r_pts = np.sqrt((y_pts - yc)**2 + (x_pts - xc)**2)
        return spl(r_pts, x_pts)
    x_mesh, y_mesh = np.meshgrid(axis, axis)
    r_mesh = np.sqrt((y_mesh - yc)**2 + (x_mesh - xc)**2)
    
    offset_map = spl(r_mesh, x_mesh)
    return offset_map

def fast_vertical_dist(arr):
    """Return each pixel's distance in rows to the nearest zero in its column.

    Zero pixels get 0, and so do the first and last rows, which count as zeros; a nonzero
    pixel next to a zero gets 1. Returns a float32 array of ``arr``'s shape. The SPHEREx
    instrument divides it by its maximum to taper a job's mosaic weight toward the edges of
    its subchannels.
    """
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