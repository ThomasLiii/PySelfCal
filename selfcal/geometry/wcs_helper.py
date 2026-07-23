import glob
import os
import h5py
from tqdm import tqdm
import numpy as np

from astropy.io import fits
from astropy.io.votable import parse_single_table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u
from reproject.mosaicking import find_optimal_celestial_wcs

def _load_det_wcs(fits_files, use_ext):
    wcs_list = []
    files_to_process = fits_files

    for file_path in tqdm(files_to_process, desc='Loading corner WCS'):
        try:
            with fits.open(file_path) as hdul:
                for ext_idx in use_ext:
                    wcs_list.append(WCS(hdul[ext_idx].header))
        except Exception as e:
            print(f'Warning: Could not process {file_path}: {e}')
    if not wcs_list:
        raise ValueError('No WCS objects could be loaded. Check FITS files and extensions.')
    return wcs_list


def _pad_wcs(wcs, shape, padding_pixels):
    new_wcs = wcs.deepcopy()
    new_wcs.wcs.crpix[0] += padding_pixels
    new_wcs.wcs.crpix[1] += padding_pixels
    ny, nx = shape
    new_shape = (int(ny + 2 * padding_pixels), int(nx + 2 * padding_pixels))
    return new_wcs, new_shape


def find_optimal_frame(exposure_list, resolution_arcsec, padding_pixels=100, use_ext = [1, 10, 37, 46]):
    if not exposure_list:
        raise ValueError('No exposure files provided to define WCS.')
    print('Defining optimal celestial WCS...')
    wcs_list = _load_det_wcs(exposure_list, use_ext)
    ref_wcs, ref_shape = find_optimal_celestial_wcs(wcs_list, resolution=resolution_arcsec * u.arcsec, auto_rotate=False)
    ref_wcs, ref_shape = _pad_wcs(ref_wcs, ref_shape, padding_pixels)
    return ref_wcs, ref_shape


# CTYPE/CUNIT keys are strings, CDELT/CRVAL/PC/CD are numeric. Projection
# equality means: same CTYPE/CUNIT, and CRVAL/CDELT/PC/CD agree to within
# this tolerance. CRPIX deliberately excluded — it shifts when we resize.
_PROJ_RTOL = 1e-9


def projection_signature(wcs):
    """Tuple uniquely identifying the projection (CRVAL/CDELT/CTYPE/PC) of a
    WCS, excluding CRPIX/shape. Used to test whether two WCSes share a
    projection and only differ in pixel-grid origin/size."""
    w = wcs.wcs
    cd = w.cd.tolist() if w.has_cd() else None
    pc = w.pc.tolist() if w.has_pc() else None
    return (
        tuple(str(c) for c in w.ctype),
        tuple(str(u_) for u_ in w.cunit),
        tuple(float(v) for v in w.crval),
        tuple(float(v) for v in w.cdelt),
        cd,
        pc,
    )


def projections_match(wcs_a, wcs_b, rtol=_PROJ_RTOL):
    """True iff wcs_a and wcs_b have the same projection (same CTYPE/CUNIT
    exactly; CRVAL/CDELT/PC/CD numerically close). CRPIX is ignored."""
    sa, sb = projection_signature(wcs_a), projection_signature(wcs_b)
    if sa[0] != sb[0] or sa[1] != sb[1]:
        return False
    def _close(a, b):
        if a is None and b is None:
            return True
        if a is None or b is None:
            return False
        return np.allclose(np.asarray(a), np.asarray(b), rtol=rtol, atol=0.0)
    return all(_close(x, y) for x, y in zip(sa[2:], sb[2:]))


def derive_reference_from(source_ref_path, exposure_list, padding_pixels=100,
                          use_ext=(1,)):
    """Build a new ref WCS that shares the projection of ``source_ref_path``
    but is sized + recentered to contain ``exposure_list``.

    Projects each exposure extension's four detector corners through the
    source WCS to find the smallest axis-aligned pixel bounding box in the
    source projection, pads, and returns a new WCS identical to source
    (same CTYPE/CRVAL/CDELT/PC) except for shifted CRPIX. Pixel scale and
    projection are bit-for-bit shared so reprojected files from the two
    runs are directly comparable / co-registrable.

    Parameters
    ----------
    source_ref_path : str
        Path to an existing reference FITS to take the projection from.
    exposure_list : list[str]
        FITS exposure paths defining the new run's footprint.
    padding_pixels : int
        Extra pixels added to all four sides of the bbox.
    use_ext : iterable[int]
        Extensions to read from each exposure (defaults to first sci ext).
    """
    if not exposure_list:
        raise ValueError('No exposure files provided to derive reference from.')
    source_wcs, source_shape = load_from_fits(source_ref_path)
    print(f'Deriving reference from {source_ref_path} '
          f'(source shape {source_shape})')
    wcs_list = _load_det_wcs(exposure_list, use_ext)

    xs, ys = [], []
    for det_wcs in wcs_list:
        ny, nx = (int(det_wcs.array_shape[0]), int(det_wcs.array_shape[1])) \
            if det_wcs.array_shape is not None else (int(det_wcs.pixel_shape[1]),
                                                    int(det_wcs.pixel_shape[0]))
        # 0-indexed corner pixel coords
        corners_x = [0, nx - 1, 0, nx - 1]
        corners_y = [0, 0, ny - 1, ny - 1]
        sky = det_wcs.pixel_to_world(corners_x, corners_y)
        rx, ry = source_wcs.world_to_pixel(sky)
        xs.extend(np.atleast_1d(rx).tolist())
        ys.extend(np.atleast_1d(ry).tolist())

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        n_bad = int(np.sum(~np.isfinite(xs)) + np.sum(~np.isfinite(ys)))
        raise ValueError(
            f'{n_bad} non-finite corner projections; exposures may fall '
            f'outside the source projection.')

    x_min = int(np.floor(xs.min())) - int(padding_pixels)
    x_max = int(np.ceil(xs.max())) + int(padding_pixels)
    y_min = int(np.floor(ys.min())) - int(padding_pixels)
    y_max = int(np.ceil(ys.max())) + int(padding_pixels)

    new_shape = (y_max - y_min + 1, x_max - x_min + 1)
    new_wcs = source_wcs.deepcopy()
    # crpix is 1-indexed; subtracting x_min (0-indexed bbox origin) makes the
    # new array's pixel (0,0) coincide with the source array's pixel (x_min,y_min).
    new_wcs.wcs.crpix[0] -= x_min
    new_wcs.wcs.crpix[1] -= y_min
    print(f'Derived reference shape {new_shape} (bbox in source: '
          f'x=[{x_min}, {x_max}], y=[{y_min}, {y_max}])')
    return new_wcs, new_shape


def save_to_fits(wcs, shape, filename):
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    header = wcs.to_header()
    # NAXIS/NAXIS1/NAXIS2 are set automatically by PrimaryHDU from the data shape.
    hdu_0 = fits.PrimaryHDU(header=header, data=np.zeros(shape))
    hdul = fits.HDUList([hdu_0])
    hdul.writeto(filename, overwrite=True)
    print(f'Reference frame FITS saved to: {filename}')


def load_from_fits(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f'Reference WCS file not found: {file_path}')
    print(f'Loading reference frame from: {file_path}')
    ref_header = fits.open(file_path)[0].header
    ref_wcs = WCS(ref_header)
    ref_shape = (ref_header['NAXIS2'], ref_header['NAXIS1'])
    return ref_wcs, ref_shape

def upscale_wcs(wcs, factor):
    new_wcs = wcs.deepcopy()
    
    # Scale down pixel size
    if new_wcs.wcs.has_cd():
        new_wcs.wcs.cd /= factor
    elif new_wcs.wcs.has_pc():
        new_wcs.wcs.cdelt /= factor
    else:
        new_wcs.wcs.cdelt /= factor

    # Shift reference pixel to preserve alignment
    new_wcs.wcs.crpix *= factor

    return new_wcs