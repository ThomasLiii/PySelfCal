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


def save_to_fits(wcs, shape, filename):
    output_dir = os.path.dirname(filename)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    header = wcs.to_header()
    # header['NAXIS'] = 2
    # header['NAXIS1'] = shape[1]
    # header['NAXIS2'] = shape[0]
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