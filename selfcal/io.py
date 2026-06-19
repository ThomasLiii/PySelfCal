"""HDF5 I/O for reprojected exposure files."""

import os
import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from . import _state


def load_reproj_file(file_path, fields):
    """Helper to load selected fields from a single HDF5 file.
    Parameters
    ----------
    file_path : str
        Path to a reprojected HDF5 file
    fields: tup
        List of strings corresponding to name of dataset to extract from the HDF5 file
        Available fields: ['sub_data', 'sub_header', 'det_header', 'ref_coords', 'sub_foot', 'file_path',
        'sub_bitmask', 'sub_mapping']

    Returns
    -------
    data : dict
        Dictionary containing the extracted data, key is the fields and value is the corresponding datas
    """

    assert isinstance(file_path, str) and os.path.isfile(file_path), "file_path must be a valid file path"
    assert isinstance(fields, (list, tuple)), "fields must be a list or tuple of strings"

    data = {}
    is_file_missing = False

    sem = _state._hdd_io_semaphore
    if sem is not None:
        sem.acquire()
    try:
        # swmr=True allows reading while the file is being written (if supported),
        # libver='latest' supports the newer layout used in creation.
        with h5py.File(file_path, 'r', libver='latest', swmr=True) as file:

            for key in fields:
                # --- CASE 1: WCS Objects (Derived from Header Attributes) ---
                if key in ('sub_wcs', 'det_wcs'):
                    attr_key = 'sub_header' if key == 'sub_wcs' else 'det_header'
                    # Retrieve from attributes
                    if attr_key in file.attrs:
                        header_val = file.attrs[attr_key]
                        # Attributes often come out as bytes if encoded during write
                        if isinstance(header_val, bytes):
                            header_val = header_val.decode('utf-8')
                        data[key] = WCS(fits.Header.fromstring(header_val))
                    else:
                        data[key] = None # Handle missing header gracefully

                # --- CASE 2: Attributes (Metadata: headers, coords, paths) ---
                elif key in file.attrs:
                    val = file.attrs[key]
                    # Decode bytes to string if necessary (e.g., for file_path or headers)
                    if isinstance(val, bytes):
                        val = val.decode('utf-8')
                    data[key] = val

                # --- CASE 3: Datasets (Heavy Data: sub_data, sub_bitmask, etc.) ---
                elif key in file:
                    data[key] = file[key][()] # Load dataset into memory

                # --- CASE 4: Key not found ---
                else:
                    # Fallback for backward compatibility or missing keys
                    data[key] = None

        # Parse indices from filename
        det_idx = int(os.path.basename(file_path).replace('.h5', '').split('_')[-1])
        exp_idx = int(os.path.basename(file_path).replace('.h5', '').split('_')[-3])
        data['det_idx'] = det_idx
        data['exp_idx'] = exp_idx
    except Exception as e:
        print(f"Error loading {file_path}: {e}. Will use placeholders.")
        is_file_missing = True
        for key in fields:
            data[key] = None
        det_idx = None
        exp_idx = None
    finally:
        if sem is not None:
            sem.release()

    data['_is_missing_'] = is_file_missing
    return data
