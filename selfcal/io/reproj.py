"""HDF5 I/O for reprojected exposure files."""

import logging
import os
import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .. import _state

logger = logging.getLogger(__name__)

try:  # optional fast decode of shuffle+zstd datasets (numcodecs ships with zarr)
    from numcodecs import Zstd as _Zstd
    from numcodecs.shuffle import Shuffle as _Shuffle
except ImportError:  # pragma: no cover - fallback is the plain h5py read
    _Zstd = _Shuffle = None

_H5_FILTER_SHUFFLE = 2
_H5_FILTER_ZSTD = 32015


def _read_dataset(ds):
    """``ds[()]``, decoded outside HDF5 when that is cheaper.

    The reprojected frames store each dataset as ONE chunk filtered by
    shuffle + Zstandard.  HDF5's read path (filter pipeline into a chunk
    buffer, then a copy out) costs ~1.5x the raw decode; for such datasets
    the chunk is read as-is and decoded with numcodecs' zstd + C unshuffle
    into the output array.  The bytes are the same either way — the codecs
    are lossless — and any dataset that is not exactly this layout falls back
    to the h5py read.
    """
    if _Zstd is None or ds.chunks is None or ds.chunks != ds.shape or ds.ndim == 0:
        return ds[()]
    try:
        plist = ds.id.get_create_plist()
        filters = [plist.get_filter(i) for i in range(plist.get_nfilters())]
        if ([f[0] for f in filters] != [_H5_FILTER_SHUFFLE, _H5_FILTER_ZSTD]
                or tuple(filters[0][2]) != (ds.dtype.itemsize,)
                or ds.id.get_num_chunks() != 1):
            return ds[()]
        _, raw = ds.id.read_direct_chunk((0,) * ds.ndim)
    except Exception:  # any HDF5 API surprise -> the standard path
        return ds[()]
    out = np.empty(ds.shape, dtype=ds.dtype)
    _Shuffle(elementsize=ds.dtype.itemsize).decode(_Zstd().decode(raw), out=out.reshape(-1).view(np.uint8))
    return out


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

    if not (isinstance(file_path, str) and os.path.isfile(file_path)):
        raise ValueError("file_path must be a valid file path")
    if not isinstance(fields, (list, tuple)):
        raise TypeError("fields must be a list or tuple of strings")

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
                    data[key] = _read_dataset(file[key]) # Load dataset into memory

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
        # load_reproj_file runs inside _prep_subframe, which executes in
        # multiprocessing children (_prep_lsqr in core/assembly, the coadd
        # workers in core/coadd). Those children have no configured logging
        # handlers, so keep print() — a logger call would silently swallow
        # this fallback report.
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
