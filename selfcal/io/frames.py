"""Exposures in, frames out — the two data contracts of the pipeline.

**Reading raw exposures.** The reprojection stage reads each detector of each
raw exposure through a *reader*: a function
``reader(path, sci_ext, dq_ext, header_only=False) -> ExposureData``. The
default, :func:`read_fits_exposure`, reads a FITS science extension with a
celestial WCS and an optional integer data-quality extension. An instrument
whose raw data look different — another file format, an external pointing
solution, a data cube whose slices are frames, a variance or wavelength plane
per exposure, detectors placed on a common focal plane, windowed or binned
read-outs — supplies its own reader (``ExposureLayout.reader``); the
reprojection never changes. ``sci_ext`` / ``dq_ext`` are whatever the
instrument's layout lists (extension numbers, slice indices, detector names):
the reader interprets them.

**The frame file.** Every reprojected frame is one HDF5 file (the solver's,
the mosaic's and the N-pass's only input): the frame's values on a box of the
reference grid (``sub_data``), where the box sits (``ref_coords =
[y0, y1, x0, x1]``), the detector position every box pixel came from
(``sub_mapping``, ``(2, H, W)`` = x, y), an optional bit mask
(``sub_bitmask``), optional per-observation planes (``layers/<name>`` — the
source of *layer* data variables) and the exposure's header (``det_header``:
its keywords are the source of *header* frame variables). Data that do not
come from a WCS imager at all can be written directly with
:func:`write_frame` and calibrated like any other frame.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import h5py
import hdf5plugin
import numpy as np
from astropy.io import fits

from .reproj import reproj_basename

__all__ = ['ExposureData', 'read_fits_exposure', 'read_exposure', 'write_frame', 'frame_header_values',
           'standard_frame_path']


@dataclass
class ExposureData:
    """One detector frame of a raw exposure.

    ``header``: an ``astropy.io.fits.Header`` carrying the celestial WCS of the
    frame and any metadata keywords (time, filter, angle, ... — stored with the
    frame and readable as frame variables). ``data``: ``(H, W)`` values
    (``None`` when only the header was asked for; then ``shape`` gives
    ``(H, W)``). ``mask``: ``(H, W)`` integer bit mask (a set bit flags the
    pixel; ``None`` = every pixel valid). ``layers``: ``{name: (H, W) plane}``
    carried into the frame file as ``layers/<name>``. ``coords``: ``(x, y)``
    detector coordinates of every pixel (two ``(H, W)`` arrays) — the
    coordinate system the chunk maps and detector variables live in (a focal
    plane, the full detector of a windowed read-out); ``None`` = the pixel
    indices."""
    header: fits.Header
    data: np.ndarray | None = None
    mask: np.ndarray | None = None
    layers: dict = field(default_factory=dict)
    coords: tuple | None = None
    shape: tuple | None = None

    @property
    def frame_shape(self) -> tuple:
        if self.data is not None:
            return tuple(np.shape(self.data)[-2:])
        if self.shape is not None:
            return tuple(self.shape)
        return (int(self.header['NAXIS2']), int(self.header['NAXIS1']))


def read_fits_exposure(path, sci_ext, dq_ext=None, header_only=False) -> ExposureData:
    """The default reader: FITS extension ``sci_ext`` (values + WCS header) and,
    unless ``dq_ext`` is None, the integer mask in extension ``dq_ext``."""
    with fits.open(path) as hdul:
        hdu = hdul[sci_ext]
        header = hdu.header.copy()
        if header_only:
            return ExposureData(header=header, shape=tuple(hdu.shape[-2:]) if hdu.shape else None)
        data = hdu.data
        mask = hdul[dq_ext].data if dq_ext is not None else None
    return ExposureData(header=header, data=data, mask=mask)


def read_exposure(reader, path, sci_ext, dq_ext=None, header_only=False) -> ExposureData:
    """Call ``reader`` (None = :func:`read_fits_exposure`) and check what it returned."""
    fn = read_fits_exposure if reader is None else reader
    exp = fn(path, sci_ext, dq_ext, header_only=header_only)
    if not isinstance(exp, ExposureData):
        raise TypeError(f"exposure reader {fn!r} returned {type(exp).__name__}, expected ExposureData")
    return exp


def write_frame(path, sub_data, ref_coords, sub_mapping, *, bitmask=None, layers=None,
                header=None, sub_header=None, source='') -> str:
    """Write one frame file (the solver's input format; see the module docstring).

    ``path``: the output file — a directory is completed with the standard
    ``exp_<exposure>_det_<detector>.h5`` name only by callers that know the
    indices (:func:`selfcal.io.reproj.reproj_basename`); the solver parses the
    exposure and detector index from that name. ``sub_data``: ``(H, W)`` values
    on the box ``ref_coords = [y0, y1, x0, x1]`` of the reference grid (NaN =
    no data). ``sub_mapping``: ``(2, H, W)`` detector x, y of each box pixel
    (NaN outside the detector). ``bitmask``: ``(H, W)`` integer bit mask or
    None. ``layers``: ``{name: (H, W)}`` per-observation planes. ``header``:
    the frame's header (an ``astropy.io.fits.Header`` or a header string; its
    keywords become readable frame variables). Written atomically."""
    sub_data = np.asarray(sub_data, dtype=np.float32)
    H, W = sub_data.shape
    y0, y1, x0, x1 = (int(v) for v in ref_coords)
    if (y1 - y0, x1 - x0) != (H, W):
        raise ValueError(f"ref_coords {list(ref_coords)} span {(y1 - y0, x1 - x0)}, sub_data is {(H, W)}")
    sub_mapping = np.asarray(sub_mapping, dtype=np.float32)
    if sub_mapping.shape != (2, H, W):
        raise ValueError(f"sub_mapping must be (2, {H}, {W}), got {sub_mapping.shape}")
    comp = {**hdf5plugin.Zstd(clevel=5), 'shuffle': True, 'track_times': False}
    tmp = f'{path}.tmp.{os.getpid()}'
    with h5py.File(tmp, 'w', libver='latest') as hf:
        hf.create_dataset('sub_data', data=sub_data, chunks=sub_data.shape, **comp)
        if bitmask is None:
            bitmask = np.zeros((H, W), dtype=np.int32)
        hf.create_dataset('sub_bitmask', data=np.asarray(bitmask, dtype=np.int32), chunks=(H, W), **comp)
        hf.create_dataset('sub_mapping', data=sub_mapping, chunks=sub_mapping.shape, **comp)
        for name, plane in (layers or {}).items():
            plane = np.asarray(plane, dtype=np.float32)
            if plane.shape != (H, W):
                raise ValueError(f"layer {name!r} has shape {plane.shape}, expected {(H, W)}")
            hf.create_dataset(f'layers/{name}', data=plane, chunks=plane.shape, **comp)
        if header is not None:
            hs = header.tostring() if hasattr(header, 'tostring') else str(header)
            hf.attrs['det_header'] = hs.encode('utf-8')
        if sub_header is not None:
            hs = sub_header.tostring() if hasattr(sub_header, 'tostring') else str(sub_header)
            hf.attrs['sub_header'] = hs.encode('utf-8')
        hf.attrs['file_path'] = str(source)
        hf.attrs['ref_coords'] = np.array([y0, y1, x0, x1], dtype=np.int32)
    os.replace(tmp, path)
    return path


def _header_values(path, keys):
    with h5py.File(path, 'r') as f:
        h = f.attrs.get('det_header', b'')
    hdr = fits.Header.fromstring(h.decode() if isinstance(h, bytes) else str(h))
    return [hdr.get(k) for k in keys]


def frame_header_values(frames, keys, max_workers=8) -> dict:
    """``{key: (n_frames,) array}`` of header keywords stored with each frame
    (numbers as float, anything else as object; a missing keyword is NaN / None)."""
    keys = list(keys)
    with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
        rows = list(ex.map(lambda p: _header_values(p, keys), frames))
    out = {}
    for j, k in enumerate(keys):
        vals = [r[j] for r in rows]
        if all(v is None or isinstance(v, (int, float, np.number)) and not isinstance(v, bool) for v in vals):
            out[k] = np.array([np.nan if v is None else float(v) for v in vals], dtype=np.float64)
        else:
            out[k] = np.array(vals, dtype=object)
    return out


def standard_frame_path(directory, exp_idx, det_idx) -> str:
    """``directory/exp_<exposure>_det_<detector>.h5`` — the name the solver parses."""
    return os.path.join(directory, reproj_basename(exp_idx, det_idx))
