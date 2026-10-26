"""Batch reprojection of detector frames onto a common reference WCS."""

import logging
import os
import shutil

import h5py
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales
from reproject import reproject_adaptive, reproject_exact, reproject_interp
from tqdm import tqdm

from .. import _state
from ..core.shmbuf import worker_pool_context
from ..geometry.map_helper import bit_to_bool, bool_to_bit
from .frames import frame_compression, read_exposure
from .reproj import reproj_basename

logger = logging.getLogger(__name__)


def _result(task_params, output_file, success, error=None):
    """Build the per-task result dict consumed by batch_reproject's parent."""
    return {
        'success': success,
        'output_file': output_file,
        'exp_idx': task_params['exp_idx'],
        'det_idx': task_params['det_idx'],
        'input_fits': task_params['file_path'],
        'sci_ext': task_params['sci_ext'],
        'dq_ext': task_params['dq_ext'],
        'error': error,
    }


#: A reprojected bit plane flags an output pixel where its bilinear value exceeds this.
MASK_THRESHOLD = 0.01


def pack_mask(sub_bits) -> np.ndarray:
    """The frame's bit mask (``sub_bitmask``) from the bilinearly reprojected bit planes
    ``sub_bits`` (``(32, H, W)`` floats): a bit is set where its plane exceeds
    :data:`MASK_THRESHOLD`."""
    return bool_to_bit(np.asarray(sub_bits) > MASK_THRESHOLD)


def reproject_mask(det_bitmask, det_wcs, sub_wcs, shape_out, bits=None) -> np.ndarray:
    """A detector bit mask on a frame's box, as the reprojection writes ``sub_bitmask``: every
    bit plane interpolated bilinearly like the coordinates, then packed by :func:`pack_mask`.
    ``bits``: only these bit planes are interpolated (the result has no other bit set), for a
    refresh that changes some bits only."""
    expanded = bit_to_bool(np.asarray(det_bitmask), expand_bits=True)
    if bits is None:
        sub, _ = reproject_interp((np.asarray(expanded, dtype=np.float64), det_wcs), sub_wcs,
                                  shape_out=tuple(shape_out), order='bilinear')
        return pack_mask(sub)
    sel = sorted({int(b) for b in bits})
    sub, _ = reproject_interp((np.asarray(expanded[sel], dtype=np.float64), det_wcs), sub_wcs,
                              shape_out=tuple(shape_out), order='bilinear')
    full = np.zeros((expanded.shape[0],) + tuple(shape_out), dtype=bool)
    full[sel] = np.asarray(sub) > MASK_THRESHOLD
    return bool_to_bit(full)


def reproject_mask_at(det_bitmask, coords, bits=None) -> np.ndarray:
    """A detector bit mask on a frame's box, sampled at the detector positions the frame stores
    (``coords``: its ``sub_mapping``, ``(2, H, W)`` detector x and y, NaN where none) instead of
    mapping the box through the WCS again: every bit plane interpolated bilinearly at those
    positions with the interpolation :func:`reproject_mask` uses, then packed by the same rule.
    The stored positions are the detector pixel grid interpolated like the planes, kept in
    float32: a pixel whose interpolated plane lies within that rounding of
    :data:`MASK_THRESHOLD` can come out the other way (12 of 9.8 M pixels on a SPHEREx frame).
    Valid only for frames whose ``sub_mapping`` is the detector pixel grid (a reader without its
    own detector coordinates). ``bits``: only these bit planes (the result has no other bit set)."""
    from reproject.array_utils import map_coordinates
    det = np.asarray(det_bitmask).astype(np.int64) & 0xFFFFFFFF
    shape = tuple(np.shape(coords)[1:])
    pts = np.array([np.asarray(coords[1], dtype=np.float64).ravel(), np.asarray(coords[0], dtype=np.float64).ravel()])
    out = np.zeros(shape, dtype=np.uint32)        # bool_to_bit's packing, one plane at a time
    for b in (range(32) if bits is None else sorted({int(b) for b in bits})):
        plane = ((det >> b) & 1).astype(np.float64)
        if plane.any():                           # an empty plane interpolates to 0 or NaN: no bit
            v = np.asarray(map_coordinates(plane, pts, order=1, cval=np.nan, mode='constant')).reshape(shape)
            out |= (v > MASK_THRESHOLD).astype(np.uint32) << np.uint32(b)
    return out


def reproject_layers(layers, det_wcs, sub_wcs, shape_out) -> dict:
    """The per-pixel planes of an exposure (a reader's ``layers``, ``{name: (H, W)}``) on a frame's
    box: each interpolated bilinearly from the detector grid (``det_wcs``) onto the box's grid
    (``sub_wcs``, ``shape_out``), like the detector coordinates every frame stores
    (``sub_mapping``), so a plane without NaN covers exactly the pixels that have a detector
    position. Returns ``{name: (shape_out) float32}``."""
    out = {}
    for name, plane in (layers or {}).items():
        sub_plane, _ = reproject_interp((np.asarray(plane, dtype=np.float64), det_wcs), sub_wcs,
                                        shape_out=tuple(shape_out), order='bilinear')
        out[name] = sub_plane.astype(np.float32)
    return out


def _reproject_worker(task_params):
    """Individual tasks called by batch_reproject's multiprocessing instances.

    Writes its HDF5 atomically (``.tmp.<pid>`` then ``os.rename``) so a
    Ctrl-C or worker crash never leaves a half-written final file. Returns
    a per-task result dict (see ``_result``) — the parent sorts successes
    into the file list and failures into a structured log.
    """
    file_path = task_params['file_path']
    exp_idx = task_params['exp_idx']
    det_idx = task_params['det_idx']
    sci_ext = task_params['sci_ext']
    dq_ext = task_params['dq_ext']
    reproj_func = task_params['reproj_func']
    ref_wcs = task_params['ref_wcs']
    sub_width = task_params['sub_width']
    output_dir = task_params['output_dir']
    replace_existing = task_params['replace_existing']
    reproject_kwargs = task_params['reproject_kwargs']

    output_file = os.path.join(output_dir, reproj_basename(exp_idx, det_idx))
    if not replace_existing and os.path.exists(output_file):
        return _result(task_params, output_file, success=True)

    tmp_file = f'{output_file}.tmp.{os.getpid()}'
    reproj_func_dict = {'exact': reproject_exact, 'interp': reproject_interp, 'adaptive': reproject_adaptive}

    try:
        # The instrument's reader (default: FITS sci/DQ extensions) returns the
        # values, the WCS header, the mask (None: every pixel valid), extra
        # per-pixel planes and, optionally, the detector coordinates of every
        # pixel (a focal plane, a windowed read-out).
        exp = read_exposure(task_params.get('reader'), file_path, sci_ext, dq_ext)
        det_data = exp.data
        det_header = exp.header
        det_bitmask = exp.mask
        det_height, det_width = np.shape(det_data)[-2:]
        det_header_str = det_header.tostring().encode('utf-8')
        det_wcs = WCS(det_header)

        # Map detector center to world, then to reference frame pixels
        det_center = [det_data.shape[0] / 2.0, det_data.shape[1] / 2.0]
        skycoord = det_wcs.pixel_to_world(det_center[1], det_center[0])
        ref_det_center = np.array(ref_wcs.world_to_pixel(skycoord))

        # Define sub-frame boundaries in the reference frame
        ref_x_min = int(ref_det_center[0] - sub_width // 2)
        ref_x_max = ref_x_min + sub_width
        ref_y_min = int(ref_det_center[1] - sub_width // 2)
        ref_y_max = ref_y_min + sub_width

        # Create WCS for the sub-frame
        sub_wcs = ref_wcs.deepcopy()
        sub_wcs.wcs.crpix[0] -= ref_x_min # Adjust CRPIX for the sub-frame origin
        sub_wcs.wcs.crpix[1] -= ref_y_min
        sub_header_str = sub_wcs.to_header().tostring().encode('utf-8')

        # Perform reprojection
        sub_data, sub_foot = reproj_func_dict[reproj_func](
            (det_data, det_wcs),
            sub_wcs,
            shape_out=(sub_width, sub_width),
            **reproject_kwargs
        )

        # Process detector auxiliary data: the detector pixel coordinates (so
        # every reprojected pixel knows where it came from) and, when the
        # exposure carries a mask, its bit planes, all reprojected together.
        if exp.coords is not None:
            det_xmesh, det_ymesh = (np.asarray(c, dtype=np.float64) for c in exp.coords)
            if det_xmesh.shape != (det_height, det_width) or det_ymesh.shape != (det_height, det_width):
                raise ValueError(f"reader coords must be two {(det_height, det_width)} arrays")
        else:
            det_xmesh, det_ymesh = np.meshgrid(np.arange(det_width), np.arange(det_height))
        if det_bitmask is not None:
            det_expanded_mask = bit_to_bool(det_bitmask, expand_bits=True)
            det_aux = np.stack((det_xmesh, det_ymesh, *det_expanded_mask), axis=0)
        else:
            det_aux = np.stack((det_xmesh, det_ymesh), axis=0)
        sub_aux, _ = reproject_interp(
            (det_aux, det_wcs),
            sub_wcs,
            shape_out=(sub_width, sub_width),
            order='bilinear',
        )
        sub_mapping = sub_aux[0:2] # x, y
        if det_bitmask is not None:
            sub_bitmask = pack_mask(sub_aux[2:])
        else:
            sub_bitmask = np.zeros((sub_width, sub_width), dtype=np.int32)

        with h5py.File(tmp_file, 'w', libver='latest') as hf:

            # Compression: Zstd(clevel=5) + byte-shuffle — smaller files than
            # gzip at comparable speed, reducing disk I/O when the many
            # per-frame files are re-read downstream.
            comp_args = frame_compression()

            # 1. Save Data
            hf.create_dataset('sub_data', data=sub_data, dtype=np.float32, chunks=sub_data.shape, **comp_args)
            hf.create_dataset('sub_foot', data=sub_foot, dtype=np.float16, chunks=sub_foot.shape, **comp_args)
            hf.create_dataset('sub_bitmask', data=sub_bitmask, dtype=np.int32, chunks=sub_bitmask.shape, **comp_args)
            hf.create_dataset('sub_mapping', data=sub_mapping, dtype=np.float32, chunks=sub_mapping.shape, **comp_args)

            # Extra per-pixel planes of the exposure (reader layers): reprojected
            # bilinearly like the coordinates, stored as layers/<name>.
            for lname, sub_plane in reproject_layers(exp.layers, det_wcs, sub_wcs, (sub_width, sub_width)).items():
                hf.create_dataset(f'layers/{lname}', data=sub_plane, dtype=np.float32, chunks=sub_plane.shape,
                                  **comp_args)

            # 2. Save Metadata as Attributes
            hf.attrs['sub_header'] = sub_header_str
            hf.attrs['det_header'] = det_header_str
            hf.attrs['file_path'] = file_path
            hf.attrs['ref_coords'] = np.array([ref_y_min, ref_y_max, ref_x_min, ref_x_max], dtype=np.int32)

        os.replace(tmp_file, output_file)  # atomic publish on same FS
        return _result(task_params, output_file, success=True)
    except Exception as e:
        try:
            if os.path.exists(tmp_file):
                os.remove(tmp_file)
        except OSError:
            pass
        # Runs inside a multiprocessing.Pool child (see batch_reproject); worker
        # processes have no configured logging handlers, so keep print() — a
        # logger call would silently swallow this per-frame error report.
        print(f'Error processing detector {det_idx} from exposure file index {exp_idx} ({file_path}): {e}')
        return _result(task_params, output_file, success=False, error=str(e))

def batch_reproject(exposure_list, ref_wcs, ref_shape,
                    output_dir='output/', padding_percentage=0.05, num_processes=1,
                    sci_ext_list=None, dq_ext_list=None, reproj_func='interp', exp_idx_list=None, det_idx_list=None,
                    replace_existing=False, reproject_kwargs=None,
                    per_task_extensions=False, reader=None):
    """Reproject individual exposures to bounding boxes in reference frame, output sored in HDF5 files.

    Parameters
    ----------
    exposure_list : list or tup
        List of strings describing path to fits files containing the exposures
    ref : object
        An astropy.wcs.WCS object that defines the WCS of the reference (mosaic) frame
    ref_shape : list or tup
        List of shape (2, ) defining the size of the mosaic
    output_dir : str, optional
        Directory for all selfcal outputs
    padding_percentage: float, optional
        Fraction of the mosaic width to pad
    num_processes : int, optional
        Number of parallel processes to use
    sci_ext_list : list or tup
        List of integers defining the extension in the fits files containing the science data
    dq_ext_list : list or tup or None
        List of integers defining the extension in the fits files containing the data quality bitmask;
        ``None`` (or ``None`` entries) for exposures without a mask (every pixel valid)
    reproj_func : str
        Reproject function for reprojecting the science extensions
        - 'Exact': Slowest, conserves flux
        - 'Interp': Fastest, alter PSF profile, does not conserves flux
        - 'Adaptive': Faster then 'Exact', conserves flux
    exp_idx_list : list or tup, optional
        List of integers defining the exposure index in the exposure_list, if None, will use the index of the exposure_list
    det_idx_list : list or tup, optional
        List of integers defining the detector index in the exposure_list, if None, will use the index of the in fits file
    replace_existing : bool, optional
        If True, will overwrite existing files in the output directory, default is False
    reader : callable or None, optional
        The instrument's exposure reader (``selfcal.io.frames``); ``None`` reads
        FITS extensions ``sci_ext`` / ``dq_ext``.
    per_task_extensions : bool, optional
        If False (default), tasks = exposures x zip(sci_ext_list, dq_ext_list)
        (cross-product, the legacy semantics — exp_idx_list is indexed per
        exposure i, det_idx_list per extension j). If True, all of
        ``exposure_list, sci_ext_list, dq_ext_list, exp_idx_list,
        det_idx_list`` must have the SAME length and are zipped element-wise
        (one task per index). The Reprojector uses the per-task mode to
        dispatch only the pending subset of an interrupted run.

    Returns
    -------
    success_file : list[str]
        Sorted list of paths to the HDF5 files successfully written or
        already present.
    failures : list[dict]
        Per-failed-task dicts (exp_idx, det_idx, input_fits, output_file,
        sci_ext, dq_ext, error). Empty when everything succeeds.
    """

    if sci_ext_list is None:
        sci_ext_list = []
    if dq_ext_list is None:                      # no mask extension: one None per science extension
        dq_ext_list = [None] * len(sci_ext_list)
    if reproject_kwargs is None:
        reproject_kwargs = {}

    if not (isinstance(exposure_list, (list, tuple)) and exposure_list):
        raise TypeError("exposure_list must be a non-empty list or tuple")
    if not isinstance(ref_wcs, WCS):
        raise TypeError("Reference WCS must be an astropy.wcs.WCS object")
    if not (isinstance(ref_shape, (list, tuple, np.ndarray)) and len(ref_shape) == 2):
        raise TypeError("ref_shape must be a list or tuple of length 2")
    if not (isinstance(padding_percentage, float) and 0 <= padding_percentage):
        raise TypeError("padding_percentage must be a float larger than 0")
    if not (isinstance(num_processes, int) and num_processes > 0):
        raise TypeError("num_processes must be a positive integer")
    if not (sci_ext_list is None or isinstance(sci_ext_list, (list, tuple, np.ndarray))):
        raise TypeError("sci_ext_list must be a list or tuple")
    if not (dq_ext_list is None or isinstance(dq_ext_list, (list, tuple, np.ndarray))):
        raise TypeError("dq_ext_list must be a list or tuple")
    if reproj_func not in ['exact', 'interp', 'adaptive']:
        raise ValueError("reproj_func must be one of 'exact', 'interp', or 'adaptive'")
    if not (exp_idx_list is None or isinstance(exp_idx_list, (list, tuple, np.ndarray))):
        raise TypeError("exp_idx_list must be a list or tuple")
    if not (det_idx_list is None or isinstance(det_idx_list, (list, tuple, np.ndarray))):
        raise TypeError("det_idx_list must be a list or tuple")
    if type(replace_existing) is not bool:
        raise TypeError("replace_existing must be a boolean")

    os.makedirs(output_dir, exist_ok=True)
    logger.info(f'Starting batch reprojection. Output will be saved to: {output_dir}')
    # Determine sub-frame width based on a sample detector frame
    try:
        # Assuming first science extension is representative
        sci_ext_0 = sci_ext_list[0] if len(sci_ext_list) > 0 else 1
        if reader is None:
            with fits.open(exposure_list[0]) as hdul_sample:
                if sci_ext_0 >= len(hdul_sample):
                    raise ValueError(f'Sample FITS {exposure_list[0]} does not have extension {sci_ext_0}')
                det_shape_0 = hdul_sample[sci_ext_0].data.shape
                det_wcs_0 = WCS(hdul_sample[sci_ext_0].header)
        else:
            sample = read_exposure(reader, exposure_list[0], sci_ext_0, None, header_only=True)
            det_shape_0 = sample.frame_shape
            det_wcs_0 = WCS(sample.header)
    except Exception as e:
        raise ValueError(f'Could not read sample exposure {exposure_list[0]} to determine detector properties: {e}')

    ref_reso = np.abs(proj_plane_pixel_scales(ref_wcs)[0]) # Assuming square pixels
    det_reso = np.abs(proj_plane_pixel_scales(det_wcs_0)[0])

    reso_ratio = ref_reso / det_reso
    # Calculate sub_width needed to contain the diagonal of the detector frame after reprojection, plus padding
    sub_width = int(np.ceil(np.sqrt(2) * np.max(det_shape_0) / reso_ratio * (1 + 2 * padding_percentage)))

    def _make_task(file_path, sci_ext, dq_ext, exp_idx, det_idx):
        return {
            'reproj_func': reproj_func,
            'file_path': file_path,
            'exp_idx': int(exp_idx),
            'det_idx': int(det_idx),
            'sci_ext': int(sci_ext),
            'dq_ext': None if dq_ext is None else int(dq_ext),
            'ref_wcs': ref_wcs,
            'sub_width': sub_width,
            'output_dir': output_dir,
            'replace_existing': replace_existing,
            'reproject_kwargs': reproject_kwargs,
            'reader': reader,
        }

    tasks = []
    if per_task_extensions:
        n = len(exposure_list)
        for name, lst in (('sci_ext_list', sci_ext_list), ('dq_ext_list', dq_ext_list),
                         ('exp_idx_list', exp_idx_list), ('det_idx_list', det_idx_list)):
            if lst is None or len(lst) != n:
                raise ValueError(
                    f'per_task_extensions=True requires {name} length == '
                    f'len(exposure_list)={n} (got {None if lst is None else len(lst)})')
        for i in range(n):
            tasks.append(_make_task(exposure_list[i], sci_ext_list[i], dq_ext_list[i],
                                    exp_idx_list[i], det_idx_list[i]))
    else:
        for i, file_path in enumerate(exposure_list):  # cross-product (legacy)
            for j, (sci_ext, dq_ext) in enumerate(zip(sci_ext_list, dq_ext_list)):
                exp_idx = exp_idx_list[i] if exp_idx_list is not None else i
                det_idx = det_idx_list[j] if det_idx_list is not None else j
                tasks.append(_make_task(file_path, sci_ext, dq_ext, exp_idx, det_idx))

    results = []
    if num_processes > 1 and len(tasks) > 0:
        # forkserver, as every pool of the pipeline: a forked child of a process that already runs
        # threads (a notebook, a script after another action) can wait forever on a lock held at the fork
        with worker_pool_context().Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap_unordered(_reproject_worker, tasks),
                                total=len(tasks), desc='Reprojecting frames',
                                disable=not _state.progress_enabled))
    elif len(tasks) > 0:
        for task in tqdm(tasks, desc='Reprojecting frames (sequentially)',
                         disable=not _state.progress_enabled):
            results.append(_reproject_worker(task))

    # imap_unordered yields in completion order; sort for deterministic
    # downstream behavior (chunk grouping, batch IDs, etc.).
    success_file = sorted(r['output_file'] for r in results if r['success'])
    failures = [r for r in results if not r['success']]
    logger.info(f'Batch reprojection completed. {len(success_file)} frames successfully '
                f'processed out of {len(tasks)} ({len(failures)} failed).')
    return success_file, failures


def _attr_str(hf, key):
    v = hf.attrs.get(key)
    if v is None:
        raise ValueError(f"frame {hf.filename} has no attribute {key!r}")
    return v.decode('utf-8') if isinstance(v, bytes) else str(v)


def add_layers(frame_path, layers, *, replace=False) -> list[str]:
    """Add per-pixel planes of the exposure to an existing frame file as ``layers/<name>``: the
    planes its reader did not return when the frame was made (:func:`add_reader_layers` reads
    them from the source exposure). ``layers``: ``{name: (H, W) detector plane}``; each is
    reprojected as the reprojection would have (:func:`reproject_layers`, with the WCS and the box
    the frame stores), so a pass over existing frames writes what a new reprojection writes. A
    layer the file already holds is kept unless ``replace``. The frame is rewritten atomically (a
    copy with the layers added takes its place), its other datasets and attributes byte for byte.
    Returns the names written."""
    with h5py.File(frame_path, 'r') as hf:
        have = set(hf['layers'].keys()) if 'layers' in hf else set()
        det_header = fits.Header.fromstring(_attr_str(hf, 'det_header'))
        sub_header = fits.Header.fromstring(_attr_str(hf, 'sub_header'))
        shape = tuple(hf['sub_data'].shape)
    todo = {name: plane for name, plane in (layers or {}).items() if replace or name not in have}
    if not todo:
        return []
    if 'NAXIS1' in det_header and 'NAXIS2' in det_header:
        det_shape = (int(det_header['NAXIS2']), int(det_header['NAXIS1']))
        for name, plane in todo.items():
            if np.shape(plane) != det_shape:
                raise ValueError(f"layer {name!r} is {np.shape(plane)}, the frame's detector {det_shape}")
    planes = reproject_layers(todo, WCS(det_header), WCS(sub_header), shape)
    tmp = f'{frame_path}.tmp.{os.getpid()}'
    try:
        shutil.copyfile(frame_path, tmp)
        with h5py.File(tmp, 'r+', libver='latest') as hf:
            comp = frame_compression()
            for name, plane in planes.items():
                key = f'layers/{name}'
                if key in hf:
                    del hf[key]
                hf.create_dataset(key, data=plane, dtype=np.float32, chunks=plane.shape, **comp)
        os.replace(tmp, frame_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return list(planes)


def add_reader_layers(frame_path, reader, sci_ext, dq_ext=None, *, names=None, replace=False) -> list[str]:
    """Add to an existing frame the layers its reader returns now (a reader that gained layers
    after the frame was made): the source exposure (the frame's ``file_path``) is read again with
    ``reader`` (None: the default FITS reader) and ``sci_ext`` / ``dq_ext`` (the instrument's
    layout), and its layers — all of them, or ``names`` — go through :func:`add_layers`."""
    with h5py.File(frame_path, 'r') as hf:
        source = _attr_str(hf, 'file_path')
    exp = read_exposure(reader, source, sci_ext, dq_ext)
    layers = dict(exp.layers or {})
    if names is not None:
        missing = [n for n in names if n not in layers]
        if missing:
            raise ValueError(f"the reader of {source} returns no layer(s) {missing} (it returns {sorted(layers)})")
        layers = {n: layers[n] for n in names}
    return add_layers(frame_path, layers, replace=replace)


def refresh_mask(frame_path, det_bitmask, bits=None, mapping='wcs') -> int:
    """Recompute an existing frame's ``sub_bitmask`` from the detector bit mask ``det_bitmask``
    (the exposure's mask as its reader returns it now, e.g. with bits dilated:
    :func:`refresh_reader_mask` reads it), reprojected as the reprojection would
    (:func:`reproject_mask`, with the WCS and the box the frame stores). ``bits``: recompute only
    these bit planes and keep the stored mask's other bits, which is right when only those bits
    changed since the frame was made and 32 times cheaper than every plane. ``mapping``: ``'wcs'``
    maps the box through the frame's WCS again, exactly as the reprojection did; ``'stored'``
    samples the planes at the detector positions the frame stores (:func:`reproject_mask_at`:
    without the WCS mapping, most of the cost; equal up to the stored positions' float32 rounding
    at the threshold; frames whose ``sub_mapping`` is the detector pixel grid only). The frame is
    rewritten atomically, its other datasets and attributes byte for byte (copied object by
    object), when the mask changes. Returns the number of pixels whose mask changed."""
    if mapping not in ('wcs', 'stored'):
        raise ValueError(f"mapping={mapping!r}: 'wcs' or 'stored'")
    with h5py.File(frame_path, 'r') as hf:
        det_header = fits.Header.fromstring(_attr_str(hf, 'det_header'))
        sub_header = fits.Header.fromstring(_attr_str(hf, 'sub_header'))
        old = hf['sub_bitmask'][...]
        coords = hf['sub_mapping'][...] if mapping == 'stored' else None
    shape = tuple(old.shape)
    if 'NAXIS1' in det_header and 'NAXIS2' in det_header:
        det_shape = (int(det_header['NAXIS2']), int(det_header['NAXIS1']))
        if np.shape(det_bitmask) != det_shape:
            raise ValueError(f"the mask is {np.shape(det_bitmask)}, the frame's detector {det_shape}")
    if mapping == 'stored':
        new = reproject_mask_at(det_bitmask, coords, bits=bits).astype(old.dtype)
    else:
        new = reproject_mask(det_bitmask, WCS(det_header), WCS(sub_header), shape, bits=bits).astype(old.dtype)
    if bits is not None:
        others = ~sum(old.dtype.type(1 << int(b)) for b in bits)
        new = (old & others) | new
    changed = int((new != old).sum())
    if changed == 0:
        return 0
    tmp = f'{frame_path}.tmp.{os.getpid()}'
    try:
        with h5py.File(frame_path, 'r') as src, h5py.File(tmp, 'w', libver='latest') as dst:
            for name in src:
                if name != 'sub_bitmask':
                    src.copy(name, dst, name=name)
            for key, value in src.attrs.items():
                dst.attrs[key] = value
            dst.create_dataset('sub_bitmask', data=new, chunks=shape, **frame_compression())
        os.replace(tmp, frame_path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return changed


def refresh_reader_mask(frame_path, reader, sci_ext, dq_ext, bits=None, mapping='wcs') -> int:
    """Refresh an existing frame's ``sub_bitmask`` with the mask its reader returns now (a reader
    that dilates or edits bits since the frame was made): the source exposure (the frame's
    ``file_path``) is read again with ``reader`` (None: the default FITS reader) and ``sci_ext`` /
    ``dq_ext``, and its mask goes through :func:`refresh_mask` (``bits``: only those planes;
    ``mapping``: how the planes reach the frame's box). Returns the number of pixels whose mask
    changed."""
    with h5py.File(frame_path, 'r') as hf:
        source = _attr_str(hf, 'file_path')
    exp = read_exposure(reader, source, sci_ext, dq_ext)
    if exp.mask is None:
        raise ValueError(f"the reader of {source} returns no mask")
    return refresh_mask(frame_path, exp.mask, bits=bits, mapping=mapping)
