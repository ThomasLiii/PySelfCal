"""Co-addition pipeline: mean, std, sigma-clipped maps, and intermediate caching.

Design (see selfcal/README.md, "Mosaic / coadd engine"):

* Per-frame payloads are **sparse**: only the nonzero-weight pixels of a frame
  (a packed bbox mask + value vectors) travel through the intermediate cache
  and into the accumulators.  A single-channel SPHEREx frame keeps ~1 % of its
  pixels, so the cache is ~10x smaller than dense bbox crops and every pass
  reads/multiplies only what contributes.
* Each batch accumulates into lazily-zeroed full-grid local arrays (only the
  touched pages materialise) and flushes **only its union window**, **in batch
  order** through a turnstile.  The result is therefore a pure function of
  (frames, batch_size): bit-reproducible at any worker count.
* ``run_coadd_schedule`` runs the passes a mosaic needs: the cache pass also
  accumulates the mean (no separate mean pass), and the band-centre/width
  ("wav") sums are accumulated inside the sigma-clip pass from per-pixel
  values sampled once, so no extra pass and no second read of ``sub_mapping``.
* ``compute_coadd_map`` keeps the single-pass API (mode 'cache' / 'mean' /
  'std' / 'sigma_clip'); it reads both the sparse cache and the legacy dense
  one.  Per-pixel arithmetic is unchanged from the dense implementation.
"""

import logging
import os
import time
from multiprocessing import Pool, Condition, Array
from multiprocessing.shared_memory import SharedMemory

import h5py
import numpy as np
from scipy.ndimage import map_coordinates
from tqdm import tqdm

from .. import _state
from .subframe import _prep_subframe

logger = logging.getLogger(__name__)

CACHE_FORMAT = 'sparse-v1'
_COADD_MODES = ('mean', 'std', 'sigma_clip')
# Row stripes of the ordered flush (see _coadd_batch_worker): enough for the
# workers to flush concurrently, few enough that a stripe is many pages.
_FLUSH_STRIPES = int(os.environ.get('SELFCAL_COADD_FLUSH_STRIPES', '128'))


# --------------------------------------------------------------------------- shared arrays
class _SharedArrays:
    """Arrays published to the worker pool through SharedMemory (by name)."""

    def __init__(self):
        self._segs = {}
        self.meta = {}

    def put(self, name, arr):
        if arr is None:
            return
        arr = np.ascontiguousarray(arr)
        shm = SharedMemory(create=True, size=max(int(arr.nbytes), 1))
        np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[...] = arr
        self._segs[name] = shm
        self.meta[name] = (shm.name, tuple(arr.shape), arr.dtype.str)

    def zeros(self, name, shape, dtype=np.float32):
        shm = SharedMemory(create=True, size=max(int(np.prod(shape)) * np.dtype(dtype).itemsize, 1))
        np.ndarray(tuple(shape), dtype=dtype, buffer=shm.buf).fill(0)
        self._segs[name] = shm
        self.meta[name] = (shm.name, tuple(shape), np.dtype(dtype).str)

    def get(self, name):
        _, shape, dt = self.meta[name]
        return np.ndarray(shape, dtype=np.dtype(dt), buffer=self._segs[name].buf)

    def drop(self, name):
        shm = self._segs.pop(name, None)
        self.meta.pop(name, None)
        if shm is not None:
            shm.close()
            shm.unlink()

    def close(self):
        for name in list(self._segs):
            self.drop(name)


def _attach(meta, name, handles):
    """Attach a published array inside a worker (None if not published)."""
    if name not in meta:
        return None
    shm_name, shape, dt = meta[name]
    try:
        shm = SharedMemory(name=shm_name, track=False)
    except TypeError:  # Python < 3.13
        shm = SharedMemory(name=shm_name)
    handles.append(shm)
    return np.ndarray(shape, dtype=np.dtype(dt), buffer=shm.buf)


# --------------------------------------------------------------------------- sparse frames
def sparsify_frame(ref_coords, sub_data, sub_weight, sub_aux=None):
    """The nonzero-weight pixels of a prepared frame.

    Returns None when no pixel has weight, else a dict with ``ref_coords`` (the
    bbox of nonzero weight in reference-grid coordinates, the same crop the
    dense cache stored), ``sub_bbox`` (that bbox in full-subframe coordinates),
    ``shape`` (bbox shape), ``mask`` (packed bits of weight != 0 over the bbox,
    row-major), ``flat`` (their flat bbox indices), ``data`` / ``weight`` (values
    in that order) and optionally ``aux`` ``(K, n)``.
    """
    rows = np.flatnonzero(sub_weight.any(axis=1))
    if rows.size == 0:
        return None
    cols = np.flatnonzero(sub_weight.any(axis=0))
    rmin, rmax, cmin, cmax = int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    w_box = sub_weight[rmin:rmax, cmin:cmax]
    nz = w_box != 0
    flat = np.flatnonzero(nz)
    y_min, _, x_min, _ = (int(v) for v in ref_coords)
    frame = {
        'ref_coords': np.array([y_min + rmin, y_min + rmax, x_min + cmin, x_min + cmax], dtype=np.int32),
        'sub_bbox': np.array([rmin, rmax, cmin, cmax], dtype=np.int32),
        'shape': (rmax - rmin, cmax - cmin),
        'mask': np.packbits(nz),
        'flat': flat,
        'data': sub_data[rmin:rmax, cmin:cmax].ravel()[flat],
        'weight': w_box.ravel()[flat],
    }
    if sub_aux is not None:
        frame['aux'] = np.ascontiguousarray(
            sub_aux[:, rmin:rmax, cmin:cmax].reshape(sub_aux.shape[0], -1)[:, flat])
    return frame


def sample_band_maps(frame, sub_mapping, det_BC, det_BW):
    """Per-pixel band centre / width of a frame's kept pixels.

    ``map_coordinates(det_map, sub_mapping[::-1], order=1)`` on the detector
    grid, exactly as the standalone wavelength coadd samples them, evaluated
    at the kept pixels only (order-1 sampling is pointwise).
    """
    rmin, rmax, cmin, cmax = (int(v) for v in frame['sub_bbox'])
    coords = sub_mapping[:, rmin:rmax, cmin:cmax].reshape(2, -1)[:, frame['flat']][::-1]
    frame['bc'] = map_coordinates(det_BC, coords, order=1, output=np.float32)
    frame['bw'] = map_coordinates(det_BW, coords, order=1, output=np.float32)


_CACHE_KEYS = ('ref_coords', 'sub_bbox', 'mask', 'data', 'weight', 'aux', 'bc', 'bw')


def write_cached_frame(path, frame):
    with h5py.File(path, 'w') as hf:
        hf.attrs['format'] = CACHE_FORMAT
        hf.attrs['shape'] = np.asarray(frame['shape'], dtype=np.int32)
        for k in _CACHE_KEYS:
            if k in frame:
                hf.create_dataset(k, data=frame[k], track_times=False)


def read_cached_frame(path):
    """A sparse frame dict from a cache file of either format (None if empty)."""
    with h5py.File(path, 'r') as hf:
        if hf.attrs.get('format') == CACHE_FORMAT:
            frame = {k: hf[k][()] for k in _CACHE_KEYS if k in hf}
            frame['shape'] = tuple(int(v) for v in hf.attrs['shape'])
            n = frame['shape'][0] * frame['shape'][1]
            frame['flat'] = np.flatnonzero(np.unpackbits(frame['mask'], count=n))
            return frame
        # legacy dense crops (sub_data / sub_weight [/ sub_aux] + ref_coords [+ sub_bbox])
        ref_coords = hf['ref_coords'][()]
        sub_data = hf['sub_data'][()]
        sub_weight = hf['sub_weight'][()]
        sub_aux = hf['sub_aux'][()] if 'sub_aux' in hf else None
        sub_bbox = hf['sub_bbox'][()] if 'sub_bbox' in hf else None
    frame = sparsify_frame(ref_coords, sub_data, sub_weight, sub_aux)
    if frame is not None and sub_bbox is not None:
        r0, _, c0, _ = (int(v) for v in sub_bbox)
        b = frame['sub_bbox']
        frame['sub_bbox'] = np.array([r0 + b[0], r0 + b[1], c0 + b[2], c0 + b[3]], dtype=np.int32)
    return frame


def load_cached_frame_dense(path):
    """``(ref_coords, sub_data, sub_weight, sub_bbox)`` dense bbox crops from a
    cache file of either format (legacy consumers such as ``wav_coadd``)."""
    frame = read_cached_frame(path)
    if frame is None:
        z = np.zeros((0, 0), dtype=np.float32)
        return np.zeros(4, dtype=np.int32), z, z, np.zeros(4, dtype=np.int32)
    n = frame['shape'][0] * frame['shape'][1]
    data = np.zeros(n, dtype=np.float32)
    weight = np.zeros(n, dtype=np.float32)
    data[frame['flat']] = frame['data']
    weight[frame['flat']] = frame['weight']
    return (frame['ref_coords'], data.reshape(frame['shape']), weight.reshape(frame['shape']),
            frame['sub_bbox'])


def _ref_indices(frame, ref_shape):
    """Reference-grid (rows, cols) of the frame's kept pixels, plus the in-bounds
    selector to apply to the value vectors (None when all are in bounds)."""
    H, W = ref_shape
    y0, _, x0, _ = (int(v) for v in frame['ref_coords'])
    width = frame['shape'][1]
    r = y0 + frame['flat'] // width
    c = x0 + frame['flat'] % width
    keep = (r >= 0) & (r < H) & (c >= 0) & (c < W)
    if keep.all():
        return r, c, None
    return r[keep], c[keep], keep


# --------------------------------------------------------------------------- worker
def _coadd_batch_worker(task):
    """One batch: prepare (or read cached) frames, optionally write them to the
    cache, accumulate the requested statistic into local arrays, and flush the
    batch's window into the shared totals when its turn comes."""
    b = task['b']
    ref_shape = task['ref_shape']
    meta = task['shm']
    acc = task['accumulate']
    n_aux = task['n_aux']
    wav = task['wav']
    write_dir = task['write_dir']
    handles = []
    stats = {'prep': 0.0, 'io': 0.0, 'acc': 0.0, 'wait': 0.0, 'flush': 0.0, 'window': 0.0}
    cached = []
    try:
        if task['source'] == 'prep':
            chunk_maps = [_attach(meta, f'chunk_map_{m}', handles) for m in range(task['n_maps'])]
            det_aux = _attach(meta, 'det_aux', handles)
            det_BC = _attach(meta, 'det_BC', handles)
            det_BW = _attach(meta, 'det_BW', handles)
            prep = dict(task['prep'])
            prep['chunk_maps'] = chunk_maps
            prep['grid_valid_weight'] = _attach(meta, 'gvw', handles)
            prep['det_aux'] = det_aux
            offsets = task['offsets']

        if acc is not None:
            loc_d = np.zeros(ref_shape, dtype=np.float32)
            loc_w = np.zeros(ref_shape, dtype=np.float32)
            loc_aux = np.zeros((n_aux,) + ref_shape, dtype=np.float32) if n_aux else None
            loc_wav = [np.zeros(ref_shape, dtype=np.float32) for _ in range(3)] if wav else None
            win = [ref_shape[0], 0, ref_shape[1], 0]
            mean_map = _attach(meta, 'mean_map', handles) if acc in ('std', 'sigma_clip') else None
            std_map = _attach(meta, 'std_map', handles) if acc == 'sigma_clip' else None
            sigma = task['sigma']

        for j, file_path in enumerate(task['files']):
            t0 = time.perf_counter()
            if task['source'] == 'prep':
                extras = {} if (wav or write_dir is not None and det_BC is not None) else None
                ref_coords, sub_data, sub_weight, _, sub_aux = _prep_subframe(
                    file=file_path,
                    chunk_offsets=[o[j] for o in offsets] if offsets is not None else None,
                    extras=extras, **prep)
                frame = sparsify_frame(ref_coords, sub_data, sub_weight, sub_aux)
                if frame is not None and extras is not None:
                    sample_band_maps(frame, extras['sub_mapping'], det_BC, det_BW)
                del sub_data, sub_weight, sub_aux
                stats['prep'] += time.perf_counter() - t0
                if write_dir is not None and frame is not None:
                    t0 = time.perf_counter()
                    cache_path = os.path.join(write_dir, f"cached_{os.path.basename(file_path)}")
                    write_cached_frame(cache_path, frame)
                    cached.append(cache_path)
                    stats['io'] += time.perf_counter() - t0
            else:
                try:
                    frame = read_cached_frame(file_path)
                except Exception as e:
                    # Pool child: no logging handlers, keep print().
                    print(f"Error loading cached file {file_path}: {e}")
                    continue
                stats['io'] += time.perf_counter() - t0
            if acc is None or frame is None:
                continue

            t0 = time.perf_counter()
            r, c, keep = _ref_indices(frame, ref_shape)
            if r.size == 0:
                continue
            d = frame['data'] if keep is None else frame['data'][keep]
            w = frame['weight'] if keep is None else frame['weight'][keep]
            a = None
            if loc_aux is not None and 'aux' in frame:
                a = frame['aux'] if keep is None else frame['aux'][:, keep]
            if acc == 'mean':
                loc_d[r, c] += d * w
                loc_w[r, c] += w
                if a is not None:
                    for k in range(n_aux):
                        loc_aux[k][r, c] += a[k] * w
            elif acc == 'std':
                mean_v = mean_map[r, c]
                loc_d[r, c] += (d - mean_v) ** 2 * w
                loc_w[r, c] += w
                if a is not None:
                    for k in range(n_aux):
                        loc_aux[k][r, c] += (a[k] - mean_v) ** 2 * w
            else:  # sigma_clip
                mean_v = mean_map[r, c]
                std_v = std_map[r, c]
                clip_mask = np.abs(d - mean_v) <= sigma * std_v
                valid_weight = w * clip_mask
                loc_d[r, c] += d * w * clip_mask
                loc_w[r, c] += valid_weight
                if a is not None:
                    for k in range(n_aux):
                        loc_aux[k][r, c] += a[k] * valid_weight
                if loc_wav is not None:
                    bc = frame['bc'] if keep is None else frame['bc'][keep]
                    bw = frame['bw'] if keep is None else frame['bw'][keep]
                    loc_wav[0][r, c] += np.where(clip_mask, (bc * bw) * w, 0.0)
                    loc_wav[1][r, c] += np.where(clip_mask, bw * w, 0.0)
                    loc_wav[2][r, c] += np.where(clip_mask, (bw * ((bw ** 2) / 12 + bc ** 2)) * w, 0.0)
            win[0] = min(win[0], int(r.min())); win[1] = max(win[1], int(r.max()) + 1)
            win[2] = min(win[2], int(c.min())); win[3] = max(win[3], int(c.max()) + 1)
            stats['acc'] += time.perf_counter() - t0

        if acc is not None:
            tot_d = _attach(meta, 'total_data', handles)
            tot_w = _attach(meta, 'total_weight', handles)
            tot_aux = _attach(meta, 'total_aux', handles) if n_aux else None
            tot_wav = [_attach(meta, f'total_wav{i}', handles) for i in range(3)] if wav else None
            if win[1] > win[0] and win[3] > win[2]:
                stats['window'] = ((win[1] - win[0]) * (win[3] - win[2])
                                   / float(ref_shape[0] * ref_shape[1]))
            # Striped turnstile: the map rows are cut into S stripes, each with
            # its own batch counter. Batch b may add its window into stripe s
            # only once batch b-1 has finished that stripe, and it walks the
            # stripes in order, so per pixel the totals are accumulated in
            # batch order (deterministic) while different batches flush
            # different stripes concurrently (a wavefront) instead of
            # serialising on one lock.
            cond, counters = _state._coadd_turn
            n_stripes = len(counters)
            stripe_h = -(-ref_shape[0] // n_stripes)
            cols = slice(win[2], win[3])
            for st in range(n_stripes):
                t0 = time.perf_counter()
                with cond:
                    while counters[st] != b:
                        if not cond.wait(timeout=600):
                            print(f"coadd turnstile: batch {b} still waiting for batch "
                                  f"{counters[st]} on stripe {st}")
                t1 = time.perf_counter()
                y0 = max(st * stripe_h, win[0])
                y1 = min((st + 1) * stripe_h, win[1])
                if y1 > y0 and win[3] > win[2]:
                    s = np.s_[y0:y1, cols]
                    tot_d[s] += loc_d[s]
                    tot_w[s] += loc_w[s]
                    if tot_aux is not None:
                        tot_aux[(slice(None),) + s] += loc_aux[(slice(None),) + s]
                    if tot_wav is not None:
                        for i in range(3):
                            tot_wav[i][s] += loc_wav[i][s]
                t2 = time.perf_counter()
                with cond:
                    counters[st] += 1
                    cond.notify_all()
                stats['wait'] += t1 - t0
                stats['flush'] += t2 - t1
    finally:
        for shm in handles:
            shm.close()
    return cached, stats


# --------------------------------------------------------------------------- one pass
def _run_pass(*, ref_shape, files, offsets, source, accumulate, write_dir, wav, shared, n_maps, n_aux,
              prep, sigma, max_workers, batch_size, label):
    """Run one pass over ``files`` on the worker pool.  Returns (cached_list, stats)."""
    ref_shape = tuple(int(v) for v in ref_shape)
    tasks = []
    for b, start in enumerate(range(0, len(files), batch_size)):
        end = min(start + batch_size, len(files))
        tasks.append({
            'b': b, 'files': files[start:end],
            'offsets': [o[start:end] for o in offsets] if offsets is not None else None,
            'ref_shape': ref_shape, 'shm': shared.meta, 'source': source, 'accumulate': accumulate,
            'write_dir': write_dir, 'wav': wav, 'n_maps': n_maps, 'n_aux': n_aux, 'prep': prep,
            'sigma': sigma,
        })
    logger.info(f"{label}: {len(files)} files in {len(tasks)} batches on {max_workers} workers...")
    t0 = time.perf_counter()
    cached, stats = [], []
    n_stripes = max(1, min(_FLUSH_STRIPES, ref_shape[0]))
    cond, counters = Condition(), Array('i', n_stripes, lock=False)
    with Pool(processes=max_workers, initializer=_state._init_coadd_worker, initargs=(cond, counters)) as pool:
        for c, s in tqdm(pool.imap_unordered(_coadd_batch_worker, tasks), total=len(tasks),
                         disable=not _state.progress_enabled):
            cached.extend(c)
            stats.append(s)
    wall = time.perf_counter() - t0
    agg = {k: sum(s[k] for s in stats) for k in ('prep', 'io', 'acc', 'wait', 'flush')}
    window = np.mean([s['window'] for s in stats]) if stats else 0.0
    logger.info(f"{label} finished in {wall:.2f} seconds (core-s: prep {agg['prep']:.0f}, io {agg['io']:.0f}, "
                f"accumulate {agg['acc']:.0f}, turnstile wait {agg['wait']:.0f}, flush {agg['flush']:.0f}; "
                f"mean flush window {window * 100:.1f} % of the map)")
    cached.sort()
    return cached, agg


def _publish_inputs(shared, chunk_maps, grid_valid_weight, det_aux, wav_maps):
    for m, cm in enumerate(chunk_maps or []):
        shared.put(f'chunk_map_{m}', cm)
    shared.put('gvw', grid_valid_weight)
    shared.put('det_aux', np.asarray(det_aux) if det_aux is not None else None)
    if wav_maps is not None:
        shared.put('det_BC', wav_maps[0])
        shared.put('det_BW', wav_maps[1])


def _new_totals(shared, ref_shape, n_aux, wav):
    shared.zeros('total_data', ref_shape)
    shared.zeros('total_weight', ref_shape)
    if n_aux:
        shared.zeros('total_aux', (n_aux,) + tuple(ref_shape))
    if wav:
        for i in range(3):
            shared.zeros(f'total_wav{i}', ref_shape)


def _take_totals(shared, n_aux, wav):
    out = {'data': shared.get('total_data').copy(), 'weight': shared.get('total_weight').copy(),
           'aux': shared.get('total_aux').copy() if n_aux else None,
           'wav': [shared.get(f'total_wav{i}').copy() for i in range(3)] if wav else None}
    for name in ('total_data', 'total_weight', 'total_aux', 'total_wav0', 'total_wav1', 'total_wav2'):
        shared.drop(name)
    return out


def _finalize(mode, totals):
    data_sum, weight_sum, aux_sum = totals['data'], totals['weight'], totals['aux']
    if mode == 'mean' or mode == 'sigma_clip':
        result_map = np.divide(data_sum, weight_sum, out=np.zeros_like(data_sum), where=weight_sum != 0)
    else:  # std: data_sum holds squared differences
        variance = np.divide(data_sum, weight_sum, out=np.zeros_like(data_sum), where=weight_sum > 0)
        result_map = np.sqrt(variance)
    aux_map = (np.divide(aux_sum, weight_sum, out=np.zeros_like(aux_sum), where=weight_sum != 0)
               if aux_sum is not None else None)
    return result_map, weight_sum, aux_map


def _finalize_wav(wav_sums):
    BCBW_sum, BW_sum, meanvar_sum = wav_sums
    with np.errstate(divide='ignore', invalid='ignore'):
        wav_mean_map = BCBW_sum / BW_sum
        wav_std_map = np.sqrt(meanvar_sum / BW_sum - wav_mean_map ** 2)
    wav_mean_map[~np.isfinite(wav_mean_map)] = 0
    wav_std_map[~np.isfinite(wav_std_map)] = 0
    return wav_mean_map, wav_std_map


def _prep_kwargs(apply_weight, apply_mask, ignore_list, det_offset_funcs, oversample_factor, valid_threshold,
                 preprocess_func, postprocess_func):
    return {
        'apply_weight': apply_weight, 'apply_mask': apply_mask, 'ignore_list': ignore_list,
        'det_offset_funcs': det_offset_funcs, 'oversample_factor': oversample_factor,
        'valid_threshold': valid_threshold, 'for_lsqr': False,
        'preprocess_func': preprocess_func, 'postprocess_func': postprocess_func,
    }


def _validate_common(mode, ref_shape, file_list, apply_weight, apply_mask, chunk_maps, offset_lists,
                     det_offset_funcs, grid_valid_weight, max_workers, ignore_list, oversample_factor,
                     batch_size):
    if not (isinstance(ref_shape, (list, np.ndarray, tuple)) and len(ref_shape) == 2):
        raise ValueError("ref_shape must be a list or tuple of length 2")
    if not (isinstance(file_list, (list, np.ndarray)) and len(file_list)):
        raise ValueError("file_list must be a non-empty list")
    if not isinstance(apply_weight, bool):
        raise TypeError("apply_weight must be a boolean")
    if not isinstance(apply_mask, bool):
        raise TypeError("apply_mask must be a boolean")
    if not isinstance(chunk_maps, list):
        raise TypeError("chunk_maps must be a list of ndarrays")
    K = len(chunk_maps)
    if offset_lists is not None and len(offset_lists) != K:
        raise ValueError(f"offset_lists length must match chunk_maps ({K})")
    if det_offset_funcs is not None and len(det_offset_funcs) != K:
        raise ValueError(f"det_offset_funcs length must match chunk_maps ({K})")
    if not (grid_valid_weight is None or isinstance(grid_valid_weight, np.ndarray)):
        raise TypeError("grid_valid_weight must be a numpy array")
    if not (isinstance(max_workers, int) and max_workers > 0):
        raise ValueError("max_workers must be a positive integer")
    if not isinstance(ignore_list, (list, np.ndarray)):
        raise TypeError("ignore_list must be a list or array of data quality flags to ignore")
    if not (isinstance(oversample_factor, int) and oversample_factor > 0):
        raise ValueError("oversample_factor must be a positive integer")
    if not (isinstance(batch_size, int) and batch_size > 0):
        raise ValueError("batch_size must be a positive integer")


def compute_coadd_map(mode, ref_shape, file_list, mean_map=None, std_map=None, sigma=3.0,
                      offset_lists=None, apply_weight=True,
                      apply_mask=True, chunk_maps=None, grid_valid_weight=None,
                      max_workers=10, ignore_list=None, det_offset_funcs=None, oversample_factor=1,
                      batch_size=10, valid_threshold=0.99,
                      cache_dir='cache/', use_cached=False, det_aux=None,
                      preprocess_func=None, postprocess_func=None):
    """Unified mean / std / sigma-clipped-mean / cache builder, multi-chunk-map aware.

    Each per-frame subframe is offset-corrected by the sum of K per-map
    contributions ``Σ_m det_offset_funcs[m](chunk_maps[m], offset_lists[m][k])``
    before being accumulated into the requested coadd or written to the
    intermediate cache.  ``mode='cache'`` returns the sorted list of cache
    files (sparse format, see module docstring); the coadd modes return
    ``(result_map, weight_sum, aux_map)``.  With ``use_cached`` the
    ``file_list`` names cache files (either format).

    Parameters
    ----------
    chunk_maps : list of np.ndarray or None
        K chunk maps (must share shape). ``None`` or ``[]`` skips chunk-based
        offset application; callers that just want to coadd raw data leave
        all of ``chunk_maps`` / ``offset_lists`` / ``det_offset_funcs``
        unset.
    offset_lists : list of np.ndarray or None
        Length-K list of per-frame, per-chunk offset arrays
        (``(num_frames, num_chunks_m)`` each). When ``None``, no offsets are
        applied.
    det_offset_funcs : list of callable or None
        Length-K list of ``(chunk_map, chunk_offset) -> grid_offset``
        callables. ``None`` (or per-map ``None``) falls back to the standard
        ``chunk_to_det`` per map.
    """
    if ignore_list is None:
        ignore_list = []
    if mode not in ['mean', 'std', 'sigma_clip', 'cache']:
        raise ValueError("mode must be one of 'mean', 'std', 'sigma_clip', or 'cache'")
    if mode == 'cache':
        if cache_dir is None:
            raise ValueError("cache_dir must be provided when mode='cache'")
        os.makedirs(cache_dir, exist_ok=True)
    if mode == 'std' and mean_map is None:
        raise ValueError("mean_map must be provided for 'std' mode")
    if mode == 'sigma_clip':
        if mean_map is None:
            raise ValueError("mean_map must be provided for 'sigma_clip' mode")
        if std_map is None:
            raise ValueError("std_map must be provided for 'sigma_clip' mode")
        if not (isinstance(sigma, (int, float)) and sigma > 0):
            raise ValueError("sigma must be a positive number")
    if chunk_maps is None:
        chunk_maps = []
    _validate_common(mode, ref_shape, file_list, apply_weight, apply_mask, chunk_maps, offset_lists,
                     det_offset_funcs, grid_valid_weight, max_workers, ignore_list, oversample_factor,
                     batch_size)
    if use_cached and mode == 'cache':
        raise ValueError("use_cached and mode='cache' cannot both be True")
    if use_cached and not os.path.isdir(cache_dir):
        raise ValueError("cache_dir must be a valid directory when use_cached is True")

    ref_shape = tuple(int(v) for v in ref_shape)
    n_aux = 0 if det_aux is None else len(det_aux)
    shared = _SharedArrays()
    try:
        source = 'cached' if use_cached else 'prep'
        if not use_cached:
            _publish_inputs(shared, chunk_maps, grid_valid_weight, det_aux, None)
        common = dict(ref_shape=ref_shape, files=list(file_list),
                      offsets=None if use_cached else offset_lists, source=source, wav=False,
                      shared=shared, n_maps=len(chunk_maps), n_aux=n_aux,
                      prep=_prep_kwargs(apply_weight, apply_mask, ignore_list, det_offset_funcs,
                                        oversample_factor, valid_threshold, preprocess_func, postprocess_func),
                      sigma=sigma, max_workers=max_workers, batch_size=batch_size)
        if mode == 'cache':
            cached, _ = _run_pass(accumulate=None, write_dir=cache_dir, label="Cache pass", **common)
            return cached
        if mode in ('std', 'sigma_clip'):
            shared.put('mean_map', mean_map)
        if mode == 'sigma_clip':
            shared.put('std_map', std_map)
        _new_totals(shared, ref_shape, n_aux, False)
        _run_pass(accumulate=mode, write_dir=None, label=f"{mode} pass", **common)
        totals = _take_totals(shared, n_aux, False)
    finally:
        shared.close()
    return _finalize(mode, totals)


def run_coadd_schedule(ref_shape, file_list, offset_lists=None, apply_weight=True, apply_mask=True,
                       chunk_maps=None, grid_valid_weight=None, max_workers=10, ignore_list=None,
                       det_offset_funcs=None, oversample_factor=1, cache_batch_size=10, coadd_batch_size=10,
                       valid_threshold=0.99, cache_dir='cache/', cache_intermediate=False, det_aux=None,
                       preprocess_func=None, postprocess_func=None, make_std_map=False,
                       apply_sigma_clipping=False, sigma=2.0, wav_maps=None):
    """All coadd passes of a mosaic in one schedule.

    Passes: (1) prepare every frame, write the intermediate cache if
    ``cache_intermediate`` and accumulate the mean; (2) std (if
    ``make_std_map``); (3) sigma-clipped mean (if also ``apply_sigma_clipping``),
    which also accumulates the band-centre/width sums when ``wav_maps`` (a
    ``(centre, width)`` pair of detector-grid maps) is given.  Passes 2-3 read
    the cache when there is one and re-prepare the frames otherwise.

    Returns ``(maps, cached_list)`` where ``maps`` holds ``mean_map`` /
    ``std_map`` / ``sc_mean_map`` entries as ``{'data', 'weight', 'aux'}`` dicts
    (absent passes have ``None`` entries) plus ``wav_mean_map`` / ``wav_std_map``
    (``{'data': ...}``) when ``wav_maps`` was given.
    """
    if ignore_list is None:
        ignore_list = []
    if chunk_maps is None:
        chunk_maps = []
    _validate_common('mean', ref_shape, file_list, apply_weight, apply_mask, chunk_maps, offset_lists,
                     det_offset_funcs, grid_valid_weight, max_workers, ignore_list, oversample_factor,
                     coadd_batch_size)
    if not (isinstance(cache_batch_size, int) and cache_batch_size > 0):
        raise ValueError("cache_batch_size must be a positive integer")
    if wav_maps is not None and not (make_std_map and apply_sigma_clipping):
        raise ValueError("wav_maps needs make_std_map and apply_sigma_clipping (the band maps are "
                         "coadded with the sigma-clipped weights)")
    if wav_maps is not None and (len(wav_maps) != 2 or np.shape(wav_maps[0]) != np.shape(wav_maps[1])):
        raise ValueError("wav_maps must be a (centre, width) pair of equally shaped detector maps")
    if not (isinstance(sigma, (int, float)) and sigma > 0):
        raise ValueError("sigma must be a positive number")

    ref_shape = tuple(int(v) for v in ref_shape)
    n_aux = 0 if det_aux is None else len(det_aux)
    do_wav = wav_maps is not None
    maps = {k: {'data': None, 'weight': None, 'aux': None} for k in ('mean_map', 'std_map', 'sc_mean_map')}
    cached = []
    shared = _SharedArrays()
    try:
        _publish_inputs(shared, chunk_maps, grid_valid_weight, det_aux, wav_maps)
        common = dict(ref_shape=ref_shape, shared=shared, n_maps=len(chunk_maps), n_aux=n_aux,
                      prep=_prep_kwargs(apply_weight, apply_mask, ignore_list, det_offset_funcs,
                                        oversample_factor, valid_threshold, preprocess_func, postprocess_func),
                      sigma=sigma, max_workers=max_workers)
        if cache_intermediate:
            os.makedirs(cache_dir, exist_ok=True)
        _new_totals(shared, ref_shape, n_aux, False)
        cached, _ = _run_pass(files=list(file_list), offsets=offset_lists, source='prep', accumulate='mean',
                              write_dir=cache_dir if cache_intermediate else None, wav=False,
                              batch_size=cache_batch_size if cache_intermediate else coadd_batch_size,
                              label="Mean map computation" + (" + cache" if cache_intermediate else ""),
                              **common)
        data, weight, aux = _finalize('mean', _take_totals(shared, n_aux, False))
        maps['mean_map'] = {'data': data, 'weight': weight, 'aux': aux}
        if cache_intermediate:
            src = dict(files=cached, offsets=None, source='cached')
        else:
            src = dict(files=list(file_list), offsets=offset_lists, source='prep')
        if make_std_map:
            shared.put('mean_map', maps['mean_map']['data'])
            _new_totals(shared, ref_shape, n_aux, False)
            _run_pass(accumulate='std', write_dir=None, wav=False, batch_size=coadd_batch_size,
                      label="Std map computation", **src, **common)
            data, weight, aux = _finalize('std', _take_totals(shared, n_aux, False))
            maps['std_map'] = {'data': data, 'weight': weight, 'aux': aux}
        if make_std_map and apply_sigma_clipping:
            shared.put('std_map', maps['std_map']['data'])
            _new_totals(shared, ref_shape, n_aux, do_wav)
            _run_pass(accumulate='sigma_clip', write_dir=None, wav=do_wav, batch_size=coadd_batch_size,
                      label="Sigma-clipped mean map computation" + (" + wavelength" if do_wav else ""),
                      **src, **common)
            totals = _take_totals(shared, n_aux, do_wav)
            data, weight, aux = _finalize('sigma_clip', totals)
            maps['sc_mean_map'] = {'data': data, 'weight': weight, 'aux': aux}
            if do_wav:
                wav_mean, wav_std = _finalize_wav(totals['wav'])
                maps['wav_mean_map'] = {'data': wav_mean, 'weight': None, 'aux': None}
                maps['wav_std_map'] = {'data': wav_std, 'weight': None, 'aux': None}
    finally:
        shared.close()
    return maps, cached
