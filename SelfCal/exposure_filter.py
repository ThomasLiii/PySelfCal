"""Cached FITS-header reads for driver-side exposure filtering.

The reproject driver typically opens every exposure once just to read one
header keyword (e.g. ``FINAST`` for SPHEREx astrometry quality). On a
~30k-exposure RAID that loop takes minutes and dominates rerun time. This
module caches header values keyed on ``(path, mtime)`` so identical reruns
hit the JSON cache instead of the FITS files.

The cache stores raw header values, decoupled from the predicate, so the
same cache can serve different filter logic. Cache writes are atomic
(tmp + rename), so a Ctrl-C never leaves a corrupt file behind.
"""
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from astropy.io import fits
from tqdm import tqdm


_CACHE_SCHEMA = 1


def _coerce(val):
    """Cast a FITS header value to a JSON-safe scalar (or None)."""
    if val is None:
        return None
    if isinstance(val, (bool, int, float, str)):
        return val
    # bytes / numpy scalars / Undefined / Card / etc.
    try:
        return str(val)
    except Exception:
        return None


def _load_cache(cache_path):
    if cache_path is None or not os.path.exists(cache_path):
        return {}
    try:
        with open(cache_path, 'r') as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}
    if payload.get('schema_version') != _CACHE_SCHEMA:
        return {}
    return payload.get('entries', {})


def _save_cache(cache_path, entries):
    if cache_path is None:
        return
    os.makedirs(os.path.dirname(os.path.abspath(cache_path)) or '.', exist_ok=True)
    payload = {'schema_version': _CACHE_SCHEMA, 'entries': entries}
    tmp = cache_path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(payload, f)
    os.replace(tmp, cache_path)


def _read_one(path, keys, ext):
    """Return (path, mtime, {key: value}, error). On failure, last
    element is the exception string and the value dict is empty."""
    try:
        mtime = os.path.getmtime(path)
        with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
            hdr = hdul[ext].header
            vals = {k: _coerce(hdr.get(k)) for k in keys}
        return (path, mtime, vals, None)
    except Exception as e:
        return (path, None, {}, str(e))


def cached_header_values(exposure_list, keys, ext=1, cache_path=None,
                         max_workers=8, refresh=False, verbose=True):
    """Read selected FITS header keys for each path in ``exposure_list``,
    using ``cache_path`` (JSON) keyed on ``(path, mtime, ext, keys)`` to
    avoid redundant opens.

    Parameters
    ----------
    exposure_list : list[str]
        FITS paths to query.
    keys : list[str]
        Header keywords to read from extension ``ext`` of each FITS.
    ext : int
        FITS extension index whose header is read. The cache is keyed on
        this value, so cache entries from a different extension are not
        reused.
    cache_path : str or None
        JSON cache file. ``None`` disables caching.
    max_workers : int
        Concurrent FITS-open threads for the uncached entries.
    refresh : bool
        If True, ignore the cache and re-read every entry. Cache is still
        updated.
    verbose : bool
        Print a one-line cache-hit summary.

    Returns
    -------
    list[dict]
        One dict per input path (same order). Successful reads contain the
        requested keys (value or None if absent). Failures contain a single
        ``'_error_'`` key with the exception string.
    """
    cache = _load_cache(cache_path)
    keys_t = tuple(keys)

    def _cache_hit(path):
        if refresh:
            return None
        entry = cache.get(path)
        if not entry:
            return None
        try:
            cur_mtime = os.path.getmtime(path)
        except OSError:
            return None
        if (entry.get('mtime') != cur_mtime
                or entry.get('ext') != ext
                or tuple(entry.get('keys', [])) != keys_t):
            return None
        return entry.get('values', {})

    results = [None] * len(exposure_list)
    pending_idx = []
    n_hit = 0
    for i, path in enumerate(exposure_list):
        hit = _cache_hit(path)
        if hit is not None:
            results[i] = dict(hit)
            n_hit += 1
        else:
            pending_idx.append(i)

    if verbose:
        print(f'[cached_header_values] {n_hit}/{len(exposure_list)} cache hits, '
              f'{len(pending_idx)} to read')

    if pending_idx:
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futs = {ex.submit(_read_one, exposure_list[i], keys, ext): i
                    for i in pending_idx}
            for fut in tqdm(as_completed(futs), total=len(futs),
                            desc='Reading FITS headers',
                            disable=len(futs) < 100):
                i = futs[fut]
                path, mtime, vals, err = fut.result()
                if err is not None:
                    results[i] = {'_error_': err}
                else:
                    results[i] = vals
                    cache[path] = {
                        'mtime': mtime,
                        'ext': ext,
                        'keys': list(keys_t),
                        'values': vals,
                    }

    if cache_path is not None:
        _save_cache(cache_path, cache)

    return results


def filter_exposures_by_header(exposure_list, predicate, keys, ext=1,
                               cache_path=None, max_workers=8, verbose=True):
    """Filter ``exposure_list`` by a header-value predicate.

    ``predicate(header_dict) -> bool`` returns True to keep the exposure.
    For files that fail to open the predicate is not called and the
    exposure is dropped (with the error logged when ``verbose=True``).

    Returns ``(kept, dropped)`` lists of paths in original order. The
    underlying header reads are cached via ``cached_header_values``.
    """
    values = cached_header_values(
        exposure_list, keys=keys, ext=ext, cache_path=cache_path,
        max_workers=max_workers, verbose=verbose)
    kept, dropped = [], []
    n_err = 0
    for path, vals in zip(exposure_list, values):
        if '_error_' in vals:
            dropped.append(path)
            n_err += 1
            if verbose:
                print(f'  dropped {path}: header read failed ({vals["_error_"]})')
            continue
        if predicate(vals):
            kept.append(path)
        else:
            dropped.append(path)
    if verbose:
        print(f'[filter_exposures_by_header] kept {len(kept)}, '
              f'dropped {len(dropped)} (incl. {n_err} read errors)')
    return kept, dropped
