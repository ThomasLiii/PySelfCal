"""Build per-frame zodi predictions for a SelfCal cal file via ZodiPy.

For each reproj file listed in the cal's /reproj_list, read MJD +
pointing, sample the channel-valid mask centroid, evaluate ZodiPy at
(RA, Dec, MJD, channel-mean wavelength), and aggregate to a single
per-frame mean in MJy/sr. Writes the result as an .npz that drops
directly into apply_zodi_anchor.py --zodi-pred.

ENVIRONMENT
-----------
zodipy 1.1.3 hard-pins numpy<2.0, which conflicts with the main
`selfcal` conda env's numpy-2.x dependencies. Run this script in the
sidecar `selfcal-zodipy` env:

    /home/thomasli/anaconda3/envs/selfcal-zodipy/bin/python \\
        selfcal_scripts/zodi_anchor/build_zodi_predictions.py --cal ...

The script intentionally has no SelfCal package imports so the sidecar
env can stay minimal (numpy, scipy, astropy, h5py, hdf5plugin, zodipy).

The module-level functions are import-safe so a batch driver can
extract MJD+WCS once for a detector and reuse across channels.
"""
import argparse
import datetime
import os
import re
from concurrent.futures import ThreadPoolExecutor

import astropy.units as u
import h5py
import hdf5plugin  # noqa: F401  -- registers zstd plugin for reproj h5 reads
import numpy as np
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS

try:
    import zodipy
except ImportError as e:
    raise SystemExit(
        "zodipy not installed. Run in the selfcal-zodipy env:\n"
        "    /home/thomasli/anaconda3/envs/selfcal-zodipy/bin/python ...\n"
        f"({e})")


DEFAULT_CALIBRATION_DIR = '/home/thomasli/spherex/SPHEREx_Spectral_Calibration'
DET_BC_TEMPLATE = '20250901_SSDC_BC_Band{detector}.fits'
DEFAULT_METADATA_CACHE_TEMPLATE = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 'cache', 'metadata_D{detector}.h5'))
VALID_CHUNK_THRESH = 0.05
# cov_frac noise floor: chunks outside the channel mask still get
# tiny accidental coverage (~1e-3 per frame from interp footprint
# spilling); chunks inside the mask hit 0.1-1.0. Threshold 0.05
# separates the two regimes robustly.


# ---------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------

def _decode_attr(val):
    if isinstance(val, bytes):
        return val.decode('utf-8')
    return val


def extract_reproj_metadata(reproj_path):
    """Read (det_header_str, MJD) for one reproj h5 file.

    Returns dict {'det_header_str', 'mjd', 'error'}. det_header_str is
    a FITS-header string ready for ``WCS(fits.Header.fromstring(s))``.
    Storing the string (not the WCS object) is cache-friendly:
    ``astropy.wcs.WCS`` is not directly pickleable and not stable
    across astropy versions, but a header string is.
    """
    try:
        with h5py.File(reproj_path, 'r', libver='latest', swmr=True) as f:
            det_header_str = _decode_attr(f.attrs['det_header'])
            fits_path = _decode_attr(f.attrs['file_path'])
        with fits.open(fits_path) as hdul:
            mjd = hdul[1].header.get('MJD-AVG')
        if mjd is None:
            return dict(det_header_str='', mjd=np.nan,
                        error=f'no MJD-AVG in {fits_path}')
        return dict(det_header_str=det_header_str, mjd=float(mjd), error=None)
    except Exception as e:
        return dict(det_header_str='', mjd=np.nan,
                    error=f'{reproj_path}: {e!r}')


def load_metadata_cache(path):
    """Load a persistent metadata cache file.

    Returns dict ``{reproj_path: {'det_header_str', 'mjd', 'error'}}``
    or ``{}`` if the cache doesn't exist / can't be read.
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        with h5py.File(path, 'r') as f:
            mjds = f['mjds'][:]
            det_headers = f['det_headers'][:]
            paths = f['reproj_list'][:]
    except Exception as e:
        print(f"WARNING: could not load metadata cache {path}: {e!r}")
        return {}
    out = {}
    for p, m, h in zip(paths, mjds, det_headers):
        p = p.decode() if isinstance(p, bytes) else p
        h = h.decode() if isinstance(h, bytes) else h
        out[p] = dict(det_header_str=h, mjd=float(m),
                      error=None if np.isfinite(m) else 'cached-as-error')
    return out


def save_metadata_cache(path, cache_dict):
    """Write the cache dict to disk (h5py with variable-length strings)."""
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    paths, mjds, headers = [], [], []
    for p, r in cache_dict.items():
        if r.get('error'):
            continue  # don't cache errors
        paths.append(p)
        mjds.append(r['mjd'])
        headers.append(r['det_header_str'])
    str_dt = h5py.string_dtype()
    with h5py.File(path, 'w') as f:
        f.create_dataset('mjds', data=np.array(mjds, dtype=np.float64))
        f.create_dataset('det_headers', data=np.array(headers, dtype=object),
                         dtype=str_dt)
        f.create_dataset('reproj_list', data=np.array(paths, dtype=object),
                         dtype=str_dt)
        f.attrs['created_iso'] = datetime.datetime.now().isoformat()
        f.attrs['n_frames'] = len(paths)
    print(f"  metadata cache saved: {path} ({len(paths)} frames)")


def extract_metadata_for_reproj_list(reproj_paths, num_workers=30,
                                     desc=None,
                                     metadata_cache_path=None):
    """Parallel extract (WCS, MJD) for every path in reproj_paths,
    with optional persistent caching.

    If ``metadata_cache_path`` is set, load any pre-extracted entries
    from there and only re-extract the missing ones; update the cache
    on disk before returning.

    Returns (wcs_list, mjds, errors).
    """
    cache = load_metadata_cache(metadata_cache_path)
    if cache:
        print(f"  metadata cache loaded: {metadata_cache_path} "
              f"({len(cache)} frames)")
    to_extract = [p for p in reproj_paths if p not in cache]
    if to_extract:
        if desc:
            print(f"{desc}  (cache miss for {len(to_extract)} / "
                  f"{len(reproj_paths)} frames)")
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            extracted = list(ex.map(extract_reproj_metadata, to_extract))
        for p, r in zip(to_extract, extracted):
            cache[p] = r
        if metadata_cache_path:
            save_metadata_cache(metadata_cache_path, cache)
    else:
        if cache:
            print(f"  all {len(reproj_paths)} frames found in cache; "
                  "skipping I/O.")

    wcs_list = []
    mjds = np.empty(len(reproj_paths), dtype=np.float64)
    errors = []
    for i, p in enumerate(reproj_paths):
        r = cache[p]
        if r.get('error'):
            wcs_list.append(None)
            mjds[i] = np.nan
            errors.append(r['error'])
        else:
            wcs_list.append(WCS(fits.Header.fromstring(r['det_header_str'])))
            mjds[i] = r['mjd']
    if errors:
        print(f"WARNING: {len(errors)} frames had read errors. First 3:")
        for e in errors[:3]:
            print(f"  {e}")
    return wcs_list, mjds, errors


# ---------------------------------------------------------------------
# Channel-valid mask + grid + wavelength + ZodiPy
# ---------------------------------------------------------------------

def channel_valid_mask_from_cal(cal_h5):
    """Reconstruct the (det_h, det_w) channel-valid mask from the cal
    file's chunk_maps + offset_coverage_frac (no filename parsing)."""
    if ('chunk_maps' not in cal_h5
            or 'map_0' not in cal_h5['chunk_maps']):
        raise ValueError(
            "cal file lacks /chunk_maps/map_0 (legacy schema). Anchor "
            "requires the multi-map schema (production since 2026-04).")
    det_chunk_map = cal_h5['chunk_maps/map_0'][:]
    cov_frac = cal_h5['offset_coverage_frac/map_0'][:]
    valid_chunks = np.where((cov_frac > VALID_CHUNK_THRESH).any(axis=0))[0]
    det_valid_mask = np.isin(det_chunk_map, valid_chunks)
    return det_valid_mask, valid_chunks, det_chunk_map


def build_grid_points(det_valid_mask, grid_size):
    """Sample detector-pixel coords representative of the channel mask.

    grid_size <= 1: returns the (y, x) centroid of the valid mask.
    grid_size  > 1: NxN regular grid over the mask bbox, filtered to
                    mask-True pixels.
    """
    ys, xs = np.where(det_valid_mask)
    if ys.size == 0:
        raise ValueError("det_valid_mask is empty -- check cal file.")
    if grid_size <= 1:
        return np.array([ys.mean()]), np.array([xs.mean()])
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    gy = np.linspace(y_min, y_max, grid_size).round().astype(int)
    gx = np.linspace(x_min, x_max, grid_size).round().astype(int)
    gy, gx = np.meshgrid(gy, gx, indexing='ij')
    gy = gy.ravel()
    gx = gx.ravel()
    keep = det_valid_mask[gy, gx]
    if not keep.any():
        raise ValueError(
            "All grid points fell outside the channel-valid mask. "
            "Increase grid_size.")
    return gy[keep], gx[keep]


def wavelength_for_channel(det_BC, det_valid_mask):
    return float(np.mean(det_BC[det_valid_mask]))


def evaluate_zodi_per_frame(wcs_list, mjds, grid_ys, grid_xs,
                            wavelength_um, model_name='dirbe',
                            nprocesses=20):
    """Evaluate ZodiPy at the channel centroid per frame.

    Sorts obstime ascending before calling model.evaluate (zodipy 1.1.3
    silently returns all-zeros otherwise). Result is un-sorted back to
    original frame order.

    Returns (zodi_pred, n_grid_valid).
    """
    num_frames = len(wcs_list)
    n_grid = len(grid_ys)

    ras = np.full((num_frames, n_grid), np.nan, dtype=np.float64)
    decs = np.full((num_frames, n_grid), np.nan, dtype=np.float64)
    for k, wcs_k in enumerate(wcs_list):
        if wcs_k is None:
            continue
        sky = wcs_k.pixel_to_world(grid_xs, grid_ys)
        ras[k] = sky.ra.deg
        decs[k] = sky.dec.deg

    valid_frame = np.isfinite(mjds) & np.isfinite(ras[:, 0])
    n_valid = int(valid_frame.sum())
    if n_valid == 0:
        raise ValueError("No valid frames; aborting.")

    # extrapolate=True lets us use models whose nominal frequency range
    # doesn't cover the SPHEREx 0.75-5 um band (e.g., rrm-experimental,
    # planck*, odegard). For DIRBE the request is inside the calibrated
    # range so this is a no-op.
    model = zodipy.Model(wavelength_um * u.micron, name=model_name,
                         extrapolate=True)

    flat_ra = ras[valid_frame].ravel()
    flat_dec = decs[valid_frame].ravel()
    flat_time_mjd = np.repeat(mjds[valid_frame], n_grid)
    order = np.argsort(flat_time_mjd, kind='stable')
    unorder = np.empty_like(order)
    unorder[order] = np.arange(len(order))

    coords = SkyCoord(
        flat_ra[order] * u.deg,
        flat_dec[order] * u.deg,
        frame='icrs',
        obstime=Time(flat_time_mjd[order], format='mjd'),
    )
    emission = model.evaluate(coords, nprocesses=nprocesses)
    em_arr = emission.to(u.MJy / u.sr).value[unorder].reshape(n_valid, n_grid)

    zodi_pred = np.full(num_frames, np.nan, dtype=np.float64)
    zodi_pred[valid_frame] = em_arr.mean(axis=1)
    return zodi_pred, n_grid


def _build_zodi_pred_inner(det_valid_mask, det_chunk_map, det_BC,
                           wcs_list, mjds, reproj_list_bytes,
                           detector, model_name, grid_size, nprocesses,
                           valid_chunks=None,
                           valid_chunks_label=''):
    """Shared inner: grid + wavelength + ZodiPy eval. Returns the
    full result dict ready for ``save_predictions_npz``."""
    grid_ys, grid_xs = build_grid_points(det_valid_mask, grid_size)
    wavelength_um = wavelength_for_channel(det_BC, det_valid_mask)
    nvc = (len(valid_chunks) if valid_chunks is not None else
           int(det_valid_mask.sum() and (np.unique(det_chunk_map[det_valid_mask]).size)))
    n_chunks_total = int(det_chunk_map.max() + 1)
    print(f"  valid chunks{valid_chunks_label}: {nvc} / {n_chunks_total}; "
          f"wavelength = {wavelength_um:.4f} um; grid = {len(grid_ys)} pt(s)")

    zodi_pred, n_grid_valid = evaluate_zodi_per_frame(
        wcs_list, mjds, grid_ys, grid_xs, wavelength_um,
        model_name=model_name, nprocesses=nprocesses)
    print(f"  zodi_pred: mean={np.nanmean(zodi_pred):.4g}, "
          f"std={np.nanstd(zodi_pred):.4g}, "
          f"range [{np.nanmin(zodi_pred):.4g}, "
          f"{np.nanmax(zodi_pred):.4g}] MJy/sr")

    return dict(
        zodi_pred=zodi_pred,
        mjds=mjds,
        reproj_list=reproj_list_bytes,
        wavelength_um=wavelength_um,
        n_grid_valid=n_grid_valid,
        grid_size=grid_size,
        model_name=model_name,
        detector=detector,
    )


def build_for_channel(cal_path, wcs_list, mjds, det_BC, detector,
                      model_name='dirbe', grid_size=1, nprocesses=20):
    """Compute zodi_pred for one channel given pre-cached per-frame
    (WCS, MJD). Reads valid mask from the cal file's offset_coverage_frac."""
    with h5py.File(cal_path, 'r') as f:
        det_valid_mask, valid_chunks, det_chunk_map = (
            channel_valid_mask_from_cal(f))
        reproj_list_bytes = f['reproj_list'][:]
    result = _build_zodi_pred_inner(
        det_valid_mask=det_valid_mask, det_chunk_map=det_chunk_map,
        det_BC=det_BC, wcs_list=wcs_list, mjds=mjds,
        reproj_list_bytes=reproj_list_bytes,
        detector=detector, model_name=model_name,
        grid_size=grid_size, nprocesses=nprocesses,
        valid_chunks=valid_chunks,
        valid_chunks_label=f' (cov_frac > {VALID_CHUNK_THRESH})')
    result['cal_path'] = os.path.abspath(cal_path)
    return result


def build_for_channel_theoretical(ch, num_subchannels, num_channels, num_columns,
                                  wcs_list, mjds, det_BC, det_chunk_map,
                                  reproj_list_bytes, detector,
                                  model_name='dirbe', grid_size=1,
                                  nprocesses=20, subchannel_padding=0):
    """Compute zodi_pred for one channel WITHOUT a cal file.

    Channel mask comes from
    ``SelfCal.SPHERExUtility.make_stripped_chunk_valid_mask(ch=[ch], ...)``
    — purely a function of the LVF geometry and channel number, no
    coverage-based thresholding. Use when the cal file doesn't exist
    yet (e.g. anchor predictions in parallel with the cal solve).
    """
    # Import here to keep zodi-only env paths optional; SelfCal is
    # available in both envs.
    from SelfCal.SPHERExUtility import make_stripped_chunk_valid_mask
    # SPHERExUtility returns a float64 0/1 mask; cast to bool so the
    # downstream `det_BC[det_valid_mask]` works.
    chunk_valid_mask_1d = make_stripped_chunk_valid_mask(
        ch=[ch], num_subchannels=num_subchannels,
        num_channels=num_channels, num_columns=num_columns,
        subchannel_padding=subchannel_padding).astype(bool)
    det_valid_mask = chunk_valid_mask_1d[det_chunk_map]
    valid_chunks = np.where(chunk_valid_mask_1d)[0]
    return _build_zodi_pred_inner(
        det_valid_mask=det_valid_mask, det_chunk_map=det_chunk_map,
        det_BC=det_BC, wcs_list=wcs_list, mjds=mjds,
        reproj_list_bytes=reproj_list_bytes,
        detector=detector, model_name=model_name,
        grid_size=grid_size, nprocesses=nprocesses,
        valid_chunks=valid_chunks,
        valid_chunks_label=f' (theoretical, padding={subchannel_padding})')


def save_predictions_npz(out_path, result):
    """Save a build_for_channel result to .npz."""
    np.savez(
        out_path,
        zodi_pred=result['zodi_pred'],
        mjds=result['mjds'],
        reproj_list=result['reproj_list'],
        model_name=np.array(result['model_name']),
        wavelength_um=np.float64(result['wavelength_um']),
        grid_size=np.int64(result['grid_size']),
        n_grid_valid=np.int64(result['n_grid_valid']),
        detector=np.int64(result['detector']),
        cal_path=np.array(result.get('cal_path', '')),
        created_iso=np.array(datetime.datetime.now().isoformat()),
    )


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--cal', required=True,
                   help='Path to cal_*.h5 with /chunk_maps/map_0 and '
                        '/frame_scalar (multi-map schema).')
    p.add_argument('--calibration-dir', default=DEFAULT_CALIBRATION_DIR,
                   help='Dir with 20250901_SSDC_BC_Band{D}.fits (LVF '
                        'band centers).')
    p.add_argument('--detector', type=int, default=None,
                   help='Detector index (1-6). Auto-parsed from filename '
                        'if omitted.')
    p.add_argument('--model', default='dirbe',
                   help='ZodiPy model name (default: dirbe = Kelsall+98).')
    p.add_argument('--grid-size', type=int, default=1,
                   help='1 = single centroid per frame (default; fast and '
                        'accurate to within ZodiPy model error). N>1 = '
                        'NxN grid over channel-mask bbox, mean-averaged.')
    p.add_argument('--num-workers', type=int, default=30,
                   help='Parallel workers for reproj+FITS reads.')
    p.add_argument('--nprocesses', type=int, default=20,
                   help='ZodiPy multiprocess pool size for evaluate().')
    p.add_argument('--out', default=None,
                   help='Output .npz path (default: zodi_pred_<tag>.npz '
                        'next to cal).')
    p.add_argument('--metadata-cache', default=None,
                   help='Persistent metadata cache path (per detector). '
                        'Stores MJD + WCS header per reproj file so '
                        'subsequent runs skip the 10-15 min per-frame '
                        'I/O. Default: '
                        f'{DEFAULT_METADATA_CACHE_TEMPLATE}')
    return p.parse_args()


def parse_detector_from_filename(cal_path):
    m = re.search(r'cal_Detector(\d+)_', os.path.basename(cal_path))
    return int(m.group(1)) if m else None


def default_output_path(cal_path):
    base = os.path.basename(cal_path)
    if not (base.startswith('cal_') and base.endswith('.h5')):
        return os.path.join(os.path.dirname(cal_path),
                            'zodi_pred_' + base + '.npz')
    tag = base[len('cal_'):-len('.h5')]
    return os.path.join(os.path.dirname(cal_path), f'zodi_pred_{tag}.npz')


def main():
    args = parse_args()
    detector = args.detector or parse_detector_from_filename(args.cal)
    if detector is None:
        raise SystemExit(
            "Could not parse Detector from filename. "
            "Pass --detector N explicitly.")
    out_path = args.out or default_output_path(args.cal)
    metadata_cache_path = (args.metadata_cache
                           or DEFAULT_METADATA_CACHE_TEMPLATE.format(
                               detector=detector))

    print(f"cal:        {args.cal}")
    print(f"detector:   {detector}")
    print(f"model:      {args.model}")
    print(f"grid_size:  {args.grid_size}")
    print(f"out:        {out_path}")
    print(f"cache:      {metadata_cache_path}")

    with h5py.File(args.cal, 'r') as f:
        if 'frame_scalar' not in f:
            raise SystemExit(
                "Anchor requires use_per_frame_scalar=True cal runs; "
                f"{args.cal} lacks /frame_scalar.")
        reproj_paths = [s.decode() if isinstance(s, (bytes, np.bytes_)) else s
                        for s in f['reproj_list'][:]]

    wcs_list, mjds, _ = extract_metadata_for_reproj_list(
        reproj_paths, num_workers=args.num_workers,
        desc=f"Reading {len(reproj_paths)} (reproj + source FITS) headers"
             f" with {args.num_workers} workers...",
        metadata_cache_path=metadata_cache_path)

    bc_path = os.path.join(
        args.calibration_dir, DET_BC_TEMPLATE.format(detector=detector))
    det_BC = fits.getdata(bc_path)

    result = build_for_channel(
        args.cal, wcs_list, mjds, det_BC, detector,
        model_name=args.model, grid_size=args.grid_size,
        nprocesses=args.nprocesses)
    save_predictions_npz(out_path, result)
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
