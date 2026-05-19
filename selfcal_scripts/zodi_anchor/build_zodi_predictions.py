"""Build per-frame zodi predictions for a SelfCal cal file via ZodiPy.

For each reproj file listed in the cal's /reproj_list, read MJD +
pointing, sample grid points across the channel-valid region of the
detector, evaluate ZodiPy at (RA, Dec, MJD, channel-mean wavelength),
and aggregate to a single per-frame mean in MJy/sr. Writes the result
as an .npz that drops directly into apply_zodi_anchor.py --zodi-pred.

ENVIRONMENT
-----------
zodipy 1.1.3 hard-pins numpy<2.0, which conflicts with the main
`selfcal` conda env's numpy-2.x dependencies. Run this script in the
sidecar `selfcal-zodipy` env:

    /home/thomasli/anaconda3/envs/selfcal-zodipy/bin/python \\
        selfcal_scripts/build_zodi_predictions.py --cal ...

The script intentionally has no SelfCal package imports so the sidecar
env can stay minimal (numpy, scipy, astropy, h5py, hdf5plugin, zodipy).
"""
import argparse
import datetime
import os
import re
import sys
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
                        'accurate to within ZodiPy model error since zodi '
                        'varies slowly across the FOV). N>1 = N x N grid '
                        'over channel-mask bbox, mean-averaged per frame.')
    p.add_argument('--num-workers', type=int, default=30,
                   help='Parallel workers for reproj+FITS reads.')
    p.add_argument('--nprocesses', type=int, default=20,
                   help='ZodiPy multiprocess pool size for evaluate() '
                        '(default: 20). ZodiPy is single-threaded per '
                        'process, so this is the dominant performance '
                        'knob for the eval step.')
    p.add_argument('--out', default=None,
                   help='Output .npz path (default: zodi_pred_<tag>.npz '
                        'next to the cal file).')
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


def _decode_attr(val):
    if isinstance(val, bytes):
        return val.decode('utf-8')
    return val


def extract_reproj_metadata(reproj_path):
    """Returns dict with 'wcs', 'mjd', 'error' (None if OK)."""
    try:
        with h5py.File(reproj_path, 'r', libver='latest', swmr=True) as f:
            det_header_str = _decode_attr(f.attrs['det_header'])
            fits_path = _decode_attr(f.attrs['file_path'])
        header = fits.Header.fromstring(det_header_str)
        wcs = WCS(header)
        with fits.open(fits_path) as hdul:
            mjd = hdul[1].header.get('MJD-AVG')
        if mjd is None:
            return dict(wcs=None, mjd=np.nan,
                        error=f'no MJD-AVG in {fits_path}')
        return dict(wcs=wcs, mjd=float(mjd), error=None)
    except Exception as e:
        return dict(wcs=None, mjd=np.nan, error=f'{reproj_path}: {e!r}')


VALID_CHUNK_THRESH = 0.05
# cov_frac noise floor: chunks outside the channel mask still get
# tiny accidental coverage (~1e-3 per frame from interp footprint
# spilling); chunks inside the mask hit 0.1-1.0. Threshold 0.05
# separates the two regimes robustly.


def build_grid_points(det_valid_mask, grid_size):
    """Sample detector-pixel coords representative of the channel mask.

    grid_size=1: return the (y, x) centroid of the valid mask -- single
        representative point per frame. Zodi varies on degree scales,
        so a single point is accurate to within model error.
    grid_size>1: regular grid_size x grid_size grid over the mask
        bbox, filtered to mask-True pixels.
    """
    ys, xs = np.where(det_valid_mask)
    if ys.size == 0:
        raise SystemExit("det_valid_mask is empty -- check cal file.")
    if grid_size <= 1:
        return np.array([ys.mean()]), np.array([xs.mean()])
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    grid_y = np.linspace(y_min, y_max, grid_size).round().astype(int)
    grid_x = np.linspace(x_min, x_max, grid_size).round().astype(int)
    gy, gx = np.meshgrid(grid_y, grid_x, indexing='ij')
    gy = gy.ravel()
    gx = gx.ravel()
    keep = det_valid_mask[gy, gx]
    if not keep.any():
        raise SystemExit(
            "All grid points fell outside the channel-valid mask. "
            "Increase --grid-size.")
    return gy[keep], gx[keep]


def main():
    args = parse_args()

    detector = args.detector or parse_detector_from_filename(args.cal)
    if detector is None:
        raise SystemExit(
            "Could not parse Detector from filename. "
            "Pass --detector N explicitly.")

    out_path = args.out or default_output_path(args.cal)

    print(f"cal:        {args.cal}")
    print(f"detector:   {detector}")
    print(f"model:      {args.model}")
    print(f"grid_size:  {args.grid_size}")
    print(f"out:        {out_path}")

    # ---- Load cal: reproj list + channel-valid mask
    with h5py.File(args.cal, 'r') as f:
        if 'chunk_maps' not in f or 'map_0' not in f['chunk_maps']:
            raise SystemExit(
                "Cal file lacks /chunk_maps/map_0 (legacy schema). "
                "This helper requires the multi-map schema (production "
                "default since 2026-04).")
        det_chunk_map = f['chunk_maps/map_0'][:]
        cov_frac = f['offset_coverage_frac/map_0'][:]
        reproj_list_bytes = f['reproj_list'][:]
    reproj_paths = [s.decode() if isinstance(s, (bytes, np.bytes_)) else s
                    for s in reproj_list_bytes]
    num_frames = len(reproj_paths)
    print(f"num_frames: {num_frames}")

    valid_chunks = np.where((cov_frac > VALID_CHUNK_THRESH).any(axis=0))[0]
    det_valid_mask = np.isin(det_chunk_map, valid_chunks)
    print(f"valid chunks (cov_frac > {VALID_CHUNK_THRESH}): "
          f"{len(valid_chunks)} / {det_chunk_map.max()+1}; "
          f"valid pixels: {int(det_valid_mask.sum())} / "
          f"{det_valid_mask.size}")

    # ---- Wavelength
    bc_path = os.path.join(
        args.calibration_dir, DET_BC_TEMPLATE.format(detector=detector))
    det_BC = fits.getdata(bc_path)
    wavelength_um = float(np.mean(det_BC[det_valid_mask]))
    print(f"wavelength: {wavelength_um:.4f} um  (mean of det_BC over "
          f"valid mask, from {os.path.basename(bc_path)})")

    # ---- Grid points
    grid_ys, grid_xs = build_grid_points(det_valid_mask, args.grid_size)
    n_grid = len(grid_ys)
    print(f"grid: {n_grid} valid points (of {args.grid_size**2} requested)")

    # ---- Per-frame WCS + MJD (parallel)
    print(f"Reading {num_frames} (reproj + source FITS) headers with "
          f"{args.num_workers} workers...")
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        results = list(ex.map(extract_reproj_metadata, reproj_paths))

    wcs_list = [r['wcs'] for r in results]
    mjds = np.array([r['mjd'] for r in results], dtype=np.float64)
    errors = [r['error'] for r in results if r['error'] is not None]
    if errors:
        print(f"WARNING: {len(errors)} frames had read errors. First 3:")
        for e in errors[:3]:
            print(f"  {e}")

    # ---- Compute (RA, Dec) at each (frame, grid_point)
    print("Computing per-frame (RA, Dec) on grid...")
    ras = np.full((num_frames, n_grid), np.nan, dtype=np.float64)
    decs = np.full((num_frames, n_grid), np.nan, dtype=np.float64)
    # WCS.pixel_to_world expects (x, y) order
    for k, wcs_k in enumerate(wcs_list):
        if wcs_k is None:
            continue
        sky = wcs_k.pixel_to_world(grid_xs, grid_ys)
        ras[k] = sky.ra.deg
        decs[k] = sky.dec.deg

    valid_frame = np.isfinite(mjds) & np.isfinite(ras[:, 0])
    n_valid = int(valid_frame.sum())
    print(f"Valid frames after I/O: {n_valid} / {num_frames}")
    if n_valid == 0:
        raise SystemExit("No valid frames; aborting.")

    # ---- ZodiPy: one batched call over valid frames x grid.
    # zodipy 1.1.3 silently returns all-zeros when obstime is unsorted
    # (internal scipy CubicSpline interp over Earth position requires
    # sorted x). Sort before evaluate, unsort the result back to the
    # original (frame, grid) order.
    print(f"Evaluating ZodiPy ({args.model} @ {wavelength_um:.3f} um) "
          f"on {n_valid * n_grid} coords...")
    model = zodipy.Model(wavelength_um * u.micron, name=args.model)

    flat_ra_arr = ras[valid_frame].ravel()
    flat_dec_arr = decs[valid_frame].ravel()
    flat_time_mjd = np.repeat(mjds[valid_frame], n_grid)

    order = np.argsort(flat_time_mjd, kind='stable')
    unorder = np.empty_like(order)
    unorder[order] = np.arange(len(order))

    coords = SkyCoord(
        flat_ra_arr[order] * u.deg,
        flat_dec_arr[order] * u.deg,
        frame='icrs',
        obstime=Time(flat_time_mjd[order], format='mjd'),
    )
    emission_sorted = model.evaluate(coords, nprocesses=args.nprocesses)
    em_arr = emission_sorted.to(u.MJy / u.sr).value[unorder].reshape(
        n_valid, n_grid)

    zodi_pred = np.full(num_frames, np.nan, dtype=np.float64)
    zodi_pred[valid_frame] = em_arr.mean(axis=1)
    print(f"zodi_pred: mean={np.nanmean(zodi_pred):.4g}, "
          f"std={np.nanstd(zodi_pred):.4g}, "
          f"min={np.nanmin(zodi_pred):.4g}, "
          f"max={np.nanmax(zodi_pred):.4g} MJy/sr")

    np.savez(
        out_path,
        zodi_pred=zodi_pred,
        reproj_list=reproj_list_bytes,
        model_name=np.array(args.model),
        wavelength_um=np.float64(wavelength_um),
        grid_size=np.int64(args.grid_size),
        n_grid_valid=np.int64(n_grid),
        detector=np.int64(detector),
        cal_path=np.array(os.path.abspath(args.cal)),
        created_iso=np.array(datetime.datetime.now().isoformat()),
    )
    print(f"Wrote {out_path}")


if __name__ == '__main__':
    main()
