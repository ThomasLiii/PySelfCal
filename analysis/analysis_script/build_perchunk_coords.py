"""Build per-(exposure, chunk) sky/ecliptic coordinates.

Each chunk has a fixed detector centroid (depends only on chunk_map). Each
exposure has its own WCS, so projecting those centroids through the WCS
gives per-chunk (RA, Dec) for that exposure. Combined with the exposure
MJD and astropy ecliptic transforms, we get per-chunk
(ecl_lon, ecl_lat, helio_lon, elongation).

Output: cache/perchunk_coords_det{detector}.npz with arrays of shape
(n_exposures, n_chunks).  This replaces the per-exposure-only metadata
used by the meeting plots.
"""
import argparse
import os
import sys

import h5py
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_SELFCAL_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))
if _SELFCAL_ROOT not in sys.path:
    sys.path.insert(0, _SELFCAL_ROOT)

from selfcal.io.reproj import load_reproj_file
from selfcal.instruments.spherex.spherex_utility import make_stripped_chunk_map, load_lvf_params
from zodi_utils import data_path, cal_path

NUM_SUB, NUM_CH, NUM_COL = 10, 34, 3
TOT_SUB = NUM_SUB * NUM_CH + 2
N_CHUNKS = TOT_SUB * NUM_COL

_GLOBAL_CENTROIDS = None  # filled by worker initializer


def compute_chunk_centroids(detector):
    """Return (n_chunks, 2) of (y_c, x_c) detector-pixel centroids per chunk.

    Chunks not present in det_chunk_map (e.g. corner regions) get NaN.
    """
    lvf = load_lvf_params(f'lvf_params_D{detector}.npy')
    det_chunk_map, _, _, _ = make_stripped_chunk_map(
        detector, num_subchannels=NUM_SUB, num_channels=NUM_CH,
        num_columns=NUM_COL, oversample_factor=1, lvf_params=lvf,
    )
    centroids = np.full((N_CHUNKS, 2), np.nan)
    for cid in range(N_CHUNKS):
        m = det_chunk_map == cid
        if m.any():
            ys, xs = np.where(m)
            centroids[cid] = (ys.mean(), xs.mean())
    return centroids


def _project_one_exposure(reproj_path):
    """Worker: project chunk centroids through the exposure WCS, return
    (ra, dec, mjd, sun_ra, sun_dec, sun_ecl_lon) for use by the parent."""
    centroids = _GLOBAL_CENTROIDS
    try:
        fpath = load_reproj_file(reproj_path, fields=['file_path'])['file_path']
        with fits.open(fpath) as hdul:
            h = hdul[1].header
            w = WCS(h)
            mjd = float(h.get('MJD-AVG'))
        valid = ~np.isnan(centroids[:, 0])
        ra = np.full(N_CHUNKS, np.nan)
        dec = np.full(N_CHUNKS, np.nan)
        rs, ds = w.pixel_to_world_values(centroids[valid, 1], centroids[valid, 0])
        ra[valid] = rs
        dec[valid] = ds
        return mjd, ra, dec
    except Exception:
        return np.nan, np.full(N_CHUNKS, np.nan), np.full(N_CHUNKS, np.nan)


def _init_worker(centroids):
    global _GLOBAL_CENTROIDS
    _GLOBAL_CENTROIDS = centroids


def build(detector):
    centroids = compute_chunk_centroids(detector)
    n_valid = int((~np.isnan(centroids[:, 0])).sum())
    print(f'chunk centroids computed: {n_valid}/{N_CHUNKS} populated')

    # Reuse the reproj list from any one cal file (channel 17 is fine).
    with h5py.File(cal_path(detector, 17), 'r') as f:
        reproj_list = [s.decode('utf-8') for s in f['reproj_list'][:]]
    n_exp = len(reproj_list)
    print(f'{n_exp} exposures to project')

    # 1) WCS projection (parallel).
    ra_arr = np.full((n_exp, N_CHUNKS), np.nan, dtype=np.float32)
    dec_arr = np.full((n_exp, N_CHUNKS), np.nan, dtype=np.float32)
    mjd_arr = np.full(n_exp, np.nan, dtype=np.float64)
    with ProcessPoolExecutor(max_workers=20,
                             initializer=_init_worker,
                             initargs=(centroids,)) as ex:
        for i, (mjd, ra, dec) in enumerate(
                tqdm(ex.map(_project_one_exposure, reproj_list, chunksize=20),
                     total=n_exp, desc='WCS projection')):
            mjd_arr[i] = mjd
            ra_arr[i] = ra.astype(np.float32)
            dec_arr[i] = dec.astype(np.float32)

    # 2) Ecliptic + sun geometry (vectorised across exposures, looped chunks
    # to control memory).  We can do this in chunks of ~100 chunks at a time
    # using astropy SkyCoord vectorised over (n_exp,) per chunk_id.
    print('Computing per-(exp, chunk) ecliptic + sun geometry...')
    ecl_lon = np.full((n_exp, N_CHUNKS), np.nan, dtype=np.float32)
    ecl_lat = np.full((n_exp, N_CHUNKS), np.nan, dtype=np.float32)
    helio_lon = np.full((n_exp, N_CHUNKS), np.nan, dtype=np.float32)
    elongation = np.full((n_exp, N_CHUNKS), np.nan, dtype=np.float32)

    # Sun direction is per-exposure; compute once.
    print('  computing per-exposure Sun direction...')
    times = Time(mjd_arr, format='mjd', scale='utc')
    sun_gcrs = get_sun(times)
    sun_dir = SkyCoord(sun_gcrs.ra, sun_gcrs.dec, frame='icrs')
    sun_ecl_lon = sun_gcrs.transform_to('geocentrictrueecliptic').lon.deg

    chunk_ids = np.where(~np.isnan(centroids[:, 0]))[0]
    for cid in tqdm(chunk_ids, desc='per-chunk geometry'):
        ras = ra_arr[:, cid]; decs = dec_arr[:, cid]
        ok = ~np.isnan(ras)
        if not ok.any():
            continue
        c = SkyCoord(ras[ok] * u.deg, decs[ok] * u.deg, frame='icrs')
        c_ecl = c.transform_to('geocentrictrueecliptic')
        ecl_lon[ok, cid] = c_ecl.lon.deg
        ecl_lat[ok, cid] = c_ecl.lat.deg
        # helio_lon = ecl_lon - sun_ecl_lon, wrapped
        hl = np.mod(c_ecl.lon.deg - sun_ecl_lon[ok] + 180.0, 360.0) - 180.0
        helio_lon[ok, cid] = hl
        # elongation against direction-only sun
        elongation[ok, cid] = c.separation(sun_dir[ok]).deg

    out = data_path(f'perchunk_coords_det{detector}.npz')
    np.savez_compressed(out,
                        chunk_centroids=centroids,
                        mjd=mjd_arr,
                        ra=ra_arr, dec=dec_arr,
                        ecl_lon=ecl_lon, ecl_lat=ecl_lat,
                        helio_lon=helio_lon, elongation=elongation,
                        reproj_paths=np.array(reproj_list, dtype='S'))
    print(f'wrote {out}  (size {os.path.getsize(out) / 1e6:.0f} MB)')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    args = p.parse_args()
    build(args.detector)
