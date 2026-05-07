"""Build a multi-channel cache for tasks 1-3.

For each channel in CHANNELS we load the calibration file and record per
exposure: mean offset over valid chunks, and the three column means
(col0/col1/col2). FITS headers are extracted ONCE (shared across channels)
and augmented with the detector x-axis position angle on sky (needed for
Task 1 -- alignment vs Sun direction).

Outputs (next to this script):
    multichannel_det{det}.pkl      : long-format DataFrame, one row per
                                     (exposure, channel)
    detector_templates_det{det}.pkl: dict {channel: (342, 3) chunk pattern}
                                     where each entry is the time average of
                                     (offset - per-exposure DC) per chunk,
                                     in the (subchannel, column) layout.
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord, get_sun
from astropy.time import Time
import astropy.units as u

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SELFCAL_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
if _SELFCAL_ROOT not in sys.path:
    sys.path.insert(0, _SELFCAL_ROOT)

from SelfCal.MakeMap import load_reproj_file
from SelfCal.SPHERExUtility import make_stripped_chunk_valid_mask

from zodi_utils import (
    cal_path,
    compute_ecliptic_geometry,
    fit_sine,
    data_path,
    load_cal_offsets,
)

NUM_SUB, NUM_CH, NUM_COL = 10, 34, 3
TOT_SUB = NUM_SUB * NUM_CH + 2
CHANNELS = list(range(1, 35))


def _extract_header_and_pa(reproj_path):
    """Pull CRVAL1/2, MJD-AVG, and position angle of the detector +X axis.

    PA is defined as the angle (east-of-north) of the vector from the
    detector centre to the detector +X neighbour pixel on the sky.
    """
    try:
        fpath = load_reproj_file(reproj_path, fields=['file_path'])['file_path']
        with fits.open(fpath) as hdul:
            h = hdul[1].header
            ra0 = h.get('CRVAL1')
            dec0 = h.get('CRVAL2')
            mjd = h.get('MJD-AVG')
            w = WCS(h)

        ny = int(h.get('NAXIS2', 2040))
        nx = int(h.get('NAXIS1', 2040))
        yc, xc = ny // 2, nx // 2
        # Sky coordinates of the centre and the +x neighbour.
        c_ra, c_dec = w.pixel_to_world_values(xc, yc)
        r_ra, r_dec = w.pixel_to_world_values(xc + 1, yc)

        # PA east-of-north via astropy great-circle bearing.
        a = SkyCoord(c_ra, c_dec, unit='deg', frame='icrs')
        b = SkyCoord(r_ra, r_dec, unit='deg', frame='icrs')
        pa_x = float(a.position_angle(b).deg)
        return (ra0, dec0, mjd, pa_x)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)


def _load_channel_cal(detector, channel):
    """Return offset cube + valid-subchannel index list for one channel."""
    with h5py.File(cal_path(detector, channel), 'r') as f:
        off = load_cal_offsets(f)[0]                   # (N, 342*3)
        reproj_list = [s.decode('utf-8') for s in f['reproj_list'][:]]
    off = off.reshape(off.shape[0], TOT_SUB, NUM_COL)  # (N, 342, 3)
    mask = make_stripped_chunk_valid_mask(
        ch=[channel], num_subchannels=NUM_SUB, num_channels=NUM_CH,
        num_columns=NUM_COL, subchannel_padding=0,
    ).reshape(TOT_SUB, NUM_COL)
    valid_sub = np.where(mask.any(axis=1))[0]
    return off, reproj_list, valid_sub


def build(detector, overwrite=False):
    out_long = data_path(f'multichannel_det{detector}.pkl')
    out_tpl = data_path(f'detector_templates_det{detector}.pkl')

    if os.path.exists(out_long) and os.path.exists(out_tpl) and not overwrite:
        print('Caches exist; pass --overwrite to rebuild.')
        return out_long, out_tpl

    print('Loading reproj list from channel 17 (shared across channels)...')
    _, reproj_list, _ = _load_channel_cal(detector, 17)
    print(f'  {len(reproj_list)} exposures')

    print('Extracting FITS headers + per-exposure PA of detector +X axis...')
    records = []
    with ProcessPoolExecutor(max_workers=20) as ex:
        for r in tqdm(ex.map(_extract_header_and_pa, reproj_list, chunksize=40),
                      total=len(reproj_list)):
            records.append(r)
    base = pd.DataFrame(records, columns=['CRVAL1', 'CRVAL2', 'MJD_AVG', 'pa_x_deg'])
    base = base.dropna(subset=['CRVAL1', 'CRVAL2', 'MJD_AVG']).reset_index(drop=True)

    print('Computing ecliptic + Sun geometry per exposure...')
    geom = compute_ecliptic_geometry(
        base['CRVAL1'].values, base['CRVAL2'].values, base['MJD_AVG'].values)
    for k, v in geom.items():
        base[k] = v

    # Position angle from target to Sun (east-of-north). IMPORTANT: we must
    # use the Sun's apparent DIRECTION, not the `get_sun` return value which
    # carries a 1 AU heliocentric distance. If the distance is kept, astropy
    # uses 3D positions and the Earth-Sun parallax flips the result.
    t = Time(base['MJD_AVG'].values, format='mjd')
    target = SkyCoord(base['CRVAL1'].values * u.deg,
                      base['CRVAL2'].values * u.deg, frame='icrs')
    sun_gcrs = get_sun(t)
    sun_dir = SkyCoord(sun_gcrs.ra, sun_gcrs.dec, frame='icrs')
    base['pa_to_sun_deg'] = target.position_angle(sun_dir).deg

    # Now iterate channels to collect mean_offset + column means + templates.
    print(f'Loading cal files for channels {CHANNELS[0]}..{CHANNELS[-1]}...')
    all_rows = []
    templates = {}
    for ch in tqdm(CHANNELS, desc='channels'):
        off, rlist, valid_sub = _load_channel_cal(detector, ch)
        assert len(rlist) == len(reproj_list), 'reproj list mismatch across channels'
        # Per-exposure: collapse over valid subchannels to a single scalar per column.
        col_means = off[:, valid_sub, :].mean(axis=1)  # (N, 3)
        mean_offset = col_means.mean(axis=1)           # (N,)
        df_ch = base.copy()
        df_ch['channel'] = ch
        df_ch['mean_offset'] = mean_offset
        df_ch['col0'] = col_means[:, 0]
        df_ch['col1'] = col_means[:, 1]
        df_ch['col2'] = col_means[:, 2]
        df_ch['grad_col02'] = col_means[:, 0] - col_means[:, 2]
        all_rows.append(df_ch)

        # Detector-fixed pattern for this channel:
        #   (1) compute per-exposure DC from the UNPADDED valid subchannels
        #       (most reliable region),
        #   (2) subtract that DC from every (subch, col) element of the
        #       offset cube, and
        #   (3) average across exposures.
        # We store values at the PADDED subchannels so the downstream
        # stitching step can use the overlap between adjacent channels.
        padded_mask = make_stripped_chunk_valid_mask(
            ch=[ch], num_subchannels=NUM_SUB, num_channels=NUM_CH,
            num_columns=NUM_COL, subchannel_padding=1,
        ).reshape(TOT_SUB, NUM_COL)
        padded_sub = np.where(padded_mask.any(axis=1))[0]

        # DC from unpadded valid region.
        valid_mat_unpad = off[:, valid_sub, :]               # (N, n_unpad, 3)
        dc = valid_mat_unpad.reshape(valid_mat_unpad.shape[0], -1).mean(axis=1)[:, None, None]
        # Template values at padded subchannels.
        valid_mat_pad = off[:, padded_sub, :]                # (N, n_pad, 3)
        dev_pad = valid_mat_pad - dc                          # (N, n_pad, 3)
        template_sub = dev_pad.mean(axis=0)                   # (n_pad, 3)
        tpl = np.full((TOT_SUB, NUM_COL), np.nan)
        tpl[padded_sub, :] = template_sub
        templates[ch] = tpl

    long_df = pd.concat(all_rows, ignore_index=True)
    long_df.to_pickle(out_long)
    pd.to_pickle(templates, out_tpl)
    print(f'wrote {out_long}  ({len(long_df)} rows)')
    print(f'wrote {out_tpl}   ({len(templates)} channels)')
    return out_long, out_tpl


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    p.add_argument('--overwrite', action='store_true')
    args = p.parse_args()
    build(args.detector, overwrite=args.overwrite)
