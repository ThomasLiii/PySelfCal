"""Build the per-exposure metadata cache for zodi analysis.

Slow step: reads thousands of FITS headers to pair each exposure's mean
self-cal offset with (MJD, CRVAL1, CRVAL2), then tags on the ecliptic /
solar geometry. Run this once per (detector, channel); downstream plotting
scripts load the cached parquet.

Usage
-----
    python build_metadata.py --detector 4 --channel 14
    python build_metadata.py --detector 5 --channel 20 --workers 20

The cache is written next to this script as
    exposure_df_det{detector}_ch{channel}.pkl
"""
import argparse
import os

import numpy as np
import pandas as pd

from zodi_utils import (
    build_header_table,
    compute_ecliptic_geometry,
    load_single_channel_offset,
    data_path,
)


def cache_path(detector, channel):
    return data_path(f'exposure_df_det{detector}_ch{channel}.pkl')


def build(detector, channel, workers=10, overwrite=False):
    out = cache_path(detector, channel)
    if os.path.exists(out) and not overwrite:
        print(f'Cache exists, skipping: {out}')
        print('Re-run with --overwrite to rebuild.')
        return out

    print(f'Loading cal file for detector {detector}, channel {channel}...')
    mean_offset, _, _, reproj_list = load_single_channel_offset(detector, channel)
    print(f'  {len(reproj_list)} frames, mean_offset range '
          f'[{np.nanmin(mean_offset):.3f}, {np.nanmax(mean_offset):.3f}]')

    print('Extracting FITS headers...')
    df = build_header_table(reproj_list, max_workers=workers)
    df['mean_offset'] = mean_offset
    df['reproj_path'] = reproj_list

    n0 = len(df)
    df = df.dropna(subset=['CRVAL1', 'CRVAL2', 'MJD_AVG', 'mean_offset']).reset_index(drop=True)
    if len(df) < n0:
        print(f'  dropped {n0 - len(df)} rows with missing metadata')

    print('Computing ecliptic / solar geometry...')
    geom = compute_ecliptic_geometry(df['CRVAL1'].values, df['CRVAL2'].values,
                                     df['MJD_AVG'].values)
    for k, v in geom.items():
        df[k] = v

    df['detector'] = detector
    df['channel'] = channel

    df.to_pickle(out)
    print(f'Wrote {out}  ({len(df)} rows, {len(df.columns)} columns)')
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detector', type=int, default=4)
    p.add_argument('--channel', type=int, default=14)
    p.add_argument('--workers', type=int, default=10)
    p.add_argument('--overwrite', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    build(args.detector, args.channel, workers=args.workers, overwrite=args.overwrite)
