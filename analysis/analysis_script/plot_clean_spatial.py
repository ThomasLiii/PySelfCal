"""Clean spatial-zodi diagnostic: subtract the global annual sine first,
then bin-average the residuals in static sky coordinates.

Reasoning
---------
The global temporal fit  offset(t) = C + A sin(2 pi t / 365.25 + phi)
captures the dominant mode (amplitude ~0.018 MJy/sr, residual std ~0.006).
It averages over all sky pointings, so it is essentially a sky-averaged
annual lightcurve of the zodi.

What's *left* after subtracting that fit is, per exposure,
    dz_i = offset_i - sine_model(t_i)
          ~  (zodi(sky_i, t_i) - <zodi(t_i)>_sky)   + noise
i.e. the deviation of the instantaneous zodi at the exposure's sky direction
from the all-sky average at that time. When we spatially bin dz_i in
(ecl_lon, ecl_lat) and average, noise -> 0 and the per-time-average also
integrates to a single *static* spatial pattern (provided each sky bin is
revisited at many times, which is true for this survey).

So the pipeline is:
    1. Global annual sine fit (reuse the existing one)
    2. Residual = offset - sine(t)
    3. Bin-average the residual on (ecl_lon, ecl_lat)
    4. Show the 2-D map, plus the 1-D projections vs ecl_lat and ecl_lon.

Panels whose binned means have SEM << trend amplitude are the "conclusive"
ones; the script prints the trend-to-SEM ratio so you can judge.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zodi_utils import data_path, fig_path

SIDEREAL_YEAR_DAYS = 365.25


def sine_model(t, A, phi, C, f=1.0 / SIDEREAL_YEAR_DAYS):
    return A * np.sin(2.0 * np.pi * f * t + phi) + C


def binned_mean(x, y, edges, min_count=10):
    idx = np.digitize(x, edges) - 1
    n_bins = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan)
    sems = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for b in range(n_bins):
        m = idx == b
        counts[b] = int(m.sum())
        if counts[b] >= min_count:
            means[b] = np.mean(y[m])
            sems[b] = np.std(y[m]) / np.sqrt(counts[b])
    return centers, means, sems, counts


def clip_outliers(y, low=0.5, high=99.5):
    lo, hi = np.nanpercentile(y, [low, high])
    return (y >= lo) & (y <= hi)


def fit_global_sine(t, y):
    """Match the fit in analyze_zodi_spatial.py (soft-L1 loss, 365.25d period)."""
    from scipy.optimize import curve_fit
    p0 = (0.5 * (y.max() - y.min()), 0.0, float(np.median(y)))
    popt, _ = curve_fit(
        lambda tt, A, phi, C: sine_model(tt, A, phi, C),
        t, y, p0=p0, method='trf', loss='soft_l1', f_scale=0.1, max_nfev=5000,
    )
    A, phi, C = popt
    if A < 0:
        A = -A
        phi = phi + np.pi
    phi = (phi + np.pi) % (2 * np.pi) - np.pi
    return float(A), float(phi), float(C)


def grid_2d_mean(x, y, z, x_edges, y_edges, min_count=10):
    ix = np.digitize(x, x_edges) - 1
    iy = np.digitize(y, y_edges) - 1
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    sums = np.zeros((nx, ny))
    counts = np.zeros((nx, ny), dtype=int)
    mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    np.add.at(sums, (ix[mask], iy[mask]), z[mask])
    np.add.at(counts, (ix[mask], iy[mask]), 1)
    out = np.where(counts >= min_count, sums / np.maximum(counts, 1), np.nan)
    return out, counts


def main(detector, channel):
    cache = data_path(f'exposure_df_det{detector}_ch{channel}.pkl')
    df = pd.read_pickle(cache)
    df = df[clip_outliers(df['mean_offset'].values)].reset_index(drop=True)
    print(f'{len(df)} exposures after outlier clip.')

    t = df['MJD_AVG'].values
    offset = df['mean_offset'].values
    ecl_lon = df['ecl_lon'].values
    ecl_lat = df['ecl_lat'].values

    # Step 1: global annual sine (dominant mode).
    A, phi, C = fit_global_sine(t, offset)
    print(f'Global sine:  C={C:.4f}  A={A:.4f}  phi={np.degrees(phi):.1f} deg')

    # Step 2: per-exposure residual after removing the time sine.
    resid = offset - sine_model(t, A, phi, C)
    print(f'Residual std (per exposure): {np.std(resid):.5f}')

    # Step 3a: 1-D projection vs ecl_lat.
    lat_lo, lat_hi = np.percentile(ecl_lat, [0.5, 99.5])
    lat_edges = np.linspace(lat_lo, lat_hi, 21)
    lat_c, lat_m, lat_s, lat_n = binned_mean(ecl_lat, resid, lat_edges, min_count=50)

    # Step 3b: 1-D projection vs ecl_lon.
    lon_edges = np.linspace(0, 360, 37)  # 10 deg bins, ecl_lon is naturally 0..360
    # Wrap ecl_lon to [0, 360) just to be safe.
    lon = np.mod(ecl_lon, 360.0)
    lon_c, lon_m, lon_s, lon_n = binned_mean(lon, resid, lon_edges, min_count=50)

    # Step 3c: 2-D spatial map.
    map_edges_lon = np.linspace(0, 360, 37)
    map_edges_lat = np.linspace(lat_lo, lat_hi, 15)
    grid, grid_n = grid_2d_mean(lon, ecl_lat, resid, map_edges_lon, map_edges_lat, min_count=8)

    # Report trend-to-noise ratios.
    lat_pp = np.nanmax(lat_m) - np.nanmin(lat_m)
    lon_pp = np.nanmax(lon_m) - np.nanmin(lon_m)
    lat_sem = np.nanmedian(lat_s)
    lon_sem = np.nanmedian(lon_s)
    print(f'ecl_lat residual trend: peak-to-peak {lat_pp:.4f}  SEM {lat_sem:.5f}  '
          f'SNR ~ {lat_pp / lat_sem:.1f}')
    print(f'ecl_lon residual trend: peak-to-peak {lon_pp:.4f}  SEM {lon_sem:.5f}  '
          f'SNR ~ {lon_pp / lon_sem:.1f}')

    # ---------- Plot ----------
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.5))

    # Panel 1: residual vs ecl_lat (sky-static latitude trend).
    ax = axes[0]
    ax.errorbar(lat_c, lat_m, yerr=lat_s, fmt='o-', color='C0',
                lw=1.5, ms=5, capsize=3,
                label=f'binned mean (N ~ {int(np.nanmedian(lat_n))} / bin)')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel='ecliptic latitude [deg]',
           ylabel=r'$\langle$ offset $-$ global sine $\rangle$ [MJy/sr]',
           title=f'Spatial trend vs ecl_lat  (peak-to-peak {lat_pp:.4f}, SEM {lat_sem:.5f})')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 2: residual vs ecl_lon.
    ax = axes[1]
    ax.errorbar(lon_c, lon_m, yerr=lon_s, fmt='o-', color='C3',
                lw=1.5, ms=5, capsize=3,
                label=f'binned mean (N ~ {int(np.nanmedian(lon_n))} / bin)')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel='ecliptic longitude [deg]',
           ylabel=r'$\langle$ offset $-$ global sine $\rangle$ [MJy/sr]',
           title=f'Spatial trend vs ecl_lon  (peak-to-peak {lon_pp:.4f}, SEM {lon_sem:.5f})')
    ax.legend()
    ax.grid(alpha=0.3)

    # Panel 3: 2-D map of the residual.
    ax = axes[2]
    v = np.nanmax(np.abs(grid))
    im = ax.imshow(
        grid.T, origin='lower', aspect='auto', cmap='RdBu_r',
        extent=(map_edges_lon[0], map_edges_lon[-1],
                map_edges_lat[0], map_edges_lat[-1]),
        vmin=-v, vmax=v,
    )
    ax.set(xlabel='ecliptic longitude [deg]',
           ylabel='ecliptic latitude [deg]',
           title='Static spatial map  (residual after global sine)')
    plt.colorbar(im, ax=ax, label='residual [MJy/sr]')

    fig.suptitle(
        f'det{detector} ch{channel}  |  global sine removed; per-exposure residuals '
        f'binned in static sky coords',
        y=1.02,
    )
    fig.tight_layout()
    out = fig_path(f'det{detector}_ch{channel}__clean_spatial.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    p.add_argument('--channel', type=int, default=17)
    args = p.parse_args()
    main(args.detector, args.channel)
