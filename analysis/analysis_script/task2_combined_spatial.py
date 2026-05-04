"""Task 2: Combine channels 1-17 after removing per-channel spectral DC + scale.

Each channel has a different spectral sensitivity, so the raw offset level
(C) and the annual amplitude (A) from the global sine fit differ between
channels. We normalize per-channel using the fit, then concatenate all
exposures into one long DataFrame and rebuild the "clean spatial" plots
on this combined set. With ~17x more rows per spatial bin, any coherent
zodi spatial structure shows up more cleanly than in a single channel.

Normalization options
---------------------
- 'subtract C'  :  offset - C_channel   (preserves amplitude units, removes
                                         inter-channel DC)
- 'standardise' :  (offset - C_channel) / A_channel (unitless; normalizes
                                                     both DC and amplitude)

Default: subtract C; also save a standardised variant for comparison.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_SELFCAL_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))
if _SELFCAL_ROOT not in sys.path:
    sys.path.insert(0, _SELFCAL_ROOT)

from zodi_utils import sine_model, fit_sine, SIDEREAL_YEAR_DAYS, data_path, fig_path


def clip_outliers(y, low=0.5, high=99.5):
    lo, hi = np.nanpercentile(y, [low, high])
    return (y >= lo) & (y <= hi)


def binned_mean(x, y, edges, min_count=20):
    idx = np.digitize(x, edges) - 1
    n = len(edges) - 1
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n, np.nan)
    sems = np.full(n, np.nan)
    for b in range(n):
        m = idx == b
        c = int(m.sum())
        if c >= min_count:
            means[b] = np.mean(y[m])
            sems[b] = np.std(y[m]) / np.sqrt(c)
    return centers, means, sems


def grid_2d_mean(x, y, z, x_edges, y_edges, min_count=20):
    ix = np.digitize(x, x_edges) - 1
    iy = np.digitize(y, y_edges) - 1
    nx, ny = len(x_edges) - 1, len(y_edges) - 1
    sums = np.zeros((nx, ny))
    counts = np.zeros((nx, ny), dtype=int)
    mask = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    np.add.at(sums, (ix[mask], iy[mask]), z[mask])
    np.add.at(counts, (ix[mask], iy[mask]), 1)
    return np.where(counts >= min_count, sums / np.maximum(counts, 1), np.nan), counts


def main(detector):
    cache = data_path(f'multichannel_det{detector}.pkl')
    df = pd.read_pickle(cache)
    print(f'Loaded {len(df)} (exposure x channel) rows, '
          f'{df["channel"].nunique()} channels, '
          f'{df["MJD_AVG"].nunique()} unique MJDs.')

    # Per-channel global sine fits -> C, A, phi.
    fits = {}
    for ch, g in df.groupby('channel'):
        mask = clip_outliers(g['mean_offset'].values)
        f = fit_sine(g['MJD_AVG'].values[mask], g['mean_offset'].values[mask])
        fits[ch] = f
        print(f'  ch{ch:2d}: C={f["C"]:.4f}  A={f["A"]:.4f}  '
              f'phi={np.degrees(f["phi"]):+.1f}  resid_std={f["residual_std"]:.4f}')

    # Annotate each row with its per-channel fit parameters and the
    # per-exposure time-sine prediction, then compute residuals two ways.
    df = df.copy()
    df['C'] = df['channel'].map(lambda c: fits[c]['C'])
    df['A'] = df['channel'].map(lambda c: fits[c]['A'])
    df['phi'] = df['channel'].map(lambda c: fits[c]['phi'])
    df['sine_pred'] = sine_model(df['MJD_AVG'].values,
                                 df['A'].values, df['phi'].values,
                                 df['C'].values)
    df['resid_sub'] = df['mean_offset'] - df['sine_pred']
    df['resid_std'] = df['resid_sub'] / df['A']

    # Outlier clip + drop any residual rows with bad metadata.
    mask = clip_outliers(df['resid_sub'].values)
    df = df[mask].reset_index(drop=True)

    # Spatial plots built on the combined df, using resid_sub
    # (subtract C + global annual sine => units preserved).
    ecl_lat = df['ecl_lat'].values
    ecl_lon = np.mod(df['ecl_lon'].values, 360.0)
    resid = df['resid_sub'].values

    lat_lo, lat_hi = np.percentile(ecl_lat, [0.5, 99.5])
    lat_edges = np.linspace(lat_lo, lat_hi, 21)
    lon_edges = np.linspace(0, 360, 37)

    lat_c, lat_m, lat_s = binned_mean(ecl_lat, resid, lat_edges, min_count=80)
    lon_c, lon_m, lon_s = binned_mean(ecl_lon, resid, lon_edges, min_count=80)

    # 2-D map.
    grid, grid_n = grid_2d_mean(ecl_lon, ecl_lat, resid,
                                lon_edges, lat_edges, min_count=60)

    lat_pp = np.nanmax(lat_m) - np.nanmin(lat_m)
    lon_pp = np.nanmax(lon_m) - np.nanmin(lon_m)
    print(f'\nCombined (ch {df["channel"].min()}..{df["channel"].max()}):')
    print(f'  ecl_lat trend: peak-to-peak {lat_pp:.5f}  '
          f'median SEM {np.nanmedian(lat_s):.6f}  '
          f'SNR {lat_pp/np.nanmedian(lat_s):.1f}')
    print(f'  ecl_lon trend: peak-to-peak {lon_pp:.5f}  '
          f'median SEM {np.nanmedian(lon_s):.6f}  '
          f'SNR {lon_pp/np.nanmedian(lon_s):.1f}')

    # Plot.
    fig, axes = plt.subplots(2, 3, figsize=(19, 10))

    # row 1: combined spatial trends.
    ax = axes[0, 0]
    ax.errorbar(lat_c, lat_m, yerr=lat_s, fmt='o-', color='C0',
                lw=1.5, ms=5, capsize=3,
                label=f'combined (N~{int(len(df)/(len(lat_edges)-1))}/bin)')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel='ecliptic latitude [deg]',
           ylabel=r'$\langle$ offset - (C_ch + A_ch sin) $\rangle$ [MJy/sr]',
           title=f'(a) combined spatial trend vs ecl_lat  '
                 f'(peak-to-peak {lat_pp:.5f})')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.errorbar(lon_c, lon_m, yerr=lon_s, fmt='o-', color='C3',
                lw=1.5, ms=5, capsize=3,
                label=f'combined (N~{int(len(df)/(len(lon_edges)-1))}/bin)')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel='ecliptic longitude [deg]',
           ylabel=r'$\langle$ offset - (C_ch + A_ch sin) $\rangle$ [MJy/sr]',
           title=f'(b) combined spatial trend vs ecl_lon  '
                 f'(peak-to-peak {lon_pp:.5f})')
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[0, 2]
    v = np.nanmax(np.abs(grid))
    im = ax.imshow(grid.T, origin='lower', aspect='auto', cmap='RdBu_r',
                   extent=(lon_edges[0], lon_edges[-1],
                           lat_edges[0], lat_edges[-1]),
                   vmin=-v, vmax=v)
    ax.set(xlabel='ecliptic longitude [deg]',
           ylabel='ecliptic latitude [deg]',
           title='(c) combined 2-D residual map')
    plt.colorbar(im, ax=ax, label='residual [MJy/sr]')

    # row 2: per-channel comparison of the same projection.
    ax = axes[1, 0]
    for ch, g in df.groupby('channel'):
        c, m, _ = binned_mean(g['ecl_lat'].values, g['resid_sub'].values,
                              lat_edges, min_count=20)
        ax.plot(c, m, lw=0.8, alpha=0.5, color=plt.cm.viridis(ch / 17))
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel='ecliptic latitude [deg]',
           ylabel='residual [MJy/sr]',
           title='(d) per-channel ecl_lat curves (viridis=channel)')
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for ch, g in df.groupby('channel'):
        c, m, _ = binned_mean(np.mod(g['ecl_lon'].values, 360.0),
                              g['resid_sub'].values, lon_edges, min_count=20)
        ax.plot(c, m, lw=0.8, alpha=0.5, color=plt.cm.viridis(ch / 17))
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel='ecliptic longitude [deg]',
           ylabel='residual [MJy/sr]',
           title='(e) per-channel ecl_lon curves')
    ax.grid(alpha=0.3)

    ax = axes[1, 2]
    chans = sorted(fits.keys())
    Cs = [fits[c]['C'] for c in chans]
    As = [fits[c]['A'] for c in chans]
    ax2 = ax.twinx()
    ax.plot(chans, Cs, 'o-', color='C0', label='C (DC)')
    ax2.plot(chans, As, 's--', color='C3', label='A (amplitude)')
    ax.set(xlabel='channel', ylabel='C [MJy/sr]',
           title='(f) Per-channel C and A from the global sine fit')
    ax2.set_ylabel('A [MJy/sr]', color='C3')
    ax.grid(alpha=0.3)

    fig.suptitle(
        f'Combined-channel spatial signal  det{detector}  '
        f'(channels {df["channel"].min()}..{df["channel"].max()})',
        y=1.01,
    )
    fig.tight_layout()
    out = fig_path(f'det{detector}__combined_spatial.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    args = p.parse_args()
    main(args.detector)
