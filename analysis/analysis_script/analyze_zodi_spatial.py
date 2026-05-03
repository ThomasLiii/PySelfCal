"""Decouple spatial zodi structure from the annual time modulation.

Strategy
--------
The offset term at a fixed pointing can be modeled as

    offset(t) = C(lon, lat) + A(lon, lat) * sin(2 pi t / T + phi(lon, lat))

so the *bin-wise DC term* C(lon, lat) is a time-averaged map of zodi brightness
that is not biased by when a given pointing happened to be observed. We do the
following:

  1. Global annual sine fit for reference.
  2. Bin exposures in ecliptic (helio_lon, ecl_lat) and fit a per-bin sine.
     Extract C (time-averaged zodi), A (annual amplitude), phi (phase).
  3. Plot the resulting C, A, phi maps and their 1-D projections against
     candidate zodi geometry angles (ecliptic latitude, helio-ecliptic
     longitude, solar elongation).
  4. Also project against (RA, DEC) for a sanity check.

The "straight-line" diagnostic plots worth checking:
  - log|C - C_pole| vs |sin(ecl_lat)|   -> Kelsall-style exponential decay
  - phi vs helio_lon                    -> dust-plane tilt signature
  - C vs elongation (if wide coverage)  -> radial profile of the cloud

For the NEP field |ecl_lat| is saturated near 90 deg, so the first plot will
be nearly a point cloud; the phi vs helio_lon view usually gives the cleanest
linear behaviour for polar-field data.

Usage
-----
    python analyze_zodi_spatial.py --detector 4 --channel 14
    python analyze_zodi_spatial.py --detector 4 --channel 14 --n-bins 12

Output plots are written alongside this script as PNG files tagged with the
detector and channel.
"""
import argparse
import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from zodi_utils import (
    SIDEREAL_YEAR_DAYS,
    sine_model,
    fit_sine,
    fit_sine_per_bin,
    bin_edges,
    assign_2d_bin,
    data_path,
    fig_path as _fig_path,
)


def cache_path(detector, channel):
    return data_path(f'exposure_df_det{detector}_ch{channel}.pkl')


def fig_path(detector, channel, name):
    return _fig_path(f'det{detector}_ch{channel}__{name}.png')


def savefig(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    print(f'  wrote {path}')


# ---------------------------------------------------------------------------
# 0. Survey overview
# ---------------------------------------------------------------------------
def plot_survey_overview(df, detector, channel):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax = axes[0, 0]
    ax.scatter(df['CRVAL1'], df['CRVAL2'], s=2, alpha=0.3, c=df['MJD_AVG'], cmap='viridis')
    ax.set(xlabel='RA [deg]', ylabel='DEC [deg]', title='Pointings coloured by MJD')

    ax = axes[0, 1]
    ax.scatter(df['ecl_lon'], df['ecl_lat'], s=2, alpha=0.3, c=df['MJD_AVG'], cmap='viridis')
    ax.set(xlabel='ecl_lon [deg]', ylabel='ecl_lat [deg]', title='Ecliptic pointings')

    ax = axes[1, 0]
    ax.hist(df['elongation'], bins=60)
    ax.set(xlabel='Solar elongation [deg]', ylabel='# exposures')

    ax = axes[1, 1]
    ax.hist(df['helio_lon'], bins=60)
    ax.set(xlabel='Helio-ecliptic lon [deg]', ylabel='# exposures')

    fig.suptitle(f'Survey overview  det{detector} ch{channel}')
    savefig(fig, fig_path(detector, channel, 'survey_overview'))


# ---------------------------------------------------------------------------
# 1. Global temporal fit
# ---------------------------------------------------------------------------
def global_temporal_fit(df, detector, channel):
    t = df['MJD_AVG'].values
    y = df['mean_offset'].values
    fit = fit_sine(t, y)
    print(f'Global sine fit: C={fit["C"]:.4f}  A={fit["A"]:.4f}  '
          f'phi={np.degrees(fit["phi"]):.1f} deg  resid_std={fit["residual_std"]:.4f}')

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(t, y, s=3, alpha=0.2, label='exposures')
    t_fit = np.linspace(t.min(), t.max(), 1000)
    ax.plot(t_fit, sine_model(t_fit, fit['A'], fit['phi'], fit['C']),
            color='red', lw=1.5, label='global annual sine')
    ax.set(xlabel='MJD', ylabel='mean offset',
           title=f'Global temporal fit  det{detector} ch{channel}')
    ax.legend()
    savefig(fig, fig_path(detector, channel, 'global_temporal_fit'))

    return fit


def plot_residual_vs_space(df, global_fit, detector, channel):
    """Residuals after the global sine removed -- any spatial pattern left is
    evidence that the sine DC/amp varies with location.

    Three panels: residual vs ecl_lon, ecl_lat, and helio_lon. The helio_lon
    panel is the most physically meaningful for zodi work since helio_lon
    parameterises the target's sky direction relative to the Sun -- which is
    what sets zodi brightness.
    """
    t = df['MJD_AVG'].values
    model = sine_model(t, global_fit['A'], global_fit['phi'], global_fit['C'])
    resid = df['mean_offset'].values - model
    ylo, yhi = np.nanpercentile(resid, [0.5, 99.5])

    panels = [
        ('ecl_lon', 'ecl_lon [deg]', 'helio_lon', 'helio_lon [deg]', 'twilight'),
        ('ecl_lat', 'ecl_lat [deg]', 'helio_lon', 'helio_lon [deg]', 'twilight'),
        ('helio_lon', 'helio_lon [deg]', 'ecl_lat', 'ecl_lat [deg]', 'viridis'),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (xcol, xlabel, ccol, clabel, cmap) in zip(axes, panels):
        sc = ax.scatter(df[xcol], resid, s=2, alpha=0.3,
                        c=df[ccol], cmap=cmap)
        ax.axhline(0, color='k', lw=0.7, ls='--')
        ax.set(xlabel=xlabel, ylabel='residual after global sine',
               ylim=(ylo, yhi))
        plt.colorbar(sc, ax=ax, label=clabel)
    fig.suptitle(f'Global-sine residuals vs ecliptic coords  det{detector} ch{channel}')
    savefig(fig, fig_path(detector, channel, 'residual_vs_space'))


# ---------------------------------------------------------------------------
# 2. Per-bin temporal fit: the core decoupling step
# ---------------------------------------------------------------------------
def per_bin_temporal_fit(df, detector, channel, n_bins=10, min_points=40,
                         xcol='helio_lon', ycol='ecl_lat'):
    x = df[xcol].values
    y_space = df[ycol].values
    t = df['MJD_AVG'].values
    y = df['mean_offset'].values

    xe = bin_edges(x, n_bins)
    ye = bin_edges(y_space, n_bins)

    bin_flat, ix, iy = assign_2d_bin(x, y_space, xe, ye)
    fits = fit_sine_per_bin(t, y, bin_flat, min_points=min_points)

    if fits.empty:
        print('WARNING: no bins met min_points threshold.')
        return fits, xe, ye, xcol, ycol

    fits['ix'] = fits['bin'] // n_bins
    fits['iy'] = fits['bin'] % n_bins
    fits[xcol + '_center'] = 0.5 * (xe[fits['ix'].values] + xe[fits['ix'].values + 1])
    fits[ycol + '_center'] = 0.5 * (ye[fits['iy'].values] + ye[fits['iy'].values + 1])
    fits['phi_deg'] = np.degrees(fits['phi'])

    fits.attrs = {'xcol': xcol, 'ycol': ycol, 'n_bins': n_bins}
    out_csv = _fig_path(f'det{detector}_ch{channel}__per_bin_fits.csv')
    fits.to_csv(out_csv, index=False)
    print(f'  wrote {out_csv}  ({len(fits)} non-empty bins)')
    return fits, xe, ye, xcol, ycol


def _imshow_from_bins(ax, fits, value_col, ix_key, iy_key, n_bins,
                      xe, ye, xlabel, ylabel, title, cmap='viridis',
                      vmin=None, vmax=None):
    grid = np.full((n_bins, n_bins), np.nan)
    grid[fits[ix_key].values, fits[iy_key].values] = fits[value_col].values
    # Display with rows = iy (y), cols = ix (x) so axes look natural.
    im = ax.imshow(
        grid.T, origin='lower', aspect='auto', cmap=cmap,
        extent=(xe[0], xe[-1], ye[0], ye[-1]),
        vmin=vmin, vmax=vmax,
    )
    ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
    plt.colorbar(im, ax=ax)


def plot_per_bin_maps(fits, xe, ye, xcol, ycol, detector, channel, n_bins):
    if fits.empty:
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    # DC term (time-averaged zodi)
    vmin, vmax = np.nanpercentile(fits['C'], [5, 95])
    _imshow_from_bins(axes[0], fits, 'C', 'ix', 'iy', n_bins,
                      xe, ye, f'{xcol} [deg]', f'{ycol} [deg]',
                      f'C (time-averaged zodi)  det{detector} ch{channel}',
                      vmin=vmin, vmax=vmax)

    # Amplitude
    vmin, vmax = np.nanpercentile(fits['A'], [5, 95])
    _imshow_from_bins(axes[1], fits, 'A', 'ix', 'iy', n_bins,
                      xe, ye, f'{xcol} [deg]', f'{ycol} [deg]',
                      'A (annual amplitude)',
                      vmin=vmin, vmax=vmax)

    # Phase
    phi_deg_wrap = np.mod(fits['phi_deg'].values + 180, 360) - 180
    fits_phase = fits.copy()
    fits_phase['phi_deg_wrap'] = phi_deg_wrap
    _imshow_from_bins(axes[2], fits_phase, 'phi_deg_wrap', 'ix', 'iy', n_bins,
                      xe, ye, f'{xcol} [deg]', f'{ycol} [deg]',
                      'phi (annual phase) [deg]', cmap='twilight',
                      vmin=-180, vmax=180)

    savefig(fig, fig_path(detector, channel, 'per_bin_maps'))


# ---------------------------------------------------------------------------
# 3. 1-D projections vs candidate "zodi geometry angles"
# ---------------------------------------------------------------------------
def plot_dc_vs_angles(fits, detector, channel):
    """Project C, A, phi against plausible zodi geometry axes."""
    if fits.empty:
        return

    xcol = fits.attrs.get('xcol', 'helio_lon')
    ycol = fits.attrs.get('ycol', 'ecl_lat')
    x = fits[xcol + '_center'].values
    y = fits[ycol + '_center'].values
    C = fits['C'].values
    A = fits['A'].values
    phi_deg = np.degrees(fits['phi'].values)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # Row 1: DC zodi (C) vs each angle.
    ax = axes[0, 0]
    sc = ax.scatter(y, C, c=x, cmap='twilight', s=40)
    ax.set(xlabel=f'{ycol} [deg]', ylabel='C (time-averaged offset)',
           title='DC zodi vs ecl_lat')
    plt.colorbar(sc, ax=ax, label=f'{xcol} [deg]')

    ax = axes[0, 1]
    sc = ax.scatter(x, C, c=y, cmap='viridis', s=40)
    ax.set(xlabel=f'{xcol} [deg]', ylabel='C',
           title='DC zodi vs helio-ecliptic lon')
    plt.colorbar(sc, ax=ax, label=f'{ycol} [deg]')

    # The "straight line" candidate for an exponential β-decay.
    ax = axes[0, 2]
    sin_abs_beta = np.abs(np.sin(np.radians(y)))
    C_floor = np.nanmin(C)
    dC = C - C_floor + 1e-6
    ax.scatter(sin_abs_beta, np.log(dC), s=40, c=x, cmap='twilight')
    ax.set(xlabel='|sin(ecl_lat)|',
           ylabel='log(C - C_min)',
           title='Kelsall-style: linear if exp(-|sinβ|/β0)')

    # Row 2: Amplitude and phase.
    ax = axes[1, 0]
    sc = ax.scatter(y, A, c=x, cmap='twilight', s=40)
    ax.set(xlabel=f'{ycol} [deg]', ylabel='A',
           title='Annual amplitude vs ecl_lat')
    plt.colorbar(sc, ax=ax, label=f'{xcol} [deg]')

    # Phase vs helio-ecliptic longitude is often the cleanest straight line
    # in NEP data because it tracks the dust-plane tilt.
    ax = axes[1, 1]
    ax.scatter(x, phi_deg, c=y, cmap='viridis', s=40)
    ax.set(xlabel=f'{xcol} [deg]', ylabel='phi [deg]',
           title='Annual phase vs helio-ecliptic lon')

    # A/C ratio -- normalizes out overall brightness.
    ax = axes[1, 2]
    ax.scatter(y, A / np.maximum(C, 1e-6), c=x, cmap='twilight', s=40)
    ax.set(xlabel=f'{ycol} [deg]', ylabel='A / C',
           title='Fractional annual modulation')

    fig.suptitle(f'Per-bin fit projections  det{detector} ch{channel}')
    savefig(fig, fig_path(detector, channel, 'dc_vs_angles'))


def plot_A_vs_helio_lon(fits, detector, channel):
    """A vs helio-ecliptic longitude, plus quadrature components.

    At fixed ecl_lat the annual modulation amplitude should vary smoothly with
    helio_lon because the zodi dust plane is tilted relative to the ecliptic.
    Decomposing each bin's sine into quadrature components
        S = A sin(phi),   T = A cos(phi)
    often makes the helio_lon dependence fall onto a single smooth curve
    (or a straight line in the (S, T) plane coloured by helio_lon).
    """
    if fits.empty:
        return

    x = fits['helio_lon_center'].values
    y = fits['ecl_lat_center'].values
    A = fits['A'].values
    phi = fits['phi'].values
    S = A * np.sin(phi)
    T = A * np.cos(phi)

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    sc = ax.scatter(x, A, c=y, cmap='viridis', s=40)
    ax.set(xlabel='helio-ecliptic lon [deg]', ylabel='A (annual amplitude)',
           title='A vs helio_lon (coloured by ecl_lat)')
    plt.colorbar(sc, ax=ax, label='ecl_lat [deg]')

    ax = axes[0, 1]
    sc = ax.scatter(x, np.degrees(phi), c=y, cmap='viridis', s=40)
    ax.set(xlabel='helio-ecliptic lon [deg]', ylabel='phi [deg]',
           title='phi vs helio_lon (coloured by ecl_lat)')
    plt.colorbar(sc, ax=ax, label='ecl_lat [deg]')

    ax = axes[1, 0]
    sc = ax.scatter(x, S, c=y, cmap='viridis', s=40, label='S = A sin(phi)')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set(xlabel='helio-ecliptic lon [deg]', ylabel='A sin(phi)',
           title='Quadrature: S vs helio_lon')
    plt.colorbar(sc, ax=ax, label='ecl_lat [deg]')

    ax = axes[1, 1]
    sc = ax.scatter(x, T, c=y, cmap='viridis', s=40, label='T = A cos(phi)')
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set(xlabel='helio-ecliptic lon [deg]', ylabel='A cos(phi)',
           title='Quadrature: T vs helio_lon')
    plt.colorbar(sc, ax=ax, label='ecl_lat [deg]')

    fig.suptitle(f'Annual modulation vs helio_lon  det{detector} ch{channel}')
    savefig(fig, fig_path(detector, channel, 'A_vs_helio_lon'))


# ---------------------------------------------------------------------------
# 4. Static spatial map built from per-bin DC
# ---------------------------------------------------------------------------
def plot_ecl_scatter(df, detector, channel):
    """The raw mean_offset painted on (helio_lon, ecl_lat) and on (RA, DEC)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    vmin, vmax = np.nanpercentile(df['mean_offset'], [5, 95])

    ax = axes[0]
    sc = ax.scatter(df['helio_lon'], df['ecl_lat'], c=df['mean_offset'],
                    s=3, alpha=0.6, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set(xlabel='helio-ecliptic lon [deg]', ylabel='ecliptic lat [deg]',
           title='mean_offset on ecliptic coords (not time-decoupled)')
    plt.colorbar(sc, ax=ax, label='offset')

    ax = axes[1]
    sc = ax.scatter(df['CRVAL1'], df['CRVAL2'], c=df['mean_offset'],
                    s=3, alpha=0.6, cmap='viridis', vmin=vmin, vmax=vmax)
    ax.set(xlabel='RA [deg]', ylabel='DEC [deg]',
           title='mean_offset on (RA, DEC)')
    plt.colorbar(sc, ax=ax, label='offset')

    fig.suptitle(f'Raw (time-coupled) spatial offset  det{detector} ch{channel}')
    savefig(fig, fig_path(detector, channel, 'mean_offset_ecl_radec'))


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------
def run(detector, channel, n_bins=10, min_points=40):
    path = cache_path(detector, channel)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f'Cache not found: {path}\n'
            f'Run:  python build_metadata.py --detector {detector} --channel {channel}'
        )
    df = pd.read_pickle(path)
    print(f'Loaded {len(df)} exposures from {path}')

    plot_survey_overview(df, detector, channel)
    plot_ecl_scatter(df, detector, channel)
    global_fit = global_temporal_fit(df, detector, channel)
    plot_residual_vs_space(df, global_fit, detector, channel)

    fits, xe, ye, xcol, ycol = per_bin_temporal_fit(
        df, detector, channel, n_bins=n_bins, min_points=min_points,
        xcol='helio_lon', ycol='ecl_lat',
    )
    plot_per_bin_maps(fits, xe, ye, xcol, ycol, detector, channel, n_bins)
    plot_dc_vs_angles(fits, detector, channel)
    plot_A_vs_helio_lon(fits, detector, channel)

    summary = {
        'detector': detector,
        'channel': channel,
        'n_exposures': len(df),
        'n_bins_per_axis': n_bins,
        'global_fit': {k: v for k, v in global_fit.items() if k != 'n'},
    }
    summary_path = _fig_path(f'det{detector}_ch{channel}__summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'Wrote {summary_path}')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--detector', type=int, default=4)
    p.add_argument('--channel', type=int, default=14)
    p.add_argument('--n-bins', type=int, default=10,
                   help='# bins along each ecliptic axis for per-bin sine fits')
    p.add_argument('--min-points', type=int, default=40,
                   help='minimum exposures in a bin to fit a sine')
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run(args.detector, args.channel,
        n_bins=args.n_bins, min_points=args.min_points)
