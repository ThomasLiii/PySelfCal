"""Decompose the col0 - col2 gradient signal into:

  1. A detector-fixed component:   a constant offset in detector coordinates
     (scattered light, bias gradient, anything that always points "col0 high,
     col2 low" regardless of pointing/time).
  2. A spatial-temporal component: the part that varies with sky direction
     and time of observation -- the real zodi/signal content.
  3. A residual                 :   per-exposure scatter left after (1)+(2).

Two models, selected by --model:

  --model harmonic  (default, parametric; fewer d.o.f., low overfit risk)
      grad = a_det
           + sum_{k=1..K_hl} [ b_k sin(k*helio_lon) + c_k cos(k*helio_lon) ]
           + d * (ecl_lat - <ecl_lat>)
           + e1 * sin(2pi*MJD/year)  +  e2 * cos(2pi*MJD/year)
           + (interaction) f1 * sin(helio_lon) * (ecl_lat - <ecl_lat>)
                         + f2 * cos(helio_lon) * (ecl_lat - <ecl_lat>)
      Fit by ordinary least squares. ~11-17 free parameters depending on K_hl.

  --model binned  (non-parametric; more flexible, overfit risk)
      Lookup table: grad - a_det averaged in a 3-D bin
      (helio_lon x ecl_lat x MJD), with 2-D and global fallbacks for sparse
      bins. No assumed functional form.

Outputs go to det{det}_ch{ch}__numcol_decomp_{model}.png so the two plots
don't overwrite each other.
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_SELFCAL_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))
if _SELFCAL_ROOT not in sys.path:
    sys.path.insert(0, _SELFCAL_ROOT)

from SelfCal.SPHERExUtility import make_stripped_chunk_valid_mask
from zodi_utils import data_path, fig_path

CAL_PATH = ('/mnt/md124/thomasli/selfcal/outputs/'
            'SPHEREx_nep_qr2_det{det}_6p2arcsec/calibration/'
            'cal_Detector{det}_NumSub10_NumCh34_NumCol3_Ch{ch}_'
            'damp0p1_reg0p1_outThresh5_sigma2.h5')

NUM_SUB, NUM_CH, NUM_COL = 10, 34, 3
TOT_SUB = NUM_SUB * NUM_CH + 2


def load_grad(detector, channel):
    """Return per-exposure col0-col2 gradient (averaged over valid subchannels)."""
    with h5py.File(CAL_PATH.format(det=detector, ch=channel), 'r') as f:
        off = f['offset'][:].reshape(-1, TOT_SUB, NUM_COL)
    mask = make_stripped_chunk_valid_mask(
        ch=[channel], num_subchannels=NUM_SUB, num_channels=NUM_CH,
        num_columns=NUM_COL, subchannel_padding=0,
    ).reshape(TOT_SUB, NUM_COL)
    valid_sub = np.where(mask.any(axis=1))[0]
    col_means = off[:, valid_sub, :].mean(axis=1)   # (N, 3)
    return col_means[:, 0] - col_means[:, 2], col_means


SIDEREAL_YEAR_DAYS = 365.25


def build_harmonic_design(helio_lon_deg, ecl_lat_deg, mjd, ecl_lat_ref, K_hl=4):
    """Parametric design matrix for the harmonic decomposition.

    Columns:
      0            : constant  -> a_det
      1..2K_hl     : sin(k*hl), cos(k*hl)  for k=1..K_hl
      next         : (ecl_lat - ecl_lat_ref)
      next 2       : sin(2pi*MJD/year), cos(2pi*MJD/year)
      next 2       : sin(hl)*(ecl_lat - ref), cos(hl)*(ecl_lat - ref)
    """
    hl = np.radians(helio_lon_deg)
    lat = ecl_lat_deg - ecl_lat_ref
    t_phase = 2 * np.pi * mjd / SIDEREAL_YEAR_DAYS

    cols = [np.ones_like(hl)]
    names = ['a_det']
    for k in range(1, K_hl + 1):
        cols.extend([np.sin(k * hl), np.cos(k * hl)])
        names.extend([f'sin_{k}hl', f'cos_{k}hl'])
    cols.append(lat)
    names.append('ecl_lat_slope')
    cols.extend([np.sin(t_phase), np.cos(t_phase)])
    names.extend(['sin_annual_t', 'cos_annual_t'])
    cols.extend([np.sin(hl) * lat, np.cos(hl) * lat])
    names.extend(['sin_hl_x_lat', 'cos_hl_x_lat'])
    return np.column_stack(cols), names


def binned_mean_3d(helio_lon, ecl_lat, mjd, y,
                   n_hl=24, n_lat=6, n_t=12, min_count=8):
    """Non-parametric spatial-temporal model.

    Bin `y` in (helio_lon, ecl_lat, MJD) and return the per-exposure bin mean
    (i.e. every exposure is assigned the mean of its 3-D bin). Sparse bins
    (count < min_count) fall back to the 2-D (helio_lon, ecl_lat) bin mean,
    and then to the global mean, so every exposure always gets a value.
    """
    hl_edges = np.linspace(-180, 180, n_hl + 1)
    lat_edges = np.linspace(np.percentile(ecl_lat, 0.5),
                            np.percentile(ecl_lat, 99.5), n_lat + 1)
    t_edges = np.linspace(mjd.min(), mjd.max(), n_t + 1)

    i_hl = np.clip(np.digitize(helio_lon, hl_edges) - 1, 0, n_hl - 1)
    i_lat = np.clip(np.digitize(ecl_lat, lat_edges) - 1, 0, n_lat - 1)
    i_t = np.clip(np.digitize(mjd, t_edges) - 1, 0, n_t - 1)

    # 3-D bin mean.
    sums3 = np.zeros((n_hl, n_lat, n_t))
    cnts3 = np.zeros_like(sums3, dtype=int)
    np.add.at(sums3, (i_hl, i_lat, i_t), y)
    np.add.at(cnts3, (i_hl, i_lat, i_t), 1)
    means3 = np.where(cnts3 >= min_count, sums3 / np.maximum(cnts3, 1), np.nan)

    # 2-D fallback on (helio_lon, ecl_lat).
    sums2 = np.zeros((n_hl, n_lat))
    cnts2 = np.zeros_like(sums2, dtype=int)
    np.add.at(sums2, (i_hl, i_lat), y)
    np.add.at(cnts2, (i_hl, i_lat), 1)
    means2 = np.where(cnts2 >= min_count, sums2 / np.maximum(cnts2, 1), np.nan)

    global_mean = float(np.mean(y))

    est = means3[i_hl, i_lat, i_t]
    fallback_2d = means2[i_hl, i_lat]
    est = np.where(np.isnan(est), fallback_2d, est)
    est = np.where(np.isnan(est), global_mean, est)
    return est


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


def main(detector, channel, model='harmonic'):
    grad, col_means = load_grad(detector, channel)
    md = pd.read_pickle(data_path(f'exposure_df_det{detector}_ch{channel}.pkl'))
    assert len(md) == len(grad)

    keep = np.abs(grad - np.median(grad)) < 5.0 * np.std(grad)
    grad = grad[keep]
    md = md.loc[keep].reset_index(drop=True)
    print(f'{len(md)} exposures after 5-sigma outlier clip.')
    print(f'Model: {model}')

    if model == 'harmonic':
        lat_ref = float(np.median(md['ecl_lat']))
        X, names = build_harmonic_design(
            md['helio_lon'].values, md['ecl_lat'].values,
            md['MJD_AVG'].values, lat_ref, K_hl=4,
        )
        beta, *_ = np.linalg.lstsq(X, grad, rcond=None)
        a_det = float(beta[0])
        spatial_temporal = X[:, 1:] @ beta[1:]
        residual = grad - X @ beta
        grad_pred = a_det + spatial_temporal

        print(f'\nHarmonic fit ({len(beta)} parameters):')
        for name, coef in zip(names, beta):
            print(f'  {name:18s}  {coef:+.5f}')
    elif model == 'binned':
        a_det = float(np.mean(grad))
        spatial_temporal = binned_mean_3d(
            md['helio_lon'].values, md['ecl_lat'].values,
            md['MJD_AVG'].values, grad - a_det,
            n_hl=24, n_lat=6, n_t=12, min_count=8,
        )
        residual = grad - a_det - spatial_temporal
        grad_pred = a_det + spatial_temporal
    else:
        raise ValueError(f'unknown model: {model}')

    var_total = np.var(grad)
    var_st = np.var(spatial_temporal)
    var_res = np.var(residual)
    print()
    print('=== Decomposition of col0 - col2 ===')
    print(f'                      mean [MJy/sr]    std [MJy/sr]    var share')
    print(f'  total signal     :  {grad.mean():+.5f}        {np.std(grad):.5f}       1.000')
    print(f'  detector-fixed   :  {a_det:+.5f}        0.00000          --- (constant)')
    print(f'  spatial-temporal :  {spatial_temporal.mean():+.5f}        '
          f'{np.std(spatial_temporal):.5f}       {var_st/var_total:.3f}')
    print(f'  residual         :  {residual.mean():+.5f}        '
          f'{np.std(residual):.5f}       {var_res/var_total:.3f}')

    # -------------------- Plots --------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (a) Stacked histogram: total, detector-fixed (arrow), spatial-temp.
    ax = axes[0, 0]
    bmax = np.percentile(np.abs(grad), 99)
    bins = np.linspace(-bmax, bmax, 80)
    ax.hist(grad, bins=bins, alpha=0.5, color='C3', label=f'total  (mean {grad.mean():+.4f})')
    ax.hist(residual, bins=bins, alpha=0.5, color='grey',
            label=f'residual  (std {np.std(residual):.4f})')
    ax.axvline(a_det, color='C0', lw=2,
               label=f'a_det = {a_det:+.4f}')
    ax.axvline(0, color='k', lw=0.7, ls='--')
    ax.set(xlabel='col0 - col2 [MJy/sr]', ylabel='# exposures',
           title='(a) Signal vs residual distribution')
    ax.legend(loc='upper left', fontsize=9)

    # (b) Spatial-temporal component vs helio_lon.
    ax = axes[0, 1]
    edges = np.linspace(-180, 180, 37)
    c, m, s = binned_mean(md['helio_lon'].values, spatial_temporal, edges)
    ax.errorbar(c, m, yerr=s, fmt='o-', color='C2', lw=1.5, ms=5, capsize=3,
                label='fitted spatial-temporal')
    c, m_raw, _ = binned_mean(md['helio_lon'].values, grad - a_det, edges)
    ax.plot(c, m_raw, '--', color='C3', lw=1.2,
            label='(total - a_det) binned mean')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel='helio-ecliptic lon [deg]',
           ylabel='gradient - a_det [MJy/sr]',
           title='(b) Spatial-temporal part vs helio_lon')
    ax.legend(loc='upper right', fontsize=9)

    # (c) Spatial-temporal component vs ecl_lat.
    ax = axes[0, 2]
    edges = np.linspace(np.percentile(md['ecl_lat'], 1),
                        np.percentile(md['ecl_lat'], 99), 21)
    c, m, s = binned_mean(md['ecl_lat'].values, spatial_temporal, edges)
    ax.errorbar(c, m, yerr=s, fmt='o-', color='C2', lw=1.5, ms=5, capsize=3,
                label='fitted spatial-temporal')
    c, m_raw, _ = binned_mean(md['ecl_lat'].values, grad - a_det, edges)
    ax.plot(c, m_raw, '--', color='C3', lw=1.2,
            label='(total - a_det) binned mean')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel='ecliptic latitude [deg]',
           ylabel='gradient - a_det [MJy/sr]',
           title='(c) Spatial-temporal part vs ecl_lat')
    ax.legend(loc='upper right', fontsize=9)

    # (d) Model vs data (sorted by helio_lon for visual clarity).
    ax = axes[1, 0]
    order = np.argsort(md['helio_lon'].values)
    ax.scatter(md['helio_lon'].values[order], grad[order],
               s=1, alpha=0.08, color='grey', label='per-exposure')
    ax.plot(md['helio_lon'].values[order], grad_pred[order],
            '.', ms=1, color='C2', alpha=0.5, label='model')
    ax.axhline(a_det, color='C0', lw=1.5, label=f'a_det = {a_det:+.4f}')
    ax.set(xlabel='helio-ecliptic lon [deg]',
           ylabel='col0 - col2 [MJy/sr]',
           title='(d) Total signal vs model')
    ax.legend(loc='upper right', fontsize=9)

    # (e) Residual vs MJD to check nothing time-coherent left over.
    ax = axes[1, 1]
    t = md['MJD_AVG'].values
    edges = np.linspace(t.min(), t.max(), 40)
    c, m, s = binned_mean(t, residual, edges)
    ax.errorbar(c, m, yerr=s, fmt='o-', color='grey', lw=1.2, ms=4, capsize=2)
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel='MJD', ylabel='residual [MJy/sr]',
           title='(e) Residual vs MJD  (flat = model captured time/helio_lon structure)')

    # (f) Spatial-temporal map on (helio_lon, ecl_lat).
    ax = axes[1, 2]
    lon_edges = np.linspace(-180, 180, 25)
    lat_lo, lat_hi = np.percentile(md['ecl_lat'], [1, 99])
    lat_edges = np.linspace(lat_lo, lat_hi, 15)
    grid_sum = np.zeros((len(lon_edges) - 1, len(lat_edges) - 1))
    grid_n = np.zeros_like(grid_sum, dtype=int)
    ilon = np.digitize(md['helio_lon'].values, lon_edges) - 1
    ilat = np.digitize(md['ecl_lat'].values, lat_edges) - 1
    mk = ((ilon >= 0) & (ilon < grid_sum.shape[0])
          & (ilat >= 0) & (ilat < grid_sum.shape[1]))
    np.add.at(grid_sum, (ilon[mk], ilat[mk]), spatial_temporal[mk])
    np.add.at(grid_n, (ilon[mk], ilat[mk]), 1)
    grid_mean = np.where(grid_n >= 5, grid_sum / np.maximum(grid_n, 1), np.nan)
    v = np.nanmax(np.abs(grid_mean))
    im = ax.imshow(
        grid_mean.T, origin='lower', aspect='auto', cmap='RdBu_r',
        extent=(lon_edges[0], lon_edges[-1], lat_edges[0], lat_edges[-1]),
        vmin=-v, vmax=v,
    )
    ax.set(xlabel='helio-ecliptic lon [deg]',
           ylabel='ecliptic lat [deg]',
           title='(f) Spatial-temporal map (a_det removed)')
    plt.colorbar(im, ax=ax, label='gradient - a_det [MJy/sr]')

    fig.suptitle(
        f'col0 - col2 decomposition  det{detector} ch{channel}  '
        f'|  model={model}  |  a_det = {a_det:+.4f}  '
        f'(var shares: ST {var_st/var_total:.2f}, residual {var_res/var_total:.2f})',
        y=1.02,
    )
    fig.tight_layout()
    out = fig_path(f'det{detector}_ch{channel}__numcol_decomp_{model}.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    p.add_argument('--channel', type=int, default=17)
    p.add_argument('--model', choices=['harmonic', 'binned'], default='harmonic',
                   help='harmonic: parametric OLS fit. '
                        'binned: non-parametric 3D lookup table.')
    args = p.parse_args()
    main(args.detector, args.channel, model=args.model)
