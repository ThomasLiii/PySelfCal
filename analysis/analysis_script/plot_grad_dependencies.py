"""What variables drive the col0-col2 gradient? Decompose and rank.

Strategy
--------
grad = grad_i^(ch)  -- per-exposure, per-channel signal.

Split into:
  * spectral dependence  =  <grad>_channel  (plot 1 panel: grad vs channel)
  * geometric dependence =  grad - <grad>_channel  (residual per exposure)

The spectral part captures the DC-baseline-plus-wavelength component that
changes with detector row.  The geometric residual is what's left per
exposure: this is what we scan against every candidate driver (alignment
angle, ecliptic lat/lon, helio_lon, MJD, elongation).  Peak-to-peak
amplitude of the binned mean, and the SNR (peak-to-peak divided by
median SEM), rank the drivers.

Output: figures/det{det}__grad_dependencies.png
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from zodi_utils import data_path, fig_path


def wrap_deg(x, lo=-180.0, hi=180.0):
    return (x - lo) % (hi - lo) + lo


def binned_stats(x, y, edges, min_count=50):
    idx = np.digitize(x, edges) - 1
    n = len(edges) - 1
    c = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(n, np.nan); sem = np.full(n, np.nan); cnt = np.zeros(n, int)
    for b in range(n):
        m = idx == b
        cnt[b] = int(m.sum())
        if cnt[b] >= min_count:
            mean[b] = np.mean(y[m])
            sem[b] = np.std(y[m]) / np.sqrt(cnt[b])
    return c, mean, sem, cnt


def main(detector):
    df = pd.read_pickle(data_path(f'multichannel_det{detector}.pkl'))
    df = df.copy()
    df['alpha_deg'] = wrap_deg(df['pa_x_deg'].values - df['pa_to_sun_deg'].values)

    # Per-channel 5-sigma clip to kill bright-star contamination.
    def clip(g):
        k = np.abs(g['grad_col02'] - g['grad_col02'].median()) < 5 * g['grad_col02'].std()
        return g.loc[k]
    df = df.groupby('channel', group_keys=False).apply(clip).reset_index(drop=True)

    grad = df['grad_col02'].values

    # Per-channel means (the spectral baseline).
    ch_mean = df.groupby('channel')['grad_col02'].transform('mean').values
    resid = grad - ch_mean  # geometric residual

    # ---- Configure panels (driver, edges, label, units, use_resid) ----
    panels = [
        ('channel',     'channel index',           'grad',
         np.arange(0.5, 18.5, 1.0), False),
        ('alpha_deg',   r'α = PA(+X) - PA(Sun) [deg]', 'residual',
         np.linspace(-180, 180, 37),  True),
        ('ecl_lat',     'ecliptic latitude [deg]', 'residual',
         np.linspace(np.percentile(df['ecl_lat'], 1),
                     np.percentile(df['ecl_lat'], 99), 21), True),
        ('ecl_lon',     'ecliptic longitude [deg]','residual',
         np.linspace(0, 360, 37), True),
        ('helio_lon',   'helio-ecliptic lon [deg]','residual',
         np.linspace(-180, 180, 37), True),
        ('MJD_AVG',     'MJD',                     'residual',
         np.linspace(df['MJD_AVG'].min(), df['MJD_AVG'].max(), 31), True),
        ('elongation',  'solar elongation [deg]',  'residual',
         np.linspace(np.percentile(df['elongation'], 1),
                     np.percentile(df['elongation'], 99), 21), True),
    ]
    # Wrap ecl_lon into [0, 360) in case astropy returns negatives.
    df_plot = df.copy()
    df_plot['ecl_lon'] = np.mod(df_plot['ecl_lon'].values, 360.0)

    # Compute peak-to-peak + SEM for each panel to rank drivers.
    ranking = []
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    axes = axes.flatten()
    for ax, (col, label, ytype, edges, use_resid) in zip(axes, panels):
        x = df_plot[col].values
        y = resid if use_resid else grad
        c, m, s, n = binned_stats(x, y, edges, min_count=30)
        pp = float(np.nanmax(m) - np.nanmin(m))
        med_sem = float(np.nanmedian(s))
        snr = pp / med_sem if med_sem > 0 else np.nan
        ranking.append({'driver': col, 'target': ytype,
                        'peak_to_peak': pp, 'median_sem': med_sem, 'snr': snr})

        ax.scatter(x, y, s=1, alpha=0.02, color='grey')
        ax.errorbar(c, m, yerr=s, fmt='o-',
                    color='C3' if use_resid else 'C0',
                    lw=1.4, ms=4, capsize=2)
        ax.axhline(0 if use_resid else np.nanmean(m),
                   color='k', lw=0.6, ls='--')
        ax.set(xlabel=label,
               ylabel=('grad - <grad>_channel [MJy/sr]'
                       if use_resid else 'grad_col02 [MJy/sr]'),
               title=f'{col} : peak-to-peak {pp:.4f}, SNR {snr:.1f}')
        ax.grid(alpha=0.3)

    # 8th panel: ranking bar chart.
    rank_df = (pd.DataFrame(ranking)
               .sort_values('peak_to_peak', ascending=False)
               .reset_index(drop=True))
    ax = axes[7]
    ax.barh(rank_df['driver'][::-1], rank_df['peak_to_peak'][::-1],
            color=['C0' if t == 'grad' else 'C3'
                   for t in rank_df['target'][::-1]])
    ax.set(xlabel='peak-to-peak amplitude [MJy/sr]',
           title='driver ranking by peak-to-peak of binned mean')
    for i, (pp, snr) in enumerate(
            zip(rank_df['peak_to_peak'][::-1], rank_df['snr'][::-1])):
        ax.text(pp, i, f'  SNR {snr:.0f}', va='center', fontsize=8)
    ax.grid(alpha=0.3, axis='x')

    # 9th panel: text summary.
    ax = axes[8]
    ax.axis('off')
    lines = [
        'Legend:',
        '  C0 (blue) = raw grad (channel is the driver)',
        '  C3 (red)  = grad - <grad>_channel  (per-exposure residuals)',
        '',
        'Raw grad range :  '
        f'[{np.min(grad):+.4f}, {np.max(grad):+.4f}] MJy/sr',
        f'Raw grad mean  :  {np.mean(grad):+.4f}',
        f'Raw grad std   :   {np.std(grad):.4f}',
        '',
        'Residual std after subtracting <grad>_channel:',
        f'  std(resid) =  {np.std(resid):.4f} MJy/sr',
        '',
        'Driver ranking (largest peak-to-peak first):',
    ] + [f'  {i+1}. {r["driver"]:12s}  pp={r["peak_to_peak"]:.4f}, '
         f'SNR={r["snr"]:4.0f}'
         for i, (_, r) in enumerate(rank_df.iterrows())]
    ax.text(0.0, 1.0, '\n'.join(lines), family='monospace',
            fontsize=10, va='top')

    fig.suptitle(
        f'What drives the col0-col2 gradient?  det{detector}',
        fontsize=14, y=0.995,
    )
    fig.tight_layout()
    out = fig_path(f'det{detector}__grad_dependencies.png')
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')

    csv_out = fig_path(f'det{detector}__grad_dependencies_ranking.csv')
    rank_df.to_csv(csv_out, index=False)
    print(f'wrote {csv_out}')
    print('\nDriver ranking:')
    print(rank_df.to_string(index=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    args = p.parse_args()
    main(args.detector)
