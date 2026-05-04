"""Task 1: Does the col0-col2 gradient correlate with the detector's
orientation relative to the Sun?

Idea
----
Zodiacal light is brighter closer to the Sun direction on the sky. So
*if* the col0-col2 gradient contains a zodi component, we expect it to
align with the Sun direction projected onto the detector's horizontal
axis. Define

  alpha = PA(detector +X axis)  -  PA(target -> Sun)

both measured east-of-north at the pointing centre. When alpha ~ 0 deg
the column axis runs straight toward the Sun; when alpha ~ 90 deg it
runs perpendicular.

For each exposure we compute

  grad_i = col0_i - col2_i
  alpha_i (deg)

and plot grad vs alpha for several channels. A zodi-driven gradient
would show a clean cos(alpha) dependence (grad large when alpha ~ 0 or
180, zero at 90). A purely detector-fixed gradient would be flat in
alpha (always the same in detector coordinates regardless of roll).
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


def main(detector, channels):
    df = pd.read_pickle(data_path(f'multichannel_det{detector}.pkl'))
    df = df[df['channel'].isin(channels)].reset_index(drop=True)

    # Alignment angle (east-of-north): PA_x (+X axis on sky) minus PA to Sun.
    alpha = wrap_deg(df['pa_x_deg'].values - df['pa_to_sun_deg'].values)
    # The column pair (col0->col2) runs along the -X or +X axis; a 180-deg
    # ambiguity only flips the gradient sign, which we handle by plotting
    # both halves of [-180, 180].
    df['alpha_deg'] = alpha

    grad = df['grad_col02'].values
    # 5-sigma clip per channel for a fair mean comparison.
    def clip(g):
        keep = np.abs(g['grad_col02'] - g['grad_col02'].median()) < 5 * g['grad_col02'].std()
        return g.loc[keep]
    df = df.groupby('channel', group_keys=False).apply(clip).reset_index(drop=True)

    fig, axes = plt.subplots(3, 2, figsize=(14, 15))

    # (a) Alignment-angle distribution to confirm we cover a full range.
    ax = axes[0, 0]
    ax.hist(df['alpha_deg'].values, bins=72)
    ax.set(xlabel=r'$\alpha$ = PA(+X) - PA(Sun)  [deg]',
           ylabel='# exposures',
           title='(a) coverage of the alignment angle')
    ax.axvline(0, color='k', lw=0.7, ls='--')
    ax.axvline(180, color='k', lw=0.7, ls=':')
    ax.axvline(-180, color='k', lw=0.7, ls=':')

    # (b) grad vs alpha -- binned mean across all selected channels, with
    #     a richer harmonic fit: 1 + cos + sin + cos(2a) + sin(2a).
    ax = axes[0, 1]
    edges = np.linspace(-180, 180, 37)
    c, m, s = binned_mean(df['alpha_deg'].values, df['grad_col02'].values,
                          edges, min_count=50)
    ax.scatter(df['alpha_deg'].values, df['grad_col02'].values,
               s=1, alpha=0.03, color='grey')
    ax.errorbar(c, m, yerr=s, fmt='o-', color='C3', lw=1.5, ms=5, capsize=2,
                label='binned mean')

    x = np.radians(df['alpha_deg'].values)
    y = df['grad_col02'].values
    # 1st-harmonic fit
    A1 = np.column_stack([np.ones_like(x), np.cos(x), np.sin(x)])
    beta1, *_ = np.linalg.lstsq(A1, y, rcond=None)
    a0_1, a1, b1 = beta1
    # 1st+2nd-harmonic fit
    A2 = np.column_stack([np.ones_like(x), np.cos(x), np.sin(x),
                          np.cos(2 * x), np.sin(2 * x)])
    beta2, *_ = np.linalg.lstsq(A2, y, rcond=None)
    a0_2, a1_2, b1_2, a2_2, b2_2 = beta2
    alpha_grid = np.linspace(-180, 180, 361)
    xg = np.radians(alpha_grid)
    ax.plot(alpha_grid,
            a0_1 + a1 * np.cos(xg) + b1 * np.sin(xg),
            color='C0', lw=1.5, ls='--',
            label=f'1st harmonic  (a0={a0_1:+.4f}, |amp1|={np.hypot(a1,b1):.4f})')
    ax.plot(alpha_grid,
            a0_2 + a1_2 * np.cos(xg) + b1_2 * np.sin(xg)
                 + a2_2 * np.cos(2 * xg) + b2_2 * np.sin(2 * xg),
            color='C2', lw=2,
            label=f'1st+2nd harmonic  (|amp2|={np.hypot(a2_2,b2_2):.4f})')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel=r'$\alpha$ [deg]', ylabel='col0 - col2  [MJy/sr]',
           title='(b) gradient vs alignment -- 1st vs 1st+2nd harmonic fits')
    ax.legend(loc='upper right', fontsize=9)

    # (c) Per-channel gradient vs alpha. Colour = channel.
    ax = axes[1, 0]
    cmap = plt.cm.viridis
    for ch in sorted(df['channel'].unique()):
        g = df[df['channel'] == ch]
        cc, mm, ss = binned_mean(g['alpha_deg'].values, g['grad_col02'].values,
                                 edges, min_count=30)
        ax.plot(cc, mm, lw=1.2, alpha=0.8, color=cmap(ch / max(channels)),
                label=f'ch{ch}' if ch in (min(channels), max(channels)) else None)
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel=r'$\alpha$ [deg]', ylabel='col0 - col2 [MJy/sr]',
           title='(c) per-channel gradient vs alpha (viridis = channel idx)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # (d) Decomposition: static part (mean a0) vs cos-amplitude per channel.
    ax = axes[1, 1]
    per_ch = []
    for ch in sorted(df['channel'].unique()):
        g = df[df['channel'] == ch]
        x = np.radians(g['alpha_deg'].values)
        A = np.column_stack([np.ones_like(x), np.cos(x), np.sin(x)])
        beta_ch, *_ = np.linalg.lstsq(A, g['grad_col02'].values, rcond=None)
        per_ch.append({'channel': ch,
                       'a0': beta_ch[0],
                       'amp': float(np.hypot(beta_ch[1], beta_ch[2])),
                       'phi0_deg': float(np.degrees(
                           np.arctan2(beta_ch[2], beta_ch[1])))})
    pc = pd.DataFrame(per_ch)
    ax.plot(pc['channel'], pc['a0'], 'o-', color='C0', label='a0 (detector-fixed part)')
    ax.plot(pc['channel'], pc['amp'], 's--', color='C3',
            label='amp of cos(alpha - phi0) (sun-aligned part)')
    ax.axhline(0, color='k', lw=0.6, ls=':')
    ax.set(xlabel='channel',
           ylabel='MJy/sr',
           title='(d) Detector-fixed a0 vs sun-aligned amplitude per channel')
    ax.legend()
    ax.grid(alpha=0.3)

    # (e) Direct even/odd decomposition at matched +|alpha|, -|alpha| bins.
    ax = axes[2, 0]
    # Bin finely, then pair bin i with bin (N - 1 - i) for symmetric folding.
    folded_edges = np.linspace(-180, 180, 73)  # 5-degree bins
    cc, mm, ss = binned_mean(df['alpha_deg'].values, df['grad_col02'].values,
                             folded_edges, min_count=30)
    n = len(cc)
    # Pair cc[i] with -cc[i] = cc[n-1-i] (since grid is symmetric around 0).
    pos_mask = cc > 0
    cc_pos = cc[pos_mask]
    mm_pos = mm[pos_mask]
    mm_neg = mm[::-1][pos_mask]                # value at -|alpha|
    # Handle NaNs
    valid = ~np.isnan(mm_pos) & ~np.isnan(mm_neg)
    cc_pos = cc_pos[valid]
    mm_pos = mm_pos[valid]
    mm_neg = mm_neg[valid]
    even = 0.5 * (mm_pos + mm_neg)
    odd = 0.5 * (mm_pos - mm_neg)
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.plot(cc_pos, even, 'o-', color='C0', lw=1.5,
            label='even:   [grad(+|a|) + grad(-|a|)] / 2')
    ax.plot(cc_pos, odd, 's--', color='C3', lw=1.5,
            label='odd:    [grad(+|a|) - grad(-|a|)] / 2')
    ax.set(xlabel=r'$|\alpha|$ [deg]',
           ylabel='decomposed gradient [MJy/sr]',
           title='(e) even (symmetric) vs odd (antisymmetric) parts of grad vs alpha')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # (f) Harmonic coefficients from the 2nd-order fit -- what's driving the
    #     shape. A pure detector-fixed signal -> only a0. A pure zodi gradient
    #     aligned with detector +X -> a0 + a1*cos(alpha). A pure sun-perpendicular
    #     -> b1*sin(alpha). Higher harmonics = survey-roll coupling.
    ax = axes[2, 1]
    labels = ['a0\n(DC)', 'a1\ncos(α)', 'b1\nsin(α)', 'a2\ncos(2α)', 'b2\nsin(2α)']
    values = [a0_2, a1_2, b1_2, a2_2, b2_2]
    colors = ['C0', 'C3', 'C3', 'C4', 'C4']
    ax.bar(labels, values, color=colors)
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(ylabel='coefficient [MJy/sr]',
           title='(f) harmonic coefficients (1st+2nd fit)')
    ax.grid(alpha=0.3, axis='y')
    # Annotate the even/odd summaries.
    even_amp = float(np.hypot(a1_2, a2_2))
    odd_amp = float(np.hypot(b1_2, b2_2))
    ax.text(0.02, 0.98,
            f'|even (cos terms)| = {even_amp:.4f}\n'
            f'|odd  (sin terms)| = {odd_amp:.4f}\n'
            f'DC = {a0_2:+.4f}',
            transform=ax.transAxes, va='top', ha='left',
            fontsize=9, family='monospace',
            bbox=dict(facecolor='white', alpha=0.8))

    fig.suptitle(
        f'col0-col2 vs Sun alignment  det{detector}  '
        f'(channels {min(channels)}..{max(channels)})',
        y=1.01,
    )
    fig.tight_layout()
    out = fig_path(f'det{detector}__col_alignment.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')

    # Save the per-channel fit table as CSV for reference.
    csv_out = fig_path(f'det{detector}__col_alignment_per_channel.csv')
    pc.to_csv(csv_out, index=False)
    print(f'wrote {csv_out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    p.add_argument('--channels', type=int, nargs='+',
                   default=list(range(1, 18)))
    args = p.parse_args()
    main(args.detector, args.channels)
