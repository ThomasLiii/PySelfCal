"""Disentangle correlated drivers of col0-col2 via a joint OLS fit.

Approach
--------
Start from the geometric residual  y_i = grad_i - <grad>_channel(i)  so the
channel (wavelength) contribution is already removed and we focus on
per-exposure geometry.

Fit jointly:
    y = b0
        + [b1 sin(alpha)   + b2 cos(alpha)
         + b3 sin(2 alpha) + b4 cos(2 alpha)]            # alpha block
        + [b5 sin(helio_lon) + b6 cos(helio_lon)]         # helio_lon block
        + [b7 sin(ecl_lon)   + b8 cos(ecl_lon)]           # ecl_lon block
        + b9 * (ecl_lat - <ecl_lat>)                       # ecl_lat block
        + b10 * (elongation - <elongation>)                # elongation block
        + [b11 sin(2 pi MJD / 365.25)
         + b12 cos(2 pi MJD / 365.25)]                    # annual-MJD block

For each block we compute the *partial R^2* = fraction of TOTAL variance
uniquely attributable to that block: fit the full model AND the model
with that block removed, compare SS_residual. Unlike single-variable
binning, this correctly apportions variance when drivers are correlated
(alpha vs helio_lon, MJD vs helio_lon, ecl_lon vs helio_lon).
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


def wrap_deg(x, lo=-180.0, hi=180.0):
    return (x - lo) % (hi - lo) + lo


def build_design(df):
    """Return (X, column_names, block_map).
    block_map is dict: block_name -> list of column indices."""
    cols, names = [], []
    block_map = {}

    def add(block, new_cols, new_names):
        start = len(names)
        cols.extend(new_cols)
        names.extend(new_names)
        block_map.setdefault(block, []).extend(range(start, start + len(new_cols)))

    add('intercept', [np.ones(len(df))], ['intercept'])

    a = np.radians(df['alpha_deg'].values)
    add('alpha', [np.sin(a), np.cos(a), np.sin(2 * a), np.cos(2 * a)],
        ['sin_a', 'cos_a', 'sin_2a', 'cos_2a'])

    h = np.radians(df['helio_lon'].values)
    add('helio_lon', [np.sin(h), np.cos(h)],
        ['sin_hl', 'cos_hl'])

    e = np.radians(df['ecl_lon'].values)
    add('ecl_lon', [np.sin(e), np.cos(e)],
        ['sin_el', 'cos_el'])

    lat = df['ecl_lat'].values - df['ecl_lat'].mean()
    add('ecl_lat', [lat], ['ecl_lat'])

    elon = df['elongation'].values - df['elongation'].mean()
    add('elongation', [elon], ['elongation'])

    t = 2 * np.pi * df['MJD_AVG'].values / SIDEREAL_YEAR_DAYS
    add('mjd_annual', [np.sin(t), np.cos(t)], ['sin_t', 'cos_t'])

    return np.column_stack(cols), names, block_map


def fit_ols(X, y):
    """Solve beta in min ||y - X beta||^2; return beta, cov, residuals, R^2."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, p = X.shape
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot
    sigma2 = ss_res / max(1, n - p)
    try:
        xtx_inv = np.linalg.inv(X.T @ X)
        se = np.sqrt(sigma2 * np.diag(xtx_inv))
    except np.linalg.LinAlgError:
        se = np.full(p, np.nan)
    return {'beta': beta, 'se': se, 'resid': resid,
            'r2': r2, 'ss_res': ss_res, 'ss_tot': ss_tot,
            'n': n, 'p': p}


def partial_r2(X, y, block_map, full_fit):
    """For each block, refit without it and compute the unique variance share."""
    results = {}
    for block, idx in block_map.items():
        if block == 'intercept':
            continue
        keep = [i for i in range(X.shape[1]) if i not in idx]
        Xr = X[:, keep]
        f = fit_ols(Xr, y)
        dR2 = full_fit['r2'] - f['r2']           # fraction of TOTAL variance
        results[block] = {
            'partial_r2': dR2,
            'ss_res_without': f['ss_res'],
            'n_params_in_block': len(idx),
        }
    return results


def main(detector):
    df = pd.read_pickle(data_path(f'multichannel_det{detector}.pkl'))
    df = df.copy()
    df['alpha_deg'] = wrap_deg(df['pa_x_deg'].values - df['pa_to_sun_deg'].values)

    # 5-sigma outlier clip per channel.
    def clip(g):
        k = np.abs(g['grad_col02'] - g['grad_col02'].median()) < 5 * g['grad_col02'].std()
        return g.loc[k]
    df = df.groupby('channel', group_keys=False).apply(clip).reset_index(drop=True)

    # Remove per-channel mean; the joint model works on the residual.
    ch_mean = df.groupby('channel')['grad_col02'].transform('mean').values
    y = df['grad_col02'].values - ch_mean

    X, names, block_map = build_design(df)
    full = fit_ols(X, y)
    parts = partial_r2(X, y, block_map, full)

    print('\n=== Joint OLS fit (on grad - per-channel mean) ===')
    print(f'N = {full["n"]}   p = {full["p"]}')
    print(f'Total R^2 = {full["r2"]:.4f}')
    print(f'RMS residual = {np.std(full["resid"]):.5f} MJy/sr '
          f'(starting RMS = {np.std(y):.5f})')
    print()
    print('Block ranking by unique (partial) R^2:')
    pr = sorted(parts.items(), key=lambda kv: -kv[1]['partial_r2'])
    for block, info in pr:
        print(f'  {block:12s}  partial R^2 = {info["partial_r2"]*100:5.2f}%   '
              f'(n_params={info["n_params_in_block"]})')
    print()
    print('Coefficients (block grouped, +- 1 sigma):')
    for block, idx in block_map.items():
        if block == 'intercept':
            continue
        print(f'  [{block}]')
        for i in idx:
            print(f'    {names[i]:12s}  {full["beta"][i]:+.5f}  '
                  f'+- {full["se"][i]:.5f}  '
                  f'(t = {full["beta"][i]/max(full["se"][i],1e-12):+6.1f})')

    # ------------------ Plots ------------------
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # (a) Partial R^2 bar chart (uniqueness).
    ax = axes[0, 0]
    blocks = [b for b, _ in pr]
    r2_vals = [info['partial_r2'] * 100 for _, info in pr]
    colors = ['C3'] * len(blocks)
    ax.barh(blocks[::-1], r2_vals[::-1], color=colors)
    ax.set(xlabel='unique (partial) variance share [%]',
           title=f'(a) Each block\'s unique contribution '
                 f'(total R^2 = {full["r2"]*100:.1f}%)')
    for i, v in enumerate(r2_vals[::-1]):
        ax.text(v, i, f'  {v:.2f}%', va='center', fontsize=9)
    ax.grid(alpha=0.3, axis='x')

    # (b) Coefficient plot with error bars.
    ax = axes[0, 1]
    order = []
    for block, idx in block_map.items():
        if block == 'intercept':
            continue
        for i in idx:
            order.append((block, names[i], full['beta'][i], full['se'][i]))
    labels = [f'[{b}] {n}' for b, n, _, _ in order]
    vals = [v for _, _, v, _ in order]
    ses = [s for _, _, _, s in order]
    y_pos = np.arange(len(labels))[::-1]
    ax.errorbar(vals, y_pos, xerr=ses, fmt='o', color='C0',
                capsize=3, ms=4, lw=1)
    ax.axvline(0, color='k', lw=0.6, ls='--')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8, family='monospace')
    ax.set(xlabel='coefficient [MJy/sr]',
           title='(b) Fitted coefficients (joint OLS, +- 1 sigma)')
    ax.grid(alpha=0.3, axis='x')

    # (c) Post-fit residual distribution.
    ax = axes[0, 2]
    ax.hist(full['resid'], bins=120, alpha=0.75, color='grey')
    ax.set(xlabel='post-fit residual [MJy/sr]', ylabel='# exposures',
           title=f'(c) post-fit residual distribution  '
                 f'(std = {np.std(full["resid"]):.4f})')
    ax.grid(alpha=0.3)

    # (d-f) For the top 3 blocks, show partial-residual plots
    # (y after removing other blocks' predictions, vs the block's driver).
    top3 = [b for b, _ in pr[:3]]
    # Map each block to a single 1-D x-axis for plotting.
    xaxis = {
        'alpha':     ('alpha_deg',   r'$\alpha$ [deg]'),
        'helio_lon': ('helio_lon',   'helio-ecliptic lon [deg]'),
        'ecl_lon':   ('ecl_lon',     'ecliptic lon [deg]'),
        'ecl_lat':   ('ecl_lat',     'ecliptic lat [deg]'),
        'elongation':('elongation',  'elongation [deg]'),
        'mjd_annual':('MJD_AVG',     'MJD'),
    }
    for ax, block in zip(axes[1], top3):
        col, label = xaxis[block]
        # Partial residual for this block: y - X_other @ beta_other.
        keep_other = [i for i in range(X.shape[1]) if i not in block_map[block]]
        pred_other = X[:, keep_other] @ full['beta'][keep_other]
        part_resid = y - pred_other
        # Model prediction from this block alone:
        model_block = X[:, block_map[block]] @ full['beta'][block_map[block]]
        # Binned mean for visualisation.
        x = df[col].values
        if col == 'MJD_AVG':
            edges = np.linspace(x.min(), x.max(), 37)
        elif col == 'ecl_lon':
            edges = np.linspace(0, 360, 37)
        elif 'lat' in col:
            edges = np.linspace(np.percentile(x, 1), np.percentile(x, 99), 21)
        else:
            edges = np.linspace(-180, 180, 37)
        idx = np.digitize(x, edges) - 1
        centers, means, sems = [], [], []
        for b in range(len(edges) - 1):
            m = idx == b
            if m.sum() >= 30:
                centers.append(0.5 * (edges[b] + edges[b + 1]))
                means.append(np.mean(part_resid[m]))
                sems.append(np.std(part_resid[m]) / np.sqrt(m.sum()))
        centers = np.array(centers); means = np.array(means); sems = np.array(sems)
        # Model curve at the same x values.
        order_x = np.argsort(x)
        ax.scatter(x, part_resid, s=1, alpha=0.02, color='grey')
        ax.errorbar(centers, means, yerr=sems, fmt='o', color='C3',
                    ms=4, capsize=2, lw=1.2, label='partial residual (binned)')
        ax.plot(x[order_x], model_block[order_x], color='C0', lw=1.5,
                label=f'model: this block only')
        ax.axhline(0, color='k', lw=0.6, ls='--')
        ax.set(xlabel=label, ylabel='partial residual [MJy/sr]',
               title=f'[{block}]  unique R^2 = {parts[block]["partial_r2"]*100:.2f}%')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle(
        f'Joint model of col0-col2 (after per-channel mean removal)  '
        f'det{detector}  |  total R^2 = {full["r2"]*100:.1f}%',
        y=1.01,
    )
    fig.tight_layout()
    out = fig_path(f'det{detector}__grad_joint_model.png')
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out}')

    # Also save the numeric results.
    rows = [{'block': b,
             'partial_r2': parts[b]['partial_r2'],
             'n_params': parts[b]['n_params_in_block']} for b in blocks]
    pd.DataFrame(rows).to_csv(
        fig_path(f'det{detector}__grad_joint_model_partial_r2.csv'), index=False)
    coefs = pd.DataFrame({
        'block': [b for b, _, _, _ in order],
        'name':  [n for _, n, _, _ in order],
        'coef':  vals,
        'se':    ses,
        't':     [v / max(s, 1e-12) for v, s in zip(vals, ses)],
    })
    coefs.to_csv(fig_path(f'det{detector}__grad_joint_model_coefs.csv'), index=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    args = p.parse_args()
    main(args.detector)
