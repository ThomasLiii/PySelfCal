"""Fit a zodi dust-plane orientation from the col0-col2 column gradient.

Physical model
--------------
The zodi smooth cloud has a symmetry plane tilted by inclination i (~1.57 deg
per Kelsall+1998) at ascending-node ecliptic longitude Omega (~77.7 deg).
The brightness gradient on the sky at any target direction points AWAY from
the dust-plane pole (= rotate the ecliptic pole by i around Omega).

For a detector aligned with spacecraft position angle PA_x (east-of-north),
the column-gradient projects as

    col0 - col2 = C + A * cos(PA_x - PA_to_dust_pole)

For each candidate (i, Omega) we:
  1. compute the dust-pole ICRS position,
  2. compute PA from each exposure's target (CRVAL1, CRVAL2) to that pole,
  3. fit (C, A) by OLS against col0 - col2,
  4. record the model SSR.

The (i, Omega) that minimises SSR is the best-fit dust-plane orientation.

Channel-wise DC (wavelength dependence) is absorbed by a per-channel offset
before fitting, so the model coefficients are the same across channels.
"""
import argparse
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u

from zodi_utils import data_path, fig_path


def dust_pole_icrs(i_deg, omega_deg):
    """Return (RA, Dec) of the dust-plane pole in ICRS, given (i, Omega) in
    ecliptic convention. Uses astropy to rotate ecliptic -> ICRS."""
    i_rad = np.radians(i_deg)
    omega_rad = np.radians(omega_deg)
    # Tilt the ecliptic pole (lon undefined, lat = +90 deg) around the
    # ecliptic ascending-node axis by angle i. In ecliptic spherical coords
    # the pole moves to lon = Omega + 90 deg, lat = 90 - i.
    lon = (omega_deg + 90.0) % 360.0
    lat = 90.0 - i_deg
    pole_ecl = SkyCoord(lon=lon * u.deg, lat=lat * u.deg,
                        frame='barycentrictrueecliptic')
    pole_icrs = pole_ecl.transform_to('icrs')
    return float(pole_icrs.ra.deg), float(pole_icrs.dec.deg)


def compute_pa_to_pole(df, ra_pole, dec_pole):
    """Position angle (east-of-north) from each target to the dust pole."""
    targets = SkyCoord(df['CRVAL1'].values * u.deg,
                       df['CRVAL2'].values * u.deg, frame='icrs')
    pole = SkyCoord(ra_pole * u.deg, dec_pole * u.deg, frame='icrs')
    return targets.position_angle(pole).deg


def fit_model(df, pa_pole, y):
    """Fit  y = C + A * cos(PA_x - PA_pole).  Returns (C, A, ssr)."""
    d = np.radians(df['pa_x_deg'].values - pa_pole)
    X = np.column_stack([np.ones(len(df)), np.cos(d)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    ssr = float(np.sum((y - pred) ** 2))
    return float(beta[0]), float(beta[1]), ssr


def main(detector):
    df = pd.read_pickle(data_path(f'multichannel_det{detector}.pkl'))
    df = df.copy()

    # Per-channel outlier clip, then subtract per-channel mean so the
    # wavelength-dependent DC is out of the way.
    def clip(g):
        k = np.abs(g['grad_col02'] - g['grad_col02'].median()) < 5 * g['grad_col02'].std()
        return g.loc[k]
    df = df.groupby('channel', group_keys=False).apply(clip).reset_index(drop=True)
    ch_mean = df.groupby('channel')['grad_col02'].transform('mean').values
    y = df['grad_col02'].values - ch_mean
    print(f'{len(df)} (exposure x channel) rows; y std = {np.std(y):.5f}')

    # --- Scan (i, Omega) grid ---
    i_grid = np.arange(0.0, 10.01, 0.25)          # dust-plane tilt [deg]
    omega_grid = np.arange(0.0, 360.0, 5.0)       # ascending node [deg]
    ssr = np.full((len(i_grid), len(omega_grid)), np.nan)
    A_grid = np.full_like(ssr, np.nan)
    C_grid = np.full_like(ssr, np.nan)
    print(f'Grid scan: {len(i_grid)} x {len(omega_grid)} = '
          f'{len(i_grid) * len(omega_grid)} points...')
    for ii, i_val in enumerate(i_grid):
        for jj, om_val in enumerate(omega_grid):
            ra_p, dec_p = dust_pole_icrs(i_val, om_val)
            pa_pole = compute_pa_to_pole(df, ra_p, dec_p)
            C, A, s = fit_model(df, pa_pole, y)
            ssr[ii, jj] = s
            A_grid[ii, jj] = A
            C_grid[ii, jj] = C

    # Best-fit point (lowest SSR).
    ii_best, jj_best = np.unravel_index(np.nanargmin(ssr), ssr.shape)
    i_best = float(i_grid[ii_best])
    om_best = float(omega_grid[jj_best])
    C_best = float(C_grid[ii_best, jj_best])
    A_best = float(A_grid[ii_best, jj_best])
    ss_min = float(ssr[ii_best, jj_best])
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2_best = 1.0 - ss_min / ss_tot

    # Zero-tilt reference (i=0, NEP as pole; Omega has no effect).
    ra0, dec0 = dust_pole_icrs(0.0, 0.0)
    pa0 = compute_pa_to_pole(df, ra0, dec0)
    C0, A0, s0 = fit_model(df, pa0, y)
    r2_i0 = 1.0 - s0 / ss_tot

    kelsall_i, kelsall_om = 1.57, 77.7
    ra_k, dec_k = dust_pole_icrs(kelsall_i, kelsall_om)
    pa_k = compute_pa_to_pole(df, ra_k, dec_k)
    Ck, Ak, sk = fit_model(df, pa_k, y)
    r2_k = 1.0 - sk / ss_tot

    print()
    print('=== Dust-plane grid-scan results ===')
    print(f'  Best fit:    i = {i_best:.2f} deg,  Omega = {om_best:.1f} deg '
          f'(C={C_best:+.5f}, A={A_best:+.5f}, R^2={r2_best*100:.2f}%)')
    print(f'  i = 0 (NEP): (C={C0:+.5f}, A={A0:+.5f}, R^2={r2_i0*100:.2f}%)')
    print(f'  Kelsall:     i = {kelsall_i} deg, Omega = {kelsall_om} deg '
          f'(C={Ck:+.5f}, A={Ak:+.5f}, R^2={r2_k*100:.2f}%)')
    # Convert best-fit ssr to a confidence-like surface by delta-chi2.
    # Using n-4 dof and sigma^2 = ssr_min/(n-p_effective).
    # Display delta_ssr contours.

    # --- Plots ---
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) SSR surface on (i, Omega), with best-fit + Kelsall marked.
    ax = axes[0, 0]
    im = ax.pcolormesh(omega_grid, i_grid, (ssr - ss_min) * 1e6,
                       cmap='viridis_r', shading='auto')
    ax.scatter([om_best], [i_best], marker='*', s=180, color='red',
               edgecolor='k', label=f'best fit ({om_best:.1f}, {i_best:.2f})')
    ax.scatter([kelsall_om], [kelsall_i], marker='x', s=120, color='white',
               label=f'Kelsall ({kelsall_om}, {kelsall_i})')
    ax.set(xlabel=r'$\Omega$  ascending node lon [deg]',
           ylabel=r'$i$  dust-plane tilt [deg]',
           title='(a) (SSR - min)  * 1e6   --- lower is better')
    plt.colorbar(im, ax=ax, label='delta SSR * 1e6')
    ax.legend(loc='upper right', fontsize=9)

    # (b) Same as (a) but showing R^2 improvement.
    ax = axes[0, 1]
    r2_grid = 1.0 - ssr / ss_tot
    im = ax.pcolormesh(omega_grid, i_grid, r2_grid * 100,
                       cmap='viridis', shading='auto')
    ax.scatter([om_best], [i_best], marker='*', s=180, color='red',
               edgecolor='k')
    ax.scatter([kelsall_om], [kelsall_i], marker='x', s=120, color='white')
    # Mark the i=0 reference R^2 as a contour.
    ax.contour(omega_grid, i_grid, r2_grid * 100, levels=[r2_i0 * 100],
               colors='white', linestyles='--')
    ax.set(xlabel=r'$\Omega$ [deg]', ylabel=r'$i$ [deg]',
           title=f'(b) R^2 on (i, Omega)  --- i=0 ref = {r2_i0*100:.2f}%')
    plt.colorbar(im, ax=ax, label='R^2 [%]')

    # (c) Residual vs PA_x - PA_to_dust_pole at the best-fit (and Kelsall).
    ax = axes[1, 0]
    pa_best = compute_pa_to_pole(df, *dust_pole_icrs(i_best, om_best))
    theta_best = ((df['pa_x_deg'].values - pa_best + 180) % 360) - 180
    pa_kelsall = compute_pa_to_pole(df, ra_k, dec_k)
    theta_k = ((df['pa_x_deg'].values - pa_kelsall + 180) % 360) - 180
    # Binned mean of y vs theta.
    edges = np.linspace(-180, 180, 37)
    def _bin(x, y, edges, min_count=50):
        idx = np.digitize(x, edges) - 1
        n = len(edges) - 1
        centers = 0.5 * (edges[:-1] + edges[1:])
        means = np.full(n, np.nan)
        sems = np.full(n, np.nan)
        for b in range(n):
            m = idx == b
            if int(m.sum()) >= min_count:
                means[b] = np.mean(y[m])
                sems[b] = np.std(y[m]) / np.sqrt(int(m.sum()))
        return centers, means, sems
    c, m, s = _bin(theta_best, y, edges)
    c_k, m_k, s_k = _bin(theta_k, y, edges)
    xg = np.linspace(-180, 180, 361)
    ax.errorbar(c, m, yerr=s, fmt='o', color='C3', ms=4, capsize=2,
                label=f'best (i={i_best:.2f}, Om={om_best:.1f})')
    ax.plot(xg, C_best + A_best * np.cos(np.radians(xg)),
            color='C3', lw=1.8, label='best-fit model')
    ax.errorbar(c_k, m_k, yerr=s_k, fmt='s', color='C0', ms=3, alpha=0.7,
                capsize=2, label=f'Kelsall (i={kelsall_i}, Om={kelsall_om})')
    ax.plot(xg, Ck + Ak * np.cos(np.radians(xg)),
            color='C0', lw=1.2, ls='--', label='Kelsall model')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel=r'PA(+X) - PA(target $\to$ dust pole)  [deg]',
           ylabel='grad - <grad>_channel [MJy/sr]',
           title='(c) gradient vs alignment to DUST pole (not sun)')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(alpha=0.3)

    # (d) Summary text + comparison table.
    ax = axes[1, 1]
    ax.axis('off')
    lines = [
        'Dust-plane grid fit summary',
        '===========================',
        '',
        f'Data: {len(df)} (exposure x channel) rows,',
        f'      y = col0-col2 - <grad>_channel,   std(y) = {np.std(y):.5f}',
        '',
        'Model:  y = C + A * cos( PA(+X)_i - PA(target_i -> dust pole) )',
        '',
        'Scenario         |      i [deg]     Omega [deg]    R^2 [%]',
        '-----------------|--------------------------------------------',
        f'Best fit         |      {i_best:5.2f}        {om_best:6.1f}      {r2_best*100:6.2f}',
        f'Kelsall (lit)    |      {kelsall_i:5.2f}        {kelsall_om:6.1f}      {r2_k*100:6.2f}',
        f'No tilt (i = 0)  |      0.00          n/a       {r2_i0*100:6.2f}',
        '',
        f'Best-fit A = {A_best:+.5f} MJy/sr',
        f'Best-fit C = {C_best:+.5f} MJy/sr',
        '',
        'Interpretation:',
        '  A is the amplitude of the sinusoidal gradient modulation',
        '  as the spacecraft +X axis rotates relative to the line',
        '  from the target to the zodi dust-plane pole.',
    ]
    ax.text(0.0, 1.0, '\n'.join(lines), va='top',
            family='monospace', fontsize=9)

    fig.suptitle(
        f'Zodi dust-plane fit from col0-col2  det{detector}  '
        f'(channels {df["channel"].min()}-{df["channel"].max()})',
        y=1.01,
    )
    fig.tight_layout()
    out = fig_path(f'det{detector}__dust_plane_fit.png')
    fig.savefig(out, dpi=170, bbox_inches='tight')
    plt.close(fig)
    print(f'\nwrote {out}')

    # Save the grid for further exploration.
    grid_out = data_path(f'det{detector}__dust_plane_grid.npz')
    np.savez(grid_out, i_grid=i_grid, omega_grid=omega_grid,
             ssr=ssr, r2_grid=r2_grid, A_grid=A_grid, C_grid=C_grid,
             i_best=i_best, omega_best=om_best,
             C_best=C_best, A_best=A_best, r2_best=r2_best,
             r2_kelsall=r2_k, r2_i0=r2_i0)
    print(f'wrote {grid_out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    args = p.parse_args()
    main(args.detector)
