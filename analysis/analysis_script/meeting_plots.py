"""Meeting plots: per-chunk sine fits -> residual diagnostics -> detector-fixed.

Three figures, in order:

  fig1: per-chunk annual-sine fits
        - one sample chunk's offset(t) with the fit overlaid
        - stitched C, A, phi vs subchannel (3 columns)
        - residual std per chunk

  fig2: residuals after subtracting the per-chunk sine
        - histogram of (offset - sine model)
        - per-exposure mean residual vs MJD          (should be flat)
        - per-exposure mean residual vs helio_lon    (should be flat)

  fig3: detector-fixed extraction
        - stitched C(sub, col) map: time-averaged offset (zodi + detector fixed)
        - C minus per-subchannel mean across columns: pure detector-fixed
        - the column-only piece rendered on the detector pixel grid
        - text: math justification

Math:
    sine fit: offset(t, s, c) = C(s, c) + A(s, c) sin(2*pi t/365.25 + phi(s, c))
    -> C(s, c) is, by construction, the time-average of offset(:, s, c)
       (the sine integrates to zero over a full period; with our 9-month
        sampling this holds approximately).
    -> C ~ I_zodi_avg(lambda_s) + detector_fixed(s, c)
    -> D(s, c) = C(s, c) - <C(s, .)>_cols  removes the lambda-dependent
       zodi spectrum (identical across the 3 cols at a given lambda) and
       leaves only the column-resolved detector-fixed component.
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_SELFCAL_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))
if _SELFCAL_ROOT not in sys.path:
    sys.path.insert(0, _SELFCAL_ROOT)

from SelfCal.SPHERExUtility import (
    make_stripped_chunk_map,
    make_stripped_chunk_valid_mask,
    make_spherex_stripped_offset_map,
    load_lvf_params,
)
from zodi_utils import data_path, fig_path, cal_path, load_cal_offsets

NUM_SUB, NUM_CH, NUM_COL = 10, 34, 3
TOT_SUB = NUM_SUB * NUM_CH + 2
SIDEREAL_YEAR_DAYS = 365.25


F_ANNUAL = 1.0 / SIDEREAL_YEAR_DAYS
MIN_FIT_POINTS = 50


def load_channel_offsets(detector, channel):
    """Load (offset_cube, padded_mask) for one channel."""
    with h5py.File(cal_path(detector, channel), 'r') as f:
        off = load_cal_offsets(f)[0].reshape(-1, TOT_SUB, NUM_COL)
    padded = make_stripped_chunk_valid_mask(
        ch=[channel], num_subchannels=NUM_SUB, num_channels=NUM_CH,
        num_columns=NUM_COL, subchannel_padding=1,
    ).reshape(TOT_SUB, NUM_COL).astype(bool)
    return off, padded


def chunk_moments(off, padded, s_t, c_t):
    """Per-chunk OLS-fit moments needed to evaluate SSR(phi) and (A, C | phi).

    Each returned array has shape (TOT_SUB, NUM_COL).
    """
    # Mask: valid sample AND inside the padded region. Treat masked-out values
    # as zero in the sums (they contribute nothing because v is zero).
    v = (np.isfinite(off) & padded[None, :, :]).astype(np.float64)
    y = np.where(v > 0, off, 0.0)
    s_b = s_t[:, None, None]
    c_b = c_t[:, None, None]
    return {
        'n':    v.sum(axis=0),
        'S_y':  y.sum(axis=0),
        'S_s':  (s_b * v).sum(axis=0),
        'S_c':  (c_b * v).sum(axis=0),
        'S_ys': (y * s_b).sum(axis=0),
        'S_yc': (y * c_b).sum(axis=0),
        'S_ss': (s_b * s_b * v).sum(axis=0),
        'S_cc': (c_b * c_b * v).sum(axis=0),
        'S_sc': (s_b * c_b * v).sum(axis=0),
        'S_yy': (y * y).sum(axis=0),
    }


def total_ssr(phi, moments_list):
    """Sum of OLS residual SSR across every chunk in moments_list, given phi."""
    cphi, sphi = np.cos(phi), np.sin(phi)
    total = 0.0
    for m in moments_list:
        n = m['n']
        ok = n >= MIN_FIT_POINTS
        if not np.any(ok):
            continue
        S_u  = cphi * m['S_s']  + sphi * m['S_c']
        S_yu = cphi * m['S_ys'] + sphi * m['S_yc']
        S_uu = (cphi * cphi * m['S_ss']
                + 2.0 * sphi * cphi * m['S_sc']
                + sphi * sphi * m['S_cc'])
        denom = n * S_uu - S_u * S_u
        num = n * S_yu - S_u * m['S_y']
        with np.errstate(divide='ignore', invalid='ignore'):
            ssr = m['S_yy'] - m['S_y']**2 / np.where(n > 0, n, 1) \
                  - num**2 / np.where(denom > 0, n * denom, 1)
        ssr = np.where(ok & (denom > 1e-18), ssr, 0.0)
        total += float(np.nansum(ssr))
    return total


def find_global_phi(moments_list, n_grid=721):
    """1-D search for the global phase that minimizes summed SSR."""
    from scipy.optimize import minimize_scalar
    grid = np.linspace(-np.pi, np.pi, n_grid, endpoint=False)
    ssr_grid = np.array([total_ssr(p, moments_list) for p in grid])
    i = int(np.argmin(ssr_grid))
    step = grid[1] - grid[0]
    res = minimize_scalar(
        lambda p: total_ssr(p, moments_list),
        bracket=(grid[i] - step, grid[i], grid[i] + step),
        method='brent', options={'xtol': 1e-7},
    )
    return float(res.x)


def fit_chunks_with_phi(off, padded, mjd_full, phi_global, f=F_ANNUAL):
    """Per-chunk linear OLS fit y = A * sin(2*pi f t + phi_global) + C.

    Amplitude is signed so that the phase is exactly the same for every chunk.
    """
    s_t = np.sin(2.0 * np.pi * f * mjd_full)
    c_t = np.cos(2.0 * np.pi * f * mjd_full)
    u = s_t * np.cos(phi_global) + c_t * np.sin(phi_global)

    C = np.full((TOT_SUB, NUM_COL), np.nan)
    A = np.full((TOT_SUB, NUM_COL), np.nan)
    phi = np.full((TOT_SUB, NUM_COL), np.nan)
    resid_std = np.full((TOT_SUB, NUM_COL), np.nan)
    sine_pred = np.full(off.shape, np.nan)

    for s in range(TOT_SUB):
        for c in range(NUM_COL):
            if not padded[s, c]:
                continue
            y = off[:, s, c]
            ok = np.isfinite(y)
            if int(ok.sum()) < MIN_FIT_POINTS:
                continue
            uu = u[ok]; yy = y[ok]
            mu_u = uu.mean(); mu_y = yy.mean()
            var_u = float(((uu - mu_u) ** 2).sum())
            if var_u <= 0:
                continue
            a_fit = float(((uu - mu_u) * (yy - mu_y)).sum() / var_u)
            c_fit = float(mu_y - a_fit * mu_u)
            pred = a_fit * u + c_fit
            A[s, c] = a_fit
            C[s, c] = c_fit
            phi[s, c] = phi_global
            resid_std[s, c] = float(np.std(yy - pred[ok]))
            sine_pred[:, s, c] = pred
    resid = off - sine_pred
    return {
        'C': C, 'A': A, 'phi': phi, 'resid_std': resid_std,
        'resid_cube': resid, 'padded_mask': padded,
        'offset_cube': off,
    }


def stitch_C(C_per_ch, padded_per_ch, sorted_chs):
    """Cumulative additive shift so adjacent channels' C agree on overlap."""
    shifts = {sorted_chs[0]: 0.0}
    for i in range(1, len(sorted_chs)):
        prev_c = sorted_chs[i - 1]; curr_c = sorted_chs[i]
        ov = padded_per_ch[prev_c] & padded_per_ch[curr_c]
        a = C_per_ch[prev_c]; b = C_per_ch[curr_c]
        good = ov & np.isfinite(a) & np.isfinite(b)
        d = float(np.mean(a[good] - b[good])) if good.any() else 0.0
        shifts[curr_c] = shifts[prev_c] + d
    return shifts


def main(detector, sample_sub=80, sample_col=1):
    df = pd.read_pickle(data_path(f'multichannel_det{detector}.pkl'))
    base = df[df['channel'] == 1].drop(columns=['channel', 'mean_offset',
                                                 'col0', 'col1', 'col2',
                                                 'grad_col02']).reset_index(drop=True)
    n_exp = len(base)
    mjd_full = base['MJD_AVG'].values.astype(float)
    print(f'{n_exp} exposures, MJD span {mjd_full.min():.0f}-{mjd_full.max():.0f}')

    # Load per-(exposure, chunk) coordinates for the residual scans.
    # If the cache is missing, fall back to the per-exposure central pointing.
    pc_path = data_path(f'perchunk_coords_det{detector}.npz')
    if os.path.exists(pc_path):
        pc = np.load(pc_path, allow_pickle=True)
        per_chunk_helio = pc['helio_lon']        # (n_exp, n_chunks)
        per_chunk_elon = pc['elongation']        # (n_exp, n_chunks)
        print(f'loaded per-chunk coords from {pc_path}')
        use_per_chunk = True
    else:
        per_chunk_helio = None
        per_chunk_elon = None
        helio_lon = base['helio_lon'].values
        elongation = base['elongation'].values
        use_per_chunk = False
        print('per-chunk coord cache not found; using exposure-central coords')

    # Load all offset cubes, then fit a SINGLE global phase shared across
    # every (channel, sub, col) chunk; (A, C) are still per-chunk.
    channels = list(range(1, 35))
    s_t = np.sin(2.0 * np.pi * F_ANNUAL * mjd_full)
    c_t = np.cos(2.0 * np.pi * F_ANNUAL * mjd_full)
    cubes = {}
    padded_per_ch = {}
    moments_per_ch = []
    for ch in tqdm(channels, desc='loading + moments'):
        off, padded = load_channel_offsets(detector, ch)
        cubes[ch] = off
        padded_per_ch[ch] = padded
        moments_per_ch.append(chunk_moments(off, padded, s_t, c_t))

    phi_global = find_global_phi(moments_per_ch)
    print(f'global phase phi = {np.degrees(phi_global):+.3f} deg '
          f'({phi_global:+.5f} rad)')

    fits = {}
    for ch in tqdm(channels, desc='per-chunk linear fits'):
        fits[ch] = fit_chunks_with_phi(
            cubes[ch], padded_per_ch[ch], mjd_full, phi_global,
        )

    # Stitch C across channels.
    C_per_ch = {ch: fits[ch]['C'] for ch in channels}
    padded_per_ch = {ch: fits[ch]['padded_mask'] for ch in channels}
    shifts = stitch_C(C_per_ch, padded_per_ch, channels)
    print('Stitching shifts (ch -> +offset):')
    for ch in channels:
        print(f'  ch{ch:2d}: {shifts[ch]:+.5f} MJy/sr')

    # Combined arrays: average across channels in overlap.
    def combine(metric_name):
        stack = np.stack(
            [np.where(padded_per_ch[ch], fits[ch][metric_name], np.nan)
             for ch in channels], axis=0)
        with np.errstate(invalid='ignore'):
            return np.nanmean(stack, axis=0)
    A_comb = combine('A')
    phi_comb = combine('phi')
    resid_std_comb = combine('resid_std')

    # Stitched C: apply per-channel shift before combining.
    C_stack = np.stack(
        [np.where(padded_per_ch[ch], fits[ch]['C'] + shifts[ch], np.nan)
         for ch in channels], axis=0)
    with np.errstate(invalid='ignore'):
        C_comb = np.nanmean(C_stack, axis=0)
    # Centre the global C around zero so the colormap is symmetric.
    C_comb = C_comb - np.nanmean(C_comb)

    # Per-exposure mean residual (used for the histogram + sample fit only).
    resid_scalar = np.zeros(n_exp)
    counts = np.zeros(n_exp)
    for ch in channels:
        rc = fits[ch]['resid_cube']                             # (N, 342, 3)
        m = padded_per_ch[ch][None, :, :] & np.isfinite(rc)
        resid_scalar += np.where(m, rc, 0.0).sum(axis=(1, 2))
        counts += m.sum(axis=(1, 2))
    resid_scalar = np.where(counts > 0, resid_scalar / counts, np.nan)

    # Per-(exposure, chunk) residuals -- needed for the per-chunk-coord
    # residual scans below. Combine across channels by averaging where
    # multiple channels contributed (the padded overlap region).
    resid_per_chunk_sum = np.zeros((n_exp, TOT_SUB, NUM_COL))
    resid_per_chunk_count = np.zeros((n_exp, TOT_SUB, NUM_COL))
    for ch in channels:
        rc = fits[ch]['resid_cube']
        m = padded_per_ch[ch][None, :, :] & np.isfinite(rc)
        resid_per_chunk_sum += np.where(m, rc, 0.0)
        resid_per_chunk_count += m
    with np.errstate(invalid='ignore'):
        resid_per_chunk = np.where(resid_per_chunk_count > 0,
                                   resid_per_chunk_sum / np.maximum(resid_per_chunk_count, 1),
                                   np.nan)
    # Flatten to (n_exp, n_chunks) in the sub*NUM_COL + col convention so it
    # matches the per-chunk coord cache.
    resid_per_chunk_flat = resid_per_chunk.reshape(n_exp, -1)

    # ------------------------------------------------------------------
    # FIGURE 1: per-chunk sine fits
    # Wavelength axis from lvf_params['wave_edges'].
    # ------------------------------------------------------------------
    lvf = load_lvf_params(f'lvf_params_D{detector}.npy')
    mean_wav = 0.5 * (lvf['wave_edges'][:-1] + lvf['wave_edges'][1:])  # (340,)
    sub_wav = np.full(TOT_SUB, np.nan)
    sub_wav[1:1 + len(mean_wav)] = mean_wav
    valid_sub = ~np.isnan(sub_wav)

    col_labels = ['Left', 'Mid', 'Right']
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Sample chunk: offset(t) + the fitted sine.
    for ch in channels:
        if padded_per_ch[ch][sample_sub, sample_col]:
            sample_ch = ch
            break
    sample_y = fits[sample_ch]['offset_cube'][:, sample_sub, sample_col]
    sample_pred = (fits[sample_ch]['C'][sample_sub, sample_col]
                   + fits[sample_ch]['A'][sample_sub, sample_col]
                   * np.sin(2 * np.pi * mjd_full / SIDEREAL_YEAR_DAYS
                            + fits[sample_ch]['phi'][sample_sub, sample_col]))
    ax = axes[0, 0]
    sy_lo, sy_hi = np.percentile(sample_y[np.isfinite(sample_y)], [1, 99])
    inl = (sample_y >= sy_lo) & (sample_y <= sy_hi)
    ax.scatter(mjd_full[inl], sample_y[inl], s=2, alpha=0.25, color='grey',
               label='Per-Exposure Offset')
    order = np.argsort(mjd_full)
    ax.plot(mjd_full[order], sample_pred[order], color='C3', lw=1.8,
            label=(f'Sine Fit: C={fits[sample_ch]["C"][sample_sub, sample_col]:.4f}, '
                   f'A={fits[sample_ch]["A"][sample_sub, sample_col]:.4f}, '
                   f'φ={np.degrees(fits[sample_ch]["phi"][sample_sub, sample_col]):.0f}°'))
    ax.set(xlabel='MJD', ylabel='SelfCal Offset [MJy/sr]',
           title=(f'(a) Subchannel {sample_sub}, '
                  f'Wavelength {sub_wav[sample_sub]:.3f} μm'))
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    # (b) All fitted sine curves overlaid, coloured by wavelength.
    # One curve per subchannel (averaged over the 3 columns).
    ax = axes[0, 1]
    cmap = plt.cm.viridis
    norm = mcolors.Normalize(vmin=np.nanmin(sub_wav), vmax=np.nanmax(sub_wav))
    t_dense = np.linspace(mjd_full.min(), mjd_full.max(), 250)
    n_drawn = 0
    for s in np.where(valid_sub)[0]:
        with np.errstate(invalid='ignore'):
            A_s = float(np.nanmean(A_comb[s, :]))
            phi_s = float(np.nanmean(phi_comb[s, :]))
            C_s = float(np.nanmean(C_comb[s, :]))
        if not (np.isfinite(A_s) and np.isfinite(phi_s) and np.isfinite(C_s)):
            continue
        pred = C_s + A_s * np.sin(
            2 * np.pi * t_dense / SIDEREAL_YEAR_DAYS + phi_s)
        ax.plot(t_dense, pred, color=cmap(norm(sub_wav[s])), lw=0.6, alpha=0.7)
        n_drawn += 1
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm); sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Wavelength [μm]')
    ax.set(xlabel='MJD', ylabel='Sine Model [MJy/sr]',
           title=f'(b) Fitted Sine per Subchannel ({n_drawn} curves)')
    ax.grid(alpha=0.3)

    # (c) C(sub, col) vs wavelength (originally panel b).
    ax = axes[1, 0]
    for c in range(NUM_COL):
        ax.plot(sub_wav[valid_sub], C_comb[valid_sub, c],
                lw=1.5, alpha=0.9, label=col_labels[c])
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel='Wavelength [μm]',
           ylabel='C [MJy/sr]',
           title='(c) Time-Averaged Offset Spectrum')
    ax.legend(title='Column')
    ax.grid(alpha=0.3)

    # (d) A(sub, col) vs wavelength (originally panel c).
    ax = axes[1, 1]
    for c in range(NUM_COL):
        ax.plot(sub_wav[valid_sub], A_comb[valid_sub, c],
                lw=1.5, alpha=0.9, label=col_labels[c])
    ax.set(xlabel='Wavelength [μm]',
           ylabel='A [MJy/sr]',
           title='(d) Amplitude Spectrum')
    ax.legend(title='Column')
    ax.grid(alpha=0.3)

    fig.suptitle(f'Per-Chunk Annual Sine Fits — Detector {detector}  '
                 f'(Channels 1–{max(channels)}, '
                 f'Global φ = {np.degrees(phi_global):+.1f}°)', y=1.01)
    fig.tight_layout()
    out1 = fig_path(f'meeting_det{detector}_fig1_sine_fits.png')
    fig.savefig(out1, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out1}')

    # ------------------------------------------------------------------
    # FIGURE 2: residuals after the per-chunk sine (vs MJD / helio_lon /
    # elongation). Per-chunk binning if the per-chunk coord cache is
    # available; else fall back to per-exposure-mean residuals.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    keep = np.isfinite(resid_scalar)
    rs = resid_scalar[keep]
    lo, hi = np.percentile(rs, [0.5, 99.5])
    inlier = (rs >= lo) & (rs <= hi)

    def _bin(x, y, edges, min_count=30):
        idx = np.digitize(x, edges) - 1
        n = len(edges) - 1
        c = 0.5 * (edges[:-1] + edges[1:])
        m = np.full(n, np.nan); s = np.full(n, np.nan)
        for b in range(n):
            mb = idx == b
            if int(mb.sum()) >= min_count:
                m[b] = np.mean(y[mb])
                s[b] = np.std(y[mb]) / np.sqrt(int(mb.sum()))
        return c, m, s

    # Residual scans: use PER-CHUNK coordinates if the cache is available,
    # otherwise fall back to per-exposure-mean residuals + central pointing.
    if use_per_chunk:
        # Flatten (n_exp, n_chunks) into 1-D arrays of (exposure, chunk) pairs.
        flat_resid = resid_per_chunk_flat.ravel()
        flat_mjd = np.broadcast_to(mjd_full[:, None], resid_per_chunk_flat.shape).ravel()
        flat_helio = per_chunk_helio.reshape(-1)
        flat_elon = per_chunk_elon.reshape(-1)
        ok = np.isfinite(flat_resid) & np.isfinite(flat_helio) & np.isfinite(flat_elon)
        flat_resid = flat_resid[ok]; flat_mjd = flat_mjd[ok]
        flat_helio = flat_helio[ok]; flat_elon = flat_elon[ok]
        # Outlier clip on residual.
        lo2, hi2 = np.percentile(flat_resid, [0.5, 99.5])
        m_in = (flat_resid >= lo2) & (flat_resid <= hi2)
        flat_resid = flat_resid[m_in]; flat_mjd = flat_mjd[m_in]
        flat_helio = flat_helio[m_in]; flat_elon = flat_elon[m_in]
        scan_label = 'per-chunk residual'
    else:
        flat_resid = rs[inlier]
        flat_mjd = mjd_full[keep][inlier]
        flat_helio = helio_lon[keep][inlier]
        flat_elon = elongation[keep][inlier]
        scan_label = 'per-exposure residual'

    # Subsample for the scatter background (millions of (exp, chunk) points
    # in per-chunk mode; capping at 80k keeps the plot readable + small).
    rng = np.random.RandomState(0)
    n_scatter = min(80000, len(flat_resid))
    idx_scatter = rng.choice(len(flat_resid), n_scatter, replace=False)

    def _scatter_bin_panel(ax, x, y, edges, color, xlabel, panel_label,
                           ylim_pad_factor=4.0, sci_x=None):
        c, m, s = _bin(x, y, edges, min_count=200)
        pp = float(np.nanmax(m) - np.nanmin(m))
        ax.scatter(x[idx_scatter], y[idx_scatter], s=0.5,
                   alpha=0.05, color='grey', label='Residual')
        ax.errorbar(c, m, yerr=s, fmt='o-', color=color, lw=1.6, ms=4,
                    capsize=2, label='Binned Mean')
        ax.axhline(0, color='k', lw=0.6, ls='--')
        yspan = max(np.nanmax(np.abs(m)), 1e-4) * ylim_pad_factor
        ax.set(xlabel=xlabel, ylabel='SelfCal Offset - Sine Model [MJy/sr]',
               title=f'{panel_label}  (Peak-to-Peak {pp:.5f})',
               ylim=(-yspan, yspan))
        leg = ax.legend(fontsize=9, loc='upper right')
        for handle in leg.legend_handles:
            handle.set_alpha(1.0)
        ax.grid(alpha=0.3)

    edges_t = np.linspace(mjd_full.min(), mjd_full.max(), 40)
    _scatter_bin_panel(axes[0], flat_mjd, flat_resid, edges_t,
                       color='C0', xlabel='MJD',
                       panel_label='(a) Residual vs MJD')

    edges_h = np.linspace(-180, 180, 37)
    _scatter_bin_panel(axes[1], flat_helio, flat_resid, edges_h,
                       color='C3', xlabel='Helio-Ecliptic Longitude [deg]',
                       panel_label='(b) Residual vs Helio-Ecliptic Longitude')

    e_lo, e_hi = np.percentile(flat_elon, [0.5, 99.5])
    edges_e = np.linspace(e_lo, e_hi, 25)
    _scatter_bin_panel(axes[2], flat_elon, flat_resid, edges_e,
                       color='C2', xlabel='Solar Elongation [deg]',
                       panel_label='(c) Residual vs Solar Elongation')

    fig.suptitle(f'Residuals After Per-Chunk Sine Subtraction — Detector {detector}',
                 y=1.02)
    fig.tight_layout()
    out2 = fig_path(f'meeting_det{detector}_fig2_residuals.png')
    fig.savefig(out2, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out2}')

    # ------------------------------------------------------------------
    # FIGURE 3: detector-fixed extraction, both rendered on detector grid.
    # (a) C(sub, col)  -- time-averaged offset (spectral trend + detector fixed)
    # (b) D(sub, col)  -- C minus per-subchannel mean across cols
    #                     (pure detector-fixed component)
    # ------------------------------------------------------------------
    with np.errstate(invalid='ignore'):
        D = C_comb - np.nanmean(C_comb, axis=1, keepdims=True)

    det_chunk_map, _, r_edges, x_edges = make_stripped_chunk_map(
        detector, num_subchannels=NUM_SUB, num_channels=NUM_CH,
        num_columns=NUM_COL, oversample_factor=1, lvf_params=lvf,
    )

    def _render(field):
        chunk_offset = np.nan_to_num(field.reshape(-1), nan=0.0)
        chunk_valid = np.isfinite(field).reshape(-1).astype(float)
        return make_spherex_stripped_offset_map(
            chunk_map=det_chunk_map, chunk_offset=chunk_offset,
            chunk_valid_mask=chunk_valid, lvf_params=lvf,
            r_edges=r_edges, x_edges=x_edges,
            tot_subchannels=TOT_SUB, num_columns=NUM_COL,
            fill_invalid=True,
        )
    grid_C = _render(C_comb)
    grid_D = _render(D)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    ax = axes[0]
    nz = grid_C != 0
    vabs_c = np.nanpercentile(np.abs(grid_C[nz]), 99)
    im = ax.imshow(grid_C, cmap='RdBu_r', vmin=-vabs_c, vmax=vabs_c,
                   origin='lower')
    ax.set(xlabel='Detector X [pix]', ylabel='Detector Y [pix]',
           title='(a) C (time invariant offset) on Detector Grid')
    plt.colorbar(im, ax=ax, label='[MJy/sr]')

    ax = axes[1]
    nz = grid_D != 0
    vabs_d = np.nanpercentile(np.abs(grid_D[nz]), 99)
    im = ax.imshow(grid_D, cmap='RdBu_r', vmin=-vabs_d, vmax=vabs_d,
                   origin='lower')
    ax.set(xlabel='Detector X [pix]', ylabel='Detector Y [pix]',
           title='(b) $C - <C(\\lambda)>$ on Detector Grid')
    plt.colorbar(im, ax=ax, label='[MJy/sr]')

    fig.suptitle(
        f'Detector-Fixed Pattern Extraction — Detector {detector}',
        y=1.02,
    )
    fig.tight_layout()
    out3 = fig_path(f'meeting_det{detector}_fig3_detector_fixed.png')
    fig.savefig(out3, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out3}')

    # Save the key arrays for further use.
    np.savez(data_path(f'meeting_det{detector}_per_chunk_fits.npz'),
             C_comb=C_comb, A_comb=A_comb, phi_comb=phi_comb,
             D=D, resid_std_comb=resid_std_comb,
             shifts=np.array([shifts[ch] for ch in channels]),
             channels=np.array(channels))
    print(f'wrote cache/meeting_det{detector}_per_chunk_fits.npz')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    p.add_argument('--sample-sub', type=int, default=80,
                   help='subchannel index for the sample fit panel')
    p.add_argument('--sample-col', type=int, default=1,
                   help='column index (0-2) for the sample fit panel')
    args = p.parse_args()
    main(args.detector, sample_sub=args.sample_sub, sample_col=args.sample_col)
