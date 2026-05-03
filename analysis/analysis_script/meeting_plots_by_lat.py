"""Repeat the meeting plots in latitude bins.

Same three figures as meeting_plots.py, but the analysis is done
independently in each ecliptic-latitude bin so we can compare:

  * C, A, phi spectra across bins   (Fig 1)
  * post-fit residuals across bins  (Fig 2)
  * detector-fixed pattern D(sub, col) across bins  (Fig 3)

Key check: D should NOT depend on the latitude bin (it is detector-fixed),
while C and A SHOULD depend on the bin (zodi varies with ecliptic lat).
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
from zodi_utils import data_path, fig_path, cal_path, sine_model, fit_sine

NUM_SUB, NUM_CH, NUM_COL = 10, 34, 3
TOT_SUB = NUM_SUB * NUM_CH + 2
SIDEREAL_YEAR_DAYS = 365.25
DEFAULT_LAT_EDGES = [85.5, 86.5, 88.0, 90.0]  # 3 bins


def _padded_mask(channel):
    return make_stripped_chunk_valid_mask(
        ch=[channel], num_subchannels=NUM_SUB, num_channels=NUM_CH,
        num_columns=NUM_COL, subchannel_padding=1,
    ).reshape(TOT_SUB, NUM_COL).astype(bool)


def fits_for_subset(off, mjd, padded):
    """Per-chunk sine fits for a subset of exposures.

    off: (Nsub, 342, 3) for that channel
    mjd: (Nsub,)
    padded: (342, 3) bool

    Returns dicts of (342, 3) arrays: C, A, phi, resid_std, plus
    per-exposure-mean residual scalar (Nsub,).
    """
    n = off.shape[0]
    C = np.full((TOT_SUB, NUM_COL), np.nan)
    A = np.full((TOT_SUB, NUM_COL), np.nan)
    phi = np.full((TOT_SUB, NUM_COL), np.nan)
    resid_std = np.full((TOT_SUB, NUM_COL), np.nan)
    sine_pred = np.full(off.shape, np.nan)

    A_guess = 0.5 * (np.nanmax(off[:, padded[:, 0]]) - np.nanmin(off[:, padded[:, 0]]))
    A_guess = max(A_guess, 0.005)

    for s in range(TOT_SUB):
        for c in range(NUM_COL):
            if not padded[s, c]:
                continue
            y = off[:, s, c]
            ok = np.isfinite(y)
            if ok.sum() < 50:
                continue
            try:
                res = fit_sine(mjd[ok], y[ok], p0=(A_guess, 0.0, float(np.mean(y[ok]))))
            except Exception:
                continue
            C[s, c] = res['C']
            A[s, c] = res['A']
            phi[s, c] = res['phi']
            resid_std[s, c] = res['residual_std']
            sine_pred[:, s, c] = sine_model(mjd, A[s, c], phi[s, c], C[s, c])

    resid_cube = off - sine_pred
    return {'C': C, 'A': A, 'phi': phi, 'resid_std': resid_std,
            'resid_cube': resid_cube}


def stitch_C(C_per_ch, padded_per_ch, sorted_chs):
    shifts = {sorted_chs[0]: 0.0}
    for i in range(1, len(sorted_chs)):
        prev_c = sorted_chs[i - 1]; curr_c = sorted_chs[i]
        ov = padded_per_ch[prev_c] & padded_per_ch[curr_c]
        a = C_per_ch[prev_c]; b = C_per_ch[curr_c]
        good = ov & np.isfinite(a) & np.isfinite(b)
        d = float(np.mean(a[good] - b[good])) if good.any() else 0.0
        shifts[curr_c] = shifts[prev_c] + d
    return shifts


def analyse_bin(detector, lat_lo, lat_hi, df, channels,
                per_chunk_lat=None):
    """Run per-chunk fits + stitching for chunks whose own ecl_lat is in
    [lat_lo, lat_hi).

    If per_chunk_lat is None, fall back to binning by the exposure's
    central ecl_lat (legacy behaviour).
    """
    mjd_full = df['MJD_AVG'].values
    helio_lon_full = df['helio_lon'].values
    n_exp_total = len(df)

    padded_per_ch = {ch: _padded_mask(ch) for ch in channels}

    if per_chunk_lat is None:
        # Legacy: bin by exposure-central ecl_lat. Same for all chunks of
        # an exposure.
        sel_exp = (df['ecl_lat'].values >= lat_lo) & (df['ecl_lat'].values < lat_hi)
        n_in_bin = int(sel_exp.sum())
        if n_in_bin < 200:
            return None
        mjd_bin = mjd_full[sel_exp]
        helio_bin = helio_lon_full[sel_exp]
        fits = {}
        for ch in channels:
            with h5py.File(cal_path(detector, ch), 'r') as f:
                off = f['offset'][:].reshape(-1, TOT_SUB, NUM_COL)
            off = off[sel_exp]
            fits[ch] = fits_for_subset(off, mjd_bin, padded_per_ch[ch])
        bin_kind = 'central'
    else:
        # Per-(exposure, chunk) binning. For each chunk, pick the exposures
        # in which THAT chunk's projected ecl_lat falls in the bin, then fit
        # a sine over those exposures.
        # per_chunk_lat is shape (n_exp, n_chunks).
        in_bin_mat = (per_chunk_lat >= lat_lo) & (per_chunk_lat < lat_hi)  # (n_exp, n_chunks)
        # Quick check: skip if too few (exp, chunk) hits.
        n_in_bin = int(in_bin_mat.sum())
        if n_in_bin < 1000:
            return None
        # mjd / helio_lon for plots remain per-exposure (averaged across
        # chunks in this bin per exposure).
        any_exp = in_bin_mat.any(axis=1)
        mjd_bin = mjd_full[any_exp]
        helio_bin = helio_lon_full[any_exp]

        fits = {}
        for ch in channels:
            with h5py.File(cal_path(detector, ch), 'r') as f:
                off = f['offset'][:].reshape(-1, TOT_SUB, NUM_COL)
            # For each (sub, col) in this channel, mask out exposures whose
            # chunk-specific lat is OUTSIDE this bin -> set the offset entry
            # to NaN there. Then fits_for_subset will use only the in-bin
            # exposures for that chunk's sine fit.
            n_total = off.shape[0]
            off_masked = off.copy()
            for s in range(TOT_SUB):
                for c in range(NUM_COL):
                    cid = s * NUM_COL + c
                    out_of_bin = ~in_bin_mat[:, cid]
                    off_masked[out_of_bin, s, c] = np.nan
            fits[ch] = fits_for_subset(off_masked, mjd_full, padded_per_ch[ch])
        bin_kind = 'per-chunk'
        n_in_bin_exp_equiv = int(any_exp.sum())

    # Stitched, channel-averaged C / A / phi maps.
    shifts = stitch_C({ch: fits[ch]['C'] for ch in channels},
                      padded_per_ch, channels)
    def combine(metric):
        stack = np.stack(
            [np.where(padded_per_ch[ch], fits[ch][metric], np.nan)
             for ch in channels], axis=0)
        with np.errstate(invalid='ignore'):
            return np.nanmean(stack, axis=0)
    A_comb = combine('A')
    phi_comb = combine('phi')
    resid_std_comb = combine('resid_std')
    C_stack = np.stack(
        [np.where(padded_per_ch[ch], fits[ch]['C'] + shifts[ch], np.nan)
         for ch in channels], axis=0)
    with np.errstate(invalid='ignore'):
        C_comb = np.nanmean(C_stack, axis=0)
    C_comb = C_comb - np.nanmean(C_comb)

    # Per-exposure residual scalar (mean over all in-bin valid chunks across
    # all channels). Length matches resid_cube along axis 0.
    n_resid_axis = next(iter(fits.values()))['resid_cube'].shape[0]
    resid_scalar = np.zeros(n_resid_axis); counts = np.zeros(n_resid_axis)
    for ch in channels:
        rc = fits[ch]['resid_cube']
        m = padded_per_ch[ch][None, :, :] & np.isfinite(rc)
        resid_scalar += np.where(m, rc, 0.0).sum(axis=(1, 2))
        counts += m.sum(axis=(1, 2))
    resid_scalar = np.where(counts > 0, resid_scalar / counts, np.nan)

    # If we ran the per-chunk path, the resid array spans ALL exposures and
    # has NaN for those with no in-bin chunk. Subset to valid entries and
    # take their MJD / helio_lon from the full per-exposure arrays.
    if per_chunk_lat is not None:
        ok = np.isfinite(resid_scalar)
        resid_scalar = resid_scalar[ok]
        mjd_bin = mjd_full[ok]
        helio_bin = helio_lon_full[ok]
        n_in_bin_out = int(ok.sum())
    else:
        n_in_bin_out = n_in_bin

    # Detector-fixed: subtract per-subchannel mean across columns.
    with np.errstate(invalid='ignore'):
        D = C_comb - np.nanmean(C_comb, axis=1, keepdims=True)
    return {
        'lat_lo': lat_lo, 'lat_hi': lat_hi,
        'n_exposures': n_in_bin_out,
        'binning_kind': bin_kind,
        'C': C_comb, 'A': A_comb, 'phi': phi_comb,
        'resid_std': resid_std_comb, 'D': D,
        'resid_scalar': resid_scalar,
        'mjd': mjd_bin, 'helio_lon': helio_bin,
    }


def _bin(x, y, edges, min_count=20):
    idx = np.digitize(x, edges) - 1
    n = len(edges) - 1
    c = 0.5 * (edges[:-1] + edges[1:])
    mean = np.full(n, np.nan); sem = np.full(n, np.nan)
    for b in range(n):
        m = idx == b
        if int(m.sum()) >= min_count:
            mean[b] = np.mean(y[m])
            sem[b] = np.std(y[m]) / np.sqrt(int(m.sum()))
    return c, mean, sem


def main(detector, lat_edges, per_chunk_binning=False):
    df = pd.read_pickle(data_path(f'multichannel_det{detector}.pkl'))
    df = df[df['channel'] == 1].reset_index(drop=True)
    channels = list(range(1, 18))

    # By default we bin by the exposure's central ecl_lat (CRVAL2 -> ecliptic).
    # This guarantees that within a bin every chunk uses the SAME exposure
    # subset, which is essential for a clean detector-fixed comparison
    # (D = C - <C>_cols would otherwise mix C values computed from
    # different exposure subsets per column).
    #
    # The per-chunk-binning mode (--per-chunk) assigns each (exposure, chunk)
    # pair to its own bin based on the chunk's projected sky coords. This is
    # technically a finer assignment, but it can introduce sampling bias in
    # cross-column comparisons. Use it only as a diagnostic.
    if per_chunk_binning:
        pc_path = data_path(f'perchunk_coords_det{detector}.npz')
        if not os.path.exists(pc_path):
            raise FileNotFoundError(f'per-chunk cache missing: {pc_path}')
        per_chunk_lat = np.load(pc_path)['ecl_lat']
        print(f'PER-CHUNK lat binning (diagnostic). Using {pc_path}.')
    else:
        per_chunk_lat = None
        print('PER-EXPOSURE (CRVAL2-central) lat binning.')

    bins = []
    for i in range(len(lat_edges) - 1):
        print(f'Analysing lat bin [{lat_edges[i]:.2f}, {lat_edges[i+1]:.2f}]...')
        b = analyse_bin(detector, lat_edges[i], lat_edges[i+1], df, channels,
                        per_chunk_lat=per_chunk_lat)
        if b is None:
            print('  too few exposures; skipped.')
            continue
        bins.append(b)
        print(f'  binning={b["binning_kind"]}  '
              f'{b["n_exposures"]} exposures, residual std '
              f'(per-exposure mean) = {np.nanstd(b["resid_scalar"]):.5f}')

    cmap = plt.cm.viridis
    colors = [cmap(i / max(1, len(bins) - 1)) for i in range(len(bins))]

    sub_idx = np.arange(TOT_SUB)

    # ------------------------------------------------------------------
    # FIGURE 1: C, A, phi vs subchannel, one curve per bin (col 1 only,
    # so the 3-column structure does not clutter; col 1 is the middle).
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharex=True)
    for b, color in zip(bins, colors):
        label = f'lat [{b["lat_lo"]:.1f}, {b["lat_hi"]:.1f}]  N={b["n_exposures"]}'
        axes[0].plot(sub_idx, b['C'][:, 1], color=color, lw=1.5, label=label)
        axes[1].plot(sub_idx, b['A'][:, 1], color=color, lw=1.5, label=label)
        axes[2].plot(sub_idx, np.degrees(b['phi'][:, 1]), color=color, lw=1.5,
                     label=label)
    axes[0].axhline(0, color='k', lw=0.5, ls='--')
    axes[0].set(xlabel='subchannel', ylabel='C (centred) [MJy/sr]',
                title='(a) time-averaged offset spectrum, per lat bin (col 1)')
    axes[0].legend(loc='upper left', fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[1].set(xlabel='subchannel', ylabel='A (annual amplitude) [MJy/sr]',
                title='(b) annual amplitude vs subchannel, per lat bin')
    axes[1].grid(alpha=0.3)
    axes[2].set(xlabel='subchannel', ylabel=r'$\phi$ [deg]',
                title='(c) annual phase vs subchannel, per lat bin')
    axes[2].grid(alpha=0.3)
    fig.suptitle(f'Per-chunk sine fits in latitude bins  det{detector}', y=1.02)
    fig.tight_layout()
    out1 = fig_path(f'meeting_det{detector}_bylat_fig1_sine_fits.png')
    fig.savefig(out1, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out1}')

    # ------------------------------------------------------------------
    # FIGURE 2: residuals per bin (one row per bin), 2 cols (vs MJD, vs hl).
    # ------------------------------------------------------------------
    n_bins = len(bins)
    fig, axes = plt.subplots(n_bins, 2, figsize=(13, 3.6 * n_bins), squeeze=False)
    edges_t = np.linspace(df['MJD_AVG'].min(), df['MJD_AVG'].max(), 35)
    edges_h = np.linspace(-180, 180, 37)
    for i, (b, color) in enumerate(zip(bins, colors)):
        rs = b['resid_scalar']
        keep = np.isfinite(rs)
        rs = rs[keep]
        mjd = b['mjd'][keep]; hl = b['helio_lon'][keep]
        lo, hi = np.percentile(rs, [0.5, 99.5])
        inl = (rs >= lo) & (rs <= hi)
        c, m, s = _bin(mjd[inl], rs[inl], edges_t)
        pp_t = float(np.nanmax(m) - np.nanmin(m))
        ax = axes[i, 0]
        ax.scatter(mjd[inl], rs[inl], s=1, alpha=0.04, color='grey')
        ax.errorbar(c, m, yerr=s, fmt='o-', color=color, lw=1.5, ms=4, capsize=2)
        ax.axhline(0, color='k', lw=0.5, ls='--')
        yspan = max(np.nanmax(np.abs(m)), 1e-4) * 4
        ax.set(xlabel='MJD', ylabel='mean residual [MJy/sr]',
               title=f'lat [{b["lat_lo"]:.1f}, {b["lat_hi"]:.1f}]'
                     f' (N={b["n_exposures"]})  vs MJD  pp={pp_t:.5f}',
               ylim=(-yspan, yspan))
        ax.grid(alpha=0.3)

        c, m, s = _bin(hl[inl], rs[inl], edges_h)
        pp_h = float(np.nanmax(m) - np.nanmin(m))
        ax = axes[i, 1]
        ax.scatter(hl[inl], rs[inl], s=1, alpha=0.04, color='grey')
        ax.errorbar(c, m, yerr=s, fmt='o-', color=color, lw=1.5, ms=4, capsize=2)
        ax.axhline(0, color='k', lw=0.5, ls='--')
        yspan_h = max(np.nanmax(np.abs(m)), 1e-4) * 4
        ax.set(xlabel='helio-ecliptic lon [deg]', ylabel='mean residual [MJy/sr]',
               title=f'vs helio_lon  pp={pp_h:.5f}',
               ylim=(-yspan_h, yspan_h))
        ax.grid(alpha=0.3)

    fig.suptitle(f'Residuals after per-chunk sine, by lat bin  det{detector}',
                 y=1.005)
    fig.tight_layout()
    out2 = fig_path(f'meeting_det{detector}_bylat_fig2_residuals.png')
    fig.savefig(out2, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out2}')

    # ------------------------------------------------------------------
    # FIGURE 3: detector-fixed D(sub, col) for each bin.
    # Show side-by-side maps + col-difference summary curves.
    # ------------------------------------------------------------------
    n_bins = len(bins)
    fig, axes = plt.subplots(2, max(n_bins, 2), figsize=(5 * max(n_bins, 2), 9),
                             squeeze=False)
    vabs = np.nanpercentile(
        np.abs(np.concatenate([b['D'].ravel() for b in bins])), 99)
    for i, b in enumerate(bins):
        ax = axes[0, i]
        im = ax.imshow(b['D'], aspect='auto', cmap='RdBu_r',
                       vmin=-vabs, vmax=vabs, origin='lower')
        ax.set(xlabel='column', ylabel='subchannel',
               title=f'D  lat [{b["lat_lo"]:.1f}, {b["lat_hi"]:.1f}]')
        plt.colorbar(im, ax=ax, label='[MJy/sr]')

    # Bottom row: col0-col2 vs subchannel, one curve per bin (clean comparison).
    ax = axes[1, 0]
    for b, color in zip(bins, colors):
        diff = b['D'][:, 0] - b['D'][:, 2]
        label = f'lat [{b["lat_lo"]:.1f}, {b["lat_hi"]:.1f}]'
        ax.plot(sub_idx, diff, color=color, lw=1.5, label=label)
    ax.axhline(0, color='k', lw=0.5, ls='--')
    ax.set(xlabel='subchannel',
           ylabel='D[col 0] - D[col 2]  [MJy/sr]',
           title='detector-fixed col0 - col2 across bins')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(alpha=0.3)

    # If we have <=2 lat bins, no extra panel needed; for >=3 show a difference
    # heatmap between the extreme bins to confirm D is bin-invariant.
    if n_bins >= 2:
        ax = axes[1, 1]
        diff = bins[-1]['D'] - bins[0]['D']
        vd = np.nanpercentile(np.abs(diff), 99)
        im = ax.imshow(diff, aspect='auto', cmap='PuOr_r',
                       vmin=-vd, vmax=vd, origin='lower')
        ax.set(xlabel='column', ylabel='subchannel',
               title=f'D[lat>{bins[-1]["lat_lo"]:.1f}] - D[lat<{bins[0]["lat_hi"]:.1f}]')
        plt.colorbar(im, ax=ax, label='[MJy/sr]')

    # Hide any unused axes.
    for j in range(n_bins, axes.shape[1]):
        axes[0, j].axis('off')
    if n_bins < 2:
        for j in range(2, axes.shape[1]):
            axes[1, j].axis('off')
    elif n_bins < axes.shape[1]:
        for j in range(2, axes.shape[1]):
            axes[1, j].axis('off')

    fig.suptitle(
        f'Detector-fixed D(sub, col) by lat bin  det{detector}',
        y=1.01,
    )
    fig.tight_layout()
    out3 = fig_path(f'meeting_det{detector}_bylat_fig3_detector_fixed.png')
    fig.savefig(out3, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out3}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    p.add_argument('--lat-edges', type=float, nargs='+',
                   default=DEFAULT_LAT_EDGES,
                   help='ecliptic-latitude bin edges (deg)')
    p.add_argument('--per-chunk', action='store_true',
                   help='diagnostic mode: bin per-(exposure, chunk) on the '
                        'projected lat instead of the exposure-central lat. '
                        'Cleaner geometry per chunk, but introduces sampling '
                        'bias when comparing columns within the same bin.')
    args = p.parse_args()
    main(args.detector, args.lat_edges, per_chunk_binning=args.per_chunk)
