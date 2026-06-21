"""Per-channel likelihood-ratio / BIC test for slope freedom.

Tests, per channel, whether the free-slope anchor fit is STATISTICALLY
required vs the null hypothesis slope=1 ("zodipy is correctly calibrated
for this channel"). On the same per-channel inlier set used by the anchor
file (recomputed from fit_with_clip with the same params stored as root
attrs in anchor_D{N}.h5):

    rss_free   = sum((fdc - slope_free*zp - C_free)^2)
    rss_locked = sum((fdc - zp - C_locked)^2),  C_locked = mean(fdc) - mean(zp)

    F_stat    = ((rss_locked - rss_free) / 1) / (rss_free / (n - 2))
              ~ F(1, n-2)  under H0
    p_value   = scipy.stats.f.sf(F_stat, 1, n-2)
    delta_BIC = n * log(rss_locked / rss_free) - log(n)
    sigma_slope = sqrt( (rss_free/(n-2)) / sum((zp_inl - mean(zp_inl))^2) )
    z_slope   = (slope_free - 1) / sigma_slope

Per channel report: slope_free, sigma_slope_free, F, log10(p), delta_BIC,
z_slope.

For each detector report channel-fractions surviving p<0.01, p<0.001,
p<1e-6, with the surviving channel ids and a check for wavelength
clustering.

Plot (3-panel):
  (a) log10(p_value) vs lambda for D3/D4/D5, with horizontal threshold lines
  (b) slope_free with +/-2 sigma error bars (horizontal line at 1.0)
  (c) delta_BIC vs lambda

Read-only on cal/anchor/zodi_pred files. Reuses load_detector and
per_channel_from_anchor from refit_smooth_slope.py and fit_with_clip
from SelfCal.ZodiAnchor.
"""
import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from selfcal.ZodiAnchor import fit_with_clip, load_anchor

# Add this directory to path so we can reuse refit_smooth_slope helpers.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from refit_smooth_slope import load_detector, per_channel_from_anchor  # noqa: E402


DET_COLORS = {1: 'tab:purple', 2: 'tab:orange',
              3: 'tab:green', 4: 'tab:blue', 5: 'tab:red'}
THRESHOLDS = [
    ('lenient',  1e-2,  'p < 0.01'),
    ('standard', 1e-3,  'p < 0.001'),
    ('strict',   1e-6,  'p < 1e-6'),
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--run-dir', nargs='+', required=True,
                   help='SPHEREx run directories (one per detector). Each '
                        'must contain calibration/, zodi_preds/, '
                        'and zodi_anchor/anchor_D*.h5.')
    p.add_argument('--cal-glob-pat', default='cal_*polyK1.h5',
                   help="Glob inside <run>/calibration "
                        "(default 'cal_*polyK1.h5').")
    p.add_argument('--out-plot', default=None,
                   help='Output PNG. Default: '
                        'figures/zodi_anchor/slope_lrt.png')
    p.add_argument('--out-data', default=None,
                   help='Output .npz. Default: derived from --out-plot.')
    p.add_argument('--cluster-gap-um', type=float, default=0.25,
                   help='Maximum lambda gap (um) for two surviving channels '
                        'to be in the same cluster (default 0.25).')
    return p.parse_args()


def _anchor_root_clip_attrs(anchor_path):
    with h5py.File(anchor_path, 'r') as f:
        w = float(f.attrs['clip_window_days'])
        s = float(f.attrs['clip_sigma'])
        it = int(f.attrs['clip_iters'])
        method = str(f.attrs.get('anchor_method', 'raw'))
    return dict(window_days=w, sigma=s, iters=it, anchor_method=method)


def lrt_one_channel(zp, fdc, mjds, clip):
    """Run fit_with_clip with the anchor's params, then compute LRT stats
    on the resulting inlier set.

    Returns a dict per channel with all the statistics.
    """
    slope_free, intercept_free, r, inlier = fit_with_clip(
        zp, fdc, mjds,
        window_days=clip['window_days'],
        sigma=clip['sigma'],
        iters=clip['iters'],
    )
    n = int(inlier.sum())
    out = dict(
        slope_free=float(slope_free),
        intercept_free=float(intercept_free),
        pearson_r=float(r),
        n_inliers=n,
    )
    if n < 5:
        out.update(rss_free=np.nan, rss_locked=np.nan,
                   C_locked=np.nan, sigma_slope=np.nan,
                   F=np.nan, log10_p=np.nan, delta_BIC=np.nan,
                   z_slope=np.nan)
        return out

    zp_in = zp[inlier]
    fdc_in = fdc[inlier]

    resid_free = fdc_in - (slope_free * zp_in + intercept_free)
    rss_free = float(np.sum(resid_free ** 2))

    mu_fdc = float(fdc_in.mean())
    mu_zp = float(zp_in.mean())
    C_locked = mu_fdc - mu_zp
    resid_locked = fdc_in - (zp_in + C_locked)
    rss_locked = float(np.sum(resid_locked ** 2))

    # OLS slope SE on the inlier set
    zp_centered = zp_in - mu_zp
    Sxx = float(np.sum(zp_centered ** 2))
    dof = n - 2
    if dof <= 0 or Sxx <= 0 or rss_free <= 0:
        out.update(rss_free=rss_free, rss_locked=rss_locked,
                   C_locked=C_locked, sigma_slope=np.nan,
                   F=np.nan, log10_p=np.nan, delta_BIC=np.nan,
                   z_slope=np.nan)
        return out
    sigma2 = rss_free / dof
    sigma_slope = float(np.sqrt(sigma2 / Sxx))

    # F statistic for slope=1 vs free slope
    F = ((rss_locked - rss_free) / 1.0) / sigma2
    # Guard against tiny negative due to FP if both models numerically equal
    F = float(max(F, 0.0))

    # log10(p) via logsf to avoid underflow on huge F.
    if F <= 0.0:
        log10_p = 0.0
    else:
        log_p = float(stats.f.logsf(F, 1, dof))
        log10_p = log_p / np.log(10.0)

    # delta_BIC, positive => free preferred over locked
    if rss_free > 0 and rss_locked > 0:
        delta_BIC = float(n * np.log(rss_locked / rss_free) - np.log(n))
    else:
        delta_BIC = np.nan

    z_slope = float((slope_free - 1.0) / sigma_slope)

    out.update(
        rss_free=rss_free,
        rss_locked=rss_locked,
        C_locked=float(C_locked),
        sigma_slope=sigma_slope,
        F=float(F),
        log10_p=float(log10_p),
        delta_BIC=delta_BIC,
        z_slope=z_slope,
    )
    return out


def run_detector(det_data, clip):
    """Run the LRT on every channel of a detector. Returns per-channel
    arrays of every test stat, plus wavelengths and channel ids."""
    chs = det_data['channels']
    wl = det_data['WL']
    n_ch = len(chs)
    fields = ['slope_free', 'intercept_free', 'pearson_r', 'n_inliers',
              'rss_free', 'rss_locked', 'C_locked', 'sigma_slope',
              'F', 'log10_p', 'delta_BIC', 'z_slope']
    arrs = {k: np.full(n_ch, np.nan, dtype=np.float64) for k in fields}
    arrs['n_inliers'] = np.zeros(n_ch, dtype=np.int64)
    for i, ch in enumerate(chs):
        zp = det_data['ZP'][i]
        fdc = det_data['FDC'][i]
        mjds = det_data['MJD'][i]
        if len(zp) == 0:
            continue
        out = lrt_one_channel(zp, fdc, mjds, clip)
        for k in fields:
            arrs[k][i] = out[k]
    arrs['channels'] = np.asarray(chs, dtype=np.int32)
    arrs['wl'] = np.asarray(wl, dtype=np.float64)
    return arrs


def find_clusters(wl_sorted, gap):
    """Group consecutive entries of a sorted lambda array into clusters
    separated by gaps > `gap`. Returns list of (start_idx, end_idx) (inclusive)
    in the input array."""
    n = len(wl_sorted)
    if n == 0:
        return []
    clusters = []
    start = 0
    for i in range(1, n):
        if (wl_sorted[i] - wl_sorted[i - 1]) > gap:
            clusters.append((start, i - 1))
            start = i
    clusters.append((start, n - 1))
    return clusters


def summarize_thresholds(stats_by_det, cluster_gap):
    """For each detector, report fraction surviving each threshold,
    the surviving channels and their wavelength clusters.

    Note: for very-significant channels, scipy's f.logsf underflows to
    -inf (p effectively zero); we treat that as VALID and SURVIVING for
    every threshold. Only NaN entries (LRT undefined) are dropped from
    n_total.
    """
    summary = {}
    for det, S in stats_by_det.items():
        log10_p = S['log10_p']
        chs = S['channels']
        wl = S['wl']
        # "Valid LRT" = not NaN. -inf (p underflow) counts as valid and
        # passes every threshold below.
        valid = ~np.isnan(log10_p)
        n_total = int(valid.sum())
        per_thresh = {}
        for tag, p_thr, label in THRESHOLDS:
            log_p_thr = np.log10(p_thr)
            sel = valid & (log10_p < log_p_thr)
            sel_idx = np.where(sel)[0]
            n_sel = int(sel.sum())
            wl_sel = wl[sel_idx]
            ch_sel = chs[sel_idx]
            order = np.argsort(wl_sel)
            wl_sorted = wl_sel[order]
            ch_sorted = ch_sel[order]
            clusters = find_clusters(wl_sorted, cluster_gap)
            cluster_desc = []
            for a, b in clusters:
                desc = dict(
                    lam_lo=float(wl_sorted[a]),
                    lam_hi=float(wl_sorted[b]),
                    n=int(b - a + 1),
                    channels=ch_sorted[a:b + 1].tolist(),
                )
                cluster_desc.append(desc)
            per_thresh[tag] = dict(
                p_threshold=p_thr,
                label=label,
                n_survivors=n_sel,
                n_total=n_total,
                frac=float(n_sel / n_total) if n_total > 0 else float('nan'),
                wl_sorted=wl_sorted,
                channels_sorted=ch_sorted,
                clusters=cluster_desc,
            )
        summary[det] = dict(n_total=n_total, per_thresh=per_thresh)
    return summary


def print_summary(stats_by_det, summary, anchor_paths):
    """Console report."""
    print("\n=== Per-channel LRT / BIC test for slope freedom ===")
    for det in sorted(stats_by_det):
        S = stats_by_det[det]
        n_valid = int((~np.isnan(S['log10_p'])).sum())
        n_total = len(S['channels'])
        print(f"\n--- D{det}  ({n_valid}/{n_total} channels with valid LRT) ---")
        print(f"  anchor: {anchor_paths[det]}")
        # Pretty per-channel table
        print(f"  {'ch':>3}  {'lambda':>7}  {'slope':>8}  {'sigma_s':>8}  "
              f"{'n':>6}  {'F':>11}  {'log10(p)':>10}  {'dBIC':>11}  "
              f"{'z_slope':>10}")
        order = np.argsort(S['wl'])
        for i in order:
            log10p = S['log10_p'][i]
            if np.isnan(log10p):
                log10p_s = f"{'nan':>10}"
            elif np.isneginf(log10p):
                log10p_s = f"{'-inf':>10}"
            else:
                log10p_s = f"{log10p:>10.2f}"
            print(
                f"  {int(S['channels'][i]):>3}  {S['wl'][i]:>7.3f}  "
                f"{S['slope_free'][i]:>8.4f}  {S['sigma_slope'][i]:>8.2e}  "
                f"{int(S['n_inliers'][i]):>6}  {S['F'][i]:>11.3e}  "
                f"{log10p_s}  {S['delta_BIC'][i]:>11.3e}  "
                f"{S['z_slope'][i]:>10.2f}")

    print("\n=== Threshold survival summary ===")
    for det in sorted(summary):
        ds = summary[det]
        print(f"\nD{det}  (n_total = {ds['n_total']} channels with valid LRT)")
        for tag, _p_thr, label in THRESHOLDS:
            pt = ds['per_thresh'][tag]
            print(f"  {label:<14s}  survivors: {pt['n_survivors']:>3d} / "
                  f"{pt['n_total']:>3d}  ({100.0 * pt['frac']:5.1f}%)")
            if pt['n_survivors'] == 0:
                continue
            ch_str = ', '.join(f"Ch{int(c)}({lam:.3f})" for c, lam in
                               zip(pt['channels_sorted'], pt['wl_sorted']))
            print(f"      channels: {ch_str}")
            if len(pt['clusters']) > 1 or (pt['clusters']
                                            and pt['clusters'][0]['n'] > 1):
                print(f"      {len(pt['clusters'])} wavelength cluster(s)"
                      f" (gap <= {pt.get('cluster_gap_um', 0.25)} um implicit):")
                for cl in pt['clusters']:
                    print(f"        [{cl['lam_lo']:.3f}, {cl['lam_hi']:.3f}] "
                          f"um  n={cl['n']}  chs={cl['channels']}")


def make_plot(stats_by_det, out_plot):
    fig, (ax_p, ax_slope, ax_bic) = plt.subplots(
        3, 1, figsize=(11, 11), sharex=True,
        gridspec_kw=dict(hspace=0.18))

    # ----- (a) log10(p) vs lambda -----
    # Find a finite floor for the y-axis so -inf log10(p) entries can be
    # plotted at the floor with an open down-arrow marker.
    finite_min = np.inf
    for S in stats_by_det.values():
        lp = S['log10_p']
        m = np.isfinite(lp)
        if m.any():
            finite_min = min(finite_min, float(lp[m].min()))
    if not np.isfinite(finite_min):
        finite_min = -300.0  # paranoid default
    floor = finite_min - 50.0  # leave headroom for the -inf arrow row
    for det in sorted(stats_by_det):
        S = stats_by_det[det]
        col = DET_COLORS.get(det, 'tab:purple')
        order = np.argsort(S['wl'])
        wl = S['wl'][order]
        lp = S['log10_p'][order].copy()
        # plot the finite part as a line
        finite_mask = np.isfinite(lp)
        ax_p.plot(wl[finite_mask], lp[finite_mask],
                  marker='o', mfc='none', ls='-', lw=0.9, ms=5,
                  color=col, label=f'D{det}')
        # -inf -> draw a downward-pointing marker at the floor
        neginf_mask = np.isneginf(lp)
        if neginf_mask.any():
            ax_p.scatter(wl[neginf_mask],
                         np.full(neginf_mask.sum(), floor),
                         marker='v', s=36, facecolors='none',
                         edgecolors=col)
    ax_p.set_ylim(floor - 20.0, 5.0)

    for tag, p_thr, label in THRESHOLDS:
        y = float(np.log10(p_thr))
        ax_p.axhline(y, color='black', lw=0.7, ls='--', alpha=0.6)
        ax_p.text(ax_p.get_xlim()[1] * 0.995, y, f'  {label}',
                  ha='right', va='bottom', fontsize=8, alpha=0.7,
                  transform=ax_p.transData)
    ax_p.set_ylabel(r'$\log_{10}(p)$ for slope = 1')
    ax_p.set_title('(a) Per-channel likelihood-ratio test: log10(p) vs lambda '
                   '(below the line => free slope statistically required)')
    ax_p.grid(alpha=0.3)
    ax_p.legend(loc='lower right', fontsize=8)
    # Invert y so very-significant (large negative log10 p) sits at the top.
    ax_p.invert_yaxis()

    # ----- (b) slope_free +/- 2 sigma -----
    for det in sorted(stats_by_det):
        S = stats_by_det[det]
        col = DET_COLORS.get(det, 'tab:purple')
        order = np.argsort(S['wl'])
        wl = S['wl'][order]
        slp = S['slope_free'][order]
        ss = S['sigma_slope'][order]
        # Use 2*sigma error bars
        yerr = 2.0 * ss
        ax_slope.errorbar(wl, slp, yerr=yerr, fmt='o', mfc='none', ms=5,
                          color=col, lw=0.9, capsize=2.5,
                          label=f'D{det}')
    ax_slope.axhline(1.0, color='gray', lw=0.7, ls='--', alpha=0.7)
    ax_slope.set_ylabel('slope_free (+/-2 sigma)')
    ax_slope.set_title('(b) Free-slope estimate with 2-sigma OLS error bars '
                       '(error bars often smaller than the marker)')
    ax_slope.grid(alpha=0.3)
    ax_slope.legend(loc='best', fontsize=8)

    # ----- (c) delta_BIC vs lambda -----
    for det in sorted(stats_by_det):
        S = stats_by_det[det]
        col = DET_COLORS.get(det, 'tab:purple')
        order = np.argsort(S['wl'])
        wl = S['wl'][order]
        dbic = S['delta_BIC'][order]
        ax_bic.plot(wl, dbic, marker='o', mfc='none', ls='-', lw=0.9, ms=5,
                    color=col, label=f'D{det}')
    ax_bic.axhline(0.0, color='gray', lw=0.7, ls='--', alpha=0.7)
    # Use symmetric log to handle the huge dynamic range
    try:
        ax_bic.set_yscale('symlog', linthresh=10.0)
    except Exception:
        pass
    ax_bic.set_ylabel(r'$\Delta$BIC  (BIC_locked - BIC_free)')
    ax_bic.set_xlabel(r'$\lambda$  ($\mu$m)')
    ax_bic.set_title('(c) delta_BIC vs lambda  '
                     '(positive => free-slope preferred over slope = 1)')
    ax_bic.grid(alpha=0.3, which='both')
    ax_bic.legend(loc='best', fontsize=8)

    fig.suptitle('Per-channel LR / BIC test for slope freedom '
                 '(H0: slope = 1, zodipy correctly calibrated)',
                 y=0.995, fontsize=12)
    plt.savefig(out_plot, dpi=130, bbox_inches='tight')
    print(f"\nSaved plot: {out_plot}")


def sanity_check_anchor(stats_by_det, anchor_paths):
    """Cross-check: recomputed slope_free should equal anchor.slope (the
    RAW per-channel fit, not slope_final which may have been smoothed)."""
    print("\n=== Sanity check vs anchor 'slope' / 'n_inliers' attrs ===")
    for det in sorted(stats_by_det):
        S = stats_by_det[det]
        a = load_anchor(anchor_paths[det])
        max_dslope = 0.0
        max_dn = 0
        for i, c in enumerate(S['channels']):
            c = int(c)
            if c not in a.channels:
                continue
            a_slope = float(a.channels[c]['slope'])
            a_n = int(a.channels[c]['n_inliers'])
            if np.isfinite(S['slope_free'][i]):
                d = abs(S['slope_free'][i] - a_slope)
                if d > max_dslope:
                    max_dslope = float(d)
            dn = int(abs(int(S['n_inliers'][i]) - a_n))
            if dn > max_dn:
                max_dn = dn
        print(f"  D{det}: max |slope_recomputed - anchor.slope| = "
              f"{max_dslope:.3e}; max |n_inliers diff| = {max_dn}")


def main():
    args = parse_args()
    out_plot = (args.out_plot or os.path.join(
        'figures', 'zodi_anchor', 'slope_lrt.png'))
    out_data = (args.out_data or
                (os.path.splitext(out_plot)[0] + '.npz'))
    os.makedirs(os.path.dirname(os.path.abspath(out_plot)) or '.',
                exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_data)) or '.',
                exist_ok=True)

    # ----- load detectors -----
    print("Loading detectors...")
    detectors_data = []
    anchor_paths = {}
    clip_per_det = {}
    for run_dir in args.run_dir:
        print(f"  {run_dir}")
        det_data = load_detector(run_dir, args.cal_glob_pat)
        det = det_data['detector']
        n_ch = len(det_data['channels'])
        n_frames_per_ch = [len(f) for f in det_data['FDC']]
        print(f"    D{det}: {n_ch} channels, "
              f"N_frames range = "
              f"[{min(n_frames_per_ch)}, {max(n_frames_per_ch)}]")
        anchor_paths[det] = det_data['anchor_path']
        clip_per_det[det] = _anchor_root_clip_attrs(det_data['anchor_path'])
        print(f"    anchor clip: window_days={clip_per_det[det]['window_days']}, "
              f"sigma={clip_per_det[det]['sigma']}, "
              f"iters={clip_per_det[det]['iters']}, "
              f"method='{clip_per_det[det]['anchor_method']}'")
        detectors_data.append(det_data)
    detectors_data.sort(key=lambda d: d['detector'])

    # ----- run LRT -----
    stats_by_det = {}
    for det_data in detectors_data:
        det = det_data['detector']
        print(f"\nRunning LRT for D{det}...")
        S = run_detector(det_data, clip_per_det[det])
        stats_by_det[det] = S

    # ----- sanity check -----
    sanity_check_anchor(stats_by_det, anchor_paths)

    # ----- threshold summary -----
    summary = summarize_thresholds(stats_by_det, args.cluster_gap_um)
    print_summary(stats_by_det, summary, anchor_paths)

    # ----- D4 Ch1 spotlight -----
    if 4 in stats_by_det:
        S4 = stats_by_det[4]
        idx = np.where(S4['channels'] == 1)[0]
        if len(idx):
            i = int(idx[0])
            print("\n=== D4 Ch1 spotlight (canonical 'extreme' channel) ===")
            print(f"  wavelength    = {S4['wl'][i]:.4f} um")
            print(f"  slope_free    = {S4['slope_free'][i]:.6f}")
            print(f"  sigma_slope   = {S4['sigma_slope'][i]:.3e}")
            print(f"  z_slope       = {S4['z_slope'][i]:.3f}")
            print(f"  n_inliers     = {int(S4['n_inliers'][i])}")
            print(f"  F             = {S4['F'][i]:.3e}")
            print(f"  log10(p)      = {S4['log10_p'][i]:.3f}")
            print(f"  delta_BIC     = {S4['delta_BIC'][i]:.3e}")

    # ----- save npz -----
    payload = {'detectors': np.asarray(sorted(stats_by_det),
                                       dtype=np.int32)}
    for tag, p_thr, _label in THRESHOLDS:
        payload[f'threshold_{tag}_p'] = float(p_thr)
    for det, S in stats_by_det.items():
        for k in ('channels', 'wl', 'slope_free', 'intercept_free',
                  'pearson_r', 'n_inliers', 'rss_free', 'rss_locked',
                  'C_locked', 'sigma_slope', 'F', 'log10_p',
                  'delta_BIC', 'z_slope'):
            payload[f'D{det}_{k}'] = S[k]
        clip = clip_per_det[det]
        payload[f'D{det}_clip_window_days'] = float(clip['window_days'])
        payload[f'D{det}_clip_sigma'] = float(clip['sigma'])
        payload[f'D{det}_clip_iters'] = int(clip['iters'])
        payload[f'D{det}_anchor_method'] = str(clip['anchor_method'])
        payload[f'D{det}_anchor_path'] = str(anchor_paths[det])
        for tag, _p_thr, _label in THRESHOLDS:
            pt = summary[det]['per_thresh'][tag]
            payload[f'D{det}_{tag}_n_survivors'] = int(pt['n_survivors'])
            payload[f'D{det}_{tag}_n_total'] = int(pt['n_total'])
            payload[f'D{det}_{tag}_frac'] = float(pt['frac'])
            payload[f'D{det}_{tag}_channels'] = np.asarray(
                pt['channels_sorted'], dtype=np.int32)
            payload[f'D{det}_{tag}_wl'] = np.asarray(
                pt['wl_sorted'], dtype=np.float64)

    np.savez(out_data, **payload)
    print(f"\nSaved data: {out_data}")

    # ----- plot -----
    make_plot(stats_by_det, out_plot)


if __name__ == '__main__':
    main()
