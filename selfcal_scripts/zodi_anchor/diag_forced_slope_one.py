"""Forced-slope=1 per-channel zodi anchor fit diagnostic.

The per-channel anchor (``fit_anchor_for_channel``) lets slope vary freely
in wavelength. A few-percent slope inflation eats ~10-15 mMJy/sr from C
(because ``C = mean(fdc) - slope*mean(zp)``), and the slope distribution
is systematically different on different detectors. That produces visible
discontinuities in C at the D3->D4 and D4->D5 boundaries.

This diagnostic tests the alternative: lock slope to exactly 1.0 per
channel and recompute C from the inlier means. The fit follows
``fit_with_clip`` step-for-step (same moving MJD-window sigma-clip,
window=7d, sigma=3.0, iters=2) but pins ``slope=1.0`` throughout and
skips the inner-loop slope updates, so::

    C_locked = mean(fdc[inlier]) - mean(zp[inlier])
    resid_locked[k] = fdc[k] - zp[k] - C_locked

The free-fit values (slope_free, C_free, resid_std_free) are loaded by
re-running ``fit_with_clip`` in-process so both variants use the same
in-memory (fdc, zp, mjds) and resid_std_free is directly comparable to
resid_std_locked.

Outputs:
  * 4-panel figure: (a) C_free vs C_locked vs lambda for D3/D4/D5
                    (b) resid_std_free vs resid_std_locked vs lambda
                    (c) ratio_chi2 vs lambda
                    (d) zoom on D3-D4 and D4-D5 boundaries: C_free vs C_locked
  * .npz with all per-channel arrays.
  * stdout table per channel.

Example::

    python selfcal_scripts/zodi_anchor/diag_forced_slope_one.py \\
        --run-dir /mnt/md124/.../SPHEREx_NEP_2026W17_D3_6p2arcsec \\
                  /mnt/md124/.../SPHEREx_NEP_2026W17_D4_6p2arcsec \\
                  /mnt/md124/.../SPHEREx_NEP_2026W17_D5_6p2arcsec \\
        --out-plot figures/zodi_anchor/forced_slope_one.png
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Allow direct invocation: selfcal_scripts/zodi_anchor sibling import.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from selfcal.ZodiAnchor import fit_with_clip, moving_sigma_clip_mask
from refit_smooth_slope import load_detector, per_channel_from_anchor


# Detector color map (matches the convention used in the other zodi anchor
# diagnostics).
DET_COLORS = {1: 'tab:purple', 2: 'tab:orange',
              3: 'tab:green', 4: 'tab:blue', 5: 'tab:red'}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--run-dir', nargs='+', required=True,
                   help='SPHEREx run directories (one per detector). '
                        'Each must contain calibration/, zodi_preds/, '
                        'and zodi_anchor/anchor_D*.h5.')
    p.add_argument('--sigma', type=float, default=3.0,
                   help='Sigma-clip threshold (default 3.0, matches '
                        'fit_anchor_for_channel).')
    p.add_argument('--window-days', type=float, default=7.0,
                   help='Moving MJD window for the sigma-clip '
                        '(default 7.0, matches fit_anchor_for_channel).')
    p.add_argument('--n-iter', type=int, default=2,
                   help='Number of moving sigma-clip refit iterations '
                        '(default 2, matches anchor clip_iters=2).')
    p.add_argument('--cal-glob-pat', default='cal_*polyK1.h5',
                   help="Glob inside <run>/calibration "
                        "(default: 'cal_*polyK1.h5').")
    p.add_argument('--out-plot', default=None,
                   help='Output PNG. Default: '
                        'figures/zodi_anchor/forced_slope_one.png')
    p.add_argument('--out-data', default=None,
                   help='Output .npz. Default: derived from --out-plot.')
    return p.parse_args()


# ----------------------------------------------------------------------
# Fits
# ----------------------------------------------------------------------

def fit_locked_slope_one(zp, fs, mjds, window_days, sigma, iters):
    """Per-channel anchor fit with slope LOCKED to 1.0.

    Mirrors ``fit_with_clip`` exactly (same finite-mask init, same moving
    MJD-window sigma-clip, same early-exit conditions) but:
      * starts with slope=1.0 (NOT polyfit)
      * keeps slope=1.0 throughout (skips the inner-loop slope update)
      * recomputes C from inlier means at each refit

    Returns (C, inlier_mask). For slope=1 the OLS-optimal intercept on
    inliers IS mean(fs[inlier]) - mean(zp[inlier]), matching the task spec.
    """
    slope = 1.0
    inlier = np.isfinite(zp) & np.isfinite(fs)
    if mjds is not None:
        inlier &= np.isfinite(mjds)
    n_init = int(inlier.sum())
    if n_init < 2:
        return float('nan'), inlier
    C = float(fs[inlier].mean() - slope * zp[inlier].mean())
    for it in range(int(iters)):
        if mjds is None or window_days <= 0:
            break
        resid = fs - (slope * zp + C)
        keep = moving_sigma_clip_mask(
            mjds, np.where(inlier, resid, np.inf), window_days, sigma)
        new_inlier = inlier & keep
        n_new = int(new_inlier.sum())
        if n_new < 10:
            break
        if n_new == int(inlier.sum()):
            break
        inlier = new_inlier
        C = float(fs[inlier].mean() - slope * zp[inlier].mean())
    return C, inlier


# ----------------------------------------------------------------------
# Boundary diagnostic
# ----------------------------------------------------------------------

def boundary_jump(wl_a, C_a, wl_b, C_b):
    """C jump at the boundary between detector A (long-lambda end) and
    detector B (short-lambda end) -- returns C_b[short] - C_a[long]."""
    finite_a = np.isfinite(wl_a) & np.isfinite(C_a)
    finite_b = np.isfinite(wl_b) & np.isfinite(C_b)
    if not finite_a.any() or not finite_b.any():
        return np.nan, np.nan, np.nan
    wl_a = wl_a[finite_a]
    C_a = C_a[finite_a]
    wl_b = wl_b[finite_b]
    C_b = C_b[finite_b]
    order_a = np.argsort(wl_a)
    order_b = np.argsort(wl_b)
    last_a = order_a[-1]
    first_b = order_b[0]
    return (float(C_b[first_b] - C_a[last_a]),
            float(wl_a[last_a]), float(wl_b[first_b]))


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    out_plot = args.out_plot or os.path.join(
        'figures', 'zodi_anchor', 'forced_slope_one.png')
    out_data = args.out_data or (os.path.splitext(out_plot)[0] + '.npz')
    os.makedirs(os.path.dirname(os.path.abspath(out_plot)) or '.',
                exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_data)) or '.',
                exist_ok=True)

    # ----- Load all detectors -----
    print("Loading detectors ...")
    detectors_data = []
    for run_dir in args.run_dir:
        print(f"  {run_dir}")
        det_data = load_detector(run_dir, args.cal_glob_pat)
        det = det_data['detector']
        n_ch = len(det_data['channels'])
        n_frames_per_ch = [len(f) for f in det_data['FDC']]
        print(f"    D{det}: {n_ch} channels, "
              f"N_frames range = "
              f"[{min(n_frames_per_ch)}, {max(n_frames_per_ch)}]")
        detectors_data.append(det_data)
    detectors_data.sort(key=lambda d: d['detector'])

    # ----- Per-detector results: free + locked -----
    print(f"\n--- Forced-slope=1 vs free per-channel anchor  "
          f"(window={args.window_days}d, sigma={args.sigma}, "
          f"iters={args.n_iter}) ---")
    per_det = {}
    print(f"\n{'Det':>3} {'Ch':>3} {'WL':>7}  "
          f"{'slope_f':>8} {'C_f':>9} {'C_lock':>9} {'dC':>8}  "
          f"{'rs_f':>7} {'rs_lock':>7} {'chi2':>6}")
    print("-" * 90)
    for det_data in detectors_data:
        det = det_data['detector']
        chs = det_data['channels']
        wls = det_data['WL']
        n_ch = len(chs)
        anchor_per_ch = per_channel_from_anchor(det_data['anchor_path'])
        # Build a lookup ch -> (slope, C) from the anchor file. anchor_method
        # is 'raw' for these runs so slope_final == slope, C_final == C.
        anchor_lut = {int(c): (float(s), float(C)) for c, s, C in zip(
            anchor_per_ch['channels'],
            anchor_per_ch['slope'],
            anchor_per_ch['C'])}

        slope_free = np.full(n_ch, np.nan)
        C_free = np.full(n_ch, np.nan)
        resid_std_free = np.full(n_ch, np.nan)
        n_inl_free = np.full(n_ch, 0, dtype=np.int32)
        C_locked = np.full(n_ch, np.nan)
        resid_std_locked = np.full(n_ch, np.nan)
        n_inl_locked = np.full(n_ch, 0, dtype=np.int32)
        var_free = np.full(n_ch, np.nan)
        var_locked = np.full(n_ch, np.nan)

        for i, ch in enumerate(chs):
            fdc = det_data['FDC'][i]
            zp = det_data['ZP'][i]
            mjds = det_data['MJD'][i]
            wl = float(wls[i])
            if fdc.size == 0:
                continue

            # ---- Free fit (re-run fit_with_clip to reproduce the anchor) ----
            s_f, C_f, _r, inl_f = fit_with_clip(
                zp, fdc, mjds,
                window_days=args.window_days, sigma=args.sigma,
                iters=args.n_iter)
            slope_free[i] = s_f
            C_free[i] = C_f
            n_inl_free[i] = int(inl_f.sum())
            if inl_f.sum() >= 2:
                r_f = fdc[inl_f] - (s_f * zp[inl_f] + C_f)
                resid_std_free[i] = float(np.std(r_f))
                var_free[i] = float(np.var(r_f))

            # Sanity check: anchor file slope/C should match within rounding.
            if int(ch) in anchor_lut:
                a_s, a_C = anchor_lut[int(ch)]
                if not (np.isclose(a_s, s_f, atol=1e-6, rtol=1e-6)
                        and np.isclose(a_C, C_f, atol=1e-9, rtol=1e-6)):
                    print(f"    [warn] D{det} Ch{ch}: anchor (slope, C) = "
                          f"({a_s:.6f}, {a_C:.6e}) differs from re-fit "
                          f"({s_f:.6f}, {C_f:.6e})", file=sys.stderr)

            # ---- Locked fit (slope = 1) ----
            C_l, inl_l = fit_locked_slope_one(
                zp, fdc, mjds,
                window_days=args.window_days, sigma=args.sigma,
                iters=args.n_iter)
            C_locked[i] = C_l
            n_inl_locked[i] = int(inl_l.sum())
            if inl_l.sum() >= 2 and np.isfinite(C_l):
                r_l = fdc[inl_l] - (1.0 * zp[inl_l] + C_l)
                resid_std_locked[i] = float(np.std(r_l))
                var_locked[i] = float(np.var(r_l))

            # Stdout row (units: WL um, C mMJy/sr, resid_std mMJy/sr)
            dC_mMJy = ((C_l - C_f) * 1e3
                       if (np.isfinite(C_l) and np.isfinite(C_f))
                       else float('nan'))
            ratio = (var_locked[i] / var_free[i]
                     if (np.isfinite(var_locked[i]) and np.isfinite(var_free[i])
                         and var_free[i] > 0)
                     else float('nan'))
            print(f"{det:>3} {ch:>3} {wl:>7.3f}  "
                  f"{s_f:>8.4f} {C_f * 1e3:>+9.2f} {C_l * 1e3:>+9.2f} "
                  f"{dC_mMJy:>+8.2f}  "
                  f"{resid_std_free[i] * 1e3:>7.2f} "
                  f"{resid_std_locked[i] * 1e3:>7.2f} "
                  f"{ratio:>6.3f}")

        delta_C = C_locked - C_free
        delta_resid_std = resid_std_locked - resid_std_free  # MJy/sr
        ratio_chi2 = np.where(
            (var_free > 0) & np.isfinite(var_locked) & np.isfinite(var_free),
            var_locked / np.where(var_free > 0, var_free, np.nan),
            np.nan)
        per_det[det] = dict(
            channels=chs, wl=wls,
            slope_free=slope_free,
            C_free=C_free, C_locked=C_locked, delta_C=delta_C,
            resid_std_free=resid_std_free,
            resid_std_locked=resid_std_locked,
            delta_resid_std=delta_resid_std,
            var_free=var_free, var_locked=var_locked,
            ratio_chi2=ratio_chi2,
            n_inliers_free=n_inl_free, n_inliers_locked=n_inl_locked,
        )

    # ----- Boundary jumps in C (free vs locked) -----
    det_ids = sorted(per_det)
    print("\n--- C(lambda) boundary jumps "
          "(C_short[next det] - C_long[prev det], mMJy/sr) ---")
    boundary_jumps = {'free': [], 'locked': []}
    for i in range(len(det_ids) - 1):
        dA = det_ids[i]
        dB = det_ids[i + 1]
        wlA = per_det[dA]['wl']
        wlB = per_det[dB]['wl']
        for label, key in (('free', 'C_free'), ('locked', 'C_locked')):
            j, lamA, lamB = boundary_jump(
                wlA, per_det[dA][key], wlB, per_det[dB][key])
            print(f"  D{dA} (lam={lamA:.3f}) -> D{dB} (lam={lamB:.3f})  "
                  f"{label:>6}: DeltaC = {j * 1e3:+.2f} mMJy/sr")
            boundary_jumps[label].append(dict(
                dA=dA, dB=dB, lamA=lamA, lamB=lamB, dC_mMJy=j * 1e3))

    # ----- Per-detector median ratio_chi2 + flagged channels -----
    print("\n--- Per-detector median ratio_chi2 "
          "(1.0 = slope freedom didn't matter; >1.1 = genuinely improved) ---")
    med_ratio = {}
    for d in det_ids:
        r = per_det[d]['ratio_chi2']
        m = float(np.nanmedian(r))
        med_ratio[d] = m
        print(f"  D{d}: median ratio_chi2 = {m:.4f}  "
              f"(range [{np.nanmin(r):.3f}, {np.nanmax(r):.3f}], "
              f"n_finite={int(np.isfinite(r).sum())}/{len(r)})")

    print("\n--- Channels where ratio_chi2 > 1.2 "
          "(slope freedom strongly justified by data) ---")
    strong = []
    for d in det_ids:
        r = per_det[d]['ratio_chi2']
        chs_d = per_det[d]['channels']
        wls_d = per_det[d]['wl']
        slope_d = per_det[d]['slope_free']
        for i, ch in enumerate(chs_d):
            if np.isfinite(r[i]) and r[i] > 1.2:
                strong.append(dict(det=d, ch=int(ch), wl=float(wls_d[i]),
                                   slope_free=float(slope_d[i]),
                                   ratio_chi2=float(r[i])))
                print(f"  D{d} Ch{ch:>2}  lam={wls_d[i]:.3f}  "
                      f"slope_free={slope_d[i]:.4f}  "
                      f"ratio_chi2={r[i]:.3f}")
    if not strong:
        print("  (none)")

    # ----- Save .npz -----
    npz_payload = {
        'detectors': np.asarray(det_ids, dtype=np.int32),
        'window_days': np.float64(args.window_days),
        'sigma': np.float64(args.sigma),
        'n_iter': np.int32(args.n_iter),
    }
    for d in det_ids:
        for k in ('channels', 'wl',
                  'slope_free', 'C_free', 'C_locked', 'delta_C',
                  'resid_std_free', 'resid_std_locked', 'delta_resid_std',
                  'var_free', 'var_locked', 'ratio_chi2',
                  'n_inliers_free', 'n_inliers_locked'):
            npz_payload[f'{k}_D{d}'] = per_det[d][k]
    for label in ('free', 'locked'):
        jl = boundary_jumps[label]
        npz_payload[f'boundary_dC_mMJy_{label}'] = np.array(
            [j['dC_mMJy'] for j in jl], dtype=np.float64)
        npz_payload[f'boundary_dA_{label}'] = np.array(
            [j['dA'] for j in jl], dtype=np.int32)
        npz_payload[f'boundary_dB_{label}'] = np.array(
            [j['dB'] for j in jl], dtype=np.int32)
    np.savez(out_data, **npz_payload)
    print(f"\nSaved data: {out_data}")

    # ----- Plot -----
    # Boundary count drives the zoom row width: 1 column per boundary,
    # min 2 cols so panels (b) and (c) keep half-width each.
    n_boundaries = max(0, len(det_ids) - 1)
    n_zoom_cols = max(2, n_boundaries)
    fig_w = max(13.0, 4.0 * n_zoom_cols + 1.0)
    fig = plt.figure(figsize=(fig_w, 11))
    gs = fig.add_gridspec(3, n_zoom_cols, height_ratios=[1.05, 1.05, 1.0],
                          hspace=0.36, wspace=0.28)
    ax_C = fig.add_subplot(gs[0, :])           # (a)
    # split row 1 into 2 panels regardless of zoom count
    half = n_zoom_cols // 2
    ax_rs = fig.add_subplot(gs[1, :half])      # (b)
    ax_chi2 = fig.add_subplot(gs[1, half:])    # (c)
    zoom_axes = [fig.add_subplot(gs[2, k]) for k in range(n_zoom_cols)]

    # (a) C_free (open) vs C_locked (filled) vs lambda
    for d in det_ids:
        col = DET_COLORS.get(d, 'tab:purple')
        wl = per_det[d]['wl']
        order = np.argsort(wl)
        wl_o = wl[order]
        ax_C.plot(wl_o, per_det[d]['C_free'][order] * 1e3,
                  color=col, marker='o', mfc='none', ls='-', ms=5, lw=0.9,
                  label=f'D{d} C_free')
        ax_C.plot(wl_o, per_det[d]['C_locked'][order] * 1e3,
                  color=col, marker='o', mfc=col, ls='--', ms=5, lw=0.9,
                  alpha=0.85, label=f'D{d} C_locked (slope=1)')
    ax_C.axhline(0.0, color='gray', lw=0.5, alpha=0.5)
    ax_C.set_xlabel('lambda  (um)')
    ax_C.set_ylabel('C  (mMJy/sr)')
    ax_C.set_title('(a) anchor constant C(lambda): free slope (open) '
                   'vs locked slope=1 (filled)')
    ax_C.grid(alpha=0.3)
    ax_C.legend(loc='best', fontsize=7, ncol=3)

    # (b) resid_std vs lambda (free vs locked)
    for d in det_ids:
        col = DET_COLORS.get(d, 'tab:purple')
        wl = per_det[d]['wl']
        order = np.argsort(wl)
        wl_o = wl[order]
        ax_rs.plot(wl_o, per_det[d]['resid_std_free'][order] * 1e3,
                   color=col, marker='o', mfc='none', ls='-', ms=4, lw=0.9,
                   label=f'D{d} free')
        ax_rs.plot(wl_o, per_det[d]['resid_std_locked'][order] * 1e3,
                   color=col, marker='o', mfc=col, ls='--', ms=4, lw=0.9,
                   alpha=0.85, label=f'D{d} locked')
    ax_rs.set_xlabel('lambda  (um)')
    ax_rs.set_ylabel('resid_std  (mMJy/sr)')
    ax_rs.set_title('(b) per-frame residual std: free vs locked slope=1')
    ax_rs.grid(alpha=0.3)
    ax_rs.legend(loc='best', fontsize=7, ncol=2)

    # (c) ratio_chi2 vs lambda
    for d in det_ids:
        col = DET_COLORS.get(d, 'tab:purple')
        wl = per_det[d]['wl']
        order = np.argsort(wl)
        ax_chi2.plot(wl[order], per_det[d]['ratio_chi2'][order],
                     color=col, marker='o', mfc=col, ls='-', ms=4, lw=0.9,
                     label=f'D{d} (median = {med_ratio[d]:.3f})')
    ax_chi2.axhline(1.0, color='gray', lw=0.6, ls='--', alpha=0.7,
                    label='ratio = 1')
    ax_chi2.axhline(1.2, color='red', lw=0.5, ls=':', alpha=0.5,
                    label='ratio = 1.2')
    ax_chi2.set_xlabel('lambda  (um)')
    ax_chi2.set_ylabel('var(resid_locked) / var(resid_free)')
    ax_chi2.set_title('(c) ratio_chi2(lambda): >1 -> slope freedom helped '
                      'the fit')
    ax_chi2.grid(alpha=0.3)
    ax_chi2.legend(loc='best', fontsize=7, ncol=2)

    # (d) Zoom on boundaries: C_free vs C_locked
    boundaries_lam = []
    for i in range(len(det_ids) - 1):
        dA = det_ids[i]
        dB = det_ids[i + 1]
        wlA = per_det[dA]['wl']
        wlB = per_det[dB]['wl']
        finA = np.isfinite(wlA)
        finB = np.isfinite(wlB)
        if not finA.any() or not finB.any():
            continue
        lam_mid = 0.5 * (np.nanmax(wlA[finA]) + np.nanmin(wlB[finB]))
        boundaries_lam.append((dA, dB, lam_mid))

    # Hide any leftover zoom axes if we have more cols than boundaries
    for k in range(len(boundaries_lam), len(zoom_axes)):
        zoom_axes[k].set_visible(False)
    for ax, (dA, dB, lam_mid) in zip(zoom_axes, boundaries_lam):
        lam_lo = lam_mid - 0.30
        lam_hi = lam_mid + 0.30
        for d in (dA, dB):
            col = DET_COLORS.get(d, 'tab:purple')
            wl = per_det[d]['wl']
            in_win = (wl >= lam_lo) & (wl <= lam_hi) & np.isfinite(wl)
            if not in_win.any():
                continue
            wl_in = wl[in_win]
            order = np.argsort(wl_in)
            wl_o = wl_in[order]
            ax.plot(wl_o, per_det[d]['C_free'][in_win][order] * 1e3,
                    color=col, marker='o', mfc='none', ls='-', ms=6, lw=0.9,
                    label=f'D{d} C_free')
            ax.plot(wl_o, per_det[d]['C_locked'][in_win][order] * 1e3,
                    color=col, marker='o', mfc=col, ls='--', ms=6, lw=0.9,
                    alpha=0.85, label=f'D{d} C_locked')
        ax.axvline(lam_mid, color='gray', lw=0.6, ls=':')
        # Print boundary jumps onto the axis
        txt_lines = []
        for label in ('free', 'locked'):
            j_list = boundary_jumps[label]
            entry = next((j for j in j_list if j['dA'] == dA and j['dB'] == dB),
                         None)
            if entry is None:
                continue
            txt_lines.append(
                f"{label:>6}: DeltaC = {entry['dC_mMJy']:+.2f} mMJy/sr")
        if txt_lines:
            ax.text(0.02, 0.98, '\n'.join(txt_lines),
                    transform=ax.transAxes, va='top', ha='left',
                    fontsize=8, family='monospace',
                    bbox=dict(facecolor='white', alpha=0.85, lw=0.4))
        ax.set_xlabel('lambda  (um)')
        ax.set_ylabel('C  (mMJy/sr)')
        ax.set_title(f'(d) zoom D{dA} <-> D{dB} boundary')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=7, ncol=2)

    fig.suptitle(
        f'Forced-slope=1 per-channel anchor refit  '
        f'(window = {args.window_days}d, sigma = {args.sigma}, '
        f'iters = {args.n_iter})',
        y=0.995, fontsize=11)
    plt.savefig(out_plot, dpi=130, bbox_inches='tight')
    print(f"Saved plot: {out_plot}")


if __name__ == '__main__':
    main()
