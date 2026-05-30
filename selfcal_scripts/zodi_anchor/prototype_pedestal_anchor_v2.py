"""Prototype pedestal-subtraction anchor v2 — preserves per-channel features.

The v1 model

    full_DC[D, k, c] = (1 + amp_D) * zodi_pred + C_smooth(lambda) + P_D

uses a smooth cubic ``C_smooth(lambda)`` that wipes out genuine
per-channel spectral features (PAH 3.29 um, CO2 ice 4.27 um, ...).

v2 keeps the per-channel ``C_old`` from the per-channel anchor — which
already encodes those features — and only solves for a per-detector
additive pedestal ``P_D`` by demanding boundary continuity of the
*smooth-continuum* channels (those with low joint-fit residual std).

    C_corr[D, c] = C_old[D, c] - P_D            (all channels)

Smooth-continuum mask per detector:

  smooth = (resid_std < smooth_thresh_factor * median(resid_std))
           AND (c not in {first, last})        # drop dichroic-edge spikes

P_D solve:

  * Gauge fix P_D3 = 0.
  * For each adjacent boundary (D_lo -> D_hi at lambda_b):
    - Weighted linfit ``(lambda, C_old[D_lo] - P_D_lo)`` over smooth
      channels in [lambda_b - 0.15, lambda_b - 0.005], weights = 1/sigma^2.
    - Extrapolate -> y_lo.
    - Weighted linfit ``(lambda, C_old[D_hi])`` over smooth channels in
      [lambda_b + 0.005, lambda_b + 0.15], weights = 1/sigma^2.
    - Extrapolate -> y_hi_unshifted.
    - P_D_hi = y_hi_unshifted - y_lo.

Reads ``/tmp/joint_resid_data.npz`` (produced by
``diag_joint_amp_fit.py``); writes a 3-panel diagnostic figure +
``/tmp/prototype_pedestal_v2_data.npz`` with the corrected C and the
solved per-detector pedestals.

Example::

    python prototype_pedestal_anchor_v2.py \\
        --joint-data /tmp/joint_resid_data.npz \\
        --out figures/zodi_anchor/prototype_pedestal_anchor_v2.png
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


# --- conventions shared with v1 -------------------------------------
DET_COLORS = {3: 'tab:green', 4: 'tab:blue', 5: 'tab:red'}
DET_BOUNDARIES_UM = (2.42, 3.81)  # D3|D4, D4|D5
# Astrophysical / instrumental feature wavelengths (um).
FEATURE_LINES = [
    (2.058, 'He I 2.058'),
    (2.166, r'Br$\gamma$ 2.166'),
    (2.42, 'dichroic 2.42'),
    (2.625, r'Br$\beta$ 2.625'),
    (3.29, 'PAH 3.29'),
    (3.40, 'PAH 3.40'),
    (4.052, r'Br$\alpha$ 4.052'),
    (4.27, 'CO$_2$ ice 4.27'),
]
# Features to verify are preserved in the corrected C (label, wl).
PRESERVED_FEATURES = [
    ('PAH_3_29', 3.29),
    ('CO2_ice_4_27', 4.27),
    ('Br_alpha_4_052', 4.052),
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--joint-data', default='/tmp/joint_resid_data.npz',
                   help='Path to the joint-amp resid-data npz from '
                        'diag_joint_amp_fit.py '
                        '(default: /tmp/joint_resid_data.npz). Must contain '
                        'WL_D{D}, C_old_D{D}, resid_std_D{D}, slope_old_D{D}, '
                        'amp_D{D} for each detector D in {3,4,5}.')
    p.add_argument('--detectors', nargs='+', type=int, default=[3, 4, 5],
                   help='Detector ids to process (default: 3 4 5).')
    p.add_argument('--reference-detector', type=int, default=3,
                   help='Detector whose P_D is fixed to 0 as gauge '
                        '(default: 3).')
    p.add_argument('--smooth-thresh-factor', type=float, default=2.0,
                   help='A channel is "smooth-continuum" if '
                        'resid_std < factor * median(resid_std) (default 2.0).')
    p.add_argument('--boundary-window-um', type=float, default=0.15,
                   help='Half-width (um) of the boundary fit window '
                        '(default 0.15).')
    p.add_argument('--boundary-guard-um', type=float, default=0.005,
                   help='Exclude channels within this wavelength of the '
                        'boundary itself (default 0.005).')
    p.add_argument('--out', default='figures/zodi_anchor/prototype_pedestal_anchor_v2.png',
                   help='Output PNG path '
                        '(default: figures/zodi_anchor/prototype_pedestal_anchor_v2.png).')
    p.add_argument('--out-data', default='/tmp/prototype_pedestal_v2_data.npz',
                   help='Output npz path '
                        '(default: /tmp/prototype_pedestal_v2_data.npz).')
    return p.parse_args()


# ------------------------------------------------------------------
# core fit
# ------------------------------------------------------------------
def load_inputs(path, dets):
    """Return per-det dicts of {WL, C_old, resid_std, slope_old, amp}."""
    if not os.path.exists(path):
        raise SystemExit(f"--joint-data not found: {path}")
    with np.load(path, allow_pickle=False) as z:
        files = set(z.files)
        out = {}
        for d in dets:
            needed = [f'WL_D{d}', f'C_old_D{d}', f'resid_std_D{d}',
                      f'slope_old_D{d}', f'amp_D{d}']
            for k in needed:
                if k not in files:
                    raise SystemExit(f"missing key '{k}' in {path}; "
                                     f"have: {sorted(files)}")
            out[d] = dict(
                WL=np.asarray(z[f'WL_D{d}'], dtype=np.float64),
                C_old=np.asarray(z[f'C_old_D{d}'], dtype=np.float64),
                resid_std=np.asarray(z[f'resid_std_D{d}'], dtype=np.float64),
                slope_old=np.asarray(z[f'slope_old_D{d}'], dtype=np.float64),
                amp=float(np.asarray(z[f'amp_D{d}'])),
            )
    return out


def smooth_channel_mask(resid_std, factor):
    """Smooth-continuum mask: low resid_std AND not the very edge channels."""
    rs = np.asarray(resid_std, dtype=np.float64)
    finite = np.isfinite(rs)
    if not finite.any():
        return np.zeros_like(rs, dtype=bool)
    med = float(np.nanmedian(rs))
    mask = finite & (rs < factor * med)
    # Drop very-edge channels (dichroic spikes).
    if mask.size >= 1:
        mask[0] = False
        mask[-1] = False
    return mask


def weighted_linfit(x, y, w):
    """Weighted linear fit y = m*x + b. Returns (m, b)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if x.size < 2:
        raise ValueError(f"need >=2 points for linfit, got {x.size}")
    W = np.sum(w)
    Wx = np.sum(w * x)
    Wy = np.sum(w * y)
    Wxx = np.sum(w * x * x)
    Wxy = np.sum(w * x * y)
    det = W * Wxx - Wx * Wx
    if det == 0:
        raise ValueError("singular weighted linfit (det=0)")
    m = (W * Wxy - Wx * Wy) / det
    b = (Wxx * Wy - Wx * Wxy) / det
    return float(m), float(b)


def select_boundary_window(WL, smooth_mask, lam_b, window, guard, side):
    """Return indices of smooth-channels within the boundary window.

    side='low' -> [lam_b - window, lam_b - guard]
    side='hi'  -> [lam_b + guard, lam_b + window]
    """
    if side == 'low':
        lo = lam_b - window
        hi = lam_b - guard
    elif side == 'hi':
        lo = lam_b + guard
        hi = lam_b + window
    else:
        raise ValueError(f"side must be 'low' or 'hi', got {side!r}")
    return np.flatnonzero(smooth_mask & (WL >= lo) & (WL <= hi))


def solve_pedestals(per_det, det_order, ref_det, smooth_masks,
                    window, guard):
    """Boundary-continuity pedestal solver.

    Returns dict P[D] and a list of per-boundary diagnostic dicts.
    """
    P = {ref_det: 0.0}
    diagnostics = []
    # We pin P at ref_det and propagate outward. The spec orders boundaries
    # D3->D4, D4->D5, so we walk det_order pairwise; this assumes ref_det is
    # the lowest detector (or that the user runs only adjacent dets).
    for d_lo, d_hi in zip(det_order[:-1], det_order[1:]):
        if d_lo not in P:
            raise SystemExit(f"P_D{d_lo} not yet pinned (ref_det must be the "
                             f"lowest detector in det_order; got det_order="
                             f"{det_order}, ref={ref_det})")
        # Detector boundary wavelength = closer-to of WL_lo[max] and WL_hi[min]
        wl_lo = per_det[d_lo]['WL']
        wl_hi = per_det[d_hi]['WL']
        lam_b = 0.5 * (np.nanmax(wl_lo) + np.nanmin(wl_hi))

        sm_lo = smooth_masks[d_lo]
        sm_hi = smooth_masks[d_hi]
        idx_lo = select_boundary_window(wl_lo, sm_lo, lam_b,
                                        window, guard, 'low')
        idx_hi = select_boundary_window(wl_hi, sm_hi, lam_b,
                                        window, guard, 'hi')
        if idx_lo.size < 2:
            raise SystemExit(
                f"D{d_lo}->D{d_hi}: not enough smooth channels on D{d_lo} "
                f"side of lam_b={lam_b:.3f} um (got {idx_lo.size}); "
                f"loosen --smooth-thresh-factor or --boundary-window-um")
        if idx_hi.size < 2:
            raise SystemExit(
                f"D{d_lo}->D{d_hi}: not enough smooth channels on D{d_hi} "
                f"side of lam_b={lam_b:.3f} um (got {idx_hi.size}); "
                f"loosen --smooth-thresh-factor or --boundary-window-um")

        x_lo = wl_lo[idx_lo]
        y_lo_raw = per_det[d_lo]['C_old'][idx_lo] - P[d_lo]
        w_lo = 1.0 / (per_det[d_lo]['resid_std'][idx_lo] ** 2)
        m_lo, b_lo = weighted_linfit(x_lo, y_lo_raw, w_lo)
        y_lo_extrap = m_lo * lam_b + b_lo

        x_hi = wl_hi[idx_hi]
        y_hi_raw = per_det[d_hi]['C_old'][idx_hi]
        w_hi = 1.0 / (per_det[d_hi]['resid_std'][idx_hi] ** 2)
        m_hi, b_hi = weighted_linfit(x_hi, y_hi_raw, w_hi)
        y_hi_unshifted = m_hi * lam_b + b_hi

        P[d_hi] = y_hi_unshifted - y_lo_extrap

        diagnostics.append(dict(
            d_lo=d_lo, d_hi=d_hi, lam_b=lam_b,
            idx_lo=idx_lo, idx_hi=idx_hi,
            m_lo=m_lo, b_lo=b_lo,
            m_hi=m_hi, b_hi=b_hi,
            y_lo_extrap=y_lo_extrap,
            y_hi_unshifted=y_hi_unshifted,
            y_hi_shifted=y_hi_unshifted - P[d_hi],
            P_lo=P[d_lo], P_hi=P[d_hi],
        ))
    return P, diagnostics


def boundary_jump_old(per_det, d_lo, d_hi):
    """C_old jump = C_old[D_hi, ch=1] - C_old[D_lo, ch=34]."""
    wl_lo = per_det[d_lo]['WL']
    wl_hi = per_det[d_hi]['WL']
    i_lo = int(np.nanargmax(wl_lo))
    i_hi = int(np.nanargmin(wl_hi))
    return float(per_det[d_hi]['C_old'][i_hi] - per_det[d_lo]['C_old'][i_lo])


def boundary_jump_new(per_det, d_lo, d_hi, P):
    """C_corr jump = (C_old[D_hi, ch=1] - P_hi) - (C_old[D_lo, ch=34] - P_lo)."""
    wl_lo = per_det[d_lo]['WL']
    wl_hi = per_det[d_hi]['WL']
    i_lo = int(np.nanargmax(wl_lo))
    i_hi = int(np.nanargmin(wl_hi))
    c_lo = per_det[d_lo]['C_old'][i_lo] - P[d_lo]
    c_hi = per_det[d_hi]['C_old'][i_hi] - P[d_hi]
    return float(c_hi - c_lo)


# ------------------------------------------------------------------
# feature preservation check
# ------------------------------------------------------------------
def find_detector_for_wl(per_det, det_order, wl_target):
    for d in det_order:
        WL = per_det[d]['WL']
        if np.nanmin(WL) <= wl_target <= np.nanmax(WL):
            return d
    return None


def nearest_channel(WL, wl_target):
    return int(np.nanargmin(np.abs(WL - wl_target)))


def is_local_enhancement(C_corr, idx, half=2):
    """Local enhancement: C_corr[idx] > median of neighboring channels.

    Neighbors: idx +/- 1..half, excluding idx itself; trimmed to array range.
    """
    n = C_corr.size
    nbr = [j for j in range(max(0, idx - half), min(n, idx + half + 1))
           if j != idx]
    if not nbr:
        return False
    return bool(C_corr[idx] > np.nanmedian(C_corr[nbr]))


def check_features_preserved(per_det, det_order, C_corr_per_det):
    """Return dict {feature_label: bool}. True iff still a local enhancement."""
    out = {}
    notes = []
    for label, wl in PRESERVED_FEATURES:
        d = find_detector_for_wl(per_det, det_order, wl)
        if d is None:
            out[label] = False
            notes.append(f"  {label} ({wl:.3f} um): no detector covers")
            continue
        WL = per_det[d]['WL']
        C_corr = C_corr_per_det[d]
        idx = nearest_channel(WL, wl)
        ok = is_local_enhancement(C_corr, idx, half=2)
        out[label] = ok
        notes.append(
            f"  {label} ({wl:.3f} um) -> D{d} ch idx {idx} "
            f"(WL={WL[idx]:.3f}): C_corr={C_corr[idx]:+.4g}, "
            f"enhancement={ok}")
    return out, notes


# ------------------------------------------------------------------
# plotting
# ------------------------------------------------------------------
def annotate_feature_lines(ax, label_top=False, ymax_frac=0.98):
    ymin, ymax = ax.get_ylim()
    for wl, label in FEATURE_LINES:
        ax.axvline(wl, color='magenta', lw=0.5, ls=':', alpha=0.55)
        if label_top:
            ax.text(wl, ymin + (ymax - ymin) * ymax_frac, label,
                    rotation=90, va='top', ha='right',
                    fontsize=6, color='magenta', alpha=0.85)


def plot_panel_a(ax, per_det, det_order, C_corr_per_det, smooth_masks, P):
    for d in det_order:
        color = DET_COLORS.get(d, 'tab:gray')
        WL = per_det[d]['WL']
        C_old = per_det[d]['C_old']
        C_corr = C_corr_per_det[d]
        sm = smooth_masks[d]
        order = np.argsort(WL)
        # C_old: open circles (per-channel anchor input)
        ax.plot(WL[order], C_old[order], 'o', mfc='none', mec=color,
                ms=5, mew=1.0, alpha=0.75,
                label=f'D{d} C_old (per-channel anchor)')
        # C_corr line through all channels.
        ax.plot(WL[order], C_corr[order], '-', color=color, lw=1.0, alpha=0.9)
        # smooth channels: filled markers; non-smooth: open thin markers.
        sm_o = sm[order]
        ax.plot(WL[order][sm_o], C_corr[order][sm_o], 's',
                color=color, ms=5, mew=0, alpha=0.95,
                label=f'D{d} C_corr (smooth ch, P_D={P[d]:+.4g})')
        ax.plot(WL[order][~sm_o], C_corr[order][~sm_o], 's',
                mfc='none', mec=color, ms=5, mew=0.8, alpha=0.9,
                label=f'D{d} C_corr (feature/edge ch)')
    ax.axhline(0.0, color='k', lw=0.5, alpha=0.4)
    for bx in DET_BOUNDARIES_UM:
        ax.axvline(bx, color='k', lw=0.7, ls='--', alpha=0.6)
    ax.set_ylabel(r'C  (MJy/sr)')
    ax.set_title('(a) $C_{\\rm corr}$ (per-channel, pedestal corrected) '
                 '— features preserved')
    annotate_feature_lines(ax, label_top=True)
    ax.legend(loc='best', fontsize=7, ncol=2)
    ax.grid(alpha=0.3)


def plot_panel_b(ax, per_det, det_order, smooth_masks):
    for d in det_order:
        color = DET_COLORS.get(d, 'tab:gray')
        WL = per_det[d]['WL']
        rs = per_det[d]['resid_std']
        sm = smooth_masks[d]
        order = np.argsort(WL)
        ax.plot(WL[order], rs[order], '-', color=color, lw=1.0, alpha=0.7)
        # Smooth channels: filled; non-smooth: open.
        sm_o = sm[order]
        ax.plot(WL[order][sm_o], rs[order][sm_o], 'o',
                color=color, ms=4, alpha=0.95,
                label=f'D{d} smooth (n={int(sm.sum())})')
        ax.plot(WL[order][~sm_o], rs[order][~sm_o], 'x',
                color=color, ms=6, mew=1.0, alpha=0.85,
                label=f'D{d} feature/edge')
    for bx in DET_BOUNDARIES_UM:
        ax.axvline(bx, color='k', lw=0.7, ls='--', alpha=0.6)
    ax.set_ylabel('joint-fit resid_std  (MJy/sr)')
    ax.set_title('(b) Per-channel joint-fit residual std '
                 '— high = spectral feature / edge')
    annotate_feature_lines(ax, label_top=False)
    ax.legend(loc='best', fontsize=7, ncol=3)
    ax.grid(alpha=0.3)


def plot_panel_c(fig, gs_row, per_det, det_order, C_corr_per_det,
                 diagnostics, P, window):
    """Two side-by-side subpanels of the two boundaries."""
    n_b = len(diagnostics)
    sub_gs = gs_row.subgridspec(1, max(n_b, 1), wspace=0.25)
    axes = []
    for j, diag in enumerate(diagnostics):
        ax = fig.add_subplot(sub_gs[0, j])
        axes.append(ax)
        d_lo = diag['d_lo']
        d_hi = diag['d_hi']
        lam_b = diag['lam_b']
        c_lo = DET_COLORS.get(d_lo, 'tab:gray')
        c_hi = DET_COLORS.get(d_hi, 'tab:gray')

        WL_lo = per_det[d_lo]['WL']
        WL_hi = per_det[d_hi]['WL']
        C_corr_lo = C_corr_per_det[d_lo]
        C_corr_hi = C_corr_per_det[d_hi]
        # Plot all C_corr points in the zoom window.
        sel_lo = (WL_lo >= lam_b - window - 0.05) & (WL_lo <= lam_b)
        sel_hi = (WL_hi >= lam_b) & (WL_hi <= lam_b + window + 0.05)
        ax.plot(WL_lo[sel_lo], C_corr_lo[sel_lo], 'o',
                color=c_lo, ms=5, alpha=0.85,
                label=f'D{d_lo} C_corr')
        ax.plot(WL_hi[sel_hi], C_corr_hi[sel_hi], 's',
                color=c_hi, ms=5, alpha=0.85,
                label=f'D{d_hi} C_corr')
        # Highlight smooth channels used in the boundary fit.
        ax.plot(WL_lo[diag['idx_lo']],
                C_corr_lo[diag['idx_lo']], 'o',
                color=c_lo, ms=8, mfc='none', mew=1.4,
                label=f'D{d_lo} smooth (fit)')
        ax.plot(WL_hi[diag['idx_hi']],
                C_corr_hi[diag['idx_hi']], 's',
                color=c_hi, ms=8, mfc='none', mew=1.4,
                label=f'D{d_hi} smooth (fit)')
        # Extrapolation lines (shifted into C_corr space).
        x_line_lo = np.linspace(lam_b - window, lam_b, 50)
        y_line_lo = (diag['m_lo'] * x_line_lo + diag['b_lo'])
        # already in (C_old - P_lo) space = C_corr space for D_lo.
        ax.plot(x_line_lo, y_line_lo, '--', color=c_lo, lw=1.1, alpha=0.9)
        x_line_hi = np.linspace(lam_b, lam_b + window, 50)
        y_line_hi = (diag['m_hi'] * x_line_hi + diag['b_hi']) - P[d_hi]
        # subtract P_hi to put it into C_corr space.
        ax.plot(x_line_hi, y_line_hi, '--', color=c_hi, lw=1.1, alpha=0.9)
        # Boundary line.
        ax.axvline(lam_b, color='k', ls='--', lw=0.8, alpha=0.6)
        # Annotate extrapolation values.
        y_lo_e = diag['y_lo_extrap']
        y_hi_e_corr = diag['y_hi_unshifted'] - P[d_hi]  # same as y_lo by construction
        ax.plot([lam_b], [y_lo_e], '^', color='k', ms=8,
                label=f'extrap = {y_lo_e:+.4g}')
        ax.set_title(f'(c{j+1}) D{d_lo}->D{d_hi} at {lam_b:.3f} um   '
                     f'P_D{d_hi} = {P[d_hi]:+.5g} MJy/sr',
                     fontsize=10)
        ax.set_xlabel(r'wavelength ($\mu$m)')
        if j == 0:
            ax.set_ylabel(r'$C_{\rm corr}$  (MJy/sr)')
        ax.set_xlim(lam_b - window - 0.03, lam_b + window + 0.03)
        ax.legend(loc='best', fontsize=7)
        ax.grid(alpha=0.3)
    return axes


# ------------------------------------------------------------------
# main
# ------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_data)) or '.',
                exist_ok=True)

    det_order = sorted(args.detectors)
    if args.reference_detector not in det_order:
        raise SystemExit(f"--reference-detector {args.reference_detector} "
                         f"not in --detectors {det_order}")
    if det_order[0] != args.reference_detector:
        # We propagate P outward starting from the ref; require ref = lowest.
        raise SystemExit(
            f"--reference-detector must be the lowest detector "
            f"(spec walks boundaries low->high); got ref={args.reference_detector}, "
            f"detectors={det_order}")

    print(f"=== prototype_pedestal_anchor_v2 ===")
    print(f"  joint-data            : {args.joint_data}")
    print(f"  detectors             : {det_order}")
    print(f"  reference detector    : D{args.reference_detector}")
    print(f"  smooth-thresh-factor  : {args.smooth_thresh_factor}")
    print(f"  boundary window (um)  : +/- {args.boundary_window_um}")
    print(f"  boundary guard (um)   : {args.boundary_guard_um}")

    per_det = load_inputs(args.joint_data, det_order)

    # Step 1: smooth-channel masks.
    print("\n--- Step 1: smooth-channel selection per detector ---")
    smooth_masks = {}
    for d in det_order:
        rs = per_det[d]['resid_std']
        med = float(np.nanmedian(rs))
        sm = smooth_channel_mask(rs, args.smooth_thresh_factor)
        smooth_masks[d] = sm
        n_smooth = int(sm.sum())
        n_total = int(np.isfinite(rs).sum())
        kept_idx = np.flatnonzero(sm)
        WL = per_det[d]['WL']
        print(f"  D{d}: median(resid_std) = {med:.4g}  "
              f"threshold = {args.smooth_thresh_factor*med:.4g} MJy/sr   "
              f"n_smooth = {n_smooth}/{n_total} channels")
        kept_ch = [f"ch{j+1}(WL={WL[j]:.3f})" for j in kept_idx]
        # Print compactly: just the channel indices to keep output short.
        print(f"    smooth channel indices (1-based): "
              f"{[int(j+1) for j in kept_idx]}")

    # Step 2: solve pedestals from boundary continuity.
    print("\n--- Step 2: boundary-continuity pedestal solve ---")
    P, diagnostics = solve_pedestals(
        per_det, det_order, args.reference_detector,
        smooth_masks,
        window=args.boundary_window_um,
        guard=args.boundary_guard_um,
    )
    for diag in diagnostics:
        print(f"  boundary D{diag['d_lo']}->D{diag['d_hi']} at "
              f"lam_b = {diag['lam_b']:.4f} um:")
        print(f"    D{diag['d_lo']} side fit: n = {diag['idx_lo'].size}, "
              f"slope = {diag['m_lo']:+.4g}, "
              f"y(lam_b) = {diag['y_lo_extrap']:+.5g} MJy/sr  [in C_corr space]")
        print(f"    D{diag['d_hi']} side fit: n = {diag['idx_hi'].size}, "
              f"slope = {diag['m_hi']:+.4g}, "
              f"y(lam_b)_unshifted = {diag['y_hi_unshifted']:+.5g} MJy/sr")
        print(f"    => P_D{diag['d_hi']} = "
              f"{diag['y_hi_unshifted']:+.5g} - {diag['y_lo_extrap']:+.5g} = "
              f"{P[diag['d_hi']]:+.5g} MJy/sr")

    # Step 3: apply correction.
    C_corr_per_det = {
        d: per_det[d]['C_old'] - P[d] for d in det_order
    }

    # Step 4: quantify jumps.
    print("\n--- Step 4: boundary-jump comparison ---")
    jumps_summary = []
    for d_lo, d_hi in zip(det_order[:-1], det_order[1:]):
        j_old = boundary_jump_old(per_det, d_lo, d_hi)
        j_new = boundary_jump_new(per_det, d_lo, d_hi, P)
        # Also: gap between the two extrapolation lines AFTER correction
        # (should be ~0 by construction).
        diag = next(x for x in diagnostics
                    if x['d_lo'] == d_lo and x['d_hi'] == d_hi)
        gap_extrap_after = (diag['y_hi_unshifted'] - P[d_hi]
                            - diag['y_lo_extrap'])
        jumps_summary.append((d_lo, d_hi, j_old, j_new, gap_extrap_after))
        print(f"  D{d_lo}->D{d_hi}: "
              f"OLD ch-jump (C_old[hi,1] - C_old[lo,34]) = {j_old:+.5g}")
        print(f"            NEW ch-jump (C_corr[hi,1] - C_corr[lo,34]) = "
              f"{j_new:+.5g}  (improvement = {abs(j_old) - abs(j_new):+.5g})")
        print(f"            gap between extrapolation lines after "
              f"correction = {gap_extrap_after:+.4g}  (should be ~0)")

    # Step 5: features preserved?
    print("\n--- Step 5: feature-preservation check ---")
    feat_results, feat_notes = check_features_preserved(
        per_det, det_order, C_corr_per_det)
    for note in feat_notes:
        print(note)

    # Save data.
    npz_payload = {}
    npz_payload['detectors'] = np.asarray(det_order, dtype=np.int32)
    npz_payload['ref_detector'] = np.int32(args.reference_detector)
    npz_payload['smooth_thresh_factor'] = np.float64(args.smooth_thresh_factor)
    for d in det_order:
        npz_payload[f'WL_D{d}'] = per_det[d]['WL']
        npz_payload[f'C_old_D{d}'] = per_det[d]['C_old']
        npz_payload[f'C_corr_D{d}'] = C_corr_per_det[d]
        npz_payload[f'smooth_mask_D{d}'] = smooth_masks[d].astype(np.bool_)
        npz_payload[f'P_D{d}'] = np.float64(P[d])
        npz_payload[f'slope_old_D{d}'] = per_det[d]['slope_old']
        npz_payload[f'resid_std_D{d}'] = per_det[d]['resid_std']
    np.savez(args.out_data, **npz_payload)
    print(f"\nSaved data: {args.out_data}")

    # ------ Plot ------
    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(3, 1, height_ratios=[1.3, 1.0, 1.0], hspace=0.42)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1], sharex=ax_a)
    # Panel c is a row containing 2 subpanels.
    plot_panel_a(ax_a, per_det, det_order, C_corr_per_det, smooth_masks, P)
    plot_panel_b(ax_b, per_det, det_order, smooth_masks)
    plot_panel_c(fig, gs[2], per_det, det_order, C_corr_per_det,
                 diagnostics, P, window=args.boundary_window_um)

    feat_summary = ', '.join(
        f"{k}={'OK' if v else 'LOST'}" for k, v in feat_results.items()
    )
    fig.suptitle(
        f'Pedestal anchor v2 (data-driven): '
        f'C_corr[D,c] = C_old[D,c] - P_D    '
        f'P_D3 = {P[3]:+.4g}, P_D4 = {P[4]:+.4g}, P_D5 = {P[5]:+.4g}    '
        f'features: {feat_summary}',
        y=0.995, fontsize=11,
    )
    plt.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f"Saved plot: {args.out}")

    # ------ Final summary ------
    print("\n=== SUMMARY ===")
    for d in det_order:
        print(f"  P_D{d} = {P[d]:+.5g} MJy/sr"
              + ('  [ref, fixed=0]' if d == args.reference_detector else ''))
    for d_lo, d_hi, j_old, j_new, _ in jumps_summary:
        print(f"  D{d_lo}->D{d_hi}: jump old = {j_old:+.5g}, new = {j_new:+.5g}  "
              f"({abs(j_old) - abs(j_new):+.5g} MJy/sr smaller in magnitude)")
    print(f"  Features preserved: {feat_summary}")
    n_improved = sum(1 for _, _, j_old, j_new, _ in jumps_summary
                     if abs(j_new) < abs(j_old))
    print(f"  Comparison to v1: v2 keeps per-channel features intact "
          f"(v1's cubic C smoothed them out). Boundary jumps reduced on "
          f"{n_improved}/{len(jumps_summary)} boundaries.")


if __name__ == '__main__':
    main()
