"""Slope-smoothness refit of the per-channel zodi anchor.

Attack the root cause of the per-detector C jump: the per-channel anchor
lets slope vary freely in wavelength, and slope/C trade off via
``mean(zodi_pred)`` — the OLS fit couples them as
``C = mean(full_DC) - slope * mean(zodi_pred)``, so a slope error delta
shifts C by ``-delta * mean(zodi_pred)``; at NEP zodi levels (mean pred
~0.3-0.5 MJy/sr) a few-percent slope inflation shifts C by ~10-15
mMJy/sr. If slope is constrained to vary smoothly in wavelength across
detectors, C should be continuous without the per-detector additive
pedestal correction explored in ``prototype_pedestal_anchor.py`` /
``prototype_pedestal_anchor_v2.py`` (same directory).

Three variants are compared on the same axes:

* ``per-ch`` -- the existing per-channel anchor (slope and C free per
  channel). Loaded from ``<run>/zodi_anchor/anchor_D{N}.h5``.
* ``strict`` -- one amp per detector, slope_c = (1 + amp_D), C_c free.
  Solved jointly per detector with the same moving MJD-window sigma-clip
  used by ``fit_anchor_for_channel``.
* ``smooth`` -- one global polynomial slope(lambda) across D3+D4+D5,
  C_c free per channel. Solved by normal equations on the long stacked
  design matrix, with per-channel moving sigma-clip iterations.

The script writes a 3-panel comparison figure and prints numerical
summaries (boundary jumps in C between D3->D4 and D4->D5) for each
variant.

Example::

    python selfcal_scripts/zodi_anchor/refit_smooth_slope.py \\
        --run-dir /mnt/md124/.../SPHEREx_NEP_2026W17_D3_6p2arcsec \\
                  /mnt/md124/.../SPHEREx_NEP_2026W17_D4_6p2arcsec \\
                  /mnt/md124/.../SPHEREx_NEP_2026W17_D5_6p2arcsec \\
        --poly-degree 3 \\
        --out-plot figures/zodi_anchor/refit_smooth_slope.png
"""
import argparse
import glob
import os
import re
import sys
import warnings

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from selfcal.zodi_anchor import (
    compute_full_dc,
    load_anchor,
    moving_sigma_clip_mask,
)


DET_COLORS = {1: 'tab:purple', 2: 'tab:orange',
              3: 'tab:green', 4: 'tab:blue', 5: 'tab:red'}
# Style for the three model variants.
VARIANT_STYLE = {
    'per-ch':  dict(marker='o', mfc='none', ls='-',  lw=1.0, ms=5,
                    alpha=0.95, label='per-channel anchor'),
    'strict':  dict(marker='s', mfc='none', ls='--', lw=1.0, ms=4,
                    alpha=0.95, label='strict (1+amp_D)'),
    'smooth':  dict(marker='^', mfc='none', ls='-',  lw=1.6, ms=4,
                    alpha=0.95, label='smooth poly(lambda)'),
}


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
    p.add_argument('--poly-degree', type=int, default=3,
                   help='Polynomial degree K for the smooth slope(lambda) '
                        'model (default: 3 -> 4 coefficients).')
    p.add_argument('--sigma', type=float, default=3.0,
                   help='Sigma-clip threshold on per-channel residuals '
                        '(default 3.0, matches fit_anchor_for_channel).')
    p.add_argument('--window-days', type=float, default=7.0,
                   help='Moving MJD window for the sigma-clip '
                        '(default 7.0, matches fit_anchor_for_channel).')
    p.add_argument('--n-iter', type=int, default=2,
                   help='Number of moving sigma-clip refit iterations '
                        '(default 2, matches anchor clip_iters=2).')
    p.add_argument('--cal-glob-pat', default='cal_*polyK1.h5',
                   help="Glob inside <run>/calibration (default "
                        "'cal_*polyK1.h5' -- cal files whose filename "
                        "suffix marks the linear K=1 polynomial-constraint "
                        "solve; set to match your run's cal-file suffix).")
    p.add_argument('--out-plot', default=None,
                   help='Output PNG. Default: '
                        'figures/zodi_anchor/refit_smooth_slope.png')
    p.add_argument('--out-data', default=None,
                   help='Output .npz. Default: derived from --out-plot.')
    return p.parse_args()


# ----------------------------------------------------------------------
# File parsing helpers
# ----------------------------------------------------------------------

def _detector_of(path):
    m = re.search(r'Detector(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def _channel_of(path):
    m = re.search(r'_Ch(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def _matching_npz(cal_path, npz_dir):
    base = os.path.basename(cal_path)
    tag = base[len('cal_'):-len('.h5')]
    return os.path.join(npz_dir, f'zodi_pred_{tag}.npz')


def _find_anchor(run_dir):
    cand = sorted(glob.glob(os.path.join(
        run_dir, 'zodi_anchor', 'anchor_D*.h5')))
    if len(cand) != 1:
        raise SystemExit(
            f"expected exactly one anchor_D*.h5 in {run_dir}/zodi_anchor/, "
            f"found {cand}")
    m = re.search(r'anchor_D(\d+)\.h5', os.path.basename(cand[0]))
    return cand[0], int(m.group(1))


# ----------------------------------------------------------------------
# Data loading
# ----------------------------------------------------------------------

def _load_cal_npz(cal_path, npz_path):
    """Return (full_dc, zodi_pred, mjds, wavelength_um) for one channel."""
    with h5py.File(cal_path, 'r') as f:
        fs = f['frame_scalar'][:].astype(np.float64)
        om0 = f['offsets/map_0'][:].astype(np.float64)
        cm0 = f['offset_coverage/map_0'][:].astype(np.float64)
    fdc = compute_full_dc(fs, om0, cm0)
    with np.load(npz_path, allow_pickle=False) as z:
        zp = z['zodi_pred'].astype(np.float64)
        mjds = z['mjds'].astype(np.float64)
        wl = float(z['wavelength_um'])
    if zp.shape != fdc.shape or mjds.shape != fdc.shape:
        raise SystemExit(
            f"shape mismatch in {cal_path}: fdc={fdc.shape} "
            f"zp={zp.shape} mjds={mjds.shape}")
    return fdc, zp, mjds, wl


def load_detector(run_dir, cal_glob_pat):
    """Load all channels of one detector into a dict.

    Returns dict with keys:
        detector, anchor_path,
        channels:   (n_ch,) int
        WL:         (n_ch,) float
        FDC, ZP, MJD: list of (N_frames_c,) float arrays (per channel)
    The arrays are kept ragged because N_frames can in principle differ
    per channel.
    """
    anchor_path, detector = _find_anchor(run_dir)
    cal_dir = os.path.join(run_dir, 'calibration')
    npz_dir = os.path.join(run_dir, 'zodi_preds')
    cals = sorted(
        glob.glob(os.path.join(cal_dir, cal_glob_pat)),
        key=lambda p: _channel_of(p) or -1,
    )
    by_ch = {}
    for c in cals:
        if _detector_of(c) != detector:
            continue
        ch = _channel_of(c)
        if ch is not None:
            by_ch[ch] = c
    chs = sorted(by_ch)
    if not chs:
        raise SystemExit(f"no channels parsed from cals in {cal_dir}")

    fdc_list, zp_list, mjd_list = [], [], []
    wls = np.full(len(chs), np.nan, dtype=np.float64)
    for i, ch in enumerate(chs):
        cal = by_ch[ch]
        npz = _matching_npz(cal, npz_dir)
        if not os.path.exists(npz):
            print(f"  [D{detector} Ch{ch}] npz missing: {npz}; skipping",
                  file=sys.stderr)
            fdc_list.append(np.array([], dtype=np.float64))
            zp_list.append(np.array([], dtype=np.float64))
            mjd_list.append(np.array([], dtype=np.float64))
            continue
        fdc, zp, mjds, wl = _load_cal_npz(cal, npz)
        wls[i] = wl
        fdc_list.append(fdc)
        zp_list.append(zp)
        mjd_list.append(mjds)
    return dict(
        detector=detector,
        anchor_path=anchor_path,
        channels=np.asarray(chs, dtype=np.int32),
        WL=wls,
        FDC=fdc_list,
        ZP=zp_list,
        MJD=mjd_list,
    )


# ----------------------------------------------------------------------
# Fits
# ----------------------------------------------------------------------

def _initial_inliers(fdc_list, zp_list, mjd_list):
    """Per-channel finite mask of the inputs."""
    return [
        np.isfinite(f) & np.isfinite(z) & np.isfinite(m)
        for f, z, m in zip(fdc_list, zp_list, mjd_list)
    ]


def _refresh_moving_clip(slope_per_ch, fdc_list, zp_list, mjd_list,
                         inlier_init, window_days, sigma):
    """Per-channel moving MJD-window sigma-clip on residuals.

    Returns a list of (N_frames_c,) bool inlier masks, AND-ing with the
    initial finite mask.

    `slope_per_ch[c]` is the *current* per-channel slope estimate, and
    the per-channel intercept C_c is recomputed from the inlier mean so
    that the residual is correctly centered before the clip.
    """
    new_inliers = []
    n_ch = len(fdc_list)
    for c in range(n_ch):
        fdc = fdc_list[c]
        zp = zp_list[c]
        mjds = mjd_list[c]
        init = inlier_init[c]
        if init.sum() < 10:
            new_inliers.append(init.copy())
            continue
        slope_c = slope_per_ch[c]
        # Centered intercept from current inliers
        mu_fdc = float(fdc[init].mean())
        mu_zp = float(zp[init].mean())
        C_c = mu_fdc - slope_c * mu_zp
        resid = fdc - (slope_c * zp + C_c)
        # Hide non-finite / pre-masked frames by sending residual to +inf
        resid_for_clip = np.where(init, resid, np.inf)
        keep = moving_sigma_clip_mask(mjds, resid_for_clip,
                                      window_days, sigma)
        new = init & keep
        if new.sum() < 10:
            new = init.copy()
        new_inliers.append(new)
    return new_inliers


# ---- (1) per-channel anchor stats from anchor file ------------------

def per_channel_from_anchor(anchor_path):
    """Pull slope_final, C_final, wavelength_um, mean_full_dc, mean_pred
    from the anchor file (these are already moving-clip fits)."""
    a = load_anchor(anchor_path)
    chs = sorted(a.channels)
    return dict(
        channels=np.asarray(chs, dtype=np.int32),
        wl=np.asarray([float(a.channels[c]['wavelength_um']) for c in chs]),
        slope=np.asarray([float(a.channels[c]['slope_final']) for c in chs]),
        C=np.asarray([float(a.channels[c]['C_final']) for c in chs]),
        mean_full_dc=np.asarray([float(a.channels[c]['mean_full_dc'])
                                 for c in chs]),
        mean_pred=np.asarray([float(a.channels[c]['mean_pred']) for c in chs]),
    )


# ---- (2) STRICT joint per-detector amp fit --------------------------

def fit_strict_one_detector(det_data, window_days, sigma, n_iter):
    """Fit FDC[c,f] = (1 + amp_D) * ZP[c,f] + C_c per detector.

    Uses the same moving MJD-window sigma-clip as fit_anchor_for_channel,
    applied per channel between refits.

    Returns dict: amp, C(n_ch), slope_per_ch=1+amp (length n_ch),
                  inliers (list of bool arrays), resid_std (n_ch)
    """
    fdc_list = det_data['FDC']
    zp_list = det_data['ZP']
    mjd_list = det_data['MJD']
    n_ch = len(fdc_list)

    # Start from per-channel OLS on finite frames to get a sensible
    # initial slope for the moving clip's centered residual.
    inlier_init = _initial_inliers(fdc_list, zp_list, mjd_list)
    slope_init = np.full(n_ch, 1.0, dtype=np.float64)
    for c in range(n_ch):
        msk = inlier_init[c]
        if msk.sum() < 5:
            continue
        try:
            s, _ = np.polyfit(zp_list[c][msk], fdc_list[c][msk], 1)
            slope_init[c] = float(s)
        except (np.linalg.LinAlgError, ValueError):
            pass

    # First moving clip with per-channel initial slopes.
    inliers = _refresh_moving_clip(
        slope_init, fdc_list, zp_list, mjd_list,
        inlier_init, window_days, sigma)

    amp = 0.0
    C = np.zeros(n_ch, dtype=np.float64)

    for it in range(int(n_iter) + 1):
        # Joint amp solve (per-channel centered):
        # amp = sum_c [ sum_inlier (ZPc * Yc) ] / sum_c [ sum_inlier ZPc^2 ]
        # where Y = FDC - ZP and ZPc, Yc are per-channel centered.
        num = 0.0
        den = 0.0
        muY = np.zeros(n_ch, dtype=np.float64)
        muZP = np.zeros(n_ch, dtype=np.float64)
        for c in range(n_ch):
            msk = inliers[c]
            if msk.sum() < 5:
                continue
            zp_c = zp_list[c][msk]
            fdc_c = fdc_list[c][msk]
            y_c = fdc_c - zp_c
            mY = float(y_c.mean())
            mZP = float(zp_c.mean())
            muY[c] = mY
            muZP[c] = mZP
            yc = y_c - mY
            zc = zp_c - mZP
            num += float((zc * yc).sum())
            den += float((zc * zc).sum())
        amp = float(num / den) if den > 0 else 0.0
        C = muY - amp * muZP  # exact OLS identity in the inlier sense

        # Refresh moving clip with the new common slope
        slope_per_ch = np.full(n_ch, 1.0 + amp, dtype=np.float64)
        if it < int(n_iter):
            inliers = _refresh_moving_clip(
                slope_per_ch, fdc_list, zp_list, mjd_list,
                inlier_init, window_days, sigma)

    # Final per-channel residual std on final inliers
    resid_std = np.full(n_ch, np.nan, dtype=np.float64)
    slope = 1.0 + amp
    for c in range(n_ch):
        msk = inliers[c]
        if msk.sum() < 2:
            continue
        r = fdc_list[c][msk] - (slope * zp_list[c][msk] + C[c])
        resid_std[c] = float(np.std(r))

    return dict(
        amp=amp,
        C=C,
        slope_per_ch=np.full(n_ch, 1.0 + amp, dtype=np.float64),
        resid_std=resid_std,
        inliers=inliers,
    )


# ---- (3) SMOOTH global poly slope across all detectors -------------

def fit_smooth_global(detectors_data, poly_degree,
                      window_days, sigma, n_iter):
    """Fit FDC[c,f] = poly(lambda_c; a_0..a_K) * ZP[c,f] + C_c
    across all channels of all detectors.

    Returns dict:
        coef:       (K+1,) polynomial coefficients in lambda (numpy
                    polyval convention: coef[0]*lam^K + ... + coef[K])
        C:          (N_total_ch,) per-channel constants
        slope:      (N_total_ch,) evaluated poly(lambda_c)
        WL:         (N_total_ch,) wavelengths
        det_of_ch:  (N_total_ch,) detector id for each channel
        ch_id:      (N_total_ch,) original channel id (1..34)
        resid_std:  (N_total_ch,) per-channel inlier residual std
    """
    # Flatten (det, ch) -> per-channel data
    wl_all = []
    det_all = []
    ch_all = []
    fdc_list = []
    zp_list = []
    mjd_list = []
    for det_data in detectors_data:
        det = det_data['detector']
        for i, ch in enumerate(det_data['channels']):
            wl_all.append(det_data['WL'][i])
            det_all.append(det)
            ch_all.append(int(ch))
            fdc_list.append(det_data['FDC'][i])
            zp_list.append(det_data['ZP'][i])
            mjd_list.append(det_data['MJD'][i])
    WL = np.asarray(wl_all, dtype=np.float64)
    det_of_ch = np.asarray(det_all, dtype=np.int32)
    ch_id = np.asarray(ch_all, dtype=np.int32)
    n_ch = len(fdc_list)
    K = int(poly_degree)
    n_coef = K + 1

    # Initial slope per channel from OLS (to drive the first moving clip)
    inlier_init = _initial_inliers(fdc_list, zp_list, mjd_list)
    slope_per_ch = np.ones(n_ch, dtype=np.float64)
    for c in range(n_ch):
        msk = inlier_init[c]
        if msk.sum() >= 5:
            try:
                s, _ = np.polyfit(zp_list[c][msk], fdc_list[c][msk], 1)
                slope_per_ch[c] = float(s)
            except (np.linalg.LinAlgError, ValueError):
                pass

    inliers = _refresh_moving_clip(
        slope_per_ch, fdc_list, zp_list, mjd_list,
        inlier_init, window_days, sigma)

    coef = np.zeros(n_coef, dtype=np.float64)
    C = np.zeros(n_ch, dtype=np.float64)

    for it in range(int(n_iter) + 1):
        # Build normal equations directly.
        #
        # Per (c, f) row of X (length n_coef + n_ch):
        #   [zp[c,f]*lam_c^K, ..., zp[c,f]*lam_c, zp[c,f], one-hot(c)]
        #
        # We need XtX (dense, (n_coef+n_ch)^2) and Xty (n_coef+n_ch).
        # Note: separability lets us compute these block-by-block per
        # channel without materializing X.
        #
        # Blocks:
        #   A (n_coef x n_coef): sum_c lam_c**(K-i+K-j) * sum_inlier zp^2
        #     = sum over i,j with power p_i+p_j: lam_c^{p_i+p_j} * S2_c
        #     where S2_c = sum_inlier zp^2.
        #   B (n_coef x n_ch): row i, col c -> lam_c**(K-i) * sum_inlier zp
        #     = lam_c^{p_i} * S1_c
        #   D (n_ch x n_ch): diagonal, D[c,c] = n_inlier_c
        #   right-hand side:
        #     b_top[i] = sum_c lam_c**(K-i) * sum_inlier (zp * fdc)
        #     b_bot[c] = sum_inlier fdc_c
        #
        # We compute S1_c, S2_c, T1_c=sum zp, T2_c=sum zp^2,
        # T_zfdc_c=sum zp*fdc, T_fdc_c=sum fdc per channel on inliers.

        S_zp2 = np.zeros(n_ch, dtype=np.float64)
        S_zp = np.zeros(n_ch, dtype=np.float64)
        S_zpfdc = np.zeros(n_ch, dtype=np.float64)
        S_fdc = np.zeros(n_ch, dtype=np.float64)
        S_n = np.zeros(n_ch, dtype=np.float64)
        for c in range(n_ch):
            msk = inliers[c]
            if msk.sum() < 5:
                continue
            zp_c = zp_list[c][msk]
            fdc_c = fdc_list[c][msk]
            S_zp2[c] = float((zp_c * zp_c).sum())
            S_zp[c] = float(zp_c.sum())
            S_zpfdc[c] = float((zp_c * fdc_c).sum())
            S_fdc[c] = float(fdc_c.sum())
            S_n[c] = float(msk.sum())

        # Powers of lambda per channel: lam_pow[c, k] = lam_c**(K - k)
        ks = np.arange(n_coef, dtype=np.int32)
        powers = K - ks  # [K, K-1, ..., 0]
        lam_pow = WL[:, None] ** powers[None, :]   # (n_ch, n_coef)

        # A = sum_c (S_zp2[c] * outer(lam_pow[c,:], lam_pow[c,:]))
        # via einsum -> A[i,j] = sum_c S_zp2[c] * lam_pow[c,i] * lam_pow[c,j]
        A = np.einsum('c,ci,cj->ij', S_zp2, lam_pow, lam_pow)
        # B[i, c] = lam_pow[c, i] * S_zp[c]
        B = (lam_pow * S_zp[:, None]).T  # (n_coef, n_ch)
        # D diagonal
        Ddiag = S_n
        # Right-hand side
        b_top = (lam_pow * S_zpfdc[:, None]).sum(axis=0)  # (n_coef,)
        b_bot = S_fdc  # (n_ch,)

        # Assemble full normal-equation matrix M (n_coef + n_ch square)
        M = np.zeros((n_coef + n_ch, n_coef + n_ch), dtype=np.float64)
        M[:n_coef, :n_coef] = A
        M[:n_coef, n_coef:] = B
        M[n_coef:, :n_coef] = B.T
        np.fill_diagonal(M[n_coef:, n_coef:], Ddiag)
        rhs = np.concatenate([b_top, b_bot])

        # Some channels may have S_n == 0 (no inliers); set those rows
        # to identity to keep M invertible, but their C will be nan-ed.
        zero_ch = (Ddiag == 0)
        if zero_ch.any():
            idx = np.where(zero_ch)[0] + n_coef
            for k in idx:
                M[k, :] = 0.0
                M[:, k] = 0.0
                M[k, k] = 1.0
                rhs[k] = 0.0

        sol = np.linalg.solve(M, rhs)
        coef = sol[:n_coef]
        C = sol[n_coef:]
        if zero_ch.any():
            C[zero_ch] = np.nan

        # Updated slopes for next clip
        slope_per_ch = np.polyval(coef, WL)
        if it < int(n_iter):
            inliers = _refresh_moving_clip(
                slope_per_ch, fdc_list, zp_list, mjd_list,
                inlier_init, window_days, sigma)

    # Final residual stats
    slope_per_ch = np.polyval(coef, WL)
    resid_std = np.full(n_ch, np.nan, dtype=np.float64)
    for c in range(n_ch):
        msk = inliers[c]
        if msk.sum() < 2:
            continue
        r = fdc_list[c][msk] - (slope_per_ch[c] * zp_list[c][msk] + C[c])
        resid_std[c] = float(np.std(r))

    return dict(
        coef=coef,
        C=C,
        slope=slope_per_ch,
        WL=WL,
        det_of_ch=det_of_ch,
        ch_id=ch_id,
        resid_std=resid_std,
    )


# ----------------------------------------------------------------------
# Boundary diagnostics
# ----------------------------------------------------------------------

def boundary_jump(wl_a, C_a, wl_b, C_b):
    """C jump at the boundary between detector A (long-lambda end) and
    detector B (short-lambda end) -- returns C_b[short] - C_a[long].

    Both inputs are 1-D arrays ordered as they came out of the per-channel
    fit (we sort by wavelength internally).
    """
    if len(wl_a) == 0 or len(wl_b) == 0:
        return np.nan, np.nan, np.nan
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
        'figures', 'zodi_anchor', 'refit_smooth_slope.png')
    out_data = args.out_data or (os.path.splitext(out_plot)[0] + '.npz')
    os.makedirs(os.path.dirname(os.path.abspath(out_plot)) or '.',
                exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_data)) or '.',
                exist_ok=True)

    # ------------- Load all detectors -------------
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

    # ------------- Per-channel anchor (variant 1) -------------
    print("\n--- Per-channel anchor (loaded from anchor files) ---")
    per_ch_results = {}
    for det_data in detectors_data:
        det = det_data['detector']
        per_ch = per_channel_from_anchor(det_data['anchor_path'])
        per_ch_results[det] = per_ch
        print(f"  D{det}: slope median={np.nanmedian(per_ch['slope']):.4f}  "
              f"range=[{np.nanmin(per_ch['slope']):.4f}, "
              f"{np.nanmax(per_ch['slope']):.4f}]   "
              f"C median={np.nanmedian(per_ch['C']):+.4g}  "
              f"range=[{np.nanmin(per_ch['C']):+.4g}, "
              f"{np.nanmax(per_ch['C']):+.4g}]")

    # ------------- STRICT per-detector (variant 2) -------------
    print(f"\n--- STRICT per-detector amp fit  "
          f"(window={args.window_days}d, sigma={args.sigma}, "
          f"iters={args.n_iter}) ---")
    strict_results = {}
    for det_data in detectors_data:
        det = det_data['detector']
        res = fit_strict_one_detector(
            det_data, args.window_days, args.sigma, args.n_iter)
        strict_results[det] = res
        print(f"  D{det}: amp = {res['amp']:+.5f}  (1+amp = "
              f"{1.0 + res['amp']:.5f})   "
              f"C median={np.nanmedian(res['C']):+.4g}   "
              f"resid_std median={np.nanmedian(res['resid_std']):.4g}")

    # ------------- SMOOTH global poly (variant 3) -------------
    print(f"\n--- SMOOTH global poly(lambda) slope  "
          f"K={args.poly_degree} ---")
    smooth = fit_smooth_global(
        detectors_data, args.poly_degree,
        args.window_days, args.sigma, args.n_iter)
    coef = smooth['coef']
    coef_str = ', '.join(f"a{args.poly_degree - i}={c:+.4e}"
                         for i, c in enumerate(coef))
    print(f"  poly coef (high->low order): {coef_str}")
    print(f"  slope(lambda) range = "
          f"[{np.nanmin(smooth['slope']):.4f}, "
          f"{np.nanmax(smooth['slope']):.4f}]")
    print(f"  C median = {np.nanmedian(smooth['C']):+.4g}   "
          f"resid_std median = "
          f"{np.nanmedian(smooth['resid_std']):.4g}")

    # ------------- Boundary jump summary -------------
    print("\n--- C(lambda) boundary jumps (C_short[next det] - C_long[prev det]) ---")
    det_ids = sorted(per_ch_results)
    # For convenience pull per-variant C and WL aligned per detector.
    summaries = {}
    for variant, get in [
        ('per-ch',
         lambda d: (per_ch_results[d]['wl'], per_ch_results[d]['C'])),
        ('strict',
         lambda d: (per_ch_results[d]['wl'], strict_results[d]['C'])),
    ]:
        summaries[variant] = {d: get(d) for d in det_ids}

    # smooth variant: split by detector
    smooth_per_det = {}
    for d in det_ids:
        msk = (smooth['det_of_ch'] == d)
        smooth_per_det[d] = (smooth['WL'][msk], smooth['C'][msk])
    summaries['smooth'] = smooth_per_det

    boundary_jumps = {}
    for v, per_det in summaries.items():
        print(f"  variant {v!r}:")
        jumps = []
        for i in range(len(det_ids) - 1):
            dA = det_ids[i]
            dB = det_ids[i + 1]
            wlA, CA = per_det[dA]
            wlB, CB = per_det[dB]
            j, lamA, lamB = boundary_jump(wlA, CA, wlB, CB)
            print(f"    D{dA} (lam={lamA:.3f}) -> D{dB} (lam={lamB:.3f}): "
                  f"DeltaC = {j*1e3:+.2f} mMJy/sr")
            jumps.append(dict(dA=dA, dB=dB, lamA=lamA, lamB=lamB,
                              dC_mMJy=j * 1e3))
        boundary_jumps[v] = jumps

    # Also report sample-level slope dispersion within each detector
    print("\n--- Slope dispersion within each detector ---")
    for d in det_ids:
        s_pc = per_ch_results[d]['slope']
        s_str = strict_results[d]['slope_per_ch']
        msk = (smooth['det_of_ch'] == d)
        s_sm = smooth['slope'][msk]
        print(f"  D{d}: per-ch std = {np.nanstd(s_pc):.4f}, "
              f"strict std = {np.nanstd(s_str):.4f} (single 1+amp), "
              f"smooth std (within det) = {np.nanstd(s_sm):.4f}")

    # ------------- Save .npz -------------
    npz_payload = {
        'detectors': np.asarray(det_ids, dtype=np.int32),
        'poly_degree': np.int32(args.poly_degree),
        'smooth_coef_high_to_low': coef,
        'smooth_WL': smooth['WL'],
        'smooth_C': smooth['C'],
        'smooth_slope': smooth['slope'],
        'smooth_det_of_ch': smooth['det_of_ch'],
        'smooth_ch_id': smooth['ch_id'],
        'smooth_resid_std': smooth['resid_std'],
    }
    for d in det_ids:
        npz_payload[f'WL_D{d}'] = per_ch_results[d]['wl']
        npz_payload[f'channels_D{d}'] = per_ch_results[d]['channels']
        npz_payload[f'slope_perch_D{d}'] = per_ch_results[d]['slope']
        npz_payload[f'C_perch_D{d}'] = per_ch_results[d]['C']
        npz_payload[f'slope_strict_D{d}'] = strict_results[d]['slope_per_ch']
        npz_payload[f'C_strict_D{d}'] = strict_results[d]['C']
        npz_payload[f'amp_strict_D{d}'] = np.float64(strict_results[d]['amp'])
        npz_payload[f'resid_std_strict_D{d}'] = strict_results[d]['resid_std']
    np.savez(out_data, **npz_payload)
    print(f"\nSaved data: {out_data}")

    # ------------- Plot -------------
    # One zoom panel per detector boundary; min 2 cols so top rows stay wide.
    n_boundaries = max(0, len(det_ids) - 1)
    n_zoom_cols = max(2, n_boundaries)
    fig_w = max(13.0, 4.0 * n_zoom_cols + 1.0)
    fig = plt.figure(figsize=(fig_w, 11))
    gs = fig.add_gridspec(3, n_zoom_cols, height_ratios=[1.05, 1.05, 1.0],
                          hspace=0.36, wspace=0.28)
    ax_slope = fig.add_subplot(gs[0, :])
    ax_C = fig.add_subplot(gs[1, :])
    zoom_axes_list = [fig.add_subplot(gs[2, k]) for k in range(n_zoom_cols)]
    ax_z1 = zoom_axes_list[0]
    ax_z2 = zoom_axes_list[1]

    # (a) Slope vs lambda
    for d in det_ids:
        col = DET_COLORS.get(d, 'tab:purple')
        pc = per_ch_results[d]
        order = np.argsort(pc['wl'])
        st = VARIANT_STYLE['per-ch'].copy()
        st['label'] = f'D{d} {st["label"]}'
        ax_slope.plot(pc['wl'][order], pc['slope'][order], color=col, **st)
        # strict (1+amp_D) horizontal line spanning its band
        amp = strict_results[d]['amp']
        ax_slope.plot(pc['wl'][order],
                      np.full(order.shape, 1.0 + amp),
                      color=col, ls=':', lw=2.2,
                      label=f'D{d} strict 1+amp = {1+amp:.4f}')
    # smooth (one global curve)
    wl_fine = np.linspace(np.nanmin(smooth['WL']) - 0.05,
                          np.nanmax(smooth['WL']) + 0.05, 400)
    slope_fine = np.polyval(coef, wl_fine)
    ax_slope.plot(wl_fine, slope_fine, color='k', lw=1.8,
                  label=f'smooth poly(K={args.poly_degree})')
    ax_slope.axhline(1.0, color='gray', lw=0.5, alpha=0.5)
    ax_slope.set_ylabel('slope')
    ax_slope.set_title('(a) slope(lambda): per-channel anchor vs '
                       'strict (1+amp_D) vs smooth poly')
    ax_slope.grid(alpha=0.3)
    ax_slope.legend(loc='best', fontsize=7, ncol=3)

    # (b) C vs lambda
    for d in det_ids:
        col = DET_COLORS.get(d, 'tab:purple')
        pc = per_ch_results[d]
        order = np.argsort(pc['wl'])
        wl = pc['wl'][order]
        ax_C.plot(wl, pc['C'][order], color=col,
                  marker='o', mfc='none', ls='-', ms=5, lw=0.9,
                  label=f'D{d} per-ch')
        ax_C.plot(wl, strict_results[d]['C'][order], color=col,
                  marker='s', mfc='none', ls='--', ms=4, lw=0.9,
                  label=f'D{d} strict')
        msk = (smooth['det_of_ch'] == d)
        order_sm = np.argsort(smooth['WL'][msk])
        ax_C.plot(smooth['WL'][msk][order_sm], smooth['C'][msk][order_sm],
                  color=col, marker='^', mfc='none', ls='-', ms=4, lw=1.6,
                  label=f'D{d} smooth')
    ax_C.axhline(0.0, color='gray', lw=0.5, alpha=0.5)
    ax_C.set_ylabel('C  (MJy/sr)')
    ax_C.set_title('(b) anchor constant C(lambda): per-channel vs '
                   'strict vs smooth')
    ax_C.grid(alpha=0.3)
    ax_C.legend(loc='best', fontsize=7, ncol=3)

    # (c, d) Zoom around each detector boundary
    boundaries_lam = []
    for i in range(len(det_ids) - 1):
        dA = det_ids[i]
        dB = det_ids[i + 1]
        wlA = per_ch_results[dA]['wl']
        wlB = per_ch_results[dB]['wl']
        if len(wlA) == 0 or len(wlB) == 0:
            continue
        lam_mid = 0.5 * (wlA.max() + wlB.min())
        boundaries_lam.append((dA, dB, lam_mid))

    zoom_axes = zoom_axes_list
    for k in range(len(boundaries_lam), len(zoom_axes)):
        zoom_axes[k].set_visible(False)
    for ax, (dA, dB, lam_mid) in zip(zoom_axes, boundaries_lam):
        # Plot a 0.4-um window around the boundary
        lam_lo = lam_mid - 0.25
        lam_hi = lam_mid + 0.25
        for d in (dA, dB):
            col = DET_COLORS.get(d, 'tab:purple')
            pc = per_ch_results[d]
            in_win = (pc['wl'] >= lam_lo) & (pc['wl'] <= lam_hi)
            order = np.argsort(pc['wl'][in_win])
            wl = pc['wl'][in_win][order]
            ax.plot(wl, pc['C'][in_win][order], color=col,
                    marker='o', mfc='none', ls='-', ms=6, lw=0.9,
                    label=f'D{d} per-ch')
            ax.plot(wl, strict_results[d]['C'][in_win][order], color=col,
                    marker='s', mfc='none', ls='--', ms=5, lw=0.9,
                    label=f'D{d} strict')
            msk = (smooth['det_of_ch'] == d)
            wl_sm = smooth['WL'][msk]
            C_sm = smooth['C'][msk]
            in_win_sm = (wl_sm >= lam_lo) & (wl_sm <= lam_hi)
            order_sm = np.argsort(wl_sm[in_win_sm])
            ax.plot(wl_sm[in_win_sm][order_sm],
                    C_sm[in_win_sm][order_sm],
                    color=col, marker='^', mfc='none', ls='-', ms=5, lw=1.4,
                    label=f'D{d} smooth')
        ax.axvline(lam_mid, color='gray', lw=0.6, ls=':')
        # Print boundary jumps onto the axis
        txt_lines = []
        for v in ('per-ch', 'strict', 'smooth'):
            wlA_, CA_ = summaries[v][dA]
            wlB_, CB_ = summaries[v][dB]
            j, _, _ = boundary_jump(wlA_, CA_, wlB_, CB_)
            txt_lines.append(f"{v:>7s}: DeltaC = {j*1e3:+.2f} mMJy/sr")
        ax.text(0.02, 0.98, '\n'.join(txt_lines),
                transform=ax.transAxes, va='top', ha='left',
                fontsize=8, family='monospace',
                bbox=dict(facecolor='white', alpha=0.85, lw=0.4))
        ax.set_xlabel('lambda  (um)')
        ax.set_ylabel('C  (MJy/sr)')
        ax.set_title(f'(zoom) D{dA} <-> D{dB} boundary')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=7, ncol=2)

    ax_z1.set_xlabel('lambda  (um)')
    ax_C.set_xlabel('lambda  (um)')

    fig.suptitle(
        f'Slope-smoothness refit of the zodi anchor  '
        f'(poly degree K = {args.poly_degree}, '
        f'sigma = {args.sigma}, window = {args.window_days}d, '
        f'iters = {args.n_iter})',
        y=0.995, fontsize=11)
    plt.savefig(out_plot, dpi=130, bbox_inches='tight')
    print(f"Saved plot: {out_plot}")


if __name__ == '__main__':
    main()
