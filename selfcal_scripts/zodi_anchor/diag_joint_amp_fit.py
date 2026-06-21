"""Joint per-detector amplification + per-channel C anchor fit diagnostic.

The current anchor fits ``full_DC[k, c] = slope_c * zodi_pred[k, c] + C_c``
independently per channel. This diagnostic fits a joint model that shares
ONE amplification across all 34 channels of a detector::

    full_DC[k, c] = (1 + amp_D) * zodi_pred[k, c] + C_c

with ``amp_D`` shared across the detector's 34 channels and ``C_c`` kept
per-channel. The idea is to test whether the per-channel slope variation
is mostly a single per-detector amplification (``amp_D`` non-trivial) or
genuine per-channel structure.

For each ``--run-dir`` we read the pristine cal files, the matching
``zodi_preds/zodi_pred_*.npz``, and the existing per-channel anchor file
``zodi_anchor/anchor_D{N}.h5`` (only for the old slope/C comparison).
Cal/anchor files are NOT modified.

Example::

    python diag_joint_amp_fit.py \\
        --run-dir /mnt/md124/thomasli/selfcal/outputs/<RUN_D3> \\
                  /mnt/md124/thomasli/selfcal/outputs/<RUN_D4> \\
                  /mnt/md124/thomasli/selfcal/outputs/<RUN_D5> \\
        --out-plot figures/zodi_anchor/diag_joint_amp_fit.png
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

from selfcal.zodi_anchor import compute_full_dc, load_anchor


# Detector color map (matches the convention used in the cross-channel
# overlay plots).
DET_COLORS = {1: 'tab:purple', 2: 'tab:orange',
              3: 'tab:green', 4: 'tab:blue', 5: 'tab:red'}
# Detector wavelength boundaries to mark on the spectrum (um).
# D1|D2 boundary is ~1.107 um (bands overlap slightly), D2|D3 ~1.65 um,
# D3|D4 = 2.42 um, D4|D5 = 3.81 um.
DET_BOUNDARIES_UM = (1.107, 1.65, 2.42, 3.81)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--run-dir', nargs='+', required=True,
                   help='One or more SPHEREx run directories. Each must '
                        'contain calibration/, zodi_preds/, and '
                        'zodi_anchor/anchor_D*.h5.')
    p.add_argument('--out-data', default=None,
                   help='Output .npz path for per-detector arrays. Default: '
                        'same dir as --out-plot but with .npz extension. If '
                        'neither given: /tmp/diag_joint_amp_fit.npz')
    p.add_argument('--out-plot', default=None,
                   help='Output PNG path for the 3-panel comparison plot. '
                        'Default: figures/zodi_anchor/diag_joint_amp_fit.png')
    p.add_argument('--sigma', type=float, default=3.0,
                   help='Per-channel sigma-clip threshold on residuals '
                        '(default 3.0).')
    p.add_argument('--n-iter', type=int, default=2,
                   help='Number of sigma-clip refit iterations '
                        '(default 2).')
    p.add_argument('--cal-glob-pat', default='cal_*polyK1.h5',
                   help="Glob pattern for cal files inside <run>/calibration "
                        "(default: 'cal_*polyK1.h5').")
    return p.parse_args()


def parse_detector_from_filename(path):
    m = re.search(r'Detector(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def parse_channel_from_filename(path):
    m = re.search(r'_Ch(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def matching_npz(cal_path, npz_dir):
    """zodi_pred_<tag>.npz for cal_<tag>.h5."""
    base = os.path.basename(cal_path)
    tag = base[len('cal_'):-len('.h5')]
    return os.path.join(npz_dir, f'zodi_pred_{tag}.npz')


def find_anchor_file(run_dir):
    """Return (anchor_path, detector). Errors if 0 or >1 anchors present."""
    cand = sorted(glob.glob(os.path.join(
        run_dir, 'zodi_anchor', 'anchor_D*.h5')))
    if not cand:
        raise SystemExit(f"no anchor_D*.h5 in {run_dir}/zodi_anchor/")
    if len(cand) > 1:
        raise SystemExit(f"ambiguous anchor files in {run_dir}/zodi_anchor/: "
                         f"{cand}")
    path = cand[0]
    m = re.search(r'anchor_D(\d+)\.h5', os.path.basename(path))
    if m is None:
        raise SystemExit(f"cannot parse detector from {path}")
    return path, int(m.group(1))


def load_cal_and_pred(cal_path, npz_path):
    """Read full_DC and zodi_pred (both shape (N_frames,)) + wavelength_um."""
    with h5py.File(cal_path, 'r') as f:
        frame_scalar = f['frame_scalar'][:].astype(np.float64)
        offsets_m0 = f['offsets/map_0'][:].astype(np.float64)
        cov_m0 = f['offset_coverage/map_0'][:].astype(np.float64)
    full_dc = compute_full_dc(frame_scalar, offsets_m0, cov_m0)
    with np.load(npz_path) as z:
        zodi_pred = z['zodi_pred'].astype(np.float64)
        wavelength_um = float(z['wavelength_um'])
    if zodi_pred.shape != full_dc.shape:
        raise SystemExit(f"shape mismatch for {cal_path} vs {npz_path}: "
                         f"{full_dc.shape} vs {zodi_pred.shape}")
    return full_dc, zodi_pred, wavelength_um


def joint_amp_fit(FDC, ZP, sigma=3.0, n_iter=2):
    """Joint fit of FDC = (1 + amp) * ZP + C_c.

    Parameters
    ----------
    FDC, ZP : (N_frames, N_channels) float arrays
    sigma   : per-channel sigma-clip threshold on residuals
    n_iter  : number of clip-and-refit iterations

    Returns
    -------
    dict with keys:
        amp        : float, the shared (1 + amp) - 1
        C          : (N_channels,) float, the per-channel constant
        resid_std  : (N_channels,) float, std of inlier residuals
        inlier     : (N_frames, N_channels) bool, inlier mask
    """
    FDC = np.asarray(FDC, dtype=np.float64)
    ZP = np.asarray(ZP, dtype=np.float64)
    Y = FDC - ZP  # so Y = amp * ZP + C_c

    finite = np.isfinite(Y) & np.isfinite(ZP)
    # Replace non-finite entries with 0 so that masked sums don't propagate
    # NaN. The inlier weight matrix w (below) gates which cells contribute.
    Y_safe = np.where(finite, Y, 0.0)
    ZP_safe = np.where(finite, ZP, 0.0)
    inlier = finite.copy()

    amp = 0.0
    C = np.zeros(Y.shape[1], dtype=np.float64)

    for _ in range(int(n_iter) + 1):
        # Mask non-inliers to 0 so nansum-style means work; we use sums
        # over the boolean inlier mask directly to handle per-channel
        # inlier counts.
        w = inlier.astype(np.float64)
        n_per_ch = w.sum(axis=0)
        # Guard against fully-masked channels
        safe = n_per_ch > 0
        muY = np.full(Y.shape[1], np.nan)
        muZP = np.full(Y.shape[1], np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            muY[safe] = (Y_safe * w).sum(axis=0)[safe] / n_per_ch[safe]
            muZP[safe] = (ZP_safe * w).sum(axis=0)[safe] / n_per_ch[safe]
        # For centered terms, fill the unsafe channels with 0 to avoid NaN
        # propagation; their w==0 rows zero them anyway.
        muY_filled = np.where(safe, muY, 0.0)
        muZP_filled = np.where(safe, muZP, 0.0)
        Yc = (Y_safe - muY_filled[None, :])
        ZPc = (ZP_safe - muZP_filled[None, :])
        # amp = sum_inlier(ZPc * Yc) / sum_inlier(ZPc^2). Mask via w.
        num = (ZPc * Yc * w).sum()
        den = (ZPc * ZPc * w).sum()
        amp = float(num / den) if den > 0 else 0.0
        C = muY - amp * muZP

        resid = Y_safe - amp * ZP_safe - np.where(safe, C, 0.0)[None, :]
        # Per-channel sigma-clip on residuals.
        new_inlier = finite.copy()
        for c in range(Y.shape[1]):
            r = resid[:, c]
            mask_c = finite[:, c] & inlier[:, c]
            if mask_c.sum() < 5:
                continue
            std_c = float(np.std(r[mask_c]))
            if std_c == 0:
                continue
            keep = np.abs(r) < sigma * std_c
            new_inlier[:, c] = finite[:, c] & keep
        inlier = new_inlier

    # Final stats
    resid = Y - amp * ZP - C[None, :]
    resid_std = np.full(Y.shape[1], np.nan)
    for c in range(Y.shape[1]):
        mask_c = inlier[:, c]
        if mask_c.sum() >= 2:
            resid_std[c] = float(np.std(resid[mask_c, c]))
    return dict(amp=amp, C=C, resid_std=resid_std, inlier=inlier)


def process_run(run_dir, cal_glob_pat):
    """Load all 34 channels of one run, run the joint fit, and return arrays."""
    anchor_path, detector = find_anchor_file(run_dir)
    cal_dir = os.path.join(run_dir, 'calibration')
    npz_dir = os.path.join(run_dir, 'zodi_preds')
    cals = sorted(glob.glob(os.path.join(cal_dir, cal_glob_pat)))
    if not cals:
        raise SystemExit(f"no cal files in {cal_dir} matching {cal_glob_pat}")

    # Map channel -> cal path; we expect Ch1..34 for this detector.
    by_ch = {}
    for c in cals:
        det_c = parse_detector_from_filename(c)
        if det_c != detector:
            continue
        ch = parse_channel_from_filename(c)
        if ch is None:
            continue
        by_ch[ch] = c
    chs = sorted(by_ch)
    if not chs:
        raise SystemExit(f"no channels parsed from cals in {cal_dir}")

    # First pass: determine N_frames from the first channel.
    first_fdc, first_zp, first_wl = load_cal_and_pred(
        by_ch[chs[0]], matching_npz(by_ch[chs[0]], npz_dir))
    n_frames = first_fdc.shape[0]
    n_channels = len(chs)
    FDC = np.full((n_frames, n_channels), np.nan, dtype=np.float64)
    ZP = np.full((n_frames, n_channels), np.nan, dtype=np.float64)
    WL = np.full(n_channels, np.nan, dtype=np.float64)
    FDC[:, 0] = first_fdc
    ZP[:, 0] = first_zp
    WL[0] = first_wl

    for i, ch in enumerate(chs[1:], start=1):
        cal = by_ch[ch]
        npz = matching_npz(cal, npz_dir)
        if not os.path.exists(npz):
            print(f"  [D{detector} Ch{ch}] npz missing: {npz}; "
                  f"channel skipped", file=sys.stderr)
            continue
        fdc, zp, wl = load_cal_and_pred(cal, npz)
        if fdc.shape[0] != n_frames:
            raise SystemExit(f"frame-count mismatch for {cal}: "
                             f"{fdc.shape[0]} vs {n_frames}")
        FDC[:, i] = fdc
        ZP[:, i] = zp
        WL[i] = wl

    return dict(
        detector=detector,
        anchor_path=anchor_path,
        channels=np.asarray(chs, dtype=np.int32),
        WL=WL, FDC=FDC, ZP=ZP,
    )


def main():
    args = parse_args()

    # Resolve outputs
    out_plot = args.out_plot or os.path.join(
        'figures', 'zodi_anchor', 'diag_joint_amp_fit.png')
    if args.out_data:
        out_data = args.out_data
    elif args.out_plot:
        out_data = os.path.splitext(args.out_plot)[0] + '.npz'
    else:
        out_data = '/tmp/diag_joint_amp_fit.npz'

    os.makedirs(os.path.dirname(os.path.abspath(out_plot)) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_data)) or '.', exist_ok=True)

    npz_payload = {}
    per_det = []  # list of dicts for plotting
    detectors = []

    for run_dir in args.run_dir:
        print(f"\n=== run-dir: {run_dir} ===")
        info = process_run(run_dir, args.cal_glob_pat)
        det = info['detector']
        chs = info['channels']
        WL = info['WL']
        print(f"  D{det}: {len(chs)} channels (anchor: {info['anchor_path']})")

        fit = joint_amp_fit(info['FDC'], info['ZP'],
                            sigma=args.sigma, n_iter=args.n_iter)
        amp = fit['amp']
        C_new = fit['C']
        resid_std = fit['resid_std']

        # Old per-channel slope / C from anchor file
        anchor = load_anchor(info['anchor_path'])
        slope_old = np.full(len(chs), np.nan, dtype=np.float64)
        C_old = np.full(len(chs), np.nan, dtype=np.float64)
        for i, ch in enumerate(chs):
            if int(ch) in anchor.channels:
                slope_old[i] = float(anchor.channels[int(ch)]['slope_final'])
                C_old[i] = float(anchor.channels[int(ch)]['C_final'])

        # Print summary
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            print(f"  amp_D{det}                 = {amp:+.5f}  "
                  f"(1 + amp = {1 + amp:.5f})")
            print(f"  slope_old median           = {np.nanmedian(slope_old):.4f}")
            print(f"  slope_old range            = "
                  f"[{np.nanmin(slope_old):.4f}, {np.nanmax(slope_old):.4f}]")
            print(f"  C_old median  (MJy/sr)     = {np.nanmedian(C_old):+.4g}")
            print(f"  C_new median  (MJy/sr)     = {np.nanmedian(C_new):+.4g}")
            print(f"  resid_std median (MJy/sr)  = "
                  f"{np.nanmedian(resid_std):.4g}")

        npz_payload[f'WL_D{det}'] = WL
        npz_payload[f'resid_std_D{det}'] = resid_std
        npz_payload[f'C_new_D{det}'] = C_new
        npz_payload[f'amp_D{det}'] = np.float64(amp)
        npz_payload[f'slope_old_D{det}'] = slope_old
        npz_payload[f'C_old_D{det}'] = C_old

        per_det.append(dict(
            det=det, WL=WL, amp=amp, C_new=C_new,
            slope_old=slope_old, C_old=C_old, resid_std=resid_std,
        ))
        detectors.append(det)

    detectors = sorted(set(detectors))
    npz_payload['detectors'] = np.asarray(detectors, dtype=np.int32)
    np.savez(out_data, **npz_payload)
    print(f"\nSaved data: {out_data}")

    # ---- Plot ----
    per_det.sort(key=lambda d: d['det'])
    fig, axes = plt.subplots(3, 1, figsize=(13, 11), sharex=True)

    ax_s, ax_c, ax_r = axes

    for d in per_det:
        det = d['det']
        color = DET_COLORS.get(det, 'tab:purple')
        wl = d['WL']
        order = np.argsort(wl)
        wls = wl[order]
        slope_old = d['slope_old'][order]
        C_old = d['C_old'][order]
        C_new = d['C_new'][order]
        resid_std = d['resid_std'][order]

        # (a) slopes
        ax_s.plot(wls, slope_old, '-o', ms=4, lw=1, color=color,
                  label=f'D{det} per-channel slope_old')
        ax_s.axhline(1 + d['amp'], color=color, lw=1.2, ls=':',
                     alpha=0.9,
                     label=f'D{det} joint 1+amp = {1 + d["amp"]:.4f}')

        # (b) C
        ax_c.plot(wls, C_old, '-o', mfc='none', ms=5, lw=1, color=color,
                  label=f'D{det} C_old (per-channel)')
        ax_c.plot(wls, C_new, '-s', ms=4, lw=1, color=color,
                  label=f'D{det} C_new (joint amp)')

        # (c) residual std
        ax_r.plot(wls, resid_std, '-o', ms=4, lw=1, color=color,
                  label=f'D{det} joint resid std')

    # Reference / cosmetic lines
    ax_s.axhline(1.0, color='k', lw=0.5, alpha=0.4)
    ax_c.axhline(0.0, color='k', lw=0.5, alpha=0.4)

    for bx in DET_BOUNDARIES_UM:
        for a in axes:
            a.axvline(bx, color='gray', lw=0.5, alpha=0.4)

    ax_s.set_ylabel('slope')
    ax_s.set_title('(a) Per-channel slope_old vs joint (1 + amp_D)')
    ax_s.legend(loc='best', fontsize=8, ncol=2)
    ax_s.grid(alpha=0.3)

    ax_c.set_ylabel('C  (MJy/sr)')
    ax_c.set_title('(b) Anchor constant C: per-channel (open) vs joint (solid)')
    ax_c.legend(loc='best', fontsize=8, ncol=2)
    ax_c.grid(alpha=0.3)

    ax_r.set_xlabel('Channel mean wavelength (um)')
    ax_r.set_ylabel('residual std  (MJy/sr)')
    ax_r.set_title('(c) Joint-fit per-channel residual std (inliers)')
    ax_r.legend(loc='best', fontsize=8)
    ax_r.grid(alpha=0.3)

    fig.suptitle('Joint per-detector amp + per-channel C anchor fit', y=1.00)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=130, bbox_inches='tight')
    print(f"Saved plot: {out_plot}")


if __name__ == '__main__':
    main()
