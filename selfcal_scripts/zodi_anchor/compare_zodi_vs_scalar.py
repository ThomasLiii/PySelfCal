"""Quick diagnostic: compare per-frame zodi_pred vs cal full_DC.

full_DC = frame_scalar + Σ_c (N_c[k]/N[k]) · offset[k, c]   (pixel-weighted)

is the FULL per-frame DC (frame_scalar alone misses the chunk-leakage
component because mean_offsets_list pins only the unit-weighted chunk sum,
not the pixel-weighted sum).

The anchor only fixes the global mean; this checks that the per-frame
variation also tracks the zodi model, which validates the time/pointing/
wavelength wiring in build_zodi_predictions.py.

Linear fit `full_DC = slope * zp + intercept` with optional moving
sigma-clip in MJD space. The intercept is the anchor; the slope is the
validation (should be ~1 if the zodi model captures the per-frame
variation correctly).
"""
import argparse

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from SelfCal.ZodiAnchor import compute_full_dc, fit_with_clip


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cal', required=True)
    p.add_argument('--zodi-pred', required=True)
    p.add_argument('--out', default='/tmp/zodi_vs_scalar.png')
    p.add_argument('--clip-window-days', type=float, default=7.0)
    p.add_argument('--clip-sigma', type=float, default=3.0)
    p.add_argument('--clip-iters', type=int, default=2)
    args = p.parse_args()

    with h5py.File(args.cal, 'r') as f:
        frame_scalar = f['frame_scalar'][:].astype(np.float64)
        offsets_m0 = f['offsets/map_0'][:].astype(np.float64)
        cov_m0 = f['offset_coverage/map_0'][:].astype(np.float64)
    full_DC = compute_full_dc(frame_scalar, offsets_m0, cov_m0)
    # fs is full_DC throughout the rest of this script
    fs = full_DC
    z_npz = np.load(args.zodi_pred)
    zp = z_npz['zodi_pred'].astype(np.float64)
    mjds = (z_npz['mjds'].astype(np.float64) if 'mjds' in z_npz.files
            else None)
    wavelength = float(z_npz['wavelength_um'])
    model_name = str(z_npz['model_name'])

    assert len(fs) == len(zp), f"length mismatch: {len(fs)} vs {len(zp)}"
    if mjds is None:
        print("WARNING: .npz has no 'mjds' — falling back to frame-index "
              "x-axis and disabling sigma-clip.")

    slope, intercept, r, inlier = fit_with_clip(
        zp, fs, mjds,
        window_days=args.clip_window_days,
        sigma=args.clip_sigma,
        iters=args.clip_iters)
    n_inlier = int(inlier.sum())
    n_total_valid = int((np.isfinite(zp) & np.isfinite(fs)).sum())
    n_outliers = n_total_valid - n_inlier

    fs_in = fs[inlier]
    zp_in = zp[inlier]
    C = float(intercept)
    print(f"frames in fit:     {n_inlier} (rejected {n_outliers} outliers)")
    print(f"mean(full_DC)      = {fs_in.mean():.6g} MJy/sr (inliers)")
    print(f"mean(zodi_pred)    = {zp_in.mean():.6g} MJy/sr (inliers)")
    print(f"std(full_DC)       = {fs_in.std():.6g}")
    print(f"std(zodi_pred)     = {zp_in.std():.6g}")
    print(f"linfit slope       = {slope:.4f}")
    print(f"linfit intercept   = {intercept:.6g} MJy/sr  <- anchor C")
    print(f"Pearson r          = {r:.4f}")
    resid_in = fs_in - (slope * zp_in + intercept)
    print(f"residual std (inliers, post-fit) = {resid_in.std():.6g}")

    outlier = ~inlier & np.isfinite(zp) & np.isfinite(fs)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel 1: per-frame DC vs MJD (or frame index fallback)
    ax = axes[0]
    if mjds is not None:
        ord_in = np.argsort(mjds[inlier])
        ord_out = np.argsort(mjds[outlier])
        ax.scatter(mjds[inlier][ord_in], fs[inlier][ord_in], s=2, alpha=0.5,
                   c='tab:blue', label='full_DC (inlier)')
        ax.scatter(mjds[outlier], fs[outlier], s=4, c='red', marker='x',
                   alpha=0.7, label=f'full_DC outlier (clipped, n={n_outliers})')
        # Plot zodi_pred sorted in MJD (smoother)
        all_valid = np.isfinite(mjds) & np.isfinite(zp)
        ord_all = np.argsort(mjds[all_valid])
        ax.plot(mjds[all_valid][ord_all], zp[all_valid][ord_all],
                color='tab:orange', lw=0.7, alpha=0.8,
                label=f'zodi_pred ({model_name} @ {wavelength:.2f} um)')
        ax.set_xlabel('MJD')
    else:
        ax.plot(fs, label='full_DC', lw=0.5, alpha=0.7)
        ax.plot(zp, label=f'zodi_pred ({model_name} @ {wavelength:.2f} um)',
                lw=0.5, alpha=0.7)
        ax.scatter(np.where(outlier)[0], fs[outlier], s=4, c='red',
                   marker='x', alpha=0.7,
                   label=f'outlier (clipped, n={n_outliers})')
        ax.set_xlabel('frame index')
    ax.set_ylabel('MJy/sr')
    ax.set_title('Per-frame DC: solved vs predicted')
    ax.legend(loc='upper right', fontsize=8)

    # Panel 2: scatter with linfit
    ax = axes[1]
    ax.scatter(zp_in, fs_in, s=1, alpha=0.3, c='tab:blue', label='inlier')
    if outlier.any():
        ax.scatter(zp[outlier], fs[outlier], s=4, c='red', marker='x',
                   alpha=0.5, label=f'outlier (n={n_outliers})')
    xx = np.linspace(zp_in.min(), zp_in.max(), 100)
    ax.plot(xx, slope * xx + intercept, 'r-', lw=1,
            label=f'fit: y = {slope:.3f} x + {intercept:.4g}')
    ax.plot(xx, xx, 'k--', lw=1, alpha=0.5, label='y = x')
    ax.set_xlabel('zodi_pred (MJy/sr)')
    ax.set_ylabel('full_DC (MJy/sr)')
    ax.set_title(f'Scatter: Pearson r = {r:.3f}, slope = {slope:.3f}')
    ax.legend(loc='upper left', fontsize=8)

    # Panel 3: residuals (fs - linfit(zp))
    ax = axes[2]
    resid_all = fs - (slope * zp + intercept)
    if mjds is not None:
        ord_in = np.argsort(mjds[inlier])
        ax.scatter(mjds[inlier][ord_in], resid_all[inlier][ord_in],
                   s=2, alpha=0.5, c='tab:blue', label='inlier')
        if outlier.any():
            ax.scatter(mjds[outlier], resid_all[outlier], s=4, c='red',
                       marker='x', alpha=0.7, label='outlier')
        ax.set_xlabel('MJD')
    else:
        ax.scatter(np.where(inlier)[0], resid_all[inlier], s=2, alpha=0.5,
                   c='tab:blue', label='inlier')
        if outlier.any():
            ax.scatter(np.where(outlier)[0], resid_all[outlier], s=4,
                       c='red', marker='x', alpha=0.7, label='outlier')
        ax.set_xlabel('frame index')
    ax.axhline(0, color='r', lw=0.5)
    ax.set_ylabel('residual = fs - (slope·zp + intercept) (MJy/sr)')
    ax.set_title(f'Residuals (inlier std = {resid_in.std():.4g})')
    ax.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
