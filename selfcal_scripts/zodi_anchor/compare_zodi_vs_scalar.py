"""Quick diagnostic: compare per-frame zodi_pred vs cal frame_scalar.

The anchor only fixes the global mean; this checks that the per-frame
variation also tracks the zodi model, which validates the time/pointing/
wavelength wiring in build_zodi_predictions.py.
"""
import argparse
import os

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--cal', required=True)
    p.add_argument('--zodi-pred', required=True)
    p.add_argument('--out', default='/tmp/zodi_vs_scalar.png')
    args = p.parse_args()

    with h5py.File(args.cal, 'r') as f:
        fs = f['frame_scalar'][:].astype(np.float64)
    z_npz = np.load(args.zodi_pred)
    zp = z_npz['zodi_pred'].astype(np.float64)
    wavelength = float(z_npz['wavelength_um'])
    model_name = str(z_npz['model_name'])

    assert len(fs) == len(zp), f"length mismatch: {len(fs)} vs {len(zp)}"
    valid = np.isfinite(zp) & np.isfinite(fs)
    fs_v = fs[valid]
    zp_v = zp[valid]
    print(f"frames valid: {valid.sum()} / {len(fs)}")

    # Stats
    r = np.corrcoef(fs_v, zp_v)[0, 1]
    slope, intercept = np.polyfit(zp_v, fs_v, 1)
    print(f"mean(frame_scalar) = {fs_v.mean():.6g} MJy/sr")
    print(f"mean(zodi_pred)    = {zp_v.mean():.6g} MJy/sr")
    print(f"std(frame_scalar)  = {fs_v.std():.6g}")
    print(f"std(zodi_pred)     = {zp_v.std():.6g}")
    print(f"Pearson r          = {r:.4f}")
    print(f"linfit slope       = {slope:.4f}")
    print(f"linfit intercept   = {intercept:.6g} MJy/sr")

    C = fs_v.mean() - zp_v.mean()  # the anchor shift
    print(f"anchor C           = mean(fs) - mean(zp) = {C:.6g} MJy/sr")
    print(f"residual std       = std(fs - zp - 0) = "
          f"{(fs_v - zp_v).std():.6g} (no shift)")
    print(f"residual std after = std(fs - zp - C)  = "
          f"{(fs_v - zp_v - C).std():.6g}  (== std(fs)-zp-driven if r=1)")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(fs, label='frame_scalar', lw=0.5, alpha=0.7)
    ax.plot(zp, label=f'zodi_pred ({model_name} @ {wavelength:.2f} um)',
            lw=0.5, alpha=0.7)
    ax.set_xlabel('frame index')
    ax.set_ylabel('MJy/sr')
    ax.set_title('Per-frame DC: solved vs predicted')
    ax.legend(loc='upper right')

    ax = axes[1]
    ax.scatter(zp_v, fs_v, s=1, alpha=0.3)
    xx = np.linspace(zp_v.min(), zp_v.max(), 100)
    ax.plot(xx, slope * xx + intercept, 'r-', lw=1,
            label=f'fit: y = {slope:.3f} x + {intercept:.4g}')
    ax.plot(xx, xx, 'k--', lw=1, alpha=0.5, label='y = x')
    ax.set_xlabel('zodi_pred (MJy/sr)')
    ax.set_ylabel('frame_scalar (MJy/sr)')
    ax.set_title(f'Scatter: Pearson r = {r:.3f}')
    ax.legend(loc='upper left')

    ax = axes[2]
    resid = fs - zp - C
    ax.plot(resid, lw=0.5, alpha=0.7)
    ax.axhline(0, color='r', lw=0.5)
    ax.set_xlabel('frame index')
    ax.set_ylabel('residual (MJy/sr)')
    ax.set_title(f'frame_scalar - zodi_pred - C   (std={(fs_v-zp_v-C).std():.4g})')

    plt.tight_layout()
    plt.savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == '__main__':
    main()
