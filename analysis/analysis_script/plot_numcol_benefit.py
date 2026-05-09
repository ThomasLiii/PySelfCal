"""Demonstrate that NumCol=3 calibration captures spatial zodi structure
that a NumCol=1 calibration would lose.

Argument
--------
When the calibrator runs with NumCol=3, each valid subchannel in each
exposure gets three offset values, one per vertical strip of the detector.
If the zodi brightness were spatially uniform across the ~3.5 deg detector
FoV, those three values would agree within noise. Any systematic spread is
structure that a NumCol=1 calibration (one offset per subchannel per
exposure) is forced to average away, absorbing the missed gradient into
the recovered skymap.

We establish three points:
  (1) The per-exposure spread across the 3 columns exceeds the within-column
      (across-subchannel) noise. That excess is real spatial structure.
  (2) The sign of the across-column gradient is highly persistent, not random,
      so it can't be solver noise.
  (3) The gradient varies smoothly with exposure time / sky position, as
      expected for a zodi signal that depends on geometry.

If (1) and (2) hold, using NumCol=1 absorbs a non-zero, coherent signal
into the skymap per exposure.
"""
import argparse
import os
import sys

import h5py
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_SELFCAL_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))
if _SELFCAL_ROOT not in sys.path:
    sys.path.insert(0, _SELFCAL_ROOT)

from SelfCal.SPHERExUtility import make_stripped_chunk_valid_mask
from zodi_utils import data_path, fig_path, load_cal_offsets


CAL_PATH = ('/mnt/md124/thomasli/selfcal/outputs/'
            'SPHEREx_nep_qr2_det{det}_6p2arcsec/calibration/'
            'cal_Detector{det}_NumSub10_NumCh34_NumCol3_Ch{ch}_'
            'damp0p1_reg0p1_outThresh5_sigma2.h5')

NUM_SUB, NUM_CH, NUM_COL = 10, 34, 3
TOT_SUB = NUM_SUB * NUM_CH + 2  # 342


def load_columns(detector, channel):
    """Return a (num_frames, num_valid_subchannels, 3) tensor of offsets.

    Unpadded subchannels only, so we avoid the padded subchannels whose
    offsets are constrained mostly by regularization.
    """
    path = CAL_PATH.format(det=detector, ch=channel)
    with h5py.File(path, 'r') as f:
        off = load_cal_offsets(f)[0]            # (N, 342 * 3)
    off3 = off.reshape(off.shape[0], TOT_SUB, NUM_COL)

    mask = make_stripped_chunk_valid_mask(
        ch=[channel], num_subchannels=NUM_SUB, num_channels=NUM_CH,
        num_columns=NUM_COL, subchannel_padding=0,
    ).reshape(TOT_SUB, NUM_COL)
    # A subchannel is valid if at least one column is flagged in the mask.
    valid_sub = np.where(mask.any(axis=1))[0]
    return off3[:, valid_sub, :]  # (N, n_sub, 3)


def load_exposure_metadata(detector, channel):
    pkl = data_path(f'exposure_df_det{detector}_ch{channel}.pkl')
    if not os.path.exists(pkl):
        raise FileNotFoundError(
            f'Metadata cache missing: {pkl}\n'
            f'Run  python build_metadata.py --detector {detector} --channel {channel}'
        )
    return pd.read_pickle(pkl)


def main(detector, channel):
    cube = load_columns(detector, channel)           # (N, n_sub, 3)
    n_frames, n_sub, _ = cube.shape
    print(f'cube shape: {cube.shape}')

    # Per-exposure per-column mean over valid subchannels (shape N x 3).
    col_means = cube.mean(axis=1)

    # Within-column (across-subchannel) scatter, averaged over frames.
    # This is our "noise" baseline: at fixed frame and fixed column, the 10
    # subchannels sample slightly different sky arcs, but small enough that
    # most of the scatter is LSQR solver noise.
    within_col_std = cube.std(axis=1).mean(axis=1)   # (N,)
    # Across-column spread per frame.
    across_col_std = col_means.std(axis=1)           # (N,)

    grad = col_means[:, 0] - col_means[:, 2]         # signed left-right gradient

    # Fraction of frames with col0 > col1 > col2 (or the reverse) -- a pure
    # random-sign null would give 2/6 = 0.333.
    order_frac = np.mean(
        ((col_means[:, 0] > col_means[:, 1]) & (col_means[:, 1] > col_means[:, 2]))
        | ((col_means[:, 0] < col_means[:, 1]) & (col_means[:, 1] < col_means[:, 2]))
    )
    print(f'Fraction of frames with monotonic col ordering: {order_frac:.3f}'
          f'   (random null = 0.333)')

    # Persistence of gradient sign.
    pos_frac = float(np.mean(grad > 0))
    print(f'Fraction of frames with col0 > col2: {pos_frac:.3f}   (random null = 0.5)')

    median_within = float(np.median(within_col_std))
    median_across = float(np.median(across_col_std))
    print(f'Median within-column std   (~ noise) : {median_within:.5f} MJy/sr')
    print(f'Median across-column std (signal?)   : {median_across:.5f} MJy/sr')
    print(f'Ratio across / within: {median_across / max(median_within, 1e-9):.2f}')

    # --- Hook up metadata (for the time-series + helio_lon plots) ---------
    md = load_exposure_metadata(detector, channel)
    md = md.copy()
    md['col0'] = col_means[:, 0]
    md['col1'] = col_means[:, 1]
    md['col2'] = col_means[:, 2]
    md['grad'] = grad
    md['across_std'] = across_col_std
    md['within_std'] = within_col_std

    # Clip a few bright-pixel outliers so they don't swamp the axes.
    clip = np.abs(md['grad']) < np.percentile(np.abs(md['grad']), 99)
    md = md[clip].reset_index(drop=True)

    # ---------------- Plot ----------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Within- vs across-column spread distributions.
    ax = axes[0, 0]
    bins = np.linspace(0, np.percentile(
        np.concatenate([within_col_std, across_col_std]), 99), 60)
    ax.hist(within_col_std, bins=bins, alpha=0.55, color='grey',
            label=f'within-column std (~noise)\nmedian {median_within:.4f}')
    ax.hist(across_col_std, bins=bins, alpha=0.55, color='C3',
            label=f'across-column std (signal)\nmedian {median_across:.4f}')
    ax.axvline(median_within, color='grey', lw=1.2, ls='--')
    ax.axvline(median_across, color='C3', lw=1.2, ls='--')
    ax.set(xlabel='per-exposure std [MJy/sr]',
           ylabel='# exposures',
           title=(f'(a) Across-column spread is {median_across / median_within:.1f}x '
                  f'the noise floor'))
    ax.legend(loc='upper right')

    # (b) Gradient sign histogram vs a random-shuffle null.
    ax = axes[0, 1]
    # Build a null by random shuffling columns within each frame.
    rng = np.random.default_rng(0)
    shuf = np.take_along_axis(
        col_means,
        rng.permuted(np.tile(np.arange(3), (len(col_means), 1)), axis=1),
        axis=1,
    )
    null_grad = shuf[:, 0] - shuf[:, 2]
    bmax = np.percentile(np.abs(np.concatenate([grad, null_grad])), 99)
    bins = np.linspace(-bmax, bmax, 80)
    ax.hist(null_grad, bins=bins, alpha=0.5, color='grey',
            label='column-shuffled null')
    ax.hist(md['grad'], bins=bins, alpha=0.5, color='C3',
            label=f'actual  (mean={md["grad"].mean():.4f})')
    ax.axvline(0, color='k', lw=0.7, ls='--')
    ax.set(xlabel='col0 - col2  [MJy/sr]',
           ylabel='# exposures',
           title=f'(b) Gradient sign is persistent (col0>col2 in {pos_frac:.0%})')
    ax.legend(loc='upper right')

    # (c) Gradient vs MJD -- coherent time structure, not random.
    ax = axes[1, 0]
    order = np.argsort(md['MJD_AVG'].values)
    t = md['MJD_AVG'].values[order]
    g = md['grad'].values[order]
    ax.scatter(t, g, s=2, alpha=0.25, color='C0')
    # Running median for readability.
    edges = np.linspace(t.min(), t.max(), 40)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.array([
        np.median(g[(t >= edges[i]) & (t < edges[i + 1])])
        if np.any((t >= edges[i]) & (t < edges[i + 1])) else np.nan
        for i in range(len(centers))
    ])
    ax.plot(centers, med, color='C3', lw=2, label='running median')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel='MJD', ylabel='col0 - col2 [MJy/sr]',
           title='(c) Gradient varies smoothly in time (zodi-like, not noise)')
    ax.legend()

    # (d) Gradient vs helio-ecliptic longitude (binned).
    ax = axes[1, 1]
    hl = md['helio_lon'].values
    g = md['grad'].values
    edges = np.linspace(-180, 180, 37)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.array([
        np.mean(g[(hl >= edges[i]) & (hl < edges[i + 1])])
        if np.any((hl >= edges[i]) & (hl < edges[i + 1])) else np.nan
        for i in range(len(centers))
    ])
    sems = np.array([
        (np.std(g[(hl >= edges[i]) & (hl < edges[i + 1])])
         / np.sqrt(max(1, np.sum((hl >= edges[i]) & (hl < edges[i + 1])))))
        if np.any((hl >= edges[i]) & (hl < edges[i + 1])) else np.nan
        for i in range(len(centers))
    ])
    ax.scatter(hl, g, s=1, alpha=0.06, color='grey')
    ax.errorbar(centers, means, yerr=sems, fmt='o-', color='C3',
                lw=1.5, ms=5, capsize=2, label='binned mean')
    ax.axhline(0, color='k', lw=0.6, ls='--')
    ax.set(xlabel='helio-ecliptic longitude [deg]',
           ylabel='col0 - col2 [MJy/sr]',
           title='(d) Gradient also varies with helio_lon (spatial zodi)')
    ax.legend(loc='upper right')

    fig.suptitle(
        f'NumCol=3 vs NumCol=1 benefit  |  det{detector} ch{channel}  '
        f'|  {n_frames} exposures, {n_sub} valid subchannels',
        y=1.02,
    )
    fig.tight_layout()
    out = fig_path(f'det{detector}_ch{channel}__numcol_benefit.png')
    fig.savefig(out, dpi=180, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    p.add_argument('--channel', type=int, default=17)
    args = p.parse_args()
    main(args.detector, args.channel)
