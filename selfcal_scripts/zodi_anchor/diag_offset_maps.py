#!/usr/bin/env python
"""Render the per-map offset terms of a K-map cal_*.h5 onto the detector frame.

For a multi-chunk-map calibration (K>=2, the ``offsets/map_{m}`` schema) the
per-pixel offset is an additive sum of K block-constant chunk maps plus an
optional per-frame DC scalar:

    offset(pixel, frame) = frame_scalar[frame]
                           + sum_m  offsets/map_m[frame, c_m(pixel)]

where ``c_m(pixel)`` is read straight off the stored ``chunk_maps/map_m``
(a (det_h, det_w) int array assigning every detector pixel to a chunk of
map m). This script paints each map's solved offsets back onto the detector
grid via that stored chunk_map -- the faithful, block-constant rendering the
solver actually fit (the mosaic stage instead renders map_0 through a
mean-preserving spline; see PIPELINE.md `det_offset_funcs`).

Example (D3 Ch17, the baseline_poly_k2 two-map run):

    python selfcal_scripts/zodi_anchor/diag_offset_maps.py \\
        --detector 3 --channel 17 \\
        --cal /mnt/md124/thomasli/selfcal/outputs/SPHEREx_nep_qr2_det3_6p2arcsec/\
calibration/cal_Detector3_NumSub10_NumCh34_NumCol10_Ch17_baseline_poly_k2_fixed.h5

map_0 there is the spectral subchannel x column map (3420 chunks, per frame);
map_1 is a 32-column detector-fixed (frame-independent) readout-stripe map.
Output -> figures/offsets/offset_maps_D{det}_Ch{ch}_frame{F}.png .
"""
import argparse
import os

import h5py
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# repo root = two up from this file (selfcal_scripts/zodi_anchor/ -> repo)
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_FIG_DIR = os.path.join(_REPO, 'figures', 'offsets')


def pick_frame(f, requested):
    """Return a frame index. If requested is None, pick the fully-covered
    frame whose frame_scalar is closest to the median (a 'typical' exposure)."""
    if requested is not None:
        return int(requested)
    if 'frame_scalar' in f:
        fs = f['frame_scalar'][:]
        order = np.argsort(np.abs(fs - np.median(fs)))
    else:
        order = np.arange(f['offsets']['map_0'].shape[0])
    cov0 = f['offset_coverage']['map_0']
    n_chunks = cov0.shape[1]
    for idx in order[:200]:
        if np.count_nonzero(cov0[int(idx)]) == n_chunks:
            return int(idx)
    return int(order[0])


def render_map(chunk_map, offsets_frame, coverage_frame):
    """Paint a (n_chunks,) offset vector onto the detector grid via chunk_map.
    Pixels whose chunk has zero coverage this frame are set to NaN."""
    img = offsets_frame[chunk_map]
    if coverage_frame is not None:
        bad = coverage_frame <= 0
        if bad.any():
            img = np.where(bad[chunk_map], np.nan, img)
    return img


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--cal', required=True, help='path to the K-map cal_*.h5')
    ap.add_argument('--detector', type=int, required=True)
    ap.add_argument('--channel', type=int, required=True)
    ap.add_argument('--frame', type=int, default=None,
                    help='example frame index (default: a typical covered frame)')
    ap.add_argument('--out-dir', default=DEFAULT_FIG_DIR)
    args = ap.parse_args()

    with h5py.File(args.cal, 'r') as f:
        if 'offsets' not in f:
            raise SystemExit(f'{args.cal} is a legacy single-offset file '
                             '(no offsets/ group); nothing to split.')
        K = int(f.attrs.get('num_maps', len(f['offsets'])))
        frame = pick_frame(f, args.frame)
        fs = float(f['frame_scalar'][frame]) if 'frame_scalar' in f else None

        maps = []
        for m in range(K):
            cm = f['chunk_maps'][f'map_{m}'][:]
            off = f['offsets'][f'map_{m}'][frame]
            cov = (f['offset_coverage'][f'map_{m}'][frame]
                   if 'offset_coverage' in f else None)
            # detector-fixed maps are identical across frames; flag it
            grp = f['offsets'][f'map_{m}']
            fixed = False
            if grp.shape[0] > 1:
                fixed = np.array_equal(grp[0], grp[min(5000, grp.shape[0] - 1)])
            maps.append((m, render_map(cm, off, cov), off, cov, fixed,
                         len(np.unique(cm))))

    os.makedirs(args.out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, K, figsize=(6.4 * K, 6.0), squeeze=False)
    axes = axes[0]
    for ax, (m, img, off, cov, fixed, n_chunks) in zip(axes, maps):
        finite = np.isfinite(img) & (img != 0)
        vabs = np.nanpercentile(np.abs(img[finite]), 99) if finite.any() else 1.0
        vabs = vabs or 1.0
        im = ax.imshow(img, cmap='RdBu_r', vmin=-vabs, vmax=vabs, origin='lower',
                       interpolation='nearest')
        kind = 'detector-fixed' if fixed else 'per-frame'
        ax.set(xlabel='Detector X [pix]', ylabel='Detector Y [pix]',
               title=f'map_{m}  ({n_chunks} chunks, {kind})')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='offset [MJy/sr]')

    sup = (f'D{args.detector} Ch{args.channel} offset terms on detector frame '
           f'— example frame {frame}')
    if fs is not None:
        sup += f'  (frame_scalar DC = {fs:.4f} MJy/sr, removed from both panels)'
    fig.suptitle(sup, y=1.02)
    fig.tight_layout()

    out = os.path.join(args.out_dir,
                       f'offset_maps_D{args.detector}_Ch{args.channel}_frame{frame}.png')
    fig.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')
    print(f'  K={K} maps, example frame={frame}, frame_scalar={fs}')


if __name__ == '__main__':
    main()
