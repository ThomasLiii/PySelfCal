"""Plot the SPHEREx chunk_map with chunk edges highlighted.

Two panels:
  (a) Full detector, coloured by subchannel index (viridis), with all chunk
      edges drawn as thin black lines so the reader can see how the
      detector is partitioned into (subchannel arc x column strip) chunks.
  (b) Zoomed-in view to show a few individual chunks clearly.
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_SELFCAL_ROOT = os.path.dirname(os.path.dirname(_PKG_DIR))
if _SELFCAL_ROOT not in sys.path:
    sys.path.insert(0, _SELFCAL_ROOT)

from SelfCal.SPHERExUtility import make_stripped_chunk_map, load_lvf_params
from zodi_utils import fig_path

NUM_SUB, NUM_CH, NUM_COL = 10, 34, 3
TOT_SUB = NUM_SUB * NUM_CH + 2


def _arc_segments(r_edges, xc, yc, x_lo, x_hi, y_lo, y_hi, n_pts=2000):
    """Yield (xs, ys) sample points for each arc r = R lying inside the
    [x_lo, x_hi] x [y_lo, y_hi] window. Arc form: y = -sqrt(R^2 - (x-xc)^2) + yc.
    """
    for R in r_edges:
        x_arc_lo = max(x_lo, xc - R)
        x_arc_hi = min(x_hi, xc + R)
        if x_arc_lo >= x_arc_hi:
            continue
        xs = np.linspace(x_arc_lo, x_arc_hi, n_pts)
        ys = -np.sqrt(np.maximum(R * R - (xs - xc) ** 2, 0.0)) + yc
        m = (ys >= y_lo) & (ys <= y_hi)
        if m.any():
            yield xs[m], ys[m]


def main(detector, zoom_x0, zoom_y0, zoom_size, dpi):
    lvf = load_lvf_params(f'lvf_params_D{detector}.npy')
    det_chunk_map, _, r_edges, x_edges = make_stripped_chunk_map(
        detector, num_subchannels=NUM_SUB, num_channels=NUM_CH,
        num_columns=NUM_COL, oversample_factor=1, lvf_params=lvf,
    )
    xc = float(lvf['xc'])
    yc = float(lvf['yc'])

    # subchannel index per pixel (used only as the colour background).
    sub_per_pixel = (det_chunk_map // NUM_COL).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # (a) Full detector (square).
    ax = axes[0]
    im = ax.imshow(sub_per_pixel, cmap='viridis', origin='lower',
                   aspect='equal', interpolation='nearest')
    # Smooth analytic subchannel arcs from the LVF spline (r = R, centre xc/yc).
    for xs, ys in _arc_segments(r_edges, xc, yc,
                                 x_lo=0, x_hi=2040,
                                 y_lo=0, y_hi=2040):
        ax.plot(xs, ys, color='black', lw=0.35, alpha=0.7)
    # Column-strip boundaries.
    for xe in x_edges[1:-1]:
        ax.axvline(xe, color='white', lw=2.0, alpha=0.85)
    # Column labels INSIDE the panel near the top so they don't get clipped.
    col_centres = 0.5 * (x_edges[:-1] + x_edges[1:])
    for c, lbl in zip(col_centres, ['Left', 'Mid', 'Right']):
        ax.text(c, 1990, f'col: {lbl}', color='white',
                ha='center', va='top', fontsize=11, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='black',
                          alpha=0.55, lw=0))
    ax.set(xlim=(0, 2040), ylim=(0, 2040),
           xlabel='Detector X [pix]', ylabel='Detector Y [pix]',
           title=(f'(a) Detector {detector} chunk_map  '
                  f'({TOT_SUB} subchannels × {NUM_COL} columns '
                  f'= {TOT_SUB * NUM_COL} chunks)'))
    plt.colorbar(im, ax=ax, label='Subchannel Index')
    # Indicate the zoom box.
    ax.add_patch(plt.Rectangle((zoom_x0, zoom_y0), zoom_size, zoom_size,
                               linewidth=2.0, edgecolor='red',
                               facecolor='none'))

    # (b) Zoom: square area, equal aspect so the chunk geometry isn't distorted.
    sl_y = slice(zoom_y0, zoom_y0 + zoom_size)
    sl_x = slice(zoom_x0, zoom_x0 + zoom_size)
    sub_zoom = sub_per_pixel[sl_y, sl_x]
    chunk_zoom = det_chunk_map[sl_y, sl_x]
    ax = axes[1]
    im = ax.imshow(sub_zoom, cmap='viridis', origin='lower',
                   aspect='equal', interpolation='nearest',
                   extent=(zoom_x0, zoom_x0 + zoom_size,
                           zoom_y0, zoom_y0 + zoom_size))
    # Smooth arcs in the zoom window.
    for xs, ys in _arc_segments(r_edges, xc, yc,
                                 x_lo=zoom_x0, x_hi=zoom_x0 + zoom_size,
                                 y_lo=zoom_y0, y_hi=zoom_y0 + zoom_size):
        ax.plot(xs, ys, color='black', lw=1.2, alpha=0.85)
    for xe in x_edges[1:-1]:
        if zoom_x0 <= xe <= zoom_x0 + zoom_size:
            ax.axvline(xe, color='white', lw=2.0, alpha=0.85)
    ax.set_xlim(zoom_x0, zoom_x0 + zoom_size)
    ax.set_ylim(zoom_y0, zoom_y0 + zoom_size)
    plt.colorbar(im, ax=ax, label='Subchannel Index')

    n_chunks_visible = int(np.unique(chunk_zoom).size)
    ax.set(xlabel='Detector X [pix]', ylabel='Detector Y [pix]',
           title=(f'(b) Zoom of red box ({zoom_size}×{zoom_size} pix, '
                  f'{n_chunks_visible} chunks)'))

    fig.suptitle(
        f'SPHEREx chunk_map structure — Detector {detector}',
        y=1.02,
    )
    fig.tight_layout()
    out = fig_path(f'chunkmap_det{detector}.png')
    fig.savefig(out, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out} (dpi={dpi})')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--detector', type=int, default=5)
    p.add_argument('--zoom-x0', type=int, default=545,
                   help='zoom lower-left x (default crosses Left/Mid boundary)')
    p.add_argument('--zoom-y0', type=int, default=965,
                   help='zoom lower-left y')
    p.add_argument('--zoom-size', type=int, default=270,
                   help='square zoom side length in pixels (~3x smaller than full panel)')
    p.add_argument('--dpi', type=int, default=300)
    args = p.parse_args()
    main(args.detector, args.zoom_x0, args.zoom_y0, args.zoom_size, args.dpi)
