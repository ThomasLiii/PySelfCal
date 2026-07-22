"""Design the adaptive-overlap tile set for a tiled multi-line spectral cal.

Tiled selfcal hard-partitions frames by footprint center (frame_filter='center'),
which truncates per-pixel wavelength diversity at tile seams -> the read-time I_P
mask blanks stripes there (SPHEREx NEP D4: a frame footprint is 3156 px, LARGER
than most tiles, so the partition is very lossy). This script builds tiles that
OVERLAP into the sparse outskirts (where the extra frames are cheap and the
diversity is genuinely missing) while leaving the dense hub hard-partitioned
(diversity already high; a full-overlap disk there holds >budget frames and would
OOM). Overlapping bboxes share any frame whose center lands in the overlap
(frame_filter='center'); the Fisher stitch is tile-shape-agnostic.

Algorithm:
  1. frame-balanced binary median split into cores of <= MAX_CORE frames;
  2. grow each core's bbox one side at a time, cheapest-first, but only INTO
     SPARSE space (areal frame-density of the growth strip < DENSITY_THRESH) and
     only up to MARGIN px (a footprint half-extent) and BUDGET frames/tile.

Prints the TOML ``[tiled].tiles`` array (paste into the run config) and, with
--save, writes an npz of bboxes. Regenerates the layout shipped in
selfcal_scripts/configs/multiline_nep.toml.

Usage:
  python design_overlap_tiles.py --reproj-dir DIR [--ref-shape H W]
      [--margin 1578] [--budget 2400] [--max-core 1300] [--density 2e-4]
      [--save tiles.npz]
"""
import argparse
import glob
import os

import numpy as np

from selfcal.io.frame_select import load_ref_coords_table


def _count(cy, cx, bb):
    y0, y1, x0, x1 = bb
    return int(((cy >= y0) & (cy < y1) & (cx >= x0) & (cx < x1)).sum())


def _split_cores(cy, cx, H, W, n, max_core):
    def rec(bb, idx):
        if len(idx) <= max_core:
            return [bb]
        y0, y1, x0, x1 = bb
        if (y1 - y0) >= (x1 - x0):
            m = int(np.clip(np.median(cy[idx]), y0 + 1, y1 - 1))
            return rec((y0, m, x0, x1), idx[cy[idx] < m]) + rec((m, y1, x0, x1), idx[cy[idx] >= m])
        m = int(np.clip(np.median(cx[idx]), x0 + 1, x1 - 1))
        return rec((y0, y1, x0, m), idx[cx[idx] < m]) + rec((y0, y1, m, x1), idx[cx[idx] >= m])
    return rec((0, H, 0, W), np.arange(n))


def _grow(cy, cx, H, W, core, margin, budget, density, step=150):
    y0, y1, x0, x1 = core
    m = [0, 0, 0, 0]                                  # -y, +y, -x, +x
    lims = [y0, H - y1, x0, W - x1]
    changed = True
    while changed:
        changed = False
        base = _count(cy, cx, (y0 - m[0], y1 + m[1], x0 - m[2], x1 + m[3]))
        for s in range(4):
            if m[s] >= min(margin, lims[s]):
                continue
            mm = m.copy()
            mm[s] = min(mm[s] + step, margin, lims[s])
            bb = (y0 - mm[0], y1 + mm[1], x0 - mm[2], x1 + mm[3])
            c = _count(cy, cx, bb)
            strip = ((bb[3] - bb[2]) if s < 2 else (bb[1] - bb[0])) * step
            if c <= budget and (c - base) / max(strip, 1) < density:
                m = mm
                changed = True
                break
    return (y0 - m[0], y1 + m[1], x0 - m[2], x1 + m[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reproj-dir", required=True)
    ap.add_argument("--frame-glob", default="exp_*_det_00.h5")
    ap.add_argument("--ref-shape", type=int, nargs=2, default=[12676, 12672])
    ap.add_argument("--margin", type=int, default=1578, help="footprint half-extent (px)")
    ap.add_argument("--budget", type=int, default=2400, help="max frames/tile")
    ap.add_argument("--max-core", type=int, default=1300)
    ap.add_argument("--density", type=float, default=2e-4, help="grow-into-sparse threshold (frames/px^2)")
    ap.add_argument("--save", default=None)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.reproj_dir, a.frame_glob)),
                   key=lambda p: int(os.path.basename(p).split('_')[1]))
    tab = load_ref_coords_table(files)
    cy = (tab[:, 0] + tab[:, 1]) / 2.0
    cx = (tab[:, 2] + tab[:, 3]) / 2.0
    H, W = a.ref_shape
    cores = _split_cores(cy, cx, H, W, len(files), a.max_core)
    tiles = [_grow(cy, cx, H, W, c, a.margin, a.budget, a.density) for c in cores]
    counts = [_count(cy, cx, bb) for bb in tiles]
    print(f"{len(tiles)} tiles | frames/tile min/med/max "
          f"{min(counts)}/{int(np.median(counts))}/{max(counts)} | "
          f"total solves {sum(counts)} ({sum(counts)/len(files):.2f}x)")
    print("\n[tiled] tiles (paste into the run config):\ntiles = [")
    for i, b in enumerate(tiles):
        print(f'  {{ name = "M{i+1:02d}", bbox = [{b[0]}, {b[1]}, {b[2]}, {b[3]}] }},')
    print("]")
    if a.save:
        np.savez(a.save, bboxes=np.array(tiles, dtype=int),
                 cores=np.array(cores, dtype=int))
        print(f"\n[saved] {a.save}")


if __name__ == "__main__":
    main()
