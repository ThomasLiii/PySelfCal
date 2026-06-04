"""Build the per-detector zodi anchor file (summary-only schema).

Reads PRISTINE cal files + their matching zodi_pred_*.npz, fits the
per-channel anchor (slope, C, r, ...) via
SelfCal.ZodiAnchor.fit_anchor_for_channel, and writes one
anchor_D{N}.h5 per detector. Never mutates cal/mosaic.

Default output: <run>/zodi_anchor/anchor_D{N}.h5
(co-located with the zodi_preds/ it references).

Runs in the selfcal env (no zodipy needed — the expensive zodipy step
already produced the npz files; this only does the cheap linear fit).

    python build_anchor.py --run-dir /mnt/.../SPHEREx_NEP_2026W17_D1_6p2arcsec

Pass --smooth to also run the Phase-1 r-weighted slope smoothing in-place
right after building (equivalent to a follow-up smooth_anchor.py). ONLY
for atmospheric-contaminated detectors (D1 He I/OI; D2) — NOT D4/D5:

    python build_anchor.py --run-dir /mnt/.../D1_... --smooth

See workspace/zodi_anchor_refactor/refactor.md for the architecture.
"""
import argparse
import glob
import os
import re
import sys
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from SelfCal.ZodiAnchor import (fit_anchor_for_channel, write_anchor,
                                smooth_anchor_file)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--run-dir', nargs='+', required=True,
                   help='Run directories (each with calibration/ + '
                        'zodi_preds/). One anchor file written per detector.')
    p.add_argument('--out-dir', default=None,
                   help='Override output dir. Default: <run>/zodi_anchor/.')
    p.add_argument('--clip-window-days', type=float, default=7.0)
    p.add_argument('--clip-sigma', type=float, default=3.0)
    p.add_argument('--clip-iters', type=int, default=2)
    p.add_argument('--cal-glob-pat', default='cal_*.h5',
                   help='Glob inside calibration/ for cal files.')
    p.add_argument('--smooth', action='store_true',
                   help='After building, run the r-weighted slope smoothing '
                        'in-place (Phase 1). ONLY for atmospheric-contaminated '
                        'detectors (D1 He I/OI; D2) — do NOT use for D4/D5, '
                        'whose low-r channels are real features. Equivalent '
                        'to running smooth_anchor.py afterward.')
    p.add_argument('--smooth-r-threshold', type=float, default=0.9,
                   help='Smoothing: flag channels with Pearson r below this '
                        '(default 0.9). Only used with --smooth.')
    p.add_argument('--smooth-s-factor', type=float, default=1.0,
                   help='Smoothing: slope-spline strength (default '
                        '1.0). Only used with --smooth.')
    return p.parse_args()


def parse_detector(path):
    m = re.search(r'Detector(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def parse_channel(path):
    m = re.search(r'_Ch(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def matching_npz(cal_path, npz_dir):
    """zodi_pred_<tag>.npz for cal_<tag>.h5."""
    base = os.path.basename(cal_path)
    tag = base[len('cal_'):-len('.h5')]
    return os.path.join(npz_dir, f'zodi_pred_{tag}.npz')


def build_one_run(run_dir, out_dir, clip, cal_glob_pat='cal_*.h5',
                  smooth=False, smooth_r_threshold=0.9, smooth_s_factor=1.0):
    cal_dir = os.path.join(run_dir, 'calibration')
    npz_dir = os.path.join(run_dir, 'zodi_preds')
    cals = sorted(glob.glob(os.path.join(cal_dir, cal_glob_pat)))
    if not cals:
        print(f"  no cal files in {cal_dir}", file=sys.stderr)
        return None

    detector = parse_detector(cals[0])
    if detector is None:
        print(f"  cannot parse detector from {cals[0]}", file=sys.stderr)
        return None

    # Sanity: warn if any cal still carries an in-place anchor (should be
    # reverted before building the anchor file).
    results = {}
    skipped = []
    for cal in cals:
        ch = parse_channel(cal)
        npz = matching_npz(cal, npz_dir)
        if not os.path.exists(npz):
            skipped.append((ch, 'npz missing'))
            continue
        with h5py.File(cal, 'r') as f:
            if 'zodi_anchor_C' in f.attrs:
                skipped.append((ch, 'cal still anchored in-place '
                                    '(revert first)'))
                continue
        try:
            res = fit_anchor_for_channel(
                cal, npz,
                clip_window_days=clip['clip_window_days'],
                clip_sigma=clip['clip_sigma'],
                clip_iters=clip['clip_iters'])
        except Exception as e:
            skipped.append((ch, f'fit error: {e}'))
            continue
        results[ch] = res
        print(f"  Ch{ch:>2}  wl={res['wavelength_um']:6.3f}  "
              f"slope={res['slope']:+.3f}  C={res['intercept']:+.5g}  "
              f"r={res['pearson_r']:+.3f}  "
              f"n_in={res['n_inliers']}/{res['n_inliers'] + res['n_outliers']}")

    if not results:
        print(f"  no channels fit for {run_dir}", file=sys.stderr)
        return None

    od = out_dir or os.path.join(run_dir, 'zodi_anchor')
    out_path = os.path.join(od, f'anchor_D{detector}.h5')
    write_anchor(out_path, detector, os.path.basename(run_dir.rstrip('/')),
                 results, clip, anchor_method='raw')
    print(f"  -> wrote {out_path} ({len(results)} channels"
          + (f", skipped {len(skipped)}" if skipped else "") + ")")
    for ch, why in skipped:
        print(f"     skip Ch{ch}: {why}")

    if smooth:
        summary = smooth_anchor_file(
            out_path, r_threshold=smooth_r_threshold,
            s_factor=smooth_s_factor)
        contam = summary['result']['contaminated']
        n_rep = int(contam.sum())
        flagged = [f"Ch{summary['chs'][i]}" for i in range(len(contam))
                   if contam[i]]
        print(f"  -> smoothed {n_rep} channel(s) in-place "
              f"(r<{smooth_r_threshold}): {', '.join(flagged) or 'none'}")
        if summary['result']['extrapolated'].any():
            print("     WARNING: some smoothed channels were extrapolated "
                  "(outside clean span) — inspect with smooth_anchor.py --plot.")
    return out_path


def main():
    args = parse_args()
    clip = dict(clip_window_days=args.clip_window_days,
                clip_sigma=args.clip_sigma,
                clip_iters=args.clip_iters)
    for run_dir in args.run_dir:
        print(f"=== {run_dir} ===")
        t0 = time.time()
        build_one_run(run_dir, args.out_dir, clip,
                      cal_glob_pat=args.cal_glob_pat,
                      smooth=args.smooth,
                      smooth_r_threshold=args.smooth_r_threshold,
                      smooth_s_factor=args.smooth_s_factor)
        print(f"  ({time.time() - t0:.1f}s)")


if __name__ == '__main__':
    main()
