"""Apply the zodi anchor in-place to many existing cal+mosaic pairs.

Loops over a cal glob, locates the matching mosaic and
``zodi_pred_<tag>.npz``, and shifts both files in-place via
``SelfCal.ZodiAnchor.apply_anchor_to_file(in_place=True)``.

Use after ``build_zodi_predictions.py`` (or
``run_multi_channel.py --skip-anchor``) has produced the .npz files.

This is a thin loop on top of the library; everything happens in one
Python process so we don't pay the Python+SelfCal import cost
per-channel (which a bash for-loop with the CLI would).

Run from the main `selfcal` env (no zodipy needed — that lives in the
sidecar `selfcal-zodipy` env where the .npz files were built).
"""
import argparse
import glob
import os
import time

import h5py

from SelfCal.ZodiAnchor import apply_anchor_to_file


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--cal-glob', required=True,
                   help="Glob pattern for cal_*.h5 files. Example: "
                        "'/data3/.../calibration/cal_Detector5_*polyK1.h5'")
    p.add_argument('--zodi-pred-dir', required=True,
                   help='Dir containing zodi_pred_<tag>.npz files.')
    p.add_argument('--clip-window-days', type=float, default=7.0)
    p.add_argument('--clip-sigma', type=float, default=3.0)
    p.add_argument('--clip-iters', type=int, default=2)
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip cals that already carry zodi_anchor_C attr.')
    p.add_argument('--dry-run', action='store_true',
                   help='List what would be done; do not modify files.')
    return p.parse_args()


def find_mosaic(cal_path):
    """Match cal_*.h5 in /<...>/calibration/ to mosaic_*.fits in /<...>/mosaic/."""
    return (cal_path
            .replace('/calibration/cal_', '/mosaic/mosaic_')
            .replace('.h5', '.fits'))


def main():
    args = parse_args()
    cals = sorted(glob.glob(args.cal_glob))
    if not cals:
        raise SystemExit(f'no cals matched: {args.cal_glob}')
    print(f'Found {len(cals)} cal files.')
    print(f'.npz dir:       {args.zodi_pred_dir}')
    print(f'sigma-clip:     {args.clip_window_days}-day window, '
          f'{args.clip_sigma}sigma, {args.clip_iters} iters')
    print(f'mode:           {"DRY-RUN (no writes)" if args.dry_run else "in-place"}')
    if args.skip_existing:
        print('--skip-existing: cals with zodi_anchor_C already set will be skipped.')

    summary = []
    t_total = time.time()
    for cal in cals:
        base = os.path.basename(cal)
        if not (base.startswith('cal_') and base.endswith('.h5')):
            print(f'\n  unexpected filename {base!r}; skipping')
            continue
        tag = base[len('cal_'):-len('.h5')]
        mos = find_mosaic(cal)
        npz = os.path.join(args.zodi_pred_dir, f'zodi_pred_{tag}.npz')

        print(f'\n=== {tag} ===')
        print(f'  cal:    {cal}')
        print(f'  mosaic: {mos}  (exists: {os.path.exists(mos)})')
        print(f'  npz:    {npz}  (exists: {os.path.exists(npz)})')

        if not os.path.exists(mos):
            print('  -> mosaic missing; skipping')
            continue
        if not os.path.exists(npz):
            print('  -> zodi_pred missing; skipping')
            continue

        if args.skip_existing:
            with h5py.File(cal, 'r') as f:
                if 'zodi_anchor_C' in f.attrs:
                    existing_C = float(f.attrs['zodi_anchor_C'])
                    print(f'  -> already anchored (C={existing_C:.4g}); skipping')
                    continue

        if args.dry_run:
            print('  -> DRY-RUN: would call apply_anchor_to_file(in_place=True)')
            continue

        t0 = time.time()
        try:
            r = apply_anchor_to_file(
                cal_in=cal, mosaic_in=mos, zodi_pred_npz=npz,
                in_place=True,
                clip_window_days=args.clip_window_days,
                clip_sigma=args.clip_sigma,
                clip_iters=args.clip_iters,
            )
            dt = time.time() - t0
            n_in = r['n_inliers']
            n_total_valid = n_in + r['n_outliers']
            print(f"  -> C={r['C']:.4g} MJy/sr, slope={r['slope']:.4f}, "
                  f"r={r['r']:.4f}, inliers={n_in}/{n_total_valid}, "
                  f"shifted HDUs={r['shifted_extnames']} ({dt:.1f}s)")
            summary.append({'tag': tag, 'C': r['C'], 'slope': r['slope'],
                            'r': r['r'], 'n_inliers': n_in,
                            'n_outliers': r['n_outliers'], 'time_s': dt})
        except Exception as e:
            print(f'  -> FAILED: {type(e).__name__}: {e}')

    print()
    print(f'=== Summary: {len(summary)} channels anchored in '
          f'{time.time() - t_total:.1f}s ===')
    if summary:
        print(f'{"tag":80s}  {"C":>10s}  {"slope":>7s}  {"r":>7s}  {"n_in":>6s}  {"n_out":>6s}')
        for s in summary:
            print(f'{s["tag"][:80]:80s}  {s["C"]:>10.4g}  '
                  f'{s["slope"]:>7.3f}  {s["r"]:>7.3f}  '
                  f'{s["n_inliers"]:>6d}  {s["n_outliers"]:>6d}')


if __name__ == '__main__':
    main()
