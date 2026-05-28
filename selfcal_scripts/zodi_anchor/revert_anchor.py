"""Undo a LEGACY in-place zodi anchor on cal + mosaic files.

The old anchor (since removed) mutated SelfCal pipeline outputs in place:

    cal:    skymap[coverage > 0] += C
            frame_scalar          -= C
            attrs zodi_anchor_*
            dataset zodi_anchor_pred
    mosaic: MEAN_MAP[weight > 0]    += C
            SC_MEAN_MAP[weight > 0] += C
            primary + ext headers: ZODIANCH/ZODISLOP/ZODICORR/ZODIMEAN

This script applies the symmetric inverse so the files return to their
pre-anchor pipeline state. It is idempotent: cals/mosaics without the
anchor markers are skipped silently.

Historical migration tool: the current anchor is non-mutating (it writes
<run>/zodi_anchor/anchor_D{N}.h5 and applies at read time), so nothing
produces in-place anchored files anymore. Kept in case legacy files
resurface (e.g. from backups). See todo/zodi_anchor_refactor.md.

Usage:
    python revert_anchor.py --run-dir /mnt/.../SPHEREx_NEP_2026W17_D1_6p2arcsec
    python revert_anchor.py --run-dir /mnt/.../D{1,4,5}_... --apply
    python revert_anchor.py --cal /path/to/cal.h5 --mosaic /path/to/mosaic.fits --apply

By default runs as a *dry-run* (reports what would change but writes
nothing). Pass --apply to actually mutate the files.
"""
import argparse
import datetime
import glob
import os
import re
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from astropy.io import fits


SHIFTED_EXTNAMES = ('MEAN_MAP', 'SC_MEAN_MAP')
CAL_ANCHOR_ATTRS = (
    'zodi_anchor_C', 'zodi_anchor_slope', 'zodi_anchor_intercept',
    'zodi_anchor_pearson_r', 'zodi_anchor_mean_full_dc',
    'zodi_anchor_mean_scalar', 'zodi_anchor_mean_pred',
    'zodi_anchor_n_inliers', 'zodi_anchor_n_outliers',
    'zodi_anchor_clip_window_days', 'zodi_anchor_clip_sigma',
    'zodi_anchor_created_iso',
)
CAL_ANCHOR_DATASETS = ('zodi_anchor_pred',)
MOSAIC_HEADER_KEYS = ('ZODIANCH', 'ZODISLOP', 'ZODICORR', 'ZODIMEAN')


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--run-dir', nargs='+',
                     help='One or more run directories; each must have '
                          'calibration/ and mosaic/ subdirs. All cal_*.h5 '
                          'inside are reverted (with their matching mosaics).')
    src.add_argument('--cal-glob',
                     help='Glob for cal_*.h5 files; mosaic is found by '
                          'replacing /calibration/ -> /mosaic/ and .h5 -> .fits.')
    src.add_argument('--cal',
                     help='Single cal file (pair with --mosaic).')
    p.add_argument('--mosaic',
                   help='Single mosaic FITS (only with --cal).')
    p.add_argument('--apply', action='store_true',
                   help='Actually mutate files. Default is dry-run.')
    p.add_argument('--skip-mosaic', action='store_true',
                   help='Revert cal files only; leave mosaics alone.')
    p.add_argument('--c-tolerance', type=float, default=1e-6,
                   help='Tolerance (MJy/sr) for mismatch between '
                        'cal.zodi_anchor_C and mosaic.ZODIANCH; mismatch '
                        'aborts the file. Default 1e-6.')
    return p.parse_args()


def find_mosaic(cal_path):
    cdir = os.path.dirname(cal_path)
    if '/calibration' not in cdir:
        return None
    mdir = cdir.replace('/calibration', '/mosaic')
    base = os.path.basename(cal_path).replace('cal_', 'mosaic_', 1).replace('.h5', '.fits')
    return os.path.join(mdir, base)


def collect_pairs(args):
    pairs = []
    if args.cal:
        if not args.mosaic and not args.skip_mosaic:
            mos = find_mosaic(args.cal)
            if mos is None:
                raise SystemExit("--cal not under a /calibration/ dir; "
                                 "pass --mosaic explicitly or --skip-mosaic.")
        else:
            mos = args.mosaic
        pairs.append((args.cal, mos))
    elif args.cal_glob:
        cals = sorted(glob.glob(args.cal_glob))
        for c in cals:
            pairs.append((c, None if args.skip_mosaic else find_mosaic(c)))
    elif args.run_dir:
        for rd in args.run_dir:
            cdir = os.path.join(rd, 'calibration')
            cals = sorted(glob.glob(os.path.join(cdir, 'cal_*.h5')))
            for c in cals:
                pairs.append((c, None if args.skip_mosaic else find_mosaic(c)))
    return pairs


def revert_cal(cal_path, apply):
    """Return (status, C, message). status in {'reverted', 'pristine', 'error'}."""
    with h5py.File(cal_path, 'r') as f:
        if 'zodi_anchor_C' not in f.attrs:
            return ('pristine', None, 'no zodi_anchor_C attr')
        C = float(f.attrs['zodi_anchor_C'])
    if not apply:
        return ('reverted', C, 'dry-run')
    with h5py.File(cal_path, 'r+') as f:
        sky = f['skymap'][...]
        if 'skymap_coverage' in f:
            cov = f['skymap_coverage'][...]
            sky[cov > 0] -= C
        else:
            sky -= C
        f['skymap'][...] = sky
        fs = f['frame_scalar']
        fs[...] = fs[...] + C
        for ds in CAL_ANCHOR_DATASETS:
            if ds in f:
                del f[ds]
        for a in CAL_ANCHOR_ATTRS:
            if a in f.attrs:
                del f.attrs[a]
        # Record the revert (cheap audit trail).
        f.attrs['zodi_anchor_reverted_iso'] = datetime.datetime.now().isoformat()
        f.attrs['zodi_anchor_reverted_C'] = C
    return ('reverted', C, 'ok')


def revert_mosaic(mos_path, expected_C, c_tol, apply):
    """Return (status, C, message)."""
    if mos_path is None or not os.path.exists(mos_path):
        return ('missing', None, f'{mos_path}')
    with fits.open(mos_path, memmap=False) as hdul:
        # Grab the stored C from the primary header or any HDU.
        stored_C = None
        for h in hdul:
            if 'ZODIANCH' in h.header:
                stored_C = float(h.header['ZODIANCH'])
                break
        if stored_C is None:
            return ('pristine', None, 'no ZODIANCH header')
    if expected_C is not None and abs(stored_C - expected_C) > c_tol:
        return ('error', stored_C,
                f'cal C={expected_C:.6g} but mosaic ZODIANCH={stored_C:.6g} '
                f'(diff {abs(stored_C - expected_C):.3g} > tol {c_tol:.1g})')
    if not apply:
        return ('reverted', stored_C, 'dry-run')
    C = stored_C
    shifted = []
    with fits.open(mos_path, mode='update') as hdul:
        ext_to_hdu = {h.header.get('EXTNAME', ''): h for h in hdul[1:]}
        # Strip header keys from everything.
        for h in hdul:
            for k in MOSAIC_HEADER_KEYS:
                if k in h.header:
                    del h.header[k]
            h.header['ZODIRVRT'] = (True, 'Zodi anchor reverted by revert_anchor.py')
        for extname in SHIFTED_EXTNAMES:
            hdu = ext_to_hdu.get(extname)
            if hdu is None or hdu.data is None:
                continue
            weight_hdu = ext_to_hdu.get(f'{extname}_WEIGHT')
            data = hdu.data
            C_typed = np.array(C, dtype=data.dtype)
            if weight_hdu is not None and weight_hdu.data is not None:
                covered = weight_hdu.data > 0
                data[covered] -= C_typed
            else:
                data -= C_typed
            hdu.data = data
            shifted.append(extname)
    return ('reverted', C, f'shifted {",".join(shifted)}')


def main():
    args = parse_args()
    pairs = collect_pairs(args)
    if not pairs:
        raise SystemExit("no cal files matched")

    if not args.apply:
        print("=== DRY RUN (no files modified). Re-run with --apply to mutate. ===\n")
    print(f"Found {len(pairs)} (cal, mosaic) pair(s)")

    n_cal_reverted = n_cal_pristine = n_cal_error = 0
    n_mos_reverted = n_mos_pristine = n_mos_missing = n_mos_error = 0

    for cal, mos in pairs:
        cal_status, cal_C, cal_msg = revert_cal(cal, args.apply)
        if cal_status == 'reverted':
            n_cal_reverted += 1
        elif cal_status == 'pristine':
            n_cal_pristine += 1
        else:
            n_cal_error += 1
        line = f"  cal  [{cal_status:8s}] C={cal_C if cal_C is not None else 'n/a':>10}  {cal_msg}"
        print(line, '  ', os.path.basename(cal))
        if mos is not None:
            mos_status, mos_C, mos_msg = revert_mosaic(
                mos, cal_C if cal_status == 'reverted' else None,
                args.c_tolerance, args.apply)
            if mos_status == 'reverted':
                n_mos_reverted += 1
            elif mos_status == 'pristine':
                n_mos_pristine += 1
            elif mos_status == 'missing':
                n_mos_missing += 1
            else:
                n_mos_error += 1
                print(f"    ERROR: {mos_msg}", file=sys.stderr)
            line = f"  mos  [{mos_status:8s}] C={mos_C if mos_C is not None else 'n/a':>10}  {mos_msg}"
            print(line, '  ', os.path.basename(mos) if mos else '(none)')

    print()
    print(f"cal:    reverted={n_cal_reverted}  pristine={n_cal_pristine}  error={n_cal_error}")
    if not args.skip_mosaic:
        print(f"mosaic: reverted={n_mos_reverted}  pristine={n_mos_pristine}  "
              f"missing={n_mos_missing}  error={n_mos_error}")
    if n_cal_error or n_mos_error:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
