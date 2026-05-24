"""Multi-channel batch: build zodi predictions + apply anchor + render
comparison plots for every cal file matching a given pattern.

Optimization: per-frame (MJD, WCS) extraction is done ONCE for the
detector's reproj_list and reused across all channels (since all channels
of one detector share the same exposure set). For 7 channels of D5 this
takes ~10 min instead of ~70.

Runs in the selfcal-zodipy env (zodipy needs numpy<2):

    /home/thomasli/anaconda3/envs/selfcal-zodipy/bin/python \\
        selfcal_scripts/zodi_anchor/run_multi_channel.py \\
        --cal-glob '/mnt/md124/.../cal_Detector5_*polyK1.h5' \\
        --out-dir /tmp/d5_anchors

The compare step needs matplotlib which lives in selfcal env, not
selfcal-zodipy, so the compare step is invoked via the selfcal env's
python automatically (configurable via --compare-python).
"""
import argparse
import datetime
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from astropy.io import fits

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from build_predictions import (  # noqa: E402
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_METADATA_CACHE_TEMPLATE,
    DET_BC_TEMPLATE,
    build_for_channel,
    extract_metadata_for_reproj_list,
    parse_detector_from_filename,
    save_predictions_npz,
)


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--cal-glob', required=True,
                   help='Glob for cal_*.h5 files to process. e.g. '
                        "'/mnt/.../calibration/cal_Detector5_*polyK1.h5'")
    p.add_argument('--out-dir', required=True,
                   help='Output dir for zodi_pred_*.npz, anchored cals + '
                        'mosaics, and comparison plots.')
    p.add_argument('--calibration-dir', default=DEFAULT_CALIBRATION_DIR)
    p.add_argument('--model', default='dirbe')
    p.add_argument('--grid-size', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=30)
    p.add_argument('--nprocesses', type=int, default=20)
    p.add_argument('--clip-window-days', type=float, default=7.0)
    p.add_argument('--clip-sigma', type=float, default=3.0)
    p.add_argument('--clip-iters', type=int, default=2)
    p.add_argument('--out-suffix', default='_zodianch')
    p.add_argument('--compare-python', default='/home/thomasli/anaconda3/envs/selfcal/bin/python',
                   help='Python interpreter for the comparison + anchor '
                        'steps (matplotlib lives in selfcal env, not '
                        'selfcal-zodipy).')
    p.add_argument('--skip-anchor', action='store_true',
                   help='Only build predictions; skip apply + compare.')
    p.add_argument('--skip-compare', action='store_true',
                   help='Skip the comparison plot step.')
    p.add_argument('--skip-existing', action='store_true',
                   help='Skip channels whose anchored outputs already exist.')
    p.add_argument('--skip-build', action='store_true',
                   help='Reuse existing zodi_pred_*.npz files; skip the '
                        'MJD+WCS extraction and per-channel ZodiPy eval. '
                        'Useful for re-running apply+compare after code '
                        'changes to those steps.')
    p.add_argument('--skip-existing-npz', action='store_true',
                   help='Per-channel: if zodi_pred_<tag>.npz already '
                        'exists in --out-dir, skip the ZodiPy eval for '
                        'that channel and reuse it. Useful when running '
                        'incrementally (e.g. new cals just finished and '
                        'you want to extend a previous build).')
    p.add_argument('--metadata-cache', default=None,
                   help='Persistent metadata cache file (per detector). '
                        f'Default: {DEFAULT_METADATA_CACHE_TEMPLATE}')
    return p.parse_args()


def find_mosaic(cal_path):
    cdir = os.path.dirname(cal_path)
    mdir = cdir.replace('/calibration', '/mosaic')
    base = os.path.basename(cal_path).replace('cal_', 'mosaic_', 1).replace('.h5', '.fits')
    return os.path.join(mdir, base)


def parse_channel_from_filename(cal_path):
    m = re.search(r'_Ch(\d+)_', os.path.basename(cal_path))
    return int(m.group(1)) if m else None


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cal_files = sorted(glob.glob(args.cal_glob))
    if not cal_files:
        raise SystemExit(f"no cal files matched: {args.cal_glob}")
    print(f"Found {len(cal_files)} cal files matching glob.")
    for f in cal_files:
        print(f"  {os.path.basename(f)}")

    # Validate consistency: all must share detector + reproj_list.
    detector = parse_detector_from_filename(cal_files[0])
    if detector is None:
        raise SystemExit(
            "Could not parse detector from filename; pass cal files "
            "named cal_Detector<N>_...h5.")
    print(f"detector: {detector}")

    # Determine reproj_list from the first cal; verify others match.
    with h5py.File(cal_files[0], 'r') as f:
        reproj_paths_b = f['reproj_list'][:]
    reproj_paths = [s.decode() if isinstance(s, (bytes, np.bytes_)) else s
                    for s in reproj_paths_b]
    for cal in cal_files[1:]:
        with h5py.File(cal, 'r') as f:
            other = f['reproj_list'][:]
        if not np.array_equal(reproj_paths_b, other):
            raise SystemExit(
                f"reproj_list mismatch: {cal} differs from {cal_files[0]}. "
                "Cannot share MJD/WCS across these cals.")
    if args.skip_build:
        print("--skip-build: reusing existing zodi_pred_*.npz files, "
              "skipping MJD+WCS extraction and ZodiPy eval.")
        wcs_list = mjds = det_BC = None
    else:
        print(f"All cals share the same {len(reproj_paths)}-frame reproj_list. "
              "Extracting MJD+WCS once...")
        meta_cache_path = (args.metadata_cache
                           or DEFAULT_METADATA_CACHE_TEMPLATE.format(
                               detector=detector))
        t0 = time.time()
        wcs_list, mjds, errors = extract_metadata_for_reproj_list(
            reproj_paths, num_workers=args.num_workers,
            desc=f"Reading {len(reproj_paths)} (reproj + source FITS) "
                 f"headers with {args.num_workers} workers...",
            metadata_cache_path=meta_cache_path)
        print(f"Metadata stage finished in {time.time() - t0:.1f}s "
              f"({len(errors)} read errors).")
        bc_path = os.path.join(
            args.calibration_dir, DET_BC_TEMPLATE.format(detector=detector))
        det_BC = fits.getdata(bc_path)
        print(f"Loaded det_BC: {bc_path}")

    # Iterate over channels.
    summary = []
    for cal in cal_files:
        ch = parse_channel_from_filename(cal)
        tag = os.path.basename(cal)[len('cal_'):-len('.h5')]
        npz_path = os.path.join(args.out_dir, f'zodi_pred_{tag}.npz')
        cal_anch_path = os.path.join(args.out_dir,
                                     f'cal_{tag}{args.out_suffix}.h5')
        mos_path = find_mosaic(cal)
        mos_anch_path = os.path.join(
            args.out_dir, os.path.basename(mos_path).replace(
                '.fits', f'{args.out_suffix}.fits'))
        cmp_path = os.path.join(args.out_dir, f'compare_{tag}.png')

        print()
        print(f"=== Ch{ch} ===")
        print(f"  cal:    {cal}")
        print(f"  mosaic: {mos_path}  (exists: {os.path.exists(mos_path)})")
        print(f"  npz:    {npz_path}")

        if args.skip_existing and os.path.exists(cmp_path) and os.path.exists(cal_anch_path):
            print(f"  -> already done; skipping (--skip-existing)")
            continue

        # 1) Build predictions (unless reusing existing .npz).
        if args.skip_build:
            if not os.path.exists(npz_path):
                print(f"  --skip-build but {npz_path} missing; "
                      f"cannot proceed for Ch{ch}.")
                continue
            with np.load(npz_path) as z:
                wavelength_um = float(z['wavelength_um'])
            result = {'wavelength_um': wavelength_um}
            print(f"  reusing {npz_path}; wavelength = {wavelength_um:.4f} um")
        else:
            t_b = time.time()
            result = build_for_channel(
                cal, wcs_list, mjds, det_BC, detector,
                model_name=args.model, grid_size=args.grid_size,
                nprocesses=args.nprocesses)
            save_predictions_npz(npz_path, result)
            print(f"  build: {time.time() - t_b:.1f}s")

        if args.skip_anchor:
            summary.append({'ch': ch, 'npz': npz_path,
                            'wavelength_um': result['wavelength_um']})
            continue

        if not os.path.exists(mos_path):
            print(f"  WARNING: mosaic missing, skipping anchor for Ch{ch}")
            continue

        # 2) Apply anchor (writes directly to out_dir via --out-dir).
        anchor_script = os.path.join(_HERE, 'apply_zodi_anchor.py')
        cmd = [args.compare_python, anchor_script,
               '--cal', cal, '--mosaic', mos_path,
               '--zodi-pred', npz_path,
               '--out-suffix', args.out_suffix,
               '--out-dir', args.out_dir,
               '--clip-window-days', str(args.clip_window_days),
               '--clip-sigma', str(args.clip_sigma),
               '--clip-iters', str(args.clip_iters),
               '--overwrite']
        t_a = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  ANCHOR FAILED for Ch{ch}: rc={r.returncode}")
            print(r.stdout[-1000:])
            print(r.stderr[-1000:])
            continue
        stats = parse_anchor_stdout(r.stdout)
        print(f"  apply: {time.time() - t_a:.1f}s  -> C={stats.get('C', '?')}, "
              f"slope={stats.get('slope', '?')}, r={stats.get('r', '?')}")

        # 3) Comparison plot.
        if not args.skip_compare:
            cmp_script = os.path.join(_HERE, 'compare_zodi_vs_scalar.py')
            cmd = [args.compare_python, cmp_script,
                   '--cal', cal, '--zodi-pred', npz_path,
                   '--out', cmp_path,
                   '--clip-window-days', str(args.clip_window_days),
                   '--clip-sigma', str(args.clip_sigma),
                   '--clip-iters', str(args.clip_iters)]
            t_c = time.time()
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                print(f"  COMPARE FAILED for Ch{ch}: rc={r.returncode}")
                print(r.stderr[-800:])
            else:
                print(f"  compare: {time.time() - t_c:.1f}s -> {cmp_path}")

        summary.append({
            'ch': ch,
            'cal': cal,
            'mosaic': mos_path,
            'cal_anchored': cal_anch_path,
            'mosaic_anchored': mos_anch_path,
            'npz': npz_path,
            'compare_png': cmp_path,
            'wavelength_um': result['wavelength_um'],
            **stats,
        })

    # Save summary
    summary_path = os.path.join(args.out_dir, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'detector': detector,
            'created_iso': datetime.datetime.now().isoformat(),
            'channels': summary,
        }, f, indent=2)
    print()
    print(f"=== Summary table ===")
    print(f"{'Ch':>3}  {'wavelength_um':>15}  {'C':>10}  {'slope':>7}  {'r':>7}  {'n_in':>5}  {'n_out':>5}")
    for row in summary:
        print(f"{row.get('ch', '?'):>3}  "
              f"{row.get('wavelength_um', 0):>15.4f}  "
              f"{row.get('C', float('nan')):>10.4g}  "
              f"{row.get('slope', float('nan')):>7.3f}  "
              f"{row.get('r', float('nan')):>7.3f}  "
              f"{int(row.get('n_inliers', 0) or 0):>5}  "
              f"{int(row.get('n_outliers', 0) or 0):>5}")
    print(f"Saved summary: {summary_path}")


def parse_anchor_stdout(text):
    """Pull C / slope / r / n_inliers / n_outliers out of apply_zodi_anchor stdout."""
    out = {}
    patterns = {
        'C': r'linfit intercept\s*=\s*([-\d\.eE+]+)\s*MJy/sr',
        'slope': r'linfit slope\s*=\s*([-\d\.eE+]+)',
        'r': r'Pearson r\s*=\s*([-\d\.eE+]+)',
        'n_inliers': r'frames in fit:\s*(\d+)',
        'n_outliers': r'rejected\s*(\d+)\s*outliers',
        'mean_scalar': r'mean\(frame_scalar\)\s*=\s*([-\d\.eE+]+)',
        'mean_pred': r'mean\(zodi_pred\)\s*=\s*([-\d\.eE+]+)',
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            try:
                out[key] = float(m.group(1)) if '.' in m.group(1) or 'e' in m.group(1).lower() else int(m.group(1))
            except ValueError:
                pass
    return out


if __name__ == '__main__':
    main()
