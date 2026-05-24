"""Compare zodi IPD models on the same cal files.

For each (model, channel) pair: build zodi_pred (sharing MJD+WCS across
all combinations), compute full_DC from cal file, run linfit + sigma-clip,
record (slope, intercept, r). Output a JSON summary and a comparison
figure (slope vs lambda + intercept vs lambda, one curve per model).

Does NOT write anchored cal/mosaic files — this is purely a diagnostic to
see how the choice of IPD model affects slope and the anchor constant.

Run in the selfcal-zodipy env (needs zodipy + matplotlib).
"""
import argparse
import datetime
import glob
import json
import os
import re
import sys

import h5py
import hdf5plugin  # noqa: F401
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
from SelfCal.ZodiAnchor import compute_full_dc, fit_with_clip  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--cal-glob', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--models', nargs='+',
                   default=['dirbe', 'rrm-experimental', 'odegard', 'planck18'],
                   help='Space-separated list of zodipy model names.')
    p.add_argument('--calibration-dir', default=DEFAULT_CALIBRATION_DIR)
    p.add_argument('--grid-size', type=int, default=1)
    p.add_argument('--num-workers', type=int, default=30)
    p.add_argument('--nprocesses', type=int, default=20)
    p.add_argument('--clip-window-days', type=float, default=7.0)
    p.add_argument('--clip-sigma', type=float, default=3.0)
    p.add_argument('--clip-iters', type=int, default=2)
    p.add_argument('--skip-existing-npz', action='store_true',
                   help='Reuse zodi_pred_*.npz files in <out-dir>/<model>/ '
                        'if present (skip ZodiPy eval for that model).')
    p.add_argument('--metadata-cache', default=None,
                   help='Persistent metadata cache (per detector). '
                        f'Default: {DEFAULT_METADATA_CACHE_TEMPLATE}')
    return p.parse_args()


def parse_channel_from_filename(path):
    m = re.search(r'_Ch(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def full_dc_and_fit(cal_path, zodi_pred, mjds, args):
    """Compute full_DC for the cal and linfit vs zodi_pred."""
    with h5py.File(cal_path, 'r') as f:
        frame_scalar = f['frame_scalar'][:].astype(np.float64)
        offsets_m0 = f['offsets/map_0'][:].astype(np.float64)
        cov_m0 = f['offset_coverage/map_0'][:].astype(np.float64)
    full_DC = compute_full_dc(frame_scalar, offsets_m0, cov_m0)
    slope, intercept, r, inlier = fit_with_clip(
        zodi_pred, full_DC, mjds,
        window_days=args.clip_window_days,
        sigma=args.clip_sigma,
        iters=args.clip_iters)
    return dict(
        slope=float(slope), intercept=float(intercept), r=float(r),
        n_inliers=int(inlier.sum()),
        n_outliers=int((np.isfinite(zodi_pred) & np.isfinite(full_DC)).sum() - inlier.sum()),
        mean_full_dc=float(np.mean(full_DC[inlier])),
        mean_zodi=float(np.mean(zodi_pred[inlier])),
        mean_scalar=float(np.mean(frame_scalar[inlier])),
    )


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    cal_files = sorted(glob.glob(args.cal_glob))
    if not cal_files:
        raise SystemExit(f"no cals matched: {args.cal_glob}")
    print(f"Found {len(cal_files)} cal files, {len(args.models)} models -> "
          f"{len(cal_files) * len(args.models)} combinations")
    print(f"models: {args.models}")

    detector = parse_detector_from_filename(cal_files[0])
    if detector is None:
        raise SystemExit("Could not parse detector from filename.")

    # Verify shared reproj_list
    with h5py.File(cal_files[0], 'r') as f:
        reproj_paths_b = f['reproj_list'][:]
    reproj_paths = [s.decode() if isinstance(s, (bytes, np.bytes_)) else s
                    for s in reproj_paths_b]
    for cal in cal_files[1:]:
        with h5py.File(cal, 'r') as f:
            if not np.array_equal(reproj_paths_b, f['reproj_list'][:]):
                raise SystemExit(f"reproj_list mismatch: {cal}")

    # Single MJD+WCS extraction (cached per detector)
    meta_cache_path = (args.metadata_cache
                       or DEFAULT_METADATA_CACHE_TEMPLATE.format(
                           detector=detector))
    print(f"Extracting MJD+WCS for {len(reproj_paths)} frames "
          f"(cache: {meta_cache_path})...")
    wcs_list, mjds, errors = extract_metadata_for_reproj_list(
        reproj_paths, num_workers=args.num_workers,
        metadata_cache_path=meta_cache_path)
    print(f"  ({len(errors)} errors)")

    bc_path = os.path.join(args.calibration_dir,
                           DET_BC_TEMPLATE.format(detector=detector))
    det_BC = fits.getdata(bc_path)

    results = []  # list of dicts {model, ch, wavelength_um, slope, intercept, r, ...}

    for model in args.models:
        print()
        print(f"### model = {model} ###")
        model_dir = os.path.join(args.out_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        for cal in cal_files:
            ch = parse_channel_from_filename(cal)
            tag = os.path.basename(cal)[len('cal_'):-len('.h5')]
            npz_path = os.path.join(model_dir, f'zodi_pred_{tag}.npz')

            # Build .npz (or reuse)
            if args.skip_existing_npz and os.path.exists(npz_path):
                with np.load(npz_path) as z:
                    zodi_pred = z['zodi_pred'].astype(np.float64)
                    wavelength_um = float(z['wavelength_um'])
                    npz_mjds = z['mjds'].astype(np.float64) if 'mjds' in z.files else mjds
                print(f"  [{model}/Ch{ch}] reused {npz_path}")
            else:
                result = build_for_channel(
                    cal, wcs_list, mjds, det_BC, detector,
                    model_name=model, grid_size=args.grid_size,
                    nprocesses=args.nprocesses)
                save_predictions_npz(npz_path, result)
                zodi_pred = result['zodi_pred']
                wavelength_um = result['wavelength_um']
                npz_mjds = mjds
                print(f"  [{model}/Ch{ch}] built {npz_path}")

            fit = full_dc_and_fit(cal, zodi_pred, npz_mjds, args)
            print(f"    slope={fit['slope']:.4f}, "
                  f"intercept={fit['intercept']:.4g}, "
                  f"r={fit['r']:.4f}, n_in={fit['n_inliers']}")
            results.append({
                'model': model,
                'ch': ch,
                'wavelength_um': wavelength_um,
                **fit,
            })

    # Save JSON
    summary_path = os.path.join(args.out_dir, 'compare_models_summary.json')
    with open(summary_path, 'w') as f:
        json.dump({
            'detector': detector,
            'created_iso': datetime.datetime.now().isoformat(),
            'cal_glob': args.cal_glob,
            'models': args.models,
            'clip_window_days': args.clip_window_days,
            'clip_sigma': args.clip_sigma,
            'results': results,
        }, f, indent=2)
    print(f"\nSaved {summary_path}")

    # Plot
    out_png = os.path.join(args.out_dir, 'compare_models.png')
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True)
    colors = plt.cm.tab10(np.linspace(0, 0.6, len(args.models)))
    by_model = {m: [r for r in results if r['model'] == m] for m in args.models}
    for color, model in zip(colors, args.models):
        rows = sorted(by_model[model], key=lambda r: r['ch'])
        wls = [r['wavelength_um'] for r in rows]
        slopes = [r['slope'] for r in rows]
        intercepts = [r['intercept'] for r in rows]
        rs = [r['r'] for r in rows]
        axes[0].plot(wls, slopes, 'o-', color=color, label=model)
        axes[1].plot(wls, intercepts, 'o-', color=color, label=model)
        axes[2].plot(wls, rs, 'o-', color=color, label=model)

    axes[0].axhline(1.0, color='k', lw=0.5, alpha=0.5,
                    label='ideal slope = 1')
    axes[0].set_ylabel('linfit slope')
    axes[0].set_title('Slope vs wavelength, per IPD model')
    axes[0].legend(loc='best', fontsize=9)
    axes[0].grid(alpha=0.3)

    axes[1].set_ylabel('intercept (MJy/sr) = anchor C')
    axes[1].set_title('Anchor C vs wavelength, per IPD model')
    axes[1].legend(loc='best', fontsize=9)
    axes[1].grid(alpha=0.3)

    axes[2].set_xlabel('Channel mean wavelength (um)')
    axes[2].set_ylabel('Pearson r')
    axes[2].set_title('Goodness of fit')
    axes[2].legend(loc='best', fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_png, dpi=120)
    print(f"Saved {out_png}")

    # Print summary table
    print()
    print(f"=== Summary: slope per (model, channel) ===")
    print(f"{'Ch':>3} " + " ".join(f"{m:>14}" for m in args.models))
    chs = sorted({r['ch'] for r in results})
    for ch in chs:
        row = [f"{ch:>3}"]
        for m in args.models:
            r = next((r for r in results if r['model'] == m and r['ch'] == ch), None)
            row.append(f"{r['slope']:>14.4f}" if r else f"{'?':>14}")
        print(" ".join(row))

    print()
    print(f"=== Summary: intercept per (model, channel) ===")
    print(f"{'Ch':>3} " + " ".join(f"{m:>14}" for m in args.models))
    for ch in chs:
        row = [f"{ch:>3}"]
        for m in args.models:
            r = next((r for r in results if r['model'] == m and r['ch'] == ch), None)
            row.append(f"{r['intercept']:>14.4g}" if r else f"{'?':>14}")
        print(" ".join(row))


if __name__ == '__main__':
    main()
