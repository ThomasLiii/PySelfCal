"""Apply a post-hoc zodi anchor to a SelfCal cal file and its matching
mosaic FITS.

The LSQR solve has one global additive degeneracy: ``sky -> sky + C`` and
``scalar[k] -> scalar[k] - C`` (same ``C`` for all frames) leaves the
data model unchanged. This script fits a linear model
``full_DC[k] = slope * zodi_pred + intercept`` over all frames and uses
the intercept as the anchor ``C``.

Crucially, ``full_DC[k]`` is the FULL per-frame DC, not just
``frame_scalar[k]``. The LSQR's ``mean_offsets_list`` constraint pins
the unit-weighted chunk-sum to 0 (``Σ_c offset[k, c] = 0``), but the
per-frame DC contribution from chunks is the PIXEL-weighted sum
``Σ_c (N_c[k]/N[k]) · offset[k, c]``, which is generally non-zero.
Using only ``frame_scalar`` therefore biases the slope upward
(empirically ~1.07-1.10 on D5 NEP 2026W17). The full-DC formulation
should bring slope close to 1.

    full_DC[k] = frame_scalar[k] + Σ_c (N_c[k] / N[k]) · offset[k, c]

Where ``N_c[k] = offset_coverage[k, c]`` from the cal file. All inputs
are already on disk in the cal file.

The slope is a validation check (should be ~1 if the zodi model
captures per-frame variation correctly). A slope far from 1 or a low
Pearson r indicates a model issue (bad astrometry, wrong wavelength,
model error).

The shift is applied uniformly: cal file (``skymap += C``,
``frame_scalar -= C`` — offsets are NOT touched), and mosaic
(``MEAN_MAP += C``, ``SC_MEAN_MAP += C``).

Originals are never mutated; anchored copies are written alongside with
``--out-suffix`` or in ``--out-dir``.
"""
import argparse
import os
import shutil

import h5py
import numpy as np
from astropy.io import fits


SHIFTED_EXTNAMES = ('MEAN_MAP', 'SC_MEAN_MAP')


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--cal', required=True,
                   help='Path to cal_*.h5 (must have /frame_scalar)')
    p.add_argument('--mosaic', required=True,
                   help='Path to matching mosaic_*.fits')
    p.add_argument('--zodi-pred', required=True,
                   help='Path to .npz with key "zodi_pred" (1D, length '
                        'num_frames, units MJy/sr, same order as cal '
                        'reproj_list). Optional key "reproj_list" '
                        'triggers a strict order check.')
    p.add_argument('--out-suffix', default='_zodianch',
                   help='Suffix for output files (default: _zodianch)')
    p.add_argument('--out-dir', default=None,
                   help='Override the directory for anchored outputs. '
                        'Default: write alongside each input file.')
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing output files')
    p.add_argument('--clip-window-days', type=float, default=7.0,
                   help='Moving sigma-clip window width in MJD days '
                        '(default: 7). Set 0 to disable clipping.')
    p.add_argument('--clip-sigma', type=float, default=3.0,
                   help='Sigma threshold for the moving clip (MAD-based) '
                        '(default: 3.0).')
    p.add_argument('--clip-iters', type=int, default=2,
                   help='Number of (clip, refit) iterations (default: 2).')
    return p.parse_args()


def load_zodi_pred_and_mjds(path, cal_reproj_list):
    """Load zodi_pred + optional mjds + optional reproj_list from .npz."""
    npz = np.load(path, allow_pickle=False)
    if 'zodi_pred' not in npz.files:
        raise SystemExit(
            f"{path} lacks 'zodi_pred' key. Available: {npz.files}")
    z = np.asarray(npz['zodi_pred'], dtype=np.float64).ravel()
    mjds = (np.asarray(npz['mjds'], dtype=np.float64).ravel()
            if 'mjds' in npz.files else None)
    if 'reproj_list' in npz.files:
        expected = [s.decode() if isinstance(s, (bytes, np.bytes_)) else s
                    for s in npz['reproj_list']]
        actual = [s.decode() if isinstance(s, (bytes, np.bytes_)) else s
                  for s in cal_reproj_list]
        if expected != actual:
            for i, (e, a) in enumerate(zip(expected, actual)):
                if e != a:
                    raise SystemExit(
                        f"zodi_pred reproj_list does not match cal "
                        f"reproj_list at index {i}:\n"
                        f"  zodi_pred: {e}\n  cal:       {a}")
            if len(expected) != len(actual):
                raise SystemExit(
                    f"zodi_pred reproj_list length {len(expected)} != "
                    f"cal reproj_list length {len(actual)}")
        print("Verified reproj_list ordering matches.")
    return z, mjds


def moving_sigma_clip_mask(mjds, residuals, window_days, sigma):
    """Boolean inlier mask via a sliding MJD-window MAD clip.

    For each frame, compute median + MAD of `residuals` within
    [mjd - W/2, mjd + W/2]; reject if |residual - local_med| > sigma * MAD/0.6745.
    Returns mask in the ORIGINAL (unsorted) frame order.
    """
    order = np.argsort(mjds, kind='stable')
    mjds_s = mjds[order]
    resid_s = residuals[order]
    n = len(mjds_s)
    keep_s = np.ones(n, dtype=bool)
    half = float(window_days) / 2.0
    # window indices via searchsorted (O(n log n) total)
    los = np.searchsorted(mjds_s, mjds_s - half, side='left')
    his = np.searchsorted(mjds_s, mjds_s + half, side='right')
    for i in range(n):
        local = resid_s[los[i]:his[i]]
        if local.size < 5:
            continue  # too few points in window to clip
        med = np.median(local)
        mad = np.median(np.abs(local - med))
        thresh = sigma * mad / 0.6745  # convert MAD to sigma-equivalent
        if thresh == 0:
            continue
        if abs(resid_s[i] - med) > thresh:
            keep_s[i] = False
    keep = np.empty_like(keep_s)
    keep[order] = keep_s
    return keep


def fit_with_clip(zp, fs, mjds, window_days, sigma, iters):
    """Iteratively linfit `fs = slope*zp + intercept` with moving
    sigma-clip on residuals. Returns slope, intercept, r, inlier_mask."""
    inlier = np.isfinite(zp) & np.isfinite(fs)
    if mjds is not None:
        inlier &= np.isfinite(mjds)
    slope, intercept = np.polyfit(zp[inlier], fs[inlier], 1)
    for it in range(int(iters)):
        if mjds is None or window_days <= 0:
            break  # no clip without mjds or with disabled window
        resid = fs - (slope * zp + intercept)
        keep = moving_sigma_clip_mask(
            mjds, np.where(inlier, resid, np.inf), window_days, sigma)
        new_inlier = inlier & keep
        n_new = int(new_inlier.sum())
        if n_new < 10:
            print(f"  clip iter {it+1}: only {n_new} inliers left, stopping")
            break
        if n_new == int(inlier.sum()):
            break  # converged
        inlier = new_inlier
        slope, intercept = np.polyfit(zp[inlier], fs[inlier], 1)
    r = float(np.corrcoef(zp[inlier], fs[inlier])[0, 1])
    return float(slope), float(intercept), r, inlier


def output_path(in_path, suffix, out_dir=None):
    if out_dir is None:
        root, ext = os.path.splitext(in_path)
        return f'{root}{suffix}{ext}'
    base = os.path.basename(in_path)
    root, ext = os.path.splitext(base)
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f'{root}{suffix}{ext}')


def load_zodi_pred(path, cal_reproj_list):
    """Load zodi_pred from .npz, optionally checking reproj_list order."""
    npz = np.load(path, allow_pickle=False)
    if 'zodi_pred' not in npz.files:
        raise SystemExit(
            f"{path} lacks 'zodi_pred' key. Available: {npz.files}")
    z = np.asarray(npz['zodi_pred'], dtype=np.float64).ravel()
    if 'reproj_list' in npz.files:
        expected = [s.decode() if isinstance(s, (bytes, np.bytes_)) else s
                    for s in npz['reproj_list']]
        actual = [s.decode() if isinstance(s, (bytes, np.bytes_)) else s
                  for s in cal_reproj_list]
        if expected != actual:
            for i, (e, a) in enumerate(zip(expected, actual)):
                if e != a:
                    raise SystemExit(
                        f"zodi_pred reproj_list does not match cal "
                        f"reproj_list at index {i}:\n"
                        f"  zodi_pred: {e}\n  cal:       {a}")
            if len(expected) != len(actual):
                raise SystemExit(
                    f"zodi_pred reproj_list length {len(expected)} != "
                    f"cal reproj_list length {len(actual)}")
        print("Verified reproj_list ordering matches.")
    return z


def anchor_cal(cal_in, cal_out, C, zodi_pred, slope, intercept, r,
               mean_scalar, mean_full_dc, mean_zodi,
               n_inliers, n_outliers,
               clip_window_days, clip_sigma):
    shutil.copyfile(cal_in, cal_out)
    with h5py.File(cal_out, 'r+') as f:
        sky = f['skymap']
        sky[...] = sky[...] + C
        fs = f['frame_scalar']
        fs[...] = fs[...] - C
        f.attrs['zodi_anchor_C'] = float(C)
        f.attrs['zodi_anchor_slope'] = float(slope)
        f.attrs['zodi_anchor_intercept'] = float(intercept)
        f.attrs['zodi_anchor_pearson_r'] = float(r)
        f.attrs['zodi_anchor_mean_full_dc'] = float(mean_full_dc)
        f.attrs['zodi_anchor_mean_scalar'] = float(mean_scalar)
        f.attrs['zodi_anchor_mean_pred'] = float(mean_zodi)
        f.attrs['zodi_anchor_n_inliers'] = int(n_inliers)
        f.attrs['zodi_anchor_n_outliers'] = int(n_outliers)
        f.attrs['zodi_anchor_clip_window_days'] = float(clip_window_days)
        f.attrs['zodi_anchor_clip_sigma'] = float(clip_sigma)
        if 'zodi_anchor_pred' in f:
            del f['zodi_anchor_pred']
        f.create_dataset('zodi_anchor_pred',
                         data=zodi_pred.astype(np.float32),
                         compression='gzip')
    print(f"Wrote anchored cal:    {cal_out}")


def anchor_mosaic(mos_in, mos_out, C, slope, r, mean_zodi):
    shutil.copyfile(mos_in, mos_out)
    shifted = []
    def stamp(header):
        header['ZODIANCH'] = (float(C), 'Zodi-anchor shift (MJy/sr) = intercept')
        header['ZODISLOP'] = (float(slope), 'Linfit slope (validation; ~1 expected)')
        header['ZODICORR'] = (float(r), 'Pearson r of frame_scalar vs zodi_pred')
        header['ZODIMEAN'] = (float(mean_zodi), 'Mean predicted zodi (MJy/sr)')
    with fits.open(mos_out, mode='update') as hdul:
        stamp(hdul[0].header)
        for hdu in hdul[1:]:
            extname = hdu.header.get('EXTNAME', '')
            if extname in SHIFTED_EXTNAMES and hdu.data is not None:
                hdu.data += np.array(C, dtype=hdu.data.dtype)
                stamp(hdu.header)
                shifted.append(extname)
    print(f"Wrote anchored mosaic: {mos_out}")
    print(f"  shifted HDUs: {shifted}")


def main():
    args = parse_args()

    with h5py.File(args.cal, 'r') as f:
        if 'frame_scalar' not in f:
            raise SystemExit(
                "Anchor requires use_per_frame_scalar=True cal runs; "
                f"{args.cal} lacks /frame_scalar.")
        frame_scalar = f['frame_scalar'][:].astype(np.float64)
        cal_reproj_list = list(f['reproj_list'][:])
        # Read offsets + offset_coverage for the full_DC computation
        if ('offsets' not in f or 'map_0' not in f['offsets']
                or 'offset_coverage' not in f
                or 'map_0' not in f['offset_coverage']):
            raise SystemExit(
                "Anchor requires multi-map schema (offsets/map_0 + "
                f"offset_coverage/map_0); {args.cal} lacks one of these.")
        offsets_m0 = f['offsets/map_0'][:].astype(np.float64)
        cov_m0 = f['offset_coverage/map_0'][:].astype(np.float64)
        if 'zodi_anchor_C' in f.attrs:
            print(f"WARNING: {args.cal} already carries "
                  f"zodi_anchor_C={float(f.attrs['zodi_anchor_C']):.6g}. "
                  f"Re-anchoring stacks on top of that shift.")

    # full_DC[k] = scalar[k] + Σ_c (N_c[k]/N[k]) * offset[k, c]
    N_per_frame = cov_m0.sum(axis=1)
    safe = N_per_frame > 0
    chunk_weighted = np.full_like(frame_scalar, np.nan)
    chunk_weighted[safe] = (
        (offsets_m0[safe] * cov_m0[safe]).sum(axis=1) / N_per_frame[safe])
    full_DC = frame_scalar + chunk_weighted
    print(f"full_DC = frame_scalar + pixel-weighted-mean(offsets):")
    print(f"  frame_scalar:    mean={np.nanmean(frame_scalar):.6g}  "
          f"std={np.nanstd(frame_scalar):.6g}")
    print(f"  chunk-DC term:   mean={np.nanmean(chunk_weighted):.6g}  "
          f"std={np.nanstd(chunk_weighted):.6g}")
    print(f"  full_DC:         mean={np.nanmean(full_DC):.6g}  "
          f"std={np.nanstd(full_DC):.6g}")

    zodi_pred, mjds = load_zodi_pred_and_mjds(args.zodi_pred, cal_reproj_list)
    if len(zodi_pred) != len(frame_scalar):
        raise SystemExit(
            f"zodi_pred length {len(zodi_pred)} != frame_scalar length "
            f"{len(frame_scalar)}")
    if mjds is not None and len(mjds) != len(frame_scalar):
        raise SystemExit(
            f"mjds length {len(mjds)} != frame_scalar length "
            f"{len(frame_scalar)}")

    if mjds is None and args.clip_window_days > 0:
        print("WARNING: .npz has no 'mjds' key — moving sigma-clip disabled. "
              "Regenerate with current build_zodi_predictions.py to enable.")

    # Linfit full_DC vs zodi_pred (full_DC = scalar + pixel-weighted-chunk-mean(offsets))
    slope, intercept, r, inlier = fit_with_clip(
        zodi_pred, full_DC, mjds,
        window_days=args.clip_window_days,
        sigma=args.clip_sigma,
        iters=args.clip_iters)
    n_finite = int((np.isfinite(zodi_pred) & np.isfinite(full_DC)).sum())
    n_used = int(inlier.sum())
    n_outl = n_finite - n_used
    mean_full_dc = float(np.mean(full_DC[inlier]))
    mean_scalar = float(np.mean(frame_scalar[inlier]))
    mean_zodi = float(np.mean(zodi_pred[inlier]))
    C = float(intercept)

    print(f"frames in fit:    {n_used} (rejected {n_outl} outliers via "
          f"{args.clip_window_days}-day MJD window, {args.clip_sigma}sigma, "
          f"{args.clip_iters} iters)")
    print(f"mean(full_DC)      = {mean_full_dc:.6g} MJy/sr  (inliers only)")
    print(f"mean(frame_scalar) = {mean_scalar:.6g} MJy/sr  (inliers only)")
    print(f"mean(zodi_pred)    = {mean_zodi:.6g} MJy/sr  (inliers only)")
    print(f"linfit slope       = {slope:.4f}  (validation: ~1 expected)")
    print(f"linfit intercept   = {intercept:.6g} MJy/sr  <- anchor C")
    print(f"Pearson r          = {r:.4f}")

    warnings = []
    if abs(slope - 1.0) > 0.2:
        warnings.append(
            f"slope {slope:.3f} is far from 1.0 — zodi model may be "
            f"miscapturing per-frame variation amplitude")
    if r < 0.5:
        warnings.append(
            f"Pearson r {r:.3f} is low — frame_scalar may not be "
            f"tracking zodi (bad astrometry? wrong wavelength?)")
    if warnings:
        print()
        print("=" * 60)
        for w in warnings:
            print(f"WARNING: {w}")
        print("Anchor will still be applied with C = intercept, but the")
        print("absolute level is suspect. Inspect compare_zodi_vs_scalar")
        print("output before trusting the result.")
        print("=" * 60)

    cal_out = output_path(args.cal, args.out_suffix, args.out_dir)
    mos_out = output_path(args.mosaic, args.out_suffix, args.out_dir)
    for out in (cal_out, mos_out):
        if os.path.exists(out) and not args.overwrite:
            raise SystemExit(
                f"{out} already exists; pass --overwrite to replace.")
        if os.path.exists(out):
            os.remove(out)

    anchor_cal(args.cal, cal_out, C, zodi_pred, slope, intercept, r,
               mean_scalar, mean_full_dc, mean_zodi,
               n_inliers=n_used, n_outliers=n_outl,
               clip_window_days=args.clip_window_days,
               clip_sigma=args.clip_sigma)
    anchor_mosaic(args.mosaic, mos_out, C, slope, r, mean_zodi)


if __name__ == '__main__':
    main()
