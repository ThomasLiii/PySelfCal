"""CLI wrapper around ``SelfCal.ZodiAnchor.apply_anchor_to_file``.

Applies a post-hoc zodi anchor to a cal_*.h5 + matching mosaic_*.fits.

See ``SelfCal/ZodiAnchor.py`` for the math. In short:
  - full_DC = frame_scalar + Σ_c (N_c[k] / N[k]) * offset[k, c]
  - linfit full_DC = slope * zodi_pred + intercept (with moving sigma-clip)
  - C = intercept
  - sky += C, frame_scalar -= C; MEAN_MAP + SC_MEAN_MAP += C

Originals are never mutated; anchored copies are written alongside via
``--out-suffix`` or in a separate dir via ``--out-dir``.
"""
import argparse

from SelfCal.ZodiAnchor import apply_anchor_to_file


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
                        'num_frames, units MJy/sr, ordered to match the '
                        'cal file\'s reproj_list).')
    p.add_argument('--out-suffix', default='_zodianch',
                   help='Suffix for output files (default: _zodianch).')
    p.add_argument('--out-dir', default=None,
                   help='Override directory for anchored outputs. '
                        'Default: write alongside each input file.')
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing output files.')
    p.add_argument('--clip-window-days', type=float, default=7.0,
                   help='Moving sigma-clip window width in MJD days '
                        '(default: 7). Set 0 to disable clipping.')
    p.add_argument('--clip-sigma', type=float, default=3.0,
                   help='Sigma threshold for the moving clip (MAD-based, '
                        'default: 3.0).')
    p.add_argument('--clip-iters', type=int, default=2,
                   help='(clip, refit) iterations (default: 2).')
    return p.parse_args()


def main():
    args = parse_args()
    result = apply_anchor_to_file(
        cal_in=args.cal, mosaic_in=args.mosaic,
        zodi_pred_npz=args.zodi_pred,
        out_dir=args.out_dir, out_suffix=args.out_suffix,
        clip_window_days=args.clip_window_days,
        clip_sigma=args.clip_sigma,
        clip_iters=args.clip_iters,
        overwrite=args.overwrite,
    )
    print(f"frames in fit:    {result['n_inliers']} "
          f"(rejected {result['n_outliers']} outliers via "
          f"{args.clip_window_days}-day MJD window, "
          f"{args.clip_sigma}sigma, {args.clip_iters} iters)")
    print(f"mean(full_DC)      = {result['mean_full_dc']:.6g} MJy/sr (inliers)")
    print(f"mean(frame_scalar) = {result['mean_scalar']:.6g} MJy/sr (inliers)")
    print(f"mean(zodi_pred)    = {result['mean_zodi']:.6g} MJy/sr (inliers)")
    print(f"linfit slope       = {result['slope']:.4f}  (~1 expected)")
    print(f"linfit intercept   = {result['intercept']:.6g} MJy/sr  <- anchor C")
    print(f"Pearson r          = {result['r']:.4f}")
    if abs(result['slope'] - 1.0) > 0.2 or result['r'] < 0.5:
        print()
        print("=" * 60)
        if abs(result['slope'] - 1.0) > 0.2:
            print(f"WARNING: slope {result['slope']:.3f} is far from 1.0 "
                  "(model amplitude may be off)")
        if result['r'] < 0.5:
            print(f"WARNING: Pearson r {result['r']:.3f} is low "
                  "(astrometry / wavelength / model issue?)")
        print("=" * 60)
    print(f"Wrote anchored cal:    {result['cal_out']}")
    if result['mosaic_out']:
        print(f"Wrote anchored mosaic: {result['mosaic_out']}")
        print(f"  shifted HDUs: {result['shifted_extnames']}")


if __name__ == '__main__':
    main()
