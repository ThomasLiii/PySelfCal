"""Apply a post-hoc zodi anchor (mean-only global shift) to a SelfCal cal
file and its matching mosaic FITS.

The LSQR solve has one global additive degeneracy: ``sky -> sky + C`` and
``scalar[k] -> scalar[k] - C`` (same ``C`` for all frames) leaves the
data model unchanged. This script picks ``C`` so that the post-shift
``mean(frame_scalar)`` equals a user-supplied ``mean(zodi_pred)`` and
applies it uniformly to the cal file (``skymap += C``, ``frame_scalar -=
C``) and the mosaic (``MEAN_MAP += C``, ``SC_MEAN_MAP += C``).

Originals are never mutated; anchored copies are written alongside with
``--out-suffix``.
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
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing output files')
    return p.parse_args()


def output_path(in_path, suffix):
    root, ext = os.path.splitext(in_path)
    return f'{root}{suffix}{ext}'


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


def anchor_cal(cal_in, cal_out, C, zodi_pred, mean_zodi):
    shutil.copyfile(cal_in, cal_out)
    with h5py.File(cal_out, 'r+') as f:
        sky = f['skymap']
        sky[...] = sky[...] + C
        fs = f['frame_scalar']
        fs[...] = fs[...] - C
        f.attrs['zodi_anchor_C'] = float(C)
        f.attrs['zodi_anchor_mean_pred'] = float(mean_zodi)
        if 'zodi_anchor_pred' in f:
            del f['zodi_anchor_pred']
        f.create_dataset('zodi_anchor_pred',
                         data=zodi_pred.astype(np.float32),
                         compression='gzip')
    print(f"Wrote anchored cal:    {cal_out}")


def anchor_mosaic(mos_in, mos_out, C, mean_zodi):
    shutil.copyfile(mos_in, mos_out)
    shifted = []
    with fits.open(mos_out, mode='update') as hdul:
        hdul[0].header['ZODIANCH'] = (float(C), 'Zodi-anchor shift (MJy/sr)')
        hdul[0].header['ZODIMEAN'] = (float(mean_zodi),
                                      'Mean predicted zodi (MJy/sr)')
        for hdu in hdul[1:]:
            extname = hdu.header.get('EXTNAME', '')
            if extname in SHIFTED_EXTNAMES and hdu.data is not None:
                hdu.data += np.array(C, dtype=hdu.data.dtype)
                hdu.header['ZODIANCH'] = (float(C),
                                          'Zodi-anchor shift (MJy/sr)')
                hdu.header['ZODIMEAN'] = (float(mean_zodi),
                                          'Mean predicted zodi (MJy/sr)')
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
        if 'zodi_anchor_C' in f.attrs:
            print(f"WARNING: {args.cal} already carries "
                  f"zodi_anchor_C={float(f.attrs['zodi_anchor_C']):.6g}. "
                  f"Re-anchoring stacks on top of that shift.")

    zodi_pred = load_zodi_pred(args.zodi_pred, cal_reproj_list)
    if len(zodi_pred) != len(frame_scalar):
        raise SystemExit(
            f"zodi_pred length {len(zodi_pred)} != frame_scalar length "
            f"{len(frame_scalar)}")

    mean_scalar = float(np.mean(frame_scalar))
    mean_zodi = float(np.mean(zodi_pred))
    C = mean_scalar - mean_zodi
    print(f"mean(frame_scalar) = {mean_scalar:.6g} MJy/sr")
    print(f"mean(zodi_pred)    = {mean_zodi:.6g} MJy/sr")
    print(f"shift C = mean(frame_scalar) - mean(zodi_pred) = "
          f"{C:.6g} MJy/sr")

    cal_out = output_path(args.cal, args.out_suffix)
    mos_out = output_path(args.mosaic, args.out_suffix)
    for out in (cal_out, mos_out):
        if os.path.exists(out) and not args.overwrite:
            raise SystemExit(
                f"{out} already exists; pass --overwrite to replace.")
        if os.path.exists(out):
            os.remove(out)

    anchor_cal(args.cal, cal_out, C, zodi_pred, mean_zodi)
    anchor_mosaic(args.mosaic, mos_out, C, mean_zodi)


if __name__ == '__main__':
    main()
