"""Write anchored copies of mosaics for ds9 / publication / hand-off.

The anchor is normally applied at read time (selfcal.zodi_anchor.load_anchor
/ load_anchored_mosaic) so the pipeline mosaics stay pristine. This script
is the OPT-IN path for when you want a materialized FITS that already has
the anchor C baked into MEAN_MAP / SC_MEAN_MAP — e.g. to drop into ds9 or
share with a collaborator who doesn't have the anchor file.

Reads a run's PRISTINE mosaics + its per-detector anchor file, writes
anchored copies to <run>/anchored_mosaics/ (never overwrites the
pipeline mosaic). Each shifted HDU gets a ZODIANCH header.

    python materialize_anchored_mosaic.py --run-dir /mnt/.../D1_...
    python materialize_anchored_mosaic.py --run-dir /mnt/.../D1_... --channels 31 32 33

Reads only the anchor file + mosaics (no cal/npz). Runs in the selfcal env.
"""
import argparse
import glob
import os
import re
import shutil
import sys

from astropy.io import fits

from selfcal.zodi_anchor import load_anchor


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--run-dir', nargs='+', required=True,
                   help='Run dir(s) with mosaic/ + zodi_anchor/anchor_D{N}.h5.')
    p.add_argument('--out-dir', default=None,
                   help='Override output dir. Default: <run>/anchored_mosaics/.')
    p.add_argument('--channels', type=int, nargs='+', default=None,
                   help='Only these channels (default: all mosaics present).')
    p.add_argument('--overwrite', action='store_true',
                   help='Overwrite existing anchored copies.')
    return p.parse_args()


def parse_detector(name):
    m = re.search(r'Detector(\d+)_', os.path.basename(name))
    return int(m.group(1)) if m else None


def parse_channel(name):
    m = re.search(r'_Ch(\d+)_', os.path.basename(name))
    return int(m.group(1)) if m else None


def materialize_run(run_dir, out_dir, channels, overwrite):
    mdir = os.path.join(run_dir, 'mosaic')
    mosaics = sorted(glob.glob(os.path.join(mdir, 'mosaic_*.fits')))
    if not mosaics:
        print(f"  no mosaics in {mdir}", file=sys.stderr)
        return
    detector = parse_detector(mosaics[0])
    anchor_path = os.path.join(run_dir, 'zodi_anchor', f'anchor_D{detector}.h5')
    if not os.path.exists(anchor_path):
        print(f"  no anchor file {anchor_path}", file=sys.stderr)
        return
    anchor = load_anchor(anchor_path)
    od = out_dir or os.path.join(run_dir, 'anchored_mosaics')
    os.makedirs(od, exist_ok=True)
    print(f"  detector D{detector}, anchor {anchor} -> {od}")

    n_done = n_skip = 0
    for mos in mosaics:
        ch = parse_channel(mos)
        if channels is not None and ch not in channels:
            continue
        if ch not in anchor.channels:
            print(f"    Ch{ch}: not in anchor file; skipping")
            n_skip += 1
            continue
        out = os.path.join(od, os.path.basename(mos).replace(
            '.fits', '_zodianch.fits'))
        if os.path.exists(out) and not overwrite:
            print(f"    Ch{ch}: exists (use --overwrite): {out}")
            n_skip += 1
            continue
        shutil.copyfile(mos, out)
        with fits.open(out, mode='update') as hdul:
            shifted = anchor.apply_to_mosaic_hdul(hdul, ch)
            hdul[0].header['ZODIANCH'] = (float(anchor.C(ch)),
                                          'Zodi anchor C added (MJy/sr)')
            hdul[0].header['ZODIMETH'] = (anchor.channels[ch].get(
                'smooth_method', 'raw'), 'Anchor method for this channel')
        n_done += 1
        print(f"    Ch{ch}: C={anchor.C(ch):+.4g} -> {os.path.basename(out)} "
              f"(shifted {', '.join(shifted)})")
    print(f"  wrote {n_done}, skipped {n_skip}")


def main():
    args = parse_args()
    for run_dir in args.run_dir:
        print(f"=== {run_dir} ===")
        materialize_run(run_dir, args.out_dir, args.channels, args.overwrite)


if __name__ == '__main__':
    main()
