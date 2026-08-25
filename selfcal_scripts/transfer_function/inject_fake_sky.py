#!/usr/bin/env python3
"""Inject a fake sky into real reprojected frames, for a transfer-function run.

Workflow: to measure the selfcal pipeline's transfer function you replace the
real sky in each reprojected frame with a KNOWN fake sky, run the fiducial
calibration + mosaic, and compare the output mosaic to the fake sky you put in.

This script does the injection. For every real reprojected ``.h5`` frame it
writes a copy whose ``sub_data`` dataset is replaced by the matching crop of your
fake sky, leaving EVERYTHING ELSE untouched (``sub_mapping``, ``sub_bitmask``,
``sub_foot``, ``ref_coords``, the WCS headers, and the filename). Because the
pipeline stores ``sub_data`` as the exposure ALREADY reprojected onto the
reference grid -- bbox-cropped to ``ref_coords = [y0, y1, x0, x1]`` -- the
injection is a plain rectangular crop::

    sub_data_new = fake_sky[y0:y1, x0:x1]

with NaN kept wherever the real frame was NaN, so the per-frame footprint
(the region the detector actually observed) is preserved exactly.

Requirements on the fake sky:
  * It is defined on the SAME reference grid as the frames were reprojected onto
    -- i.e. its shape equals the ref.fits image shape for that detector. Pass
    --ref-fits and the script checks this.
  * Same detector as the frames (do not mix detectors).

Reads/writes Zstd-compressed HDF5, so ``hdf5plugin`` MUST be importable (it is a
selfcal dependency). Run under the selfcal env.

Example
-------
    python inject_fake_sky.py \
        --frames-in  /mnt/md124/.../SPHEREx_..._D3_.../reprojected \
        --frames-out /scratch/tf/D3_fakesky_frames \
        --fake-sky   /scratch/tf/fake_sky_D3.npy \
        --ref-fits   /mnt/md124/.../SPHEREx_..._D3_.../ref.fits \
        --workers 16
"""
import argparse
import glob
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import hdf5plugin  # noqa: F401  -- REQUIRED so h5py can read/write the Zstd frames
import numpy as np
from astropy.io import fits


def load_fake_sky(path, ref_shape):
    """Load the fake sky (.npy or .fits) and check it matches the ref grid."""
    if path.endswith((".fits", ".fit", ".fits.gz")):
        sky = fits.getdata(path).astype(np.float32)
    else:
        sky = np.load(path).astype(np.float32)
    if sky.shape != ref_shape:
        raise ValueError(
            f"fake sky shape {sky.shape} != reference grid shape {ref_shape}. "
            "The fake sky must be on the SAME grid the frames were reprojected "
            "onto (the detector's ref.fits image shape).")
    return sky


def _inject_one(args):
    src, dst, fake_sky_path, ref_shape, preserve_footprint = args
    # Load fake sky per-worker (cheap vs. the h5 IO; keeps the pool picklable).
    sky = load_fake_sky(fake_sky_path, ref_shape)
    shutil.copy2(src, dst)  # copy the whole frame (all datasets/attrs, filename)
    with h5py.File(dst, "r+") as f:
        y0, y1, x0, x1 = (int(v) for v in f.attrs["ref_coords"])
        orig = f["sub_data"][:]
        crop = sky[y0:y1, x0:x1].astype(np.float32, copy=True)
        if crop.shape != orig.shape:
            raise ValueError(
                f"{os.path.basename(src)}: crop {crop.shape} != sub_data "
                f"{orig.shape} (ref_coords={[y0, y1, x0, x1]}). The fake sky "
                "grid does not line up with this frame's bbox.")
        if preserve_footprint:
            crop[np.isnan(orig)] = np.nan  # keep the observed footprint exactly
        # In-place write into the existing dataset preserves its shape, dtype,
        # chunking and Zstd compression.
        f["sub_data"][...] = crop
    return os.path.basename(dst)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-in", required=True,
                    help="Directory of REAL reprojected .h5 frames (read-only).")
    ap.add_argument("--frames-out", required=True,
                    help="Output directory for the fake-sky frames (created).")
    ap.add_argument("--fake-sky", required=True,
                    help="Fake sky on the detector's ref grid (.npy or .fits).")
    ap.add_argument("--ref-fits", required=True,
                    help="The detector's ref.fits (defines the ref grid shape).")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--glob", default="*.h5", help="Frame glob (default *.h5).")
    ap.add_argument("--no-preserve-footprint", action="store_true",
                    help="Inject sky into every bbox pixel, including ones the "
                         "detector never observed (NOT recommended -- changes "
                         "coverage and biases the transfer function).")
    args = ap.parse_args()

    ref_shape = fits.getdata(args.ref_fits).shape
    # Validate the fake sky up front (fail fast before copying 900 files).
    load_fake_sky(args.fake_sky, ref_shape)

    src_files = sorted(glob.glob(os.path.join(args.frames_in, args.glob)))
    if not src_files:
        raise SystemExit(f"No frames matched {args.frames_in}/{args.glob}")
    os.makedirs(args.frames_out, exist_ok=True)
    print(f"Injecting fake sky into {len(src_files)} frames "
          f"({args.frames_in} -> {args.frames_out}) with {args.workers} workers...")

    jobs = [(s, os.path.join(args.frames_out, os.path.basename(s)),
             args.fake_sky, ref_shape, not args.no_preserve_footprint)
            for s in src_files]

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_inject_one, j) for j in jobs]
        for fut in as_completed(futs):
            fut.result()
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)}")
    print(f"Done. {len(jobs)} fake-sky frames written to {args.frames_out}")
    print("Filenames are preserved, so the exp/det indices still parse correctly.")


if __name__ == "__main__":
    main()
