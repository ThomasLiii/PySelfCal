"""Compare two mosaic FITS files HDU-by-HDU within float32 tolerance.

Usage:
    python compare_mosaic.py <baseline.fits> <candidate.fits> [--rtol 1e-4] [--atol 1e-6]

Prints, per shared image HDU, max |a-b|, max relative diff, and the count of
pixels exceeding the float32 tolerance. Exits 0 if every HDU is within tol.

Because the coadd accumulates float32 sums under imap_unordered (flush order is
non-deterministic across runs), even the *baseline against itself* differs at
the ULP level. Compare the candidate-vs-baseline numbers against a
baseline-vs-baseline2 run to confirm the change stays inside that noise floor.
"""
import argparse
import sys

import numpy as np
from astropy.io import fits


def hdu_map(path):
    out = {}
    with fits.open(path) as hdul:
        for h in hdul:
            name = h.header.get('EXTNAME')
            if name and h.data is not None:
                out[name] = h.data.astype(np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--rtol', type=float, default=1e-4)
    ap.add_argument('--atol', type=float, default=1e-6)
    args = ap.parse_args()

    A = hdu_map(args.a)
    B = hdu_map(args.b)
    shared = [k for k in A if k in B]
    print(f"comparing {len(shared)} HDUs (rtol={args.rtol}, atol={args.atol})")
    print(f"{'HDU':18s} {'max|a-b|':>12s} {'max_rel':>12s} {'n_exceed':>10s} {'n_nonzero':>12s}  verdict")

    failures = 0
    for k in shared:
        a, b = A[k], B[k]
        if a.shape != b.shape:
            print(f"{k:18s}  SHAPE MISMATCH {a.shape} vs {b.shape}")
            failures += 1
            continue
        absdiff = np.abs(a - b)
        denom = np.maximum(np.abs(a), np.abs(b))
        nz = denom > 0
        reldiff = np.zeros_like(absdiff)
        reldiff[nz] = absdiff[nz] / denom[nz]
        tol = args.atol + args.rtol * denom
        exceed = absdiff > tol
        n_exceed = int(exceed.sum())
        verdict = 'OK' if n_exceed == 0 else 'EXCEEDS'
        if n_exceed > 0:
            failures += 1
        print(f"{k:18s} {absdiff.max():12.3e} {reldiff.max():12.3e} "
              f"{n_exceed:10d} {int(nz.sum()):12d}  {verdict}")

    if failures == 0:
        print("\nALL HDUs WITHIN TOLERANCE")
        return 0
    print(f"\n{failures} HDU(s) EXCEED TOLERANCE")
    return 1


if __name__ == '__main__':
    sys.exit(main())
