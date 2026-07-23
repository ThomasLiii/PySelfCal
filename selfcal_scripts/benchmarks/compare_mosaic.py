"""Compare two mosaic FITS files HDU-by-HDU within float32 tolerance.

Usage:
    python compare_mosaic.py <baseline.fits> <candidate.fits> [--rtol 1e-4] [--atol 1e-6]
        [--meaningful-thr FLOAT]
        [--meaningful-extname-thr EXTNAME=FLOAT [EXTNAME=FLOAT ...]]

Prints, per shared image HDU, max |a-b|, max relative diff, and the count of
pixels exceeding the float32 tolerance. Exits 0 if every HDU is within tol.

Because the coadd accumulates float32 sums under imap_unordered (flush order is
non-deterministic across runs), even the *baseline against itself* differs at
the ULP level. Compare the candidate-vs-baseline numbers against a
baseline-vs-baseline2 run to confirm the change stays inside that noise floor.

Meaningful-pixel mode
---------------------
The default per-HDU summary reports diffs over *every* pixel, including
near-zero baseline pixels where any absolute diff is large *relative* to a
near-zero baseline. That tends to flag harmless noise as EXCEEDS. A more
robust gating criterion is to restrict each HDU's diff to pixels where
``|baseline| > thr`` (baseline = the FIRST argument, ``a``).

When ``--meaningful-thr`` or ``--meaningful-extname-thr`` is provided, a SECOND
summary block is emitted with that restriction applied per HDU. Different HDUs
have different physical scales (MEAN_MAP ~1e-3, STD_MAP ~1e-3..1e-5,
SC_MEAN_MAP ~1e-3, WAV_* ~µm), so per-HDU overrides exist via
``--meaningful-extname-thr``. Per-HDU overrides take precedence over the
global ``--meaningful-thr`` for the matching HDU. If neither flag is set,
only the full-pixel summary is emitted.

Examples:
    # global threshold across every HDU
    python compare_mosaic.py a.fits b.fits --meaningful-thr 1e-3

    # per-HDU thresholds; STD_MAP gets a tighter cut, others fall through to global
    python compare_mosaic.py a.fits b.fits --meaningful-thr 1e-3 \\
        --meaningful-extname-thr STD_MAP=1e-5 WAV_STD=1e-5
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


def _parse_extname_thr(values):
    """Parse list of EXTNAME=FLOAT tokens into {EXTNAME: thr}."""
    out = {}
    if not values:
        return out
    for tok in values:
        if '=' not in tok:
            raise SystemExit(f"--meaningful-extname-thr expects EXTNAME=FLOAT, got: {tok!r}")
        name, val = tok.split('=', 1)
        try:
            out[name] = float(val)
        except ValueError:
            raise SystemExit(f"--meaningful-extname-thr value not a float: {tok!r}")
    return out


def _summarize_unrestricted(A, B, shared, rtol, atol):
    """Original full-pixel summary (kept for backward compatibility)."""
    print(f"{'HDU':18s} {'max|a-b|':>12s} {'max_rel':>12s} "
          f"{'n_exceed':>10s} {'n_nonzero':>12s}  verdict")
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
        tol = atol + rtol * denom
        exceed = absdiff > tol
        n_exceed = int(exceed.sum())
        verdict = 'OK' if n_exceed == 0 else 'EXCEEDS'
        if n_exceed > 0:
            failures += 1
        print(f"{k:18s} {absdiff.max():12.3e} {reldiff.max():12.3e} "
              f"{n_exceed:10d} {int(nz.sum()):12d}  {verdict}")
    return failures


def _summarize_meaningful(A, B, shared, rtol, atol, thr_for):
    """Per-HDU summary restricted to |baseline|>thr; thr_for(name) -> thr or None.

    HDUs whose thr_for returns None are skipped.
    """
    print(f"{'HDU':18s} {'max|a-b|':>12s} {'max_rel':>12s} "
          f"{'n_exceed':>10s} {'n_meaningful':>13s}  verdict")
    failures = 0
    for k in shared:
        a, b = A[k], B[k]
        if a.shape != b.shape:
            print(f"{k:18s}  SHAPE MISMATCH {a.shape} vs {b.shape}")
            failures += 1
            continue
        thr = thr_for(k)
        if thr is None:
            print(f"{k:18s}  (no threshold set; skipped)")
            continue
        mask = np.abs(a) > thr
        n_meaningful = int(mask.sum())
        if n_meaningful == 0:
            print(f"{k:18s}  (0 pixels survive |a|>{thr:g}; skipped)")
            continue
        am, bm = a[mask], b[mask]
        absdiff = np.abs(am - bm)
        denom = np.maximum(np.abs(am), np.abs(bm))
        nz = denom > 0
        reldiff = np.zeros_like(absdiff)
        reldiff[nz] = absdiff[nz] / denom[nz]
        tol = atol + rtol * denom
        exceed = absdiff > tol
        n_exceed = int(exceed.sum())
        verdict = 'OK' if n_exceed == 0 else 'EXCEEDS'
        if n_exceed > 0:
            failures += 1
        print(f"{k:18s} {absdiff.max():12.3e} {reldiff.max():12.3e} "
              f"{n_exceed:10d} {n_meaningful:13d}  {verdict}")
    return failures


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--rtol', type=float, default=1e-4)
    ap.add_argument('--atol', type=float, default=1e-6)
    ap.add_argument('--meaningful-thr', type=float, default=None,
                    help='Global |baseline|>thr restriction for the second summary block.')
    ap.add_argument('--meaningful-extname-thr', nargs='*', default=None,
                    help='Per-HDU thresholds as EXTNAME=FLOAT; overrides --meaningful-thr.')
    args = ap.parse_args()

    A = hdu_map(args.a)
    B = hdu_map(args.b)
    shared = [k for k in A if k in B]
    print(f"comparing {len(shared)} HDUs (rtol={args.rtol}, atol={args.atol})")

    # ---- Block 1: existing behavior, every pixel ----
    failures = _summarize_unrestricted(A, B, shared, args.rtol, args.atol)

    # ---- Block 2: meaningful-pixel restriction (opt-in) ----
    per_hdu = _parse_extname_thr(args.meaningful_extname_thr)
    if args.meaningful_thr is not None or per_hdu:
        print("\n" + "=" * 78)
        print("MEANINGFUL-PIXEL RESTRICTION  (mask = |baseline|>thr; baseline=arg `a`)")
        if args.meaningful_thr is not None:
            print(f"  global --meaningful-thr = {args.meaningful_thr:g}")
        for name, thr in per_hdu.items():
            print(f"  per-HDU {name} = {thr:g}")
        print("=" * 78)

        def thr_for(name):
            if name in per_hdu:
                return per_hdu[name]
            return args.meaningful_thr  # may be None → HDU is skipped

        # Meaningful-block failures count toward the exit status — they are
        # the intended gate. Full-pixel (block 1) failures also still count,
        # so invocations without the meaningful flags keep their original
        # pass/fail semantics.
        failures += _summarize_meaningful(A, B, shared, args.rtol, args.atol, thr_for)

    if failures == 0:
        print("\nALL HDUs WITHIN TOLERANCE")
        return 0
    print(f"\n{failures} HDU(s) EXCEED TOLERANCE")
    return 1


if __name__ == '__main__':
    sys.exit(main())
