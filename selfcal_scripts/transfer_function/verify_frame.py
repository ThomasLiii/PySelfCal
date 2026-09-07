#!/usr/bin/env python3
"""Sanity-check a reprojected frame against the schema the selfcal pipeline reads.

Run this on ONE of your fake-sky frames before launching a full run, so a schema
mistake surfaces on 1 file instead of after processing 900. Optionally pass the
matching real frame with --orig to confirm the injection preserved everything
except sub_data (and preserved the footprint).

    python verify_frame.py /scratch/tf/D3_fakesky_frames/exp_000000_det_0.h5 \
        --orig /mnt/md124/.../reprojected/exp_000000_det_0.h5
"""
import argparse
import os

import h5py
import hdf5plugin  # noqa: F401  -- REQUIRED to read the Zstd frames
import numpy as np

REQUIRED_DATASETS = ["sub_data", "sub_mapping"]        # read by the LSQR path
OPTIONAL_DATASETS = ["sub_bitmask", "sub_foot"]        # sub_bitmask read iff apply_mask
REQUIRED_ATTRS = ["ref_coords"]                        # sub_header/det_header used by wav path


def check(path, orig=None):
    ok = True
    with h5py.File(path, "r") as f:
        keys, attrs = set(f.keys()), set(f.attrs.keys())
        for d in REQUIRED_DATASETS:
            present = d in keys
            ok &= present
            print(f"  [{'OK' if present else 'MISSING'}] dataset {d}")
        for d in OPTIONAL_DATASETS:
            print(f"  [{'ok' if d in keys else '--'}] dataset {d} (optional)")
        for a in REQUIRED_ATTRS:
            present = a in attrs
            ok &= present
            print(f"  [{'OK' if present else 'MISSING'}] attr {a}")

        rc = [int(v) for v in f.attrs["ref_coords"]]
        y0, y1, x0, x1 = rc
        sd = f["sub_data"][:]
        match = sd.shape == (y1 - y0, x1 - x0)
        ok &= match
        print(f"  [{'OK' if match else 'BAD'}] sub_data shape {sd.shape} == "
              f"bbox {(y1 - y0, x1 - x0)} from ref_coords {rc}")
        # filename must parse to exp/det indices
        base = os.path.basename(path)
        try:
            exp, det = int(base.split("_")[1]), int(base.split("_")[3].removesuffix(".h5"))
            print(f"  [OK] filename parses: exp={exp} det={det}")
        except Exception:
            ok = False
            print(f"  [BAD] filename {base} does not parse as exp_<n>_det_<n>.h5")
        nan_frac = float(np.isnan(sd).mean())
        print(f"  [info] sub_data NaN fraction (footprint): {nan_frac:.3f}")

        if orig is not None:
            with h5py.File(orig, "r") as g:
                # footprint preserved?
                osd = g["sub_data"][:]
                same_fp = np.array_equal(np.isnan(osd), np.isnan(sd))
                ok &= same_fp
                print(f"  [{'OK' if same_fp else 'BAD'}] footprint (NaN pattern) "
                      "matches the real frame")
                # everything except sub_data unchanged? (sub_mapping/sub_foot
                # carry NaN in the unobserved footprint, so compare NaN-aware:
                # equal_nan for float dtypes, plain equality for integer masks.)
                def _eq(a, b):
                    if np.issubdtype(a.dtype, np.floating):
                        return np.array_equal(a, b, equal_nan=True)
                    return np.array_equal(a, b)
                for d in ["sub_mapping", "sub_bitmask", "sub_foot"]:
                    if d in g and d in f:
                        same = _eq(g[d][:], f[d][:])
                        ok &= same
                        print(f"  [{'OK' if same else 'BAD'}] {d} unchanged vs real frame")
                same_rc = [int(v) for v in g.attrs["ref_coords"]] == rc
                ok &= same_rc
                print(f"  [{'OK' if same_rc else 'BAD'}] ref_coords unchanged")
                changed = not np.array_equal(np.nan_to_num(osd), np.nan_to_num(sd))
                print(f"  [{'OK' if changed else 'WARN'}] sub_data differs from "
                      "real frame (injection actually happened)")
    print("RESULT:", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("frame")
    ap.add_argument("--orig", default=None,
                    help="Matching real frame, to confirm only sub_data changed.")
    args = ap.parse_args()
    raise SystemExit(0 if check(args.frame, args.orig) else 1)
