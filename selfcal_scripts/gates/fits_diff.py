"""Byte-equality of two multi-extension FITS mosaics, extension by extension.
Exit 0 when every extension's data are identical (dtype, shape, bytes); 1 otherwise.
Usage: fits_diff.py <a.fits> <b.fits>"""
import sys
import numpy as np
from astropy.io import fits

a_path, b_path = sys.argv[1:3]
ok = True
with fits.open(a_path) as A, fits.open(b_path) as B:
    names_a = [h.name for h in A]; names_b = [h.name for h in B]
    if names_a != names_b:
        print(f"EXTENSIONS DIFFER: {names_a} vs {names_b}"); ok = False
    for name in names_a:
        if name not in names_b:
            continue
        da, db = A[name].data, B[name].data
        if da is None and db is None:
            print(f"{name:20s} (no data)"); continue
        same = (da is not None and db is not None and da.dtype == db.dtype and da.shape == db.shape
                and np.array_equal(da.view(np.uint8), db.view(np.uint8)))
        if same:
            print(f"{name:20s} IDENTICAL")
        else:
            ok = False
            if da is not None and db is not None and da.shape == db.shape:
                d = np.abs(da.astype(np.float64) - db.astype(np.float64))
                print(f"{name:20s} DIFFER  max|d|={np.nanmax(d):.4g}  n(d!=0)={int(np.count_nonzero(d))}")
            else:
                print(f"{name:20s} DIFFER  shape/dtype {getattr(da,'shape',None)}/{getattr(da,'dtype',None)} vs {getattr(db,'shape',None)}/{getattr(db,'dtype',None)}")
print("ALL EXTENSIONS BYTE-EQUAL" if ok else "MOSAIC DIFFERS")
sys.exit(0 if ok else 1)
