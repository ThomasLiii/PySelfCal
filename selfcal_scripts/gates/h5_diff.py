"""Generic HDF5 byte-equality: every dataset (shape, dtype, raw bytes) and every attribute of
both files must match. Usage: h5_diff.py A.h5 B.h5 [--rounding-ok REGEX[,REGEX] [--tol 1e-12]]
-> per-dataset verdicts, then 'ALL DATASETS BYTE-EQUAL', 'ALL DATASETS EQUAL (k WITHIN ROUNDING)'
or 'N DATASETS DIFFER'. Datasets whose name matches a --rounding-ok regex may differ by at most
--tol (absolute) — for the per-pixel moment folds (Fisher / separability / closed-form sky) that
the assembly accumulates in batch-completion order (rounding-level run-to-run noise on a busy
box). Exit 0 iff no dataset differs beyond that."""
import re
import sys
import numpy as np
import h5py
import hdf5plugin  # noqa: F401  (zstd-compressed datasets)

def walk(f):
    out = {}
    def visit(name, obj):
        if isinstance(obj, h5py.Dataset):
            out[name] = obj
    f.visititems(visit)
    return out

IGNORE_ATTRS = {'sky_cal'}      # name-bearing attrs (paths of sibling products) differ between runs by design


def attrs_equal(a, b):
    a = {k: v for k, v in a.items() if k not in IGNORE_ATTRS}
    b = {k: v for k, v in b.items() if k not in IGNORE_ATTRS}
    if set(a) != set(b):
        return False
    for k in a:
        x, y = a[k], b[k]
        try:
            if isinstance(x, np.ndarray) or isinstance(y, np.ndarray):
                if not (np.asarray(x).shape == np.asarray(y).shape and np.asarray(x).tobytes() == np.asarray(y).tobytes()):
                    return False
            elif x != y:
                return False
        except Exception:
            return False
    return True

def main(pa, pb, rounding_ok=(), tol=1e-12):
    bad = 0
    rounding = 0
    with h5py.File(pa, 'r') as fa, h5py.File(pb, 'r') as fb:
        da, db = walk(fa), walk(fb)
        if set(da) != set(db):
            print('dataset sets differ:', sorted(set(da) ^ set(db))); bad += 1
        for k in sorted(set(da) & set(db)):
            A, B = da[k], db[k]
            if A.shape != B.shape or A.dtype != B.dtype:
                print(f'{k}: shape/dtype differ {A.shape}/{A.dtype} vs {B.shape}/{B.dtype}'); bad += 1; continue
            x, y = A[()], B[()]
            same = np.asarray(x).tobytes() == np.asarray(y).tobytes() if A.dtype.kind != 'O' else list(x) == list(y)
            if not same:
                if A.dtype.kind in 'fiu':
                    d = np.asarray(x, dtype=np.float64) - np.asarray(y, dtype=np.float64)
                    n, mx = int(np.count_nonzero(d)), float(np.nanmax(np.abs(d)))
                    if any(re.fullmatch(r, k) for r in rounding_ok) and mx <= tol:
                        rounding += 1
                        print(f'{k}: equal within rounding  n={n} max|d|={mx:.3e}')
                    else:
                        bad += 1
                        print(f'{k}: DIFFERS  n={n} max|d|={mx:.3e}')
                else:
                    bad += 1
                    print(f'{k}: DIFFERS')
            if not attrs_equal(dict(A.attrs), dict(B.attrs)):
                print(f'{k}: attrs differ'); bad += 1
        if not attrs_equal(dict(fa.attrs), dict(fb.attrs)):
            print('root attrs differ:', dict(fa.attrs), dict(fb.attrs)); bad += 1
    if bad:
        print(f'{bad} DATASETS DIFFER')
    elif rounding:
        print(f'ALL DATASETS EQUAL ({rounding} WITHIN ROUNDING, tol {tol:g})')
    else:
        print('ALL DATASETS BYTE-EQUAL')
    return 0 if bad == 0 else 1

if __name__ == '__main__':
    args = sys.argv[1:]
    rounding_ok, tol = (), 1e-12
    if '--rounding-ok' in args:
        i = args.index('--rounding-ok'); rounding_ok = tuple(args[i + 1].split(',')); del args[i:i + 2]
    if '--tol' in args:
        i = args.index('--tol'); tol = float(args[i + 1]); del args[i:i + 2]
    sys.exit(main(args[0], args[1], rounding_ok, tol))
