"""Byte-equality diff for two cal_*.h5 files.

Usage:
    python diff_cal_h5.py <baseline.h5> <candidate.h5>

Exits 0 if every dataset matches element-wise; nonzero otherwise.
"""
import sys
import h5py
import numpy as np


def diff(a_path, b_path):
    failures = 0
    with h5py.File(a_path, 'r') as A, h5py.File(b_path, 'r') as B:
        keys_a = set()
        keys_b = set()
        A.visit(keys_a.add)
        B.visit(keys_b.add)
        all_keys = sorted(keys_a | keys_b)
        for k in all_keys:
            if k not in keys_a:
                print(f'MISSING in baseline: {k}')
                failures += 1
                continue
            if k not in keys_b:
                print(f'MISSING in candidate: {k}')
                failures += 1
                continue
            if not isinstance(A[k], h5py.Dataset):
                continue
            try:
                np.testing.assert_array_equal(A[k][...], B[k][...])
                print(f'OK   {k:30s}  shape={A[k].shape}  dtype={A[k].dtype}')
            except AssertionError as e:
                print(f'DIFF {k}:')
                a_arr = A[k][...]
                b_arr = B[k][...]
                if a_arr.shape != b_arr.shape:
                    print(f'  shape mismatch: {a_arr.shape} vs {b_arr.shape}')
                else:
                    diff_mask = a_arr != b_arr
                    n_diff = int(diff_mask.sum())
                    print(f'  {n_diff} differing elements out of {a_arr.size}')
                    if n_diff > 0 and n_diff <= 20:
                        idx = np.argwhere(diff_mask)
                        for i in idx[:20]:
                            print(f'    at {tuple(i)}: {a_arr[tuple(i)]} vs {b_arr[tuple(i)]}')
                    elif n_diff > 0:
                        max_abs = float(np.max(np.abs(a_arr[diff_mask] - b_arr[diff_mask])))
                        print(f'  max |a-b| = {max_abs:.3e}')
                failures += 1
    if failures == 0:
        print(f'\nALL {len(all_keys)} DATASETS BYTE-EQUAL')
        return 0
    print(f'\n{failures} DATASETS DIFFER')
    return 1


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(diff(sys.argv[1], sys.argv[2]))
