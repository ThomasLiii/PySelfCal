"""Byte-equality diff for two cal_*.h5 files (schema-aware).

Usage:
    python diff_cal_h5.py <baseline.h5> <candidate.h5>

Handles both schemas:
  - Legacy: top-level ``offset``, ``offset_coverage``, ``offset_coverage_frac``.
  - Multi-chunk-map (Commit 3+): ``offsets/map_m``, ``offset_coverage/map_m``,
    ``offset_coverage_frac/map_m``, plus optional top-level ``frame_scalar``
    and ``chunk_maps/map_m``.

When one side is legacy and the other is new, the per-map arrays
(``offsets/map_0`` + ``frame_scalar`` if present) are compared against the
legacy single ``offset``. Coverage arrays are compared the same way.

Exits 0 if every dataset matches element-wise; nonzero otherwise.
"""
import sys

import h5py
import numpy as np


def _has_new_schema(f):
    return 'offsets' in f


def _read_offset(f, m=0):
    """Return per-frame offset for map m (with frame_scalar baked in for m=0)."""
    if _has_new_schema(f):
        off = f['offsets'][f'map_{m}'][...]
        if m == 0 and 'frame_scalar' in f:
            off = off + f['frame_scalar'][...][:, None]
        return off
    if m != 0:
        raise KeyError(f"legacy schema only has map 0 (requested map_{m})")
    return f['offset'][...]


def _read_coverage(f, kind, m=0):
    """kind in {'offset_coverage', 'offset_coverage_frac'}."""
    if _has_new_schema(f):
        return f[kind][f'map_{m}'][...]
    return f[kind][...]


def _num_maps(f):
    if not _has_new_schema(f):
        return 1
    return int(f.attrs.get('num_maps', len(f['offsets'])))


def _diff_array(label, a_arr, b_arr):
    try:
        np.testing.assert_array_equal(a_arr, b_arr)
        print(f'OK   {label:30s}  shape={a_arr.shape}  dtype={a_arr.dtype}')
        return 0
    except AssertionError:
        print(f'DIFF {label}:')
        if a_arr.shape != b_arr.shape:
            print(f'  shape mismatch: {a_arr.shape} vs {b_arr.shape}')
        else:
            diff_mask = a_arr != b_arr
            n_diff = int(diff_mask.sum())
            print(f'  {n_diff} differing elements out of {a_arr.size}')
            if 0 < n_diff <= 20:
                idx = np.argwhere(diff_mask)
                for i in idx[:20]:
                    print(f'    at {tuple(i)}: {a_arr[tuple(i)]} vs {b_arr[tuple(i)]}')
            elif n_diff > 0:
                max_abs = float(np.max(np.abs(a_arr[diff_mask] - b_arr[diff_mask])))
                print(f'  max |a-b| = {max_abs:.3e}')
        return 1


def diff(a_path, b_path):
    failures = 0
    with h5py.File(a_path, 'r') as A, h5py.File(b_path, 'r') as B:
        # Top-level always-present datasets.
        for k in ('skymap', 'skymap_coverage', 'reproj_list'):
            failures += _diff_array(k, A[k][...], B[k][...])

        # Number of maps must agree across schemas.
        K_a, K_b = _num_maps(A), _num_maps(B)
        if K_a != K_b:
            print(f'DIFF num_maps: {K_a} vs {K_b}')
            failures += 1
        K = min(K_a, K_b)

        for m in range(K):
            failures += _diff_array(f'offset[map_{m}]', _read_offset(A, m), _read_offset(B, m))
            failures += _diff_array(f'offset_coverage[map_{m}]',
                                    _read_coverage(A, 'offset_coverage', m),
                                    _read_coverage(B, 'offset_coverage', m))
            failures += _diff_array(f'offset_coverage_frac[map_{m}]',
                                    _read_coverage(A, 'offset_coverage_frac', m),
                                    _read_coverage(B, 'offset_coverage_frac', m))

            # Per-map chunk_map dataset is only present in the new schema; skip
            # when either side lacks it.
            if _has_new_schema(A) and _has_new_schema(B) and 'chunk_maps' in A and 'chunk_maps' in B:
                failures += _diff_array(f'chunk_maps[map_{m}]',
                                        A['chunk_maps'][f'map_{m}'][...],
                                        B['chunk_maps'][f'map_{m}'][...])

    if failures == 0:
        print('\nALL DATASETS BYTE-EQUAL')
        return 0
    print(f'\n{failures} DATASETS DIFFER')
    return 1


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(diff(sys.argv[1], sys.argv[2]))
