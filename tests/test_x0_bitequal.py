"""Bit-equality property test: new compute_x0_scalar_only (CSR-direct) vs the
legacy algorithm (compute_x0_from_Ab + chunk-block zeroing, unchanged in
solution.py).

Covers: f32/f64 data, int32/int64 indices, with/without active_mask, empty
rows, dense/sparse scalar coverage, zero-coverage scalar columns, and forced
multi-chunk selection (via _chunk_target_entries) to prove chunked gathering +
single global bincount reproduces the one-shot path bit-for-bit.

Ported from workspace/memopt/scripts/test_x0_bitequal.py. Runnable as
``python tests/test_x0_bitequal.py`` or under pytest.
"""
import numpy as np
from scipy.sparse import csr_matrix, random as sprandom

from selfcal.core.solution import compute_x0_from_Ab, compute_x0_scalar_only

# Seed preserved verbatim from the source script (deterministic case sweep).
_SEED = 20260718


def legacy_scalar_only(A, b, ref_shape, scalar_col_start, num_sky_blocks=1,
                       active_mask=None):
    x0 = compute_x0_from_Ab(A, b, ref_shape, num_sky_blocks=num_sky_blocks,
                            active_mask=active_mask)
    ref_h, ref_w = ref_shape
    num_sky_eff = num_sky_blocks * ref_h * ref_w
    x0[num_sky_eff:scalar_col_start] = 0.0
    return x0


def build_case(rng, n_rows, ref_shape, num_sky_blocks, n_chunk_cols,
               n_scalar_cols, data_dtype, idx_dtype, use_active_mask,
               density=0.05):
    num_sky = ref_shape[0] * ref_shape[1] * num_sky_blocks
    num_cols_full = num_sky + n_chunk_cols + n_scalar_cols
    A_full = sprandom(n_rows, num_cols_full, density=density, format='csr',
                      random_state=rng.integers(2**31),
                      data_rvs=lambda n: rng.standard_normal(n))
    # kill some rows entirely; leave some scalar columns empty
    kill_rows = rng.integers(0, n_rows, size=max(1, n_rows // 10))
    for r in kill_rows:
        A_full.data[A_full.indptr[r]:A_full.indptr[r + 1]] = 0.0
    A_full.eliminate_zeros()
    A_full.data = A_full.data.astype(data_dtype)
    b = rng.standard_normal(n_rows)  # float64, as in production

    scalar_col_start = num_sky + n_chunk_cols
    if use_active_mask:
        col_nnz = np.bincount(
            A_full.indices, minlength=num_cols_full)
        active_mask = col_nnz > 0
        # legacy compact matrix: drop inactive columns (order-preserving)
        keep = np.flatnonzero(active_mask)
        remap = np.full(num_cols_full, -1)
        remap[keep] = np.arange(keep.size)
        A = csr_matrix((A_full.data, remap[A_full.indices], A_full.indptr),
                       shape=(n_rows, keep.size))
    else:
        active_mask = None
        A = A_full
    A.indices = A.indices.astype(idx_dtype)
    A.indptr = A.indptr.astype(idx_dtype)
    return A, b, scalar_col_start, active_mask


def test_x0_scalar_only_bitequal():
    """Full parameter sweep: 2*2*2*3*3*3 = 216 cases, each asserted bit-equal.

    The nested loop is kept (rather than pytest.parametrize) so a single seeded
    rng advances through every case exactly as the source script did.
    """
    rng = np.random.default_rng(_SEED)
    ref_shape = (7, 9)
    n_cases = 0
    for data_dtype in (np.float32, np.float64):
        for idx_dtype in (np.int32, np.int64):
            for use_mask in (False, True):
                for num_sky_blocks in (1, 2, 4):
                    for n_scalar in (1, 5, 40):
                        for chunk_target in (10**9, 50, 7):
                            n_cases += 1
                            A, b, scs, am = build_case(
                                rng, n_rows=200, ref_shape=ref_shape,
                                num_sky_blocks=num_sky_blocks,
                                n_chunk_cols=30, n_scalar_cols=n_scalar,
                                data_dtype=data_dtype, idx_dtype=idx_dtype,
                                use_active_mask=use_mask)
                            ref = legacy_scalar_only(
                                A, b, ref_shape, scs,
                                num_sky_blocks=num_sky_blocks, active_mask=am)
                            new = compute_x0_scalar_only(
                                A, b, ref_shape, scs,
                                num_sky_blocks=num_sky_blocks, active_mask=am,
                                _chunk_target_entries=chunk_target)
                            assert ref.dtype == new.dtype and ref.shape == new.shape, (
                                f"dtype/shape mismatch "
                                f"dtype={data_dtype.__name__} "
                                f"idx={idx_dtype.__name__} mask={use_mask} "
                                f"J={num_sky_blocks} S={n_scalar} "
                                f"chunk={chunk_target}: "
                                f"ref=({ref.dtype},{ref.shape}) "
                                f"new=({new.dtype},{new.shape})")
                            assert ref.tobytes() == new.tobytes(), (
                                f"x0 not bit-equal "
                                f"dtype={data_dtype.__name__} "
                                f"idx={idx_dtype.__name__} mask={use_mask} "
                                f"J={num_sky_blocks} S={n_scalar} "
                                f"chunk={chunk_target}: "
                                f"{int((ref != new).sum())} of {ref.size} differ")
    # Guard against silently shrinking coverage.
    assert n_cases == 216


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")


if __name__ == '__main__':
    _run_all()
