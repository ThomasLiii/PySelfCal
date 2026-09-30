"""The row-split parallel transpose product (the default ``A^T @ y``).

It is deliberately NOT bit-identical to the sequential kernel — each column's
sum is a fixed tree instead of one chain — so its contract is different from
the rest of the bit-equality suite:

1. deterministic: two operators built the same way return identical bytes;
2. layout-independent: unified CSR, BlockCSR and ColSplitCSR give identical
   bytes for the same thread count (a thread scatters its rows in row order
   whatever the storage, so every per-thread partial is the same sequence);
3. close to the sequential kernel at float32 reassociation level;
4. ``SELFCAL_PARALLEL_RMATVEC=1`` selects the sequential kernel, which IS
   bit-identical to scipy's CSC product (the byte-exact verification mode).

Runnable as ``python tests/test_rowsplit_rmatvec.py`` or under pytest.
"""
import os

import numpy as np
from scipy.sparse import csr_matrix, random as sprandom

from selfcal.core import solve as SV
from selfcal.core.blockcsr import build_block_csr, partition_block_csr

_SEED = 47


def _ops(A, n_threads, T, rng):
    m, n = A.shape

    def bcsr():
        return build_block_csr(A.data.copy(), A.indices.astype(np.int32),
                               A.indptr.astype(np.int64), (m, n),
                               int(rng.integers(30, 2000)))
    cuts = np.linspace(0, n, T + 1, dtype=np.int64)
    uni = SV._make_parallel_operator(csr_matrix(A, copy=True), n_threads)
    blk = SV._make_parallel_operator_blocks(bcsr(), n_threads, a_owned=False)
    col = SV._make_parallel_operator_colsplit(partition_block_csr(bcsr(), cuts),
                                              n_threads, T)
    return uni, blk, col


def _close(ops):
    for op in ops:
        for name in ('_executor', '_rmatvec_executor', '_rowsplit_executor'):
            ex = getattr(op, name, None)
            if ex is not None:
                ex.shutdown(wait=False)


def test_rowsplit_deterministic_layout_independent_and_close(trials=10):
    rng = np.random.default_rng(_SEED)
    os.environ.pop('SELFCAL_PARALLEL_RMATVEC', None)
    checked = 0
    for _ in range(trials):
        m, n = int(rng.integers(200, 1500)), int(rng.integers(50, 600))
        A = sprandom(m, n, density=float(rng.choice([0.02, 0.1])),
                     random_state=int(rng.integers(1 << 30)), format='csr').astype(np.float32)
        A.sort_indices()
        if A.nnz == 0:
            continue
        y = rng.standard_normal(m).astype(np.float32)
        n_threads, T = int(rng.integers(2, 7)), int(rng.integers(1, 5))
        uni, blk, col = _ops(A, n_threads, T, rng)
        ru, rb, rc = uni.rmatvec(y), blk.rmatvec(y), col.rmatvec(y)
        # 1. deterministic (same operator, repeated; and a fresh build)
        assert np.array_equal(ru.view(np.uint32), uni.rmatvec(y).view(np.uint32))
        uni2, blk2, col2 = _ops(A, n_threads, T, rng)
        assert np.array_equal(rb.view(np.uint32), blk2.rmatvec(y).view(np.uint32))
        assert np.array_equal(rc.view(np.uint32), col2.rmatvec(y).view(np.uint32))
        # 2. layout independent
        assert np.array_equal(ru.view(np.uint32), rb.view(np.uint32)), "unified != BlockCSR"
        assert np.array_equal(ru.view(np.uint32), rc.view(np.uint32)), "unified != ColSplitCSR"
        # 3. close to the exact product
        exact = (A.T @ y.astype(np.float64))
        np.testing.assert_allclose(ru.astype(np.float64), exact, rtol=2e-5, atol=1e-5)
        _close((uni, blk, col, uni2, blk2, col2))
        checked += 1
    assert checked > 0


def test_sequential_mode_is_bitexact_scipy():
    rng = np.random.default_rng(_SEED + 1)
    os.environ['SELFCAL_PARALLEL_RMATVEC'] = '1'
    try:
        for _ in range(6):
            m, n = int(rng.integers(200, 1500)), int(rng.integers(50, 600))
            A = sprandom(m, n, density=0.05, random_state=int(rng.integers(1 << 30)),
                         format='csr').astype(np.float32)
            A.sort_indices()
            y = rng.standard_normal(m).astype(np.float32)
            uni, blk, col = _ops(A, 4, 3, rng)
            ref = (A.T @ y).astype(np.float32)          # scipy's CSC product
            for op in (uni, blk, col):
                assert np.array_equal(op.rmatvec(y).view(np.uint32), ref.view(np.uint32))
            _close((uni, blk, col))
    finally:
        os.environ.pop('SELFCAL_PARALLEL_RMATVEC', None)


if __name__ == '__main__':
    test_rowsplit_deterministic_layout_independent_and_close()
    print("OK row-split rmatvec: deterministic, layout-independent, close to exact")
    test_sequential_mode_is_bitexact_scipy()
    print("OK SELFCAL_PARALLEL_RMATVEC=1: bit-identical to scipy's CSC product")
