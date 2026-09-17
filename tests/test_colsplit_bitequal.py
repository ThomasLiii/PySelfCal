"""Bit-equality tests for the column-partitioned (ColSplitCSR) path.

``setup_lsqr`` can emit the design matrix already split into storage blocks x
column ranges — the layout the bit-equal parallel SpMV consumes. Everything
downstream must see exactly the bytes the unified / BlockCSR path produces:

1. placement: one batch laid out by ``_scatter_one_batch`` == the stable
   sort-by-row + within-row-cumcount placement it replaced, and == scipy's
   own COO->CSR;
2. partitioned placement: every (block, range) piece == the unified matrix
   restricted to those rows and columns, entry for entry;
3. preconditioner: column square norms from ``_iter_range_pieces_colsplit``
   == the unified chunk loop, bitwise, at arbitrary chunk sizes;
4. consumers: ``compute_x0_scalar_only``, the SpMV operator, and full
   ``apply_lsqr`` solves agree bit for bit with the BlockCSR path.

Runnable as ``python tests/test_colsplit_bitequal.py`` or under pytest.
"""
import contextlib
import io
import os

import numpy as np
from scipy.sparse import _sparsetools, csr_matrix, random as sprandom

from selfcal.core import solve as SV
from selfcal.core.blockcsr import ColSplitCSR, build_block_csr, partition_block_csr
from selfcal.core.solution import compute_x0_scalar_only
from selfcal.core.system import _range_ids, _scatter_block_split, _scatter_one_batch

_SEED_PLACEMENT = 11
_SEED_CONSUMERS = 21


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def legacy_place(rows, cols, data, num_rows):
    """The placement kernel `_scatter_one_batch` replaced: stable sort by row,
    then a within-row cumcount for the slot of each entry."""
    n = rows.shape[0]
    out_i = np.empty(n, np.int32)
    out_d = np.empty(n, np.float32)
    indptr = np.zeros(num_rows + 1, np.int64)
    np.cumsum(np.bincount(rows, minlength=num_rows), out=indptr[1:])
    order = np.argsort(rows, kind="stable")
    rows_s, cols_s, data_s = rows[order], cols[order], data[order]
    is_new = np.empty(n, bool)
    is_new[0] = True
    is_new[1:] = rows_s[1:] != rows_s[:-1]
    starts = np.flatnonzero(is_new)
    within = np.arange(n, dtype=np.int64) - starts[np.cumsum(is_new, dtype=np.int64) - 1]
    slots = indptr[rows_s] + within
    out_d[slots] = data_s
    out_i[slots] = cols_s
    return indptr, out_i, out_d


def make_batched_case(rng):
    """A random matrix whose rows come in contiguous groups (the spill
    batches / constraint blocks of a real run), with entries shuffled inside
    each group and no duplicate (row, col) pair."""
    n_cols = int(rng.integers(40, 400))
    rows_g = [int(rng.integers(1, 40)) for _ in range(int(rng.integers(2, 6)))]
    total_rows = sum(rows_g)
    per_row = rng.integers(0, 9, total_rows)
    rows = np.repeat(np.arange(total_rows, dtype=np.int64), per_row)
    cols = np.empty(rows.size, np.int64)
    at = 0
    for k in per_row:
        if k:
            cols[at:at + k] = rng.permutation(n_cols)[:k]
            at += k
    data = rng.standard_normal(rows.size).astype(np.float32) * 7
    starts = np.concatenate(([0], np.cumsum(rows_g)))
    perm = np.concatenate([rng.permutation(np.flatnonzero(
        (rows >= starts[g]) & (rows < starts[g + 1]))) for g in range(len(rows_g))])
    return n_cols, rows_g, starts, rows[perm], cols[perm], data[perm]


def canonicalise(d, i, ip):
    n_rows = ip.shape[0] - 1
    if not _sparsetools.csr_has_sorted_indices(n_rows, ip, i):
        _sparsetools.csr_sort_indices(n_rows, ip, i, d)
    return bool(_sparsetools.csr_has_canonical_format(n_rows, ip, i))


# --------------------------------------------------------------------------
# 1 + 2 + 3: placement and the preconditioner iterator
# --------------------------------------------------------------------------
def test_placement_and_preconditioner_bitequal(trials=40):
    rng = np.random.default_rng(_SEED_PLACEMENT)
    checked = 0
    for _ in range(trials):
        n_cols, rows_g, starts, rows, cols, data = make_batched_case(rng)
        total_rows, nnz = int(starts[-1]), rows.size
        if nnz == 0:
            continue
        compact = bool(rng.integers(0, 2))
        if compact:
            active = np.zeros(n_cols, bool)
            active[np.unique(cols)] = True
            n_active = int(active.sum())
            col_map = np.cumsum(active, dtype=np.int32) - 1
        else:
            active, n_active, col_map = None, n_cols, None

        # ---- unified placement
        indptr = np.zeros(total_rows + 1, np.int64)
        np.cumsum(np.bincount(rows, minlength=total_rows), out=indptr[1:])
        u_d, u_i = np.zeros(nnz, np.float32), np.zeros(nnz, np.int32)
        for g in range(len(rows_g)):
            m = (rows >= starts[g]) & (rows < starts[g + 1])
            _scatter_one_batch((rows[m] - starts[g]).astype(np.int32),
                               cols[m].astype(np.int32), data[m], int(starts[g]),
                               rows_g[g], indptr, col_map, u_d, u_i, n_cols)
            if not m.any():
                continue
            e0, e1 = int(indptr[starts[g]]), int(indptr[starts[g + 1]])
            _, li, ld = legacy_place((rows[m] - starts[g]).astype(np.int64),
                                     cols[m], data[m], rows_g[g])
            if col_map is not None:
                li = col_map[li]
            assert np.array_equal(li, u_i[e0:e1]), "indices differ from the legacy kernel"
            assert np.array_equal(ld.view(np.uint32), u_d[e0:e1].view(np.uint32)), \
                "data differ from the legacy kernel"

        ref = csr_matrix((data, (rows, col_map[cols] if compact else cols)),
                         shape=(total_rows, n_active))
        ref.sum_duplicates()
        ref.sort_indices()
        uni = csr_matrix((u_d.copy(), u_i.copy(), indptr.copy()),
                         shape=(total_rows, n_active))
        uni.sort_indices()
        assert np.array_equal(uni.indices, ref.indices), "indices differ from scipy"
        assert np.array_equal(uni.data.view(np.uint32), ref.data.view(np.uint32)), \
            "data differ from scipy"

        # ---- partitioned placement, one storage block per row group
        T = int(rng.integers(2, 5))
        full_cuts = np.linspace(0, n_cols, T + 1, dtype=np.int64)
        cuts_c = (np.array([int(active[:int(c)].sum()) for c in full_cuts], np.int64)
                  if compact else full_cuts.copy())
        rid_all = _range_ids(cols, full_cuts)
        nb = len(rows_g)
        blk_rnnz, blk_start, ip_off = [], [0], [0]
        for g in range(nb):
            m = (rows >= starts[g]) & (rows < starts[g + 1])
            blk_rnnz.append(np.bincount(rid_all[m], minlength=T).astype(np.int64))
            blk_start.append(blk_start[-1] + int(m.sum()))
            ip_off.append(ip_off[-1] + T * (rows_g[g] + 1))
        p_d, p_i = np.zeros(nnz, np.float32), np.zeros(nnz, np.int32)
        ip_all = np.zeros(ip_off[-1], np.int32)
        for g in range(nb):
            m = (rows >= starts[g]) & (rows < starts[g + 1])
            layout = (T, full_cuts, cuts_c, blk_start[g], int(m.sum()),
                      ip_off[g], blk_rnnz[g])
            _scatter_block_split((rows[m] - starts[g]).astype(np.int32),
                                 cols[m].astype(np.int32), data[m], rows_g[g],
                                 layout, col_map, p_d, p_i, ip_all, n_cols)

        sub = [[None] * T for _ in range(nb)]
        for g in range(nb):
            off = 0
            for t in range(T):
                e0 = blk_start[g] + off
                e1 = e0 + int(blk_rnnz[g][t])
                off += int(blk_rnnz[g][t])
                ip = ip_all[ip_off[g] + t * (rows_g[g] + 1):
                            ip_off[g] + (t + 1) * (rows_g[g] + 1)]
                assert int(ip[0]) == 0 and int(ip[-1]) == e1 - e0
                assert canonicalise(p_d[e0:e1], p_i[e0:e1], ip), "piece not canonical"
                sub[g][t] = (p_d[e0:e1], p_i[e0:e1], ip)
                exp = uni[starts[g]:starts[g + 1],
                          int(cuts_c[t]):int(cuts_c[t + 1])]
                exp.sort_indices()
                assert np.array_equal(ip, exp.indptr), f"piece {g},{t} indptr"
                assert np.array_equal(p_i[e0:e1], exp.indices), f"piece {g},{t} cols"
                assert np.array_equal(p_d[e0:e1].view(np.uint32),
                                      exp.data.view(np.uint32)), f"piece {g},{t} data"

        # ---- preconditioner column norms, bitwise, over several chunkings
        A = ColSplitCSR(sub, np.asarray(starts, np.int64), cuts_c,
                        (total_rows, n_active))
        for chunk in (1, 3, 17, max(1, nnz // 2), nnz, nnz + 5):
            want = np.zeros(n_active, np.float32)
            for s0 in range(0, nnz, chunk):
                dd = uni.data[s0:s0 + chunk]
                want += np.bincount(uni.indices[s0:s0 + chunk], weights=dd * dd,
                                    minlength=n_active).astype(np.float32)
            got = np.zeros(n_active, np.float32)
            for pieces in SV._iter_range_pieces_colsplit(A, chunk):
                for d, i, c0, w in pieces:
                    got[c0:c0 + w] += np.bincount(i, weights=d * d,
                                                  minlength=w).astype(np.float32)
            assert np.array_equal(want.view(np.uint32), got.view(np.uint32)), \
                f"column norms differ at chunk_size={chunk}"
        checked += 1
    assert checked > 0


# --------------------------------------------------------------------------
# 4: the consumers (x0, SpMV operator, full solves)
# --------------------------------------------------------------------------
def test_consumers_bitequal(trials=12):
    rng = np.random.default_rng(_SEED_CONSUMERS)
    checked = 0
    for trial in range(trials):
        dtype = np.float32 if trial % 3 else np.float64
        m, n = int(rng.integers(40, 700)), int(rng.integers(30, 400))
        A = sprandom(m, n, density=float(rng.choice([0.03, 0.1, 0.3])),
                     random_state=int(rng.integers(1e9)), format='csr').astype(dtype)
        n_sc = int(rng.integers(2, 6))
        scalar_col_start = n - n_sc
        A = A.tolil()
        for r in range(m):                     # a scalar-block entry in every row
            A[r, scalar_col_start + int(rng.integers(0, n_sc))] = float(rng.standard_normal())
        A = A.tocsr().astype(dtype)
        A.sort_indices()
        A.sum_duplicates()
        if A.nnz == 0:
            continue

        def bcsr():
            return build_block_csr(A.data.copy(), A.indices.astype(np.int32),
                                   A.indptr.astype(np.int64), (m, n),
                                   int(rng.integers(30, 3000)))

        T = int(rng.choice([2, 3, 4, 6]))
        cuts = np.linspace(0, n, T + 1, dtype=np.int64)
        if cuts[-2] > scalar_col_start:        # scalar block must sit in the last range
            cuts[-2] = scalar_col_start - 1 if scalar_col_start > 1 else 0
            cuts = np.unique(cuts)
            if len(cuts) - 1 < 2:
                continue
        B, C = bcsr(), partition_block_csr(bcsr(), cuts)
        assert C.nnz == B.nnz

        b = rng.standard_normal(m).astype(dtype)
        x0a = compute_x0_scalar_only(B, b, (1, 1), scalar_col_start, num_sky_blocks=1)
        x0b = compute_x0_scalar_only(C, b, (1, 1), scalar_col_start, num_sky_blocks=1)
        assert np.array_equal(x0a.view(np.uint8), x0b.view(np.uint8)), "x0 differs"

        os.environ['SELFCAL_RMATVEC_SPLIT'] = '1'   # keep the reference sequential
        try:
            op_ref = SV._make_parallel_operator_blocks(bcsr(), 4, a_owned=False)
            op_new = SV._make_parallel_operator_colsplit(
                partition_block_csr(bcsr(), cuts), 4, T)
            x = rng.standard_normal(n).astype(dtype)
            y = rng.standard_normal(m).astype(dtype)
            assert np.array_equal(op_ref.matvec(x).view(np.uint8),
                                  op_new.matvec(x).view(np.uint8)), "matvec differs"
            assert np.array_equal(op_ref.rmatvec(y).view(np.uint8),
                                  op_new.rmatvec(y).view(np.uint8)), "rmatvec differs"
            for op in (op_ref, op_new):
                op._executor.shutdown(wait=False)
                if getattr(op, '_rmatvec_executor', None) is not None:
                    op._rmatvec_executor.shutdown(wait=False)

            with contextlib.redirect_stdout(io.StringIO()):
                kw = dict(atol=1e-8, btol=1e-8, damp=0, iter_lim=20,
                          precondition=True, solver='lsqr',
                          use_float32=(dtype == np.float32), n_threads=4,
                          a_owned=True)
                xa = SV.apply_lsqr(bcsr(), b.copy(), (1, 1), x0=x0a.copy(), **kw)
                xb = SV.apply_lsqr(partition_block_csr(bcsr(), cuts), b.copy(),
                                   (1, 1), x0=x0b.copy(), **kw)
        finally:
            os.environ.pop('SELFCAL_RMATVEC_SPLIT', None)
        assert xa.dtype == xb.dtype
        assert np.array_equal(xa.view(np.uint8), xb.view(np.uint8)), "solve differs"
        checked += 1
    assert checked > 0


if __name__ == '__main__':
    test_placement_and_preconditioner_bitequal()
    print("OK placement + preconditioner: bit-equal to the unified matrix")
    test_consumers_bitequal()
    print("OK consumers: x0 / SpMV / full solves bit-equal to BlockCSR")
