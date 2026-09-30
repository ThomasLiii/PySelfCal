"""LSQR solve: preconditioning, parallel SpMV operator, and the solver call.

The solver stage of ``selfcal.core``: matrix/RHS assembly lives in
``assembly.py`` and ``system.py``; this module only consumes the returned
(A, b). ``apply_lsqr`` runs scipy lsqr/lsmr against a thread-parallel SpMV
``LinearOperator``, with Jacobi column-norm preconditioning and — when
``setup_lsqr`` has already dropped all-zero columns and passes an
``active_mask`` — expansion of the compact solution back to the full column
layout.
"""
from __future__ import annotations

import logging
import mmap as _mmap
import os

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import coo_matrix, csr_matrix
try:
    # Private scipy kernel: ``_sparsetools.csc_matvec`` drives the bit-exact
    # BlockCSR transpose product (a CSR row-block reinterpreted as CSC).
    # This is scipy-private API and can move between releases; it is present
    # and verified through scipy 1.16.
    from scipy.sparse import _sparsetools
except ImportError as e:  # pragma: no cover - depends on scipy internals
    raise ImportError(
        "selfcal.core.solve couples to scipy's private sparse kernels "
        "(scipy.sparse._sparsetools.csc_matvec) for the bit-exact BlockCSR "
        "transpose product; that private module is present and verified "
        "through scipy 1.16. Pin a compatible scipy version or upgrade selfcal."
    ) from e
from scipy.sparse.linalg import lsqr, lsmr, LinearOperator
from threadpoolctl import threadpool_limits

from .blockcsr import BlockCSR, ColSplitCSR, _csr_shell
from .lsqr_inplace import lsqr_inplace

logger = logging.getLogger(__name__)

__all__ = ["apply_lsqr"]


def _partition_csr(A, n_blocks):
    """Split CSR matrix into row-blocks sharing data/indices arrays (zero-copy).

    The blocks are assembled by direct attribute assignment rather than the
    ``csr_matrix((data, indices, indptr), ...)`` constructor: that constructor
    runs scipy's index-dtype unification with ``check_contents=True``, and when
    the parent has int64 indices (forced whenever total nnz > 2^31) every
    block's contents fit int32, so it silently downcast-COPIES all indices —
    ~nnz*4 bytes of duplicates held for the whole solve (e.g. ~89 GiB at
    nnz ~2.4e10, one production-size tile; this is scipy's csr_matrix
    constructor behavior at least through scipy 1.16). Attribute assignment keeps true
    views of the parent's data/indices; only the per-block shifted indptr is a
    fresh (small) array. Index dtype does not enter the float arithmetic, so
    matvec results are bit-identical either way.
    """
    n_rows = A.shape[0]
    boundaries = np.linspace(0, n_rows, n_blocks + 1, dtype=int)
    blocks = []
    for i in range(n_blocks):
        sr, er = int(boundaries[i]), int(boundaries[i + 1])
        nnz_s = A.indptr[sr]
        blk = csr_matrix.__new__(csr_matrix)
        blk._shape = (er - sr, int(A.shape[1]))
        blk.data = A.data[A.indptr[sr]:A.indptr[er]]
        blk.indices = A.indices[A.indptr[sr]:A.indptr[er]]
        blk.indptr = A.indptr[sr:er + 1] - nnz_s
        blocks.append(blk)
    return blocks, boundaries

def _make_parallel_operator(A_csr, n_threads):
    """Build a LinearOperator with thread-parallel matvec, scipy-native rmatvec.

    matvec: per-thread row-block partition of A_csr (GIL released during scipy CSR SpMV).
    rmatvec: A_csr.T as a zero-copy CSC view; scipy CSC SpMV handles A.T @ y directly.

    Do NOT replace the view with AT = A_csr.T.tocsr(): that copies all of A's
    storage (data + indices, ~8 bytes per nonzero — e.g. ~88 GB at nnz ~1e10).
    The CSC view shares A_csr's storage and costs O(1) — scipy's CSC @ vec is
    fast and releases the GIL.
    """
    m, n = A_csr.shape

    logger.info(f"Building parallel SpMV operator ({n_threads} threads)...")
    AT_view = A_csr.T  # zero-copy CSC view sharing storage with A_csr

    A_blocks, A_bounds = _partition_csr(A_csr, n_threads)

    executor = ThreadPoolExecutor(max_workers=n_threads)
    dtype = A_csr.dtype

    def _matvec(x):
        out = np.empty(m, dtype=dtype)
        def _work(i):
            out[A_bounds[i]:A_bounds[i+1]] = A_blocks[i] @ x
        list(executor.map(_work, range(n_threads)))
        return out

    _rmv_threads = rmatvec_threads(n_threads, n, dtype.itemsize)
    _rowsplit, _rex = (None, None)
    if _rmv_threads > 1 and A_csr.indptr.dtype == A_csr.indices.dtype:
        _rowsplit, _rex, _ = _make_rowsplit_rmatvec(
            m, n, dtype, _rmv_threads, np.array([0, m], dtype=np.int64),
            _scatter_block_csr([A_csr], np.array([0, m], dtype=np.int64), n))

    def _rmatvec(y):
        if _rowsplit is not None and np.promote_types(dtype, y.dtype) == dtype:
            return _rowsplit(y)
        # Sequential reference: one scipy CSC SpMV over the zero-copy
        # transpose view (the kernel releases the GIL).
        return AT_view @ y

    op = LinearOperator((m, n), matvec=_matvec, rmatvec=_rmatvec, dtype=A_csr.dtype)
    op._executor = executor
    op._rmatvec_executor = _rex
    op._AT_view = AT_view  # prevent GC
    return op


def _iter_global_entry_chunks(blocks, chunk_size):
    """Yield (data, cols) pairs cut at GLOBAL entry-stream boundaries.

    The preconditioner accumulates float32 partial sums per chunk, so the
    chunk CUTS are part of the byte-equal contract: they must fall at the
    same multiples of ``chunk_size`` over the logical concatenation of all
    blocks that the unified-matrix loop would use. A chunk that straddles a
    block boundary is therefore CONCATENATED (small copy, <= chunk_size)
    rather than split — splitting would change the partial-sum tree.
    """
    buf_d, buf_c, buf_n = [], [], 0
    for blk in blocks:
        pos, size = 0, blk.data.size
        while pos < size:
            take = min(size - pos, chunk_size - buf_n)
            buf_d.append(blk.data[pos:pos + take])
            buf_c.append(blk.indices[pos:pos + take])
            buf_n += take
            pos += take
            if buf_n == chunk_size:
                yield ((buf_d[0], buf_c[0]) if len(buf_d) == 1
                       else (np.concatenate(buf_d), np.concatenate(buf_c)))
                buf_d, buf_c, buf_n = [], [], 0
    if buf_n:
        yield ((buf_d[0], buf_c[0]) if len(buf_d) == 1
               else (np.concatenate(buf_d), np.concatenate(buf_c)))


def _iter_range_pieces_colsplit(A, chunk_size):
    """Cut a ColSplitCSR at the same GLOBAL entry boundaries as
    :func:`_iter_global_entry_chunks`, and yield each chunk as its PER-RANGE
    slices rather than one merged array.

    The logical stream is row-major with each row's ranges in order (rows are
    column-sorted), so a global chunk is, per block, one contiguous slice of
    every range's arrays — the boundary rows split at whichever range holds
    the chunk edge. Yielding those slices with their column offset (instead
    of concatenating them under global column ids) keeps the indices as
    int32 VIEWS: no per-chunk int64 copy of the ids, and no per-block
    segment/prefix tables. Each column's entries stay in row order within
    the chunk, and a column belongs to exactly one range, so the caller's
    per-range float32 partial sums are bit-identical to the unified
    matrix's single partial per chunk.

    Yields
    ------
    list of (data, indices, col_offset, width)
        ``indices`` are local to the range; add ``col_offset`` for global.
    """
    T = A.nranges
    cuts = A.cuts
    pend = [[] for _ in range(T)]
    pend_n = 0

    def _emit():
        out = []
        for t in range(T):
            if not pend[t]:
                continue
            if len(pend[t]) == 1:
                d, i = pend[t][0]
            else:
                d = np.concatenate([p[0] for p in pend[t]])
                i = np.concatenate([p[1] for p in pend[t]])
            out.append((d, i, int(cuts[t]), int(cuts[t + 1] - cuts[t])))
            pend[t].clear()
        return out

    for b in range(A.nblocks):
        per = A.sub[b]
        ips = [p[2] for p in per]
        n_rows = ips[0].shape[0] - 1

        def _offset(r, ips=ips):
            """Entries of the block before row ``r``, over all ranges."""
            return sum(int(ip[r]) for ip in ips)

        def _row_at(target, lo, hi):
            """Last row whose start offset is <= target (searchsorted 'right'
            minus one, evaluated without materialising the offsets: only the
            few chunk boundaries per block are ever located)."""
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if _offset(mid) <= target:
                    lo = mid
                else:
                    hi = mid - 1
            return lo

        nnz_b = _offset(n_rows)
        lpos = 0
        row_lo = 0                    # boundaries advance, so the search does
        while lpos < nnz_b:
            take = min(nnz_b - lpos, chunk_size - pend_n)
            l0, l1 = lpos, lpos + take
            r0 = _row_at(l0, row_lo, n_rows)
            q0 = l0 - _offset(r0)
            if l1 == nnz_b:                       # chunk ends at the block end
                r1 = n_rows - 1
                q1 = nnz_b - _offset(r1)
            else:
                r1 = _row_at(l1, r0, n_rows)
                q1 = l1 - _offset(r1)
            row_lo = r0
            acc0 = acc1 = 0                       # entries of the boundary rows
            for t in range(T):                    # in ranges before t
                ip = ips[t]
                w0, w1 = int(ip[r0]), int(ip[r1])
                seg0, seg1 = int(ip[r0 + 1]) - w0, int(ip[r1 + 1]) - w1
                s0 = w0 + min(max(q0 - acc0, 0), seg0)
                s1 = w1 + min(max(q1 - acc1, 0), seg1)
                acc0 += seg0
                acc1 += seg1
                if s1 > s0:
                    d_t, i_t, _ = per[t]
                    pend[t].append((d_t[s0:s1], i_t[s0:s1]))
            pend_n += take
            lpos = l1
            if pend_n == chunk_size:
                yield _emit()
                pend_n = 0
    if pend_n:
        yield _emit()


def rmatvec_threads(n_threads, n_cols, itemsize=4):
    """Threads for the row-split ``A^T @ y`` — the default transpose product.

    ``A^T @ y`` is a scatter (every matrix row adds into scattered output
    columns), so it cannot be threaded without changing the order in which
    each output column's contributions are summed. The row-split kernel gives
    each thread a private output buffer for its own rows and reduces the
    buffers in a fixed order afterwards: race-free, DETERMINISTIC for a given
    thread count, and 5.2x faster than the sequential kernel on the real
    1k-frame matrix once the columns are compacted (2.07 s vs 10.69 s). It is
    NOT bit-identical to the sequential kernel — a column's sum is a fixed
    tree rather than one chain, a float32 reassociation of ~1e-7 per product
    that reaches the converged maps at the ~1e-6 level of their own values
    (integer coverage, Fisher and separability outputs are untouched). The
    solution therefore depends on the thread count at that level; a fixed
    thread count reproduces the same bytes.

    Default: as many threads as the matvec uses, capped so the private
    buffers (``threads x n_cols x itemsize``) stay under
    ``SELFCAL_RMATVEC_BUFFER_GB`` (default 16). ``SELFCAL_PARALLEL_RMATVEC=<n>``
    pins the count; ``1`` (or ``0``) selects the sequential kernel — the
    byte-exact reference the regression goldens were re-baselined against.
    """
    raw = os.environ.get('SELFCAL_PARALLEL_RMATVEC')
    if raw not in (None, '', 'auto'):
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    try:
        budget = float(os.environ.get('SELFCAL_RMATVEC_BUFFER_GB', '16')) * 1e9
    except ValueError:
        budget = 16e9
    per_thread = max(1, int(n_cols)) * int(itemsize)
    return max(1, min(int(n_threads), int(budget // per_thread)))


def _row_spans(row_bounds, m, n_pieces):
    """Cut rows ``0..m`` into ``n_pieces`` contiguous ranges and intersect each
    with the storage blocks: ``[(r0, r1, [(block, lo, hi), ...]), ...]``."""
    cuts = np.linspace(0, m, max(1, n_pieces) + 1, dtype=np.int64)
    pieces = []
    for r0, r1 in zip(cuts[:-1], cuts[1:]):
        r0, r1 = int(r0), int(r1)
        if r1 <= r0:
            continue
        spans = []
        for bi in range(len(row_bounds) - 1):
            lo, hi = max(r0, int(row_bounds[bi])), min(r1, int(row_bounds[bi + 1]))
            if lo < hi:
                spans.append((bi, lo, hi))
        if spans:
            pieces.append((r0, r1, spans))
    return pieces


def _make_rowsplit_rmatvec(m, n, dtype, n_threads, row_bounds, scatter):
    """Row-split ``A^T @ y`` over any storage: ``scatter(bi, lo, hi, y_seg, buf)``
    must add rows ``lo:hi`` (global ids, inside storage block ``bi``) into
    ``buf`` (length ``n``) with scipy's ``csc_matvec``.

    Each thread owns a contiguous row range and a private buffer, allocated
    once per operator. The reduction walks the buffers in index order for
    every element, so it is the same left fold whichever thread evaluates a
    column chunk — deterministic, and parallel. Rows within a thread are
    scattered in row order block by block, so for a column the per-thread
    partial is the same sequence whatever the storage layout.
    """
    pieces = _row_spans(row_bounds, m, n_threads)
    bufs = [np.zeros(n, dtype=dtype) for _ in pieces]
    ex = ThreadPoolExecutor(max_workers=len(pieces))
    logger.info(f"  rmatvec: row-split over {len(pieces)} threads "
                f"(+{len(pieces) * n * np.dtype(dtype).itemsize / 2**30:.2f} GB "
                "of private buffers; deterministic for this thread count, not "
                "bit-identical to the sequential kernel).")
    red_cuts = np.linspace(0, n, len(pieces) + 1, dtype=np.int64)

    def _rmatvec(y):
        y = np.ascontiguousarray(y, dtype=dtype)

        def _scatter_piece(i):
            buf = bufs[i]
            buf[...] = 0
            for bi, lo, hi in pieces[i][2]:
                scatter(bi, lo, hi, y[lo:hi], buf)

        list(ex.map(_scatter_piece, range(len(pieces))))
        out = np.empty(n, dtype=dtype)

        def _reduce_chunk(k):
            c0, c1 = int(red_cuts[k]), int(red_cuts[k + 1])
            out[c0:c1] = bufs[0][c0:c1]
            for b in bufs[1:]:                 # fixed order => same fold everywhere
                out[c0:c1] += b[c0:c1]

        list(ex.map(_reduce_chunk, range(len(pieces))))
        return out

    return _rmatvec, ex, len(pieces)


def _scatter_block_csr(blocks, row_bounds, n):
    """``scatter`` for a BlockCSR (or a single unified CSR passed as one block):
    the indptr slice keeps its absolute offsets, so the whole index/data
    arrays are passed and nothing is copied."""
    def scatter(bi, lo, hi, y_seg, buf):
        blk = blocks[bi]
        lr0, lr1 = lo - int(row_bounds[bi]), hi - int(row_bounds[bi])
        _sparsetools.csc_matvec(n, hi - lo, blk.indptr[lr0:lr1 + 1],
                                blk.indices, blk.data, y_seg, buf)
    return scatter



def _rmatvec_split_ranges():
    """Column-range count of the partitioned storage (SELFCAL_RMATVEC_SPLIT,
    default 1). More than one range only matters to the byte-exact
    verification kernel (``SELFCAL_PARALLEL_RMATVEC=1``): each range's columns
    are then folded by one thread in the sequential order. The default
    row-split kernel needs no ranges."""
    try:
        return max(1, int(os.environ.get('SELFCAL_RMATVEC_SPLIT', '1')))
    except ValueError:
        return 1


def _madvise_dontneed(arr, i0, i1):
    """Release the physical pages backing arr[i0:i1] when arr sits in an
    anonymous shared mmap (the parallel-scatter buffers). Page-aligned
    inward; silently a no-op for ordinary arrays."""
    base = arr
    while not isinstance(base, _mmap.mmap):
        if isinstance(base, memoryview):
            base = base.obj              # np.frombuffer's .base is a memoryview
            continue
        nxt = getattr(base, 'base', None)
        if nxt is None:
            return                       # ordinary array — nothing to punch
        base = nxt
    itemsize = arr.dtype.itemsize
    off = arr.ctypes.data - np.frombuffer(base, dtype=np.uint8).ctypes.data
    b0 = off + i0 * itemsize
    b1 = off + i1 * itemsize
    page = _mmap.PAGESIZE
    b0 = (b0 + page - 1) // page * page
    b1 = b1 // page * page
    if b1 > b0:
        # Shared-anonymous (shmem) pages are NOT freed by MADV_DONTNEED —
        # it only drops the mappings while the pages stay charged to the
        # shmem object. MADV_REMOVE hole-punches them (actually frees).
        try:
            base.madvise(getattr(_mmap, 'MADV_REMOVE', _mmap.MADV_DONTNEED),
                         b0, b1 - b0)
        except (ValueError, OSError):
            try:
                base.madvise(_mmap.MADV_DONTNEED, b0, b1 - b0)
            except (ValueError, OSError):
                pass


def _partition_block_columns(blk, cuts, chunk=64_000_000, workers=8,
                             count_chunk_rows=16_000_000):
    """Split one canonical (column-sorted) CSR row-block into per-column-range
    sub-CSRs with LOCAL int32 column ids.

    Entries are taken in storage order and stable-filtered per range, so each
    sub-block's rows keep their original within-row entry order and each
    COLUMN's complete entry sequence lands in exactly one range — the
    property the bit-equal partitioned SpMV rests on. Pure data movement,
    so both stages run in threads: range classification chunk-parallel,
    then one gather thread per range (each writes only its own arrays).
    """
    n_rows = blk.shape[0]
    nranges = len(cuts) - 1
    nnz = blk.indices.shape[0]
    rid = np.empty(nnz, dtype=np.int8)

    def _classify(s0):
        s1 = min(s0 + chunk, nnz)
        rid[s0:s1] = np.searchsorted(cuts[1:-1], blk.indices[s0:s1],
                                     side='right').astype(np.int8)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(_classify, range(0, nnz, chunk)))

    subs = [None] * nranges
    cr = count_chunk_rows   # bounds the row-id expansion for the per-row
                            # counts at ~128 MB instead of 8 B per entry

    def _gather(t):
        m_t = rid == t
        cnt = np.zeros(n_rows, dtype=np.int64)
        for r0 in range(0, n_rows, cr):
            r1 = min(r0 + cr, n_rows)
            s0, s1 = int(blk.indptr[r0]), int(blk.indptr[r1])
            if s0 == s1:
                continue
            row_local = np.repeat(np.arange(r1 - r0, dtype=np.int64),
                                  np.diff(blk.indptr[r0:r1 + 1]))
            cnt[r0:r1] = np.bincount(row_local[m_t[s0:s1]],
                                     minlength=r1 - r0)
            del row_local
        indptr_t = np.zeros(n_rows + 1, dtype=np.int32)
        np.cumsum(cnt, out=indptr_t[1:])
        data_t = blk.data[m_t]
        # int32 subtrahend: int32 - int64 scalar would promote to an
        # int64 intermediate (8 B x selected) under NEP 50.
        idx_t = blk.indices[m_t]
        np.subtract(idx_t, np.int32(cuts[t]), out=idx_t)
        subs[t] = (data_t, idx_t, indptr_t)
    with ThreadPoolExecutor(max_workers=nranges) as ex:
        list(ex.map(_gather, range(nranges)))
    return subs


class _ColSplit:
    """Column-partitioned copy of a BlockCSR for the bit-equal parallel SpMV.

    ``sub[b][t]`` is the (data, local int32 indices, int32 indptr) of storage
    block ``b`` restricted to columns ``cuts[t]:cuts[t+1]``. Each source
    block's pages are released (madvise) right after its split, so the build
    transient stays around one block; steady overhead is the extra per-range
    indptrs ((T-1) x 4 B/row).
    """

    def __init__(self, bcsr, nranges):
        m, n = bcsr.shape
        self.shape = bcsr.shape
        self.dtype = bcsr.dtype
        self.row_bounds = bcsr.row_bounds.copy()
        self.cuts = np.linspace(0, n, nranges + 1, dtype=np.int64)
        self.sub = []
        for blk in bcsr.blocks:
            self.sub.append(_partition_block_columns(blk, self.cuts))
            d, i = blk.data, blk.indices
            blk.data = blk.indices = blk.indptr = None
            _madvise_dontneed(d, 0, d.shape[0])
            _madvise_dontneed(i, 0, i.shape[0])
        bcsr.blocks = []


def _make_parallel_operator_colsplit(bcsr, n_threads, nranges):
    """Bit-equal FULLY-parallel SpMV from a column-partitioned copy.

    rmatvec: thread t owns columns ``cuts[t]:cuts[t+1]`` and walks the row
    pieces IN ORDER with the same raw ``csc_matvec`` kernel the sequential
    path uses — a column's complete fold lives in exactly one thread and
    sees its entries in the identical global row order, so every output
    element is bit-identical to the single-thread scatter.

    matvec: the output starts at zero and the ranges are applied as ORDERED
    continuation passes (``csr_matvec`` resumes each row's sum from
    ``Yx[i]``), each pass row-parallel. Rows are column-sorted (canonical),
    so the concatenation of the per-range segments IS the original storage
    order — the per-row addition sequence is unchanged.

    Consumes ``bcsr`` (its blocks are released as they are split).
    """
    if isinstance(bcsr, ColSplitCSR):
        split = bcsr                     # storage already partitioned by setup
        nranges = split.nranges
        logger.info(f"Building column-partitioned SpMV operator from partitioned "
                    f"storage ({n_threads} matvec threads, {nranges} rmatvec ranges, "
                    f"{split.nblocks} storage blocks)...")
    else:
        logger.info(f"Building column-partitioned SpMV operator ({n_threads} matvec "
                    f"threads, {nranges} rmatvec column ranges, "
                    f"{len(bcsr.blocks)} storage blocks)...")
        split = _ColSplit(bcsr, nranges)
    m, n = split.shape
    dtype = split.dtype
    row_bounds = split.row_bounds.copy()
    thread_cuts = np.linspace(0, m, max(1, n_threads) + 1, dtype=np.int64)
    bounds = np.unique(np.concatenate((row_bounds, thread_cuts)))
    cuts = split.cuts

    # Piece-level views: (r0, r1, per-range (data, idx, indptr)). Every
    # entry here is a VIEW — the indptr slice keeps the stored absolute
    # offsets and the data/index arrays are passed whole, which is what the
    # scipy kernels index with, so the steady indptr overhead stays the
    # nranges x 4 B/row the storage already holds (no per-piece copies).
    pieces = []
    for r0, r1 in zip(bounds[:-1], bounds[1:]):
        bi = int(np.searchsorted(row_bounds, r0, side='right') - 1)
        lr0 = int(r0 - row_bounds[bi])
        lr1 = int(r1 - row_bounds[bi])
        pieces.append((int(r0), int(r1),
                       [(data_t, idx_t, ip_t[lr0:lr1 + 1])
                        for data_t, idx_t, ip_t in split.sub[bi]]))
    split_sub = split.sub  # per-block (data, idx, indptr) views for the row-split scatter
    split.sub = None       # the container's own list dies; the views live on

    executor = ThreadPoolExecutor(max_workers=max(1, n_threads))
    rexecutor = ThreadPoolExecutor(max_workers=nranges)

    def _matvec(x):
        od = dtype                      # n_threads>1 regime: output in A dtype
        out = np.zeros(m, dtype=od)
        xs = x if x.dtype == od else np.ascontiguousarray(x, dtype=od)
        for t in range(nranges):        # ordered continuation passes
            c0, c1 = int(cuts[t]), int(cuts[t + 1])
            x_t = np.ascontiguousarray(xs[c0:c1])

            def _work(p, t=t, x_t=x_t, c1c0=c1 - c0):
                r0, r1, per_range = p
                data_t, idx_t, ip_t = per_range[t]
                if data_t.dtype == od:
                    _sparsetools.csr_matvec(r1 - r0, c1c0, ip_t, idx_t, data_t,
                                            x_t, out[r0:r1])
                    return
                s0, s1 = int(ip_t[0]), int(ip_t[-1])
                _sparsetools.csr_matvec(r1 - r0, c1c0, ip_t - np.int32(s0),
                                        idx_t[s0:s1], data_t[s0:s1].astype(od),
                                        x_t, out[r0:r1])
            list(executor.map(_work, pieces))
        return out

    # Row-split parallel scatter (the default): a thread owns a row range and
    # scatters each of its storage blocks' T range pieces into its private
    # buffer's column window — a column's entries still arrive in row order.
    _rmv_threads = rmatvec_threads(n_threads, n, np.dtype(dtype).itemsize)
    _rowsplit, _rsex = (None, None)
    if _rmv_threads > 1:
        _sub = split_sub          # captured before split.sub is dropped below

        def _scatter_split(bi, lo, hi, y_seg, buf):
            lr0, lr1 = lo - int(row_bounds[bi]), hi - int(row_bounds[bi])
            for t, (data_t, idx_t, ip_t) in enumerate(_sub[bi]):
                c0, c1 = int(cuts[t]), int(cuts[t + 1])
                _sparsetools.csc_matvec(c1 - c0, hi - lo, ip_t[lr0:lr1 + 1],
                                        idx_t, data_t, y_seg, buf[c0:c1])

        _rowsplit, _rsex, _ = _make_rowsplit_rmatvec(
            m, n, dtype, _rmv_threads, row_bounds, _scatter_split)

    def _rmatvec(y):
        if _rowsplit is not None and np.promote_types(dtype, y.dtype) == dtype:
            return _rowsplit(y)
        # Byte-exact verification kernel: one thread per column range, each
        # folding its columns in the sequential order.
        out_dtype = np.promote_types(dtype, y.dtype)
        out = np.zeros(n, dtype=out_dtype)
        yc = np.ascontiguousarray(y, dtype=out_dtype)

        def _work(t):
            c0, c1 = int(cuts[t]), int(cuts[t + 1])
            out_t = out[c0:c1]
            for r0, r1, per_range in pieces:      # global row order
                data_t, idx_t, ip_t = per_range[t]
                if data_t.dtype == out_dtype:
                    _sparsetools.csc_matvec(c1 - c0, r1 - r0, ip_t, idx_t,
                                            data_t, yc[r0:r1], out_t)
                    continue
                s0, s1 = int(ip_t[0]), int(ip_t[-1])
                _sparsetools.csc_matvec(c1 - c0, r1 - r0, ip_t - np.int32(s0),
                                        idx_t[s0:s1],
                                        data_t[s0:s1].astype(out_dtype),
                                        yc[r0:r1], out_t)
        list(rexecutor.map(_work, range(nranges)))
        return out

    op = LinearOperator((m, n), matvec=_matvec, rmatvec=_rmatvec, dtype=dtype)
    op._executor = executor
    op._rmatvec_executor = rexecutor
    op._rowsplit_executor = _rsex
    op._pieces = pieces
    op._colsplit = split
    op._split_sub = split_sub
    return op


def _make_parallel_operator_blocks(bcsr, n_threads, a_owned=False):
    _nranges = _rmatvec_split_ranges()
    # The partition pays a build whose transients must stay NUMA-node-local
    # to be worth it: three M11-scale runs measured the build at 3,000-4,850 s
    # (net loss at iter200) once the copies overflow the 386 GB node, while
    # at <=1k-tile scale it is a clear win (−21..25 %/iteration for a ~170 s
    # build). Auto-enable only when the matrix leaves comfortable node
    # headroom for the build's in-flight copies; SELFCAL_RMATVEC_SPLIT_MAX_GB
    # overrides the threshold (raise it to force the split at large scale).
    try:
        _split_max = float(os.environ.get('SELFCAL_RMATVEC_SPLIT_MAX_GB', '170'))
    except ValueError:
        _split_max = 170.0
    if (a_owned and _nranges > 1 and n_threads > 1
            and rmatvec_threads(n_threads, bcsr.shape[1]) <= 1
            and len(bcsr.blocks) > 0 and bcsr.nnz * 8 <= _split_max * 1e9):
        # Column-partitioned bit-equal parallel SpMV; consumes the blocks,
        # hence only when the caller handed A over (keep_state=False).
        return _make_parallel_operator_colsplit(bcsr, n_threads, _nranges)

    """Thread-parallel matvec + bit-exact rmatvec for a BlockCSR.

    matvec: rows are cut at the union of storage-block boundaries and an
    ``n_threads``-way linspace; each piece is a zero-copy shell into one
    storage block. A row's dot product depends only on its own entries, so
    ANY row partition is bit-identical.

    rmatvec: A^T @ y must reproduce the unified CSC scatter's per-element
    addition ORDER, so blocks are scattered SEQUENTIALLY into one shared
    output via scipy's raw ``csc_matvec`` kernel (a CSR block reinterpreted
    as CSC is its transpose, and the kernel accumulates with ``+=``): the
    same C loop as the one-matrix product, split at row boundaries.
    Single-threaded, like the unified path's CSC-view rmatvec.
    """
    m, n = bcsr.shape
    dtype = bcsr.dtype
    logger.info(f"Building parallel SpMV operator ({n_threads} threads, "
                f"{len(bcsr.blocks)} int32 storage blocks)...")
    thread_cuts = np.linspace(0, m, max(1, n_threads) + 1, dtype=np.int64)
    bounds = np.unique(np.concatenate((bcsr.row_bounds, thread_cuts)))
    pieces = []
    for r0, r1 in zip(bounds[:-1], bounds[1:]):
        bi = int(np.searchsorted(bcsr.row_bounds, r0, side='right') - 1)
        blk = bcsr.blocks[bi]
        lr0 = int(r0 - bcsr.row_bounds[bi])
        lr1 = int(r1 - bcsr.row_bounds[bi])
        s0, s1 = int(blk.indptr[lr0]), int(blk.indptr[lr1])
        shell = _csr_shell(blk.data[s0:s1], blk.indices[s0:s1],
                           blk.indptr[lr0:lr1 + 1] - blk.indptr[lr0],
                           (lr1 - lr0, n))
        pieces.append((int(r0), int(r1), shell))

    executor = ThreadPoolExecutor(max_workers=max(1, n_threads))

    # Dtype mimicry of the unified path (bit-equal contract): at n_threads>1
    # the unified custom operator allocates its matvec output in A's dtype
    # (mixed-dtype products get truncated on assignment), while at
    # n_threads<=1 scipy wraps the raw matrix and PROMOTES. Production runs
    # use_float32=True where both agree; we reproduce each regime exactly.
    promote_matvec = n_threads <= 1

    def _matvec(x):
        od = np.promote_types(dtype, x.dtype) if promote_matvec else dtype
        out = np.empty(m, dtype=od)
        def _work(i):
            r0, r1, shell = pieces[i]
            out[r0:r1] = shell @ x
        list(executor.map(_work, range(len(pieces))))
        return out

    def _rmatvec_sequential(y):
        # Match scipy's own mixed-dtype coercion (e.g. f32 data x f64 y in
        # non-float32 runs): products and accumulation in the promoted dtype,
        # same as the unified CSC-view path, so bits are unchanged. In
        # production (use_float32=True) everything is f32 and no copy happens.
        out_dtype = np.promote_types(dtype, y.dtype)
        out = np.zeros(n, dtype=out_dtype)
        y = np.ascontiguousarray(y, dtype=out_dtype)
        for bi, blk in enumerate(bcsr.blocks):
            sr = int(bcsr.row_bounds[bi])
            er = int(bcsr.row_bounds[bi + 1])
            bd = (blk.data if blk.data.dtype == out_dtype
                  else blk.data.astype(out_dtype))
            _sparsetools.csc_matvec(n, er - sr, blk.indptr, blk.indices,
                                    bd, y[sr:er], out)
        return out

    # Row-split parallel scatter (the default). Only for the all-one-dtype
    # case: the private buffers are typed at construction, so a y of a
    # different dtype would change the promotion and is left to the
    # sequential kernel.
    _par_threads = rmatvec_threads(n_threads, n, np.dtype(dtype).itemsize)
    _rmatvec_parallel, _par_ex = (None, None)
    if _par_threads > 1:
        _rmatvec_parallel, _par_ex, _ = _make_rowsplit_rmatvec(
            m, n, dtype, _par_threads, bcsr.row_bounds,
            _scatter_block_csr(bcsr.blocks, bcsr.row_bounds, n))

    def _rmatvec(y):
        if (_rmatvec_parallel is not None
                and np.promote_types(dtype, y.dtype) == dtype):
            return _rmatvec_parallel(y)
        return _rmatvec_sequential(y)

    op = LinearOperator((m, n), matvec=_matvec, rmatvec=_rmatvec, dtype=dtype)
    op._executor = executor
    op._rmatvec_executor = _par_ex
    op._pieces = pieces
    op._bcsr = bcsr  # prevent GC of the storage blocks
    return op

def apply_lsqr(A: coo_matrix | csr_matrix | BlockCSR, b: np.ndarray,
                ref_shape: tuple[int, int], x0: np.ndarray | None = None,
                atol: float = 1e-05, btol: float = 1e-05, damp: float = 1e-2,
                iter_lim: int = 100, precondition: bool = True,
                solver: str = 'lsmr', use_float32: bool = False, n_threads: int = 32,
                active_mask: np.ndarray | None = None,
                num_cols_full: int | None = None,
                a_owned: bool = False) -> np.ndarray:
    """Applies LSQR or LSMR to solve for the sky and detector offsets.

    Parameters
    ----------
    A : coo_matrix or csr_matrix
        Sparse system matrix. When ``csr_matrix`` is passed together with an
        ``active_mask``, setup_lsqr is assumed to have already compacted the
        zero columns; ``apply_lsqr`` skips its own column elimination and
        uses ``active_mask`` only to expand the solution back to the full
        column space at the end.
    b : np.ndarray
        Right-hand-side vector (may arrive float32 from ``setup_lsqr``; it is
        upcast to float64 for float64 solves).
    ref_shape : tuple of int
        (height, width) of the reference frame; ``ref_h * ref_w`` sky columns.
    x0 : np.ndarray or None, optional
        Initial guess in the FULL (uncompacted) column layout, or None to
        start from zero. Compacted internally when ``active_mask`` is set.
    atol : float, optional
        Absolute tolerance passed to the solver (default 1e-05).
    btol : float, optional
        Relative tolerance passed to the solver (default 1e-05).
    damp : float, optional
        Tikhonov damping applied by the solver (default 1e-2).
    iter_lim : int, optional
        Maximum solver iterations (default 100).
    precondition : bool, optional
        Apply Jacobi column-norm preconditioning before solving (default True).
    solver : str, optional
        Solver to use: 'lsmr' (default, faster convergence) or 'lsqr'.
    use_float32 : bool, optional
        If True, cast matrix data and b to float32 before solving.
        Reduces memory bandwidth (~2x faster SpMV) at the cost of precision.
    n_threads : int, optional
        Thread count for the parallel SpMV ``LinearOperator``; <=1 uses scipy's
        serial matvec (default 32).
    active_mask : np.ndarray of bool, optional
        When set, marks the columns of the original (uncompacted) layout
        that are present in the supplied compact CSR. Used to expand the
        compact solution back to the full column space on return.
    num_cols_full : int, optional
        Original (uncompacted) column count. Required when ``active_mask``
        is given. Equals ``A.shape[1]`` when no compaction happened upstream.

    Returns
    -------
    x : np.ndarray
        Solution vector in the full (uncompacted) column layout: expanded via
        ``active_mask`` when compaction ran, otherwise the raw solver output.
    """
    if not isinstance(A, (coo_matrix, csr_matrix, BlockCSR, ColSplitCSR)):
        raise TypeError(
            "A must be a scipy.sparse.coo_matrix, csr_matrix, BlockCSR or ColSplitCSR")
    if not isinstance(b, np.ndarray):
        raise TypeError("b must be a numpy array")
    if not (isinstance(ref_shape, (list, np.ndarray, tuple)) and len(ref_shape) == 2):
        raise ValueError("ref_shape must be a list or tuple of length 2")

    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w

    # setup_lsqr may emit f32 b (exactly-f32-representable values only). For
    # use_float32 solves that's the wanted dtype already; for f64 solves,
    # upcasting is exact (every f32 value is exactly representable in f64),
    # so the solver sees a b bit-identical to one built in f64 from the start.
    if not use_float32 and b.dtype == np.float32:
        b = b.astype(np.float64)

    # ---- Pre-compacted fast path: setup_lsqr already dropped all-zero
    # ---- columns; A arrives as compact CSR (or int32 BlockCSR).
    if isinstance(A, (csr_matrix, BlockCSR, ColSplitCSR)):
        is_block = isinstance(A, BlockCSR)
        is_split = isinstance(A, ColSplitCSR)
        if active_mask is not None:
            if num_cols_full is None:
                raise ValueError(
                    "num_cols_full must be supplied alongside active_mask")
            num_cols = int(num_cols_full)
            n_active = int(active_mask.sum())
            if A.shape[1] != n_active:
                raise ValueError(
                    f"A.shape[1]={A.shape[1]} != active_mask.sum()={n_active}")
            x0_compressed = x0[active_mask] if x0 is not None else None
        else:
            num_cols = A.shape[1]
            n_active = num_cols
            x0_compressed = x0
        # Drop this function's reference to the full-layout x0 (the caller
        # transferred ownership and dropped its own reference — see
        # Calibrator.apply_lsqr) so the f64 original can be freed once
        # x0_compressed is cast to float32 below.
        x0 = None

        A_shape = A.shape
        if use_float32:
            logger.info("Downcasting to float32 for faster SpMV...")
            if is_split:
                for _b in range(A.nblocks):
                    for _t in range(A.nranges):
                        _d, _i, _ip = A.sub[_b][_t]
                        if _d.dtype != np.float32:
                            A.sub[_b][_t] = (_d.astype(np.float32), _i, _ip)
            else:
                for _blk in (A.blocks if is_block else (A,)):
                    if _blk.data.dtype != np.float32:
                        _blk.data = _blk.data.astype(np.float32)
            _b_in = b
            b = _b_in.astype(np.float32)
            del _b_in
            if x0_compressed is not None:
                x0_compressed = x0_compressed.astype(np.float32)

        if precondition:
            logger.info("Applying column-norm preconditioning...")
            chunk_size = 64_000_000  # ~256 MB per chunk at f32
            col_sq_norm = np.zeros(n_active, dtype=np.float32)

            # Same GLOBAL chunk cuts for blocks and unified (the float32
            # partial-sum tree is part of the byte-equal contract);
            # straddling block chunks are concatenated inside the iterator,
            # never split.
            def _chunk_pieces():
                """Each global chunk as [(data, indices, col_offset, width)]:
                one piece per column range for partitioned storage, a single
                whole-width piece otherwise."""
                if is_split:
                    yield from _iter_range_pieces_colsplit(A, chunk_size)
                elif is_block:
                    for _d, _c in _iter_global_entry_chunks(A.blocks, chunk_size):
                        yield [(_d, _c, 0, n_active)]
                else:
                    _d, _c = A.data, A.indices
                    for start in range(0, _d.size, chunk_size):
                        stop = min(start + chunk_size, _d.size)
                        yield [(_d[start:stop], _c[start:stop], 0, n_active)]

            def _partial(pieces):
                # A column lives in exactly one range, so folding a range's
                # entries into its own slice sees the same addends in the
                # same order as one whole-width bincount would.
                return [(c0, np.bincount(i, weights=d * d,
                                         minlength=w).astype(np.float32))
                        for d, i, c0, w in pieces]

            def _fold(parts):
                for c0, p in parts:
                    col_sq_norm[c0:c0 + p.size] += p

            # Each chunk's bincount is a pure function; computing a few of
            # them in threads (np.bincount releases the GIL) and applying
            # `+=` strictly in chunk order reproduces the serial loop's
            # partial-sum tree bit for bit. An in-flight partial costs up to
            # ~12 B x n_active (float64 bincount result + float32 cast), so
            # the window is capped to keep the transient under the solve
            # plateau; window 1 degenerates to the serial loop.
            _win = max(1, min(8, int(32e9 // max(1, n_active * 12))))
            if _win > 1:
                from collections import deque
                with ThreadPoolExecutor(max_workers=_win) as _ex:
                    _q = deque()
                    for _pieces in _chunk_pieces():
                        _q.append(_ex.submit(_partial, _pieces))
                        if len(_q) >= _win:
                            _fold(_q.popleft().result())
                    while _q:
                        _fold(_q.popleft().result())
            else:
                for _pieces in _chunk_pieces():
                    _fold(_partial(_pieces))

            col_norms = np.sqrt(col_sq_norm)
            col_norms[col_norms == 0] = 1.0
            M_inv = col_norms
            M = 1.0 / M_inv
            # Elementwise in-place scaling over disjoint tiles: no entry is
            # read or written by two tiles and multiplication is elementwise,
            # so any execution order gives identical bytes — threaded. Each
            # tile's gather temp is ~256 MB (M[indices] copy).
            _tiles = []
            if is_split:
                for _b in range(A.nblocks):
                    for _t in range(A.nranges):
                        _d, _i, _ip = A.sub[_b][_t]
                        # range-local ids index the range's slice of M
                        _Mt = M[int(A.cuts[_t]):int(A.cuts[_t + 1])]
                        for start in range(0, _d.size, chunk_size):
                            _tiles.append(((_d, _i, _Mt), start,
                                           min(start + chunk_size, _d.size)))
            else:
                for _blk in (A.blocks if is_block else (A,)):
                    for start in range(0, _blk.data.size, chunk_size):
                        _tiles.append((_blk, start,
                                       min(start + chunk_size, _blk.data.size)))

            def _scale_tile(t):
                _blk, s0, s1 = t
                if isinstance(_blk, tuple):        # partitioned storage: local ids
                    _bd, _bc, _Mt = _blk
                    _bd[s0:s1] *= _Mt[_bc[s0:s1]].astype(_bd.dtype, copy=False)
                    return
                _bd, _bc = _blk.data, _blk.indices
                _bd[s0:s1] *= M[_bc[s0:s1]].astype(_bd.dtype, copy=False)

            _nsc = min(8, max(1, n_threads), max(1, len(_tiles)))
            if _nsc > 1:
                with ThreadPoolExecutor(max_workers=_nsc) as _ex:
                    list(_ex.map(_scale_tile, _tiles))
            else:
                for _t in _tiles:
                    _scale_tile(_t)
            del _tiles
            x0_solver = x0_compressed * M_inv.astype(x0_compressed.dtype) if x0_compressed is not None else None
            # Only M (post-solve unscaling) is needed from here on; the
            # squared norms, the norm vector and the unscaled x0 are dead
            # weight for the whole solve (3 n-length vectors).
            del col_sq_norm, col_norms, M_inv
            x0_compressed = None
        else:
            M = None
            x0_solver = x0_compressed
            x0_compressed = None

        logger.info(f"Solving least squares for {n_active} unknowns with {A_shape[0]} equations (solver={solver}).")
        A_csr = A
        del A
        # x0_solver is a private temporary: hand it to lsqr_inplace as the
        # solution buffer (x0_owned) instead of letting it copy it.
        _x0_owned = [x0_solver]
        del x0_solver
        # lsqr_inplace (bit-identical to scipy's lsqr, in-place vector
        # updates) reads b exactly once; hand it over without keeping a
        # reference so the solver can release the m-length vector for the
        # rest of the solve. lsmr keeps scipy's implementation.
        _b_owned = [b]
        del b

        if is_block or is_split:
            op = (_make_parallel_operator_colsplit(A_csr, n_threads, A_csr.nranges)
                  if is_split else
                  _make_parallel_operator_blocks(A_csr, n_threads, a_owned=a_owned))
            try:
                with threadpool_limits(limits=1, user_api='blas'):
                    if solver == 'lsmr':
                        result = lsmr(op, _b_owned[0], x0=_x0_owned[0], show=True, atol=atol, btol=btol, damp=damp, maxiter=iter_lim)
                    elif solver == 'lsqr':
                        result = lsqr_inplace(op, _b_owned.pop(), x0=_x0_owned.pop(), x0_owned=True, show=True, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim)
                    else:
                        raise ValueError(f"Unknown solver: {solver}. Use 'lsqr' or 'lsmr'.")
            finally:
                for _ex_name in ('_executor', '_rmatvec_executor', '_rowsplit_executor'):
                    if getattr(op, _ex_name, None) is not None:
                        getattr(op, _ex_name).shutdown(wait=False)
        elif n_threads > 1:
            op = _make_parallel_operator(A_csr, n_threads)
            try:
                with threadpool_limits(limits=1, user_api='blas'):
                    if solver == 'lsmr':
                        result = lsmr(op, _b_owned[0], x0=_x0_owned[0], show=True, atol=atol, btol=btol, damp=damp, maxiter=iter_lim)
                    elif solver == 'lsqr':
                        result = lsqr_inplace(op, _b_owned.pop(), x0=_x0_owned.pop(), x0_owned=True, show=True, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim)
                    else:
                        raise ValueError(f"Unknown solver: {solver}. Use 'lsqr' or 'lsmr'.")
            finally:
                op._executor.shutdown(wait=False)
                if getattr(op, '_rmatvec_executor', None) is not None:
                    op._rmatvec_executor.shutdown(wait=False)
        else:
            if solver == 'lsmr':
                result = lsmr(A_csr, _b_owned[0], x0=_x0_owned[0], show=True, atol=atol, btol=btol, damp=damp, maxiter=iter_lim)
            elif solver == 'lsqr':
                result = lsqr_inplace(A_csr, _b_owned.pop(), x0=_x0_owned.pop(), x0_owned=True, show=True, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim)
            else:
                raise ValueError(f"Unknown solver: {solver}. Use 'lsqr' or 'lsmr'.")
        x_solver = result[0]
        del A_csr
        if precondition:
            x_solver = x_solver * M
        if active_mask is not None:
            x = np.zeros(num_cols, dtype=x_solver.dtype)
            x[active_mask] = x_solver
        else:
            x = x_solver
        return x

    # ---- Legacy path: A is a COO. apply_lsqr does the compaction itself.
    num_cols = A.shape[1]

    # --- Fused preprocessing: column elimination + float32 + preconditioning + CSR ---
    col_nnz = np.bincount(A.col, minlength=num_cols)
    active_mask = col_nnz > 0
    num_active = int(np.sum(active_mask))

    if num_active < num_cols:
        logger.info(f"Eliminating {num_cols - num_active} zero columns ({num_active}/{num_cols} active)...")
        col_map = np.full(num_cols, -1, dtype=A.col.dtype)
        col_map[active_mask] = np.arange(num_active, dtype=A.col.dtype)
        new_col = col_map[A.col]
        x0_compressed = x0[active_mask] if x0 is not None else None
    else:
        new_col = A.col
        x0_compressed = x0
        active_mask = None
    x0 = None  # release the full-layout x0 (see fast-path comment)

    n_active = num_active if active_mask is not None else num_cols

    if use_float32:
        logger.info("Downcasting to float32 for faster SpMV...")
        # setup_lsqr workers always emit float32 for sub_data_vec, so A.data
        # is already f32 in production; skip the redundant nnz-sized f32 copy
        # (4 bytes per nonzero — tens of GB at production nnz ~1e10).
        if A.data.dtype == np.float32:
            data = A.data
        else:
            data = A.data.astype(np.float32)
        # Drop the f64 b reference once the f32 cast exists (caller already
        # released self.b; this releases the local f64 reference — 8 bytes
        # per equation, e.g. ~80 GB at ~1e10 equations).
        _b_in = b
        b = _b_in.astype(np.float32)
        del _b_in
        if x0_compressed is not None:
            x0_compressed = x0_compressed.astype(np.float32)
    else:
        data = A.data

    if precondition:
        logger.info("Applying column-norm preconditioning...")
        # Chunked float32 accumulation of column-squared-norms.
        # Avoids materializing the full nnz-sized f64 (data**2) temp (8 bytes
        # per nonzero — e.g. ~56 GB at nnz ~7e9). f32 sum is safe: max per-column sum is bounded
        # (max data ~10 from apply_weight * max ~17k contributors ~1.7M, << f32 max 3.4e38).
        chunk_size = 64_000_000  # ~256 MB per chunk at f32
        col_sq_norm = np.zeros(n_active, dtype=np.float32)
        for start in range(0, data.size, chunk_size):
            stop = min(start + chunk_size, data.size)
            d_chunk = data[start:stop]
            c_chunk = new_col[start:stop]
            col_sq_norm += np.bincount(c_chunk, weights=d_chunk * d_chunk, minlength=n_active).astype(np.float32)
        col_norms = np.sqrt(col_sq_norm)
        col_norms[col_norms == 0] = 1.0
        M_inv = col_norms
        M = 1.0 / M_inv
        # Chunked in-place gather-multiply: avoids two full-nnz transients
        # (the M[new_col] gather and the .astype(data.dtype) copy, ~4 bytes
        # per nonzero each — e.g. ~52 GB combined at production nnz). M values are tiny
        # (n_active entries), so per-chunk gather is cheap.
        chunk_size = 64_000_000  # ~256 MB per chunk at f32
        for start in range(0, data.size, chunk_size):
            stop = min(start + chunk_size, data.size)
            data[start:stop] *= M[new_col[start:stop]].astype(data.dtype, copy=False)
        x0_solver = x0_compressed * M_inv.astype(x0_compressed.dtype) if x0_compressed is not None else None
    else:
        M = None
        x0_solver = x0_compressed

    # Bind the row array + shape locally so we can drop the COO container immediately
    # after CSR build. Caller already released its reference (see Calibrator.apply_lsqr);
    # this lets the COO's row/col/data arrays (~12-16 bytes per nonzero —
    # e.g. ~140 GB at nnz ~1e10) be freed as soon as CSR construction
    # finishes. row/col are scipy properties without deleters, so we
    # drop A itself after rebinding row locally.
    A_row = A.row
    A_shape = A.shape
    del A
    logger.info(f"Solving least squares for {n_active} unknowns with {A_shape[0]} equations (solver={solver}).")
    A_csr = coo_matrix((data, (A_row, new_col)), shape=(A_shape[0], n_active)).tocsr()
    del data, new_col, A_row

    # --- Build parallel operator or use CSR directly ---
    if n_threads > 1:
        op = _make_parallel_operator(A_csr, n_threads)
        try:
            with threadpool_limits(limits=1, user_api='blas'):
                if solver == 'lsmr':
                    result = lsmr(op, b, x0=x0_solver, show=True, atol=atol, btol=btol, damp=damp, maxiter=iter_lim)
                elif solver == 'lsqr':
                    result = lsqr(op, b, x0=x0_solver, show=True, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim)
                else:
                    raise ValueError(f"Unknown solver: {solver}. Use 'lsqr' or 'lsmr'.")
        finally:
            op._executor.shutdown(wait=False)
            if getattr(op, '_rmatvec_executor', None) is not None:
                op._rmatvec_executor.shutdown(wait=False)
    else:
        if solver == 'lsmr':
            result = lsmr(A_csr, b, x0=x0_solver, show=True, atol=atol, btol=btol, damp=damp, maxiter=iter_lim)
        elif solver == 'lsqr':
            result = lsqr(A_csr, b, x0=x0_solver, show=True, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim)
        else:
            raise ValueError(f"Unknown solver: {solver}. Use 'lsqr' or 'lsmr'.")
    x_solver = result[0]
    del A_csr

    # --- Undo preconditioning ---
    if precondition:
        x_solver = x_solver * M

    # --- Expand back to full column space ---
    if active_mask is not None:
        x = np.zeros(num_cols, dtype=x_solver.dtype)
        x[active_mask] = x_solver
    else:
        x = x_solver

    return x
