"""LSQR solve: preconditioning, parallel SpMV operator, and the solver call.

The solver stage of ``selfcal.core``: matrix/RHS assembly lives in
``assembly.py`` and ``system.py``; this module only consumes the returned
(A, b). ``apply_lsqr`` runs scipy lsqr/lsmr against a thread-parallel SpMV
``LinearOperator``, with Jacobi column-norm preconditioning and — when
``setup_lsqr`` has already dropped all-zero columns and passes an
``active_mask`` — expansion of the compact solution back to the full column
layout.
"""
import os

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import coo_matrix, csr_matrix, _sparsetools
from scipy.sparse.linalg import lsqr, lsmr, LinearOperator
from threadpoolctl import threadpool_limits

from .blockcsr import BlockCSR, _csr_shell


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

    print(f"Building parallel SpMV operator ({n_threads} threads)...")
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

    def _rmatvec(y):
        # Single scipy CSC SpMV — no per-thread slicing needed; the kernel is already
        # vectorized and releases the GIL.
        return AT_view @ y

    op = LinearOperator((m, n), matvec=_matvec, rmatvec=_rmatvec, dtype=A_csr.dtype)
    op._executor = executor
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


def parallel_rmatvec_threads():
    """Thread count for the parallel rmatvec; 0 = off (the default).

    ``A^T @ y`` is a scatter (every matrix row adds into scattered output
    columns), so unlike matvec it cannot be threaded without changing the
    order in which each output column's contributions are summed. The
    sequential kernel is therefore the default, and it dominates the solve:
    matvec runs on many threads while this runs on one, so at production
    scale (nnz ~1e10, far larger than any cache) the LSQR solve spends most
    of its per-iteration time here.

    Enabling ``SELFCAL_PARALLEL_RMATVEC=<n>`` gives each thread a private
    output buffer and reduces them afterwards — race-free and DETERMINISTIC
    (a fixed thread count always yields the same bytes). Two consequences,
    both measured on a production-size tile (one 2x2-tiling block of a
    full-NEP detector run, ~1100 frames, nnz ~1e10), both the reason this
    is opt-in:

    * NOT bit-identical to the sequential kernel. Per-column sums become a
      tree instead of one chain, a float32 reassociation of ~1.5e-6 L2 in
      rmatvec (median exactly 0) that propagates to ~1e-4 of each converged
      sky map's own scatter (Pearson 1.0 vs the sequential cal; well inside
      the solver's atol/btol=1e-6). Scientifically equivalent, but the
      byte-equality regression baselines (the reference cal_*.h5 outputs
      that diff_cal_h5.py compares new runs against) must be regenerated
      once to adopt it, and the output then depends on the thread count (a
      different ``n`` -> a different partition -> different bytes at the
      ~1e-6 level).
    * The real speedup is modest. rmatvec is memory-bandwidth-bound at
      production scale, so parallelising it gives ~1.5x per LSQR iteration
      and ~1.3x on the whole tile (e.g. 8994 -> 6812 s on the tile above) —
      not the larger figure an isolated-kernel microbenchmark suggests (its
      smaller matrix fits cache; the real one does not). 8 threads was the
      sweet spot on a 192-physical-core box; past that the O(n_threads x
      n_cols) per-thread-buffer reduction costs more than it saves.
    """
    try:
        n = int(os.environ.get('SELFCAL_PARALLEL_RMATVEC', '0'))
    except ValueError:
        n = 0
    return max(0, n)


def _row_pieces(bcsr, n_pieces):
    """Split the rows into ``n_pieces`` contiguous chunks of storage shells.

    Each piece is a list of (block, local_start, local_end) covering one
    contiguous global row range, so a thread can scatter its own rows without
    touching another thread's.
    """
    m = bcsr.shape[0]
    cuts = np.linspace(0, m, max(1, n_pieces) + 1, dtype=np.int64)
    pieces = []
    for r0, r1 in zip(cuts[:-1], cuts[1:]):
        r0, r1 = int(r0), int(r1)
        if r1 <= r0:
            continue
        spans = []
        for bi, blk in enumerate(bcsr.blocks):
            b0 = int(bcsr.row_bounds[bi])
            b1 = int(bcsr.row_bounds[bi + 1])
            lo, hi = max(r0, b0), min(r1, b1)
            if lo < hi:
                spans.append((blk, b0, lo, hi))
        if spans:
            pieces.append((r0, r1, spans))
    return pieces


def _make_parallel_rmatvec(bcsr, n_threads, out_n, dtype):
    """Parallel scatter rmatvec: private per-thread buffers + reduction.

    Each thread scatters a disjoint row range with scipy's own C kernel (which
    releases the GIL) into a buffer it alone owns, so no numba and no locking.
    The buffers are allocated ONCE per operator, not per call.

    Determinism: the row partition and the reduction order are both fixed at
    construction, so repeated calls on the same input return identical bytes.
    """
    pieces = _row_pieces(bcsr, n_threads)
    bufs = [np.zeros(out_n, dtype=dtype) for _ in pieces]
    ex = ThreadPoolExecutor(max_workers=len(pieces))
    print(f"  rmatvec: PARALLEL scatter over {len(pieces)} threads "
          f"(+{len(pieces) * out_n * np.dtype(dtype).itemsize / 2**30:.2f} GB "
          f"of private buffers) — NOT bit-identical to the sequential kernel.")

    def _rmatvec(y):
        y = np.ascontiguousarray(y, dtype=dtype)

        def _work(i):
            r0, r1, spans = pieces[i]
            buf = bufs[i]
            buf[:] = 0
            for blk, b0, lo, hi in spans:
                l0, l1 = lo - b0, hi - b0
                s0 = int(blk.indptr[l0])
                _sparsetools.csc_matvec(
                    out_n, hi - lo,
                    blk.indptr[l0:l1 + 1] - blk.indptr[l0],
                    blk.indices[s0:int(blk.indptr[l1])],
                    blk.data[s0:int(blk.indptr[l1])],
                    y[lo:hi], buf)

        list(ex.map(_work, range(len(pieces))))
        out = bufs[0].copy()
        for i in range(1, len(bufs)):        # fixed order => deterministic
            out += bufs[i]
        return out

    return _rmatvec, ex



def _make_parallel_operator_blocks(bcsr, n_threads):
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
    print(f"Building parallel SpMV operator ({n_threads} threads, "
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

    # Opt-in parallel scatter. Only for the all-one-dtype case: the private
    # buffers are typed at construction, so a y of a different dtype would
    # change the promotion and is left to the sequential kernel.
    _par_threads = parallel_rmatvec_threads()
    _rmatvec_parallel, _par_ex = (
        _make_parallel_rmatvec(bcsr, _par_threads, n, dtype)
        if _par_threads > 1 else (None, None))

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

def apply_lsqr(A, b, ref_shape, x0=None,
                atol=1e-05, btol=1e-05, damp=1e-2, iter_lim=100, precondition=True,
                solver='lsmr', use_float32=False, n_threads=32,
                active_mask=None, num_cols_full=None):
    """Applies LSQR or LSMR to solve for the sky and detector offsets.

    Parameters
    ----------
    A : coo_matrix or csr_matrix
        Sparse system matrix. When ``csr_matrix`` is passed together with an
        ``active_mask``, setup_lsqr is assumed to have already compacted the
        zero columns; ``apply_lsqr`` skips its own column elimination and
        uses ``active_mask`` only to expand the solution back to the full
        column space at the end.
    solver : str, optional
        Solver to use: 'lsmr' (default, faster convergence) or 'lsqr'.
    use_float32 : bool, optional
        If True, cast matrix data and b to float32 before solving.
        Reduces memory bandwidth (~2x faster SpMV) at the cost of precision.
    active_mask : np.ndarray of bool, optional
        When set, marks the columns of the original (uncompacted) layout
        that are present in the supplied compact CSR. Used to expand the
        compact solution back to the full column space on return.
    num_cols_full : int, optional
        Original (uncompacted) column count. Required when ``active_mask``
        is given. Equals ``A.shape[1]`` when no compaction happened upstream.
    """
    assert isinstance(A, (coo_matrix, csr_matrix, BlockCSR)), \
        "A must be a scipy.sparse.coo_matrix, csr_matrix, or BlockCSR"
    assert isinstance(b, np.ndarray), "b must be a numpy array"
    assert isinstance(ref_shape, (list, np.ndarray, tuple)) and len(ref_shape) == 2, "ref_shape must be a list or tuple of length 2"

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
    if isinstance(A, (csr_matrix, BlockCSR)):
        is_block = isinstance(A, BlockCSR)
        if active_mask is not None:
            assert num_cols_full is not None, \
                "num_cols_full must be supplied alongside active_mask"
            num_cols = int(num_cols_full)
            n_active = int(active_mask.sum())
            assert A.shape[1] == n_active, (
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
            print("Downcasting to float32 for faster SpMV...")
            for _blk in (A.blocks if is_block else (A,)):
                if _blk.data.dtype != np.float32:
                    _blk.data = _blk.data.astype(np.float32)
            _b_in = b
            b = _b_in.astype(np.float32)
            del _b_in
            if x0_compressed is not None:
                x0_compressed = x0_compressed.astype(np.float32)

        if precondition:
            print("Applying column-norm preconditioning...")
            chunk_size = 64_000_000  # ~256 MB per chunk at f32
            col_sq_norm = np.zeros(n_active, dtype=np.float32)
            if is_block:
                # Same GLOBAL chunk cuts as the unified loop below (the
                # float32 partial-sum tree is part of the byte-equal
                # contract); straddling chunks are concatenated inside the
                # iterator, never split.
                for d_chunk, c_chunk in _iter_global_entry_chunks(A.blocks, chunk_size):
                    col_sq_norm += np.bincount(c_chunk, weights=d_chunk * d_chunk, minlength=n_active).astype(np.float32)
            else:
                data = A.data
                new_col = A.indices
                for start in range(0, data.size, chunk_size):
                    stop = min(start + chunk_size, data.size)
                    d_chunk = data[start:stop]
                    c_chunk = new_col[start:stop]
                    col_sq_norm += np.bincount(c_chunk, weights=d_chunk * d_chunk, minlength=n_active).astype(np.float32)
            col_norms = np.sqrt(col_sq_norm)
            col_norms[col_norms == 0] = 1.0
            M_inv = col_norms
            M = 1.0 / M_inv
            # Elementwise in-place scaling: chunk boundaries are free here
            # (no cross-entry accumulation), so the block path scales each
            # block's arrays directly.
            for _blk in (A.blocks if is_block else (A,)):
                _bd, _bc = _blk.data, _blk.indices
                for start in range(0, _bd.size, chunk_size):
                    stop = min(start + chunk_size, _bd.size)
                    _bd[start:stop] *= M[_bc[start:stop]].astype(_bd.dtype, copy=False)
            x0_solver = x0_compressed * M_inv.astype(x0_compressed.dtype) if x0_compressed is not None else None
        else:
            M = None
            x0_solver = x0_compressed

        print(f"Solving least squares for {n_active} unknowns with {A_shape[0]} equations (solver={solver}).")
        A_csr = A
        del A

        if is_block:
            op = _make_parallel_operator_blocks(A_csr, n_threads)
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
        elif n_threads > 1:
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
                # _make_parallel_operator has no rmatvec executor to shut
                # down (only the BlockCSR operator creates one).
                op._executor.shutdown(wait=False)
        else:
            if solver == 'lsmr':
                result = lsmr(A_csr, b, x0=x0_solver, show=True, atol=atol, btol=btol, damp=damp, maxiter=iter_lim)
            elif solver == 'lsqr':
                result = lsqr(A_csr, b, x0=x0_solver, show=True, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim)
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
        print(f"Eliminating {num_cols - num_active} zero columns ({num_active}/{num_cols} active)...")
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
        print("Downcasting to float32 for faster SpMV...")
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
        print("Applying column-norm preconditioning...")
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
    print(f"Solving least squares for {n_active} unknowns with {A_shape[0]} equations (solver={solver}).")
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
