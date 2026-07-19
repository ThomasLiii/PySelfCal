"""LSQR solve: preconditioning, parallel SpMV operator, and the solver call.

Split out of the former monolithic lsqr.py. ``apply_lsqr`` runs scipy
lsqr/lsmr against a thread-parallel SpMV ``LinearOperator``, with Jacobi
column-norm preconditioning and Top-2 column-compaction expansion. Independent
of the assembly/system halves (operates on the returned A, b).
"""
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.linalg import lsqr, lsmr, LinearOperator
from threadpoolctl import threadpool_limits


def _partition_csr(A, n_blocks):
    """Split CSR matrix into row-blocks sharing data/indices arrays (zero-copy).

    The blocks are assembled by direct attribute assignment rather than the
    ``csr_matrix((data, indices, indptr), ...)`` constructor: that constructor
    runs scipy's index-dtype unification with ``check_contents=True``, and when
    the parent has int64 indices (forced whenever total nnz > 2^31) every
    block's contents fit int32, so it silently downcast-COPIES all indices —
    ~nnz*4 bytes of duplicates held for the whole solve (~89 GiB at full-NEP
    tile scale; verified on scipy 1.15/1.16). Attribute assignment keeps true
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

    The previous version materialized AT_csr = A_csr.T.tocsr() as an explicit copy, which
    at no-srcmask region-10k scale was ~88 GB. The CSC view shares A_csr's storage and
    costs O(1) — scipy's CSC @ vec is fast and releases the GIL.
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
        zero columns (Top 2 path); ``apply_lsqr`` skips its own column
        elimination and uses ``active_mask`` only to expand the solution
        back to the full column space at the end.
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
    assert isinstance(A, (coo_matrix, csr_matrix)), \
        "A must be a scipy.sparse.coo_matrix or csr_matrix"
    assert isinstance(b, np.ndarray), "b must be a numpy array"
    assert isinstance(ref_shape, (list, np.ndarray, tuple)) and len(ref_shape) == 2, "ref_shape must be a list or tuple of length 2"

    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w

    # ---- Top 2 fast path: A is already compact CSR ------------------
    if isinstance(A, csr_matrix):
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
        # Drop this frame's ref to the full-layout x0 (the caller released its
        # own via the ownership hand-off) so the f64 original can be freed once
        # x0_compressed is cast to float32 below.
        x0 = None

        A_shape = A.shape
        if use_float32:
            print("Downcasting to float32 for faster SpMV...")
            if A.data.dtype != np.float32:
                A.data = A.data.astype(np.float32)
            _b_in = b
            b = _b_in.astype(np.float32)
            del _b_in
            if x0_compressed is not None:
                x0_compressed = x0_compressed.astype(np.float32)

        data = A.data
        new_col = A.indices

        if precondition:
            print("Applying column-norm preconditioning...")
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
            for start in range(0, data.size, chunk_size):
                stop = min(start + chunk_size, data.size)
                data[start:stop] *= M[new_col[start:stop]].astype(data.dtype, copy=False)
            x0_solver = x0_compressed * M_inv.astype(x0_compressed.dtype) if x0_compressed is not None else None
        else:
            M = None
            x0_solver = x0_compressed

        print(f"Solving least squares for {n_active} unknowns with {A_shape[0]} equations (solver={solver}).")
        A_csr = A
        del A

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
        # is already f32 in production; skip the redundant ~25 GB copy.
        if A.data.dtype == np.float32:
            data = A.data
        else:
            data = A.data.astype(np.float32)
        # Drop the f64 b reference once the f32 cast exists (caller already released self.b;
        # this releases the local f64 reference, saving ~80 GB at no-srcmask region-10k scale).
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
        # Avoids materializing the full nnz-sized f64 (data**2) temp (~56 GB at no-srcmask
        # region-10k scale). f32 sum is safe: max per-column sum is bounded
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
        # (the M[new_col] gather and the .astype(data.dtype) copy together
        # held ~52 GB at no-srcmask region-10k scale). M values are tiny
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
    # this lets the (~140 GB at no-srcmask region-10k scale) COO arrays be freed as soon
    # as CSR construction finishes. row/col are scipy properties without deleters, so we
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
