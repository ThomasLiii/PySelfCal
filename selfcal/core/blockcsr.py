"""Row-block representation of a CSR matrix too large for int32 indices.

A single scipy CSR must store ``indptr`` values up to nnz, so once total nnz
exceeds 2**31 scipy forces BOTH indptr and indices to int64 — pure index
overhead of nnz*4 bytes (e.g. ~78 GiB at nnz ~2.1e10, typical of a
multi-thousand-frame production tile) plus an
nnz*8 upcast copy at construction, even though the column ids themselves fit
int32 comfortably. Splitting the rows into blocks whose per-block nnz stays
below 2**31 keeps every stored integer in int32; nothing numerical reads the
index width, and both SpMV directions can be made bit-identical to the
unified matrix (matvec is row-local, so blocking cannot change any row's
accumulation; the transpose product is reproduced by scattering each block
sequentially into one shared output — the same C loop split across row
ranges).

``BlockCSR`` is intentionally minimal: an ordered list of scipy
``csr_matrix`` row-blocks plus their global row boundaries. Only
``core.solve.apply_lsqr`` and ``core.solution.compute_x0_scalar_only``
consume it. ``setup_lsqr`` emits it only when total nnz crosses the int64
threshold (env ``SELFCAL_BLOCK_NNZ`` overrides, for tests), so every
smaller run keeps the plain ``csr_matrix`` path byte-for-byte.
"""
import numpy as np
from scipy.sparse import csr_matrix


class BlockCSR:
    """Ordered row-blocks of one logical CSR matrix.

    Attributes
    ----------
    blocks : list of scipy.sparse.csr_matrix
        Row-blocks in row order. Each block's nnz < 2**31 so its
        indices/indptr are int32.
    row_bounds : np.ndarray, shape (n_blocks + 1,)
        Global row index of each block boundary; block i covers rows
        ``row_bounds[i]:row_bounds[i+1]``.
    shape : tuple
        Logical (n_rows, n_cols) of the full matrix.
    """

    def __init__(self, blocks, row_bounds, shape):
        self.blocks = list(blocks)
        self.row_bounds = np.asarray(row_bounds, dtype=np.int64)
        self.shape = (int(shape[0]), int(shape[1]))

    @property
    def dtype(self):
        return self.blocks[0].data.dtype

    @property
    def nnz(self):
        return int(sum(blk.nnz for blk in self.blocks))

    def __repr__(self):
        return (f"<BlockCSR shape={self.shape} nnz={self.nnz} "
                f"blocks={len(self.blocks)}>")


def _csr_shell(data, indices, indptr, shape):
    """Zero-copy csr_matrix from pre-validated arrays.

    The public ``csr_matrix((data, indices, indptr))`` constructor runs
    index-dtype unification with content checks and will silently COPY the
    index arrays whenever their dtypes can be "improved" — exactly the copy
    this module exists to avoid. ``__new__`` + attribute assignment keeps the
    supplied arrays as-is; callers guarantee consistency.
    """
    blk = csr_matrix.__new__(csr_matrix)
    blk._shape = (int(shape[0]), int(shape[1]))
    blk.data = data
    blk.indices = indices
    blk.indptr = indptr
    return blk


def build_block_csr(data, indices, indptr, shape, target_nnz):
    """Split (data, indices int32, indptr int64) into a BlockCSR.

    Row cuts are nnz-aware: each block gets ~``target_nnz`` entries (never
    more than target_nnz + the largest single row, and a row always fits —
    row nnz <= n_cols < 2**31). The int32 ``indices`` array is SLICED into
    views (no copy); only each block's shifted indptr is a fresh int32 array.
    The caller must not reuse ``indptr`` afterwards (it can be freed).
    """
    n_rows = int(shape[0])
    total_nnz = int(indptr[-1])
    n_blocks = max(1, int(np.ceil(total_nnz / float(target_nnz))))
    # Row boundaries where the cumulative nnz crosses each target multiple.
    targets = (np.arange(1, n_blocks) * (total_nnz / n_blocks)).astype(np.int64)
    cuts = np.searchsorted(indptr, targets, side='left')
    bounds = np.unique(np.concatenate(([0], cuts, [n_rows])))
    blocks = []
    for i in range(len(bounds) - 1):
        sr, er = int(bounds[i]), int(bounds[i + 1])
        s0, s1 = int(indptr[sr]), int(indptr[er])
        if s1 - s0 >= 2**31:
            raise ValueError(
                f"block rows [{sr},{er}) hold {s1 - s0} entries (>= 2^31); "
                "single rows this dense cannot be int32-indexed")
        local_indptr = (indptr[sr:er + 1] - s0).astype(np.int32)
        blocks.append(_csr_shell(data[s0:s1], indices[s0:s1], local_indptr,
                                 (er - sr, shape[1])))
    return BlockCSR(blocks, bounds, shape)


def _move_forward(arr, src, dst, n, chunk=1 << 26):
    """In-place ``arr[dst:dst+n] = arr[src:src+n]`` for ``dst <= src``.

    Walks forward in fixed chunks so no chunk overwrites a later chunk's
    source (``dst + k + chunk <= src + k + chunk``); ``np.copyto`` buffers the
    intra-chunk overlap itself. Memory: one chunk (256 MB at 4 B/entry).
    """
    assert dst <= src
    for k in range(0, n, chunk):
        m = min(chunk, n - k)
        np.copyto(arr[dst + k:dst + k + m], arr[src + k:src + k + m])


def merge_duplicates_inplace(blocks, arrays, starts):
    """Sort + merge duplicate (row, col) entries of row-blocks WITHOUT copies.

    Equivalent to ``blk.sort_indices(); blk.sum_duplicates()`` on every block
    (the same scipy kernels, ``csr_sort_indices`` and ``csr_sum_duplicates``,
    run on the same per-row entry sequences, so the merged values are
    bit-identical), but the storage is handled here instead of by scipy's
    ``prune()``. ``prune()`` copies a block's ``indices``/``data`` whenever the
    post-merge view is smaller than half its base array — true for every
    block of a BlockCSR — so the pre-merge base arrays stay pinned while
    per-block copies pile up: a ~2x-the-final-CSR transient at the end of
    ``setup_lsqr`` whenever a mode emits duplicates (template / hard
    poly-basis offsets emit one entry per chunk-contrib into the same
    per-frame column). Here each block is merged in place (the kernel
    front-packs within the block's slice), the packed runs are slid forward
    into one contiguous prefix of the base arrays, and the base arrays are
    shrunk with ``ndarray.resize`` — an in-place ``realloc``, no second copy
    of the matrix at any moment.

    Parameters
    ----------
    blocks : list of csr_matrix
        Row-blocks whose ``data`` / ``indices`` are VIEWS into ``arrays``
        starting at ``starts[i]`` (``build_block_csr`` output, or a single
        unified ``csr_matrix`` with ``starts=[0]``). ``indptr`` must be a
        private array per block (it is rewritten in place).
    arrays : list ``[data, indices]``
        The two base arrays, passed in a list the CALLER NO LONGER REFERENCES
        by name: ``ndarray.resize`` refuses when any other reference or view
        exists. The list is emptied and the (possibly shrunk) arrays returned.
    starts : list of int
        Offset of each block's first entry in the base arrays (pre-merge).

    Returns
    -------
    (data, indices, nnz_before, nnz_after)
    """
    from scipy.sparse import _sparsetools
    data, indices = arrays.pop(0), arrays.pop(0)
    assert not arrays
    nnz_before = int(sum(int(b.indptr[-1]) for b in blocks))
    spans = []
    g = 0
    for blk, s0 in zip(blocks, starts):
        assert int(blk.indptr[-1]) == blk.indices.shape[0] == blk.data.shape[0]
        blk.has_sorted_indices = False
        blk.sort_indices()                      # in place, per row
        n_rows, n_cols = blk.shape
        _sparsetools.csr_sum_duplicates(n_rows, n_cols, blk.indptr,
                                        blk.indices, blk.data)   # in place
        n_blk = int(blk.indptr[-1])
        if g != s0:
            _move_forward(data, s0, g, n_blk)
            _move_forward(indices, s0, g, n_blk)
        spans.append((g, n_blk))
        g += n_blk
    if g < nnz_before:
        # Drop every view before shrinking; glibc realloc shrinks a large
        # mapping in place (mremap), so this releases the slack without a copy.
        for blk in blocks:
            blk.data = blk.indices = None
        data.resize(g, refcheck=True)
        indices.resize(g, refcheck=True)
    for blk, (g0, n) in zip(blocks, spans):
        blk.data = data[g0:g0 + n]
        blk.indices = indices[g0:g0 + n]
        blk.has_canonical_format = True        # sorted + duplicate-free now
    return data, indices, nnz_before, g
