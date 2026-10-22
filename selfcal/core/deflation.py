"""Deflation: solve the smooth sky patterns of a selfcal system exactly, iterate on the rest.

The slowest directions of a selfcal least-squares system are sky patterns much larger than a frame:
each frame's offsets absorb almost all of such a pattern's signal, so the data constrain it only
weakly, and a Krylov solver (LSQR, LSMR) resolves it last. How slowly depends on the offset model
(the more of a smooth pattern its offsets absorb, the slower) and on the field (the larger, the
more such patterns): from tens of iterations to thousands. Deflation removes them from the
iterations.

**The space.** For each sky term, the patterns ``f_a(x) f_b(y)`` (``a, b = 0 ... order``) over the
bounding box of the term's covered pixels: cosines ``cos(pi a u)`` (``u`` from 0 to 1 across the
box) or Legendre polynomials, plus the Legendre polynomials of degree at most :data:`POLY_ALWAYS`
(so a plane or quadratic gauge is in the space whole, never in part). A pattern alone is not slow (the offsets cannot leave it); a pattern
together with the offsets that absorb it best is. So each pattern ``F`` is paired with its exact
offset response,

    g(F) = argmin_g | B_sky F + B_off g |,

the least squares over every non-sky column (offsets of any form, per-frame scalars, templates ...),
solved directly: the normal matrix of those columns, ``B_off^T B_off``, is built from the system
itself and solved exactly on each of its connected blocks (a block per frame, or per group of frames
that share offsets), by the pseudo-inverse (the response of minimum norm). A column that many frames
share (a detector-fixed pattern, whatever its term) would merge every block it touches into one; such
columns, coupled to far more columns than the block's others, are eliminated first by a Schur
complement over them, which leaves the blocks per frame (:func:`_schur_pinv_solve`). The pairs ``z = (F, g(F))``
span the deflation space. A combination the data cannot see at all (``|B z|`` below
:data:`GAUGE_RATIO` of ``|B F|``: a gauge, such as the sky's mean against per-frame scalars) is left
out, and the rest is made orthogonal to it, so the solution keeps the gauge a plain solve gives
(the minimum-norm one).

**The solve.** With ``V`` an orthonormal basis of the space and ``N = B^T B`` (``B`` the column-scaled
system the solver sees), ``E = V^T N V`` and ``W = N V``:

    x = x_c + P y,   x_c = V E^-1 V^T B^T b,   P = I - V E^-1 W^T,

where ``y`` solves ``min | B P y - (b - B x_c) |`` with the unchanged solver (LSQR or LSMR, in the
same precision): the range of ``B P`` is orthogonal to that of ``B V``, so the part in the space is
solved exactly at the start and the iterations run on the complement only, where the slow patterns
are gone. ``x`` is the least-squares solution of the full system (the same one, gauge included);
its residual and gradient are those of the full system at every iteration. With LSQR or LSMR, the
solver's right Lanczos vectors are also projected onto the complement of the space after each
normalisation (in exact arithmetic they are there already; rounding, which the recurrence amplifies
once the solve has converged, would otherwise let the iterate drift along the directions ``B P``
cannot see: :meth:`Deflation.constrain`); CGLS's gradients stay there by construction, and it takes
the two projections only. At the end, the part in the space is solved again against the final
iterate in float64 (one pass over the matrix), which removes the rounding of ``P``.

**Cost.** Setup: the patterns' products with the matrix in float64, two passes over it with all
the patterns at once (forward and transposed), plus one for the start; per iteration, four passes over
``V`` and ``W`` (kept in the products' precision, ``2 x 4 x patterns`` bytes per covered column with
float32 products, the arithmetic in float64 on cache-sized blocks: ``P`` in the product and ``P^T`` in the
transposed product; six with LSQR's guard), threaded; at the end one
pass for the re-solve (and one per snapshot). Only the covered columns take part: a system that
was not compacted (all-zero columns kept) costs as one that was.

:class:`Deflation` is used by :func:`~selfcal.core.solve.apply_lsqr` (``deflate=``; the run engine's
``sc.Fit(deflate=sc.Deflate(...))``).
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy.polynomial import legendre as _legendre
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse import _sparsetools
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, minres

from selfcal.core.projector import ColumnProjector

from .blockcsr import BlockCSR, ColSplitCSR

logger = logging.getLogger(__name__)

__all__ = ['Deflation', 'GAUGE_RATIO', 'axis_functions', 'storage_of']

#: A combination of the paired patterns whose residual ``|B z|`` is below about this fraction of the
#: residual of its sky part alone, ``|B F|``, is a gauge (a direction the data cannot see at all):
#: it is left out of the space, and the rest is orthogonal to it (:data:`DEPENDENT`).
GAUGE_RATIO = 1e-6
#: Directions of the space whose normal-equation eigenvalue (or Euclidean Gram eigenvalue) is below
#: this fraction of the largest are dropped: numerically dependent, or gauges (directions the data
#: cannot see at all: their eigenvalue is the square of their ``|B z| / |B F|``, so this is
#: :data:`GAUGE_RATIO` squared).
DEPENDENT = 1e-12
#: The polynomial patterns every basis includes: the products of Legendre polynomials of total degree
#: at most this. A gauge the patterns contain only in part (a sky pattern the offsets absorb exactly,
#: such as a plane under per-chunk slopes on a flat geometry) would be set differently from a plain
#: solve and let the iterate drift along it after convergence; with the low-degree polynomials always in
#: the space, such gauges up to this degree are recognised whole (and left out).
POLY_ALWAYS = 2
#: Rows of the matrix per chunk of the float64 passes.
_ROWS = 1 << 21
#: Connected blocks of offset columns up to this size are solved densely (larger ones by MINRES, unless
#: hub columns explain their size).
_DENSE_BLOCK = 4096
#: In a block larger than :data:`_DENSE_BLOCK`, a column coupled to more than this multiple of the
#: block's median coupling count is a hub (an offset many frames share); the hubs are eliminated by a
#: Schur complement, which leaves the per-frame blocks.
_HUB_RATIO = 3.0
#: Up to this many hub columns are eliminated densely (the Schur complement is a dense matrix of that size).
_SCHUR_MAX = 1 << 14
#: Elements of the hub-column blocks held at once in the Schur complement's passes.
_SCHUR_BATCH = 1 << 26
#: Elements per thread block of the projections.
_MIN_BLOCK = 1 << 18
#: Sky columns per call when a chunk's column sums are made and added in the second pass (each adding thread's
#: buffer holds this many columns of the products).
_ADD_COLUMNS = 1 << 16
#: Entries of the offsets' normal matrix read at once when the hub columns' blocks are taken out of it.
_EXTRACT_ENTRIES = 1 << 24
#: Threads (and items in flight) for the deflation's work in many small BLAS calls (the projections' sub-block
#: products, the Schur complement's batches): OpenBLAS serialises concurrent calls on its buffer lock, so more threads
#: run slower, and each batch in flight holds its own buffers.
_BLAS_THREADS = 6


def axis_functions(basis, order, n) -> np.ndarray:
    """The 1-D functions of a pattern basis at ``n`` cells spanning [0, 1] (cell centres), ``(order + 1, n)``:
    ``cos(pi a u)`` for ``basis='cosine'``, the Legendre polynomial ``P_a(2 u - 1)`` for ``'polynomial'``."""
    u = (np.arange(n, dtype=np.float64) + 0.5) / max(int(n), 1)
    a = np.arange(int(order) + 1)
    if basis == 'cosine':
        return np.cos(np.pi * a[:, None] * u[None, :])
    if basis == 'polynomial':
        t = 2.0 * u - 1.0
        return np.stack([_legendre.legval(t, [0] * int(k) + [1]) for k in a])
    raise ValueError(f"unknown pattern basis {basis!r}: 'cosine' or 'polynomial'")


def storage_of(operator, fallback=None):
    """The matrix storage behind one of :mod:`selfcal.core.solve`'s operators (a csr matrix, a
    :class:`~selfcal.core.blockcsr.BlockCSR` or a :class:`~selfcal.core.blockcsr.ColSplitCSR`), or
    ``fallback`` when the operator keeps none."""
    if getattr(operator, '_AT_view', None) is not None:
        return operator._AT_view.T
    bcsr = getattr(operator, '_bcsr', None)
    if bcsr is not None and len(bcsr.blocks) > 0:
        return bcsr
    split = getattr(operator, '_colsplit', None)
    sub = getattr(operator, '_split_sub', None)
    if split is not None and sub is not None:
        return ColSplitCSR(sub, split.row_bounds, split.cuts, split.shape)
    return fallback


def _piece_index(storage, rows=_ROWS):
    """The row chunks of the matrix as descriptors ``(block, lr0, lr1)`` (cheap: nothing is read), in row order."""
    if isinstance(storage, ColSplitCSR):
        sizes = [int(storage.row_bounds[bi + 1]) - int(storage.row_bounds[bi]) for bi in range(storage.nblocks)]
    else:
        blocks = storage.blocks if isinstance(storage, BlockCSR) else [storage]
        sizes = [blk.shape[0] for blk in blocks]
    out = []
    for bi, nrows in enumerate(sizes):
        for lr0 in range(0, nrows, rows):
            out.append((bi, lr0, min(nrows, lr0 + rows)))
    return out


def _piece(storage, desc, buf=None):
    """One row chunk: ``(r0, r1, [(indptr, column ids, data), ...])`` with a local indptr (int64, from 0) per
    piece; one piece per column range of a ColSplitCSR, one otherwise. ``buf`` (a :class:`_Slot`): the indptr
    and the column ids are written into its buffers (the same values; nothing allocated per chunk)."""
    bi, lr0, lr1 = desc

    def local(ip, s0, idx, s1, cut, t):
        if buf is None:
            return ip[lr0:lr1 + 1].astype(np.int64) - s0, idx[s0:s1].astype(np.int64) + cut
        lip = buf.get(f'lip{t}', lr1 - lr0 + 1, np.int64)
        np.subtract(ip[lr0:lr1 + 1], s0, out=lip, casting='unsafe')
        cols = buf.get(f'cols{t}', s1 - s0, np.int64)
        np.add(idx[s0:s1], cut, out=cols, casting='unsafe')
        return lip, cols

    if isinstance(storage, ColSplitCSR):
        base = int(storage.row_bounds[bi])
        out = []
        for t, (d, idx, ip) in enumerate(storage.sub[bi]):
            s0, s1 = int(ip[lr0]), int(ip[lr1])
            lip, cols = local(ip, s0, idx, s1, int(storage.cuts[t]), t)
            out.append((lip, cols, d[s0:s1]))
        return base + lr0, base + lr1, out
    blocks = storage.blocks if isinstance(storage, BlockCSR) else [storage]
    bounds = storage.row_bounds if isinstance(storage, BlockCSR) else (0,)
    blk, base = blocks[bi], bounds[bi]
    ip = blk.indptr
    s0, s1 = int(ip[lr0]), int(ip[lr1])
    lip, cols = local(ip, s0, blk.indices, s1, 0, 0)
    return int(base) + lr0, int(base) + lr1, [(lip, cols, blk.data[s0:s1])]


def _pieces(storage, rows=_ROWS):
    """The matrix in row chunks (:func:`_piece` of every descriptor of :func:`_piece_index`)."""
    for desc in _piece_index(storage, rows):
        yield _piece(storage, desc)


def _components(G):
    """The connected components of a symmetric sparse matrix's graph, ``(count, labels)``, numbered in the order of
    their smallest column, as scipy's undirected search numbers them: its strongly connected components (those of
    a symmetric graph), which it finds without the transposed copy of the matrix the undirected search makes."""
    n = G.shape[0]
    count, labels = connected_components(G, directed=True, connection='strong')
    first = np.full(count, n, dtype=np.int64)
    np.minimum.at(first, labels, np.arange(n, dtype=np.int64))
    rank = np.empty(count, dtype=labels.dtype)
    rank[np.argsort(first, kind='stable')] = np.arange(count, dtype=labels.dtype)
    return count, rank[labels]


def _bounded_map(pool, fn, items, width=_BLAS_THREADS):
    """``[fn(item) for item in items]``, in ``pool`` with at most ``width`` items in flight (each holds its
    buffers until it returns); the results in item order, so a sum over them keeps its order."""
    if pool is None:
        return [fn(it) for it in items]
    from collections import deque
    out, q = [], deque()
    for it in items:
        if len(q) >= width:
            out.append(q.popleft().result())
        q.append(pool.submit(fn, it))
    while q:
        out.append(q.popleft().result())
    return out


def _block_pinv_solve(G, R, project=None, pool=None):
    """``G^+ R`` for a symmetric positive semi-definite csr ``G`` and ``R`` (n x k), exact on each
    connected block of ``G``: batched dense eigen-decompositions (eigenvalues below
    :data:`DEPENDENT` of the block's largest count as zero: the solution of minimum norm), MINRES on
    blocks larger than :data:`_DENSE_BLOCK`, unless hub columns (coupled to :data:`_HUB_RATIO` times
    the block's median: offsets many frames share) explain their size: those are eliminated first by
    :func:`_schur_pinv_solve`, which leaves the per-frame blocks. ``project`` (a
    :class:`~selfcal.core.projector.ColumnProjector` over these columns, or None): the solution of the
    projected system ``P G P X = P R`` (columns the constraints join become hubs; a dense block or a MINRES
    block holding constrained columns is projected in place). Returns ``(X, info)``."""
    n = G.shape[0]
    X = np.zeros((n, R.shape[1]))
    if n == 0:
        return X, {'blocks': 0, 'largest': 0, 'zero_eigenvalues': 0}
    if project is not None and project.columns.size == 0:
        project = None
    if project is not None:
        R = project.apply(np.array(R, dtype=np.float64, copy=True))
    ncomp, labels = _components(G)
    counts = np.bincount(labels, minlength=ncomp)
    if counts.max() > _DENSE_BLOCK:
        deg = np.diff(G.indptr)
        big = counts[labels] > _DENSE_BLOCK
        hub = np.flatnonzero(big & (deg > _HUB_RATIO * np.median(deg[big]))) if big.any() else np.zeros(0, dtype=np.int64)
        if project is not None:
            hub = np.union1d(hub, project.columns[big[project.columns]])      # constrained columns of big blocks
        if 0 < hub.size <= _SCHUR_MAX:
            out = _schur_pinv_solve(G, R, hub, project=project, pool=pool)
            if out is not None:
                return out
    order = np.argsort(labels, kind='stable')
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    pos = np.empty(n, dtype=np.int64)
    pos[order] = np.arange(n) - np.repeat(starts, counts)
    Gc = G.tocoo()
    er, ec, ev = Gc.row.astype(np.int64), Gc.col.astype(np.int64), Gc.data
    ecomp = labels[er]
    zero = 0
    constrained_comp = np.zeros(ncomp, dtype=bool)
    if project is not None:
        constrained_comp[labels[project.columns]] = True
    for s in np.unique(counts):
        comps = np.flatnonzero(counts == s)
        cols = order[starts[comps][:, None] + np.arange(s)[None, :]]            # (blocks, s)
        if project is not None and constrained_comp[comps].any():
            # a block holding constrained columns: P_b G_b P_b, one block at a time
            keep = ~constrained_comp[comps]
            for ci, cc in zip(comps[~keep], cols[~keep]):
                sel = ecomp == ci
                Gb = coo_matrix((ev[sel], (pos[er[sel]], pos[ec[sel]])), shape=(s, s)).toarray()
                pb = project.restricted_to(cc)
                pb.apply_symmetric(Gb)
                w, U = np.linalg.eigh(Gb)
                ok = w > DEPENDENT * max(float(w.max()), 1e-300)
                zero += int((~ok).sum())
                X[cc] = U @ (np.where(ok, 1.0 / np.where(ok, w, 1.0), 0.0)[:, None] * (U.T @ R[cc]))
            comps, cols = comps[keep], cols[keep]
            if comps.size == 0:
                continue
        if s > _DENSE_BLOCK:
            for ci, cc in zip(comps, cols):
                sel = ecomp == ci
                Gb = coo_matrix((ev[sel], (pos[er[sel]], pos[ec[sel]])), shape=(s, s)).tocsr()
                for j in range(R.shape[1]):
                    X[cc, j] = minres(Gb, R[cc, j], rtol=1e-14, maxiter=20 * s)[0]
            continue
        bidx = np.full(ncomp, -1, dtype=np.int64)
        bidx[comps] = np.arange(comps.size)
        sel = bidx[ecomp] >= 0
        blocks = np.zeros((comps.size, s, s))
        blocks[bidx[ecomp[sel]], pos[er[sel]], pos[ec[sel]]] = ev[sel]
        w, U = np.linalg.eigh(blocks)
        top = np.maximum(w.max(axis=1, keepdims=True), 1e-300)
        ok = w > DEPENDENT * top
        zero += int((~ok).sum())
        inv = np.where(ok, 1.0 / np.where(ok, w, 1.0), 0.0)
        X[cols] = U @ (inv[:, :, None] * (np.swapaxes(U, 1, 2) @ R[cols]))
    return X, {'blocks': int(ncomp), 'largest': int(counts.max()), 'zero_eigenvalues': zero}


def _schur_factor(G, hub, project=None, pool=None):
    """The factorisation behind ``G X = R`` with the hub columns eliminated: for ``G = [[D, B], [B^T, C]]`` (``D`` the other
    columns, block-diagonal once the hubs are out; ``C`` the hubs' own part), ``X_hub`` solves the
    Schur complement ``(C - B^T D^+ B) X_hub = R_hub - B^T D^+ R_other`` by its pseudo-inverse and
    ``X_other = D^+ (R_other - B X_hub)``; ``D^+`` block by block as :func:`_block_pinv_solve` does.
    Then the solution of minimum norm, as the pseudo-inverse gives: the null space of ``G`` (offset
    combinations the data cannot see: gauges) is spanned by ``(-D^+ B v, v)`` over the Schur
    complement's null vectors ``v`` and by the blocks' null vectors that do not couple to the hubs;
    each candidate is checked against ``G`` and the verified ones are projected out of ``X`` (a response
    with a null component would put a gauge into the deflation space in part, and the iterate would
    drift along it). Returns a :class:`_SchurFactor`, or None when the other columns still hold a block larger
    than :data:`_DENSE_BLOCK` (the caller falls back). ``project`` (every constrained column a hub): the Schur
    complement is projected, ``P S P X_hub = P rhs``, so the hubs' response and the null vectors live in the
    allowed subspace. ``pool``: a thread pool for the batches of blocks (each batch's products are the same calls
    as in sequence and the sums over batches keep their order: the result does not depend on the threads)."""
    n = G.shape[0]
    is_hub = np.zeros(n, dtype=bool)
    is_hub[hub] = True
    F, H = np.flatnonzero(~is_hub), np.flatnonzero(is_hub)
    nF, nH = int(F.size), int(H.size)
    fpos = np.full(n, -1, dtype=np.int64)
    fpos[F] = np.arange(nF)
    hpos = np.full(n, -1, dtype=np.int64)
    hpos[H] = np.arange(nH)
    ph = None
    if project is not None:
        if (hpos[project.columns] < 0).any():
            raise ValueError('the Schur path needs every constrained column among the hubs')
        ph = project.mapped(hpos)
    # The other columns' block and the hubs' own part, read off G's rows a chunk at a time in G's entry order
    # (the order of G.tocoo(), so Gff is the same matrix), without a copy of G with int64 indices and masks.
    ff_r, ff_c, ff_v = [], [], []
    C = np.zeros((nH, nH))
    r0 = 0
    while r0 < n:
        r1 = max(r0 + 1, int(np.searchsorted(G.indptr, int(G.indptr[r0]) + _EXTRACT_ENTRIES, side='right')) - 1)
        r1 = min(n, r1)
        s0, s1 = int(G.indptr[r0]), int(G.indptr[r1])
        er = np.repeat(np.arange(r0, r1, dtype=np.int64), np.diff(G.indptr[r0:r1 + 1]))
        ec = G.indices[s0:s1].astype(np.int64)
        ev = G.data[s0:s1]
        hr, hc = is_hub[er], is_hub[ec]
        ff = ~hr & ~hc
        ff_r.append(fpos[er[ff]])
        ff_c.append(fpos[ec[ff]])
        ff_v.append(ev[ff])
        hh = hr & hc
        C[hpos[er[hh]], hpos[ec[hh]]] = ev[hh]
        r0 = r1
    del er, ec, ev, hr, hc, ff, hh
    Gff = coo_matrix((np.concatenate(ff_v), (np.concatenate(ff_r), np.concatenate(ff_c))), shape=(nF, nF)).tocsr()
    del ff_r, ff_c, ff_v
    ncomp, labels = _components(Gff)
    counts = np.bincount(labels, minlength=ncomp)
    if counts.max() > _DENSE_BLOCK:
        return None
    order = np.argsort(labels, kind='stable')
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    pos = np.empty(nF, dtype=np.int64)
    pos[order] = np.arange(nF) - np.repeat(starts, counts)
    Gfc = Gff.tocoo()
    er, ec, ev = Gfc.row.astype(np.int64), Gfc.col.astype(np.int64), Gfc.data
    ecomp = labels[er]
    zero = 0
    groups = []                                        # (block columns (nb, s) in F, U (nb, s, s), 1/w (nb, s))
    for s in np.unique(counts):
        comps = np.flatnonzero(counts == s)
        cols = order[starts[comps][:, None] + np.arange(s)[None, :]]
        bidx = np.full(ncomp, -1, dtype=np.int64)
        bidx[comps] = np.arange(comps.size)
        sel = bidx[ecomp] >= 0
        blocks = np.zeros((comps.size, s, s))
        blocks[bidx[ecomp[sel]], pos[er[sel]], pos[ec[sel]]] = ev[sel]
        w, U = np.linalg.eigh(blocks)
        top = np.maximum(w.max(axis=1, keepdims=True), 1e-300)
        ok = w > DEPENDENT * top
        zero += int((~ok).sum())
        groups.append((cols, U, np.where(ok, 1.0 / np.where(ok, w, 1.0), 0.0), ~ok))
    del Gfc, er, ec, ev, ecomp, Gff

    def run(fn, items):
        """``fn`` over the items, in the pool when there is one (a few in flight); the results in item order."""
        return _bounded_map(pool, fn, items)

    def hub_part(rows):
        """The rows ``rows`` of G restricted to the hub columns, dense ``(len(rows), nH)``."""
        sub = G[rows]
        m = is_hub[sub.indices]
        rid = np.repeat(np.arange(rows.size), np.diff(sub.indptr))
        out = np.zeros((rows.size, nH))
        out[rid[m], hpos[sub.indices[m]]] = sub.data[m]
        return out

    def batches():
        """The batches of blocks: (flat columns in F, blocks, block size, U, 1/w, null flags, the hub part
        (blocks, s, nH)); the hub parts extracted in the pool (independent sparse row slices)."""
        descs = []
        for cols, U, inv, bad in groups:
            nb, s = cols.shape
            step = max(1, _SCHUR_BATCH // max(s * nH, 1))
            for a in range(0, nb, step):
                c = cols[a:a + step]
                descs.append((c.ravel(), c.shape[0], s, U[a:a + step], inv[a:a + step], bad[a:a + step]))

        def with_hub(d):
            flat, nb, s = d[0], d[1], d[2]
            return d + (hub_part(F[flat]).reshape(nb, s, nH),)

        return run(with_hub, descs)

    S = C.copy()
    bl = batches()

    def first(batch):
        flat, nb, s, U, inv, _bad, Bb = batch
        Ut = np.swapaxes(U, 1, 2)
        Yb = U @ (inv[:, :, None] * (Ut @ Bb))
        return Bb.reshape(-1, nH).T @ Yb.reshape(-1, nH)

    for dS in run(first, bl):
        S -= dS
    if ph is not None:
        ph.apply_symmetric(S)
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    ok = w > DEPENDENT * max(float(w.max()), 1e-300)
    zero += int((~ok).sum())
    V0 = V[:, ~ok]                                      # the Schur complement's null vectors (nH x m0)
    nulls = np.concatenate((np.zeros((nF, V0.shape[1])), V0))         # (-D^+ B v, v): the F part filled below
    extra = []                                          # the blocks' null vectors that do not couple to the hubs

    def second(batch):
        flat, nb, s, U, inv, bad, Bb = batch
        Ut = np.swapaxes(U, 1, 2)
        if V0.shape[1]:
            nulls[flat] = -(U @ (inv[:, :, None] * (Ut @ (Bb @ V0)))).reshape(-1, V0.shape[1])
        found = []
        if bad.any():
            bi, ei = np.nonzero(bad)
            for i, e in zip(bi, ei):
                vec = U[i][:, e]
                if np.linalg.norm(Bb[i].T @ vec) <= 1e-10 * max(np.linalg.norm(Bb[i]), 1e-300):
                    u = np.zeros(n)
                    u[F[flat[i * s:(i + 1) * s]]] = vec
                    found.append(u)
        return found

    for found in run(second, bl):
        for u in found:
            if len(extra) < 64:
                extra.append(u)
    cands = []
    if V0.shape[1]:
        full = np.zeros((n, V0.shape[1]))
        full[F] = nulls[:nF]
        full[H] = nulls[nF:]
        cands.append(full)
    if extra:
        cands.append(np.stack(extra, axis=1))
    Q = None
    if cands:
        Nc = np.concatenate(cands, axis=1)
        gscale = max(float(np.abs(G.diagonal()).max()), 1e-300)
        GN = G @ (project.apply(Nc.copy()) if project is not None else Nc)
        if project is not None:
            project.apply(GN)                              # the projected G: P G P
        good = np.linalg.norm(GN, axis=0) <= 1e-9 * gscale * np.maximum(np.linalg.norm(Nc, axis=0), 1e-300)
        if good.any():
            Q = np.linalg.qr(Nc[:, good])[0]
    return _SchurFactor(n=n, F=F, H=H, nF=nF, nH=nH, ph=ph, bl=bl, V=V, inv_w=np.where(ok, 1.0 / np.where(ok, w, 1.0), 0.0),
                        Q=Q, info={'blocks': int(ncomp), 'largest': int(counts.max()), 'zero_eigenvalues': zero,
                                   'hubs': nH, 'shared_columns': H})


class _SchurFactor:
    """The R-independent part of :func:`_schur_pinv_solve` (the blocks' eigendecompositions with their hub parts, the
    Schur complement's eigendecomposition, the null-space projector), reused by :meth:`solve` for any right-hand
    side: a second setup on the same offsets (the automatic exclusion's) costs no block work."""

    def __init__(self, **kw):
        self.__dict__.update(kw)

    def solve(self, R, pool=None):
        """``G^+ R`` as :func:`_schur_pinv_solve` returns it, ``(X, info)``, the same operations in the same order;
        the batches in ``pool`` when given."""
        n, k, F, H, nF, nH, bl = self.n, R.shape[1], self.F, self.H, self.nF, self.nH, self.bl

        def run(fn, items):
            return _bounded_map(pool, fn, items)
        RF, RH = R[F], R[H]
        Rh = RH.copy()

        def first(batch):
            flat, nb, s, U, inv, _bad, Bb = batch
            Ut = np.swapaxes(U, 1, 2)
            Rb = RF[flat].reshape(nb, s, k)
            Zb = U @ (inv[:, :, None] * (Ut @ Rb))
            return Bb.reshape(-1, nH).T @ Zb.reshape(-1, k)

        for dR in run(first, bl):
            Rh -= dR
        if self.ph is not None:
            self.ph.apply(Rh)
        V = self.V
        Xh = V @ (self.inv_w[:, None] * (V.T @ Rh))
        X = np.zeros((n, k))
        X[H] = Xh

        def second(batch):
            flat, nb, s, U, inv, _bad, Bb = batch
            Ut = np.swapaxes(U, 1, 2)
            rhs = RF[flat].reshape(nb, s, k) - Bb @ Xh
            X[F[flat]] = (U @ (inv[:, :, None] * (Ut @ rhs))).reshape(-1, k)

        run(second, bl)
        if self.Q is not None:
            X -= self.Q @ (self.Q.T @ X)
        info = dict(self.info)
        info['factor'] = self
        return X, info


def _schur_pinv_solve(G, R, hub, project=None, pool=None):
    """``G X = R`` with the hub columns eliminated (see :func:`_schur_factor`): the factorisation, then its
    solve. None when the factorisation cannot serve."""
    fac = _schur_factor(G, hub, project=project, pool=pool)
    return None if fac is None else fac.solve(R, pool=pool)



def _prepare_piece(nr, lip, cols, data, colmap, n_sky_S, buf=None, tag=0):
    """A piece's entries as the passes read them, computed once per chunk: ``(lip, pos, d64, sky, off)`` with ``pos``
    the solver columns, ``d64`` the values, ``sky = (indptr, columns, values, rows)`` the sky entries as a csr over
    the chunk's rows (None: none) and ``off = (indptr, offset columns, values)`` the offset entries likewise (None:
    none), all int64 / float64 and in storage order. ``buf`` (a :class:`_Slot`; ``tag`` tells a chunk's pieces
    apart): the arrays are written into its buffers, the same values without the allocations; each row's sky
    entries and offset entries are told apart by a running count of the offsets (a row's offset entries are those
    at or past ``n_sky_S``, in storage order)."""
    if buf is not None:
        return _prepare_piece_into(nr, lip, cols, data, colmap, n_sky_S, buf, tag)
    pos = cols if colmap is None else colmap[cols]
    d64 = data.astype(np.float64)
    rows = np.repeat(np.arange(nr, dtype=np.int64), np.diff(lip))
    o = pos >= n_sky_S
    off = sky = None
    oi = np.flatnonzero(o)
    if oi.size:
        ip = np.zeros(nr + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows[oi], minlength=nr), out=ip[1:])
        off = (ip, (pos[oi] - n_sky_S).astype(np.int64), d64[oi])
    si = np.flatnonzero(~o)
    if si.size:
        ips = np.zeros(nr + 1, dtype=np.int64)
        np.cumsum(np.bincount(rows[si], minlength=nr), out=ips[1:])
        sky = (ips, pos[si], d64[si], rows[si])
    return lip, pos, d64, sky, off


def _prepare_piece_into(nr, lip, cols, data, colmap, n_sky_S, buf, tag):
    """:func:`_prepare_piece` into the buffers of ``buf``: the same arrays (offset counts by a cumulative sum of
    the offset mask instead of a bincount over the rows; the entries selected by ``np.compress``, in storage order)."""
    n = int(cols.size)
    pos = cols if colmap is None else np.take(colmap, cols, out=buf.get(f'{tag}pos', n, np.int64))
    d64 = buf.get(f'{tag}d64', n, np.float64)
    np.copyto(d64, data)
    o = buf.get(f'{tag}o', n, np.bool_)
    np.greater_equal(pos, n_sky_S, out=o)
    c = buf.get(f'{tag}c', n + 1, np.int64)              # c[j]: offset entries among the first j
    c[0] = 0
    np.cumsum(o, out=c[1:])
    ip = buf.get(f'{tag}ip', nr + 1, np.int64)
    np.take(c, lip, out=ip)                               # the offset entries before each row
    n_off = int(c[n])
    off = sky = None
    if n_off:
        oc = buf.get(f'{tag}offcols', n_off, np.int64)
        np.compress(o, pos, out=oc)
        oc -= n_sky_S
        ov = buf.get(f'{tag}offvals', n_off, np.float64)
        np.compress(o, d64, out=ov)
        off = (ip, oc, ov)
    if n - n_off:
        no = buf.get(f'{tag}no', n, np.bool_)
        np.logical_not(o, out=no)
        ips = buf.get(f'{tag}ips', nr + 1, np.int64)
        np.subtract(lip, ip, out=ips)                     # the sky entries before each row
        sc = buf.get(f'{tag}skycols', n - n_off, np.int64)
        np.compress(no, pos, out=sc)
        sv = buf.get(f'{tag}skyvals', n - n_off, np.float64)
        np.compress(no, d64, out=sv)
        sky = (ips, sc, sv, np.repeat(np.arange(nr, dtype=np.int64), np.diff(ips)))
    return lip, pos, d64, sky, off


class _Slot:
    """Reusable buffers of one task slot: ``take(name, shape)`` returns a zeroed float64 view of that shape,
    ``get(name, size, dtype)`` an uninitialised 1-D view; each grown when a request exceeds the buffer."""

    def __init__(self):
        self.bufs = {}

    def take(self, name, shape):
        rows, cols = shape
        b = self.bufs.get(name)
        if b is None or b.shape[0] < rows or b.shape[1] != cols:
            b = self.bufs[name] = np.empty((max(rows, 1), cols))
        v = b[:rows]
        v.fill(0.0)
        return v

    def get(self, name, size, dtype):
        dtype = np.dtype(dtype)
        b = self.bufs.get(name)
        if b is None or b.size < size or b.dtype != dtype:
            b = self.bufs[name] = np.empty(max(int(size), 1), dtype=dtype)
        return b[:size]


def _combine_rows_in_place(M, C, cols=1 << 20):
    """``C M`` written over ``M`` (``C`` square, k x k; ``M`` k x n, C order), a column block at a time; returns ``M``."""
    for a in range(0, M.shape[1], cols):
        c = min(M.shape[1], a + cols)
        M[:, a:c] = C @ M[:, a:c]
    return M


def _transform_in_place(M, T, cols=1 << 20):
    """``T^T M`` written over the first ``T.shape[1]`` rows of ``M`` (k x n, C order), a column block at a
    time; returns those rows (contiguous)."""
    kk = T.shape[1]
    for a in range(0, M.shape[1], cols):
        c = min(M.shape[1], a + cols)
        M[:kk, a:c] = T.T @ M[:, a:c]
    return M[:kk]


class Deflation:
    """The deflation of one solve (see the module docstring).

    Built on the column-scaled system the solver runs on: ``storage`` (its matrix: a csr matrix, a
    :class:`~selfcal.core.blockcsr.BlockCSR` or a :class:`~selfcal.core.blockcsr.ColSplitCSR`; read
    only), ``b`` (its right-hand side, read only), ``scale`` (``x * scale`` is physical; None: no
    scaling), ``active_mask`` (the full layout's columns in the
    system: None when every column is), ``ref_shape`` and ``sky_names`` (the sky terms: the first
    ``len(sky_names) * h * w`` columns of the full layout, one grid each), and ``settings`` (with
    ``order``, ``basis``, ``terms`` and ``storage``, a :class:`~selfcal.run.recipe.Deflate`). ``threads``: of
    the projections.

    Then :meth:`start` (the part in the space, from ``b``: the right-hand side and the start the
    solver gets), :meth:`operator` (the solver's operator), :meth:`full` (an iterate as the
    solution), :meth:`resolve` (the part in the space again, against an iterate) and :meth:`summary`
    (what the solve's record keeps). :meth:`bind` gives it the solver's operator and the matrix before
    :meth:`start`. Only the columns with entries take part (all of a compacted system's)."""

    def __init__(self, storage, b, *, scale, active_mask, ref_shape, sky_names, settings, threads=1,
                 projector=None, signatures=None):
        t0 = time.perf_counter()
        self.settings = settings
        self.exclusion_info = None                    # Deflate(exclude=): the degenerate directions excluded (record)
        self.signature_info = None                    # Offsets(exclude=SkySignature): the terms constrained (record)
        if signatures and getattr(settings, 'exclude', None) is not None:
            raise ValueError("Offsets(exclude=SkySignature(...)) together with Deflate(exclude=Degenerate(...)) is not "
                             "supported: the second re-bases the pattern space on sky constraints, the first constrains "
                             "the offsets' response to it")
        m, n = int(storage.shape[0]), int(storage.shape[1])
        self.shape = (m, n)
        h, w = int(ref_shape[0]), int(ref_shape[1])
        npix = h * w
        names = tuple(sky_names) if sky_names else ('sky',)
        terms = tuple(names) if settings.terms is None else tuple(settings.terms)
        unknown = [t for t in terms if t not in names]
        if unknown:
            raise ValueError(f"Deflate(terms={settings.terms!r}): no sky term {unknown} (the model's: {list(names)})")
        nsb = len(names)
        self._pool = ThreadPoolExecutor(max_workers=int(threads)) if threads > 1 else None
        # The projections' blocks (as many as the threads: their partial sums set the arithmetic) are run by a few
        # threads: each block is many small BLAS calls, which OpenBLAS serialises across threads (_BLAS_THREADS).
        self._blas_pool = (ThreadPoolExecutor(max_workers=min(int(threads), _BLAS_THREADS)) if threads > 1
                           else None)
        self._scratch = {}
        # The columns the space lives on: those with entries (all of a compacted system's); the chunks over the
        # threads (marking a column covered commutes).
        covered = np.zeros(n, dtype=bool)
        mark_slots = {}                                           # one per thread, for this sweep

        def mark(desc):
            buf = mark_slots.get(threading.get_ident())
            if buf is None:
                buf = mark_slots[threading.get_ident()] = _Slot()
            for _lip, cols, _data in _piece(storage, desc, buf=buf)[2]:
                covered[cols] = True

        descs0 = _piece_index(storage)
        if self._pool is None:
            for d in descs0:
                mark(d)
        else:
            list(self._pool.map(mark, descs0))
        del descs0, mark_slots
        self.S = None if covered.all() else np.flatnonzero(covered)
        del covered
        nS = n if self.S is None else int(self.S.size)
        self.nS = nS
        # Solver column -> full layout, for the sky columns (the first ones, in grid order per term).
        if active_mask is not None:
            full_sky = np.flatnonzero(np.asarray(active_mask, dtype=bool)[:nsb * npix])
        else:
            full_sky = np.arange(nsb * npix, dtype=np.int64)
        n_sky = int(full_sky.size)
        sky_S = np.arange(n_sky) if self.S is None else self.S[self.S < n_sky]
        self.n_sky_S = n_sky_S = int(sky_S.size)
        n_off = nS - n_sky_S
        fs = full_sky[sky_S]
        term_of = fs // npix
        pix = fs - term_of * npix
        iy, ix = np.divmod(pix, w)
        full_sky_cols = fs if getattr(settings, 'exclude', None) is not None else None
        del fs, pix
        col_norm = (np.ones(nS) if scale is None
                    else 1.0 / np.asarray(scale if self.S is None else np.asarray(scale)[self.S], dtype=np.float64))
        # The patterns: (order + 1)^2 per deflated term, on the term's covered bounding box.
        order, basis = int(settings.order), str(settings.basis)
        # (sel, x function, y function, box corner) per pattern; the values are made row by row into the pattern
        # array below, one pattern's temporaries at a time
        labels, rows_of = [], []
        for name in terms:
            j = names.index(name)
            sel = np.flatnonzero(term_of == j)
            if sel.size == 0:
                continue
            x0, x1, y0, y1 = int(ix[sel].min()), int(ix[sel].max()), int(iy[sel].min()), int(iy[sel].max())
            fx = axis_functions(basis, order, x1 - x0 + 1)
            fy = axis_functions(basis, order, y1 - y0 + 1)
            for a in range(order + 1):
                for c in range(order + 1):
                    labels.append(f"{name}:{a},{c}")
                    rows_of.append((sel, fx[a], fy[c], x0, y0))
            if basis != 'polynomial' or order < POLY_ALWAYS:
                px = axis_functions('polynomial', POLY_ALWAYS, x1 - x0 + 1)
                py = axis_functions('polynomial', POLY_ALWAYS, y1 - y0 + 1)
                for a in range(POLY_ALWAYS + 1):
                    for c in range(POLY_ALWAYS + 1 - a):
                        if a + c == 0 or (basis == 'polynomial' and a <= order and c <= order):
                            continue
                        labels.append(f"{name}:P{a},{c}")
                        rows_of.append((sel, px[a], py[c], x0, y0))
        self._term_slices = []
        for j, name in enumerate(names):
            a_, b_ = int(np.searchsorted(term_of, j, 'left')), int(np.searchsorted(term_of, j, 'right'))
            if b_ > a_:
                self._term_slices.append((name, a_, b_))
        k = len(labels)
        if k == 0:
            raise ValueError(f"Deflate: the sky terms {list(terms)} have no covered pixel")
        self.labels = labels
        Z = np.zeros((k, nS))                     # one pattern per row (contiguous)
        for i, (sel, gx, gy, x0, y0) in enumerate(rows_of):
            Z[i, sel] = (gx[ix[sel] - x0] * gy[iy[sel] - y0]) * col_norm[sel]
        del rows_of, term_of, iy, ix
        self._colmap = None
        if self.S is not None:
            cm = np.full(n, -1, dtype=np.int64)
            cm[self.S] = np.arange(nS)
            self._colmap = cm
        # The column constraints (solver coordinates) on the offset columns of this space: the offsets'
        # normal matrix is projected, G -> P G P, so the response stays in the allowed subspace.
        self._projector = projector                  # the column constraints in solver coordinates (None: none)
        self._proj_off = None
        if projector is not None:
            cm = np.arange(n, dtype=np.int64) if self._colmap is None else self._colmap
            # The patterns live in the allowed subspace: a sky shape the constraints exclude (a bowl, say) leaves
            # the space (its projected pattern is ~0 and drops as dependent), the others lose their part along it.
            full = np.zeros(n)
            for i in range(k):
                full[...] = 0.0
                if self.S is None:
                    full[:] = Z[i]
                else:
                    full[self.S] = Z[i]
                projector.apply(full)
                Z[i] = full if self.S is None else full[self.S]
            del full
            # The offsets' part of the constraints, for the block solve (the sky's groups are not its business).
            off = np.where(cm >= n_sky_S, cm - n_sky_S, -1)
            og = [(c, Q) for c, Q in projector.groups if (cm[c[c >= 0]] >= n_sky_S).all()]
            self._proj_off = ColumnProjector(og).mapped(off) if og else None
        logger.info(f"Deflation: {k} patterns ({basis}, order {order}) of {len(terms)} sky term(s) on {nS} "
                    f"columns ({n_sky_S} sky, {n_off} offset and other)")
        try:
            # Pass 1 (float64, one sweep over the matrix for all the patterns at once, scipy's multi-vector kernels:
            # the matrix is read once, not once per pattern; the row chunks over the threads, each thread's partial
            # sums combined in a fixed order: deterministic for a thread count): each pattern's sky image U = B F,
            # the offsets' response R = B_off^T U and the offsets' normal matrix G = B_off^T B_off (a sparse product
            # per row chunk).
            t = time.perf_counter()
            Zt = np.ascontiguousarray(Z.T)                        # (nS, k): the kernels' vector layout
            del Z                                                 # the one layout until pass 2 has read it
            descs = _piece_index(storage)

            def pass1(my):
                sky_norm2 = np.zeros(k)
                Rt = np.zeros((n_off, k))
                Gt = None
                t_g = 0.0
                slot = _Slot()                                    # this share's chunk buffers, reused chunk to chunk
                for desc in my:
                    r0, r1, pieces = _piece(storage, desc, buf=slot)
                    nr = r1 - r0
                    U = slot.take('U', (nr, k))
                    offs = []
                    for ti, (lip, cols, data) in enumerate(pieces):
                        _lip, _pos, _d64, sky, oc = _prepare_piece(nr, lip, cols, data, self._colmap, n_sky_S,
                                                                   buf=slot, tag=ti)
                        if sky is not None:
                            # the sky entries alone: the table's offset rows are zero here, and a row's sky entries
                            # keep their order, so the sums are those of the whole piece
                            _sparsetools.csr_matvecs(nr, nS, k, sky[0], sky[1], sky[2], Zt, U)
                        if oc is not None:
                            offs.append(oc)
                    sky_norm2 += np.einsum('ij,ij->j', U, U)
                    for ip, ci, dd in offs:
                        _sparsetools.csc_matvecs(n_off, nr, k, ip, ci, dd, U, Rt)
                        sub = csr_matrix((dd, ci, ip), shape=(nr, n_off))
                        tg = time.perf_counter()
                        g = (sub.T @ sub).tocsr()                # the chunk's share of the offsets' normal matrix
                        g.sum_duplicates()
                        Gt = g if Gt is None else Gt + g         # the thread's running sum, in chunk order
                        t_g += time.perf_counter() - tg
                        del g, sub
                    del U, offs
                return sky_norm2, Rt, Gt, t_g

            parts = self._over_chunks(descs, pass1)
            sky_norm2 = parts[0][0].copy()
            Rt = parts[0][1]
            gthreads = [p[2] for p in parts if p[2] is not None]
            t_g = parts[0][3]
            for p in parts[1:]:
                sky_norm2 += p[0]
                Rt += p[1]
                t_g = max(t_g, p[3])
            del parts
            R = np.ascontiguousarray(Rt.T)
            del Rt
            # The threads' sums added pairwise (a fixed tree, the pairs in the pool): the result depends on the thread
            # count through the association of the floating-point sums only.
            tg = time.perf_counter()
            while len(gthreads) > 1:
                pairs = [(gthreads[i], gthreads[i + 1]) for i in range(0, len(gthreads) - 1, 2)]
                summed = [a + b_ for a, b_ in pairs] if self._pool is None else list(self._pool.map(lambda ab: ab[0] + ab[1], pairs))
                if len(gthreads) % 2:
                    summed.append(gthreads[-1])
                gthreads = summed
            G = gthreads[0].tocsr() if gthreads else coo_matrix((n_off, n_off)).tocsr()
            G.sum_duplicates()
            del gthreads
            t_g += time.perf_counter() - tg
            t_pass1 = time.perf_counter() - t
            t = time.perf_counter()
            ex = getattr(self.settings, 'exclude', None)
            X, info = _block_pinv_solve(G, R.T, project=self._proj_off, pool=self._pool)
            res = G @ (-X) + R.T
            if self._proj_off is not None:
                self._proj_off.apply(res)                 # the projected normal equations' residual
            self.pairing_residual = float(np.max(np.linalg.norm(res, axis=0) /
                                                 np.maximum(np.linalg.norm(R, axis=1), 1e-300)))
            del res
            if signatures:
                X = self._exclude_signatures(signatures, X, G, info, scale, n, n_sky_S, n_off)
            Zt[n_sky_S:, :] = -X                                  # the paired patterns (pass 2 reads them)
            del G, R, X
            t_pair = time.perf_counter() - t

            # Pass 2 (float64, one sweep): N Z = B^T (B Z) for all the patterns, and B^T b alongside (the
            # right-hand side rides along as one more column of U). Threaded in two stages per batch of row
            # chunks: the images U = B Z chunk by chunk, then the transposed products over the columns, each
            # thread owning a range of them (:meth:`_transposed_products`; the sums per column in row order, as a
            # single sweep makes them).
            t = time.perf_counter()
            pair_norm2 = np.zeros(k)
            bnorm2 = 0.0
            acc2 = []

            def images(r0, r1, prepared, buf):
                nr = r1 - r0
                U = buf.take('U', (nr, k + 1))
                for lip, pos, d64, _sky, _off in prepared:
                    _sparsetools.csr_matvecs(nr, nS, k, lip, pos, d64, Zt, U[:, :k])
                pn = np.einsum('ij,ij->j', U[:, :k], U[:, :k])
                bc = np.asarray(b[r0:r1], dtype=np.float64)
                U[:, k] = bc
                return U, (pn, float(bc @ bc))

            NZb = self._transposed_products(storage, images, k + 1, n_sky_S, acc2)
            for pn, bn in acc2:
                pair_norm2 += pn
                bnorm2 += bn
            del acc2
            ATb = np.ascontiguousarray(NZb[:, k])
            NZ = np.ascontiguousarray(NZb[:, :k].T)               # one copy, into the layout the grams read
            del NZb
            Z = np.ascontiguousarray(Zt.T)                        # the patterns, one per row again
            del Zt
            t_pass2 = time.perf_counter() - t
            self.nullness = np.sqrt(pair_norm2 / np.maximum(sky_norm2, 1e-300))

            # An orthonormal basis of the space diagonalising E = V^T N V; the directions kept: those the
            # data see (E's eigenvalue above DEPENDENT of the largest: exact gauges have ~0); the rest
            # stays orthogonal to the dropped. The obliqueness |N v| / v^T N v of each (about 1 for an
            # eigenvector of N, large for a slow direction mixed with fast ones) is kept as information.
            E = self._gram(Z, NZ)
            GZ = self._gram(Z, Z)
            NN = self._gram(NZ, NZ)
        except BaseException:
            self.close()
            raise
        w1, U1 = np.linalg.eigh(GZ)
        ok1 = w1 > DEPENDENT * w1.max()
        T = U1[:, ok1] / np.sqrt(w1[ok1])
        E2 = T.T @ E @ T
        lam, U2 = np.linalg.eigh(0.5 * (E2 + E2.T))
        T = T @ U2
        # Everything the decomposition found is kept for the record: the directions in pattern coordinates
        # (column j of `directions` = the weights of the patterns in direction j), their eigenvalues, what the
        # data see of each, and which are exact gauges (summary()).
        self.directions = np.array(T, copy=True)
        self.eigenvalues = np.array(lam, copy=True)
        wn = np.sqrt(np.maximum(np.einsum('ij,ij->j', T, NN @ T), 0.0))
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = np.where(lam > 0, wn / np.where(lam > 0, lam, 1.0), np.inf)
        lam_max = max(float(lam.max()), 1e-300)
        self.seen = np.sqrt(np.maximum(lam, 0.0) / lam_max)        # per direction: what the data see of it, relative
        # An exact gauge (eigenvalue at the rounding level) is left out of the space: the data never push along it,
        # and its value stays the start's, as a plain solve leaves it (the zero point of the sky against the frame
        # scalars, for one).
        exact = lam > DEPENDENT * lam_max                          # False: an exact gauge
        self.kept_mask, self.exact_mask = exact.copy(), ~exact
        self.gauges = int((~exact).sum())
        self.rho_max = float(rho[exact].max()) if exact.any() else math.nan
        if ex is not None:
            # Deflate(exclude=Degenerate(below)): the directions seen below the level are degenerate. Their sky
            # shapes (physical units, over the covered sky columns) join the column constraints as a sky group, and
            # the pattern space is re-based on the constrained patterns from this same setup: the projected
            # shapes are combinations of the original ones (the projection is in map space, where the shapes
            # live), so the pairings, E and the Gram matrices follow by a k x k combination; the images of the
            # re-based patterns under the constrained operator are the projected images.
            sel = self.seen < float(ex.below)
            k_sel = int(sel.sum())
            if k_sel == 0:
                raise ValueError(f"Deflate(exclude=Degenerate(below={ex.below:g})): no direction of the pattern space is seen "
                                 f"below the level (the least-seen: {float(self.seen.min()):.3g})")
            if k_sel == k:
                raise ValueError(f"Deflate(exclude=Degenerate(below={ex.below:g})): every direction of the pattern space is seen "
                                 f"below the level")
            Ts = np.ascontiguousarray(T[:, sel])                                           # (k, k_sel)
            M_sky = 1.0 / col_norm[:n_sky_S]
            P = np.empty((n_sky_S, k_sel))
            for a_ in range(0, n_sky_S, 1 << 20):
                c_ = min(n_sky_S, a_ + (1 << 20))
                P[a_:c_] = (Z[:, a_:c_].T @ Ts) * M_sky[a_:c_, None]                          # physical: x = y * M
            # the physical Gram of the patterns (S = F F^T) and the combination C: F' = C F has P^T F' = 0
            Sg = np.zeros((k, k))
            for a_ in range(0, n_sky_S, 1 << 20):                                              # S = F F^T, by column blocks
                c_ = min(n_sky_S, a_ + (1 << 20))
                Fb = Z[:, a_:c_] * M_sky[None, a_:c_]
                Sg += Fb @ Fb.T
            Sg = 0.5 * (Sg + Sg.T)
            Gs = Ts.T @ Sg @ Ts
            C = np.eye(k) - Sg @ Ts @ np.linalg.solve(Gs, Ts.T)
            comp = []
            for j in np.flatnonzero(sel):
                wv = self.directions[:, j]; wv = wv / max(float(np.abs(wv).max()), 1e-300); top = np.argsort(-np.abs(wv))[:4]
                comp.append(' '.join(f'{wv[i]:+.2f}*{self.labels[i]}' for i in top))
            self.exclusion_info = {
                'below': float(ex.below), 'directions': [int(j) for j in np.flatnonzero(sel)], 'count': k_sel,
                'seen': [float(self.seen[j]) for j in np.flatnonzero(sel)],
                'composition': comp, 'pattern_labels': list(self.labels),
                'seen_by_direction': [float(v) for v in self.seen],
                'eigenvectors': [[float(f'{v:.7g}') for v in col] for col in self.directions.T],
                'pairing_residual': float(self.pairing_residual)}
            logger.info(f"Deflation: {k_sel} degenerate direction(s) (seen below {ex.below:g}) excluded from the sky: "
                        + ' | '.join(f"seen {sv:.2g}: {cc}" for sv, cc in zip(self.exclusion_info['seen'], comp)))
            # the sky constraint joins the column constraints; the solver's operator and the solution take it
            groups = list(projector.groups) if projector is not None else []
            new = ColumnProjector.build([(full_sky_cols[None, :], P)], active_mask, scale, n)
            self._projector = projector = ColumnProjector(groups + list(new.groups))
            # The slow modes of the constrained system: the excluded shapes' solver-space images (the shapes weighted
            # by the column scale, their excluded part taken out) are nearly invisible too, and the solve would spend
            # its iterations on them. They join the space as the degenerate directions projected in solver space:
            # sky part Z_sel - Qh (Qh^T Z_sel) with Qh the group's orthonormal basis, offsets' part as paired; their
            # images need N Qh = (Qh, B_off^T B_sky Qh) (unit sky column norms), one small sweep for the offsets' part.
            Qh = new.groups[0][1][0]                                                       # (n_sky_S, k_sel), orthonormal
            del P, new
            # (One sky term with unit column norms and no compaction map for now: then N Qh's sky part is Qh itself and
            # one sweep for the offsets' part suffices; otherwise the space does without the leftovers, which costs
            # iterations, not correctness.)
            extra = (nsb == 1 and scale is not None and self.S is None)
            if extra:
                Zsel = np.ascontiguousarray(T[:, sel].T @ Z)                                   # (k_sel, nS): the degenerate directions
                NZsel = T[:, sel].T @ NZ
                coef = Zsel[:, :n_sky_S] @ Qh                                             # (k_sel, k_sel): Qh^T Z_sel
                Rq = self._offset_images(storage, Qh, n_sky_S, n_off)                      # B_off^T B_sky Qh, (n_off, k_sel)
                NQ = np.zeros((k_sel, nS))
                NQ[:, :n_sky_S] = Qh.T
                NQ[:, n_sky_S:] = Rq.T
                Zx = Zsel.copy()
                Zx[:, :n_sky_S] -= coef @ Qh.T
                NZx = NZsel - coef @ NQ
                del Zsel, NZsel, NQ, Rq
            # the re-basing: Z' = C Z (the offsets' parts follow), N' Z' = Pc (C N Z), E' = C E C^T, GZ' = C GZ C^T
            Z = _combine_rows_in_place(Z, C)
            NZ = _combine_rows_in_place(NZ, C)
            if extra:
                Z = np.concatenate([Z, Zx], axis=0)
                NZ = np.concatenate([NZ, NZx], axis=0)
                del Zx, NZx
            full = np.zeros(n)
            for i in range(Z.shape[0]):
                full[...] = 0.0
                if self.S is None:
                    full[:] = NZ[i]
                else:
                    full[self.S] = NZ[i]
                projector.apply(full)
                NZ[i] = full if self.S is None else full[self.S]
            del full
            if extra:
                E = self._gram(Z, NZ)
                GZ = self._gram(Z, Z)
            else:
                E = C @ E @ C.T
                GZ = C @ GZ @ C.T
            NN = self._gram(NZ, NZ)
            w1, U1 = np.linalg.eigh(GZ)
            ok1 = w1 > DEPENDENT * w1.max()
            T = U1[:, ok1] / np.sqrt(w1[ok1])
            E2 = T.T @ E @ T
            lam, U2 = np.linalg.eigh(0.5 * (E2 + E2.T))
            T = T @ U2
            self.directions_constrained = np.array(T, copy=True)
            self.eigenvalues_constrained = np.array(lam, copy=True)
            wn = np.sqrt(np.maximum(np.einsum('ij,ij->j', T, NN @ T), 0.0))
            with np.errstate(divide='ignore', invalid='ignore'):
                rho = np.where(lam > 0, wn / np.where(lam > 0, lam, 1.0), np.inf)
            lam_max = max(float(lam.max()), 1e-300)
            seen_c = np.sqrt(np.maximum(lam, 0.0) / lam_max)
            exact = lam > DEPENDENT * lam_max
            self.kept_mask, self.exact_mask = exact.copy(), ~exact
            self.gauges = int((~exact).sum())
            self.rho_max = float(rho[exact].max()) if exact.any() else math.nan
            self.seen_constrained = seen_c
            logger.info(f"Deflation: the constrained space's {int(exact.sum())} directions, seen fractions (the 8 smallest): "
                        + ', '.join(f'{v:.2g}' for v in np.sort(seen_c)[:8]))
        T = T[:, exact]
        self.lam = lam[exact].copy()
        self.kept = int(exact.sum())
        spectrum = ', '.join(f'{v:.2g}' for v in np.sort(self.seen)[:12])
        # the composition of the least-seen directions (below 0.1 of the best-seen): their three largest pattern weights,
        # relative to the direction's largest (the pattern coordinates are not normalised)
        comp = []
        for j in np.argsort(self.seen):
            if self.seen[j] >= 0.1 or len(comp) >= 8:
                break
            w = self.directions[:, j]; w = w / max(float(np.abs(w).max()), 1e-300); top = np.argsort(-np.abs(w))[:3]
            comp.append(f"{self.seen[j]:.2g} = " + ' '.join(f'{w[i]:+.2f}*{self.labels[i]}' for i in top))
        logger.info('Deflation: the least-seen directions, by their largest pattern weights: ' + ' | '.join(comp))
        logger.info(f"Deflation: the directions' seen fractions (sqrt(lambda / lambda_max); the {min(12, self.seen.size)} "
                    f"smallest of {self.seen.size}): {spectrum}")
        if self.kept == 0:
            self.close()
            raise ValueError("Deflate: no pattern direction is left (every one is a gauge of the system)")
        # V = T^T Z and W = T^T N Z (one direction per row), in place: the setup's peak stays at the two arrays.
        self.V = _transform_in_place(Z, T)
        self.W = _transform_in_place(NZ, T)
        del Z, NZ
        # Kept in the products' precision when that is float32 (half the traffic of every projection; the
        # arithmetic stays float64 and the rounding is that of the products: a converged float32 solve stays as
        # put as with float64 storage, measured), in float64 otherwise (float32 storage then lets a converged
        # solve drift).
        pdt = np.dtype(storage.dtype)
        want = getattr(settings, 'storage', None)
        if want == 'float32' and pdt != np.float32:
            raise ValueError("Deflate(storage='float32') needs float32 products (Fit(float32=True)): with float64 "
                             "products the stored vectors would be coarser than the products and a converged solve "
                             "would drift")
        self.store = np.dtype(np.float64) if want == 'float64' else (pdt if pdt == np.float32 else np.dtype(np.float64))
        if self.store == np.float32:
            self.V, self.W = self.V.astype(np.float32), self.W.astype(np.float32)
        # The space's least-squares amplitudes.
        self.c_star = (self.V @ ATb) / self.lam
        del ATb
        self._bnorm2 = bnorm2
        nb = 1 if self._pool is None else max(1, min(int(threads), nS // _MIN_BLOCK))
        cuts = np.linspace(0, nS, nb + 1, dtype=np.int64)
        self._blocks = [(int(a), int(c)) for a, c in zip(cuts[:-1], cuts[1:])]
        self.x_c = np.zeros(nS)
        self._sub(self.x_c, self.V, -self.c_star)
        self.setup_s = time.perf_counter() - t0
        self.resolve_change = math.nan
        self.op = None
        st = self.sweep_timing
        logger.info(
            f"Deflation: offset response solved on {info['blocks']} blocks (largest {info['largest']}, "
            f"{info['zero_eigenvalues']} zero eigenvalues"
            + (f"; {info['hubs']} columns shared by many frames eliminated by a Schur complement" if info.get('hubs') else '')
            + f"), normal-equation residual {self.pairing_residual:.1e}; "
            f"|B z| / |B F| from {self.nullness.min():.2e} to {self.nullness.max():.2e}; {self.kept} of {k} "
            f"directions kept ({self.gauges} gauge or dependent left out); passes {t_pass1:.1f} s (the offsets' normal "
            f"matrix {t_g:.1f} s of a thread's, with the reduction) + {t_pass2:.1f} s (chunks {st['work_thread_s']:.0f} thread-s over {st['work_wall_s']:.0f} s, "
            f"adds {st['add_thread_s']:.0f} thread-s over {st['add_wall_s']:.0f} s, {st['threads']} threads), blocks {t_pair:.1f} s; setup {self.setup_s:.1f} s")

    def _exclude_signatures(self, signatures, X, G, info, scale, n, n_sky_S, n_off):
        """``Offsets(exclude=SkySignature(k))``: each named term may not carry its response to the sky polynomials of
        degree 1 to k. ``X`` (``(n_off, patterns)``, solver coordinates) is the offsets' least-squares response to
        every pattern; the polynomial patterns' responses restricted to a term are the term's signature of those
        fields. The signature first loses its detector-fixed part (``chunk``: the columns of one chunk value, one per
        group, hold the configurations the same in every group; their projection is removed per chunk), so a
        pattern fixed on the detector stays free; the term's coefficients are then made orthogonal to what is left in
        the data's metric (each coefficient weighted by its information, the diagonal of ``G``): a
        :class:`~selfcal.core.projector.ColumnProjector` group joins the column constraints (the solver's operator, its
        start and the solution apply it). Every pattern's response is then solved again under the constraint, by its
        Lagrange form (the constraint spans every frame, so it is not one more block of the block solve):
        ``X_c = X - Y (C^T Y)^+ C^T X`` with ``Y = G^+ C`` (the same block solve, or its factorisation when the shared
        offsets made one). Returns ``X_c``; the record keeps what was constrained (``signature_exclusions``)."""
        from .projector import _DROP
        cm = np.arange(n, dtype=np.int64) if self._colmap is None else self._colmap
        basis = str(self.settings.basis)

        def poly_degree(label):
            """The total degree of a polynomial pattern ('<term>:Pa,c', or 'a,c' with the polynomial basis); None for a
            cosine pattern."""
            key = label.split(':', 1)[1]
            if key.startswith('P'):
                key = key[1:]
            elif basis != 'polynomial':
                return None
            a, c = key.split(',')
            return int(a) + int(c)

        degrees = [poly_degree(lbl) for lbl in self.labels]
        gdiag = np.asarray(G.diagonal(), dtype=np.float64)
        groups, cs, recs = [], [], []
        for cols, degree, name, chunk in signatures:
            cols = np.asarray(cols, dtype=np.int64)
            sp = cm[cols]
            ok = sp >= n_sky_S                                    # the term's unknowns in this space (with data)
            cols, off, chunk = cols[ok], sp[ok] - n_sky_S, np.asarray(chunk, dtype=np.int64)[ok]
            fields = [i for i, d in enumerate(degrees) if d is not None and 1 <= d <= int(degree)]
            if off.size == 0 or not fields:
                raise ValueError(f"Offsets({name!r}, exclude=SkySignature({degree})): "
                                 + ("the term has no unknown with data" if off.size == 0 else
                                    "the deflation has no polynomial pattern of degree 1 to that degree"))
            # Solver coordinates y, physical x = m y. The data's metric on the term's coefficients is the diagonal of
            # the offsets' normal matrix g (1 for unit-norm columns): <u, v> = sum g u v in solver coordinates. A
            # detector-fixed configuration (x the same in every group, one value per chunk) is f = 1 / m on the
            # chunk's columns; each signature S loses its projection on those (per chunk, in the data's metric), so
            # they stay free, and the constraint on y is (g S')^T y = 0.
            m = np.ones(off.size) if scale is None else np.asarray(scale, dtype=np.float64)[cols]
            g, f = gdiag[off], 1.0 / m
            S = np.array(X[off][:, fields], dtype=np.float64)
            norm0 = np.sqrt(np.einsum('i,ij,ij->j', g, S, S))
            _, ch = np.unique(chunk, return_inverse=True)
            den = np.bincount(ch, weights=g * f * f)
            for j in range(S.shape[1]):
                num = np.bincount(ch, weights=g * f * S[:, j], minlength=den.size)
                S[:, j] -= f * np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)[ch]
            A = g[:, None] * S
            U, sv, _ = np.linalg.svd(A, full_matrices=False)
            if float(sv.max()) <= 0.0:
                raise ValueError(f"Offsets({name!r}, exclude=SkySignature({degree})): the term's signature is "
                                 "detector-fixed throughout (one group, or every frame alike): nothing to constrain")
            keep = sv > _DROP * max(float(sv.max()), 1e-300)
            U = np.ascontiguousarray(U[:, keep])
            groups.append((cols[None, :], U[None, :, :]))
            c = np.zeros((n_off, U.shape[1]))
            c[off] = U
            cs.append(c)
            kept = np.sqrt(np.einsum('i,ij,ij->j', g, S, S))
            recs.append({'term': str(name), 'degree': int(degree), 'fields': [self.labels[i] for i in fields],
                         'columns': int(off.size), 'directions': int(U.shape[1]),
                         'singular_values': [float(v) for v in sv], 'metric': 'data (the diagonal of the normal matrix)',
                         'detector_fixed_free': True,
                         'share_constrained': [float(a / b) if b > 0 else 0.0 for a, b in zip(kept, norm0)]})
        C = np.concatenate(cs, axis=1)
        Cp = C if self._proj_off is None else self._proj_off.apply(np.array(C, copy=True))
        fac = info.get('factor')
        Y = fac.solve(Cp, pool=self._pool)[0] if fac is not None else \
            _block_pinv_solve(G, C, project=self._proj_off, pool=self._pool)[0]
        lam = np.linalg.lstsq(C.T @ Y, C.T @ X, rcond=None)[0]
        Xc = X - Y @ lam
        held = float(np.abs(C.T @ Xc).max() / max(float(np.abs(Xc).max()), 1e-300))
        groups0 = list(self._projector.groups) if self._projector is not None else []
        self._projector = ColumnProjector(groups0 + groups)
        for r in recs:
            r['constraint_held'] = held
        self.signature_info = recs
        logger.info('Deflation: sky signatures excluded from ' + '; '.join(
            f"{r['term']} ({r['columns']} unknowns, degree <= {r['degree']}: {r['directions']} directions; the share of each "
            f"field's signature that is not detector-fixed: " + ', '.join(f'{v:.2f}' for v in r['share_constrained']) + ')'
            for r in recs)
            + f"; the patterns' responses solved under the constraint (|C^T X| / |X| = {held:.1e})")
        return Xc

    def _offset_images(self, storage, Qh, n_sky_S, n_off):
        """``B_off^T B_sky Q`` for sky vectors ``Q`` ``(n_sky_S, q)`` (a single sky term, every row with one sky entry):
        one sweep over the matrix, the offsets' part of ``N Q`` (its sky part is ``Q`` itself for unit sky column
        norms). The chunks over the threads, the partial sums added in thread order."""
        q = Qh.shape[1]
        nS = self.nS
        Qt = np.zeros((nS, q))
        Qt[:n_sky_S] = Qh
        descs = _piece_index(storage)

        def sweep(my):
            Rt = np.zeros((n_off, q))
            slot = _Slot()
            for desc in my:
                r0, r1, pieces = _piece(storage, desc, buf=slot)
                nr = r1 - r0
                U = slot.take('U', (nr, q))
                offs = []
                for ti, (lip, cols, data) in enumerate(pieces):
                    _lip, _pos, _d64, sky, oc = _prepare_piece(nr, lip, cols, data, self._colmap, n_sky_S,
                                                               buf=slot, tag=ti)
                    if sky is not None:
                        _sparsetools.csr_matvecs(nr, nS, q, sky[0], sky[1], sky[2], Qt, U)
                    if oc is not None:
                        offs.append(oc)
                for ip, ci, dd in offs:
                    _sparsetools.csc_matvecs(n_off, nr, q, ip, ci, dd, U, Rt)
            return Rt

        parts = self._over_chunks(descs, sweep)
        R = parts[0]
        for p in parts[1:]:
            R += p
        return R

    @property
    def projector(self):
        """The column constraints the solve runs with: the model's, and the degenerate directions' sky shapes when
        ``Deflate(exclude=)`` added them (the solver's operator and the solution must apply this one)."""
        return self._projector

    # ---------------------------------------------------------------- the sweeps over the matrix (float64, threaded)
    def _over_chunks(self, descs, fn):
        """``fn(my)`` for each thread's share ``my`` of the row-chunk descriptors (round robin, so the share depends
        on the thread count only); returns the threads' results in thread order (one result without a pool)."""
        if self._pool is None:
            return [fn(descs)]
        nt = max(1, min(self._pool._max_workers, len(descs)))
        shares = [descs[j::nt] for j in range(nt)]
        return list(self._pool.map(fn, shares))

    def _transposed_products(self, storage, images, m, n_sky_S, extra=None):
        """``B^T U`` summed over the row chunks, ``(nS, m)`` float64, where ``images(r0, r1, pieces)`` gives each
        chunk's ``U`` (``(rows, m)``) and an extra value collected in ``extra`` (in row order). Per batch of chunks
        the images are made in parallel, and each chunk's sky entries (``pos < n_sky_S``) are sorted by column (a
        stable sort); threads that own disjoint column ranges then sum each chunk's products per column (in row
        order, from zero) and add the sums into the output; the offset columns' (few) go through the transposed
        kernel into a private buffer per chunk, added in chunk order. Every column's total is the chunk partials
        added in chunk order, so the result does not depend on the thread count."""
        out = np.zeros((self.nS, m))
        descs = _piece_index(storage)
        nt = 1 if self._pool is None else max(1, self._pool._max_workers)
        n_off = self.nS - n_sky_S

        def partials(mp, buf):
            """Per chunk: the sky entries grouped by column ``(uniq, ip, rows, values)`` (the columns ascending, each
            column's entries ``ip[i]:ip[i + 1]`` in row order; None: none), the chunk's ``U``, and the offset columns'
            ``(columns, sums)``, the chunk's own offset columns only (None: none)."""
            (r0, r1, prepared), (U, x) = mp
            nr = r1 - r0
            sky = []
            off = None
            ocs = [oc for _lip, _pos, _d64, _sk, oc in prepared if oc is not None]
            if ocs:
                # The chunk's offset columns numbered compactly, their sums in a buffer of that size: each the same
                # sum, in the same order; a column the chunk does not touch gets no addition (its zero changed nothing).
                mark = buf.bufs.get('offmark')
                if mark is None or mark.size != n_off:
                    mark = buf.bufs['offmark'] = np.zeros(n_off, dtype=bool)
                for _ip, ci, _dd in ocs:
                    mark[ci] = True
                touched = np.flatnonzero(mark)
                mark[touched] = False
                cmap = buf.get('offmap', n_off, np.int64)
                cmap[touched] = np.arange(touched.size, dtype=np.int64)
                sums = buf.take('offsums', (touched.size, m))
                for ti, (ip, ci, dd) in enumerate(ocs):
                    local = buf.get(f'{ti}offlocal', ci.size, np.int64)
                    np.take(cmap, ci, out=local)
                    _sparsetools.csc_matvecs(touched.size, nr, m, ip, local, dd, U, sums)
                off = (touched, sums)
            for _lip, _pos, _d64, sk, oc in prepared:
                if sk is not None:
                    sky.append((sk[1], sk[3], sk[2]))
            grouped = None
            if sky:
                ps = np.concatenate([q[0] for q in sky]) if len(sky) > 1 else sky[0][0]
                rs = np.concatenate([q[1] for q in sky]) if len(sky) > 1 else sky[0][1]
                ds = np.concatenate([q[2] for q in sky]) if len(sky) > 1 else sky[0][2]
                order = np.argsort(ps, kind='stable')
                ne = int(ps.size)
                ps = np.take(ps, order, out=buf.get('sortcols', ne, np.int64))
                rs = np.take(rs, order, out=buf.get('sortrows', ne, np.int64))
                ds = np.take(ds, order, out=buf.get('sortvals', ne, np.float64))
                del order
                starts = np.flatnonzero(np.diff(ps, prepend=-1))
                uniq = np.take(ps, starts, out=buf.get('uniq', starts.size, np.int64))
                ip = buf.get('sortip', starts.size + 1, np.int64)
                ip[:-1] = starts
                ip[-1] = ne
                grouped = (uniq, ip, rs, ds)
            return grouped, U, off, x

        # One buffer slot per task of a batch (the chunk's entries and U: a gigabyte per chunk at full scale), kept
        # for the whole pass: fresh allocations of that size page-fault on every chunk, which the kernel serialises
        # across threads. One more per column range, for its column sums (_ADD_COLUMNS columns at a time).
        slots = [_Slot() for _ in range(nt)]
        add_slots = [_Slot() for _ in range(nt)]
        t_work = [0.0] * nt
        t_add = [0.0] * nt
        t_wall = [0.0, 0.0]

        def work(i_desc):
            i, desc = i_desc
            t0 = time.perf_counter()
            buf = slots[i]
            r0, r1, pieces = _piece(storage, desc, buf=buf)
            prepared = [_prepare_piece(r1 - r0, lip, cols, data, self._colmap, n_sky_S, buf=buf, tag=ti)
                        for ti, (lip, cols, data) in enumerate(pieces)]
            del pieces
            out_ = partials(((r0, r1, prepared), images(r0, r1, prepared, buf)), buf)
            t_work[i] += time.perf_counter() - t0
            return out_

        for a in range(0, len(descs), nt):
            batch = list(enumerate(descs[a:a + nt]))
            t0 = time.perf_counter()
            parts = [work(d) for d in batch] if self._pool is None else list(self._pool.map(work, batch))
            t_wall[0] += time.perf_counter() - t0
            if extra is not None:
                extra.extend(pt[3] for pt in parts)
            for _g, _U, off, _x in parts:
                if off is not None:
                    touched, sums = off
                    out[n_sky_S + touched] += sums
            # The column ranges the threads own, cut where the batch's columns are (a frame covers a compact
            # region of the sky, so equal ranges would leave most threads idle): the quantiles of a sample.
            samp = np.concatenate([g[0][::997] for g, _U, _o, _x in parts if g is not None]
                                  or [np.zeros(0, dtype=np.int64)])
            samp.sort()
            if samp.size >= nt:
                cuts = np.concatenate(([0], samp[(np.arange(1, nt) * samp.size) // nt], [n_sky_S]))
                cuts = np.maximum.accumulate(cuts)
            else:
                cuts = np.linspace(0, n_sky_S, nt + 1, dtype=np.int64)
            # Each chunk's column sums, made by the thread owning the columns: the products of the column's entries
            # summed in row order from zero (a sparse product with one row per column of the chunk: the C kernel,
            # which runs without the GIL), then added at their own columns by the transposed kernel with one entry
            # per summed column (``out[c] += 1.0 * part``), so the work is the chunk's columns', not the range's.
            n_max = min(max((g[0].size for g, _U, _o, _x in parts if g is not None), default=0), _ADD_COLUMNS)
            steps = np.arange(n_max + 1, dtype=np.int64)
            units = np.ones(n_max)

            def add(j):
                lo, hi = int(cuts[j]), int(cuts[j + 1])
                if hi <= lo:
                    return
                t0 = time.perf_counter()
                view = out[lo:hi]
                pb = add_slots[j]
                for g, U, _off, _x in parts:
                    if g is None:
                        continue
                    uniq, ip, rs, ds = g
                    i0, i1 = (int(v) for v in np.searchsorted(uniq, (lo, hi)))
                    for a0 in range(i0, i1, _ADD_COLUMNS):
                        a1 = min(i1, a0 + _ADD_COLUMNS)
                        e0, e1 = int(ip[a0]), int(ip[a1])
                        lip = np.subtract(ip[a0:a1 + 1], e0, out=pb.get('ip', a1 - a0 + 1, np.int64))
                        part = pb.take('part', (a1 - a0, m))
                        _sparsetools.csr_matvecs(a1 - a0, U.shape[0], m, lip, rs[e0:e1], ds[e0:e1], U, part)
                        _sparsetools.csc_matvecs(hi - lo, a1 - a0, m, steps[:a1 - a0 + 1], uniq[a0:a1] - lo,
                                                 units[:a1 - a0], part, view)
                t_add[j] += time.perf_counter() - t0

            t0 = time.perf_counter()
            self._wait(self._submit(add, nt))
            t_wall[1] += time.perf_counter() - t0
            del parts
        self.sweep_timing = {'work_thread_s': float(sum(t_work)), 'work_wall_s': t_wall[0],
                             'add_thread_s': float(sum(t_add)), 'add_wall_s': t_wall[1], 'threads': nt}
        return out

    def _forward(self, storage, v, fn):
        """``fn(r0, r1, B[r0:r1] v)`` for every row chunk (``v``: (nS,) float64; the products in parallel over the
        chunks, each exactly as a single sweep computes it)."""
        descs = _piece_index(storage)
        slots = {}                                                # one per thread, for this sweep

        def chunk(desc):
            buf = slots.get(threading.get_ident())
            if buf is None:
                buf = slots[threading.get_ident()] = _Slot()
            r0, r1, pieces = _piece(storage, desc, buf=buf)
            nr = r1 - r0
            acc = buf.get('acc', nr, np.float64)
            acc.fill(0.0)
            for ti, (lip, cols, data) in enumerate(pieces):
                pos = cols if self._colmap is None else self._colmap[cols]
                d64 = buf.get(f'{ti}d64', data.size, np.float64)
                np.copyto(d64, data)
                _sparsetools.csr_matvec(nr, self.nS, lip, pos, d64, v, acc)
            fn(r0, r1, acc)                                       # consumed here (the buffer is the thread's next chunk's)

        if self._pool is None:
            for d in descs:
                chunk(d)
        else:
            list(self._pool.map(chunk, descs))

    # ---------------------------------------------------------------- the projections (float64, threaded)
    def _submit(self, fn, count):
        """Run ``fn(i)`` for ``i < count`` in the pool (now, without a pool); returns the futures."""
        if self._pool is None:
            for i in range(count):
                fn(i)
            return []
        return [self._pool.submit(fn, i) for i in range(count)]

    @staticmethod
    def _wait(futures):
        for f in futures:
            f.result()

    def _gram(self, A, B):
        """``A B^T`` (k x k, float64) over column blocks, the partial sums added in a fixed order."""
        n = A.shape[1]
        nb = 1 if self._pool is None else max(1, min(self._pool._max_workers, n // _MIN_BLOCK))
        cuts = np.linspace(0, n, nb + 1, dtype=np.int64)
        parts = [None] * nb

        def part(j):
            parts[j] = A[:, cuts[j]:cuts[j + 1]] @ B[:, cuts[j]:cuts[j + 1]].T

        self._wait(self._submit(part, nb))
        out = parts[0].copy()
        for q in parts[1:]:
            out += q
        return 0.5 * (out + out.T) if A is B else out

    def _map(self, fn):
        if self._blas_pool is None or len(self._blocks) == 1:
            return [fn(a, c) for a, c in self._blocks]
        return list(self._blas_pool.map(lambda ac: fn(*ac), self._blocks))

    _SUB = 32768          # columns per sub-block: a float64 copy of k' x _SUB stays in cache

    def _rows(self, M, a, c):
        """The columns ``a:c`` of ``M`` as float64 sub-blocks (a view in float64 storage; in float32 storage a
        copy into a cache-sized scratch, so the float32 bytes are read once and the arithmetic is float64)."""
        if M.dtype == np.float64:
            yield a, c, M[:, a:c]
            return
        key = ('sub', threading.get_ident(), int(M.shape[0]))        # per thread and per row count
        buf = self._scratch.get(key)
        if buf is None:
            buf = self._scratch[key] = np.empty((M.shape[0], self._SUB), dtype=np.float64)
        for s0 in range(a, c, self._SUB):
            s1 = min(c, s0 + self._SUB)
            t = buf[:, :s1 - s0]
            np.copyto(t, M[:, s0:s1])
            yield s0, s1, t

    def _dot(self, M, v):
        """``M v`` (``M``: (k', nS), one direction per row; ``v``: (nS,) float64), block partial sums in a
        fixed order."""
        def part(a, c):
            acc = None
            for s0, s1, t in self._rows(M, a, c):
                q = t @ v[s0:s1]
                acc = q if acc is None else acc + q
            return acc
        parts = self._map(part)
        out = parts[0].copy()
        for p in parts[1:]:
            out += p
        return out

    def _sub(self, v, M, c):
        """``v -= M^T c`` in place."""
        def part(a, e):
            for s0, s1, t in self._rows(M, a, e):
                np.subtract(v[s0:s1], t.T @ c, out=v[s0:s1])
        self._map(part)

    def _gather(self, y):
        """``y`` on the covered columns, float64 (nS,), written block by block in the threads into a buffer the
        deflation keeps: one gathered vector is live at a time (every caller consumes it before the next)."""
        y = np.asarray(y)
        buf = self._scratch.get('gathered')
        if buf is None:
            buf = self._scratch['gathered'] = np.empty(self.nS)
        S = self.S

        def part(j):
            a, c = self._blocks[j]
            np.copyto(buf[a:c], y[a:c] if S is None else y[S[a:c]], casting='unsafe')
        self._wait(self._submit(part, len(self._blocks)))
        return buf

    def _scatter(self, out, vS):
        """``vS`` written back to its columns of ``out`` (in ``out``'s dtype), block by block in the threads."""
        S = self.S

        def part(j):
            a, c = self._blocks[j]
            if S is None:
                np.copyto(out[a:c], vS[a:c], casting='unsafe')
            else:
                out[S[a:c]] = vS[a:c]
        self._wait(self._submit(part, len(self._blocks)))

    def project(self, y) -> np.ndarray:
        """``P y`` on the covered columns, float64 (nS,)."""
        vS = self._gather(y)
        self._sub(vS, self.V, self._dot(self.W, vS) / self.lam)
        return vS

    def _project_transposed(self, vS):
        """``P^T v`` in place."""
        self._sub(vS, self.W, self._dot(self.V, vS) / self.lam)
        return vS

    # ---------------------------------------------------------------- the solve
    def start(self, b, x0=None):
        """Subtract the part in the space from ``b`` in place (``b - B x_c``, computed in float64, stored in
        ``b``'s dtype) and return the solver's start: ``P x0`` (in ``x0``'s dtype), or None."""
        xb = self.x_c

        def shift(r0, r1, acc):
            b[r0:r1] = (np.asarray(b[r0:r1], dtype=np.float64) - acc).astype(b.dtype)

        self._forward(self._storage, xb, shift)
        if x0 is None:
            return None
        y0 = np.array(x0, copy=True)
        if self._projector is not None:
            self._projector.apply(y0)
        self._scatter(y0, self.project(y0))
        return y0

    def shifted(self, b, dtype=np.float32) -> np.ndarray:
        """The deflated right-hand side ``b - B x_c`` as a new array of ``dtype`` (``b``: the original one, an
        array or a parked vector, read in blocks; the product in float64, one pass over the matrix): what the
        solver's residual is measured against (CGLS's refresh)."""
        m = self.shape[0]
        out = np.empty(m, dtype=dtype)
        xb = self.x_c

        def shift(r0, r1, acc):
            out[r0:r1] = np.asarray(b[r0:r1], dtype=np.float64) - acc

        self._forward(self._storage, xb, shift)
        return out

    def bind(self, op, storage):
        """The solver's operator ``op`` (the column-scaled system) and the matrix it reads (for the float64
        passes of :meth:`start` and :meth:`resolve`)."""
        self.op = op
        self._storage = storage

    def operator(self) -> LinearOperator:
        """``B P`` as the solver's operator (same shape and dtype as the bound one)."""
        op = self.op
        m, n = op.shape
        dt = np.dtype(op.dtype)
        buf = np.empty(n, dtype=dt)

        pc = self._projector

        def matvec(y):
            y = np.asarray(y).reshape(-1)
            np.copyto(buf, y, casting='unsafe')
            if pc is not None:
                pc.apply(buf)                            # B P Pc: the iterate's part outside the allowed subspace
            self._scatter(buf, self.project(buf))        # goes first, so P's orthogonality to the space holds
            return op.matvec(buf)                        # (Pc and P do not commute; the constrained columns are
                                                         # mixed into W = N V)
        def rmatvec(u):
            out = np.asarray(op.rmatvec(u))
            self._scatter(out, self._project_transposed(self._gather(out)))
            if pc is not None:
                pc.apply(out)                            # the transpose: Pc P^T Pc A^T
            return out

        return LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec, dtype=dt)

    def constrain(self, v):
        """Remove from ``v`` (a solver vector, in place) its part along the space: the solver's right Lanczos
        vectors lie in the complement of the space in exact arithmetic; rounding, which the recurrence
        amplifies once the solve has converged, would otherwise let the iterate drift along directions the
        deflated operator cannot see."""
        if self._projector is not None:
            self._projector.apply(v)
        vS = self._gather(v)
        self._sub(vS, self.V, self._dot(self.V, vS))
        self._scatter(v, vS)

    def full(self, y) -> np.ndarray:
        """The solution an iterate ``y`` of the solver stands for, ``x_c + P y``, in ``y``'s dtype."""
        x = np.array(y, copy=True)
        if self._projector is not None:
            self._projector.apply(x)
        vS = self.project(x)
        vS += self.x_c
        self._scatter(x, vS)
        return x

    def resolve(self, x, b) -> np.ndarray:
        """The part in the space solved again against ``x`` (``x += V E^-1 V^T B^T (b - B x)``, float64, one pass
        over the matrix; ``b``: the original right-hand side, an array or a parked vector), in place; returns
        ``x``. Records the relative size of the change (:attr:`resolve_change`)."""
        xS = self._gather(x)

        def residual(r0, r1, prepared, buf):
            nr = r1 - r0
            r = buf.get('resid', nr, np.float64)
            np.copyto(r, b[r0:r1], casting='unsafe')
            acc = buf.get('resid_acc', nr, np.float64)
            acc.fill(0.0)
            for lip, pos, d64, _sky, _off in prepared:
                _sparsetools.csr_matvec(nr, self.nS, lip, pos, d64, xS, acc)
            r -= acc
            return r.reshape(nr, 1), None

        g = self._transposed_products(self._storage, residual, 1, self.n_sky_S)[:, 0]
        dc = self._dot(self.V, g) / self.lam
        cnorm = float(np.linalg.norm(self.c_star))
        self.resolve_change = float(np.linalg.norm(dc)) / cnorm if cnorm > 0 else 0.0
        self._sub(xS, self.V, -dc)
        self._scatter(x, xS)
        return x

    def summary(self) -> dict:
        """What the solve's record keeps (all reproducible run to run; the setup time is apart)."""
        st = self.settings
        out = {'basis': str(st.basis), 'order': int(st.order), 'storage': self.store.name,
                'terms': None if st.terms is None else list(st.terms), 'patterns': len(self.labels),
                'kept': self.kept, 'gauges': self.gauges,
                'seen': [float(v) for v in np.sort(self.seen)],
                'obliqueness_max': None if math.isnan(self.rho_max) else float(self.rho_max), 'columns': int(self.nS),
                'nullness_min': float(self.nullness.min()), 'nullness_median': float(np.median(self.nullness)),
                'pairing_residual': float(self.pairing_residual),
                'coarse_residual_drop': float(self.c_star @ (self.lam * self.c_star)),
                'resolve_change': None if math.isnan(self.resolve_change) else float(self.resolve_change),
                # the decomposition itself (small: patterns x patterns), for anyone reading the solve later
                'pattern_labels': list(self.labels),
                'nullness_by_pattern': [float(v) for v in self.nullness],
                'eigenvalues': [float(v) for v in self.eigenvalues],
                'seen_by_direction': [float(v) for v in self.seen],
                'kept_directions': [bool(v) for v in self.kept_mask],
                'exact_gauge_directions': [bool(v) for v in self.exact_mask],
                'eigenvectors': [[float(f'{v:.7g}') for v in col] for col in self.directions.T],   # direction j = weights over pattern_labels
                'excluded_degenerate': self.exclusion_info,                                        # Deflate(exclude=): the selection, or None
                'seen_constrained': None if self.exclusion_info is None else [float(v) for v in np.sort(self.seen_constrained)],
                'coefficients': [float(v) for v in self.c_star]}                                   # the kept directions' solved amplitudes, in kept order
        if self.signature_info is not None:
            out['signature_exclusions'] = self.signature_info                                      # Offsets(exclude=SkySignature): the terms constrained
        return out

    def summary_json(self) -> str:
        return json.dumps(self.summary(), sort_keys=True)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
        if getattr(self, '_blas_pool', None) is not None:
            self._blas_pool.shutdown(wait=False)
            self._blas_pool = None
