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
(the minimum-norm one). The rules (``Deflate(gauge_below=)``, ``Deflate(absorbed_by_shared=)``) may
*freeze* further directions: a frozen direction stays in the space, so the iterations never move along
it, with its coarse amplitude held at zero instead of solved.

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
from scipy.sparse import coo_matrix
from scipy.sparse import _sparsetools
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import LinearOperator, minres

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


def _pieces(storage, rows=_ROWS):
    """The matrix in row chunks: ``(r0, r1, [(indptr, column ids, data), ...])`` with a local indptr
    (int64, from 0) per piece; one piece per column range of a ColSplitCSR, one otherwise."""
    if isinstance(storage, ColSplitCSR):
        for bi in range(storage.nblocks):
            base = int(storage.row_bounds[bi])
            nrows = int(storage.row_bounds[bi + 1]) - base
            for lr0 in range(0, nrows, rows):
                lr1 = min(nrows, lr0 + rows)
                out = []
                for t, (d, idx, ip) in enumerate(storage.sub[bi]):
                    s0, s1 = int(ip[lr0]), int(ip[lr1])
                    lip = ip[lr0:lr1 + 1].astype(np.int64) - s0
                    out.append((lip, idx[s0:s1].astype(np.int64) + int(storage.cuts[t]), d[s0:s1]))
                yield base + lr0, base + lr1, out
        return
    blocks = storage.blocks if isinstance(storage, BlockCSR) else [storage]
    bounds = storage.row_bounds if isinstance(storage, BlockCSR) else (0,)
    for blk, base in zip(blocks, bounds):
        nrows = blk.shape[0]
        ip = blk.indptr
        for lr0 in range(0, nrows, rows):
            lr1 = min(nrows, lr0 + rows)
            s0, s1 = int(ip[lr0]), int(ip[lr1])
            lip = ip[lr0:lr1 + 1].astype(np.int64) - s0
            yield int(base) + lr0, int(base) + lr1, [(lip, blk.indices[s0:s1].astype(np.int64), blk.data[s0:s1])]


def _block_pinv_solve(G, R, project=None, shared=False):
    """``G^+ R`` for a symmetric positive semi-definite csr ``G`` and ``R`` (n x k), exact on each
    connected block of ``G``: batched dense eigen-decompositions (eigenvalues below
    :data:`DEPENDENT` of the block's largest count as zero: the solution of minimum norm), MINRES on
    blocks larger than :data:`_DENSE_BLOCK`, unless hub columns (coupled to :data:`_HUB_RATIO` times
    the block's median: offsets many frames share) explain their size: those are eliminated first by
    :func:`_schur_pinv_solve`, which leaves the per-frame blocks. ``project`` (a
    :class:`~selfcal.core.projector.ColumnProjector` over these columns, or None): the solution of the
    projected system ``P G P X = P R`` (columns the constraints join become hubs; a dense block or a MINRES
    block holding constrained columns is projected in place). ``shared``: look for the frame-shared columns
    (the hubs) whatever the blocks' size, for the rule that needs the response without them
    (``info['without_shared']``, ``info['shared_columns']``: absent when there are none; ValueError when the
    Schur path cannot serve). Returns ``(X, info)``."""
    n = G.shape[0]
    X = np.zeros((n, R.shape[1]))
    if n == 0:
        return X, {'blocks': 0, 'largest': 0, 'zero_eigenvalues': 0}
    if project is not None and project.columns.size == 0:
        project = None
    if project is not None:
        R = project.apply(np.array(R, dtype=np.float64, copy=True))
    ncomp, labels = connected_components(G, directed=False)
    counts = np.bincount(labels, minlength=ncomp)
    if counts.max() > _DENSE_BLOCK or shared:
        deg = np.diff(G.indptr)
        big = counts[labels] > (1 if shared else _DENSE_BLOCK)
        hub = np.flatnonzero(big & (deg > _HUB_RATIO * np.median(deg[big]))) if big.any() else np.zeros(0, dtype=np.int64)
        if project is not None:
            hub = np.union1d(hub, project.columns[big[project.columns]])      # constrained columns of big blocks
        if 0 < hub.size <= _SCHUR_MAX:
            out = _schur_pinv_solve(G, R, hub, project=project)
            if out is not None:
                return out
        if shared and hub.size:
            raise ValueError(f"Deflate(absorbed_by_shared=): the {hub.size} frame-shared offset columns cannot be "
                             f"eliminated by the Schur complement (at most {_SCHUR_MAX}, and the other columns must "
                             f"fall into blocks of at most {_DENSE_BLOCK}), so the response without them is not available")
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


def _schur_pinv_solve(G, R, hub, project=None):
    """``G X = R`` with the hub columns eliminated: for ``G = [[D, B], [B^T, C]]`` (``D`` the other
    columns, block-diagonal once the hubs are out; ``C`` the hubs' own part), ``X_hub`` solves the
    Schur complement ``(C - B^T D^+ B) X_hub = R_hub - B^T D^+ R_other`` by its pseudo-inverse and
    ``X_other = D^+ (R_other - B X_hub)``; ``D^+`` block by block as :func:`_block_pinv_solve` does.
    Then the solution of minimum norm, as the pseudo-inverse gives: the null space of ``G`` (offset
    combinations the data cannot see: gauges) is spanned by ``(-D^+ B v, v)`` over the Schur
    complement's null vectors ``v`` and by the blocks' null vectors that do not couple to the hubs;
    each candidate is checked against ``G`` and the verified ones are projected out of ``X`` (a response
    with a null component would put a gauge into the deflation space in part, and the iterate would
    drift along it). None when the other columns still hold a block larger than :data:`_DENSE_BLOCK`
    (the caller falls back). ``project`` (every constrained column a hub): the Schur complement is
    projected, ``P S P X_hub = P rhs``, so the hubs' response and the null vectors live in the allowed
    subspace."""
    n, k = G.shape[0], R.shape[1]
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
    Gc = G.tocoo()
    er, ec, ev = Gc.row.astype(np.int64), Gc.col.astype(np.int64), Gc.data
    hr, hc = is_hub[er], is_hub[ec]
    ff = ~hr & ~hc
    Gff = coo_matrix((ev[ff], (fpos[er[ff]], fpos[ec[ff]])), shape=(nF, nF)).tocsr()
    hh = hr & hc
    C = np.zeros((nH, nH))
    C[hpos[er[hh]], hpos[ec[hh]]] = ev[hh]
    del Gc, er, ec, ev, hr, hc, ff, hh
    ncomp, labels = connected_components(Gff, directed=False)
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

    def hub_part(rows):
        """The rows ``rows`` of G restricted to the hub columns, dense ``(len(rows), nH)``."""
        sub = G[rows]
        m = is_hub[sub.indices]
        rid = np.repeat(np.arange(rows.size), np.diff(sub.indptr))
        out = np.zeros((rows.size, nH))
        out[rid[m], hpos[sub.indices[m]]] = sub.data[m]
        return out

    def batches():
        for cols, U, inv, bad in groups:
            nb, s = cols.shape
            step = max(1, _SCHUR_BATCH // max(s * nH, 1))
            for a in range(0, nb, step):
                c = cols[a:a + step]
                flat = c.ravel()
                yield flat, c.shape[0], s, U[a:a + step], inv[a:a + step], bad[a:a + step], hub_part(F[flat]).reshape(c.shape[0], s, nH)

    RF, RH = R[F], R[H]
    S = C.copy()
    Rh = RH.copy()
    Xf = np.zeros((n, k))                               # the response of the per-frame columns alone (D^+ R_F)
    for flat, nb, s, U, inv, _bad, Bb in batches():
        Ut = np.swapaxes(U, 1, 2)
        Rb = RF[flat].reshape(nb, s, k)
        Yb = U @ (inv[:, :, None] * (Ut @ Bb))
        Zb = U @ (inv[:, :, None] * (Ut @ Rb))
        Xf[F[flat]] = Zb.reshape(-1, k)
        S -= Bb.reshape(-1, nH).T @ Yb.reshape(-1, nH)
        Rh -= Bb.reshape(-1, nH).T @ Zb.reshape(-1, k)
    if ph is not None:
        ph.apply_symmetric(S)
        ph.apply(Rh)
    w, V = np.linalg.eigh(0.5 * (S + S.T))
    ok = w > DEPENDENT * max(float(w.max()), 1e-300)
    zero += int((~ok).sum())
    Xh = V @ (np.where(ok, 1.0 / np.where(ok, w, 1.0), 0.0)[:, None] * (V.T @ Rh))
    V0 = V[:, ~ok]                                      # the Schur complement's null vectors (nH x m0)
    X = np.zeros((n, k))
    X[H] = Xh
    nulls = [np.concatenate((np.zeros((nF, V0.shape[1])), V0))]      # (-D^+ B v, v): the F part filled below
    extra = []                                          # the blocks' null vectors that do not couple to the hubs
    for flat, nb, s, U, inv, bad, Bb in batches():
        Ut = np.swapaxes(U, 1, 2)
        rhs = RF[flat].reshape(nb, s, k) - Bb @ Xh
        X[F[flat]] = (U @ (inv[:, :, None] * (Ut @ rhs))).reshape(-1, k)
        if V0.shape[1]:
            nulls[0][flat] = -(U @ (inv[:, :, None] * (Ut @ (Bb @ V0)))).reshape(-1, V0.shape[1])
        if bad.any() and len(extra) < 64:
            bi, ei = np.nonzero(bad)
            for i, e in zip(bi, ei):
                vec = U[i][:, e]
                if np.linalg.norm(Bb[i].T @ vec) <= 1e-10 * max(np.linalg.norm(Bb[i]), 1e-300):
                    u = np.zeros(n)
                    u[F[flat[i * s:(i + 1) * s]]] = vec
                    extra.append(u)
    cands = []
    if V0.shape[1]:
        full = np.zeros((n, V0.shape[1]))
        full[F] = nulls[0][:nF]
        full[H] = nulls[0][nF:]
        cands.append(full)
    if extra:
        cands.append(np.stack(extra, axis=1))
    if cands:
        Nc = np.concatenate(cands, axis=1)
        gscale = max(float(np.abs(G.diagonal()).max()), 1e-300)
        GN = G @ (project.apply(Nc.copy()) if project is not None else Nc)
        if project is not None:
            project.apply(GN)                              # the projected G: P G P
        good = np.linalg.norm(GN, axis=0) <= 1e-9 * gscale * np.maximum(np.linalg.norm(Nc, axis=0), 1e-300)
        if good.any():
            Q = np.linalg.qr(Nc[:, good])[0]
            X -= Q @ (Q.T @ X)
    return X, {'blocks': int(ncomp), 'largest': int(counts.max()), 'zero_eigenvalues': zero, 'hubs': nH,
               'without_shared': Xf, 'shared_columns': H}


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
                 projector=None):
        t0 = time.perf_counter()
        self.settings = settings
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
        # The columns the space lives on: those with entries (all of a compacted system's).
        covered = np.zeros(n, dtype=bool)
        for _r0, _r1, pieces in _pieces(storage):
            for _lip, cols, _data in pieces:
                covered[cols] = True
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
        del fs, pix
        col_norm = (np.ones(nS) if scale is None
                    else 1.0 / np.asarray(scale if self.S is None else np.asarray(scale)[self.S], dtype=np.float64))
        # The patterns: (order + 1)^2 per deflated term, on the term's covered bounding box.
        order, basis = int(settings.order), str(settings.basis)
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
                    rows_of.append((sel, fx[a][ix[sel] - x0] * fy[c][iy[sel] - y0]))
            if basis != 'polynomial' or order < POLY_ALWAYS:
                px = axis_functions('polynomial', POLY_ALWAYS, x1 - x0 + 1)
                py = axis_functions('polynomial', POLY_ALWAYS, y1 - y0 + 1)
                for a in range(POLY_ALWAYS + 1):
                    for c in range(POLY_ALWAYS + 1 - a):
                        if a + c == 0 or (basis == 'polynomial' and a <= order and c <= order):
                            continue
                        labels.append(f"{name}:P{a},{c}")
                        rows_of.append((sel, px[a][ix[sel] - x0] * py[c][iy[sel] - y0]))
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
        for i, (sel, vals) in enumerate(rows_of):
            Z[i, sel] = vals * col_norm[sel]
        del rows_of, term_of, iy, ix
        self._colmap = None
        if self.S is not None:
            cm = np.full(n, -1, dtype=np.int64)
            cm[self.S] = np.arange(nS)
            self._colmap = cm
        # The column constraints (solver coordinates) on the offset columns of this space: the offsets'
        # normal matrix is projected, G -> P G P, so the response stays in the allowed subspace.
        self._proj_off = None
        if projector is not None:
            cm = np.arange(n, dtype=np.int64) if self._colmap is None else self._colmap
            off = np.where(cm >= n_sky_S, cm - n_sky_S, -1)
            self._proj_off = projector.mapped(off)
        logger.info(f"Deflation: {k} patterns ({basis}, order {order}) of {len(terms)} sky term(s) on {nS} "
                    f"columns ({n_sky_S} sky, {n_off} offset and other)")
        self._pool = ThreadPoolExecutor(max_workers=int(threads)) if threads > 1 else None
        self._scratch = {}
        try:
            # Pass 1 (float64, one sweep over the matrix for all the patterns at once, scipy's multi-vector kernels:
            # the matrix is read once, not once per pattern): each pattern's sky image U = B F, the offsets' response
            # R = B_off^T U and the offsets' normal matrix G = B_off^T B_off (a sparse product per row block).
            t = time.perf_counter()
            t_g = 0.0
            Zt = np.ascontiguousarray(Z.T)                        # (nS, k): the kernels' vector layout
            sky_norm2 = np.zeros(k)
            Rt = np.zeros((n_off, k))
            gparts = []
            for r0, r1, pieces in _pieces(storage):
                nr = r1 - r0
                U = np.zeros((nr, k))
                offs = []
                for lip, cols, data in pieces:
                    pos = cols if self._colmap is None else self._colmap[cols]
                    d64 = data.astype(np.float64)
                    _sparsetools.csr_matvecs(nr, nS, k, lip, pos, d64, Zt, U)
                    o = pos >= n_sky_S
                    if o.any():
                        rr = np.repeat(np.arange(nr, dtype=np.int64), np.diff(lip))
                        offs.append((rr[o], pos[o] - n_sky_S, d64[o]))
                sky_norm2 += np.einsum('ij,ij->j', U, U)
                if offs:
                    rr = np.concatenate([o[0] for o in offs])
                    cc = np.concatenate([o[1] for o in offs])
                    dd = np.concatenate([o[2] for o in offs])
                    sub = coo_matrix((dd, (rr, cc)), shape=(nr, n_off)).tocsr()
                    _sparsetools.csc_matvecs(n_off, nr, k, sub.indptr.astype(np.int64), sub.indices.astype(np.int64), sub.data, U, Rt)
                    tg = time.perf_counter()
                    g = (sub.T @ sub).tocoo()
                    gparts.append((g.row.astype(np.int64), g.col.astype(np.int64), g.data))
                    t_g += time.perf_counter() - tg
                    del g, sub
                del U, offs
            R = np.ascontiguousarray(Rt.T)
            del Rt
            if gparts:
                G = coo_matrix((np.concatenate([p[2] for p in gparts]),
                                (np.concatenate([p[0] for p in gparts]), np.concatenate([p[1] for p in gparts]))),
                               shape=(n_off, n_off)).tocsr()
            else:
                G = coo_matrix((n_off, n_off)).tocsr()
            del gparts
            G.sum_duplicates()
            t_pass1 = time.perf_counter() - t
            t = time.perf_counter()
            want_shared = getattr(self.settings, 'absorbed_by_shared', None) is not None
            X, info = _block_pinv_solve(G, R.T, project=self._proj_off, shared=want_shared)
            res = G @ (-X) + R.T
            if self._proj_off is not None:
                self._proj_off.apply(res)                 # the projected normal equations' residual
            self.pairing_residual = float(np.max(np.linalg.norm(res, axis=0) /
                                                 np.maximum(np.linalg.norm(R, axis=1), 1e-300)))
            Z[:, n_sky_S:] = -X.T
            Zt[n_sky_S:, :] = -X                                  # the transposed copy follows (pass 2 reads it)
            # The frame-shared offsets' share of the absorption: Delta = the response with them minus the
            # response of the per-frame columns alone (nonzero on offset columns only). Pass 2 accumulates
            # what the data see of each pattern with that share removed (E_f below), a by-product of its sweep.
            Xf = info.get('without_shared')
            Delta = None if Xf is None else np.ascontiguousarray(X - Xf)          # (n_off, k)
            self.shared_offsets = 0 if Xf is None else int(np.asarray(info.get('shared_columns')).size)
            del G, R, X, res, Xf
            t_pair = time.perf_counter() - t

            # Pass 2 (float64, one sweep): N Z = B^T (B Z) for all the patterns, and B^T b alongside.
            t = time.perf_counter()
            NZt = np.zeros((nS, k))
            ATb = np.zeros(nS)
            pair_norm2 = np.zeros(k)
            bnorm2 = 0.0
            C1 = np.zeros((k, k)); C2 = np.zeros((k, k))
            for r0, r1, pieces in _pieces(storage):
                nr = r1 - r0
                chunk = []
                U = np.zeros((nr, k))
                offs = []
                for lip, cols, data in pieces:
                    pos = cols if self._colmap is None else self._colmap[cols]
                    d64 = data.astype(np.float64)
                    chunk.append((lip, pos, d64))
                    _sparsetools.csr_matvecs(nr, nS, k, lip, pos, d64, Zt, U)
                    if Delta is not None:
                        o = pos >= n_sky_S
                        if o.any():
                            rr = np.repeat(np.arange(nr, dtype=np.int64), np.diff(lip))
                            offs.append((rr[o], pos[o] - n_sky_S, d64[o]))
                pair_norm2 += np.einsum('ij,ij->j', U, U)
                if offs:
                    sub = coo_matrix((np.concatenate([o[2] for o in offs]), (np.concatenate([o[0] for o in offs]), np.concatenate([o[1] for o in offs]))),
                                     shape=(nr, n_off)).tocsr()
                    Ud = np.zeros((nr, k))
                    _sparsetools.csr_matvecs(nr, n_off, k, sub.indptr.astype(np.int64), sub.indices.astype(np.int64), sub.data, Delta, Ud)
                    C1 += U.T @ Ud
                    C2 += Ud.T @ Ud
                    del sub, Ud
                del offs
                bc = np.asarray(b[r0:r1], dtype=np.float64)
                for lip, pos, d64 in chunk:
                    _sparsetools.csc_matvecs(nS, nr, k, lip, pos, d64, U, NZt)
                    _sparsetools.csc_matvec(nS, nr, lip, pos, d64, bc, ATb)
                bnorm2 += float(bc @ bc)
                del chunk, bc, U
            del Zt
            NZ = np.ascontiguousarray(NZt.T)
            del NZt
            t_pass2 = time.perf_counter() - t
            self.nullness = np.sqrt(pair_norm2 / np.maximum(sky_norm2, 1e-300))

            # An orthonormal basis of the space diagonalising E = V^T N V; the directions kept: those the
            # data see (E's eigenvalue above DEPENDENT of the largest: exact gauges have ~0); the rest
            # stays orthogonal to the dropped. The obliqueness |N v| / v^T N v of each (about 1 for an
            # eigenvector of N, large for a slow direction mixed with fast ones) is kept as information.
            E = self._gram(Z, NZ)
            GZ = self._gram(Z, Z)
            NN = self._gram(NZ, NZ)
            # The same patterns with the frame-shared offsets' absorption removed (E itself when there are none)
            E_f = E + C1 + C1.T + C2 if Delta is not None else (E if want_shared else None)
            del Delta
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
        # data see of each, and which were kept / frozen / exact gauges (summary()).
        self.directions = np.array(T, copy=True)
        self.eigenvalues = np.array(lam, copy=True)
        wn = np.sqrt(np.maximum(np.einsum('ij,ij->j', T, NN @ T), 0.0))
        with np.errstate(divide='ignore', invalid='ignore'):
            rho = np.where(lam > 0, wn / np.where(lam > 0, lam, 1.0), np.inf)
        lam_max = max(float(lam.max()), 1e-300)
        self.seen = np.sqrt(np.maximum(lam, 0.0) / lam_max)        # per direction: what the data see of it, relative
        below = getattr(self.settings, 'gauge_below', None)
        exact = lam > DEPENDENT * lam_max                          # False: an exact gauge (left as a plain solve leaves it)
        keep = exact.copy()
        if below is not None:
            keep &= self.seen > float(below)                       # the user's rule: unseen directions are gauges
        # The frame-shared offsets' rule: a direction the data would see `ratio` times better if the shared
        # offsets could not absorb it is a sky shape the shared detector pattern mimics in every frame:
        # attributed to the instrument (frozen at zero sky amplitude) when ratio > absorbed_by_shared.
        by_shared = getattr(self.settings, 'absorbed_by_shared', None)
        self.visibility_ratio = None
        if E_f is not None:
            lam_f = np.einsum('ij,jk,ki->i', T.T, E_f, T)
            with np.errstate(divide='ignore', invalid='ignore'):
                self.visibility_ratio = np.sqrt(np.maximum(lam_f, 0.0) / np.where(lam > 0, lam, np.inf))
            if by_shared is not None:
                keep &= ~(exact & (self.visibility_ratio > float(by_shared)))
        self.absorbed_by_shared = None if by_shared is None else float(by_shared)
        frozen = exact & ~keep                                     # the rules' directions: frozen at zero amplitude
        self.kept_mask, self.frozen_mask, self.exact_mask = keep.copy(), frozen.copy(), ~exact
        self.gauges = int((~keep).sum())
        self.gauge_below = None if below is None else float(below)
        self.rho_max = float(rho[keep].max()) if keep.any() else math.nan
        # A frozen direction stays in the space (the solver's iterations are kept orthogonal to it in the data's
        # metric, like the kept ones) with its coarse amplitude held at zero instead of solved: the solution has
        # no part along it as the data measure it. (An exact gauge is left out of the space: the data never push
        # along it, and its value stays the start's, as without the rules: the zero point of the sky against the
        # frame scalars, for one.)
        T = T[:, exact]
        self.lam = lam[exact].copy()
        self._free = keep[exact]                                   # the space's directions whose amplitude is solved
        self.kept = int(self._free.sum())
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
        if self.visibility_ratio is not None:
            top = np.argsort(-np.nan_to_num(self.visibility_ratio))[:6]
            logger.info('Deflation: visibility without / with the frame-shared offsets, the 6 largest ratios: '
                        + ', '.join(f'{self.visibility_ratio[j]:.3g} (seen {self.seen[j]:.2g})' for j in top)
                        + ('' if self.absorbed_by_shared is None else f"; frozen above {self.absorbed_by_shared:g}: {int((self.frozen_mask).sum())}"))
        logger.info(f"Deflation: the directions' seen fractions (sqrt(lambda / lambda_max); the {min(12, self.seen.size)} "
                    f"smallest of {self.seen.size}): {spectrum}"
                    + ('' if self.gauge_below is None else f"; the rule gauge_below={self.gauge_below:g} leaves out {self.gauges}"))
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
        # The space's least-squares amplitudes. A frozen direction's amplitude is what the data ask for along it
        # (kept for the record and to take its signature out of the right-hand side) but the solution gets zero.
        self._c_data = (self.V @ ATb) / self.lam
        self.c_star = np.where(self._free, self._c_data, 0.0)
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
        logger.info(
            f"Deflation: offset response solved on {info['blocks']} blocks (largest {info['largest']}, "
            f"{info['zero_eigenvalues']} zero eigenvalues"
            + (f"; {info['hubs']} columns shared by many frames eliminated by a Schur complement" if info.get('hubs') else '')
            + f"), normal-equation residual {self.pairing_residual:.1e}; "
            f"|B z| / |B F| from {self.nullness.min():.2e} to {self.nullness.max():.2e}; {self.kept} of {k} "
            f"directions kept ({self.gauges} gauge or dependent left out); passes {t_pass1:.1f} s (of which the offsets' "
            f"normal matrix {t_g:.1f} s) + {t_pass2:.1f} s, blocks {t_pair:.1f} s; setup {self.setup_s:.1f} s")

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
        if self._pool is None or len(self._blocks) == 1:
            return [fn(a, c) for a, c in self._blocks]
        return list(self._pool.map(lambda ac: fn(*ac), self._blocks))

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
        y = np.asarray(y)
        return y.astype(np.float64) if self.S is None else y[self.S].astype(np.float64)

    def _scatter(self, out, vS):
        if self.S is None:
            out[...] = vS
        else:
            out[self.S] = vS

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
        ``b``'s dtype; the frozen directions' signature leaves too, :meth:`_shift_vector`) and return the
        solver's start: ``P x0`` (in ``x0``'s dtype), or None."""
        xb = self._shift_vector()
        for r0, r1, pieces in _pieces(self._storage):
            nr = r1 - r0
            acc = np.zeros(nr)
            for lip, cols, data in pieces:
                pos = cols if self._colmap is None else self._colmap[cols]
                _sparsetools.csr_matvec(nr, self.nS, lip, pos, data.astype(np.float64), xb, acc)
            b[r0:r1] = (np.asarray(b[r0:r1], dtype=np.float64) - acc).astype(b.dtype)
        if x0 is None:
            return None
        y0 = np.array(x0, copy=True)
        self._scatter(y0, self.project(x0))
        return y0

    def shifted(self, b, dtype=np.float32) -> np.ndarray:
        """The deflated right-hand side ``b - B x_c`` as a new array of ``dtype`` (``b``: the original one, an
        array or a parked vector, read in blocks; the product in float64, one pass over the matrix): what the
        solver's residual is measured against (CGLS's refresh)."""
        m = self.shape[0]
        out = np.empty(m, dtype=dtype)
        xb = self._shift_vector()
        for r0, r1, pieces in _pieces(self._storage):
            nr = r1 - r0
            acc = np.zeros(nr)
            for lip, cols, data in pieces:
                pos = cols if self._colmap is None else self._colmap[cols]
                _sparsetools.csr_matvec(nr, self.nS, lip, pos, data.astype(np.float64), xb, acc)
            out[r0:r1] = np.asarray(b[r0:r1], dtype=np.float64) - acc
        return out

    def _shift_vector(self) -> np.ndarray:
        """The part of the solution the solver's right-hand side is shifted by: ``x_c`` (the frozen directions at
        zero) plus, when directions are frozen, those directions at the amplitudes the data ask for: their
        signature leaves the right-hand side, so the shifted system is consistent (its least-squares gradient
        vanishes at the solution; a gradient that only the projection cancelled would be rounding the solver
        amplifies along the system's gauges) and the solution has zero amplitude along them all the same."""
        if self._free.all():
            return self.x_c
        xb = self.x_c.copy()
        self._sub(xb, self.V, -np.where(self._free, 0.0, self._c_data))
        return xb

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

        def matvec(y):
            y = np.asarray(y).reshape(-1)
            np.copyto(buf, y, casting='unsafe')
            self._scatter(buf, self.project(y))
            return op.matvec(buf)

        def rmatvec(u):
            out = np.asarray(op.rmatvec(u))
            self._scatter(out, self._project_transposed(self._gather(out)))
            return out

        return LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec, dtype=dt)

    def constrain(self, v):
        """Remove from ``v`` (a solver vector, in place) its part along the space: the solver's right Lanczos
        vectors lie in the complement of the space in exact arithmetic; rounding, which the recurrence
        amplifies once the solve has converged, would otherwise let the iterate drift along directions the
        deflated operator cannot see."""
        vS = self._gather(v)
        self._sub(vS, self.V, self._dot(self.V, vS))
        self._scatter(v, vS)

    def full(self, y) -> np.ndarray:
        """The solution an iterate ``y`` of the solver stands for, ``x_c + P y``, in ``y``'s dtype."""
        x = np.array(y, copy=True)
        vS = self.project(y)
        vS += self.x_c
        self._scatter(x, vS)
        return x

    def resolve(self, x, b) -> np.ndarray:
        """The part in the space solved again against ``x`` (``x += V E^-1 V^T B^T (b - B x)``, float64, one pass
        over the matrix; ``b``: the original right-hand side, an array or a parked vector), in place; returns
        ``x``. Records the relative size of the change (:attr:`resolve_change`)."""
        xS = self._gather(x)
        g = np.zeros(self.nS)
        for r0, r1, pieces in _pieces(self._storage):
            nr = r1 - r0
            r = np.array(b[r0:r1], dtype=np.float64)
            acc = np.zeros(nr)
            for lip, cols, data in pieces:
                pos = cols if self._colmap is None else self._colmap[cols]
                _sparsetools.csr_matvec(nr, self.nS, lip, pos, data.astype(np.float64), xS, acc)
            r -= acc
            for lip, cols, data in pieces:
                pos = cols if self._colmap is None else self._colmap[cols]
                _sparsetools.csc_matvec(self.nS, nr, lip, pos, data.astype(np.float64), r, g)
        dc = self._dot(self.V, g) / self.lam
        dc[~self._free] = 0.0                                       # the frozen directions stay at zero
        cnorm = float(np.linalg.norm(self.c_star))
        self.resolve_change = float(np.linalg.norm(dc)) / cnorm if cnorm > 0 else 0.0
        self._sub(xS, self.V, -dc)
        self._scatter(x, xS)
        return x

    def summary(self) -> dict:
        """What the solve's record keeps (all reproducible run to run; the setup time is apart)."""
        st = self.settings
        return {'basis': str(st.basis), 'order': int(st.order), 'storage': self.store.name,
                'terms': None if st.terms is None else list(st.terms), 'patterns': len(self.labels),
                'kept': self.kept, 'gauges': self.gauges, 'gauge_below': self.gauge_below,
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
                'absorbed_by_shared': self.absorbed_by_shared, 'shared_offsets': self.shared_offsets,
                'visibility_ratio_without_shared': None if self.visibility_ratio is None else [float(v) for v in self.visibility_ratio],
                'seen_by_direction': [float(v) for v in self.seen],
                'kept_directions': [bool(v) for v in self.kept_mask],
                'frozen_directions': [bool(v) for v in self.frozen_mask],
                'exact_gauge_directions': [bool(v) for v in self.exact_mask],
                'eigenvectors': [[float(f'{v:.7g}') for v in col] for col in self.directions.T],   # direction j = weights over pattern_labels
                'coefficients': [float(v) for v in self.c_star[self._free]],                       # the kept directions' solved amplitudes, in kept order
                'frozen_amplitudes': [float(v) for v in self._c_data[~self._free]]}                # what the data asked for along the frozen directions (held at zero), in frozen order

    def summary_json(self) -> str:
        return json.dumps(self.summary(), sort_keys=True)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
