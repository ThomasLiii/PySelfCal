"""The OFFSET pass of the N-pass solve: the offsets given the sky.

Given the sky, each frame's offsets are an independent least-squares problem: a frame's rows touch
only its own offset columns -- whatever the model's terms are (per-chunk values, a polynomial along a
chunk axis, coefficients of known functions of data variables) -- up to constraint rows,
which may tie a few frames together. An OFFSET pass subtracts the sky from every frame
(:class:`SkySubtractor`, a frame hook that also drops the observations off the sky map or too bright),
reduces each frame's rows, in the workers, to the frame's normal equations
(``setup_lsqr(..., fixed_sky=True, reduce=normal_equations)``: no matrix) and solves every frame
directly (:func:`solve_frames`); :func:`merge_offset_cals` joins the tiles' solutions. A term shared
across frames (``per="all"``, a frame group) would tie every frame together; the N-pass does not take
one.

Nothing here knows an instrument: the per-subchannel clip edges come in as an array (see
:meth:`selfcal.run.engine.RunContext.clip_group_edges`).
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix, vstack
from scipy.sparse.csgraph import connected_components

from ..core.projector import _DROP as _PROJECTOR_DROP
from ..io.atomic import atomic_path
from ..models.offset_structure import group_aux_edges

__all__ = ["group_wavelength_edges", "SkySubtractor", "FrameNormal", "normal_equations", "solve_frames",
           "merge_offset_cals", "offset_monitors"]

#: The most columns one block of an OFFSET pass's system may have: a frame's offsets, or several
#: frames' tied together by constraint rows.
MAX_BLOCK_COLUMNS = 20_000


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def group_wavelength_edges(det_wavelength, det_chunk_map, group_of_chunk, min_pixels=50):
    """Wavelength bin edges between consecutive chunk groups for the grouped
    outlier clip — :func:`selfcal.models.offset_structure.group_aux_edges`."""
    return group_aux_edges(det_wavelength, det_chunk_map, group_of_chunk, min_pixels=min_pixels)


def _basename(p):
    return os.path.basename(p.decode() if isinstance(p, bytes) else str(p))


# --------------------------------------------------------------------------- #
# the sky subtracted
# --------------------------------------------------------------------------- #
class SkySubtractor:
    """Subtract a solved sky ``Σ_j S_j(p) c_j(v)`` at each observation, for any
    number of sky blocks.

    The J maps are exported once by the parent as ``.npy`` files and memory-
    mapped in each worker (page cache shared across the pool), so J large maps
    cost one copy, not one per worker. ``sky_model`` supplies the component
    names (``sky/<name>`` in the cal) and their coefficient functions;
    ``aux_keys`` names the entries of the per-pixel aux list (default: the sky
    model's variables in order). ``variables`` (a
    :class:`~selfcal.models.variables.VariableSet`, frame values aligned with
    ``frames``) lets the coefficients read every other kind of data variable
    (per-frame values, sky maps, stored layers, functions); ``None`` keeps the
    detector-map path.

    As the frame hook of a fit of the offsets given this sky (the N-pass OFFSET
    pass), it also selects the observations by weight: ``exclude_off_map``
    gives the observations off the sky map (no sky to subtract there) weight 0,
    and ``bright_cut`` keeps only those whose modelled sky is below it -- unless
    fewer than ``min_pixels`` of the frame's weighted observations would remain.
    The hook then returns ``(sub_data, sub_weight)``.
    """

    def __init__(self, sky_cal, sky_model, export_dir, aux_keys=None, variables=None, frames=None,
                 exclude_off_map=False, bright_cut=None, min_pixels=0):
        self.exclude_off_map = bool(exclude_off_map)
        self.bright_cut = None if bright_cut is None else float(bright_cut)
        self.min_pixels = int(min_pixels)
        self.sky_cal = sky_cal
        self.sky_model = sky_model
        self.names = list(sky_model.names)
        self.aux_keys = tuple(sky_model.variables) if aux_keys is None else tuple(aux_keys)
        self.variables = variables
        self.frame_index = ({os.path.basename(f): i for i, f in enumerate(frames)}
                            if frames is not None else None)
        self._last = None
        self.export_dir = export_dir
        self.paths = {n: os.path.join(export_dir, f"{n}.npy") for n in self.names}
        self._maps = None
        os.makedirs(export_dir, exist_ok=True)
        with h5py.File(sky_cal, "r") as f:
            for n in self.names:
                src = f["sky"][n] if "sky" in f else (f["skymap"] if n == self.names[0]
                                                        else f["skymap_line"])
                if not os.path.exists(self.paths[n]):
                    with atomic_path(self.paths[n]) as tmp:
                        np.save(tmp, np.nan_to_num(src[:].astype(np.float32)))
            self.shape = tuple(f["skymap"].shape)

    def __getstate__(self):
        # What a worker process receives: the paths, not the maps a process has mapped or the last
        # prediction it cached (each worker maps the exported .npy files itself).
        state = dict(self.__dict__)
        state['_maps'] = None
        state['_last'] = None
        return state

    # --- worker side ---------------------------------------------------------
    def _load(self):
        if self._maps is None:
            self._maps = [np.load(self.paths[n], mmap_mode="r") for n in self.names]
        return self._maps

    def window(self, rc, shape):
        """The J sky maps on the subframe grid, plus an on-map mask.

        The subframe's ``ref_coords`` may overhang ANY map edge. A plain
        ``map[y0:y1, x0:x1]`` handles only the high side: a NEGATIVE y0/x0 is a
        Python negative slice start and silently returns an EMPTY window — for
        the 94 SEP frames overhanging the bottom/left edge that meant NO sky
        subtracted, and their refit offsets absorbed the LMC emission (the
        white-streak bug, fixed 2026-08-24). Off-map pixels get sky 0 and
        ``on_map=False``; fits against the residual must exclude them.
        """
        maps = self._load()
        y0, x0 = int(rc[0]), int(rc[2])
        sh, sw = shape
        H, W = self.shape
        on_map = np.zeros(shape, bool)
        out = [np.zeros(shape, np.float32) for _ in maps]
        ys, ye = max(0, -y0), min(sh, H - y0)
        xs, xe = max(0, -x0), min(sw, W - x0)
        if ye > ys and xe > xs:
            for o, m in zip(out, maps):
                o[ys:ye, xs:xe] = m[y0 + ys:y0 + ye, x0 + xs:x0 + xe]
            on_map[ys:ye, xs:xe] = True
        return out, on_map

    def coefficients(self, sub_aux, shape, ctx=None):
        """``c_j`` per subframe pixel for every term (None for a constant term)."""
        if self.variables is not None and ctx is not None:
            from ..models.variables import FrameObservations, ObservationVariables
            pixels = tuple(np.indices(shape).reshape(2, -1))
            det = ({k: np.asarray(sub_aux[i]) for i, k in enumerate(self.aux_keys) if i < len(sub_aux)}
                   if sub_aux is not None else {})
            i = (self.frame_index or {}).get(os.path.basename(ctx.file), 0)
            v = self.variables
            layers = {}
            if v.layers:
                from ..io.reproj import load_reproj_file
                got = load_reproj_file(ctx.file, fields=[f'layers/{n}' for n in v.layers])
                layers = {n: got[f'layers/{n}'] for n in v.layers}
            fctx = (FrameObservations(file=ctx.file, index=i, pixels=pixels, ref_coords=ctx.ref_coords,
                                      sub_data=ctx.sub_data, sub_weight=ctx.sub_weight,
                                      sub_mapping=ctx.sub_mapping) if v.frame_functions else None)
            aux = ObservationVariables(pixels, index=i, ref_coords=ctx.ref_coords, sub_mapping=ctx.sub_mapping,
                                       detector=det, frame_values={k: np.asarray(a)[i] for k, a in v.frame.items()},
                                       sky=v.sky, layers=layers, derived=v.derived,
                                       frame_functions=v.frame_functions, context=fctx)
        else:
            aux = {k: np.asarray(sub_aux[i]) for i, k in enumerate(self.aux_keys)
                   if sub_aux is not None and i < len(sub_aux)}
        out = []
        for comp in self.sky_model.components:
            c = comp.coefficients(aux)
            out.append(None if c is None else np.asarray(c, dtype=np.float64).reshape(shape))
        return out

    def predict(self, rc, shape, sub_aux, ctx=None):
        """Total modelled sky ``Σ_j c_j S_j`` on the subframe grid (float64), and the on-map mask.
        Without ``ctx``, the prediction the last ``__call__`` made for the same
        subframe is returned (the refit asks twice per frame)."""
        last = self._last
        if ctx is None and last is not None and last[3] is sub_aux and last[0] == (
                tuple(np.asarray(rc).tolist()), tuple(shape)):
            return last[1], last[2]
        maps, on_map = self.window(rc, shape)
        coef = self.coefficients(sub_aux, shape, ctx=ctx)
        pred = None
        for m, c in zip(maps, coef):
            term = m.astype(np.float64) if c is None else m * c
            pred = term if pred is None else pred + term
        return pred, on_map

    def __call__(self, ctx):
        sub_data = ctx.sub_data
        pred, on_map = self.predict(ctx.ref_coords, sub_data.shape, ctx.sub_aux, ctx=ctx)
        # (the same subframe's aux arrays, held so the identity check is sound)
        self._last = ((tuple(np.asarray(ctx.ref_coords).tolist()), tuple(sub_data.shape)), pred, on_map,
                      ctx.sub_aux)
        if not self.exclude_off_map and self.bright_cut is None:
            return sub_data - pred
        weight = np.asarray(ctx.sub_weight)
        keep = on_map if self.exclude_off_map else np.ones(on_map.shape, bool)
        if self.bright_cut is not None:
            faint = keep & (pred < self.bright_cut)
            if int(np.count_nonzero(faint & (weight > 0))) >= self.min_pixels:
                keep = faint
        return sub_data - pred, np.where(keep, weight, 0).astype(weight.dtype)

# --------------------------------------------------------------------------- #
# every frame solved
# --------------------------------------------------------------------------- #
@dataclass
class FrameNormal:
    """One frame's normal equations over its own columns: ``G = A_f^T A_f`` and ``h = A_f^T b_f``
    (float64) over the columns ``cols`` (sorted global column ids), ``bb = b_f . b_f`` and the frame's
    number of rows."""
    cols: np.ndarray
    G: np.ndarray
    h: np.ndarray
    bb: float
    rows: int


def normal_equations(rows, cols, data, b, num_rows) -> FrameNormal:
    """The normal equations of one frame's rows, the triplets ``setup_lsqr`` would assemble the matrix
    from (its ``reduce``): duplicate entries are summed, as the matrix sums them; the products
    accumulate in float64."""
    u, inv = np.unique(np.asarray(cols, dtype=np.int64), return_inverse=True)
    A = csr_matrix((np.asarray(data, dtype=np.float64), (np.asarray(rows), inv)), shape=(int(num_rows), u.size))
    bf = np.asarray(b, dtype=np.float64)
    return FrameNormal(cols=u, G=(A.T @ A).toarray(), h=np.asarray(A.T @ bf).ravel(), bb=float(bf @ bf),
                       rows=int(num_rows))


def _constraint_rows(constraint_blocks, num_cols):
    """All constraint rows as one CSR matrix over the full column layout, and their right-hand side."""
    if not constraint_blocks:
        return None, np.zeros(0)
    r, c, d, rhs, start = [], [], [], [], 0
    for blk in constraint_blocks:
        r.append(np.asarray(blk['rows_local'], dtype=np.int64) + start)
        c.append(np.asarray(blk['cols'], dtype=np.int64))
        d.append(np.asarray(blk['data'], dtype=np.float64))
        rhs.append(np.asarray(blk['b'], dtype=np.float64))
        start += int(blk['num_rows'])
    C = coo_matrix((np.concatenate(d), (np.concatenate(r), np.concatenate(c))), shape=(start, int(num_cols))).tocsr()
    C.sum_duplicates()
    return C, np.concatenate(rhs)


def _components(frames, C):
    """The connected sets of columns: ``(cols, label)``, the columns of the system (sorted global ids)
    and the set of each; two columns are in one set when a frame's rows or a constraint row touch both."""
    pieces = [f.cols for f in frames]
    if C is not None and C.nnz:
        pieces.append(C.indices.astype(np.int64))
    cols = np.unique(np.concatenate(pieces)) if pieces else np.zeros(0, dtype=np.int64)
    a, z = [], []
    for f in frames:                                   # a chain through each frame's columns
        if f.cols.size > 1:
            loc = np.searchsorted(cols, f.cols)
            a.append(loc[:-1])
            z.append(loc[1:])
    if C is not None and C.nnz:                        # and through each constraint row's
        loc = np.searchsorted(cols, C.indices.astype(np.int64))
        same_row = np.repeat(np.arange(C.shape[0]), np.diff(C.indptr))
        link = same_row[1:] == same_row[:-1]
        a.append(loc[:-1][link])
        z.append(loc[1:][link])
    if not a:
        return cols, np.arange(cols.size)
    a, z = np.concatenate(a), np.concatenate(z)
    graph = coo_matrix((np.ones(a.size, dtype=np.int8), (a, z)), shape=(cols.size, cols.size))
    return cols, connected_components(graph, directed=False)[1]


def _start(G, h, kind):
    """The engine's starting vector on one block (:meth:`~selfcal.run.engine.RunContext.x0`):
    ``kind`` ``'separate'``, each column at its own diagonal least-squares estimate ``h_c / G_cc`` (0
    where ``G_cc`` is 0); anything else, zero (the direct solve then gives the minimum-norm joint fit,
    the ``'offsets'`` start)."""
    if kind != 'separate':
        return np.zeros(h.size)
    d = np.diagonal(G)
    return np.divide(h, d, out=np.zeros_like(h), where=d > 0)


def _exclusion_rows(exclusions, num_cols):
    """The terms' hard constraints (:meth:`~selfcal.models.spec.ModelSpec.column_exclusions`: groups of
    columns and the directions excluded on each) as one CSR matrix of rows over the full column layout,
    one row per excluded direction of each group, or None."""
    if not exclusions:
        return None
    r, c, d, n = [], [], [], 0
    for cols, P in exclusions:
        cols = np.asarray(cols, dtype=np.int64)
        P = np.asarray(P, dtype=np.float64)
        for g in range(cols.shape[0]):
            real = cols[g] >= 0
            for j in range(P.shape[1]):
                r.append(np.full(int(real.sum()), n, dtype=np.int64))
                c.append(cols[g][real])
                d.append(P[real, j])
                n += 1
    if not n:
        return None
    return coo_matrix((np.concatenate(d), (np.concatenate(r), np.concatenate(c))), shape=(n, int(num_cols))).tocsr()


def _solve_stack(G, h, x0, *, precondition, damp, rcond, E=None):
    """Solve a stack of blocks of one size, ``G`` (n, k, k), ``h`` and ``x0`` (n, k): the column-scaled
    minimum-norm correction from ``x0``. ``E``: per block, the rows of its hard constraints ``E x = 0``
    (``(r, k)``, or None): the start is moved onto them and the correction kept in them, in the scaled
    unknowns, as the engine's solver does (:class:`~selfcal.core.projector.ColumnProjector`). Returns
    ``x`` (n, k), the number of directions left out (the constrained ones not counted) and the largest
    condition number kept."""
    d = np.sqrt(np.clip(np.diagonal(G, axis1=1, axis2=2), 0.0, None))
    live = d > 0                                       # a column no row touches keeps its start
    s = np.where(live, 1.0 / np.where(live, d, 1.0), 0.0) if precondition else live.astype(np.float64)
    n_con = 0
    Q = None
    if E is not None and any(e is not None and e.size for e in E):
        n, k = h.shape
        bases = []
        for i, e in enumerate(E):
            if e is None or not e.size:
                bases.append(np.zeros((k, 0)))
                continue
            u, sv, _ = np.linalg.svd((e * s[i][None, :]).T, full_matrices=False)   # the rows in the scaled unknowns
            bases.append(u[:, sv > _PROJECTOR_DROP * sv.max()] if sv.size and sv.max() > 0 else np.zeros((k, 0)))
        rmax = max(b.shape[1] for b in bases)
        Q = np.zeros((n, k, rmax))
        for i, b in enumerate(bases):
            Q[i, :, :b.shape[1]] = b
        n_con = int(sum(b.shape[1] for b in bases))
        y0 = np.where(live, x0 / np.where(live, s, 1.0), 0.0)
        y0 = y0 - np.einsum('nkr,nr->nk', Q, np.einsum('nkr,nk->nr', Q, y0))
        x0 = np.where(live, s * y0, x0)
    S = G * s[:, :, None] * s[:, None, :]
    if damp:
        S = S + (damp ** 2) * (live[:, :, None] & np.eye(S.shape[1], dtype=bool)[None])
    r = s * (h - np.einsum('nij,nj->ni', G, x0))
    if Q is not None:                                  # P S P and P r, P = I - Q Q^T
        SQ = np.einsum('nij,njr->nir', S, Q)
        S = (S - np.einsum('nir,njr->nij', SQ, Q) - np.einsum('nir,njr->nij', Q, SQ)
             + np.einsum('nir,nrs,njs->nij', Q, np.einsum('nkr,nks->nrs', Q, SQ), Q))
        r = r - np.einsum('nkr,nr->nk', Q, np.einsum('nkr,nk->nr', Q, r))
    w, V = np.linalg.eigh(S)                           # ascending eigenvalues
    top = w[:, -1:]
    keep = (w > rcond * top) & (top > 0)
    inv = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)
    y = np.einsum('nik,nk->ni', V, inv * np.einsum('nik,ni->nk', V, r))
    weakest = np.where(keep, w, np.inf).min(axis=1)
    cond = np.where(np.isfinite(weakest) & (weakest > 0), top[:, 0] / np.where(weakest > 0, weakest, 1.0), 1.0)
    left_out = int((live.sum(axis=1) - keep.sum(axis=1)).sum()) - n_con
    return x0 + s * y, left_out, float(cond.max()) if cond.size else 1.0


def solve_frames(frames, constraint_blocks, num_cols, *, start=None, exclusions=None, precondition=True,
                 damp=0.0, rcond=1e-12, dtype=np.float64, chunk=2048):
    """Solve an OFFSET pass's system from the frames' normal equations (``frames``: a
    :class:`FrameNormal` per frame, None for a frame with no row) and the constraint rows
    (``constraint_blocks``, the dicts ``setup_lsqr`` builds), over ``num_cols`` columns.

    The columns fall into the sets that the frames' rows and the constraint rows tie together -- a
    frame's own columns, unless a constraint row ties frames -- and each set is solved from its dense
    normal equations: the minimum-norm least-squares correction from the engine's starting vector
    (``start``: its kind, :func:`_start`) in column-scaled unknowns (``precondition``), damped by
    ``damp``, under the terms' hard constraints (``exclusions``,
    :meth:`~selfcal.models.spec.ModelSpec.column_exclusions`) -- the solution the engine's iterative
    solve converges to, so the directions the data leave undetermined (two terms that duplicate each
    other) keep the start the joint solve gives them. ``rcond``: a block's eigen-directions weaker
    than ``rcond`` times its strongest are left out.

    Returns ``(x, summary)``: the solution in the full column layout (``dtype``; zero on the columns no
    row touches) and a dict -- ``method`` ``'direct'``, ``blocks``, ``columns``, ``rows``, ``joined`` (the
    blocks of several frames or constraint rows), ``largest_block``, ``left_out`` (directions left out,
    beyond the constrained ones),
    ``largest_condition`` (of a block's scaled normal matrix, over the directions kept),
    ``true_residual`` ``|b - A x|`` and ``bnorm`` ``|b|`` over every row, data and constraint, in float64,
    and ``wall_s``. Raises ``ValueError`` when a set has more than :data:`MAX_BLOCK_COLUMNS` columns."""
    t0 = time.perf_counter()
    frames = [f for f in frames if f is not None and f.cols.size]
    C, cb = _constraint_rows(constraint_blocks, num_cols)
    X = _exclusion_rows(exclusions, num_cols)
    links = C if X is None else (X if C is None else vstack([C, X]).tocsr())
    cols, label = _components(frames, links)
    n_comp = int(label.max()) + 1 if label.size else 0
    sizes = np.bincount(label, minlength=n_comp)
    if sizes.size and sizes.max() > MAX_BLOCK_COLUMNS:
        raise ValueError(f"the offsets do not separate frame by frame: {int(sizes.max())} columns are tied "
                         f"together (by offsets shared across frames, or constraint rows over many frames); "
                         f"a direct solve takes at most {MAX_BLOCK_COLUMNS}")
    # The frames and constraint rows of each set. A set that is one frame and no constraint row (the
    # usual case) is solved from the frame's own normal equations.
    frame_comp = np.array([label[np.searchsorted(cols, f.cols[0])] for f in frames], dtype=np.int64)
    row_comp = np.zeros(0, dtype=np.int64)
    if C is not None and C.nnz:
        has = np.diff(C.indptr) > 0
        row_comp = np.full(C.shape[0], -1, dtype=np.int64)
        row_comp[has] = label[np.searchsorted(cols, C.indices[C.indptr[:-1][has]].astype(np.int64))]
    n_frames_in = np.bincount(frame_comp, minlength=n_comp)
    n_rows_in = np.bincount(row_comp[row_comp >= 0], minlength=n_comp)
    alone = n_frames_in[frame_comp] == 1
    alone &= n_rows_in[frame_comp] == 0
    # The hard constraints' rows of each set (a group of a term lies in one frame's set, or a joined one).
    x_rows_of = {}
    if X is not None and X.nnz:
        has_x = np.diff(X.indptr) > 0
        for i, k in zip(np.flatnonzero(has_x),
                        label[np.searchsorted(cols, X.indices[X.indptr[:-1][has_x]].astype(np.int64))]):
            x_rows_of.setdefault(int(k), []).append(int(i))
    blocks = [(f.cols, f.G, f.h, f.bb, int(k)) for f, a, k in zip(frames, alone, frame_comp) if a]
    joined = sorted(set(frame_comp[~alone].tolist()) | set(row_comp[row_comp >= 0].tolist()))
    if joined:
        members = {k: [] for k in joined}
        for f, k, a in zip(frames, frame_comp, alone):
            if not a:
                members[int(k)].append(f)
        rows_of = {k: [] for k in joined}
        for i, k in enumerate(row_comp):
            if k >= 0:
                rows_of[int(k)].append(i)
        order = np.argsort(label, kind='stable')
        bounds = np.concatenate(([0], np.cumsum(sizes)))
        for k in joined:
            kc = cols[order[bounds[k]:bounds[k + 1]]]
            G, h, bb = np.zeros((kc.size, kc.size)), np.zeros(kc.size), 0.0
            for f in members[k]:
                loc = np.searchsorted(kc, f.cols)
                G[np.ix_(loc, loc)] += f.G
                h[loc] += f.h
                bb += f.bb
            if rows_of[k]:
                Ck = C[rows_of[k]][:, kc].toarray()
                dk = cb[rows_of[k]]
                G += Ck.T @ Ck
                h += Ck.T @ dk
                bb += float(dk @ dk)
            blocks.append((kc, G, h, bb, int(k)))
    extra_bb = 0.0
    if C is not None:                                  # constraint rows with no entry still count in |b|
        empty = np.diff(C.indptr) == 0
        extra_bb = float(cb[empty] @ cb[empty])

    x = np.zeros(int(num_cols), dtype=dtype)
    by_size = {}
    for i, b in enumerate(blocks):
        by_size.setdefault(b[0].size, []).append(i)
    res2, bb_tot, left_out, worst = extra_bb, extra_bb + float(sum(b[3] for b in blocks)), 0, 1.0
    for _, idx in sorted(by_size.items()):
        for s in range(0, len(idx), chunk):
            part = [blocks[i] for i in idx[s:s + chunk]]
            G = np.stack([p[1] for p in part])
            h = np.stack([p[2] for p in part])
            x0 = np.stack([_start(p[1], p[2], start) for p in part])
            E = ([X[x_rows_of[p[4]]][:, p[0]].toarray() if p[4] in x_rows_of else None for p in part]
                 if x_rows_of else None)
            xs, lo, cond = _solve_stack(G, h, x0, precondition=precondition, damp=damp, rcond=rcond, E=E)
            left_out += lo
            worst = max(worst, cond)
            Gx = np.einsum('nij,nj->ni', G, xs)
            bb = np.array([p[3] for p in part])
            res2 += float(np.sum(bb - 2.0 * np.einsum('ni,ni->n', xs, h) + np.einsum('ni,ni->n', xs, Gx)))
            for j, p in enumerate(part):
                x[p[0]] = xs[j]
    n_rows = int(sum(f.rows for f in frames) + (C.shape[0] if C is not None else 0))
    return x, {"method": "direct", "blocks": len(blocks), "columns": int(cols.size), "rows": n_rows,
               "joined": len(joined), "largest_block": int(sizes.max()) if sizes.size else 0,
               "left_out": left_out, "largest_condition": worst,
               "true_residual": float(np.sqrt(max(res2, 0.0))), "bnorm": float(np.sqrt(bb_tot)),
               "wall_s": time.perf_counter() - t0}


# --------------------------------------------------------------------------- #
# the tiles' solutions joined
# --------------------------------------------------------------------------- #
def merge_offset_cals(pieces, out, *, attrs=None):
    """Join offset solutions over disjoint frame sets (the tiles of an OFFSET pass, each a cal of a
    fixed-sky solve) into one file in the cal layout: ``offsets/map_<m>`` (with each map's
    ``n_basis`` / ``basis`` attributes), ``offset_coverage`` / ``offset_coverage_frac``,
    ``chunk_maps`` and ``reproj_list``, the pieces' rows in piece order. The
    pieces must hold the same offset maps (chunk maps and functions) and no frame twice."""
    maps = None
    joined = {}
    names = []
    first_attrs = {}
    for path in pieces:
        with h5py.File(path, "r") as f:
            K = int(f.attrs.get("num_maps", 0))
            if maps is None:
                maps = [f["chunk_maps"][f"map_{m}"][()] for m in range(K)]
                first_attrs = {m: dict(f["offsets"][f"map_{m}"].attrs) for m in range(K)}
            elif K != len(maps):
                raise ValueError(f"merge_offset_cals: {os.path.basename(path)} has {K} offset maps, the first "
                                 f"piece {len(maps)}")
            for m in range(K):
                if not np.array_equal(f["chunk_maps"][f"map_{m}"][()], maps[m]):
                    raise ValueError(f"merge_offset_cals: the chunk map of map {m} differs in "
                                     f"{os.path.basename(path)}")
                if dict(f["offsets"][f"map_{m}"].attrs).get("n_basis") != first_attrs[m].get("n_basis"):
                    raise ValueError(f"merge_offset_cals: map {m} multiplies other functions in "
                                     f"{os.path.basename(path)}")
                for group in ("offsets", "offset_coverage", "offset_coverage_frac"):
                    if group in f:
                        joined.setdefault((group, m), []).append(f[group][f"map_{m}"][()])
            if "frame_scalar" in f:
                raise ValueError(f"merge_offset_cals: {os.path.basename(path)} holds a separate per-frame scalar "
                                 f"(frame_scalar), an older layout that is not read")
            names += [r.decode() if isinstance(r, bytes) else str(r) for r in f["reproj_list"][()]]
    bases = [os.path.basename(n) for n in names]
    if len(set(bases)) != len(bases):
        raise ValueError("merge_offset_cals: a frame is in more than one piece")
    with atomic_path(out) as tmp, h5py.File(tmp, "w") as f:
        f.attrs["num_maps"] = len(maps or ())
        f.attrs["fixed_sky"] = True
        f.attrs["merged_from"] = np.array([os.path.basename(os.fspath(p)).encode() for p in pieces])
        f.create_dataset("reproj_list", data=np.array([n.encode() for n in names]))
        for m, cm in enumerate(maps or ()):
            ds = f.create_dataset(f"offsets/map_{m}", data=np.concatenate(joined[("offsets", m)]),
                                  compression="gzip")
            for k, v in first_attrs[m].items():
                ds.attrs[k] = v
            for group in ("offset_coverage", "offset_coverage_frac"):
                if (group, m) in joined:
                    f.create_dataset(f"{group}/map_{m}", data=np.concatenate(joined[(group, m)]),
                                     compression="gzip")
            f.create_dataset(f"chunk_maps/map_{m}", data=cm, compression="gzip")
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
    print(f"[npass] OFFSET: {len(pieces)} piece(s), {len(names)} frames -> {out}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# monitors
# --------------------------------------------------------------------------- #
def _offset_values(path):
    """``(frame names, per-frame values of every offset map)`` of a cal or an offsets file: each
    frame's row is its maps' stored values side by side."""
    with h5py.File(path, "r") as f:
        names = [_basename(r) for r in f["reproj_list"][()]]
        if "offsets" in f:
            K = int(f.attrs.get("num_maps", len(f["offsets"])))
            parts = [f["offsets"][f"map_{m}"][()] for m in range(K)]
        else:
            parts = [f["offset"][()]] if "offset" in f else []
    cols = [np.asarray(p, dtype=np.float64).reshape(len(names), -1) for p in parts]
    vals = np.concatenate(cols, axis=1) if cols else np.zeros((len(names), 0))
    return names, vals


def offset_monitors(off_path, prev_off_path=None):
    """The solved offsets of an OFFSET pass, whatever the model's offset terms: the frames solved
    (every value finite) and, given the previous offsets (any cal or offsets file of the same terms),
    the RMS step over the frames both hold (every stored value of every map)."""
    names, vals = _offset_values(off_path)
    ok = np.all(np.isfinite(vals), axis=1)
    out = {"n_frames": len(names), "n_fit": int(ok.sum())}
    if prev_off_path:
        pnames, pvals = _offset_values(prev_off_path)
        idx = {n: i for i, n in enumerate(pnames)}
        pairs = [(i, idx[n]) for i, n in enumerate(names) if n in idx and ok[i]]
        if pairs and pvals.shape[1] == vals.shape[1]:
            a = np.array([i for i, _ in pairs])
            b = np.array([j for _, j in pairs])
            out["step_rms"] = float(np.sqrt(np.mean((vals[a] - pvals[b]) ** 2)))
            out["n_shared"] = len(pairs)
    return out
