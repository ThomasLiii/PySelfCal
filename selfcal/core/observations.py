"""The solve's matrix kept as its observations and made again when read (``sc.Compute(matrix='compact')``).

An observation is one unmasked pixel of one frame inside the job's window: a row of the least-squares system. The
stored matrix (:mod:`selfcal.core.system`) keeps every entry of the row, its column and its float32 value, 8 bytes
each and 8 to 12 entries per row. The table keeps what the row is made from, 16 bytes: the reference pixel the
observation lands on, its detector position and its weight (after the mask, the window and the weight function).
Every pass over the matrix (the products of the solve, the column norms, the start vector, the deflation's sums)
makes the rows again with the arithmetic of the build (:mod:`selfcal.core.assembly`): the same bilinear corners of
the detector position, the same float32 sums of their weights per chunk, the same products with the weight and the
model's functions, in the same column order. The entries are the stored matrix's to the bit, and so the products are
the stored path's for the same thread count.

The rows are made by compiled loops (numba, imported when a table is made); the model's functions, a sky term's
coefficient or an offset term's basis, are evaluated by the model's own code on each chunk's observation variables,
as the build evaluates them. A table holds the data rows in blocks, one per assembly batch, and the constraint rows
as a small stored matrix after them. The row-split transpose product, the column-norm chunks and the deflation's row
chunks read it through the same functions as the stored matrices (:mod:`selfcal.core.solve`,
:mod:`selfcal.core.deflation`, :mod:`selfcal.core.solution`).
"""
from __future__ import annotations

import itertools
import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.sparse import _sparsetools, csr_matrix

from ..models.variables import ObservationVariables
from .blockcsr import _csr_shell

try:
    from numba import njit as _njit
except ImportError:                      # the table refuses to exist without it (require_numba)
    _njit = None

logger = logging.getLogger(__name__)

__all__ = ['ObservationTable', 'require_numba']

#: Rows of a data block per task of the streams and the coverage-floor pass.
_TASK_ROWS = 1 << 19
#: Rows per evaluation of the model's functions and per call of the loops: the functions' temporaries (a few arrays
#: of this length per function) and the padded scratch of the expansion stay in cache at this size.
_EVAL_ROWS = 1 << 16


def require_numba():
    """Raise ``ImportError`` with the fix when numba, which compiles the table's loops, is not installed."""
    if _njit is None:
        raise ImportError("Compute(matrix='compact') keeps the matrix as an observation table and makes its rows with "
                          "compiled loops, which need the numba package: pip install numba "
                          "(or pip install 'selfcal[compact]')")


def _jit(fn):
    """numba's ``njit(nogil=True, cache=True)`` when numba is installed; the plain function otherwise (never called:
    :func:`require_numba` runs before any table exists)."""
    return fn if _njit is None else _njit(nogil=True, cache=True)(fn)


# ---------------------------------------------------------------------------------------------------- the loops
@_jit
def _corners(xi, yi, H, W):
    """The four bilinear corners of a detector position, in the interpolation matrix's storage order (top-left,
    top-right, bottom-left, bottom-right): flat pixel index, weight (float64 fractions of the coordinates, cast to
    float32, as the matrix stores them) and whether the corner lies on the detector."""
    r0 = np.int64(np.floor(yi))
    c0 = np.int64(np.floor(xi))
    r_frac = np.float64(yi) - np.float64(r0)
    c_frac = np.float64(xi) - np.float64(c0)
    rf_inv = 1.0 - r_frac
    cf_inv = 1.0 - c_frac
    cw0 = np.float32(rf_inv * cf_inv)
    cw1 = np.float32(rf_inv * c_frac)
    cw2 = np.float32(r_frac * cf_inv)
    cw3 = np.float32(r_frac * c_frac)
    in_r0 = r0 >= 0 and r0 < H
    in_c0 = c0 >= 0 and c0 < W
    in_r1 = r0 + 1 >= 0 and r0 + 1 < H
    in_c1 = c0 + 1 >= 0 and c0 + 1 < W
    p0 = r0 * W + c0
    return (p0, p0 + 1, p0 + W, p0 + W + 1, cw0, cw1, cw2, cw3,
            in_r0 and in_c0, in_r0 and in_c1, in_r1 and in_c0, in_r1 and in_c1)


@_jit
def _sample_maps(x, y, maps, H, W, out):
    """Detector maps at the observations' positions, ``out[a, i]`` for map ``a``: the float32 sum over the corners
    on the detector, in storage order, of the corner weight times the map (the build's ``interp_matrix @ map``)."""
    A = maps.shape[0]
    for i in range(x.shape[0]):
        xi = x[i]
        yi = y[i]
        for a in range(A):
            out[a, i] = np.float32(0.0)
        if not (np.isfinite(xi) and np.isfinite(yi)):
            continue
        p0, p1, p2, p3, cw0, cw1, cw2, cw3, ok0, ok1, ok2, ok3 = _corners(xi, yi, H, W)
        for a in range(A):
            s = np.float32(0.0)
            if ok0:
                s = np.float32(s + np.float32(cw0 * maps[a, p0]))
            if ok1:
                s = np.float32(s + np.float32(cw1 * maps[a, p1]))
            if ok2:
                s = np.float32(s + np.float32(cw2 * maps[a, p2]))
            if ok3:
                s = np.float32(s + np.float32(cw3 * maps[a, p3]))
            out[a, i] = s


_NO_I32 = np.zeros(0, dtype=np.int32)
_NO_I64 = np.zeros(0, dtype=np.int64)
_NO_F32 = np.zeros(0, dtype=np.float32)
_NO_F64 = np.zeros(0, dtype=np.float64)
_NO_U8 = np.zeros(0, dtype=np.uint8)
_ZERO64 = np.float64(0.0)


@_jit
def _pass(mode, lo, hi, P, x, y, w, frame_rows, base, zb, fl, J, num_sky, sky_vals, K, cmaps, kinds, H, W, nb, nboff,
          phi, use_cmap, col_map, use_scale, scale, S, vec, zero, out, drop, counts, cols_s, vals_s):
    """One pass over a block's rows ``lo:hi``: each row's entries made in column order and used as ``mode`` says.

    ======  ===================================================================================================
    0       ``out[i - lo] = A[i] . vec``: the products summed in column order from ``zero`` (a scalar of the
            dtype of the entries and ``vec`` promoted: scipy's ``csr_matvec`` on the stored row, which upcasts
            float32 entries for a float64 vector), the sum stored in ``out``'s dtype; a row the coverage floor
            zeroed gives 0.
    1       ``out[c] += A[i, c] * vec[i - lo]`` in row order (scipy's ``csc_matvec`` on the stored rows, in
            ``out``'s dtype); a floored row adds nothing.
    2       the rows padded: row ``i - lo`` holds ``counts[i - lo]`` entries from slot ``(i - lo) * S`` of
            ``cols_s`` / ``vals_s`` (copied there from the row's scratch); a floored row keeps its entries with
            the value +0.0, as the stored matrix keeps them.
    3       ``out[i - lo]`` (a uint8 array): 1 when the row has an entry in a column of ``drop`` (a boolean over
            the leading columns of the full layout; the coverage floor's columns), with the columns unmapped and
            unscaled.
    ======  ===================================================================================================

    A row's entries: the sky terms first (``j * num_sky + P``, the value ``w * c_j`` made by the model's code:
    ``sky_vals`` ``(J, n)`` at the row's ``i - lo``), then each offset map: the chunks of the bilinear corners on
    the detector, the corners' weights summed per chunk in storage order (float32, as the product of the
    interpolation matrix and the chunk map's one-hot matrix sums them; a zero sum is dropped), the chunks in
    ascending order, each with the value ``w * sum`` times every nonzero basis function (``nb[m]`` of them, from
    ``phi`` ``(sum nb, n)``; none: the value itself). An entry whose value is zero is kept only when the row's
    right-hand side is not (``zb``: the rows whose right-hand side is zero), as the build keeps them. ``kinds[m]``
    (:func:`_map_kind`) says what the map's form spares: 1, one chunk over the whole detector, so the corners'
    weights sum straight into it; 2, every pixel its own chunk, so the corners are the chunks and storage order is
    ascending order; 0, the chunk of each corner looked up and merged. Each entry's column goes through
    ``col_map`` and its value is scaled by ``scale[column]`` (float32) as it is written, when ``use_cmap`` /
    ``use_scale``. The corners are those of :func:`_corners`, written out here.

    The row is built in this loop, not in a function called per row: such a call, with these arguments (every
    array flattened into its fields), cost more per observation than the row's own work. The arguments a mode
    does not use are empty arrays."""
    keys = np.empty(4, dtype=np.int64)
    sums = np.empty(4, dtype=np.float32)
    oc = np.empty(S, dtype=np.int32)
    ov = np.empty(S, dtype=np.float32)
    L = drop.shape[0]
    f = np.searchsorted(frame_rows, lo, side='right') - 1
    zi = np.searchsorted(zb, lo)
    fi = np.searchsorted(fl, lo)
    for i in range(lo, hi):
        while i >= frame_rows[f + 1]:
            f += 1
        bz = zi < zb.shape[0] and zb[zi] == i
        if bz:
            zi += 1
        floored = fi < fl.shape[0] and fl[fi] == i
        if floored:
            fi += 1
        k = i - lo
        if mode == 0:
            out[k] = 0
        if floored and mode != 2:
            continue
        Pi = P[i]
        xi = x[i]
        yi = y[i]
        wi = w[i]
        cnt = 0
        for j in range(J):
            v = sky_vals[j, k]
            if v != 0 or not bz:
                c = j * num_sky + Pi
                if use_cmap:
                    c = col_map[c]
                if use_scale:
                    v = np.float32(v * scale[c])
                oc[cnt] = c
                ov[cnt] = v
                cnt += 1
        if K > 0 and np.isfinite(xi) and np.isfinite(yi):
            # the four bilinear corners in the interpolation matrix's storage order (top-left, top-right,
            # bottom-left, bottom-right): float64 fractions of the coordinates, the weights cast to float32
            r0 = np.int64(np.floor(yi))
            c0 = np.int64(np.floor(xi))
            r_frac = np.float64(yi) - np.float64(r0)
            c_frac = np.float64(xi) - np.float64(c0)
            rf_inv = 1.0 - r_frac
            cf_inv = 1.0 - c_frac
            cw0 = np.float32(rf_inv * cf_inv)
            cw1 = np.float32(rf_inv * c_frac)
            cw2 = np.float32(r_frac * cf_inv)
            cw3 = np.float32(r_frac * c_frac)
            in_r0 = r0 >= 0 and r0 < H
            in_c0 = c0 >= 0 and c0 < W
            in_r1 = r0 + 1 >= 0 and r0 + 1 < H
            in_c1 = c0 + 1 >= 0 and c0 + 1 < W
            ok0 = in_r0 and in_c0
            ok1 = in_r0 and in_c1
            ok2 = in_r1 and in_c0
            ok3 = in_r1 and in_c1
            p0 = r0 * W + c0
            p1 = p0 + 1
            p2 = p0 + W
            p3 = p0 + W + 1
            for m in range(K):
                kind = kinds[m]
                ns = 0
                if kind == 1:
                    if ok0:
                        sums[0] = cw0
                        ns = 1
                    if ok1:
                        if ns == 0:
                            sums[0] = cw1
                            ns = 1
                        else:
                            sums[0] = np.float32(sums[0] + cw1)
                    if ok2:
                        if ns == 0:
                            sums[0] = cw2
                            ns = 1
                        else:
                            sums[0] = np.float32(sums[0] + cw2)
                    if ok3:
                        if ns == 0:
                            sums[0] = cw3
                            ns = 1
                        else:
                            sums[0] = np.float32(sums[0] + cw3)
                    keys[0] = cmaps[m, 0]
                elif kind == 2:
                    if ok0:
                        keys[ns] = p0
                        sums[ns] = cw0
                        ns += 1
                    if ok1:
                        keys[ns] = p1
                        sums[ns] = cw1
                        ns += 1
                    if ok2:
                        keys[ns] = p2
                        sums[ns] = cw2
                        ns += 1
                    if ok3:
                        keys[ns] = p3
                        sums[ns] = cw3
                        ns += 1
                else:
                    for corner in range(4):
                        if corner == 0:
                            ok, pc, cw = ok0, p0, cw0
                        elif corner == 1:
                            ok, pc, cw = ok1, p1, cw1
                        elif corner == 2:
                            ok, pc, cw = ok2, p2, cw2
                        else:
                            ok, pc, cw = ok3, p3, cw3
                        if not ok:
                            continue
                        kk = cmaps[m, pc]
                        if kk < 0:
                            continue
                        found = -1
                        for t in range(ns):
                            if keys[t] == kk:
                                found = t
                                break
                        if found >= 0:
                            sums[found] = np.float32(sums[found] + cw)
                        else:
                            keys[ns] = kk
                            sums[ns] = cw
                            ns += 1
                    for a in range(ns):                         # ascending chunk ids (at most four: a selection sort)
                        mn = a
                        for b in range(a + 1, ns):
                            if keys[b] < keys[mn]:
                                mn = b
                        if mn != a:
                            tk = keys[a]
                            tv = sums[a]
                            keys[a] = keys[mn]
                            sums[a] = sums[mn]
                            keys[mn] = tk
                            sums[mn] = tv
                nbm = nb[m]
                b0 = base[f, m]
                for t in range(ns):
                    cv = sums[t]
                    if cv == 0:
                        continue
                    fw = np.float32(wi * cv)
                    if nbm == 0:
                        if fw != 0 or not bz:
                            c = b0 + keys[t]
                            if use_cmap:
                                c = col_map[c]
                            v = fw
                            if use_scale:
                                v = np.float32(v * scale[c])
                            oc[cnt] = c
                            ov[cnt] = v
                            cnt += 1
                    else:
                        o = nboff[m]
                        for q in range(nbm):
                            ph = phi[o + q, k]
                            if ph == 0:
                                continue
                            v = np.float32(fw * ph)
                            if v != 0 or not bz:
                                c = b0 + keys[t] * nbm + q
                                if use_cmap:
                                    c = col_map[c]
                                if use_scale:
                                    v = np.float32(v * scale[c])
                                oc[cnt] = c
                                ov[cnt] = v
                                cnt += 1
        if mode == 0:
            acc = zero
            for t in range(cnt):
                acc = acc + ov[t] * vec[oc[t]]
            out[k] = acc
        elif mode == 1:
            yv = vec[k]
            for t in range(cnt):
                out[oc[t]] += ov[t] * yv
        elif mode == 2:
            a = k * S
            for t in range(cnt):
                cols_s[a + t] = oc[t]
                vals_s[a + t] = np.float32(0.0) if floored else ov[t]
            counts[k] = cnt
        else:
            hit = 0
            for t in range(cnt):
                c = oc[t]
                if c < L and drop[c]:
                    hit = 1
                    break
            out[k] = hit


@_jit
def _compact(counts, S, cols_s, vals_s, indptr, cols, vals):
    """The padded rows of :func:`_expand` packed as CSR arrays (``indptr`` from ``counts``, written here)."""
    n = counts.shape[0]
    indptr[0] = 0
    for k in range(n):
        indptr[k + 1] = indptr[k] + counts[k]
    for k in range(n):
        a = indptr[k]
        o = k * S
        for s in range(counts[k]):
            cols[a + s] = cols_s[o + s]
            vals[a + s] = vals_s[o + s]


@_jit
def _fold_sums(indptr, cols, vals, b64, boundary, AtA, Atb):
    """The start vector's column sums over one row chunk, in row order: ``AtA[c] += float64(float32(v * v))`` and
    ``Atb[c] += float64(v) * b[i]`` for every entry of a column ``c >= boundary``
    (:func:`selfcal.core.solution._column_sums`'s arithmetic)."""
    for i in range(indptr.shape[0] - 1):
        bi = b64[i]
        for e in range(indptr[i], indptr[i + 1]):
            c = cols[e] - boundary
            if c >= 0:
                v = vals[e]
                AtA[c] += np.float64(np.float32(v * v))
                Atb[c] += np.float64(v) * bi


@_jit
def _split_spans(indptr, cols, vals, cuts, out_ip, out_cols, out_vals, offsets):
    """The rows' entries split by the column ranges ``cuts[s]:cuts[s + 1]`` (each row's entries ascend in column):
    range ``s`` is the CSR ``out_ip[s]``, ``out_cols[offsets[s]:]``, ``out_vals[offsets[s]:]``; ``offsets`` are its
    entry counts on entry and its starts on return."""
    nr = indptr.shape[0] - 1
    ns = cuts.shape[0] - 1
    total = 0
    for s in range(ns):
        c = offsets[s]
        offsets[s] = total
        total += c
    for s in range(ns):
        out_ip[s, 0] = 0
    pos = np.empty(ns, dtype=np.int64)
    for s in range(ns):
        pos[s] = offsets[s]
    for i in range(nr):
        s = 0
        for e in range(indptr[i], indptr[i + 1]):
            c = cols[e]
            while c >= cuts[s + 1]:
                s += 1
            out_cols[pos[s]] = c
            out_vals[pos[s]] = vals[e]
            pos[s] += 1
        for t in range(ns):
            out_ip[t, i + 1] = pos[t] - offsets[t]


# ---------------------------------------------------------------------------------------------------- the table
class _Block:
    """The data rows of one assembly batch: per observation its reference pixel ``sky`` (int32), detector position
    ``x``, ``y`` (the frames' coordinate dtype) and weight ``w`` (float32); the frames (``frames``, their index in the
    solve), their row ranges (``frame_rows``, length ``nf + 1``) and their column base per offset map (``base``,
    ``(nf, K)``); the rows whose right-hand side is zero (``zb``, ascending) and those the coverage floor zeroed
    (``fl``, ascending); the entry count ``nnz``."""

    __slots__ = ('frames', 'frame_rows', 'sky', 'x', 'y', 'w', 'base', 'zb', 'fl', 'nnz')

    def __init__(self, frames, frame_rows, sky, x, y, w, zb, nnz):
        self.frames = np.asarray(frames, dtype=np.int64)
        self.frame_rows = np.asarray(frame_rows, dtype=np.int64)
        self.sky = np.ascontiguousarray(sky, dtype=np.int32)
        self.x = np.ascontiguousarray(x)
        self.y = np.ascontiguousarray(y)
        self.w = np.ascontiguousarray(w, dtype=np.float32)
        self.base = None                       # set once the layout's column map is known
        self.zb = np.asarray(zb, dtype=np.int64)
        self.fl = np.zeros(0, dtype=np.int64)
        self.nnz = int(nnz)

    @property
    def n(self):
        return int(self.sky.shape[0])

    @property
    def nbytes(self):
        return self.sky.nbytes + self.x.nbytes + self.y.nbytes + self.w.nbytes


def _map_kind(cm):
    """What a chunk map's form spares the loop (:func:`_pass`): 1 when one chunk covers the whole detector (no
    pixel outside it), 2 when every pixel is its own chunk, 0 otherwise."""
    cm = np.asarray(cm)
    if cm.size and cm.min() == cm.max() and cm.min() >= 0:
        return 1
    if cm.size and np.array_equal(cm, np.arange(cm.size, dtype=cm.dtype)):
        return 2
    return 0


class _Expansion:
    """The model's detector side, as the rows are made from it: the sky terms and their coefficients, the chunk
    maps and the offset bases, the detector maps the functions read and the other variable sources, the column
    layout. ``values(block, lo, hi)`` evaluates the model's functions over a chunk of a block's observations, as
    the build evaluates them on a subframe's, and returns the sky entries' values and the basis values the compiled
    loops take."""

    def __init__(self, *, ref_shape, sky_components, sky_entries, chunk_maps, det_shape, bases, aux_keys, aux_maps,
                 frame_values, sky_variables, derived, col_bases, num_chunks_list, frame_to_group_list, col_map):
        self.ref_w = int(ref_shape[1])
        self.num_sky = int(ref_shape[0]) * int(ref_shape[1])
        self.components = list(sky_components) if sky_entries else []
        self.J = len(self.components)
        self.identity = [c.coefficient is None for c in self.components]
        self.H, self.W = int(det_shape[0]), int(det_shape[1])
        K = len(chunk_maps)
        self.K = K
        self.cmaps = (np.stack([np.asarray(cm).ravel() for cm in chunk_maps]).astype(np.int32)
                      if K else np.zeros((0, self.H * self.W), dtype=np.int32))
        self.kinds = np.array([_map_kind(cm) for cm in self.cmaps], dtype=np.int64)
        self.bases = list(bases)
        self.nb = np.array([0 if bm is None or int(bm.n) == 0 else int(bm.n) for bm in self.bases], dtype=np.int64)
        self.nboff = np.zeros(K + 1, dtype=np.int64)
        np.cumsum(self.nb, out=self.nboff[1:])
        self.aux_keys = list(aux_keys)
        self.aux_maps = (np.stack([np.asarray(a, dtype=np.float32).ravel() for a in aux_maps])
                         if len(aux_maps) else np.zeros((0, self.H * self.W), dtype=np.float32))
        self.frame_values = {k: np.asarray(v) for k, v in (frame_values or {}).items()}
        self.sky_variables = {k: np.ascontiguousarray(v, dtype=np.float32) for k, v in (sky_variables or {}).items()}
        self.derived = dict(derived or {})
        self.col_bases = [int(c) for c in col_bases]
        self.num_chunks_list = [int(c) for c in num_chunks_list]
        self.frame_to_group_list = [np.asarray(g) for g in frame_to_group_list]
        self.col_map = None if col_map is None else np.ascontiguousarray(col_map, dtype=np.int32)
        #: The most entries a row can hold: the sky terms, and per map four corners times its functions.
        self.S = int(self.J + sum(4 * max(1, int(n)) for n in self.nb))
        #: Every data variable the model's functions read.
        self.wanted = set()
        for c in self.components:
            if c.coefficient is not None:
                self.wanted.update(c.coefficient.variables)
        for bm in self.bases:
            if bm is not None:
                self.wanted.update(bm.variables)
        #: Whether the observations' reference position is built for the functions: a sky map, a derived variable
        #: (its inputs are its own) or ``sky_x`` / ``sky_y`` among the variables read.
        self.needs_position = bool(self.sky_variables) or bool(self.derived) or bool(self.wanted & {'sky_x', 'sky_y'})

    def column_bases(self, frames):
        """``(nf, K)``: the column base of each frame's chunks per map (``col_bases[m] + group * chunks``)."""
        frames = np.asarray(frames, dtype=np.int64)
        out = np.zeros((frames.size, self.K), dtype=np.int64)
        for m in range(self.K):
            out[:, m] = self.col_bases[m] + self.frame_to_group_list[m][frames] * self.num_chunks_list[m]
        return out

    def values(self, block, lo, hi):
        """The sky entries' values ``(J, n)`` (``w * c_j``, float32: the build's ``S_data``) and the bases' values
        ``(sum nb, n)`` (float32, one function per contiguous row, each cast once from the function's values) of the
        block's rows ``lo:hi``, from the model's own functions on the rows' observation variables."""
        n = hi - lo
        x, y, w = block.x[lo:hi], block.y[lo:hi], block.w[lo:hi]
        det = {'det_x': x, 'det_y': y}
        if self.aux_maps.shape[0]:
            aux = np.empty((self.aux_maps.shape[0], n), dtype=np.float32)
            _sample_maps(x, y, self.aux_maps, self.H, self.W, aux)
            det.update(zip(self.aux_keys, aux))
        if self.frame_values or 'frame' in self.wanted:
            f0 = int(np.searchsorted(block.frame_rows, lo, side='right') - 1)
            f1 = int(np.searchsorted(block.frame_rows, hi - 1, side='right'))
            bounds = np.clip(block.frame_rows[f0:f1 + 1], lo, hi)
            frame_of = np.repeat(block.frames[f0:f1], np.diff(bounds))
            det['frame'] = frame_of.astype(np.float64)
            for name, vals in self.frame_values.items():
                v = vals[frame_of]
                det[name] = v.astype(np.float64) if v.dtype.kind in 'biuf' else v
        if self.needs_position:
            P = block.sky[lo:hi].astype(np.int64)
            obs = ObservationVariables((P // self.ref_w, P % self.ref_w), ref_coords=(0, 0, 0, 0), detector=det,
                                       sky=self.sky_variables, derived=self.derived)
        else:                       # no variable reads the reference position: the stand-in only sets the length
            obs = ObservationVariables((block.sky[lo:hi], block.sky[lo:hi]), ref_coords=None, detector=det,
                                       sky={}, derived=self.derived)
        sky_vals = np.empty((self.J, n), dtype=np.float32)
        for j, comp in enumerate(self.components):
            if self.identity[j]:
                sky_vals[j] = w
            else:
                sky_vals[j] = w * comp.coefficients(obs)             # the dtype of the product, then float32
        if self.nboff[-1]:
            phi = np.empty((int(self.nboff[-1]), n), dtype=np.float32)
            for m, bm in enumerate(self.bases):
                if self.nb[m]:
                    bm.evaluate(obs, rows=True, out=phi[self.nboff[m]:self.nboff[m + 1]])
        else:
            phi = np.zeros((0, n), dtype=np.float32)
        return sky_vals, phi


class ObservationTable:
    """The matrix of a solve as its observations (module docstring): ``blocks`` (:class:`_Block`, the data rows in
    row order), ``tail`` (the constraint rows, a csr matrix in solver columns, or None), ``shape`` and ``dtype``
    (float32, the entries') as a stored matrix has them, ``scale`` (the column scale of every entry, None until the
    preconditioner sets it) and ``threads`` (the threads of its passes). The expansion (``model``) holds the model's
    detector side and the column map of a compacted system."""

    def __init__(self, blocks, tail, n_cols, model, threads=1):
        require_numba()
        self.blocks = list(blocks)
        self.tail = tail
        self.model = model
        self.scale = None
        self.threads = max(1, int(threads))
        if not self.blocks:
            raise ValueError("an observation table needs at least one data block")
        dt = {b.x.dtype for b in self.blocks} | {b.y.dtype for b in self.blocks}
        if len(dt) != 1:
            raise ValueError(f"the frames' detector coordinates come in several dtypes ({sorted(map(str, dt))}); "
                             "an observation table needs one")
        sizes = [b.n for b in self.blocks] + ([int(tail.shape[0])] if tail is not None else [])
        self.row_bounds = np.zeros(len(sizes) + 1, dtype=np.int64)
        np.cumsum(sizes, out=self.row_bounds[1:])
        self.shape = (int(self.row_bounds[-1]), int(n_cols))
        self._scratch = {}
        self._lock = threading.Lock()

    # ---- what a stored matrix tells ---------------------------------------------------------------------------
    dtype = np.dtype(np.float32)

    @property
    def nblocks(self):
        """The data blocks (the tail, when present, is the block after them: ``row_bounds`` has ``nblocks + 2``
        boundaries then)."""
        return len(self.blocks)

    @property
    def nnz(self):
        """The entries of the matrix the table stands for."""
        return sum(b.nnz for b in self.blocks) + (0 if self.tail is None else int(self.tail.nnz))

    @property
    def nbytes(self):
        """The bytes of the table (the observations; the tail's stored entries aside)."""
        return sum(b.nbytes for b in self.blocks)

    @property
    def n_data_rows(self):
        return int(self.row_bounds[len(self.blocks)])

    def __repr__(self):
        return (f"<ObservationTable shape={self.shape} observations={self.n_data_rows} nnz={self.nnz} "
                f"blocks={len(self.blocks)} tail={0 if self.tail is None else self.tail.shape[0]} rows>")

    # ---- the rows ---------------------------------------------------------------------------------------------
    def _args(self, bi, lo, hi, scaled=True):
        """The compiled loops' arguments for rows ``lo:hi`` of block ``bi`` (local rows): the block's arrays, the
        model's values over those rows and the layout."""
        blk = self.blocks[bi]
        md = self.model
        sky_vals, phi = md.values(blk, lo, hi)
        use_cmap = md.col_map is not None
        use_scale = scaled and self.scale is not None
        return (lo, hi, blk.sky, blk.x, blk.y, blk.w, blk.frame_rows, blk.base, blk.zb, blk.fl, md.J, md.num_sky,
                sky_vals, md.K, md.cmaps, md.kinds, md.H, md.W, md.nb, md.nboff, phi,
                use_cmap, md.col_map if use_cmap else np.zeros(0, dtype=np.int32),
                use_scale, self.scale if use_scale else np.zeros(0, dtype=np.float32), md.S)

    def _thread_scratch(self, n):
        """This thread's padded scratch for ``n`` rows (grown as needed)."""
        key = threading.get_ident()
        with self._lock:
            sc = self._scratch.get(key)
        need = n * self.model.S
        if sc is None or sc[0].shape[0] < need:
            sc = (np.empty(max(need, 1), dtype=np.int32), np.empty(max(need, 1), dtype=np.float32),
                  np.empty(max(n, 1), dtype=np.int64))
            with self._lock:
                self._scratch[key] = sc
        return sc

    def matvec_rows(self, bi, lo, hi, p, out):
        """``out[:] = A[lo:hi] @ p`` for the global rows ``lo:hi`` of block ``bi`` (the tail included), with
        scipy's arithmetic for the entries' and ``p``'s dtypes (float32 entries times a float64 vector are summed in
        float64), the sums stored in ``out``'s dtype."""
        base = int(self.row_bounds[bi])
        if bi == len(self.blocks):
            lr0, lr1 = lo - base, hi - base
            ip = self.tail.indptr[lr0:lr1 + 1]
            s0, s1 = int(ip[0]), int(ip[-1])
            out[...] = _csr_shell(self.tail.data[s0:s1], self.tail.indices[s0:s1], ip - np.int32(s0),
                                  (hi - lo, self.shape[1])) @ p
            return
        zero = np.zeros((), dtype=np.promote_types(self.dtype, p.dtype))[()]
        for a, b in self._chunks(lo - base, hi - base):
            _pass(0, *self._args(bi, a, b), p, zero, out[a - (lo - base):b - (lo - base)], _NO_U8, _NO_I64,
                  _NO_I32, _NO_F32)

    def rmatvec_rows(self, bi, lo, hi, y_seg, buf):
        """``buf[c] += A[lo:hi, c]^T y_seg`` for the global rows ``lo:hi`` of block ``bi`` (the tail included), the
        rows in order."""
        base = int(self.row_bounds[bi])
        if bi == len(self.blocks):
            lr0, lr1 = lo - base, hi - base
            d = self.tail.data
            if d.dtype != buf.dtype:
                d = d.astype(buf.dtype)
            _sparsetools.csc_matvec(self.shape[1], hi - lo, self.tail.indptr[lr0:lr1 + 1], self.tail.indices, d,
                                    y_seg, buf)
            return
        for a, b in self._chunks(lo - base, hi - base):
            _pass(1, *self._args(bi, a, b), y_seg[a - (lo - base):b - (lo - base)], _ZERO64, buf, _NO_U8, _NO_I64,
                  _NO_I32, _NO_F32)

    @staticmethod
    def _chunks(lo, hi, rows=_EVAL_ROWS):
        """The row chunks ``(a, b)`` of ``lo:hi``, ``rows`` each."""
        return [(a, min(hi, a + rows)) for a in range(lo, hi, rows)]

    def expand(self, bi, lo, hi, scaled=True, buf=None):
        """The rows ``lo:hi`` of block ``bi`` (local rows; the tail is block ``nblocks``) as CSR arrays ``(indptr,
        cols, vals)``: int32 local indptr, int32 solver columns, float32 values (scaled by ``scale`` when set and
        ``scaled``). With ``buf`` (a holder with ``get(name, size, dtype)``) the arrays are written into its buffers,
        else fresh ones."""
        n = hi - lo
        if bi == len(self.blocks):
            ip = self.tail.indptr[lo:hi + 1]
            s0, s1 = int(ip[0]), int(ip[-1])
            lip = (ip - np.int32(s0)).astype(np.int32)
            cols, vals = self.tail.indices[s0:s1], self.tail.data[s0:s1]
            if buf is None:
                return lip, cols.copy(), vals.copy()
            li = buf.get('lip', n + 1, np.int32)
            np.copyto(li, lip)
            co = buf.get('cols', cols.size, np.int32)
            np.copyto(co, cols)
            va = buf.get('vals', vals.size, np.float32)
            np.copyto(va, vals)
            return li, co, va
        S = self.model.S
        parts, total = [], 0
        for a, b in self._chunks(lo, hi):
            m = b - a
            cols_s, vals_s, counts = self._thread_scratch(m)
            _pass(2, *self._args(bi, a, b, scaled=scaled), _NO_F64, _ZERO64, _NO_F64, _NO_U8, counts[:m], cols_s,
                  vals_s)
            k = int(counts[:m].sum())
            ip, co, va = np.empty(m + 1, dtype=np.int32), np.empty(k, dtype=np.int32), np.empty(k, dtype=np.float32)
            _compact(counts[:m], S, cols_s, vals_s, ip, co, va)
            parts.append((ip, co, va))
            total += k
        if len(parts) == 1 and buf is None:
            return parts[0]
        if buf is None:
            indptr, cols, vals = (np.empty(n + 1, dtype=np.int32), np.empty(total, dtype=np.int32),
                                  np.empty(total, dtype=np.float32))
        else:
            indptr, cols, vals = (buf.get('lip', n + 1, np.int32), buf.get('cols', total, np.int32),
                                  buf.get('vals', total, np.float32))
        indptr[0] = 0
        o, r = 0, 0
        for ip, co, va in parts:
            m, k = ip.shape[0] - 1, co.shape[0]
            np.add(ip[1:], np.int32(o), out=indptr[r + 1:r + m + 1])
            cols[o:o + k] = co
            vals[o:o + k] = va
            o += k
            r += m
        return indptr, cols, vals

    # ---- the deflation's and the start vector's reads ----------------------------------------------------------
    def piece_index(self, rows):
        """The row chunks as descriptors ``(block, lr0, lr1)`` in row order, at most ``rows`` per chunk of each
        block; the tail is the last block. The blocks are the assembly batches, as the partitioned storage's are, so
        a reader that sums per chunk (the deflation) sums the same chunks."""
        rows = max(1, int(rows))
        sizes = [b.n for b in self.blocks] + ([int(self.tail.shape[0])] if self.tail is not None else [])
        return [(bi, lr0, min(nrows, lr0 + rows)) for bi, nrows in enumerate(sizes) for lr0 in range(0, nrows, rows)]

    def piece(self, desc, buf=None, spans=None):
        """One row chunk as :func:`selfcal.core.deflation._piece` gives it: ``(r0, r1, [(indptr, columns,
        values), ...])``, one triple per span ``(lo, hi, 0)`` (the entries of the columns ``lo:hi``, local indptr
        from 0), int32 with ``buf``, int64 without."""
        bi, lr0, lr1 = desc
        r0 = int(self.row_bounds[bi]) + lr0
        nr = lr1 - lr0
        indptr, cols, vals = self.expand(bi, lr0, lr1, buf=buf)
        if spans is None or (len(spans) == 1 and spans[0][0] == 0 and spans[0][1] >= self.shape[1]):
            out = [(indptr, cols, vals)]
        else:
            cuts = np.array([s[0] for s in spans] + [spans[-1][1]], dtype=np.int64)
            offsets = np.zeros(len(spans), dtype=np.int64)
            span_of = np.searchsorted(cuts, cols, side='right') - 1
            np.add.at(offsets, span_of, 1)
            del span_of
            total = int(cols.size)
            if buf is None:
                ips, oc, ov = (np.empty((len(spans), nr + 1), dtype=np.int32), np.empty(total, dtype=np.int32),
                               np.empty(total, dtype=np.float32))
            else:
                ips = buf.get('span_ip', len(spans) * (nr + 1), np.int32).reshape(len(spans), nr + 1)
                oc = buf.get('span_cols', total, np.int32)
                ov = buf.get('span_vals', total, np.float32)
            _split_spans(indptr, cols, vals, cuts, ips, oc, ov, offsets)
            out = []
            for s in range(len(spans)):
                a = int(offsets[s])
                b = int(offsets[s + 1]) if s + 1 < len(spans) else total
                out.append((ips[s], oc[a:b], ov[a:b]))
        if buf is None:
            out = [(ip.astype(np.int64), co.astype(np.int64), va) for ip, co, va in out]
        return r0, r0 + nr, out

    def stream(self, scaled=True, rows=_TASK_ROWS, threads=None):
        """Every row chunk of the data blocks in row order, expanded in threads ahead of the consumer: yields
        ``(r0, r1, indptr, cols, vals)`` (global rows, fresh arrays)."""
        threads = self.threads if threads is None else max(1, int(threads))
        tasks = [(bi, lr0, lr1) for bi, lr0, lr1 in self.piece_index(rows) if bi < len(self.blocks)]

        def work(t):
            bi, lr0, lr1 = t
            r0 = int(self.row_bounds[bi]) + lr0
            return (r0, r0 + lr1 - lr0) + self.expand(bi, lr0, lr1, scaled=scaled)

        if threads <= 1:
            for t in tasks:
                yield work(t)
            return
        with ThreadPoolExecutor(max_workers=threads) as ex:
            q = deque()
            for t in tasks:
                q.append(ex.submit(work, t))
                if len(q) >= 2 * threads:
                    yield q.popleft().result()
            while q:
                yield q.popleft().result()

    def entry_chunks(self, chunk_size, scaled=True):
        """The entries in row order cut at global multiples of ``chunk_size``, each chunk as its pieces
        ``[(values, columns), ...]`` (the column-norm chunks of :func:`selfcal.core.solve.apply_lsqr`; the stored
        storages' :func:`selfcal.core.solve._iter_global_entry_chunks`)."""
        buf_d, buf_c, buf_n = [], [], 0
        sources = ((d, c) for _r0, _r1, _ip, c, d in self.stream(scaled=scaled))
        if self.tail is not None:
            sources = itertools.chain(sources, [(self.tail.data, self.tail.indices)])
        for d, c in sources:
            pos, size = 0, d.size
            while pos < size:
                take = min(size - pos, chunk_size - buf_n)
                buf_d.append(d[pos:pos + take])
                buf_c.append(c[pos:pos + take])
                buf_n += take
                pos += take
                if buf_n == chunk_size:
                    yield list(zip(buf_d, buf_c))
                    buf_d, buf_c, buf_n = [], [], 0
        if buf_n:
            yield list(zip(buf_d, buf_c))

    def column_sums(self, b, boundary):
        """``sum_i A[i, j]^2`` and ``sum_i A[i, j] b[i]`` of every column ``j >= boundary``, with the arithmetic of
        :func:`selfcal.core.solution._column_sums` (each column's entries folded in row order from 0.0, the square
        in float32 then float64, the product in float64)."""
        n = int(self.shape[1])
        AtA = np.zeros(n - boundary)
        Atb = np.zeros(n - boundary)
        for r0, r1, ip, cols, vals in self.stream(scaled=False):
            _fold_sums(ip, cols, vals, np.asarray(b[r0:r1], dtype=np.float64), boundary, AtA, Atb)
        if self.tail is not None:
            r0 = self.n_data_rows
            _fold_sums(self.tail.indptr, self.tail.indices, self.tail.data,
                       np.asarray(b[r0:r0 + self.tail.shape[0]], dtype=np.float64), boundary, AtA, Atb)
        return AtA, Atb

    def column_squares(self, chunk=64_000_000):
        """The sum of the squared entries of each column, float64 (:func:`selfcal.core.solve._column_squares` on a
        single stored matrix: chunks of ``chunk`` entries, each a float64 bincount of the float64 squares)."""
        sq = np.zeros(int(self.shape[1]), dtype=np.float64)
        for pieces in self.entry_chunks(chunk):
            d = np.concatenate([p[0] for p in pieces]) if len(pieces) > 1 else pieces[0][0]
            i = np.concatenate([p[1] for p in pieces]) if len(pieces) > 1 else pieces[0][1]
            v = d.astype(np.float64)
            sq += np.bincount(i, weights=v * v, minlength=sq.size)[:sq.size]
        return sq

    # ---- the coverage floor -----------------------------------------------------------------------------------
    def floor(self, drop_cols, threads=None):
        """Zero the rows with an entry in a column of ``drop_cols`` (a boolean over the leading columns of the full
        layout): they keep their entries with the value +0.0 from here on. Returns their global row ids."""
        threads = self.threads if threads is None else max(1, int(threads))
        drop = np.ascontiguousarray(drop_cols, dtype=np.bool_)
        md = self.model
        out = []

        def work(t):
            bi, lo, hi = t
            blk = self.blocks[bi]
            flags = np.zeros(hi - lo, dtype=np.uint8)
            for a, b in self._chunks(lo, hi):
                sky_vals, phi = md.values(blk, a, b)
                _pass(3, a, b, blk.sky, blk.x, blk.y, blk.w, blk.frame_rows, blk.base, blk.zb, _NO_I64, md.J,
                      md.num_sky, sky_vals, md.K, md.cmaps, md.kinds, md.H, md.W, md.nb, md.nboff, phi, False,
                      _NO_I32, False, _NO_F32, md.S, _NO_F64, _ZERO64, flags[a - lo:b - lo], drop, _NO_I64, _NO_I32,
                      _NO_F32)
            return bi, lo + np.flatnonzero(flags)

        tasks = [t for t in self.piece_index(_TASK_ROWS) if t[0] < len(self.blocks)]
        if threads <= 1:
            results = [work(t) for t in tasks]
        else:
            with ThreadPoolExecutor(max_workers=threads) as ex:
                results = list(ex.map(work, tasks))
        per_block = {}
        for bi, rows in results:
            per_block.setdefault(bi, []).append(rows)
        for bi, parts in per_block.items():
            rows = np.concatenate(parts)
            self.blocks[bi].fl = np.sort(rows).astype(np.int64)
            out.append(rows + int(self.row_bounds[bi]))
        return np.sort(np.concatenate(out)) if out else np.zeros(0, dtype=np.int64)

    # ---- for tests and checks ---------------------------------------------------------------------------------
    def to_csr(self, scaled=True):
        """The whole matrix as one csr matrix (rows in order; the entries as the solve reads them)."""
        parts = []
        for bi in range(len(self.blocks)):
            ip, cols, vals = self.expand(bi, 0, self.blocks[bi].n, scaled=scaled)
            parts.append(csr_matrix((vals, cols, ip), shape=(self.blocks[bi].n, self.shape[1])))
        if self.tail is not None:
            parts.append(self.tail)
        from scipy.sparse import vstack
        out = vstack(parts, format='csr')
        out.has_sorted_indices = True
        return out


def warm_up():
    """Compile the loops once, on a one-row table, so the solve's threads do not all wait on the compilation."""
    require_numba()
    HW = 4
    P, x, y, w = np.zeros(1, np.int32), np.zeros(1, np.float32), np.zeros(1, np.float32), np.ones(1, np.float32)
    frame_rows, base = np.array([0, 1], np.int64), np.zeros((1, 1), np.int64)
    zb = fl = np.zeros(0, np.int64)
    sky_vals, phi = np.ones((1, 1), np.float32), np.ones((1, 1), np.float32)
    cmaps = np.zeros((1, HW), np.int32)
    nb, nboff = np.array([1], np.int64), np.array([0, 1], np.int64)
    no_map, no_scale = np.zeros(0, np.int32), np.zeros(0, np.float32)
    S = 5
    kinds = np.zeros(1, np.int64)
    args = (0, 1, P, x, y, w, frame_rows, base, zb, fl, 1, 4, sky_vals, 1, cmaps, kinds, 2, 2, nb, nboff, phi,
            False, no_map, False, no_scale, S)
    for pdt in (np.float32, np.float64):
        for odt in (np.float32, np.float64):
            _pass(0, *args, np.ones(S, pdt), np.zeros((), dtype=np.promote_types(np.float32, pdt))[()],
                  np.zeros(1, odt), _NO_U8, _NO_I64, _NO_I32, _NO_F32)
        _pass(1, *args, np.ones(1, pdt), _ZERO64, np.zeros(S, pdt), _NO_U8, _NO_I64, _NO_I32, _NO_F32)
    counts = np.zeros(1, np.int64)
    cols_s, vals_s = np.zeros(S, np.int32), np.zeros(S, np.float32)
    _pass(2, *args, _NO_F64, _ZERO64, _NO_F64, _NO_U8, counts, cols_s, vals_s)
    ip, co, va = np.zeros(2, np.int32), np.zeros(S, np.int32), np.zeros(S, np.float32)
    _compact(counts, S, cols_s, vals_s, ip, co, va)
    _pass(3, *args, _NO_F64, _ZERO64, np.zeros(1, np.uint8), np.zeros(1, np.bool_), _NO_I64, _NO_I32, _NO_F32)
    _fold_sums(ip, co, va, np.zeros(1), 0, np.zeros(S), np.zeros(S))
    _sample_maps(x, y, np.zeros((1, HW), np.float32), 2, 2, np.zeros((1, 1), np.float32))
    _split_spans(ip, co, va, np.array([0, 2, S], np.int64), np.zeros((2, 2), np.int32), np.zeros(S, np.int32),
                 np.zeros(S, np.float32), np.array([0, 0], np.int64))
