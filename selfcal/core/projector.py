"""Hard constraints on solver columns by reparametrisation.

An offset term may be required to carry no component along given directions of its own columns
(no polynomial of some degree along a chunk axis, say: :class:`~selfcal.models.model.Offsets`
``exclude``). The solver then works with ``A P`` instead of ``A``, ``P`` the orthogonal projector
onto the allowed subspace: every product applies ``P`` to the unknowns first (``A (P y)``) and after
the transpose (``P (A^T r)``). The solution ``P y`` satisfies the constraint exactly, whatever the
data say, with no constraint row and no weight; the removed directions are exactly null in ``A P``
and a solver started in the subspace never leaves it. ``P`` is a block-diagonal rank update: one
small orthonormal basis ``Q`` per group of columns (``P = I - Q Q^T`` on the group), so a product
costs nothing against the matrix itself.

The constraint is stated on the physical unknowns ``x``; the solver's are the column-scaled and
compacted ones, ``x = M y`` on the active columns. :meth:`ColumnProjector.build` turns the stated
directions ``P_x`` of a group into the solver's: ``Q = orth(M_g P_x)`` over the group's active columns
(an inactive column, compacted away, has no unknown and drops out of the basis; so does a column without data,
which a coverage floor keeps in the layout with zeroed entries: in the constraint it would take the value that
lifts the constraint off the group's other columns). A group measured with the columns' information (the
exclusion's third element ``'information'``) states ``P_x^T F x = 0``, ``F`` the columns' information (the sum of
their squared entries): in the solver's unknowns ``Q = orth(F_g M_g P_x)``, so a column the data barely see weighs
little in the constraint and cannot carry it for the others.
"""
from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import LinearOperator

__all__ = ['ColumnProjector']

#: A direction whose singular value is below this fraction of the group's largest is dropped (it
#: is a combination the active columns cannot express, e.g. a line with fewer active columns than
#: polynomials).
_DROP = 1e-10


class ColumnProjector:
    """``P = I - sum_g Q_g Q_g^T``: ``groups`` is a list of ``(cols, Q)`` with ``cols`` an int array
    ``(G, n)`` of column indices (``G`` groups of ``n`` columns sharing one shape; ``-1`` marks a
    column that does not exist, whose row of ``Q`` is zero) and ``Q`` a float64 array ``(G, n, r)``
    with orthonormal columns per group. The groups' columns must be disjoint."""

    def __init__(self, groups):
        self.groups = []
        self._spans = []           # per group: (lo, hi, Q2) when its real columns are one contiguous range (a sky term), else None
        for cols, Q in groups:
            cols = np.asarray(cols, dtype=np.int64)
            Q = np.asarray(Q, dtype=np.float64)
            if cols.ndim != 2 or Q.shape[:2] != cols.shape:
                raise ValueError(f"ColumnProjector: cols {cols.shape} and Q {Q.shape} disagree")
            if Q.shape[2]:
                self.groups.append((cols, Q))
                self._spans.append(self._span(cols, Q))

    @staticmethod
    def _span(cols, Q):
        """``(lo, hi, Q2)`` when the group is one line whose real columns are exactly ``lo:hi`` in order (the
        rows of ``Q`` at them), so that applying is a slice and two small products; None otherwise."""
        if cols.shape[0] != 1:
            return None
        flat = cols[0]
        real = flat >= 0
        r = flat[real]
        if r.size == 0 or not np.array_equal(r, np.arange(r[0], r[0] + r.size)):
            return None
        return int(r[0]), int(r[0] + r.size), np.ascontiguousarray(Q[0][real])

    # ---------------------------------------------------------------------------------------- building
    @classmethod
    def build(cls, exclusions, active_mask, scale, n_cols, no_data=None, information=None) -> 'ColumnProjector':
        """From ``exclusions``, a list of ``(cols, P)`` or ``(cols, P, 'information')`` in the full column layout
        (``cols`` ``(G, n)``, ``P`` ``(n, r)`` the excluded directions, the same for every group), the projector in
        the solver's coordinates: ``active_mask`` (None: no compaction) maps full columns to the compact ones,
        ``scale`` (None: unscaled) is the solver's column scale ``M`` (``x = M y``), ``n_cols`` the
        solver's column count, ``no_data`` (None: none) the solver columns without data, which drop out as
        inactive ones do, ``information`` the solver columns' information (their squared norms, unscaled),
        needed by an exclusion measured with it."""
        if active_mask is not None:
            cmap = np.full(len(active_mask), -1, dtype=np.int64)
            cmap[np.asarray(active_mask, dtype=bool)] = np.arange(int(np.sum(active_mask)))
        else:
            cmap = None
        groups = []
        for ex in exclusions:
            cols, P = np.asarray(ex[0], dtype=np.int64), np.asarray(ex[1], dtype=np.float64)
            weights = ex[2] if len(ex) > 2 else None
            if weights not in (None, 'information'):
                raise ValueError(f"ColumnProjector.build: an exclusion's weights are None or 'information', not {weights!r}")
            if weights is not None and information is None:
                raise ValueError("ColumnProjector.build: an exclusion measured with the columns' information needs it")
            if cols.ndim == 1:
                cols = cols[None, :]
            ci = cols if cmap is None else np.where(cols >= 0, cmap[np.maximum(cols, 0)], -1)
            if no_data is not None and len(no_data):
                ci = np.where(np.isin(ci, no_data), -1, ci)    # a column without data drops out as an inactive one
            if (ci >= n_cols).any():
                raise ValueError("ColumnProjector.build: a column index beyond the solver's columns")
            m = np.ones(ci.shape) if scale is None else np.where(ci >= 0, np.asarray(scale, dtype=np.float64)[np.maximum(ci, 0)], 0.0)
            if weights is not None:                           # P_x^T F x = 0: the directions F M P_x in the solver's unknowns
                m = m * np.asarray(information)[np.maximum(ci, 0)].astype(np.float64)
            m[ci < 0] = 0.0                                   # an inactive column has no unknown
            A = m[:, :, None] * P[None, :, :]                 # (G, n, r): the directions in solver units
            U, s, _ = np.linalg.svd(A, full_matrices=False)
            keep = s > _DROP * np.maximum(s.max(axis=1, keepdims=True), 1e-300)
            U = U * keep[:, None, :]
            groups.append((ci, U))
        return cls(groups)

    # ---------------------------------------------------------------------------------------- applying
    @property
    def columns(self) -> np.ndarray:
        """Every column the projector acts on (sorted, unique, the real ones)."""
        if not self.groups:
            return np.zeros(0, dtype=np.int64)
        c = np.concatenate([g[0].ravel() for g in self.groups])
        return np.unique(c[c >= 0])

    def apply(self, y):
        """``y <- P y`` in place (``y`` 1-D over the columns, or 2-D with the columns along axis 0);
        returns ``y``. The arithmetic is float64 whatever ``y``'s dtype."""
        y = np.asarray(y)
        two = y.ndim == 2
        for (cols, Q), span in zip(self.groups, self._spans):
            if span is not None:
                lo, hi, Q2 = span
                v = y[lo:hi]
                if v.dtype == np.float64:
                    v -= Q2 @ (Q2.T @ v)
                else:
                    v64 = np.asarray(v, dtype=np.float64)
                    v64 -= Q2 @ (Q2.T @ v64)
                    v[...] = v64.astype(y.dtype, copy=False)
                continue
            flat = cols.ravel()
            real = flat >= 0
            idx = np.where(real, flat, 0)
            v = np.asarray(y[idx], dtype=np.float64)
            if not two:
                v = v[:, None]
            v[~real] = 0.0
            v = v.reshape(cols.shape[0], cols.shape[1], -1)
            v -= Q @ (np.swapaxes(Q, 1, 2) @ v)
            v = v.reshape(flat.size, -1)
            y[flat[real]] = (v[real] if two else v[real, 0]).astype(y.dtype, copy=False)
        return y

    def apply_symmetric(self, S):
        """``S <- P S P`` in place for a dense square ``S`` over the columns; returns ``S``."""
        self.apply(S)
        self.apply(S.T)
        return S

    def wrap(self, op) -> LinearOperator:
        """The operator ``A P`` (matvec: project, then ``op``; rmatvec: ``op``'s transpose, then project)."""
        m, n = op.shape
        dt = np.dtype(op.dtype)

        def matvec(y):
            z = np.array(y, dtype=dt, copy=True).reshape(-1)
            self.apply(z)
            return op.matvec(z)

        def rmatvec(u):
            out = np.asarray(op.rmatvec(u))
            self.apply(out)
            return out

        wrapped = LinearOperator((m, n), matvec=matvec, rmatvec=rmatvec, dtype=dt)
        inner = getattr(op, 'normal_product', None)
        if inner is not None:                      # (A P y, P A^T A P y) from one pass of the operator inside
            def normal_product(y):
                z = np.array(y, dtype=dt, copy=True).reshape(-1)
                self.apply(z)
                q, t = inner(z)
                t = np.asarray(t)
                self.apply(t)
                return q, t
            wrapped.normal_product = normal_product
        inner_t = getattr(op, 'rmatvec_promoted', None)
        if inner_t is not None:
            def rmatvec_promoted(u):
                out = np.asarray(inner_t(u))
                self.apply(out)
                return out
            wrapped.rmatvec_promoted = rmatvec_promoted
        return wrapped

    def mapped(self, index_map) -> 'ColumnProjector':
        """The same projector on columns renumbered by ``index_map`` (an int array over the current
        columns; ``-1`` for a column absent from the new numbering, which must then carry no weight)."""
        index_map = np.asarray(index_map, dtype=np.int64)
        groups = []
        for cols, Q in self.groups:
            new = np.where(cols >= 0, index_map[np.maximum(cols, 0)], -1)
            lost = (new < 0) & (cols >= 0)
            if lost.any() and np.abs(Q[lost]).max() > 0:
                raise ValueError("ColumnProjector.mapped: a constrained column is absent from the new numbering")
            groups.append((new, Q))
        return ColumnProjector(groups)

    def restricted_to(self, cols_in) -> 'ColumnProjector | None':
        """The projector on the subset ``cols_in`` (sorted int array), renumbered to positions in it:
        None when no group touches the subset; an error when a group straddles the subset's boundary."""
        pos = {int(c): i for i, c in enumerate(np.asarray(cols_in).ravel())}
        groups = []
        for cols, Q in self.groups:
            inside = np.array([[c in pos for c in row] for row in cols]) if cols.size else np.zeros(cols.shape, bool)
            inside &= cols >= 0
            rows_in = inside.any(axis=1)
            if not rows_in.any():
                continue
            if not inside[rows_in].all():
                full = inside[rows_in] | (cols[rows_in] < 0)
                if not full.all():
                    raise ValueError("ColumnProjector.restricted_to: a constrained group straddles the subset")
            new = np.full(cols[rows_in].shape, -1, dtype=np.int64)
            for i, row in enumerate(cols[rows_in]):
                for j, c in enumerate(row):
                    if c >= 0:
                        new[i, j] = pos[int(c)]
            groups.append((new, Q[rows_in]))
        return ColumnProjector(groups) if groups else None

    def __len__(self):
        return sum(int(g[0].shape[0]) for g in self.groups)

    def __repr__(self):
        return (f"ColumnProjector({len(self)} groups, {int(sum(g[1].shape[2] for g in self.groups))} directions per group, "
                f"{self.columns.size} columns)")
