"""Priors — any linear equations on a model's unknowns, written as functions.

The built-in priors of the terms (Tikhonov ``damp``, ``reg_weight`` adjacency,
``poly`` shape, ``mean_zero``) cover the common cases. Everything else is a
*prior function*: it receives one :class:`TermInfo` per term it names and
returns the rows it adds::

    def my_prior(term, **params):
        ...
        return rows, cols, vals, rhs      # Σ_u vals[r, u] * x[cols] = rhs[r], row by row

``rows`` number the prior's own rows from 0, ``cols`` are global unknown
indices (``term.index(...)`` turns a term's own indices into them), ``rhs``
has one value per row (0 = pull toward zero; a target value = pull toward it).
The config's ``weight`` multiplies every row. A prior on several terms (a
cross-term relation) names them all and receives them in order.

``TermInfo`` describes the term's unknowns: a sky term is a map (``shape =
ref_shape``), an offset term an array ``(n_groups, n_chunks, n_basis)`` (one
group per frame unless the term shares offsets), the per-frame scalar a vector
``(n_frames,)``. ``coverage`` counts the observations touching each unknown;
``frame_group`` maps every frame to its group; ``variables`` are the solve's
data variables (frame values, sky maps, ...), ``axes`` the chunk axes of an
offset term's map. Priors run after the data rows are built (coverage is known).

The functions below are ready-made priors; they are ordinary functions, no
different from one a user writes.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ['TermInfo', 'ModelPrior', 'frame_smoothness', 'sky_smoothness', 'toward_variable']


@dataclass
class TermInfo:
    """The unknowns of one model term, as a prior sees them."""
    name: str
    kind: str                      # 'sky' | 'offset' | 'scalar'
    shape: tuple
    col_base: int
    coverage: np.ndarray
    frame_group: np.ndarray | None = None
    variables: object = None       # VariableSet of the solve
    axes: object = None            # ChunkAxes of an offset term's map
    n_frames: int = 0
    extra: dict = field(default_factory=dict)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    def index(self, *idx) -> np.ndarray:
        """Global unknown indices of the term's own indices (``(y, x)`` for a sky
        term, ``(group, chunk, function)`` for an offset term, ``(frame,)`` for
        the scalar); arrays broadcast."""
        if len(idx) == 1 and len(self.shape) == 1:
            flat = np.asarray(idx[0], dtype=np.int64)
        else:
            if len(idx) != len(self.shape):
                raise ValueError(f"term {self.name!r} has unknowns of shape {self.shape}; got {len(idx)} indices")
            flat = np.ravel_multi_index(tuple(np.asarray(i, dtype=np.int64) for i in idx), self.shape)
        return self.col_base + np.asarray(flat, dtype=np.int64)

    def frame_values(self, name) -> np.ndarray:
        """A frame variable's values, one per frame."""
        fv = getattr(self.variables, 'frame', {}) if self.variables is not None else {}
        if name == 'frame':
            return np.arange(self.n_frames, dtype=np.float64)
        if name not in fv:
            raise KeyError(f"prior on {self.name!r} reads frame variable {name!r}, which the solve does not "
                           f"define (frame variables: {sorted(fv)})")
        return np.asarray(fv[name])

    def group_values(self, name) -> np.ndarray:
        """A frame variable averaged over each group's frames (one value per group)."""
        v = self.frame_values(name).astype(np.float64)
        g = np.arange(self.n_frames) if self.frame_group is None else np.asarray(self.frame_group)
        n_g = self.shape[0]
        s = np.bincount(g, weights=v, minlength=n_g)
        c = np.bincount(g, minlength=n_g)
        return s / np.maximum(c, 1)


class ModelPrior:
    """One ``[[model.prior]]`` entry, lowered: calls ``function(*terms, **params)``
    with the :class:`TermInfo` of each named term and scales its rows by
    ``weight``. ``describe(info)`` builds the TermInfos from the solve's
    :class:`~selfcal.core.constraint_builders.SystemInfo`."""

    def __init__(self, name, term_names, function, params, weight, describe):
        self.name = name
        self.term_names = tuple(term_names)
        self.function = function
        self.params = dict(params or {})
        self.weight = float(weight)
        self._describe = describe

    def __call__(self, info):
        terms = [self._describe(info, t) for t in self.term_names]
        out = self.function(*terms, **self.params)
        if out is None:
            return None
        rows, cols, vals, rhs = out
        w = self.weight
        return (rows, cols, np.asarray(vals, dtype=np.float64) * w, np.asarray(rhs, dtype=np.float64) * w)


# ---------------------------------------------------------------------------
# ready-made priors
# ---------------------------------------------------------------------------
def frame_smoothness(term, variable='frame', power=0.0, covered_only=True):
    """Offsets of groups that are neighbours in the frame variable ``variable``
    (e.g. ``time``) are pulled together: one row ``(x[g2] − x[g1]) / |Δv|^power``
    per neighbouring pair of groups (ordered by the group's mean value), per
    chunk and basis function. ``power = 0``: plain differences; ``power =
    0.5``: a random walk in ``variable``. ``covered_only``: only pairs whose
    two unknowns are both observed."""
    if term.kind != 'offset':
        raise ValueError(f"frame_smoothness applies to an offset term, not {term.kind!r} ({term.name})")
    n_g = term.shape[0]
    if n_g < 2:
        return None
    v = term.group_values(variable)
    order = np.argsort(v, kind='stable')
    g1, g2 = order[:-1], order[1:]
    dv = np.abs(v[g2] - v[g1])
    scale = np.ones_like(dv) if power == 0 else 1.0 / np.maximum(dv, 1e-12) ** power
    inner = int(np.prod(term.shape[1:]))
    cov = term.coverage.reshape(n_g, inner)
    rows_l, cols_l, vals_l = [], [], []
    r = 0
    u = np.arange(inner)
    for a, b, s in zip(g1, g2, scale):
        keep = u if not covered_only else u[(cov[a] > 0) & (cov[b] > 0)]
        if keep.size == 0:
            continue
        rr = r + np.arange(keep.size)
        rows_l += [rr, rr]
        cols_l += [term.col_base + b * inner + keep, term.col_base + a * inner + keep]
        vals_l += [np.full(keep.size, s), np.full(keep.size, -s)]
        r += keep.size
    if r == 0:
        return None
    return (np.concatenate(rows_l), np.concatenate(cols_l), np.concatenate(vals_l), np.zeros(r))


def sky_smoothness(term, covered_only=True):
    """Neighbouring pixels of a sky map are pulled together (one row per
    horizontal and vertical neighbour pair). ``covered_only``: only pairs
    whose two pixels are both observed; ``False`` also smooths into gaps."""
    if term.kind != 'sky':
        raise ValueError(f"sky_smoothness applies to a sky term, not {term.kind!r} ({term.name})")
    H, W = term.shape
    cov = term.coverage.reshape(H, W) > 0
    parts = []
    for (dy, dx) in ((0, 1), (1, 0)):
        y, x = np.mgrid[0:H - dy, 0:W - dx]
        y, x = y.ravel(), x.ravel()
        if covered_only:
            ok = cov[y, x] & cov[y + dy, x + dx]
            y, x = y[ok], x[ok]
        parts.append((term.index(y, x), term.index(y + dy, x + dx)))
    a = np.concatenate([p[0] for p in parts])
    b = np.concatenate([p[1] for p in parts])
    n = a.size
    if n == 0:
        return None
    rows = np.arange(n)
    return (np.concatenate([rows, rows]), np.concatenate([a, b]),
            np.concatenate([np.ones(n), -np.ones(n)]), np.zeros(n))


def toward_variable(term, variable, covered_only=True):
    """Pull a term's unknowns toward known values: a sky term toward the sky
    variable ``variable`` (a reference-grid map); the per-frame scalar, or an
    offset term's every chunk, toward the frame variable ``variable`` (a
    group's frames averaged). One row ``x = target`` per unknown."""
    cov = term.coverage.reshape(-1) > 0
    if term.kind == 'sky':
        sky = getattr(term.variables, 'sky', {}) if term.variables is not None else {}
        if variable not in sky:
            raise KeyError(f"toward_variable on sky term {term.name!r}: {variable!r} is not a sky variable")
        target = np.asarray(sky[variable], dtype=np.float64).reshape(-1)
    elif term.kind == 'scalar':
        target = term.frame_values(variable).astype(np.float64)
    else:
        per_group = term.group_values(variable)
        inner = int(np.prod(term.shape[1:]))
        target = np.repeat(per_group, inner)
    u = np.arange(term.size)
    keep = u[cov & np.isfinite(target)] if covered_only else u[np.isfinite(target)]
    if keep.size == 0:
        return None
    return (np.arange(keep.size), term.col_base + keep, np.ones(keep.size), target[keep])
