"""Offset structure from chunk AXES — the instrument-agnostic builders.

A chunk map partitions the detector into chunks; an offset block carries one
unknown per (frame, chunk). Which chunks are *regularised together*, *forced
onto a polynomial* or *share a hard polynomial basis* is a property of the
chunk grid's COORDINATES, not of the telescope: SPHEREx's stripped map is a
(subchannel, column) grid, Euclid's strip map a (strip,) line, a broadband
camera's map a (row, col) grid. Every builder here takes the grid's axes
(:class:`ChunkAxes`: per axis, the value of every chunk id and the image
direction along which neighbouring chunks differ in it) and works on any of
them, so a mode can say "adjacency along *column*" or "degree-2 polynomial
along *subchannel* per *column*" without knowing how the chunk ids were
encoded.

Byte-for-byte compatible with the SPHEREx builders they replaced
(``spherex_utility.compute_*_adjacency`` / ``compute_*_polynomial_chains`` and
the adapter's ``subchannel_poly_basis`` / ``subchannel_bc_edges``): same pairs,
same chain order, same dtypes — ``tests/test_offset_structure.py`` proves it on
the real maps.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from math import comb

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['ChunkAxis', 'ChunkAxes', 'adjacency_along', 'poly_chains_along',
           'poly_basis_along', 'group_edges_along', 'group_aux_edges', 'fd_stencil']


@dataclass(frozen=True)
class ChunkAxis:
    """One coordinate of a chunk grid: ``of_chunk[chunk_id]`` = the chunk's
    value along this axis; ``size`` = number of distinct values; ``scan`` = the
    image direction along which neighbouring chunks differ in this coordinate
    (``'x'`` = horizontal pixel neighbours, ``'y'`` = vertical, ``'both'``)."""
    name: str
    size: int
    of_chunk: np.ndarray
    scan: str = 'both'


class ChunkAxes:
    """The coordinates of a chunk grid, by name."""

    def __init__(self, axes):
        self._axes = tuple(axes)
        n = {len(a.of_chunk) for a in self._axes}
        if len(n) != 1:
            raise ValueError(f"axis arrays disagree on the chunk count: {n}")
        self.n_chunks = n.pop()
        self.names = [a.name for a in self._axes]

    @classmethod
    def row_major(cls, names, sizes, scans):
        """The dense encoding ``chunk = i_0 * (n_1 * ...) + i_1 * ... + i_last``.
        SPHEREx: ``chunk = subchannel * num_col + column`` ->
        ``row_major(('subchannel', 'column'), (n_sub, num_col), ('y', 'x'))``."""
        sizes = [int(s) for s in sizes]
        ids = np.arange(int(np.prod(sizes)))
        values, rest = {}, ids
        for name, size in zip(reversed(list(names)), reversed(sizes)):
            values[name] = rest % size
            rest = rest // size
        return cls(ChunkAxis(n, s, values[n], sc) for n, s, sc in zip(names, sizes, scans))

    def __iter__(self):
        return iter(self._axes)

    def __len__(self):
        return len(self._axes)

    def __contains__(self, name):
        return name in self.names

    def __getitem__(self, name) -> ChunkAxis:
        for a in self._axes:
            if a.name == name:
                return a
        raise KeyError(f"no chunk axis {name!r} (axes: {self.names})")

    def others(self, name):
        return [a for a in self._axes if a.name != name]

    def __repr__(self):
        return 'ChunkAxes(' + ', '.join(f'{a.name}[{a.size}]/{a.scan}' for a in self._axes) + ')'


def _pixel_transitions(chunk_map, scan):
    """(u, v) chunk-id pairs of neighbouring pixels that belong to different
    chunks, scanning ``'x'`` (horizontal) or ``'y'`` (vertical)."""
    if scan == 'x':
        a, b = chunk_map[:, :-1], chunk_map[:, 1:]
    elif scan == 'y':
        a, b = chunk_map[:-1, :], chunk_map[1:, :]
    else:
        raise ValueError(f"scan must be 'x' or 'y', got {scan!r}")
    mask = (a != -1) & (b != -1) & (a != b)
    return a[mask], b[mask]


def adjacency_along(chunk_map, axes: ChunkAxes, axis: str, step: int | None = None):
    """Regularisation pairs between chunks that touch in the image (scanning
    pixel neighbours along the axis' ``scan`` direction) and differ ONLY along
    ``axis`` (every other axis equal); with ``step`` the difference along
    ``axis`` must be exactly ``step``. Returns ``(i, j)`` unique sorted pairs
    (empty arrays of the chunk map's dtype when there are none).

    SPHEREx: ``adjacency_along(cm, axes, 'column')`` is the column
    (vertical-strip) adjacency; ``adjacency_along(cm, axes, 'subchannel',
    step=1)`` the subchannel adjacency.
    """
    ax = axes[axis]
    scans = ('x', 'y') if ax.scan == 'both' else (ax.scan,)
    us, vs = zip(*(_pixel_transitions(chunk_map, s) for s in scans))
    u, v = np.concatenate(us), np.concatenate(vs)
    keep = np.ones(u.shape, dtype=bool)
    for other in axes.others(axis):
        keep &= other.of_chunk[u] == other.of_chunk[v]
    if step is not None:
        keep &= np.abs(ax.of_chunk[u] - ax.of_chunk[v]) == step
    u, v = u[keep], v[keep]
    if u.size == 0:
        logger.info(f"Found 0 {axis} adjacency pairs.")
        empty = np.array([], dtype=np.asarray(chunk_map).dtype)
        return empty, empty.copy()
    pairs = np.sort(np.stack([u, v], axis=1), axis=1)
    unique_pairs = np.unique(pairs, axis=0)
    logger.info(f"Found {len(unique_pairs)} {axis} adjacency pairs.")
    return unique_pairs[:, 0], unique_pairs[:, 1]


def adjacency_union(chunk_map, axes: ChunkAxes, axis_names):
    """Pairs of :func:`adjacency_along` over several axes, merged (unique, sorted)."""
    parts = [adjacency_along(chunk_map, axes, a) for a in axis_names]
    parts = [p for p in parts if len(p[0])]
    if not parts:
        empty = np.array([], dtype=np.asarray(chunk_map).dtype)
        return empty, empty.copy()
    if len(parts) == 1:
        return parts[0]
    i = np.concatenate([p[0] for p in parts]); j = np.concatenate([p[1] for p in parts])
    unique_pairs = np.unique(np.stack([i, j], axis=1), axis=0)
    return unique_pairs[:, 0], unique_pairs[:, 1]


def fd_stencil(degree):
    """The ``(degree+1)``-th finite-difference stencil ``(-1)^k C(degree+1, k)``,
    length ``degree + 2``: annihilates any polynomial of degree ``<= degree``."""
    if degree < 0:
        raise ValueError(f"degree must be >= 0 (got {degree})")
    return np.array([(-1) ** k * comb(degree + 1, k) for k in range(degree + 2)], dtype=np.float64)


def chunk_table(axes: ChunkAxes, names):
    """Dense lookup ``table[i_0, i_1, ...] = chunk id`` over the axes ``names``
    (-1 where no chunk has those coordinates)."""
    shape = tuple(axes[n].size for n in names)
    table = np.full(shape, -1, dtype=np.int64)
    table[tuple(axes[n].of_chunk for n in names)] = np.arange(axes.n_chunks)
    return table


def poly_chains_along(axes: ChunkAxes, axis: str, degree: int, lo: int | None = None,
                      hi: int | None = None):
    """Chains + stencil of the soft polynomial constraint along ``axis``: for
    every combination of the OTHER axes' values, sliding windows of
    ``L = degree + 2`` consecutive ``axis`` values form the chains, optionally
    restricted to windows whose first value lies in ``[lo, hi - L + 1]``
    (``hi`` = the last value a chain may reach, inclusive). Chains are ordered
    by their first chunk id (the order the row assembly emits them in).

    Every chain must be fully populated (the axes must form a dense grid, as
    SPHEREx's and any row-major encoding do).
    """
    L = degree + 2
    size = axes[axis].size
    if size < L:
        raise ValueError(f"axis {axis!r} has {size} values, too few for degree={degree}: need >= {L}")
    s_lo = 0 if lo is None else int(lo)
    s_hi = (size - L) if hi is None else int(hi) - L + 1
    if s_lo < 0 or s_hi > size - L:
        raise ValueError(f"window lo={lo}, hi={hi} (chain start range [{s_lo}, {s_hi}]) outside "
                         f"valid [0, {size - L}]")
    if s_hi < s_lo:
        raise ValueError(f"window [{lo}, {hi}] yields no length-{L} chains along {axis!r}")
    names = [axis] + [a.name for a in axes.others(axis)]
    table = chunk_table(axes, names)                                     # (size_axis, *other_sizes)
    starts = np.arange(s_lo, s_hi + 1, dtype=np.int64)
    win = starts[:, None] + np.arange(L, dtype=np.int64)[None, :]        # (n_starts, L)
    chains = table[win]                                                  # (n_starts, L, *other_sizes)
    chains = np.moveaxis(chains, 1, -1).reshape(-1, L)                   # (n_starts * n_other, L)
    if (chains < 0).any():
        raise ValueError(f"the chunk grid is not dense along {axis!r}: a chain hits a missing chunk")
    chains = chains[np.argsort(chains[:, 0], kind='stable')]
    return chains, fd_stencil(degree)


def poly_basis_along(axes: ChunkAxes, coord_axis: str, group_axis: str, degree: int,
                     lo: int, hi: int, segments=None):
    """The hard polynomial-basis descriptor consumed by
    :mod:`selfcal.models.offset_basis`: a degree-``degree`` Chebyshev in
    ``coord_axis`` over the window ``[lo, hi]``, one independent polynomial
    per value of ``group_axis``; with ``segments`` (inclusive ``[lo, hi]``
    ranges inside the window) an independent polynomial per segment."""
    pb = {
        'degree': int(degree),
        'num_groups': int(axes[group_axis].size),
        'coord_lo': int(lo), 'coord_hi': int(hi),
        'chunk_coord': np.asarray(axes[coord_axis].of_chunk),
        'chunk_group': np.asarray(axes[group_axis].of_chunk),
    }
    if segments:
        segs = [(int(a), int(b)) for a, b in segments]
        if segs[0][0] < int(lo) or segs[-1][1] > int(hi):
            raise ValueError(f"segments {segs} must lie inside the window [{lo}, {hi}]")
        pb['segments'] = segs
    return pb


def group_aux_edges(det_aux, det_chunk_map, group_of_chunk, min_pixels=50):
    """Bin edges in an aux quantity between consecutive chunk GROUPS (e.g. the
    wavelength boundaries between consecutive subchannels), for the grouped
    outlier clip: midpoints between the sorted per-group mean aux values over
    groups with at least ``min_pixels`` finite positive pixels.
    ``group_of_chunk[chunk]`` maps a chunk id to its group."""
    w = np.asarray(det_aux, dtype=np.float64)
    cm = np.asarray(det_chunk_map)
    grp = np.where(cm >= 0, np.asarray(group_of_chunk)[np.maximum(cm, 0)], -1)
    ngrp = int(grp.max()) + 1
    mean = np.full(ngrp, np.nan)
    valid = np.isfinite(w) & (w > 0) & (grp >= 0)
    cnt = np.bincount(grp[valid].ravel(), minlength=ngrp)
    sums = np.bincount(grp[valid].ravel(), weights=w[valid].ravel(), minlength=ngrp)
    ok = cnt >= min_pixels
    mean[ok] = sums[ok] / cnt[ok]
    ws = np.sort(mean[ok])
    return 0.5 * (ws[:-1] + ws[1:])


def group_edges_along(aux_map, chunk_map, axes: ChunkAxes, axis: str, min_pixels=50):
    """:func:`group_aux_edges` with ``group_of_chunk = axes[axis].of_chunk``."""
    return group_aux_edges(aux_map, chunk_map, np.asarray(axes[axis].of_chunk), min_pixels=min_pixels)
