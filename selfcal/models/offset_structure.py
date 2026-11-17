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

import numpy as np

logger = logging.getLogger(__name__)

__all__ = ['ChunkAxis', 'ChunkAxes', 'adjacency_along', 'adjacency_union', 'group_edges_along', 'group_aux_edges']


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
        """The axes other than ``name``, in order (a list of :class:`ChunkAxis`)."""
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
