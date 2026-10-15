"""Rectangular chunk maps: the helpers the instruments' geometries share.

:func:`rect_grid_chunk_map` cuts a detector into ``ny x nx`` rectangular chunks (the chunk map of
:class:`~selfcal.instruments.camera.Camera` and of
:meth:`~selfcal.instruments.base.ChunkMap.rectangles`); :func:`upsample_chunk_map` samples a
detector-resolution chunk map onto the oversampled grid the mosaic uses.
"""
from __future__ import annotations

import numpy as np


def upsample_chunk_map(det_chunk_map, factor):
    """Replicate each detector pixel into a (factor x factor) block, preserving ids."""
    if factor == 1:
        return det_chunk_map
    return np.kron(det_chunk_map, np.ones((factor, factor), dtype=det_chunk_map.dtype))


def rect_grid_chunk_map(det_shape, ny, nx):
    """``ny x nx`` rectangular chunks over ``det_shape``; chunk id = row * nx + col.

    Pixel row ``r`` falls in chunk row ``r * ny // H`` (columns likewise), so chunk sides
    differ by at most one pixel when ``ny`` or ``nx`` does not divide the detector
    (the same layout as :func:`~selfcal.geometry.map_helper.make_grid_chunk_map`
    for ``ny == nx``)."""
    H, W = det_shape
    rows = np.minimum(np.arange(H) * ny // H, ny - 1)
    cols = np.minimum(np.arange(W) * nx // W, nx - 1)
    return (rows[:, None] * nx + cols[None, :]).astype(np.int32)
