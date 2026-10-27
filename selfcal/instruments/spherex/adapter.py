"""Helpers of the SPHEREx geometry: the detector-column and readout-channel maps.

The chunk maps a model solves on are built from these by the model side (``selfcal_scripts/spherex/chunks.py``);
the LVF arc maps and their renderer are in :mod:`~selfcal.instruments.spherex.spherex_utility`.
"""
from __future__ import annotations

import numpy as np


def make_column_chunk_map(det_shape=(2040, 2040)):
    """One chunk per detector column at detector resolution: chunk id = the column index
    (every row, the reference columns included). Returns (chunk_map int32, n_chunks). An offset
    term on this map shared by every frame (``sc.Offsets(on='detector_column', per='all')``) is a
    detector-fixed column pattern: the survey-mean of what the per-frame terms do not represent
    along x, which one-sided coverage otherwise amplifies into the map's outskirts (2026-10)."""
    H, W = det_shape
    return np.tile(np.arange(W, dtype=np.int32), (H, 1)), int(W)


def make_readout_chunk_map(det_shape=(2040, 2040), col_start=60, col_width=64):
    """Per-readout-channel chunk map at detector resolution (H2RG, post 4px trim).
    Chunk 0 covers the first ``col_start`` reference columns; then one chunk per
    ``col_width``-wide readout column, plus a final partial chunk for any
    remaining columns. Returns (chunk_map int32, n_chunks)."""
    H, W = det_shape
    chunk_map = np.full(det_shape, -1, dtype=np.int32)
    chunk_map[:, :col_start] = 0
    n_full = (W - col_start) // col_width
    for i in range(n_full):
        x0 = col_start + i * col_width
        chunk_map[:, x0: x0 + col_width] = i + 1
    right_start = col_start + n_full * col_width
    n_chunks = n_full + 1
    if right_start < W:
        chunk_map[:, right_start:] = n_chunks
        n_chunks += 1
    # Internal invariant: verifies this function's own tiling of the map it just
    # built left no pixel unassigned (not caller-input validation) -> keep assert.
    assert (chunk_map >= 0).all(), "every pixel must be assigned a readout channel"
    return chunk_map, n_chunks
