"""Helpers of the SPHEREx geometry: the named subchannel windows and the readout-channel map.

Everything SPHEREx-LVF-specific of a run (the stripped (arc) chunk map and its ``(subchannel,
column)`` chunk axes, the H2RG readout-channel map, the BC/BW wavelength maps, a job's subchannel
masks, the smooth arc offset renderer, the LVF wavelength coadd, the FINAST astrometry filter) is
the contract of :class:`~selfcal.instruments.spherex.settings.SPHEREx`; it builds its chunk maps
with these.
"""
from __future__ import annotations

import numpy as np

# Named subchannel windows -> (inclusive-low, exclusive-high) subch index range.
# Aromatic/Aliphatic are stable; the PAH-fit window is run-dependent (different
# production runs have chosen different subchannel ranges), so it is NOT a
# global preset — such runs give the window's subchannels explicitly
# (spherex.window(name, subchannels=range(lo, hi))).
SUBCH_WINDOWS = {
    'Aromatic': (225, 236),
    'Aliphatic': (249, 260),
}


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
