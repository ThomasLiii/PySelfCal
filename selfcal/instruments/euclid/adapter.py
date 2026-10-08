"""Helpers of the Euclid NISP instrument: its strip chunk maps, mosaic renderers and edge taper.

NISP exposures hold 16 detectors per FITS file (science at extension 3k+1, the
data-quality mask at 3k+3), in ELECTRONS. The self-calibration model the EDFN
mosaics were made with (the frozen Y/J/H recipe of 2026-08) is::

    sc.Model(offsets=[
        # a detector-fixed pattern: one N x N grid per detector
        sc.Offsets(on="grid", per="detector", smooth=0.1, smooth_along=("row", "col"), mean_zero=True,
                   exact_group_rows=True),
        # per-frame readout stripes, damped toward 0
        sc.Offsets(on="col_strips", smooth_along=(), damping=0.3),
        sc.Offsets(on="row_strips", smooth_along=(), damping=0.3)])      # and the per-frame scalar

with, optionally, a per-frame linear tilt (``col_tilt`` / ``row_tilt`` maps, a degree-1
polynomial offset). The instrument contract of :class:`~selfcal.instruments.euclid.settings.Euclid`
provides those chunk maps, the exposure layout, an optional detector-edge taper (``edge_zero_px`` /
``edge_ramp_px``) and the mosaic renderers (mean-preserving spline for the grid, piecewise-constant
strips, linear ramps), from the helpers below; the recipe's per-frame hooks are in
:mod:`~selfcal.instruments.euclid.hooks`.
"""
from __future__ import annotations

import numpy as np

from ...geometry.map_helper import fill_invalid_offsets, mean_preserving_spline_2d
from ...models.offset_structure import ChunkAxes, ChunkAxis
from . import conventions as ec


# ---------------------------------------------------------------------------
# chunk maps
# ---------------------------------------------------------------------------
def make_strip_chunk_maps(n_strips, det_shape=ec.DET_SHAPE):
    """(x_strip_map, y_strip_map): chunk id = strip index along x (resp. y)."""
    h, w = det_shape
    col_ids = (np.arange(w, dtype=np.int64) * n_strips) // w
    row_ids = (np.arange(h, dtype=np.int64) * n_strips) // h
    x_map = np.broadcast_to(col_ids[None, :], det_shape).copy()
    y_map = np.broadcast_to(row_ids[:, None], det_shape).copy()
    return x_map, y_map


def _strip_axes(n_strips, scan):
    """A strip map's axes: the strip index (``strip``) and a single group (``all``)
    so a degree-1 polynomial basis in ``strip`` per ``all`` is one ramp per frame."""
    return ChunkAxes((ChunkAxis('strip', int(n_strips), np.arange(n_strips), scan),
                      ChunkAxis('all', 1, np.zeros(n_strips, dtype=np.int64), 'both')))


# ---------------------------------------------------------------------------
# mosaic renderers
# ---------------------------------------------------------------------------
def make_grid_offset_map(chunk_map, chunk_offset, n_side, det_shape=ec.DET_SHAPE):
    """Smooth a per-chunk offset vector of the N x N grid onto the (possibly
    oversampled) detector grid with a mean-preserving 2-D spline; chunks the
    solve left at 0 are filled from their neighbours first."""
    offset_grid = fill_invalid_offsets(chunk_offset.reshape(n_side, n_side).copy())
    y_edges = np.linspace(0, det_shape[0], n_side + 1)
    x_edges = np.linspace(0, det_shape[1], n_side + 1)
    spl = mean_preserving_spline_2d(y_edges, x_edges, offset_grid)
    h, w = chunk_map.shape
    oversample = h // det_shape[0]
    shift = 0.5 / oversample
    inc = 1.0 / oversample
    y_mesh, x_mesh = np.meshgrid(
        np.arange(shift, det_shape[0] + shift, inc),
        np.arange(shift, det_shape[1] + shift, inc), indexing="ij")
    return spl(y_mesh, x_mesh)


def make_free_strip_offset_map(chunk_map, chunk_offset, det_shape=ec.DET_SHAPE):
    """Renderer for a FREE per-strip offset map: each strip's fitted value is
    broadcast over that strip's pixels. Piecewise constant on purpose —
    readout-channel structure is discontinuous at channel boundaries, so a
    mean-preserving spline would smear it."""
    vals = np.nan_to_num(np.asarray(chunk_offset, dtype=np.float64), nan=0.0)
    return vals[chunk_map]


def make_strip_offset_map(chunk_map, chunk_offset, axis, det_shape=ec.DET_SHAPE):
    """Renderer for a tilt map: the saved per-strip offsets are exactly linear in
    the strip index by construction, so fit the line and evaluate it at pixel
    centres (a smooth ramp, no staircase). ``axis=1`` -> ramp along x
    (vertical strips); ``axis=0`` -> along y."""
    n_strips = len(chunk_offset)
    span = det_shape[axis]
    centers = (np.arange(n_strips) + 0.5) * (span / n_strips)
    slope, intercept = np.polyfit(centers, np.asarray(chunk_offset, float), 1)
    h, w = chunk_map.shape
    oversample = h // det_shape[0]
    shift = 0.5 / oversample
    inc = 1.0 / oversample
    coord = np.arange(shift, span + shift, inc)[: (h if axis == 0 else w)]
    ramp = intercept + slope * coord
    if axis == 0:
        return np.broadcast_to(ramp[:, None], (h, w)).copy()
    return np.broadcast_to(ramp[None, :], (h, w)).copy()


def make_edge_taper_weight(det_shape, zero_px, ramp_px):
    """Detector-plane weight that ZEROES the outermost ``zero_px`` and ramps
    linearly to 1 over the next ``ramp_px`` — each frame's own edge rim then
    contributes nothing (the raw frames carry a rim / trough locked to the
    detector edge that no offset model removes)."""
    h, w = det_shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.minimum.reduce([yy, h - 1 - yy, xx, w - 1 - xx]).astype(np.float64)
    wt = np.clip((dist - zero_px) / max(ramp_px, 1), 0.0, 1.0)
    return wt.astype(np.float32)
