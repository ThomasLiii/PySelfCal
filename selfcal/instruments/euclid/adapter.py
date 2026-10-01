"""Euclid NISP instrument — the broadband, multi-detector reference implementation.

NISP exposures hold 16 detectors per FITS file (science at extension 3k+1, the
data-quality mask at 3k+3), in ELECTRONS. The self-calibration model the EDFN
mosaics were made with (the frozen Y/J/H recipe of 2026-08) is, in the
vocabulary of :mod:`selfcal.models.spec`::

    [[model.offset]]                      # a detector-fixed pattern: one N x N grid
    map = "grid"; kind = "grouped"; groups = "detector"; reg_weight = 0.1
    adjacency = ["row", "col"]; mean_zero = true; exact_group_rows = true
    [[model.offset]]                      # per-frame readout stripes, damped toward 0
    map = "col_strips"; kind = "free"; adjacency = []; damp = 0.3
    [[model.offset]]
    map = "row_strips"; kind = "free"; adjacency = []; damp = 0.3
    scalar = true                          # per-frame DC

with, optionally, a per-frame linear tilt (``col_tilt`` / ``row_tilt`` maps,
``kind = "polybasis"``, degree 1). This adapter provides those chunk maps, the
exposure layout, an optional detector-edge taper (``[instrument]
edge_zero_px`` / ``edge_ramp_px``), the mosaic renderers (mean-preserving
spline for the grid, piecewise-constant strips, linear ramps) and the two
per-frame hooks of the recipe (``star_position_mask``, ``residual_mask``; see
``hooks.py``).

``[instrument]`` keys: ``band`` (Y | J | H; the product-name job), ``chunks``
(grid side N, default 40), ``strips`` (stripe count, default = chunks),
``tilt_strips`` (default 60), ``edge_zero_px`` / ``edge_ramp_px`` (default 0),
``detectors`` (default 16), ``det_shape`` (default [2040, 2040]).
"""
from __future__ import annotations

from functools import partial

import numpy as np

from ...geometry.map_helper import fill_invalid_offsets, make_grid_chunk_map, mean_preserving_spline_2d
from ...models.offset_structure import ChunkAxes, ChunkAxis
from ..base import (Instrument, register_instrument, Job, ChunkMap, DetectorGeometry, JobGeometry,
                    ExposureLayout)
from ..grid import upsample_chunk_map
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


# ---------------------------------------------------------------------------
# the instrument
# ---------------------------------------------------------------------------
@register_instrument('euclid')
class EuclidInstrument(Instrument):
    """Euclid NISP: broadband, 16 detectors per exposure, electrons."""

    capabilities = frozenset()

    # ---- run layout ------------------------------------------------------------
    def jobs(self, inst_cfg):
        return [Job(name=str(inst_cfg.get('band', 'Y')))]

    def frame_tag(self, inst_cfg):
        return str(inst_cfg.get('tag', 'EDFN'))

    def exposure_layout(self, inst_cfg):
        n = int(inst_cfg.get('detectors', ec.N_DETECTORS))
        return ExposureLayout(
            sci_ext=[int(e) for e in ec.sci_ext_list(n)], dq_ext=[int(e) for e in ec.dq_ext_list(n)],
            detector_ids=[int(d) for d in ec.det_idx_list(n)],
            ref_use_ext=tuple(int(e) for e in inst_cfg.get('ref_use_ext', [1, 10, 37, 46])),
            cache_tag=f"euclid_{inst_cfg.get('band', 'Y')}",
            default_ignore_bits=tuple(ec.DQ_IGNORE))

    # ---- geometry -------------------------------------------------------------------
    def detector_geometry(self, inst_cfg, oversample):
        det_shape = tuple(int(v) for v in inst_cfg.get('det_shape', ec.DET_SHAPE))
        n = int(inst_cfg.get('chunks', 40))
        n_strips = int(inst_cfg.get('strips', n))
        n_tilt = int(inst_cfg.get('tilt_strips', 60))
        o = int(oversample)
        grid = make_grid_chunk_map(det_shape, n)                     # int64, chunk = row * n + col
        maps = {'grid': ChunkMap(name='grid', det=grid, grid=upsample_chunk_map(grid, o),
                                 axes=ChunkAxes.row_major(('row', 'col'), (n, n), ('y', 'x')),
                                 adjacency_axes=('row', 'col'), spectral_axis=None, group_axis='row')}
        xs, ys = make_strip_chunk_maps(n_strips, det_shape)
        maps['col_strips'] = ChunkMap(name='col_strips', det=xs, grid=upsample_chunk_map(xs, o),
                                      axes=_strip_axes(n_strips, 'x'), group_axis='all')
        maps['row_strips'] = ChunkMap(name='row_strips', det=ys, grid=upsample_chunk_map(ys, o),
                                      axes=_strip_axes(n_strips, 'y'), group_axis='all')
        xt, yt = make_strip_chunk_maps(n_tilt, det_shape)
        maps['col_tilt'] = ChunkMap(name='col_tilt', det=xt, grid=upsample_chunk_map(xt, o),
                                    axes=_strip_axes(n_tilt, 'x'), spectral_axis='strip', group_axis='all')
        maps['row_tilt'] = ChunkMap(name='row_tilt', det=yt, grid=upsample_chunk_map(yt, o),
                                    axes=_strip_axes(n_tilt, 'y'), spectral_axis='strip', group_axis='all')
        return DetectorGeometry(shape=det_shape, chunk_maps=maps, primary='grid',
                                extra={'n_side': n, 'n_strips': n_strips, 'n_tilt': n_tilt,
                                       'det_shape': det_shape})

    def job_geometry(self, inst_cfg, geom, job):
        zero_px = int(inst_cfg.get('edge_zero_px', 0))
        ramp_px = int(inst_cfg.get('edge_ramp_px', 0))
        if zero_px > 0:
            det_w = make_edge_taper_weight(geom.shape, zero_px, ramp_px)
        else:
            det_w = np.ones(geom.shape)
        o = geom.chunk_map.grid.shape[0] // geom.shape[0]
        grid_w = det_w if o == 1 else np.kron(det_w, np.ones((o, o), dtype=det_w.dtype))
        n = geom.chunk_map.n_chunks
        return JobGeometry(det_valid_weight=det_w, grid_valid_weight=grid_w,
                           chunk_valid=np.ones(n, dtype=bool), chunk_valid_strict=np.ones(n, dtype=bool))

    # ---- mosaic ----------------------------------------------------------------------------
    def offset_renderer(self, inst_cfg, geom, jobgeom, map_name=None, render=None):
        det_shape = geom.extra['det_shape']
        name = map_name or geom.primary
        render = render or {'grid': 'spline', 'col_strips': 'strip', 'row_strips': 'strip',
                            'col_tilt': 'ramp', 'row_tilt': 'ramp'}.get(name)
        if render == 'spline':
            return partial(make_grid_offset_map, n_side=geom.extra['n_side'], det_shape=det_shape)
        if render == 'strip':
            return partial(make_free_strip_offset_map, det_shape=det_shape)
        if render == 'ramp':
            return partial(make_strip_offset_map, axis=1 if name.startswith('col') else 0, det_shape=det_shape)
        if render in (None, 'constant'):
            return None
        raise ValueError(f"unknown renderer {render!r} for map {name!r} (spline | strip | ramp | constant)")

    def data_unit(self, inst_cfg):
        return 'electron'

    # ---- hooks ---------------------------------------------------------------------------------
    def hooks(self):
        from .hooks import HOOKS
        return dict(HOOKS)
