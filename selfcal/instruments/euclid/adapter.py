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
        """Return one job, named after ``[instrument].band`` (default ``Y``).

        The band only names the products (``cal_<frame_tag>_<band><suffix>.h5``)
        and selects no data: the exposures are chosen at reprojection by the
        ``[reproject]`` file pattern, which can refer to it as ``{band}`` (e.g.
        ``"/*_{band}*.fits"``)."""
        return [Job(name=str(inst_cfg.get('band', 'Y')))]

    def frame_tag(self, inst_cfg):
        """Return the product-name tag, ``[instrument].tag`` (default ``EDFN``).

        With the job name it forms every product name, e.g. ``cal_EDFN_Y<suffix>.h5``
        and ``mosaic_EDFN_Y<suffix>.fits``."""
        return str(inst_cfg.get('tag', 'EDFN'))

    def exposure_layout(self, inst_cfg):
        """Return the layout of a NISP exposure file of ``detectors`` (default 16) detectors.

        Detector ``k`` (0-based) has its science image in FITS extension ``3k + 1``
        and its data-quality mask in ``3k + 3``, and its reprojected frames carry
        detector index ``k``. The WCS of the extensions in ``ref_use_ext`` (default
        ``[1, 10, 37, 46]``, the science extensions of detectors 0, 3, 12 and 15;
        ``[reproject].use_ext`` overrides it) defines the reference frame. There is no
        header filter and no custom reader. ``default_ignore_bits`` records the DQ bits
        11 and 15 (:data:`~selfcal.instruments.euclid.conventions.DQ_IGNORE`), but
        nothing reads it: the bits a run ignores are the ``ignore_list`` of its
        ``[calibration]`` and ``[mosaic]`` tables."""
        n = int(inst_cfg.get('detectors', ec.N_DETECTORS))
        return ExposureLayout(
            sci_ext=[int(e) for e in ec.sci_ext_list(n)], dq_ext=[int(e) for e in ec.dq_ext_list(n)],
            detector_ids=[int(d) for d in ec.det_idx_list(n)],
            ref_use_ext=tuple(int(e) for e in inst_cfg.get('ref_use_ext', [1, 10, 37, 46])),
            cache_tag=f"euclid_{inst_cfg.get('band', 'Y')}",
            default_ignore_bits=tuple(ec.DQ_IGNORE))

    # ---- geometry -------------------------------------------------------------------
    def detector_geometry(self, inst_cfg, oversample):
        """Return the five chunk maps of a NISP detector, the square ``grid`` being primary.

        - ``grid``: ``chunks`` x ``chunks`` square cells (default 40; chunk id
          ``row * chunks + col``) with the axes ``row`` and ``col``, regularised along
          both by default. When ``chunks`` does not divide the detector side (40 and
          60 divide 2040), the cell sides differ by at most one pixel
          (:func:`~selfcal.geometry.map_helper.make_grid_chunk_map`).
        - ``col_strips`` / ``row_strips``: ``strips`` (default ``chunks``) vertical /
          horizontal strips, for the per-frame readout stripes.
        - ``col_tilt`` / ``row_tilt``: ``tilt_strips`` (default 60) strips whose
          ``strip`` axis is the spectral axis, so that a degree-1 ``polybasis`` term
          on one is a single linear ramp per frame.

        The strip maps have the axes ``strip`` and ``all`` (a single group) and no
        default adjacency. The detector is ``det_shape`` pixels (default 2040 x 2040);
        each map is also built on the reference grid, every detector pixel replicated
        into an ``oversample`` x ``oversample`` block. There are no aux maps (the sky is
        broadband); ``extra`` holds ``n_side``, ``n_strips``, ``n_tilt`` and
        ``det_shape`` for the renderers."""
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
        """Return the job's pixel weights: 1 everywhere, or an optional taper at the detector edges.

        With ``edge_zero_px`` > 0 the weight is ``clip((d - edge_zero_px) /
        max(edge_ramp_px, 1), 0, 1)``, where ``d`` is a pixel's distance in pixels from
        the nearest detector edge (0 for the outermost pixels; see
        :func:`make_edge_taper_weight`); ``edge_ramp_px`` alone has no effect. The same
        weight serves the solve (detector grid) and, replicated into blocks of the
        oversampling factor, the mosaic (reference grid). Every chunk of the primary
        map is valid, and ``job`` is not read."""
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
        """Return the function that draws one chunk map's offsets for the mosaic, or ``None``.

        The mosaic calls it per frame as ``renderer(chunk_map, offsets)`` and
        subtracts the image it returns. ``render`` (an offset term's ``render`` key)
        picks the renderer; when it is ``None`` the map does (``map_name=None`` is the
        primary map, ``grid``):

        - ``'spline'``, the default for ``grid``: :func:`make_grid_offset_map`, a
          mean-preserving 2-D spline;
        - ``'strip'``, the default for ``col_strips`` / ``row_strips``:
          :func:`make_free_strip_offset_map`, constant over each strip;
        - ``'ramp'``, the default for ``col_tilt`` / ``row_tilt``:
          :func:`make_strip_offset_map`, a line fitted to the strip values, along x
          for a map whose name starts with ``col`` and along y otherwise;
        - ``'constant'``, or a map with no default: ``None``, which the mosaic renders
          constant over each chunk.

        Any other ``render`` raises ``ValueError``; ``jobgeom`` is not read."""
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
        """Return ``'electron'``, written as the mosaic ``BUNIT``."""
        return 'electron'

    # ---- hooks ---------------------------------------------------------------------------------
    def hooks(self):
        """Return the recipe's per-frame hook factories, ``star_position_mask`` and ``residual_mask``.

        A run config selects one by name as ``pre_cal``, ``post_cal`` or ``post_mosaic``
        in its ``[hooks]`` table; the entry's other keys are the factory's parameters
        (see :mod:`~selfcal.instruments.euclid.hooks`). The dict is a copy of
        :data:`~selfcal.instruments.euclid.hooks.HOOKS`."""
        from .hooks import HOOKS
        return dict(HOOKS)
