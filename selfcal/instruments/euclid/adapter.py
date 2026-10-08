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
``kind = "polybasis"``, degree 1). The instrument contract of
:class:`~selfcal.instruments.euclid.settings.Euclid` provides those chunk maps, the
exposure layout, an optional detector-edge taper (``edge_zero_px`` /
``edge_ramp_px``), the mosaic renderers (mean-preserving spline for the grid,
piecewise-constant strips, linear ramps) and the two per-frame hooks of the recipe
(``star_position_mask``, ``residual_mask``; see ``hooks.py``), from the helpers
below. The registered ``euclid`` instrument of the TOML configs
(:class:`EuclidInstrument`) reads its ``[instrument]`` table as those settings.

``[instrument]`` keys: ``band`` (Y | J | H; the product-name job), ``chunks``
(grid side N, default 40), ``strips`` (stripe count, default = chunks),
``tilt_strips`` (default 60), ``edge_zero_px`` / ``edge_ramp_px`` (default 0),
``detectors`` (default 16), ``det_shape`` (default [2040, 2040]).
"""
from __future__ import annotations

import numpy as np

from ...geometry.map_helper import fill_invalid_offsets, mean_preserving_spline_2d
from ...models.offset_structure import ChunkAxes, ChunkAxis
from ..base import Instrument, register_instrument
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
def _settings(inst_cfg):
    """The ``[instrument]`` table as :class:`~selfcal.instruments.euclid.settings.Euclid` settings."""
    from .settings import Euclid  # loaded on first use (see the package docstring)
    return Euclid.from_table(inst_cfg)


@register_instrument('euclid')
class EuclidInstrument(Instrument):
    """Euclid NISP: broadband, 16 detectors per exposure, electrons. The ``[instrument]``
    table is read as :class:`~selfcal.instruments.euclid.settings.Euclid` settings, whose
    methods do the work."""

    capabilities = frozenset()

    # ---- run layout ------------------------------------------------------------
    def jobs(self, inst_cfg):
        """Return one job, named after ``[instrument].band`` (default ``Y``).

        The band only names the products (``cal_<frame_tag>_<band><suffix>.h5``)
        and selects no data: the exposures are chosen at reprojection by the
        ``[reproject]`` file pattern, which can refer to it as ``{band}`` (e.g.
        ``"/*_{band}*.fits"``)."""
        return [j.engine_job() for j in _settings(inst_cfg).default_jobs()]

    def frame_tag(self, inst_cfg):
        """Return the product-name tag, ``[instrument].tag`` (default ``EDFN``).

        With the job name it forms every product name, e.g. ``cal_EDFN_Y<suffix>.h5``
        and ``mosaic_EDFN_Y<suffix>.fits``."""
        return _settings(inst_cfg).product_tag

    def exposure_layout(self, inst_cfg):
        """Return the layout of a NISP exposure file of ``detectors`` (default 16) detectors
        (:meth:`~selfcal.instruments.euclid.settings.Euclid.layout`)."""
        return _settings(inst_cfg).layout()

    # ---- geometry -------------------------------------------------------------------
    def detector_geometry(self, inst_cfg, oversample):
        """Return the five chunk maps of a NISP detector, the square ``grid`` being primary
        (:meth:`~selfcal.instruments.euclid.settings.Euclid.geometry`)."""
        return _settings(inst_cfg).geometry(oversample)

    def job_geometry(self, inst_cfg, geom, job):
        """Return the job's pixel weights: 1 everywhere, or an optional taper at the detector edges
        (:meth:`~selfcal.instruments.euclid.settings.Euclid.job_geometry`)."""
        return _settings(inst_cfg).job_geometry(geom, job)

    # ---- mosaic ----------------------------------------------------------------------------
    def offset_renderer(self, inst_cfg, geom, jobgeom, map_name=None, render=None):
        """Return the function that draws one chunk map's offsets for the mosaic, or ``None``
        (:meth:`~selfcal.instruments.euclid.settings.Euclid.offset_renderer`)."""
        return _settings(inst_cfg).offset_renderer(geom, jobgeom, map_name=map_name, render=render)

    def data_unit(self, inst_cfg):
        """Return ``'electron'``, written as the mosaic ``BUNIT``."""
        from .settings import Euclid
        return Euclid.unit

    # ---- hooks ---------------------------------------------------------------------------------
    def hooks(self):
        """Return the recipe's per-frame hook factories, ``star_position_mask`` and ``residual_mask``
        (:meth:`~selfcal.instruments.euclid.settings.Euclid.hooks`)."""
        from .settings import Euclid
        return Euclid.hooks()
