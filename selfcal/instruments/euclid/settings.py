"""Euclid NISP as settings: ``sc.Euclid(band="Y")``.

:class:`Euclid` implements the instrument contract
(:class:`~selfcal.instruments.contract.Instrument`): the exposure layout of 16 detectors per
file, the optional detector-edge taper, and the default chunk maps a model gets when it names
none: the square grid (:func:`grid_cells`), the row and column stripes (:func:`strips`) and the
tilted stripes (:func:`tilt_strips`), each with its mosaic renderers (the helpers are in
:mod:`~selfcal.instruments.euclid.adapter`).
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from functools import partial
from typing import Literal

import numpy as np

from ...config.base import ConfigError
from ...geometry.map_helper import make_grid_chunk_map
from ...models.model import ChunkMap
from ...models.offset_structure import ChunkAxes
from ..base import Chunks, ExposureLayout, JobGeometry
from ..contract import Geometry, Instrument
from . import conventions as ec
from .adapter import (
    _strip_axes,
    make_edge_taper_weight,
    make_free_strip_offset_map,
    make_grid_offset_map,
    make_strip_chunk_maps,
    make_strip_offset_map,
)

__all__ = ['Euclid', 'grid_cells', 'strips', 'tilt_strips']


# ---- the mosaic renderers of the maps: make(chunks, jobgeom) -> renderer(chunk_map, offsets) -> image
def _spline(chunks, jobgeom):
    """A mean-preserving 2-D spline through the cells' offsets
    (:func:`~selfcal.instruments.euclid.adapter.make_grid_offset_map`); zeroed cells are filled from
    their neighbours."""
    renderer = partial(make_grid_offset_map, n_side=chunks.extra['n_side'], det_shape=chunks.extra['det_shape'])
    renderer.fills_invalid = True
    return renderer


def _strip(chunks, jobgeom):
    """Constant over each strip (:func:`~selfcal.instruments.euclid.adapter.make_free_strip_offset_map`)."""
    return partial(make_free_strip_offset_map, det_shape=chunks.extra['det_shape'])


def _ramp(chunks, jobgeom):
    """A line fitted to the strip values along the strips' scan direction
    (:func:`~selfcal.instruments.euclid.adapter.make_strip_offset_map`)."""
    return partial(make_strip_offset_map, axis=1 if chunks.extra['scan'] == 'x' else 0,
                   det_shape=chunks.extra['det_shape'])


# ---- the chunk maps
def grid_cells(geometry, n) -> Chunks:
    """``n x n`` square cells (chunk id ``row * n + col``) with the axes ``row`` and ``col``,
    regularised along both by default, ``row`` the group axis; when ``n`` does not divide the
    detector side (40 and 60 divide 2040), the cell sides differ by at most one pixel
    (:func:`~selfcal.geometry.map_helper.make_grid_chunk_map`). Rendered by ``spline`` (the
    default) or constant over each cell."""
    n = int(n)
    det = make_grid_chunk_map(tuple(geometry.shape), n)                     # int64, chunk = row * n + col
    return Chunks(det, ChunkAxes.row_major(('row', 'col'), (n, n), ('y', 'x')), adjacency_axes=('row', 'col'),
                  spectral_axis=None, group_axis='row', renderers={'spline': _spline}, render='spline',
                  extra={'n_side': n, 'det_shape': tuple(geometry.shape)})


def strips(geometry, n, scan) -> Chunks:
    """``n`` strips across the detector, vertical ones (``scan='x'``: columns, for the per-frame
    readout stripes) or horizontal ones (``'y'``), with the axes ``strip`` and ``all`` (a single
    group) and no default adjacency. Rendered by ``strip`` (constant over each strip, the default)
    or ``ramp`` (a line along the scan direction)."""
    n = int(n)
    xs, ys = make_strip_chunk_maps(n, tuple(geometry.shape))
    return Chunks(xs if scan == 'x' else ys, _strip_axes(n, scan), group_axis='all',
                  renderers={'strip': _strip, 'ramp': _ramp}, render='strip',
                  extra={'scan': scan, 'det_shape': tuple(geometry.shape)})


def tilt_strips(geometry, n, scan) -> Chunks:
    """:func:`strips` whose ``strip`` axis is the spectral axis, so that a degree-1 ``polybasis``
    term on one is a single linear ramp per frame; rendered by ``ramp`` (the default) or
    ``strip``."""
    cm = strips(geometry, n, scan)
    return Chunks(cm.det, cm.axes, group_axis='all', spectral_axis='strip', renderers=cm.renderers, render='ramp',
                  extra=cm.extra)


@dataclass(frozen=True)
class Euclid(Instrument):
    """Euclid NISP: ``detectors`` detectors of ``det_shape`` pixels per exposure, in the photometric
    ``band``. Its default chunk maps: a square grid of ``chunks`` x ``chunks`` (``grid``, primary),
    row and column stripes of ``strips`` (default ``chunks``; ``col_strips``, ``row_strips``) and
    tilted stripes of ``tilt_strips`` (``col_tilt``, ``row_tilt``); detector edges are zeroed over
    ``edge_zero_px`` and tapered over ``edge_ramp_px``. ``reference_ext``: the extensions whose WCS
    define the reference grid. ``tag``: the products' tag (with the band as the job's name:
    ``cal_EDFN_Y<suffix>.h5``)."""
    _: KW_ONLY
    band: Literal['Y', 'J', 'H'] = 'Y'
    chunks: int = 40
    strips: int | None = None
    tilt_strips: int = 60
    edge_zero_px: int = 0
    edge_ramp_px: int = 0
    detectors: int = ec.N_DETECTORS
    det_shape: tuple[int, int] = tuple(ec.DET_SHAPE)
    reference_ext: tuple[int, ...] = (1, 10, 37, 46)
    tag: str = 'EDFN'

    # The unit of the data (the mosaic's BUNIT): a constant, not a setting (without an
    # annotation it is no dataclass field, so it stays out of the products' fingerprints). The
    # geometry reads the settings only: the engine may keep it between actions.
    unit = 'electron'
    geometry_is_pure = True

    def _validate(self):
        for k in ('chunks', 'tilt_strips', 'detectors'):
            if getattr(self, k) < 1:
                raise ConfigError(f"Euclid({k}={getattr(self, k)}): at least 1")
        if self.strips is not None and self.strips < 1:
            raise ConfigError(f"Euclid(strips={self.strips}): at least 1")

    @property
    def product_tag(self) -> str:
        return self.tag

    def default_ignore_flags(self):
        return tuple(ec.DQ_IGNORE)

    # ---- the contract -------------------------------------------------------------------
    def layout(self) -> ExposureLayout:
        """Return the layout of a NISP exposure file of ``detectors`` (default 16) detectors.

        Detector ``k`` (0-based) has its science image in FITS extension ``3k + 1``
        and its data-quality mask in ``3k + 3``, and its reprojected frames carry
        detector index ``k``. The WCS of the extensions in ``reference_ext`` (default
        ``(1, 10, 37, 46)``, the science extensions of detectors 0, 3, 12 and 15) defines the
        reference frame. There is no
        header filter and no custom reader. ``default_ignore_bits`` records the DQ bits
        11 and 15 (:data:`~selfcal.instruments.euclid.conventions.DQ_IGNORE`), the bits
        :meth:`default_ignore_flags` ignores."""
        n = self.detectors
        return ExposureLayout(
            sci_ext=[int(e) for e in ec.sci_ext_list(n)], dq_ext=[int(e) for e in ec.dq_ext_list(n)],
            detector_ids=[int(d) for d in ec.det_idx_list(n)],
            ref_use_ext=tuple(int(e) for e in self.reference_ext),
            cache_tag=f"euclid_{self.band}",
            default_ignore_bits=tuple(ec.DQ_IGNORE))

    def geometry(self, oversample=1):
        """The detector: ``det_shape`` pixels (default 2040 x 2040), no aux maps (the sky is
        broadband)."""
        return Geometry(tuple(int(v) for v in self.det_shape), oversample)

    def default_chunk_maps(self):
        """The five maps of a NISP detector, the square ``grid`` primary: ``grid``
        (:func:`grid_cells` of ``chunks``), ``col_strips`` / ``row_strips`` (:func:`strips` of
        ``strips``) and ``col_tilt`` / ``row_tilt`` (:func:`tilt_strips` of ``tilt_strips``)."""
        n_strips = int(self.strips if self.strips is not None else self.chunks)
        return {'grid': ChunkMap(grid_cells, n=int(self.chunks)),
                'col_strips': ChunkMap(strips, n=n_strips, scan='x'),
                'row_strips': ChunkMap(strips, n=n_strips, scan='y'),
                'col_tilt': ChunkMap(tilt_strips, n=int(self.tilt_strips), scan='x'),
                'row_tilt': ChunkMap(tilt_strips, n=int(self.tilt_strips), scan='y')}

    def job_geometry(self, geom, job) -> JobGeometry:
        """Return the job's pixel weights: 1 everywhere, or an optional taper at the detector edges.

        With ``edge_zero_px`` > 0 the weight is ``clip((d - edge_zero_px) /
        max(edge_ramp_px, 1), 0, 1)`` (float32), where ``d`` is a pixel's distance in pixels
        from the nearest detector edge (0 for the outermost pixels; see
        :func:`~selfcal.instruments.euclid.adapter.make_edge_taper_weight`); otherwise float64
        ones; ``edge_ramp_px`` alone has no effect. The same weight serves the solve (detector
        grid) and, replicated into blocks of the oversampling factor, the mosaic (reference
        grid). Every chunk of the primary map is valid, and ``job`` is not read."""
        zero_px = int(self.edge_zero_px)
        ramp_px = int(self.edge_ramp_px)
        if zero_px > 0:
            det_w = make_edge_taper_weight(geom.shape, zero_px, ramp_px)
        else:
            det_w = np.ones(geom.shape)
        o = geom.chunk_map.grid.shape[0] // geom.shape[0]
        grid_w = det_w if o == 1 else np.kron(det_w, np.ones((o, o), dtype=det_w.dtype))
        n = geom.chunk_map.n_chunks
        return JobGeometry(det_valid_weight=det_w, grid_valid_weight=grid_w,
                           chunk_valid=np.ones(n, dtype=bool), chunk_valid_strict=np.ones(n, dtype=bool))
