"""Euclid NISP as settings: ``sc.Euclid(band="Y")``.

:class:`Euclid` implements the instrument contract
(:class:`~selfcal.instruments.contract.Instrument`): the exposure layout of 16 detectors per
file, the grid, stripe and tilt chunk maps, the optional detector-edge taper and the mosaic
renderers (the helpers are in :mod:`~selfcal.instruments.euclid.adapter`).
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from functools import partial
from typing import Literal

import numpy as np

from ...config.base import ConfigError
from ...geometry.map_helper import make_grid_chunk_map
from ...models.offset_structure import ChunkAxes
from ..base import ChunkMap, DetectorGeometry, ExposureLayout, JobGeometry
from ..contract import Instrument, Job
from ..grid import upsample_chunk_map
from . import conventions as ec
from .adapter import (
    _strip_axes,
    make_edge_taper_weight,
    make_free_strip_offset_map,
    make_grid_offset_map,
    make_strip_chunk_maps,
    make_strip_offset_map,
)

__all__ = ['Euclid']


@dataclass(frozen=True)
class Euclid(Instrument):
    """Euclid NISP: ``detectors`` detectors of ``det_shape`` pixels per exposure, in the photometric
    ``band``. Chunk maps: a square grid of ``chunks`` x ``chunks`` (primary), row and column stripes
    of ``strips`` (default ``chunks``) and tilted stripes of ``tilt_strips``; detector edges are
    zeroed over ``edge_zero_px`` and tapered over ``edge_ramp_px``. ``reference_ext``: the
    extensions whose WCS define the reference grid. ``tag``: the products' tag (with the band:
    ``cal_EDFN_Y<suffix>.h5``). One job, named after the band."""
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
    # annotation it is no dataclass field, so it stays out of the products' fingerprints).
    unit = 'electron'

    def _validate(self):
        for k in ('chunks', 'tilt_strips', 'detectors'):
            if getattr(self, k) < 1:
                raise ConfigError(f"Euclid({k}={getattr(self, k)}): at least 1")
        if self.strips is not None and self.strips < 1:
            raise ConfigError(f"Euclid(strips={self.strips}): at least 1")

    @property
    def product_tag(self) -> str:
        return self.tag

    def default_jobs(self):
        return (Job(self.band),)

    def default_ignore_flags(self):
        return tuple(ec.DQ_IGNORE)

    def check_jobs(self, jobs):
        if len(jobs) != 1 or jobs[0].name != self.band:
            raise ConfigError(f"Euclid: one job, the band ({self.band!r}); got {[getattr(j, 'name', j) for j in jobs]}")

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

    def geometry(self, oversample=1) -> DetectorGeometry:
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
        default adjacency. The maps are ``int64``. The detector is ``det_shape`` pixels
        (default 2040 x 2040); each map is also built on the reference grid, every detector
        pixel replicated into an ``oversample`` x ``oversample`` block. There are no aux maps
        (the sky is broadband); ``extra`` holds ``n_side``, ``n_strips``, ``n_tilt`` and
        ``det_shape`` for the renderers."""
        det_shape = tuple(int(v) for v in self.det_shape)
        n = int(self.chunks)
        n_strips = int(self.strips if self.strips is not None else n)
        n_tilt = int(self.tilt_strips)
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

    def offset_renderer(self, geom, jobgeom, map_name=None, render=None):
        """Return the function that draws one chunk map's offsets for the mosaic, or ``None``.

        The mosaic calls it per frame as ``renderer(chunk_map, offsets)`` and
        subtracts the image it returns. ``render`` (an offset term's ``render``)
        picks the renderer; when it is ``None`` the map does (``map_name=None`` is the
        primary map, ``grid``):

        - ``'spline'``, the default for ``grid``:
          :func:`~selfcal.instruments.euclid.adapter.make_grid_offset_map`, a
          mean-preserving 2-D spline;
        - ``'strip'``, the default for ``col_strips`` / ``row_strips``:
          :func:`~selfcal.instruments.euclid.adapter.make_free_strip_offset_map`, constant
          over each strip;
        - ``'ramp'``, the default for ``col_tilt`` / ``row_tilt``:
          :func:`~selfcal.instruments.euclid.adapter.make_strip_offset_map`, a line fitted
          to the strip values, along x for a map whose name starts with ``col`` and along y
          otherwise;
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
