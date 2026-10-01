"""The built-in ``grid`` instrument — any imager, configured without code.

A detector of ``detector_shape`` pixels, partitioned into an ``ny x nx`` grid
of rectangular chunks (``chunks``); one job; no wavelength maps (continuum
sky). Everything comes from the ``[instrument]`` table::

    [instrument]
    name = "grid"
    detector_shape = [2048, 2048]   # (rows, cols) of one science array
    chunks = [8, 8]                 # chunk grid (rows, cols); [8] = 8 x 8
    sci_ext = 1                     # science extension of an exposure file
    dq_ext = 2                      # data-quality mask extension (omit or -1: no mask)
    ref_use_ext = [1]               # extensions whose WCS define the reference frame
    tag = "MyCam"                   # optional product-name tag (default Grid<H>x<W>)
    job_name = "All"                # optional

Offset structure the standard modes give it: adjacency along both grid axes
(``row`` scans vertically, ``col`` horizontally), a soft polynomial along
``row`` (one per ``col``; ``[params].poly_axis`` picks the other axis) when
``[params].poly_weight`` is set. Instruments with a
non-rectangular chunk geometry, per-pixel wavelength maps or several detectors
per file subclass :class:`~selfcal.instruments.base.Instrument` instead (see
``spherex/adapter.py``).
"""
from __future__ import annotations

import numpy as np

from ..geometry.map_helper import make_grid_chunk_map
from ..models.offset_structure import ChunkAxes
from .base import (Instrument, register_instrument, Job, ChunkMap, DetectorGeometry, JobGeometry,
                   ExposureLayout)


def upsample_chunk_map(det_chunk_map, factor):
    """Replicate each detector pixel into a (factor x factor) block, preserving ids."""
    if factor == 1:
        return det_chunk_map
    return np.kron(det_chunk_map, np.ones((factor, factor), dtype=det_chunk_map.dtype))


def _chunk_grid(inst_cfg):
    c = inst_cfg.get('chunks', [4])
    c = [int(c)] if np.isscalar(c) else [int(v) for v in c]
    return (c[0], c[0]) if len(c) == 1 else (c[0], c[1])


def rect_grid_chunk_map(det_shape, ny, nx):
    """``ny x nx`` rectangular chunks over ``det_shape``; chunk id = row * nx + col."""
    if ny == nx:
        return make_grid_chunk_map(det_shape, ny).astype(np.int32)
    H, W = det_shape
    rows = np.minimum(np.arange(H) * ny // H, ny - 1)
    cols = np.minimum(np.arange(W) * nx // W, nx - 1)
    return (rows[:, None] * nx + cols[None, :]).astype(np.int32)


@register_instrument('grid')
class GridInstrument(Instrument):
    """Rectangular-chunk imager, fully described by the ``[instrument]`` table."""

    capabilities = frozenset()

    def jobs(self, inst_cfg):
        """Return one job for the whole detector, named ``[instrument].job_name`` (default ``All``).

        The name is the job component of every product name, e.g.
        ``cal_<frame_tag>_All<suffix>.h5``."""
        return [Job(name=str(inst_cfg.get('job_name', 'All')))]

    def frame_tag(self, inst_cfg):
        """Return the product-name tag ``<tag>_Chunks<ny>x<nx>``, e.g. ``MyCam_Chunks8x8``.

        ``tag`` defaults to ``Grid<H>x<W>`` from ``detector_shape`` (required
        even when ``tag`` is set) and ``ny x nx`` is the chunk grid of
        ``chunks``: a 2048 x 2048 detector in 8 x 8 chunks without a ``tag``
        gives ``Grid2048x2048_Chunks8x8``."""
        H, W = (int(v) for v in inst_cfg['detector_shape'])
        ny, nx = _chunk_grid(inst_cfg)
        return str(inst_cfg.get('tag', f'Grid{H}x{W}')) + f'_Chunks{ny}x{nx}'

    def exposure_layout(self, inst_cfg):
        """Return the layout of a FITS exposure file that holds one detector.

        The science image is in extension ``sci_ext`` (default 1) and the
        data-quality mask in ``dq_ext`` (omitted or negative: no mask, every
        pixel valid); the detector index is 0, and the WCS of the
        ``ref_use_ext`` extensions (default: ``[sci_ext]``) defines the
        reference frame. There is no header filter and no custom reader: the
        default FITS reader is used."""
        dq = inst_cfg.get('dq_ext', -1)
        return ExposureLayout(
            sci_ext=[int(inst_cfg.get('sci_ext', 1))],
            dq_ext=None if dq is None or int(dq) < 0 else [int(dq)],
            detector_ids=[0],
            ref_use_ext=tuple(int(e) for e in inst_cfg.get('ref_use_ext', [inst_cfg.get('sci_ext', 1)])),
            cache_tag=f"headers_{inst_cfg.get('tag', 'grid')}")

    def detector_geometry(self, inst_cfg, oversample):
        """Return the detector geometry: one ``ny x nx`` rectangular chunk map named ``grid``.

        ``detector_shape`` gives the detector's ``(H, W)`` and ``chunks`` the
        chunk grid (``[ny, nx]``, or one number for a square grid; default
        4 x 4). The ``int32`` map comes from :func:`rect_grid_chunk_map` (chunk
        id ``row * nx + col``) and is replicated onto the reference grid in
        ``oversample x oversample`` blocks (:func:`upsample_chunk_map`). Its
        axes are ``row`` (size ``ny``, scanned vertically) and ``col`` (size
        ``nx``, scanned horizontally); the standard offset block regularises
        along both, and ``row`` is the default group axis of a
        polynomial-basis offset term. There are no aux maps (no wavelength
        map) and no spectral axis."""
        H, W = (int(v) for v in inst_cfg['detector_shape'])
        ny, nx = _chunk_grid(inst_cfg)
        det = rect_grid_chunk_map((H, W), ny, nx)
        axes = ChunkAxes.row_major(('row', 'col'), (ny, nx), ('y', 'x'))
        cm = ChunkMap(name='grid', det=det, grid=upsample_chunk_map(det, int(oversample)), axes=axes,
                      adjacency_axes=('row', 'col'), spectral_axis=None, group_axis='row')
        return DetectorGeometry(shape=(H, W), chunk_maps={'grid': cm}, primary='grid')

    def job_geometry(self, inst_cfg, geom, job):
        """Return uniform validity: every pixel and chunk valid, every weight 1.

        The weights and masks are ``float32`` ones on the detector grid
        (``geom.shape``) and on the reference grid (the shape of the primary
        chunk map's ``grid``); ``chunk_valid`` and ``chunk_valid_strict`` are
        all ``True``. There is no edge taper, and the result does not depend
        on ``inst_cfg`` or ``job``."""
        ones_det = np.ones(geom.shape, dtype=np.float32)
        ones_grid = np.ones(geom.chunk_map.grid.shape, dtype=np.float32)
        n = geom.chunk_map.n_chunks
        return JobGeometry(det_valid_weight=ones_det, grid_valid_weight=ones_grid,
                           chunk_valid=np.ones(n, dtype=bool), chunk_valid_strict=np.ones(n, dtype=bool),
                           det_valid_mask=ones_det, grid_valid_mask=ones_grid)

    def data_unit(self, inst_cfg):
        """Return ``[instrument].unit`` (default: empty), written as the mosaic ``BUNIT``."""
        return str(inst_cfg.get('unit', ''))
