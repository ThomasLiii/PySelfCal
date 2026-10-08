"""The typed geometry an instrument gives the run engine.

An instrument (:class:`~selfcal.instruments.contract.Instrument`: ``sc.SPHEREx``, ``sc.Euclid``,
``sc.Camera`` or a subclass of your own) describes its detector with these types:

* :class:`ChunkMap`: a chunk partition of the detector, with the *axes* of its chunk grid
  (:class:`~selfcal.models.offset_structure.ChunkAxes`), so that the offset structure can be
  expressed generically ("adjacency along *column*", "degree-2 polynomial along *subchannel* per
  *column*");
* :class:`DetectorGeometry`: the chunk maps by name (one primary), and the named per-pixel
  detector maps (a wavelength map for spectral fits; none for a broadband imager);
* :class:`JobGeometry`: which pixels are valid for a job, and the edge-taper weights of the solve
  and the mosaic;
* :class:`ExposureLayout`: how a raw exposure file is read (its science and data-quality entries,
  the detectors it holds, an optional header filter and reader).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..models.offset_structure import ChunkAxes

__all__ = ['ChunkMap', 'DetectorGeometry', 'JobGeometry', 'ExposureLayout']


@dataclass(frozen=True)
class ChunkMap:
    """A chunk partition of the detector at detector resolution (``det``) and
    sampled ``oversample`` times per detector pixel along each axis (``grid``,
    what the mosaic samples), with the axes of its chunk grid. ``-1`` marks
    pixels outside every chunk.

    ``adjacency_axes``: the axes along which the standard offset block
    regularises neighbouring chunks by default (SPHEREx: ``('column',)``; a
    camera grid: ``('row', 'col')``). ``spectral_axis``: the axis along which
    the dispersion runs (SPHEREx: ``'subchannel'``), ``None`` for broadband.
    ``group_axis``: the axis that indexes independent polynomials in the hard
    poly-basis offset (SPHEREx: ``'column'``)."""
    name: str
    det: np.ndarray
    grid: np.ndarray
    axes: ChunkAxes | None = None
    adjacency_axes: tuple[str, ...] = ()
    spectral_axis: str | None = None
    group_axis: str | None = None

    @property
    def n_chunks(self) -> int:
        """The number of chunks: the largest chunk id in ``det`` plus one."""
        return int(self.det.max()) + 1

    @classmethod
    def rectangles(cls, name, shape, chunks, axes=('row', 'col'), adjacency=None, group_axis=None) -> ChunkMap:
        """``ny x nx`` rectangular chunks over a detector of ``shape`` (chunk id ``row * nx + col``;
        sides differ by at most one pixel), at detector resolution (``grid`` is ``det``:
        ``selfcal.instruments.contract.Geometry`` oversamples it). ``axes`` names the chunk rows and
        columns; the chunks are smoothed along both unless ``adjacency`` says otherwise, and
        ``group_axis`` (default the first axis) indexes a polynomial basis's groups."""
        from .grid import rect_grid_chunk_map
        ny, nx = (int(v) for v in chunks)
        det = rect_grid_chunk_map(tuple(int(v) for v in shape), ny, nx)
        axes = tuple(axes)
        return cls(name=name, det=det, grid=det, axes=ChunkAxes.row_major(axes, (ny, nx), ('y', 'x')),
                   adjacency_axes=axes if adjacency is None else tuple(adjacency), spectral_axis=None,
                   group_axis=group_axis or axes[0])


@dataclass(frozen=True)
class DetectorGeometry:
    """Detector-level geometry built once per run.

    ``chunk_maps``: by name; ``primary`` names the one the standard offset
    block and the mosaic use. ``aux``: named per-pixel maps on the detector
    grid (SPHEREx: ``{'BC': band centre, 'BW': band width}``);
    ``wavelength_key`` / ``width_key`` say which of them are the wavelength
    and its width (``None`` for a broadband instrument). ``extra`` is
    instrument-private state its own renderers and hooks need (never read by
    the engine)."""
    shape: tuple[int, int]
    chunk_maps: dict[str, ChunkMap]
    primary: str = 'primary'
    aux: dict[str, np.ndarray] = field(default_factory=dict)
    wavelength_key: str | None = None
    width_key: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def chunk_map(self) -> ChunkMap:
        """The primary chunk map ``chunk_maps[primary]``, used by offset terms that name no map."""
        return self.chunk_maps[self.primary]

    @property
    def aux_keys(self) -> tuple[str, ...]:
        """The names of the ``aux`` maps, in the order :attr:`aux_list` returns the maps."""
        return tuple(self.aux)

    @property
    def aux_list(self) -> list:
        """The aux maps in key order (the positional form the core consumes)."""
        return [self.aux[k] for k in self.aux]


@dataclass(frozen=True)
class JobGeometry:
    """Per-job validity + weights. ``det_valid_weight`` (detector grid) is the
    solve's per-pixel valid weight; ``grid_valid_weight`` (the detector grid
    sampled ``oversample`` times per pixel along each axis) the mosaic's;
    ``chunk_valid`` / ``chunk_valid_strict`` are per chunk of the primary map
    (the padded set overlaps neighbouring jobs for stitching)."""
    det_valid_weight: np.ndarray
    grid_valid_weight: np.ndarray
    chunk_valid: np.ndarray | None = None
    chunk_valid_strict: np.ndarray | None = None
    det_valid_mask: np.ndarray | None = None
    grid_valid_mask: np.ndarray | None = None


@dataclass(frozen=True)
class ExposureLayout:
    """How the reprojection stage reads a raw exposure file.

    ``sci_ext`` / ``dq_ext``: the science and data-quality entries of each
    detector frame the file holds (``dq_ext=None``: no mask, all pixels
    valid) — FITS extension numbers for the default reader, any integer the
    instrument's ``reader`` understands otherwise (slice indices of a cube,
    detector numbers, ...); ``detector_ids``: the detector index each entry
    yields; ``ref_use_ext``: the entries whose WCS define the reference frame;
    ``reader``: ``reader(path, sci_ext, dq_ext, header_only=False) ->
    selfcal.io.frames.ExposureData`` (values, WCS header + metadata, mask,
    extra per-pixel planes, detector coordinates; ``None`` = FITS
    extensions); ``header_predicate`` keeps an exposure iff it returns True on
    the FITS header of ``header_ext`` (``header_keys`` are the keys it reads,
    so the filter can be cached); ``cache_tag`` names that cache."""
    sci_ext: list[int]
    dq_ext: list[int] | None
    detector_ids: list[int]
    ref_use_ext: tuple[int, ...] = (1,)
    header_predicate: Callable | None = None
    header_keys: tuple[str, ...] = ()
    header_ext: int = 1
    cache_tag: str = 'exposures'
    default_ignore_bits: tuple[int, ...] = ()
    reader: Callable | None = None
