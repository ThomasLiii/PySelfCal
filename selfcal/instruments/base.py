"""The typed geometry the run engine works on.

An instrument (:class:`~selfcal.instruments.contract.Instrument`: ``sc.SPHEREx``, ``sc.Euclid``,
``sc.Camera`` or a subclass of your own) describes its detector with a :class:`DetectorGeometry`:
its shape, the named per-pixel detector maps (a wavelength map for spectral fits; none for a
broadband imager) and what its own functions need. The chunk maps are the model's: each
``sc.ChunkMap(fn, **params)`` of the model (or of the instrument's defaults) is evaluated on that
geometry into a :class:`Chunks`, a partition of the detector with the *axes* of its chunk grid
(:class:`~selfcal.models.offset_structure.ChunkAxes`), so that the offset structure and the jobs
can be expressed generically ("adjacency along *column*", "the chunks whose *subchannel* lies in
150..160"). The engine attaches the evaluated maps to the geometry it hands the model.

* :class:`JobGeometry`: which pixels are valid for a job, and the per-pixel weights of the solve
  and the mosaic (Euclid: an optional edge taper);
* :class:`ExposureLayout`: how a raw exposure file is read (its science and data-quality entries,
  the detectors it holds, an optional header filter and reader).
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import Callable

import numpy as np

from ..models.offset_structure import ChunkAxes

__all__ = ['Chunks', 'DetectorGeometry', 'JobGeometry', 'ExposureLayout']


@dataclass(frozen=True)
class Chunks:
    """A chunk partition of the detector, what a chunk-map function returns: ``det``, the chunk id of
    every detector pixel (``-1`` marks pixels outside every chunk), and ``axes``, the coordinates
    of its chunk grid. ``grid`` is the same partition sampled ``oversample`` times per detector
    pixel along each axis (what the mosaic samples); None: the engine replicates ``det`` into
    blocks. The engine names the map after the model's key (``name``).

    ``adjacency_axes``: the axes along which the standard offset block regularises neighbouring
    chunks by default (a camera grid: ``('row', 'col')``). ``spectral_axis``: the axis along
    which the dispersion runs (SPHEREx: ``'subchannel'``), ``None`` for broadband.
    ``group_axis``: the axis that indexes independent polynomials in the hard poly-basis offset.

    ``renderers``: how the mosaic draws this map's offsets, by name: each
    ``make(chunks, jobgeom) -> renderer`` returns the ``renderer(chunk_map, offsets) -> image`` of
    one job (a renderer that fills the offsets the mosaic zeroed, a chunk below
    ``Coadd(min_chunk_coverage)``, from the frame's other chunks says so with a true attribute
    ``fills_invalid``). ``render`` names the one used unless the offset term names another
    (``Offsets(render=...)``); None: constant over each chunk. ``extra``: what the map's own
    functions need later (a detector map read from it, its renderers); the engine never reads it."""
    det: np.ndarray
    axes: ChunkAxes | None = None
    _: KW_ONLY
    grid: np.ndarray | None = None
    name: str = ''
    adjacency_axes: tuple[str, ...] = ()
    spectral_axis: str | None = None
    group_axis: str | None = None
    renderers: dict[str, Callable] = field(default_factory=dict)
    render: str | None = None
    extra: dict = field(default_factory=dict)

    @property
    def n_chunks(self) -> int:
        """The number of chunks: the largest chunk id in ``det`` plus one."""
        return int(self.det.max()) + 1

    @classmethod
    def rectangles(cls, shape, chunks, axes=('row', 'col'), adjacency=None, group_axis=None) -> Chunks:
        """``ny x nx`` rectangular chunks over a detector of ``shape`` (chunk id ``row * nx + col``;
        sides differ by at most one pixel), at detector resolution. ``axes`` names the chunk rows
        and columns; the chunks are smoothed along both unless ``adjacency`` says otherwise, and
        ``group_axis`` (default the first axis) indexes a polynomial basis's groups."""
        from .grid import rect_grid_chunk_map
        ny, nx = (int(v) for v in chunks)
        det = rect_grid_chunk_map(tuple(int(v) for v in shape), ny, nx)
        axes = tuple(axes)
        return cls(det, ChunkAxes.row_major(axes, (ny, nx), ('y', 'x')),
                   adjacency_axes=axes if adjacency is None else tuple(adjacency), spectral_axis=None,
                   group_axis=group_axis or axes[0])


@dataclass(frozen=True)
class DetectorGeometry:
    """Detector-level geometry built once per run.

    ``shape``: the detector's pixels; ``oversample``: how many times per pixel the mosaic samples
    the detector along each axis (the chunk maps' ``grid``). ``aux``: named per-pixel maps on the
    detector grid (SPHEREx: ``{'BC': band centre, 'BW': band width}``); ``wavelength_key`` /
    ``width_key`` say which of them are the wavelength and its width (``None`` for a broadband
    instrument). ``extra`` is instrument-private state its own functions need (a chunk-map
    function, a renderer, a hook; never read by the engine). ``chunk_maps``: the model's maps by
    name once the engine has evaluated them on this geometry, ``primary`` the first one (the
    offset terms' default map, the jobs' and the mosaic's)."""
    shape: tuple[int, int]
    oversample: int = 1
    aux: dict[str, np.ndarray] = field(default_factory=dict)
    wavelength_key: str | None = None
    width_key: str | None = None
    extra: dict = field(default_factory=dict)
    chunk_maps: dict[str, Chunks] = field(default_factory=dict)
    primary: str | None = None

    @property
    def chunk_map(self) -> Chunks:
        """The primary chunk map ``chunk_maps[primary]``, used by offset terms that name no map."""
        if self.primary is None:
            raise ValueError("the geometry has no chunk maps yet: the model's (Model(chunk_maps=...)) or the "
                             "instrument's defaults are evaluated by the run engine")
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
    (the solve's set, which may overlap neighbouring jobs for stitching, and the
    coadd's)."""
    det_valid_weight: np.ndarray
    grid_valid_weight: np.ndarray
    chunk_valid: np.ndarray | None = None
    chunk_valid_strict: np.ndarray | None = None


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
