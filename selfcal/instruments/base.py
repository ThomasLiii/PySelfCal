"""The Instrument contract — the ONE place a telescope enters the pipeline.

The numerical layers (``selfcal.core``, ``models``, ``geometry``, ``io``) take
plain arrays and callables and never import this package. The run engine
(``selfcal.run``) drives a calibration entirely through the
:class:`Instrument` interface below plus the ``CalMode`` recipe interface; it
never names a telescope. A new instrument is one subclass registered with
:func:`register_instrument` (or published through the ``selfcal.instruments``
entry-point group of any installed package); a new telescope never touches
the engine. Users without custom geometry need no code at all: the built-in
``grid`` instrument (``selfcal.instruments.grid``) is configured entirely from
the ``[instrument]`` table.

What an instrument provides
---------------------------
* the **run layout**: how a config selects *jobs* (SPHEREx: spectral channels
  or subchannel windows; a camera: one job), the product-name tag, and how raw
  exposures are read (:class:`ExposureLayout`);
* the **detector geometry** (:class:`DetectorGeometry`): one or more chunk
  maps, each with the *axes* of its chunk grid
  (:class:`~selfcal.models.offset_structure.ChunkAxes`) so the offset structure
  can be expressed generically ("adjacency along *column*", "degree-2
  polynomial along *subchannel* per *column*"), plus named per-pixel aux maps
  (a wavelength map for spectral fits; ``{}`` for broadband);
* the **per-job geometry** (:class:`JobGeometry`): which pixels are valid for
  the job and the edge-taper weights the solve and the mosaic use;
* **data variables** (:mod:`selfcal.models.variables`): detector maps
  (``DetectorGeometry.aux``) and per-frame values (:meth:`frame_variables`:
  time, filter, angle, ...) that the model's functions read by name, beside
  the model's own variables and the built-ins (detector / sky coordinates);
* an optional **exposure reader** (``ExposureLayout.reader``) for raw data
  that is not a FITS image with a WCS, extra per-pixel planes, or detector
  coordinates on a focal plane;
* optional **hooks**: a smooth chunk->grid offset renderer for the mosaic,
  per-pixel maps to coadd with the data, a mosaic finaliser, a catalogue of
  named coefficients, post-calibration hooks (SPHEREx: the zodi anchor), the
  data unit, and a rarely-run geometry precompute.

The required surface is the five abstract methods; everything else has a
default. ``spherex/adapter.py`` is the full reference implementation and
``grid.py`` the minimal one: both read the table as settings whose methods
hold the instrument's code (``spherex/settings.py``'s
:class:`~selfcal.instruments.spherex.settings.SPHEREx`, ``camera.py``'s
:class:`~selfcal.instruments.camera.Camera`).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..models.offset_structure import ChunkAxes

__all__ = ['Instrument', 'register_instrument', 'get_instrument', 'available_instruments',
           'Job', 'ChunkMap', 'DetectorGeometry', 'JobGeometry', 'ExposureLayout']

_INSTRUMENT_REGISTRY: dict[str, type] = {}
_ENTRY_POINT_GROUP = 'selfcal.instruments'


def register_instrument(name: str):
    """Class decorator: make the ``Instrument`` subclass selectable as
    ``[instrument].name = "<name>"``."""
    def deco(cls):
        cls.name = name
        _INSTRUMENT_REGISTRY[name] = cls
        return cls
    return deco


def _load_entry_points():
    """Instruments published by other installed packages
    (``[project.entry-points."selfcal.instruments"]`` in their pyproject)."""
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group=_ENTRY_POINT_GROUP)
    except Exception:
        return
    for ep in eps:
        if ep.name not in _INSTRUMENT_REGISTRY:
            try:
                cls = ep.load()
            except Exception as e:                       # a broken plugin must not break the built-ins
                import logging
                logging.getLogger(__name__).warning(f"instrument entry point {ep.name!r} failed to load: {e}")
                continue
            cls.name = ep.name
            _INSTRUMENT_REGISTRY[ep.name] = cls


def get_instrument(name: str) -> "Instrument":
    """The registered instrument called ``name``: a built-in (registered when
    ``selfcal.instruments`` is imported) or an entry-point plugin."""
    if name not in _INSTRUMENT_REGISTRY:
        _load_entry_points()
    if name not in _INSTRUMENT_REGISTRY:
        raise ValueError(f"unknown instrument {name!r}; available: {available_instruments()}")
    return _INSTRUMENT_REGISTRY[name]()


def available_instruments() -> list[str]:
    """Return the sorted names of all registered instruments, plugins included.

    The ``selfcal.instruments`` entry points are loaded first (a plugin that
    fails to load is skipped with a logged warning), so the list holds the
    built-ins, any instrument registered in this process with
    :func:`register_instrument`, and those published by installed packages.
    Each name is a valid ``[instrument].name`` and argument of
    :func:`get_instrument`."""
    _load_entry_points()
    return sorted(_INSTRUMENT_REGISTRY)


# ---------------------------------------------------------------------------
# Typed results
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Job:
    """One unit of the run's job loop: a name (a component of every product
    file name) and an instrument-private selection (``kind``/``value``)."""
    name: str
    kind: str = 'all'
    value: object = None


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
    the engine or a mode)."""
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
    sampled ``oversample`` times per pixel along each axis) the mosaic's; ``chunk_valid`` / ``chunk_valid_strict`` are per chunk of the
    primary map (the padded set overlaps neighbouring jobs for stitching)."""
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


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------
class Instrument(ABC):
    """Subclass, decorate with ``@register_instrument("<name>")``, implement
    the five abstract methods; override the hooks you need."""

    name: str = None
    capabilities: frozenset = frozenset()       # e.g. {'wavelength'}: tags a mode can ``require``

    # ---- run layout -----------------------------------------------------------
    @abstractmethod
    def jobs(self, inst_cfg) -> list[Job]:
        """Expand the [instrument] table into the run's jobs."""

    @abstractmethod
    def frame_tag(self, inst_cfg) -> str:
        """The product-name component: ``cal_<frame_tag>_<job><suffix>.h5``."""

    @abstractmethod
    def exposure_layout(self, inst_cfg) -> ExposureLayout:
        """How raw exposures are read (``reproject`` task)."""

    # ---- geometry ---------------------------------------------------------------
    @abstractmethod
    def detector_geometry(self, inst_cfg, oversample) -> DetectorGeometry:
        """Chunk maps (with axes), aux maps, edges — built once per run. Must NOT
        include adjacency or constraints: those are the mode's."""

    @abstractmethod
    def job_geometry(self, inst_cfg, geom: DetectorGeometry, job: Job) -> JobGeometry:
        """Validity + weights of one job."""

    # ---- hooks (defaults) -------------------------------------------------------
    def offset_renderer(self, inst_cfg, geom, jobgeom, map_name=None, render=None) -> Callable | None:
        """``(chunk_map, offsets) -> grid`` renderer the mosaic uses to draw the
        offsets of chunk map ``map_name`` (None = the primary map); ``render``
        names one of the instrument's renderers when the model asks for a
        specific one. ``None`` = block-constant (``chunk_to_det``)."""
        return None

    def frame_variable_names(self, inst_cfg) -> tuple[str, ...]:
        """Names of the frame variables :meth:`frame_variables` provides (used to
        check a model before any frame is read)."""
        return ('exposure', 'detector')

    def frame_variables(self, frames, inst_cfg=None) -> dict:
        """Per-frame data variables the instrument defines: ``{name: (n_frames,)
        array}`` — one value per frame (time, filter, angle, temperature, ...),
        broadcast over the frame's observations and usable by any model
        function and as an offset grouping. Default: ``exposure`` and
        ``detector``, the indices in each frame's file name. Override (keeping
        the defaults) to read header keywords (``selfcal.io.frames.
        frame_header_values``), tables or anything else."""
        from ..io.reproj import parse_reproj_basename
        idx = np.array([parse_reproj_basename(f) for f in frames], dtype=np.int64).reshape(-1, 2)
        return {'exposure': idx[:, 0], 'detector': idx[:, 1]}

    def frame_groups(self, frames) -> dict:
        """Named per-frame groupings a grouped offset term can share an offset
        over: ``{name: int array (n_frames,)}``. Default: the instrument's
        integer frame variables (``detector`` = each frame's detector index,
        ``exposure``)."""
        return {k: v for k, v in self.frame_variables(frames).items()
                if np.asarray(v).dtype.kind in 'iu'}

    def hooks(self) -> dict:
        """Named per-frame hook factories a config can select in ``[hooks]``:
        ``{name: factory(**params) -> callable(FrameContext) -> sub_data}``."""
        return {}

    def aux_coadds(self, geom) -> tuple | None:
        """The ``(band centre, band width)`` detector-grid maps whose per-pixel
        weighted mean / std the mosaic coadds alongside the data (spectral
        instruments); ``None`` = nothing to coadd."""
        return None

    def finalize_mosaic(self, geom, mosaicker, maps, sigma) -> None:
        """Label / append instrument products on the finished mosaic (called
        only when the mode asks for a full mosaic and ``aux_coadds`` is set)."""

    def coefficient_catalog(self) -> dict[str, Callable]:
        """Named sky-term coefficients: ``{name: factory(**overrides) -> Coefficient}``,
        referred to as ``coefficient = { catalog = "<name>" }``."""
        return {}

    def postcal_hooks(self, cfg) -> list[Callable]:
        """Callables ``hook(ctx, job, cal_path, mosaic_path)`` run after each job's
        mosaic (``cfg`` is the whole run config)."""
        return []

    def data_unit(self, inst_cfg) -> str:
        """Unit of the calibrated data (mosaic ``BUNIT``)."""
        return ''

    def precompute(self, inst_cfg) -> None:
        """Rarely-run geometry generator (task ``precompute``)."""
        raise NotImplementedError(f"instrument {self.name!r} has nothing to precompute")
