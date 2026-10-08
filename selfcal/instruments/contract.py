"""Instruments as settings: what took the data, as an object the run's actions take.

``sc.SPHEREx(4)``, ``sc.Euclid(band="Y")`` and ``sc.Camera((2048, 2048), chunks=(8, 8))`` are
the built-in instruments; their jobs are typed objects (``spherex.channel(17)``). A new
telescope is a frozen-dataclass subclass of :class:`Instrument` whose fields are its settings
and which implements :meth:`Instrument.geometry` (its chunk maps); everything else has a
default::

    @dataclass(frozen=True, kw_only=True)
    class Owl(sc.Instrument):
        chunks: int = 8
        tag: str = "Owl"
        unit: str = "e-/s"

        def geometry(self, oversample):
            grid = sc.ChunkMap.rectangles("grid", (2048, 2048), (self.chunks, self.chunks))
            return sc.Geometry((2048, 2048), oversample, maps=[grid])

The run engine receives the instrument object itself and calls the methods below; no registry
is needed.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, replace
from typing import Any

import numpy as np

from ..config.base import Config, ConfigError
from ..models.offset_structure import ChunkAxes
from . import base
from .grid import upsample_chunk_map

__all__ = ['Instrument', 'Job', 'Geometry', 'ChunkMap', 'ChunkAxes', 'JobGeometry', 'ExposureLayout']

ChunkMap = base.ChunkMap
JobGeometry = base.JobGeometry
ExposureLayout = base.ExposureLayout


def Geometry(shape, oversample, maps, *, primary=None, aux=None, wavelength=None, width=None):
    """The detector geometry: the chunk ``maps`` (:class:`ChunkMap`, at detector resolution; their
    oversampled grids are made here), the ``primary`` one (default the first), per-pixel ``aux``
    maps ``{name: array}``, and which of them are the ``wavelength`` and its ``width``."""
    maps = list(maps)
    if not maps:
        raise ConfigError("Geometry: at least one chunk map")
    out = {}
    for cm in maps:
        out[cm.name] = replace(cm, grid=upsample_chunk_map(np.asarray(cm.det), int(oversample)))
    return base.DetectorGeometry(shape=tuple(int(v) for v in shape), chunk_maps=out,
                                   primary=primary or maps[0].name, aux=dict(aux or {}),
                                   wavelength_key=wavelength, width_key=width)


@dataclass(frozen=True)
class Job(Config):
    """One unit of a run: a ``name`` (part of every product name) and an instrument-private
    selection (``kind``, ``value``)."""
    name: str = 'All'
    _: KW_ONLY
    kind: str = 'all'
    value: Any = None

    def __repr__(self):
        if self.kind == 'channels':                 # SPHEREx
            chans = tuple(self.value)
            return f"spherex.channel({chans[0]})" if len(chans) == 1 else f"spherex.group{chans!r}"
        if self.kind == 'window':
            lo, hi = self.value
            return f"spherex.window({self.name!r}, subchannels=range({lo}, {hi}))"
        return f"Job({self.name!r})" if self.kind == 'all' and self.value is None else \
            f"Job({self.name!r}, kind={self.kind!r}, value={self.value!r})"


class Instrument(Config):
    """Base of the instrument settings (see the module docstring): the contract the run engine
    calls.

    A subclass implements :meth:`geometry`; it may override :meth:`layout` (how raw exposure
    files are read), :meth:`default_jobs`, :meth:`job_geometry` (valid pixels and weights of a
    job), :meth:`frame_variables` (per-frame data variables) and the mosaic's hooks
    (:meth:`offset_renderer`, :meth:`aux_coadds`, :meth:`finalize_mosaic`), and name its
    coefficients (:meth:`coefficient_catalog`) and the data files its geometry reads
    (:meth:`geometry_files`). A ``tag`` setting names its products (default: the class name);
    ``unit`` is the mosaic's ``BUNIT``.
    """

    #: The unit of the calibrated data, the mosaic's ``BUNIT`` (a constant here, or a setting).
    unit = ''

    # ---- the contract -------------------------------------------------------------------
    def geometry(self, oversample=1) -> base.DetectorGeometry:
        """The detector geometry: chunk maps with their axes, per-pixel maps (:func:`Geometry`),
        the detector-plane maps sampled ``oversample`` times per pixel. A new instrument
        implements it, as the built-in ones (``sc.Camera``, ``sc.SPHEREx``, ``sc.Euclid``) do."""
        raise NotImplementedError(f"{type(self).__name__}: a subclass of sc.Instrument implements "
                                  f"geometry(oversample)")

    def layout(self) -> base.ExposureLayout:
        """How a raw exposure file is read (default: a FITS file, science in extension 1, no mask)."""
        return base.ExposureLayout(sci_ext=[1], dq_ext=None, detector_ids=[0], ref_use_ext=(1,),
                                     cache_tag=f'headers_{self.product_tag}')

    def default_jobs(self) -> tuple:
        """The jobs a run makes when the action names none (default: one, ``All``)."""
        return (Job('All'),)

    def job_geometry(self, geom, job) -> base.JobGeometry:
        """Valid pixels and weights of ``job`` (default: every pixel of a chunk, weight 1)."""
        cm = geom.chunk_map
        det = (np.asarray(cm.det) >= 0).astype(np.float32)
        grid = (np.asarray(cm.grid) >= 0).astype(np.float32)
        n = cm.n_chunks
        return base.JobGeometry(det_valid_weight=det, grid_valid_weight=grid,
                                  chunk_valid=np.ones(n, dtype=bool), chunk_valid_strict=np.ones(n, dtype=bool),
                                  det_valid_mask=det, grid_valid_mask=grid)

    def frame_variables(self, frames) -> dict:
        """Per-frame data variables ``{name: (n_frames,) array}``: one value per frame (time,
        filter, angle, temperature, ...), usable by any model function and as an offset grouping.
        Default: ``exposure`` and ``detector``, the indices in each frame's file name. Override
        (keeping the defaults) to read header keywords (``selfcal.io.frames.frame_header_values``),
        tables or anything else."""
        from ..io.reproj import parse_reproj_basename
        idx = np.array([parse_reproj_basename(f) for f in frames], dtype=np.int64).reshape(-1, 2)
        return {'exposure': idx[:, 0], 'detector': idx[:, 1]}

    def frame_variable_names(self) -> tuple:
        """The names :meth:`frame_variables` provides (checked before any frame is read)."""
        return ('exposure', 'detector')

    def frame_groups(self, frames) -> dict:
        """Named per-frame groupings a grouped offset term can share an offset over:
        ``{name: integer array (n_frames,)}``. Default: the integer-valued :meth:`frame_variables`
        (``detector``: each frame's detector; ``exposure``)."""
        return {k: v for k, v in self.frame_variables(frames).items() if np.asarray(v).dtype.kind in 'iu'}

    def offset_renderer(self, geom, jobgeom, map_name=None, render=None):
        """The ``(chunk_map, offsets) -> grid`` function the mosaic draws the offsets of chunk map
        ``map_name`` (None: the primary map) with, for one job; ``render`` names one of the
        instrument's renderers when the model asks for one. Default None: constant over each chunk."""
        return None

    def aux_coadds(self, geom):
        """The ``(band centre, band width)`` detector maps whose per-pixel weighted mean and std the
        mosaic coadds with the data (``Coadd(instrument_maps=True)``); default None: nothing."""
        return None

    def finalize_mosaic(self, geom, mosaicker, maps, sigma):
        """Label or complete the mosaic's instrument maps once the coadd is done (only when
        :meth:`aux_coadds` gives some)."""

    def coefficient_catalog(self) -> dict:
        """Named sky coefficients ``{name: factory(**overrides) -> Coefficient}`` (``sc.catalog(name)``);
        default none."""
        return {}

    def geometry_files(self) -> tuple:
        """The data files :meth:`geometry` reads, beyond the settings (default none): a geometry kept
        by the run engine is built again when one of them changes."""
        return ()

    def default_ignore_flags(self) -> tuple:
        """The data-quality bits that do not flag a pixel unless a recipe says otherwise."""
        return tuple(getattr(self.layout(), 'default_ignore_bits', ()) or ())

    # ---- naming --------------------------------------------------------------------------
    @property
    def product_tag(self) -> str:
        """The products' tag (``cal_<tag>_<job><suffix>.h5``)."""
        return str(getattr(self, 'tag', None) or type(self).__name__)

    def check_jobs(self, jobs):
        """Raise :class:`~selfcal.config.base.ConfigError` for a job this instrument cannot run."""
        for j in jobs:
            if not isinstance(j, Job):
                raise ConfigError(f"{type(self).__name__}: a job is a Job, got {j!r}")
