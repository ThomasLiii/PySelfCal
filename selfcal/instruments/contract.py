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

The engine receives the instrument object itself; no registry is needed. Under the hood a
built-in instrument lowers to the engine's registered instrument and its ``[instrument]``
table (:meth:`Instrument.engine`), the form the TOML run configs use.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, replace
from typing import Any

import numpy as np

from ..config.base import Config, ConfigError
from ..models.offset_structure import ChunkAxes
from . import base as engine
from .grid import upsample_chunk_map

__all__ = ['Instrument', 'Job', 'Geometry', 'ChunkMap', 'ChunkAxes', 'JobGeometry', 'ExposureLayout']

ChunkMap = engine.ChunkMap
JobGeometry = engine.JobGeometry
ExposureLayout = engine.ExposureLayout


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
    return engine.DetectorGeometry(shape=tuple(int(v) for v in shape), chunk_maps=out,
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

    def engine_job(self):
        return engine.Job(name=self.name, kind=self.kind, value=self.value)


class Instrument(Config):
    """Base of the instrument settings (see the module docstring).

    A subclass implements :meth:`geometry`; it may override :meth:`layout` (how raw exposure
    files are read), :meth:`default_jobs`, :meth:`job_geometry` (valid pixels and weights of a
    job), :meth:`frame_variables` (per-frame data variables). A ``tag`` setting names its
    products (default: the class name); ``unit`` is the mosaic's ``BUNIT``.

    It may also define the engine's optional hooks, which then reach the run: ``offset_renderer(geom,
    jobgeom, map_name=None, render=None)`` (a ``(chunk_map, offsets) -> grid`` renderer for the
    mosaic; None: constant over each chunk), ``aux_coadds(geom)`` and ``finalize_mosaic(geom,
    mosaicker, maps, sigma)`` (per-pixel maps to coadd alongside the data, and their labelling),
    and ``coefficient_catalog()`` (named coefficients, ``sc.catalog(name)``); see
    :class:`selfcal.instruments.base.Instrument`.
    """

    # ---- the contract -------------------------------------------------------------------
    def geometry(self, oversample=1) -> engine.DetectorGeometry:
        """The detector geometry: chunk maps with their axes, per-pixel maps (:func:`Geometry`),
        the detector-plane maps sampled ``oversample`` times per pixel. A new instrument
        implements it; a built-in one (``sc.Camera``, ``sc.SPHEREx``, ``sc.Euclid``) returns the
        run engine's."""
        inst, table = self.engine(())
        if isinstance(inst, _ContractAdapter):
            raise NotImplementedError(f"{type(self).__name__}: a subclass of sc.Instrument implements "
                                      f"geometry(oversample)")
        if isinstance(inst, str):
            inst = engine.get_instrument(inst)
        return inst.detector_geometry(table, oversample)

    def layout(self) -> engine.ExposureLayout:
        """How a raw exposure file is read (default: a FITS file, science in extension 1, no mask)."""
        return engine.ExposureLayout(sci_ext=[1], dq_ext=None, detector_ids=[0], ref_use_ext=(1,),
                                     cache_tag=f'headers_{self.product_tag}')

    def default_jobs(self) -> tuple:
        """The jobs a run makes when the action names none (default: one, ``All``)."""
        return (Job('All'),)

    def job_geometry(self, geom, job) -> engine.JobGeometry:
        """Valid pixels and weights of ``job`` (default: every pixel of a chunk, weight 1)."""
        cm = geom.chunk_map
        det = (np.asarray(cm.det) >= 0).astype(np.float32)
        grid = (np.asarray(cm.grid) >= 0).astype(np.float32)
        n = cm.n_chunks
        return engine.JobGeometry(det_valid_weight=det, grid_valid_weight=grid,
                                  chunk_valid=np.ones(n, dtype=bool), chunk_valid_strict=np.ones(n, dtype=bool),
                                  det_valid_mask=det, grid_valid_mask=grid)

    def frame_variables(self, frames) -> dict:
        """Per-frame data variables ``{name: (n_frames,) array}`` (default: ``exposure`` and
        ``detector``, from the frame file names)."""
        return engine.Instrument.frame_variables(None, frames)

    def frame_variable_names(self) -> tuple:
        """The names :meth:`frame_variables` provides (checked before any frame is read)."""
        return ('exposure', 'detector')

    def default_ignore_flags(self) -> tuple:
        """The data-quality bits that do not flag a pixel unless a recipe says otherwise."""
        return tuple(getattr(self.layout(), 'default_ignore_bits', ()) or ())

    # ---- naming --------------------------------------------------------------------------
    @property
    def product_tag(self) -> str:
        """The products' tag (``cal_<tag>_<job><suffix>.h5``)."""
        return str(getattr(self, 'tag', None) or type(self).__name__)

    # ---- lowering -------------------------------------------------------------------------
    def engine(self, jobs):
        """``(instrument, table)`` the run engine takes: an engine instrument (a registered name or
        an object) and the ``[instrument]`` table that selects ``jobs``."""
        return _ContractAdapter(self), {'name': type(self).__name__, 'jobs': tuple(jobs)}

    def check_jobs(self, jobs):
        """Raise :class:`~selfcal.config.base.ConfigError` for a job this instrument cannot run."""
        for j in jobs:
            if not isinstance(j, Job):
                raise ConfigError(f"{type(self).__name__}: a job is a Job, got {j!r}")


class _ContractAdapter(engine.Instrument):
    """The engine's interface over an :class:`Instrument` settings object (a user subclass)."""

    def __init__(self, inst):
        self.inst = inst
        self.name = type(inst).__name__
        self.capabilities = frozenset(getattr(inst, 'capabilities', ()))

    def __reduce__(self):
        return (_ContractAdapter, (self.inst,))

    def jobs(self, inst_cfg):
        return [j.engine_job() for j in (inst_cfg.get('jobs') or self.inst.default_jobs())]

    def frame_tag(self, inst_cfg):
        return self.inst.product_tag

    def exposure_layout(self, inst_cfg):
        return self.inst.layout()

    def detector_geometry(self, inst_cfg, oversample):
        return self.inst.geometry(oversample)

    def job_geometry(self, inst_cfg, geom, job):
        return self.inst.job_geometry(geom, job)

    def frame_variable_names(self, inst_cfg):
        return tuple(self.inst.frame_variable_names())

    def frame_variables(self, frames, inst_cfg=None):
        return dict(self.inst.frame_variables(frames))

    def data_unit(self, inst_cfg):
        return str(getattr(self.inst, 'unit', '') or '')

    # the engine's optional hooks, when the settings object defines them
    def offset_renderer(self, inst_cfg, geom, jobgeom, map_name=None, render=None):
        hook = getattr(self.inst, 'offset_renderer', None)
        return hook(geom, jobgeom, map_name=map_name, render=render) if hook else None

    def aux_coadds(self, geom):
        hook = getattr(self.inst, 'aux_coadds', None)
        return hook(geom) if hook else None

    def finalize_mosaic(self, geom, mosaicker, maps, sigma):
        hook = getattr(self.inst, 'finalize_mosaic', None)
        if hook:
            hook(geom, mosaicker, maps, sigma)

    def coefficient_catalog(self):
        hook = getattr(self.inst, 'coefficient_catalog', None)
        return dict(hook()) if hook else {}
