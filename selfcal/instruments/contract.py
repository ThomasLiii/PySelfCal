"""Instruments as settings: what took the data, as an object the run's actions take.

``sc.SPHEREx(4)``, ``sc.Euclid(band="Y")`` and ``sc.Camera((2048, 2048), chunks=(8, 8))`` are
the built-in instruments. A new telescope is a frozen-dataclass subclass of :class:`Instrument`
whose fields are its settings and which implements :meth:`Instrument.geometry` (its detector:
the shape, the per-pixel maps); everything else has a default::

    @dataclass(frozen=True, kw_only=True)
    class Owl(sc.Instrument):
        tag: str = "Owl"
        unit: str = "e-/s"

        def geometry(self, oversample):
            return sc.Geometry((2048, 2048), oversample)

The chunk maps an offset model lives on are the model's (``sc.Model(chunk_maps={...})``), or the
defaults an instrument offers (:meth:`Instrument.default_chunk_maps`); the jobs of a run
(:class:`Job`) name their chunks on the primary map. The run engine receives the instrument
object itself and calls the methods below; no registry is needed.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

import numpy as np

from ..config.base import Config, ConfigError, FrozenDict
from ..models.offset_structure import ChunkAxes
from . import base

__all__ = ['Instrument', 'Job', 'Geometry', 'Chunks', 'ChunkAxes', 'JobGeometry', 'ExposureLayout', 'chunk_window',
           'check_job']

Chunks = base.Chunks
JobGeometry = base.JobGeometry
ExposureLayout = base.ExposureLayout


def Geometry(shape, oversample=1, *, aux=None, wavelength=None, width=None, extra=None):
    """The detector geometry: its ``shape``, the mosaic's ``oversample`` (samples per pixel along
    each axis), per-pixel ``aux`` maps ``{name: array}``, which of them are the ``wavelength`` and
    its ``width``, and ``extra``, what the instrument's own functions need (never read by the
    engine). The chunk maps are added by the engine from the model."""
    return base.DetectorGeometry(shape=tuple(int(v) for v in shape), oversample=int(oversample), aux=dict(aux or {}),
                                 wavelength_key=wavelength, width_key=width, extra=dict(extra or {}))


def _ranges(value, what):
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ConfigError(f"{what}: a mapping {{axis: range(lo, hi)}}, got {value!r}")
    out = {}
    for axis, r in value.items():
        if not isinstance(r, range) or r.step != 1 or len(r) < 1:
            raise ConfigError(f"{what}: axis {axis!r} takes a range of consecutive chunk indices, got {r!r}")
        out[str(axis)] = r
    return FrozenDict(out)


@dataclass(frozen=True)
class Job(Config):
    """One unit of a run: a ``name`` (part of every product name) and the chunks of the primary
    chunk map it solves. ``window``: a range per chunk axis, ``{'subchannel': range(149, 162)}``,
    selects the chunks whose index along each named axis lies in its range (every value of the
    axes not named); None: every chunk. ``strict``: the chunks whose pixels the coadd keeps
    (None: the window's); a window wider than its strict set overlaps the neighbouring jobs,
    for stitching. ``sc.Job('All')`` is the whole map."""
    name: str
    _: KW_ONLY
    window: dict[str, range] | None = None
    strict: dict[str, range] | None = None

    def _validate(self):
        if not isinstance(self.name, str) or not self.name:
            raise ConfigError(f"Job(name=...): a name, got {self.name!r}")
        object.__setattr__(self, 'window', _ranges(self.window, f"Job({self.name!r}, window=...)"))
        object.__setattr__(self, 'strict', _ranges(self.strict, f"Job({self.name!r}, strict=...)"))

    def __repr__(self):
        parts = [repr(self.name)]
        for k in ('window', 'strict'):
            v = getattr(self, k)
            if v is not None:
                parts.append(f"{k}={{{', '.join(f'{a!r}: {r!r}' for a, r in v.items())}}}")
        return f"Job({', '.join(parts)})"


def chunk_window(chunks, window) -> np.ndarray:
    """The chunks of ``chunks`` (a :class:`Chunks`) inside ``window``, a range per axis name as
    :class:`Job` takes it (None: every chunk): a boolean array, one entry per chunk."""
    n = chunks.n_chunks
    out = np.ones(n, dtype=bool)
    if not window:
        return out
    if chunks.axes is None:
        raise ConfigError(f"a job's window names the axes {sorted(window)}, and the chunk map {chunks.name!r} "
                          f"has none")
    for axis, r in window.items():
        if axis not in chunks.axes:
            raise ConfigError(f"a job's window names the axis {axis!r}; the chunk map {chunks.name!r} has "
                              f"{list(chunks.axes.names)}")
        of_chunk = np.asarray(chunks.axes[axis].of_chunk)
        if len(of_chunk) != n:
            raise ConfigError(f"the chunk map {chunks.name!r} has {n} chunks and its axes describe {len(of_chunk)}")
        out &= (of_chunk >= r.start) & (of_chunk < r.stop)
    return out


def check_job(job, chunks):
    """Raise :class:`~selfcal.config.base.ConfigError` when ``job``'s window or strict set names
    an axis the primary chunk map ``chunks`` lacks or reaches beyond its size, or when its strict
    set leaves its window."""
    for what, sel in (('window', job.window), ('strict', job.strict)):
        for axis, r in (sel or {}).items():
            if chunks.axes is None or axis not in chunks.axes:
                raise ConfigError(f"Job({job.name!r}, {what}=...): no chunk axis {axis!r} on the primary map "
                                  f"{chunks.name!r} (axes: {list(chunks.axes.names) if chunks.axes else []})")
            size = int(chunks.axes[axis].size)
            if r.start < 0 or r.stop > size:
                raise ConfigError(f"Job({job.name!r}, {what}=...): {axis!r} = {r!r} lies outside 0..{size - 1}")
    if job.strict is not None:
        window = chunk_window(chunks, job.window)
        strict = chunk_window(chunks, job.strict)
        if np.any(strict & ~window):
            raise ConfigError(f"Job({job.name!r}): its strict set leaves its window")


class Instrument(Config):
    """Base of the instrument settings (see the module docstring): the contract the run engine
    calls.

    A subclass implements :meth:`geometry`; it may override :meth:`layout` (how raw exposure
    files are read), :meth:`default_chunk_maps` (the chunk maps a model gets when it names none),
    :meth:`job_geometry` (valid pixels and weights of a job), :meth:`frame_variables` (per-frame
    data variables) and the mosaic's hooks (:meth:`aux_coadds`, :meth:`finalize_mosaic`), and
    name the data files its geometry reads (:meth:`geometry_files`). A ``tag`` setting names its
    products (default: the class name); ``unit`` is the mosaic's ``BUNIT``.

    The run engine builds the geometry once per action. It keeps it between the actions of a
    process only when the class says ``geometry_is_pure = True``: its :meth:`geometry` depends on
    nothing but its settings and the files :meth:`geometry_files` names (no module state, no other
    file). The built-in instruments say so; a subclass that overrides the geometry of one says so
    again, or its geometry is built by each action.
    """

    #: The unit of the calibrated data, the mosaic's ``BUNIT`` (a constant here, or a setting).
    unit = ''
    #: Whether :meth:`geometry` depends only on the settings and the :meth:`geometry_files`, so that
    #: the run engine may keep it between actions (a class constant, never a setting).
    geometry_is_pure = False

    # ---- the contract -------------------------------------------------------------------
    def geometry(self, oversample=1) -> base.DetectorGeometry:
        """The detector geometry (:func:`Geometry`): the shape, the per-pixel maps, what the
        instrument's own functions need, the detector-plane maps sampled ``oversample`` times per
        pixel. A new instrument implements it, as the built-in ones (``sc.Camera``, ``sc.SPHEREx``,
        ``sc.Euclid``) do."""
        raise NotImplementedError(f"{type(self).__name__}: a subclass of sc.Instrument implements "
                                  f"geometry(oversample)")

    def default_chunk_maps(self) -> dict:
        """The chunk maps a model gets when it names none, ``{name: sc.ChunkMap(...)}`` in order,
        the first one primary (default none: the model names its maps)."""
        return {}

    def layout(self) -> base.ExposureLayout:
        """How a raw exposure file is read (default: a FITS file, science in extension 1, no mask)."""
        return base.ExposureLayout(sci_ext=[1], dq_ext=None, detector_ids=[0], ref_use_ext=(1,),
                                     cache_tag=f'headers_{self.product_tag}')

    def job_geometry(self, geom, job) -> base.JobGeometry:
        """Valid chunks and pixel weights of ``job`` on the primary chunk map: the chunks inside its
        window are the solve's (``chunk_valid``), those inside its strict set the coadd's
        (``chunk_valid_strict``); every pixel of a valid chunk weighs 1 (the solve's weight in
        float64 on the detector grid, the mosaic's in float32 on the sampled grid), every other
        pixel 0."""
        cm = geom.chunk_map
        valid = chunk_window(cm, job.window)
        strict = valid if job.strict is None else chunk_window(cm, job.strict)
        det, grid = np.asarray(cm.det), np.asarray(cm.grid)
        det_w = np.where(det >= 0, valid[np.where(det >= 0, det, 0)], False).astype(np.float64)
        grid_w = np.where(grid >= 0, strict[np.where(grid >= 0, grid, 0)], False).astype(np.float32)
        return base.JobGeometry(det_valid_weight=det_w, grid_valid_weight=grid_w,
                                  chunk_valid=valid, chunk_valid_strict=strict)

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

    def aux_coadds(self, geom):
        """The ``(band centre, band width)`` detector maps whose per-pixel weighted mean and std the
        mosaic coadds with the data (``Coadd(instrument_maps=True)``); default None: nothing."""
        return None

    def finalize_mosaic(self, geom, mosaicker, maps, sigma):
        """Label or complete the mosaic's instrument maps once the coadd is done (only when
        :meth:`aux_coadds` gives some)."""

    def geometry_files(self) -> tuple:
        """The data files :meth:`geometry` reads, beyond the settings (default none): a geometry kept
        by the run engine (``geometry_is_pure``) is built again when one of them changes."""
        return ()

    def default_ignore_flags(self) -> tuple:
        """The data-quality bits that do not flag a pixel unless a recipe says otherwise."""
        return tuple(getattr(self.layout(), 'default_ignore_bits', ()) or ())

    # ---- naming --------------------------------------------------------------------------
    @property
    def product_tag(self) -> str:
        """The instrument's part of the products' tag (``cal_<tag>[_<primary map tag>]_<job><suffix>.h5``)."""
        return str(getattr(self, 'tag', None) or type(self).__name__)

    def check_jobs(self, jobs):
        """Raise :class:`~selfcal.config.base.ConfigError` for jobs this instrument cannot run
        (default: each is a :class:`Job` and their names differ)."""
        names = []
        for j in jobs:
            if not isinstance(j, Job):
                raise ConfigError(f"{type(self).__name__}: a job is a Job, got {j!r}")
            names.append(j.name)
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ConfigError(f"jobs named twice: {dup}")
