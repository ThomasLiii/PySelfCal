"""The recipe: everything that decides the numbers of a calibration.

A :class:`Recipe` is the model (:class:`~selfcal.models.model.Model`) and how it is fitted
(:class:`Fit`), coadded (:class:`Coadd`) and summed (:class:`Numerics`), under a ``name`` that
becomes the products' suffix. It holds no paths and no machine: the same recipe runs on any
field (:class:`~selfcal.run.field.Field`) and any machine (:class:`~selfcal.run.compute.Compute`).

``Numerics`` is part of the recipe because it changes bytes: the LSQR transpose product depends
on the thread count, the assembly and the coadd on their batch sizes. Its defaults are
production's, so a run on any machine reproduces production bit for bit.

Outlier clips (:class:`Clip`) judge each observation against the median of its group: the whole
frame (``per="frame"``), each chunk of the primary map (``per="chunk"``), or chunk groups the run
script defines (:class:`ChunkGroups`), e.g. SPHEREx's subchannels::

    SUBCHANNEL = sc.ChunkGroups.along("subchannel")
    fit = sc.Fit(clip=sc.Clip(5.0, per=SUBCHANNEL))
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import Callable, Literal

from ..config.base import Config, ConfigError
from ..config.functions import check_picklable
from ..models.model import Model, continuum

__all__ = ['ChunkGroups', 'Clip', 'Fit', 'Coadd', 'Numerics', 'Recipe']


@dataclass(frozen=True)
class ChunkGroups(Config):
    """Groups of chunks of a chunk map: by the values of an axis of the map
    (:meth:`along`), or by an explicit chunk-to-group list (:meth:`mapping`)."""
    axis: str | None = None
    _: KW_ONLY
    groups: tuple[int, ...] | None = None
    map: str | None = None

    def _validate(self):
        if (self.axis is None) == (self.groups is None):
            raise ConfigError("ChunkGroups: give an axis (ChunkGroups.along(axis)) or a chunk-to-group list "
                              "(ChunkGroups.mapping(groups))")

    def __repr__(self):
        m = '' if self.map is None else f", map={self.map!r}"
        if self.axis is not None:
            return f"ChunkGroups.along({self.axis!r}{m})"
        return f"ChunkGroups.mapping({list(self.groups)!r}{m})"

    @classmethod
    def along(cls, axis, map=None) -> ChunkGroups:
        """The chunks with equal values of the chunk-map axis ``axis`` form a group (SPHEREx:
        ``along("subchannel")``, one group per subchannel)."""
        return cls(axis, map=map)

    @classmethod
    def mapping(cls, chunk_to_group, map=None) -> ChunkGroups:
        """Chunk ``i`` belongs to group ``chunk_to_group[i]``."""
        return cls(groups=tuple(int(g) for g in chunk_to_group), map=map)


@dataclass(frozen=True)
class Clip(Config):
    """An outlier clip at ``sigma`` standard deviations (robust: median and MAD) of each
    observation's group: ``per="frame"`` (the frame), ``"chunk"`` (its chunk of the primary
    map) or a :class:`ChunkGroups`; or bins of a data variable, ``edges`` of ``variable``
    (default the instrument's wavelength). ``ignore_flags``: the data-quality bits that do not
    flag a pixel, where the clip sets them (the N-pass first pass)."""
    sigma: float
    _: KW_ONLY
    per: Literal['frame', 'chunk'] | ChunkGroups = 'frame'
    edges: tuple[float, ...] | None = None
    variable: str | None = None
    ignore_flags: tuple[int, ...] | None = None

    def _validate(self):
        if self.sigma <= 0:
            raise ConfigError(f"Clip({self.sigma}): sigma is positive (no clip: clip=None)")
        if self.edges is not None and self.per != 'frame':
            raise ConfigError("Clip: edges= define the groups; give them or per=, not both")
        if self.variable is not None and self.edges is None:
            raise ConfigError("Clip(variable=...): the variable is binned with edges=, which are not given")


def as_clip(value, what) -> Clip | None:
    """A clip setting: None (no clip), a number (sigma, per frame), or a :class:`Clip`."""
    if value is None or isinstance(value, Clip):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return Clip(float(value))
    raise ConfigError(f"{what}: expected a sigma, a Clip or None, got {value!r}")


def _check_hook(hook, what):
    if hook is not None:
        check_picklable(hook, what)


@dataclass(frozen=True)
class Fit(Config):
    """How the model is fitted to the frames.

    ``iterations``: of the iterative solver (``method``: ``"lsqr"`` or ``"lsmr"``; ``tolerance``:
    its stopping tolerance, one number or ``(atol, btol)``; ``damp``: its global damping;
    ``precondition``: column scaling; ``float32``: single-precision products). ``clip``: the
    outlier clip (a sigma, a :class:`Clip`, or None). ``use_mask``: drop pixels flagged in the
    data-quality mask, except the bits ``ignore_flags`` (None: the instrument's default).
    ``shot_noise_weights``: weight observations by ``1/sqrt(|value|)``.
    ``frame_hook`` / ``raw_frame_hook``: called on each frame after / before its corrections
    (an importable function or a picklable object; see :mod:`selfcal.config.functions`).
    ``line_fisher_threshold``: below this Fisher information, a pixel of a sky term after the
    first reads as unconstrained.
    """
    iterations: int = 50
    _: KW_ONLY
    clip: float | Clip | None = 5.0
    use_mask: bool = True
    ignore_flags: tuple[int, ...] | None = None
    shot_noise_weights: bool = False
    tolerance: float | tuple[float, float] = 1e-6
    method: Literal['lsqr', 'lsmr'] = 'lsqr'
    damp: float = 0.0
    precondition: bool = True
    float32: bool = True
    frame_hook: Callable | None = None
    raw_frame_hook: Callable | None = None
    line_fisher_threshold: float = 10.0

    def _validate(self):
        if self.iterations < 1:
            raise ConfigError(f"Fit(iterations={self.iterations}): at least 1")
        object.__setattr__(self, 'clip', as_clip(self.clip, 'Fit(clip=...)'))
        if self.clip is not None and self.clip.ignore_flags is not None:
            raise ConfigError("Fit: the data-quality bits to ignore are Fit(ignore_flags=...), not the clip's")
        _check_hook(self.frame_hook, 'Fit(frame_hook=...)')
        _check_hook(self.raw_frame_hook, 'Fit(raw_frame_hook=...)')

    @property
    def atol_btol(self) -> tuple[float, float]:
        t = self.tolerance
        return (t, t) if isinstance(t, float) else (float(t[0]), float(t[1]))


@dataclass(frozen=True)
class Coadd(Config):
    """How the calibrated frames are coadded into the mosaic.

    ``clip``: sigma clip against the per-pixel std (None: the mean only). ``std``: make the std
    map (the clip needs it). ``use_mask`` / ``ignore_flags`` / ``shot_noise_weights``: as in
    :class:`Fit`. ``oversample``: each detector pixel sampled ``oversample`` times per axis on the
    way (the final mosaic stays on the reference grid). ``instrument_maps``: coadd the
    instrument's per-pixel maps too (SPHEREx: the wavelength maps). ``min_chunk_coverage``: a
    chunk observed in a smaller fraction of a frame takes no offset. ``subtract_offsets``:
    subtract the solved offsets (False: coadd the raw frames). ``normalize_offsets``: remove each
    frame's mean offset first. ``frame_hook``: called on each corrected frame.
    """
    clip: float | None = 2.0
    _: KW_ONLY
    std: bool = True
    use_mask: bool = True
    ignore_flags: tuple[int, ...] | None = None
    shot_noise_weights: bool = False
    oversample: int = 1
    instrument_maps: bool = True
    min_chunk_coverage: float = 0.01
    subtract_offsets: bool = True
    normalize_offsets: bool = False
    frame_hook: Callable | None = None

    def _validate(self):
        if self.clip is not None and self.clip <= 0:
            raise ConfigError(f"Coadd(clip={self.clip}): sigma is positive (no clip: clip=None)")
        if self.clip is not None and not self.std:
            raise ConfigError("Coadd: the sigma clip needs the std map (std=True), or clip=None")
        if self.oversample < 1:
            raise ConfigError(f"Coadd(oversample={self.oversample}): at least 1")
        _check_hook(self.frame_hook, 'Coadd(frame_hook=...)')


@dataclass(frozen=True)
class Numerics(Config):
    """The summation layout: changes the last bits of the products, never the science.

    ``threads``: the solver's threads. ``rmatvec_threads``: the threads of its transpose product,
    each summing its rows into a buffer of its own (None: as many as ``threads``, capped so the
    buffers stay under 16 GB; 1: the sequential product). ``batch``: frames per assembly batch.
    ``mosaic_batch``: frames per batch of the coadd's frame cache; ``coadd_batch``: of its passes.
    The defaults are production's, so any machine reproduces production bit for bit.
    """
    threads: int = 48
    _: KW_ONLY
    batch: int = 50
    mosaic_batch: int = 50
    coadd_batch: int = 50
    rmatvec_threads: int | None = None

    def _validate(self):
        for k in ('threads', 'batch', 'mosaic_batch', 'coadd_batch'):
            if getattr(self, k) < 1:
                raise ConfigError(f"Numerics({k}={getattr(self, k)}): at least 1")
        if self.rmatvec_threads is not None and self.rmatvec_threads < 1:
            raise ConfigError(f"Numerics(rmatvec_threads={self.rmatvec_threads}): at least 1 (None: automatic)")


@dataclass(frozen=True)
class Recipe(Config):
    """The model and how it is fitted, coadded and summed, under a ``name``.

    ``model``: the :class:`~selfcal.models.model.Model` (default ``sc.continuum()``). ``fit``,
    ``coadd`` (None: no mosaic), ``numerics``. ``name``: the products' suffix
    (``cal_<tag>_<job>_<name>.h5``; empty: no suffix).
    """
    model: Model | None = None
    _: KW_ONLY
    fit: Fit = field(default_factory=Fit)
    coadd: Coadd | None = field(default_factory=Coadd)
    numerics: Numerics = field(default_factory=Numerics)
    name: str = ''

    def _validate(self):
        if self.model is None:
            object.__setattr__(self, 'model', continuum())
        if any(c in self.name for c in '/\\'):
            raise ConfigError(f"Recipe(name={self.name!r}): a name is part of a file name; no path separators")

    @property
    def suffix(self) -> str:
        """The products' suffix: ``_<name>`` (empty without a name)."""
        return f'_{self.name}' if self.name else ''
