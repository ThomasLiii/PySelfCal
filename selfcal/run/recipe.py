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

A :class:`Stop` gives the fit opt-in rules that may end the solve before its iterations
(:class:`ResidualRule`, :class:`GradientRule`, :class:`LargeScaleRule`, the solver's own tests):
they decide where the solve stops, so they are part of the recipe (:mod:`selfcal.core.monitor`)::

    fit = sc.Fit(1000, tolerance=0, stop=sc.Stop(gradient=1e-3, large_scale=1e-3, min_iterations=40))
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import Callable, Literal

from ..config.base import Config, ConfigError, added
from ..config.functions import check_picklable
from ..models.model import Model, continuum

__all__ = ['ChunkGroups', 'Clip', 'ResidualRule', 'GradientRule', 'LargeScaleRule', 'Stop', 'Fit', 'Coadd',
           'Numerics', 'Recipe']


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


def _check_window(rule):
    name = type(rule).__name__
    if rule.below <= 0:
        raise ConfigError(f"{name}(below={rule.below}): positive")
    if rule.window < 1:
        raise ConfigError(f"{name}(window={rule.window}): at least 1 iteration")
    if rule.every is not None:
        if rule.every < 1:
            raise ConfigError(f"{name}(every={rule.every}): at least 1 iteration (None: the solver's estimate, "
                              f"every iteration)")
        if rule.window % rule.every:
            raise ConfigError(f"{name}(window={rule.window}, every={rule.every}): the window is a whole number of "
                              f"checks (a multiple of every)")


@dataclass(frozen=True)
class ResidualRule(Config):
    """Stop when ``|b - A x|`` has stopped falling: its relative decrease over the last ``window``
    iterations is below ``below``.

    ``every=None`` measures the solver's estimate (``r1norm``, every iteration, free); ``every=m``
    the true residual, from a product ``A x`` every ``m`` iterations (``window`` a multiple of
    ``m``). ``sc.Stop(residual=1e-3)`` is ``ResidualRule(1e-3)``.
    """
    below: float
    _: KW_ONLY
    window: int = 10
    every: int | None = None

    def _validate(self):
        _check_window(self)


@dataclass(frozen=True)
class GradientRule(Config):
    """Stop when the largest ``|A^T r|`` over the last ``window`` iterations is at most ``below``
    times its value at the start of the solve (the window: the gradient of a CG-type solver is not
    monotone).

    ``every=None`` measures the solver's estimate (``arnorm``, every iteration, free); ``every=m``
    the true gradient, from the products ``A x`` and ``A^T r`` every ``m`` iterations (``window`` a
    multiple of ``m``). ``sc.Stop(gradient=1e-3)`` is ``GradientRule(1e-3)``.
    """
    below: float
    _: KW_ONLY
    window: int = 10
    every: int | None = None

    def _validate(self):
        _check_window(self)


@dataclass(frozen=True)
class LargeScaleRule(Config):
    """Stop when the large scales of every sky term have settled: the relative change of each
    term's smooth fit is below ``below`` at the last two checks.

    A check every ``every`` iterations (iteration 0 included) fits the monomials of degree at most
    ``degree`` in the grid's coordinates (1, x, y, x^2, xy, y^2 for 2) to the term's covered pixels
    on every ``step``-th row and column; its change is the rms of the change of the fitted surface
    over those pixels, relative to the rms of the surface, both without their means
    (:class:`~selfcal.core.monitor.SmoothFit`). ``sc.Stop(large_scale=1e-3)`` is
    ``LargeScaleRule(1e-3)``.

    Each sky term is normalised by its own large-scale amplitude (the rms of its fitted surface), and
    the rule takes the largest change over the terms: a term with little large-scale content (a
    small rms) can keep the relative change high while the terms that matter have settled. Combine
    the rule with the others (``Stop(combine=...)``) accordingly, and read each term's change in the
    history file (``large_scale_change``, one column per term) and in the stop record (``terms``).
    """
    below: float
    _: KW_ONLY
    every: int = 10
    degree: int = 2
    step: int = 4

    def _validate(self):
        if self.below <= 0:
            raise ConfigError(f"LargeScaleRule(below={self.below}): positive")
        if self.every < 1:
            raise ConfigError(f"LargeScaleRule(every={self.every}): at least 1 iteration")
        if not 1 <= self.degree <= 6:
            raise ConfigError(f"LargeScaleRule(degree={self.degree}): 1 to 6")
        if self.step < 1:
            raise ConfigError(f"LargeScaleRule(step={self.step}): at least 1")


_SOLVER_TESTS = ('compatible', 'least_squares', 'condition')


@dataclass(frozen=True)
class Stop(Config):
    """Opt-in rules that may end a solve before its iterations, recorded when they do.

    ``residual`` (a :class:`ResidualRule`, or its ``below``), ``gradient`` (a
    :class:`GradientRule`, or its ``below``), ``large_scale`` (a :class:`LargeScaleRule`, or its
    ``below``): the rules; ``solver_tests``: the solver's own tolerance tests (LSQR's or LSMR's)
    that count as a rule (``True``: all three; or some of ``"compatible"``, ``"least_squares"``,
    ``"condition"``). With a Stop, the solver's tests stop the solve only through it: ``False``, the
    default, turns them off, so the fit's ``tolerance`` (``atol``, ``btol``) and the condition limit
    are unused (the plan and the solve's record say so). ``combine``: ``"all"`` (the default: every
    rule given holds at the same iteration) or ``"any"``. ``min_iterations``: no rule ends the solve
    before. The iteration limit stays the hard cap, and machine precision still ends a solve; a rule
    that cannot hold before the limit (its window or checks need more iterations) is a note of the
    plan. The cal's ``solve`` group, the action's record and the log say which rule ended the solve
    and what it measured (:mod:`selfcal.core.monitor`).

    Residual and gradient tests do not see the weakest, largest-scale directions of a selfcal
    system (a smooth sky pattern the offsets can nearly absorb): they can pass while those still
    move. ``large_scale`` watches them; combine it with the others (``"all"``) to stop only when
    the large scales have settled too.
    """
    _: KW_ONLY
    residual: float | ResidualRule | None = None
    gradient: float | GradientRule | None = None
    large_scale: float | LargeScaleRule | None = None
    solver_tests: bool | tuple[Literal['compatible', 'least_squares', 'condition'], ...] = False
    min_iterations: int = 0
    combine: Literal['all', 'any'] = 'all'

    def _validate(self):
        for name, cls in (('residual', ResidualRule), ('gradient', GradientRule), ('large_scale', LargeScaleRule)):
            v = getattr(self, name)
            if isinstance(v, float):
                object.__setattr__(self, name, cls(v))
        tests = self.solver_tests
        if isinstance(tests, tuple):
            kept = tuple(t for t in _SOLVER_TESTS if t in tests)
            object.__setattr__(self, 'solver_tests', True if kept == _SOLVER_TESTS else (kept or False))
        if self.min_iterations < 0:
            raise ConfigError(f"Stop(min_iterations={self.min_iterations}): at least 0")
        if self.residual is None and self.gradient is None and self.large_scale is None and not self.solver_tests:
            raise ConfigError("Stop(): give at least one rule (residual=, gradient=, large_scale= or solver_tests=)")

    @property
    def rules(self) -> tuple[str, ...]:
        """The rules given, in the order ``residual``, ``gradient``, ``large_scale``, ``solver_tests``."""
        return tuple(n for n in ('residual', 'gradient', 'large_scale', 'solver_tests') if getattr(self, n))

    @property
    def tests(self) -> tuple[str, ...]:
        """The solver's tests the Stop keeps."""
        return _SOLVER_TESTS if self.solver_tests is True else (self.solver_tests or ())

    def first_iteration(self, rule) -> int:
        """The first iteration at which ``rule`` (one of :attr:`rules`) can hold: the window of a
        residual or gradient rule, two checks after the start for the large-scale rule, 1 for the
        solver's tests."""
        if rule in ('residual', 'gradient'):
            return int(getattr(self, rule).window)
        if rule == 'large_scale':
            return 2 * int(self.large_scale.every)
        return 1

    def unreachable(self, iterations) -> list[str]:
        """Why this Stop cannot end a solve of at most ``iterations`` iterations before its limit, as
        notes (empty: it can): each rule that cannot hold by then, and the whole Stop when no
        combination of its rules can (``combine="all"``: one rule that cannot; ``"any"``: none
        can), or when ``min_iterations`` is past the limit."""
        n = int(iterations)
        firsts = {r: self.first_iteration(r) for r in self.rules}
        late = [r for r, k in firsts.items() if k > n]
        out = [f"the {r.replace('_', '-')} rule cannot hold before iteration {firsts[r]}, after the "
               f"iteration limit ({n})" for r in late]
        if self.min_iterations > n:
            out.append(f"min_iterations={self.min_iterations} is after the iteration limit ({n})")
        if out and (self.min_iterations > n or (late and (self.combine == 'all' or len(late) == len(firsts)))):
            out.append(f"so no stop rule can end the solve: it runs its {n} iterations (or stops at machine "
                       f"precision)")
        return out


def _check_hook(hook, what):
    if hook is not None:
        check_picklable(hook, what)


@dataclass(frozen=True)
class Fit(Config):
    """How the model is fitted to the frames.

    ``iterations``: of the iterative solver (``method``: ``"lsqr"`` or ``"lsmr"``; ``tolerance``:
    its stopping tolerance, one number or ``(atol, btol)``, where ``tolerance=0`` turns every
    stopping test off, the condition-estimate stop too, so the solve runs exactly ``iterations``
    iterations; ``damp``: its global damping; ``precondition``: column scaling; ``float32``:
    single-precision products). Every solve is recorded: in the cal file's ``solve`` group, in the
    action's record, and its history per iteration in ``<field>/records/<cal stem>_history.npz``
    (:mod:`selfcal.core.solve_record`). ``clip``: the
    outlier clip (a sigma, a :class:`Clip`, or None). ``use_mask``: drop pixels flagged in the
    data-quality mask, except the bits ``ignore_flags`` (None: the instrument's default).
    ``shot_noise_weights``: weight observations by ``1/sqrt(|value|)``.
    ``frame_hook`` / ``raw_frame_hook``: called on each frame after / before its corrections
    (an importable function or a picklable object; see :mod:`selfcal.config.functions`).
    ``line_fisher_threshold``: below this Fisher information, a pixel of a sky term after the
    first reads as unconstrained. ``stop``: opt-in rules that may end the solve before
    ``iterations`` (a :class:`Stop`; None: the solver's own tests, by ``tolerance``; with a Stop,
    ``tolerance`` is used only by its ``solver_tests``).
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
    stop: Stop | None = added(None, since='2026-10')

    def _validate(self):
        if self.iterations < 1:
            raise ConfigError(f"Fit(iterations={self.iterations}): at least 1")
        if self.stop is not None and self.stop.solver_tests and self.exact_iterations:
            raise ConfigError("Fit(tolerance=0, stop=Stop(solver_tests=...)): tolerance=0 turns the solver's tests "
                              "off; give a tolerance, or no solver_tests")
        if min(self.atol_btol) < 0:
            raise ConfigError(f"Fit(tolerance={self.tolerance}): at least 0 (0: run exactly `iterations` iterations)")
        object.__setattr__(self, 'clip', as_clip(self.clip, 'Fit(clip=...)'))
        if self.clip is not None and self.clip.ignore_flags is not None:
            raise ConfigError("Fit: the data-quality bits to ignore are Fit(ignore_flags=...), not the clip's")
        _check_hook(self.frame_hook, 'Fit(frame_hook=...)')
        _check_hook(self.raw_frame_hook, 'Fit(raw_frame_hook=...)')

    @property
    def atol_btol(self) -> tuple[float, float]:
        """The solver's ``(atol, btol)``."""
        t = self.tolerance
        return (t, t) if isinstance(t, float) else (float(t[0]), float(t[1]))

    @property
    def exact_iterations(self) -> bool:
        """Whether the solve runs exactly ``iterations`` iterations: ``tolerance=0`` (or ``(0, 0)``)
        turns off the solver's tolerance tests and its condition-estimate stop."""
        return self.atol_btol == (0.0, 0.0)


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
