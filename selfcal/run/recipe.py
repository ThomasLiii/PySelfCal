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

A :class:`Deflate` solves the smooth sky patterns of the fit directly and runs the iterations on the
rest, for solves whose slowest part is the large scales (:mod:`selfcal.core.deflation`)::

    fit = sc.Fit(300, tolerance=0, deflate=sc.Deflate(4))
"""
from __future__ import annotations

import math
from dataclasses import KW_ONLY, dataclass, field
from typing import Callable, Literal

from ..config.base import Config, ConfigError, added
from ..config.functions import check_picklable
from ..models.model import Model, continuum

__all__ = ['ChunkGroups', 'Clip', 'ResidualRule', 'GradientRule', 'LargeScaleRule', 'ChiSquaredRule', 'Stop',
           'Deflate', 'Fit', 'Coadd', 'Numerics', 'Recipe']


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


@dataclass(frozen=True)
class ChiSquaredRule(Config):
    """Stop when iterating on can no longer change the solution by a meaningful fraction of its
    statistical error: the chi-squared the solve gained over the last ``window`` iterations is below
    ``sigma`` squared.

    The chi-squared of a solution is its squared residual in units of the data's noise variance per
    row, ``|b - A x|^2 / s^2``. A fall of it by ``g`` between two iterates means the data prefer the
    later one by ``g``; when the whole remaining fall is at most ``sigma^2``, every linear quantity of
    the map (a pixel, a large-scale mode, ...) is within ``sigma`` standard deviations of its converged
    value. So ``sigma = 0.1`` means the same on any field and model: converged to a tenth of the error
    bars. The noise variance per row is estimated from the residual itself, ``s^2 = |b - A x|^2 / (rows
    - unknowns)``, or given (``noise``, in the weighted units of the system: for a noise-free simulation,
    whose residual is not noise).

    The rule reads the true residual, from a product ``A x`` every ``every`` iterations (``window`` a
    multiple of ``every``), and compares the checks ``window`` iterations apart: the gain it sees is a
    lower bound on what remains (a solve can rest on a plateau before a slow direction moves), which
    the deflation of the smooth sky patterns and the large-scale rule guard against. A solver that
    idles once converged (CGLS, the default) makes the iteration limit a safe cap, so this rule is an
    early exit, not a safety device. ``sc.Stop(chi2=0.1)`` is ``ChiSquaredRule(0.1)``.
    """
    sigma: float = 0.1
    _: KW_ONLY
    window: int = 50
    every: int = 10
    noise: float | None = None

    def _validate(self):
        if self.sigma <= 0:
            raise ConfigError(f"ChiSquaredRule(sigma={self.sigma}): positive")
        if self.every < 1:
            raise ConfigError(f"ChiSquaredRule(every={self.every}): at least 1 iteration (the true residual)")
        if self.window < self.every or self.window % self.every:
            raise ConfigError(f"ChiSquaredRule(window={self.window}, every={self.every}): the window is a whole "
                              f"number of checks (a multiple of every, at least one)")
        if self.noise is not None and self.noise <= 0:
            raise ConfigError(f"ChiSquaredRule(noise={self.noise}): a positive noise variance per row, or None "
                              f"(estimated from the residual)")


_SOLVER_TESTS = ('compatible', 'least_squares', 'condition')


@dataclass(frozen=True)
class Stop(Config):
    """Opt-in rules that may end a solve before its iterations, recorded when they do.

    ``residual`` (a :class:`ResidualRule`, or its ``below``), ``gradient`` (a
    :class:`GradientRule`, or its ``below``), ``large_scale`` (a :class:`LargeScaleRule`, or its
    ``below``), ``chi2`` (a :class:`ChiSquaredRule`, or its ``sigma``: converged to that fraction of the
    error bars, the scale-free rule): the rules; ``solver_tests``: the solver's own tolerance tests (LSQR's or LSMR's)
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
    chi2: float | ChiSquaredRule | None = added(None, since='2026-10')

    def _validate(self):
        for name, cls in (('residual', ResidualRule), ('gradient', GradientRule), ('large_scale', LargeScaleRule),
                          ('chi2', ChiSquaredRule)):
            v = getattr(self, name)
            if isinstance(v, float):
                object.__setattr__(self, name, cls(v))
        tests = self.solver_tests
        if isinstance(tests, tuple):
            kept = tuple(t for t in _SOLVER_TESTS if t in tests)
            object.__setattr__(self, 'solver_tests', True if kept == _SOLVER_TESTS else (kept or False))
        if self.min_iterations < 0:
            raise ConfigError(f"Stop(min_iterations={self.min_iterations}): at least 0")
        if (self.residual is None and self.gradient is None and self.large_scale is None and self.chi2 is None
                and not self.solver_tests):
            raise ConfigError("Stop(): give at least one rule (residual=, gradient=, large_scale=, chi2= or "
                              "solver_tests=)")

    @property
    def rules(self) -> tuple[str, ...]:
        """The rules given, in the order ``residual``, ``gradient``, ``large_scale``, ``chi2``, ``solver_tests``."""
        return tuple(n for n in ('residual', 'gradient', 'large_scale', 'chi2', 'solver_tests') if getattr(self, n))

    @property
    def tests(self) -> tuple[str, ...]:
        """The solver's tests the Stop keeps."""
        return _SOLVER_TESTS if self.solver_tests is True else (self.solver_tests or ())

    def first_iteration(self, rule) -> int:
        """The first iteration at which ``rule`` (one of :attr:`rules`) can hold: the window of a
        residual or gradient rule, two checks after the start for the large-scale rule, 1 for the
        solver's tests."""
        if rule in ('residual', 'gradient', 'chi2'):
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


@dataclass(frozen=True)
class Degenerate(Config):
    """Which pattern directions of a :class:`Deflate` count as degenerate, to be excluded from the sky.

    The deflation's setup decomposes its pattern space (the smooth sky shapes with the offsets that absorb
    them) into directions and measures each: how well the data see it (the seen fraction, ``sqrt(lambda /
    lambda_max)``) and, when the model has frame-shared offsets, how much better they would see it without
    those offsets' absorption (the visibility ratio). The degenerate directions are the exact gauges (seen
    about 0: the data never constrain them) together with the directions a rule names: ``absorbed_by_shared``
    (a ratio: a sky shape the frame-shared offsets mimic in nearly every frame; about ``1 / sqrt(fraction of
    the data still seeing it)``; shapes the per-frame terms absorb sit at 1; None: off) and ``gauge_below`` (a
    seen fraction, a level: None: off). Their sky shapes, in map space, become a hard constraint on the sky
    (as :class:`~selfcal.models.model.Sky` ``exclude=`` does for shapes given by hand): the map has no
    component along them, what the data place there goes to the other terms, and the solve then runs on
    the constrained system. One run, nothing named by hand: on a survey whose geometry makes the sky's
    radial quadratic indistinguishable from a detector pattern, that quadratic (with the mean) is what comes
    out; elsewhere, whatever that model and footprint make degenerate within the pattern space. The record
    keeps the selection and every direction's measures.
    """
    absorbed_by_shared: float | None = 3.0
    gauge_below: float | None = None

    def _validate(self):
        if self.absorbed_by_shared is not None and not float(self.absorbed_by_shared) >= 1.0:
            raise ConfigError(f"Degenerate(absorbed_by_shared={self.absorbed_by_shared}): a ratio of visibilities, at least 1")
        if self.gauge_below is not None and not 0.0 <= float(self.gauge_below) < 1.0:
            raise ConfigError(f"Degenerate(gauge_below={self.gauge_below}): a fraction, 0 <= gauge_below < 1")


@dataclass(frozen=True)
class Deflate(Config):
    """Solve the smooth sky patterns directly and iterate on the rest (deflation).

    The slowest part of a solve is usually the sky's largest scales: each frame's offsets absorb almost
    all of a pattern much larger than a frame, so the data see it weakly and the solver resolves it last
    (hundreds to thousands of iterations when the offsets also absorb gradients, as a per-chunk slope
    does). With a Deflate, the solve takes, for each sky term, the patterns ``f_a(x) f_b(y)`` for
    ``a, b = 0 ... order`` over the bounding box of the term's covered pixels (``basis="cosine"``:
    ``cos(pi a u)``, ``u`` from 0 to 1 across the box; ``"polynomial"``: Legendre polynomials), plus the
    polynomials of degree at most 2, each with the offsets that absorb it best (computed from the system:
    any offset model), solves the least squares along them directly, and runs the iterations on the
    rest of the unknowns only. The solution is the same least-squares solution as without; the
    iterations it takes are fewer when those patterns are what is slow. ``terms``: the sky terms to
    deflate (None: all). ``storage``: the precision the deflation's two vector sets are kept in (None: the
    products', ``Fit(float32=)``; ``"float64"``: always float64, twice the memory and projection traffic with
    float32 products, the same solution to the products' rounding; ``"float32"`` needs float32 products, with
    float64 ones it would let a converged solve drift). The arithmetic of the projections is float64 either
    way.

    The solution includes the patterns' least-squares values at once: where the data barely constrain
    a pattern (the record's ``nullness``), that value is what the data say, however large, while an
    unconverged plain solve would still be near its start. Directions the data cannot see at all
    (gauges: the sky's mean against per-frame scalars; a plane or quadratic the offsets absorb
    exactly) are recognised and left as a plain solve leaves them. A gauge the patterns contain only in
    part (a higher-degree pattern the offsets absorb exactly, which needs a flat geometry and offsets
    of that degree) would make the solve drift along it once converged: deflate it whole
    (``basis="polynomial"`` of at least its degree).

    ``gauge_below``: treat as a gauge every direction of the pattern space the data see below this
    fraction of the best-seen one (``sqrt(lambda / lambda_max)``, the record's ``seen`` spectrum):
    not solved, zero sky amplitude, the offsets take it, as the exact gauges are treated. None (the
    default): exact gauges only. A near-null direction's least-squares value is what little the
    data say about it, noise and systematics included, and it can be large (a shared detector
    pattern that a smooth sky mode mimics in every frame); this rule states the convention that
    such modes have no sky amplitude, whatever the footprint or the model made them. Set it at
    a gap of the spectrum (the solve's log prints it).

    ``absorbed_by_shared``: the same freeze decided by what absorbs the direction rather than by a level,
    so one run decides it: the setup also computes each direction's visibility when the frame-shared
    offsets take no part in the absorption. Those are found in the system itself (the offset columns
    coupled to far more columns than the others: whatever term, shape or basis made them, a constant
    per detector column or any detector-fixed pattern alike; the solve's log says how many). A
    direction the data would see more than this factor better without them is a sky shape that the
    shared pattern mimics in every frame, and is attributed to the instrument: frozen at zero sky
    amplitude. Directions the per-frame terms absorb on their own (a plane, the mean) have a ratio
    near 1 and are left as they are; the ratio of a shape only the shared pattern absorbs is about
    one over the square root of the fraction of the data that still sees it, so a few sits in an empty
    gap between the two kinds (the record keeps every ratio: ``visibility_ratio_without_shared``).
    None (the default): off. Without frame-shared offsets in the model every ratio is 1 and nothing is
    frozen. A frozen direction stays in the deflation space with its amplitude held at zero (the data's
    signature along it leaves the right-hand side, so the solve stays consistent); the directions kept
    get the amplitudes they would get anyway.

    ``exclude``: a :class:`Degenerate` names the directions of the pattern space that are degenerate (the exact
    gauges and the rules' directions); their sky shapes are excluded from the sky as a hard constraint in map
    space (the mechanism of ``Sky(exclude=)``), and the solve runs on the constrained system: the
    decomposition is made first, then the deflated solve with the constraint. None (the default): off.

    Cost: two float64 passes over the matrix with all the patterns at the start, one at the end (and
    one per snapshot); memory and per-iteration work of ``2 x 4 x patterns`` bytes per covered unknown
    with float32 products (``2 x 8`` with float64; four passes over them per iteration, six with LSQR), with :attr:`patterns_per_term` patterns per deflated term.
    Undamped fits only (``Fit(damp=0)``). The solve's record says what it did
    (:mod:`selfcal.core.deflation`).
    """
    order: int = 4
    _: KW_ONLY
    basis: Literal['cosine', 'polynomial'] = 'cosine'
    terms: tuple[str, ...] | None = None
    storage: Literal['float32', 'float64'] | None = None
    gauge_below: float | None = added(None, since='2026-10')
    absorbed_by_shared: float | None = added(None, since='2026-10')
    exclude: Degenerate | None = added(None, since='2026-10')

    def _validate(self):
        if not 0 <= self.order <= 16:
            raise ConfigError(f"Deflate(order={self.order}): 0 to 16")
        if self.basis not in ('cosine', 'polynomial'):
            raise ConfigError(f"Deflate(basis={self.basis!r}): 'cosine' or 'polynomial'")
        if self.storage not in (None, 'float32', 'float64'):
            raise ConfigError(f"Deflate(storage={self.storage!r}): 'float32', 'float64' or None (the products' "
                              f"precision)")
        if self.gauge_below is not None and not 0.0 <= float(self.gauge_below) < 1.0:
            raise ConfigError(f"Deflate(gauge_below={self.gauge_below}): a fraction, 0 <= gauge_below < 1")
        if self.absorbed_by_shared is not None and not float(self.absorbed_by_shared) >= 1.0:
            raise ConfigError(f"Deflate(absorbed_by_shared={self.absorbed_by_shared}): a ratio of visibilities, at least 1")
        if self.exclude is not None and not isinstance(self.exclude, Degenerate):
            raise ConfigError(f"Deflate(exclude={self.exclude!r}): a Degenerate(...) (the rules naming the degenerate directions) or None")
        if self.terms is not None:
            terms = (self.terms,) if isinstance(self.terms, str) else tuple(self.terms)
            if not terms or not all(isinstance(t, str) and t for t in terms):
                raise ConfigError(f"Deflate(terms={self.terms!r}): sky term names, or None for all")
            object.__setattr__(self, 'terms', terms)

    @property
    def patterns_per_term(self) -> int:
        """The patterns of each deflated sky term: ``(order + 1)^2`` products, plus the polynomials of
        degree at most 2 they do not contain."""
        k = int(self.order)
        extra = [(a, c) for a in range(3) for c in range(3 - a) if (a, c) != (0, 0)
                 and not (self.basis == 'polynomial' and a <= k and c <= k)]
        return (k + 1) ** 2 + len(extra)


def _check_hook(hook, what):
    if hook is not None:
        check_picklable(hook, what)


@dataclass(frozen=True)
class Gauge(Config):
    """The post-solve gauge of a per-frame gradient model (:mod:`selfcal.core.gauge`).

    A model with, per frame, a constant per chunk (a band of detector rows) and a slope along
    detector x (an :class:`~selfcal.models.model.Offsets` term ``times`` a function
    ``(det_x - centre) / half``) cannot tell a sky plane from its image in those offsets, and the
    solve lands anywhere in that family. After the solve the survey-mean plane the offsets absorbed
    is fitted to the solved slopes (``slope``: that term's name; ``constants``: the chunk-constant
    term's, default the first term on the same chunk map without a basis) -- each frame's mean slope
    against the frame's x axis on the grid, with a slope fixed on the detector as the nuisance -- and
    moved into the sky: added to the cal's sky term and, ``apply="map"``, to the maps of every mosaic
    made from the cal (the offsets stay as solved; exact), or, ``apply="offsets"``, removed from the
    offsets instead (exact only up to the band discretisation). ``degree=2`` moves a sky quadratic
    too, fitted with the plane to every (frame, chunk) slope, with a detector-fixed slope per chunk
    and a polynomial of degree ``position_degree`` (default 2) in the frame's position as the
    nuisance. A quadratic is not a freedom of the model: its curvature along x within a frame has no
    term in the offsets, so the data pin it, weakly, and moving it can take real sky curvature out of
    the map. ``geometry``: each frame's axes and centre on the reference grid from a tangent plane
    through its stored header (``"tangent"``: pointing, reference pixel, PC matrix, scale; no
    distortion) or from its full stored WCS (``"header"``). ``centre`` and ``half`` (default: the
    detector's ``((n_x - 1) / 2, (n_y - 1) / 2)`` and ``n_x / 2``) are checked against the slope
    term's basis.
    """
    slope: str
    _: KW_ONLY
    constants: str | None = None
    degree: Literal[1, 2] = 1
    position_degree: int | None = None
    apply: Literal['map', 'offsets'] = 'map'
    geometry: Literal['tangent', 'header'] = 'tangent'
    centre: tuple[float, float] | None = None
    half: float | None = None

    def _validate(self):
        if not self.slope:
            raise ConfigError("Gauge(slope=...): the name of the per-frame x-slope term")
        if self.degree not in (1, 2):
            raise ConfigError(f"Gauge(degree={self.degree}): 1 (a plane) or 2 (plane + quadratic)")
        if self.position_degree is not None:
            if self.degree != 2:
                raise ConfigError(f"Gauge(position_degree={self.position_degree}): the nuisance of the quadratic's fit "
                                  "(degree=2); the plane's has none")
            if self.position_degree < 0:
                raise ConfigError(f"Gauge(position_degree={self.position_degree}): at least 0 (none)")
        if self.apply not in ('map', 'offsets'):
            raise ConfigError(f"Gauge(apply={self.apply!r}): 'map' or 'offsets'")
        if self.geometry not in ('tangent', 'header'):
            raise ConfigError(f"Gauge(geometry={self.geometry!r}): 'tangent' or 'header'")
        if self.centre is not None:
            object.__setattr__(self, 'centre', (float(self.centre[0]), float(self.centre[1])))
        if self.half is not None and self.half <= 0:
            raise ConfigError(f"Gauge(half={self.half}): positive")


@dataclass(frozen=True)
class Fit(Config):
    """How the model is fitted to the frames.

    ``iterations``: of the iterative solver (``method``: ``"cgls"``, the default, ``"lsqr"`` or
    ``"lsmr"``; ``tolerance``: its stopping tolerance, one number or ``(atol, btol)``, where
    ``tolerance=0`` turns every stopping test off, the condition-estimate stop too, so the solve runs
    exactly ``iterations`` iterations; ``damp``: its global damping; ``precondition``: column scaling;
    ``float32``: single-precision products; ``vectors``: the precision of the solver's own vectors,
    ``"float32"`` or ``"float64"``, None for the method's own: CGLS float64, LSQR and LSMR the products').
    The three methods take the same products per iteration and give the same iterates while a solve
    converges; CGLS carries its residual as a vector it recomputes from the data every 50 iterations and
    takes steps that vanish with the gradient, so a converged CGLS solve stays put however many
    iterations follow, where LSQR in float32 drifts. CGLS has no estimate of the condition number
    (``conlim`` and the ``"condition"`` test never stop it). With ``vectors="float64"``, LSQR and LSMR
    keep float64 vectors on a float32 matrix, with the products computed in float64 (slower). Every solve is recorded: in the cal file's ``solve`` group, in the
    action's record, and its history per iteration in ``<field>/records/<cal stem>_history.npz``
    (:mod:`selfcal.core.solve_record`). ``clip``: the
    outlier clip (a sigma, a :class:`Clip`, or None). ``use_mask``: drop pixels flagged in the
    data-quality mask, except the bits ``ignore_flags`` (None: the instrument's default).
    ``shot_noise_weights``: weight observations by ``1/sqrt(|value|)`` (the fixed Poisson weight of
    :func:`selfcal.geometry.map_helper.make_weight`; a configurable noise model is the model's
    weight: ``sc.Model(weight=sc.Variance(...))`` or ``sc.shot_noise()``).
    ``frame_hook`` / ``raw_frame_hook``: called on each frame after / before its corrections
    (an importable function or a picklable object; see :mod:`selfcal.config.functions`).
    ``line_fisher_threshold``: below this Fisher information, a pixel of a sky term after the
    first reads as unconstrained. ``stop``: opt-in rules that may end the solve before
    ``iterations`` (a :class:`Stop`; None: the solver's own tests, by ``tolerance``; with a Stop,
    ``tolerance`` is used only by its ``solver_tests``; ``Stop(chi2=0.1)``: the scale-free rule, converged to a
    tenth of the error bars). ``deflate``: solve the smooth sky patterns
    directly and iterate on the rest (a :class:`Deflate`; None: plain iterations). ``gauge``: after
    the solve, move the sky plane / quadratic a per-frame gradient model absorbed into the sky (a
    :class:`Gauge`; None: the solution as solved).
    ``seed``: the solver's starting vector: ``'scalar'`` (each frame's scalar at the mean of its data, every
    other column at zero; without scalars each non-sky column at its own diagonal estimate) or ``'offsets'``
    (every non-sky column at its frame's joint least-squares fit to the data, the sky at zero:
    :func:`~selfcal.core.solution.compute_x0_offsets_fit`) or ``'zero'`` (everything at zero). A warm start
    (``calibrate(start=)``) replaces them.
    """
    iterations: int = 50
    _: KW_ONLY
    clip: float | Clip | None = 5.0
    use_mask: bool = True
    ignore_flags: tuple[int, ...] | None = None
    shot_noise_weights: bool = False
    tolerance: float | tuple[float, float] = 1e-6
    method: Literal['cgls', 'lsqr', 'lsmr'] = 'cgls'
    damp: float = 0.0
    precondition: bool = True
    float32: bool = True
    frame_hook: Callable | None = None
    raw_frame_hook: Callable | None = None
    line_fisher_threshold: float = 10.0
    stop: Stop | None = added(None, since='2026-10')
    deflate: Deflate | None = added(None, since='2026-10')
    vectors: Literal['float32', 'float64'] | None = added(None, since='2026-10')
    gauge: Gauge | None = added(None, since='2026-10')
    seed: Literal['scalar', 'offsets', 'zero'] = added('scalar', since='2026-10')

    def _validate(self):
        if self.seed not in ('scalar', 'offsets', 'zero'):
            raise ConfigError(f"Fit(seed={self.seed!r}): 'scalar' (each frame's scalar at its data's mean, the rest at zero), "
                              f"'offsets' (every offset column at its frame's joint least-squares fit, the sky at zero) or 'zero' (everything at zero)")
        if self.gauge is not None and not isinstance(self.gauge, Gauge):
            raise ConfigError(f"Fit(gauge={self.gauge!r}): an sc.Gauge")
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
        if self.deflate is not None and self.damp:
            raise ConfigError(f"Fit(deflate=..., damp={self.damp}): deflation needs an undamped fit (damp=0)")
        if self.vectors not in (None, 'float32', 'float64'):
            raise ConfigError(f"Fit(vectors={self.vectors!r}): 'float32', 'float64' or None (the method's own)")
        if self.vectors == 'float32' and not self.float32:
            raise ConfigError("Fit(vectors='float32', float32=False): float32 vectors need float32 products")
        if (self.stop is not None and self.method == 'cgls' and isinstance(self.stop.solver_tests, tuple)
                and 'condition' in self.stop.solver_tests):
            raise ConfigError("Fit(method='cgls', stop=Stop(solver_tests=...)): CGLS has no condition estimate, so "
                              "the 'condition' test never holds; keep 'compatible' and 'least_squares' (or True: "
                              "the tests the method has)")

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
