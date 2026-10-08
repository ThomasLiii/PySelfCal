"""The self-calibration model in Python: sky terms, offset terms, data variables, priors.

These are the settings objects of the Python API (``sc.Model``, ``sc.Sky``, ``sc.Offsets``,
...). They are checked when built and lower to the model's data form,
:class:`~selfcal.models.spec.ModelSpec`, which the solver consumes; nothing numerical happens
here. The fitted equation, for every observation (one value of one frame on one reference
pixel)::

    data = Σ_j S_j[pixel] · c_j(v)  +  Σ_m O_m[group(frame), chunk, k] · φ_mk(v)  +  s(frame)

* :class:`Sky`: a sky map ``S_j`` times a known function ``c_j`` of data variables ``v``
  (``times``; none: a constant sky);
* :class:`Offsets`: an offset per chunk of a chunk map (``on``), shared by the frames of a
  group (``per``), optionally times known functions (``times``, ``basis``), or a polynomial
  (``polynomial``); with smoothness, polynomial-shape and mean-zero priors;
* the per-frame scalar ``s`` (``Model.scalar``);
* data variables beyond the built-in and instrument ones (``Model.variables``: a header
  keyword, a function of the frame list, a detector or sky map, ...);
* :class:`Prior`: any linear rows on the unknowns, written as a function.

A function of data variables is any Python function that worker processes can import
(:mod:`selfcal.config.functions`). Its parameters without defaults name the variables it
reads (``def above_80(temperature, t0=80.0)`` reads ``temperature``); :class:`Function` says
so explicitly (``sc.Function(np.sin, of="phase")``) and binds parameters.
"""
from __future__ import annotations

import functools
import inspect
from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Callable, Literal

import numpy as np

from ..config.base import Config, ConfigError, FrozenDict
from ..config.functions import by_value, function_ref


def _check(fn, what):
    """Raise unless the worker processes can get ``fn``: by import, or by value (:class:`by_value`)."""
    if not isinstance(fn, by_value):
        function_ref(fn, what)


def _ref(fn):
    """How a function is handed to the model's data form: its import path, or the by-value object itself."""
    return fn if isinstance(fn, by_value) else function_ref(fn)

__all__ = ['Function', 'Shape', 'template', 'gaussian', 'linear', 'catalog', 'Poly', 'Sky', 'Offsets',
           'Header', 'PerFrame', 'DetectorMap', 'SkyMap', 'SolvedSky', 'Layer', 'Derived', 'FrameFunction',
           'Prior', 'Model', 'continuum', 'spectral', 'two_block']


# =============================================================================== functions
def _call(name, positional=(), keywords=()):
    """``name(positional..., key=value...)``: the repr of an object as the call that rebuilds it."""
    from ..config.base import python_repr
    args = [python_repr(v) for v in positional] + [f'{k}={python_repr(v)}' for k, v in keywords]
    return f"{name}({', '.join(args)})"


def _fn_name(fn):
    if isinstance(fn, str) or isinstance(fn, by_value):
        return fn if isinstance(fn, str) else repr(fn)
    return getattr(fn, '__qualname__', getattr(fn, '__name__', repr(fn)))


class _Named(str):
    """A name shown without quotes in a repr (a function's name)."""

    def __repr__(self):
        return str(self)


def _unwrap(fn, params, what):
    """``(function, params)`` with a keyword ``functools.partial`` unwrapped into the params."""
    while isinstance(fn, functools.partial):
        if fn.args:
            raise ConfigError(f"{what}: a functools.partial with positional arguments cannot say which data "
                              f"variables the function reads; bind keywords only, or pass the function with "
                              f"of=... and its parameters")
        params = {**dict(fn.keywords), **params}
        fn = fn.func
    if not callable(fn):
        raise ConfigError(f"{what}: expected a function, got {fn!r}")
    return fn, params


def _signature(fn):
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


def _check_params(fn, params, n_variables, what):
    """The function accepts ``n_variables`` positional values and the keyword ``params``."""
    sig = _signature(fn)
    if sig is None:
        return
    has_kwargs = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    unknown = [k for k in params if k not in sig.parameters and not has_kwargs]
    if unknown:
        raise ConfigError(f"{what}: {getattr(fn, '__name__', fn)}() has no parameter "
                          f"{', '.join(map(repr, unknown))} (parameters: {', '.join(sig.parameters)})")
    try:
        sig.bind(*range(n_variables), **params)
    except TypeError as e:
        raise ConfigError(f"{what}: {getattr(fn, '__name__', fn)}() cannot be called with {n_variables} data "
                          f"variable(s) and parameters {sorted(params)}: {e}") from None


def _infer_variables(fn, params, what):
    """The data variables a bare function reads: its parameters without defaults that the
    params do not bind."""
    sig = _signature(fn)
    name = getattr(fn, '__name__', repr(fn))
    if sig is None:
        raise ConfigError(f"{what}: cannot read the parameters of {name}; say which data variables it reads: "
                          f"sc.Function({name}, of=('variable', ...))")
    names = []
    for p in sig.parameters.values():
        if p.kind is p.VAR_POSITIONAL:
            raise ConfigError(f"{what}: {name}(*{p.name}) takes any number of values; say which data variables "
                              f"it reads: sc.Function({name}, of=('variable', ...))")
        if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD) and p.default is p.empty and p.name not in params:
            names.append(p.name)
    if not names:
        raise ConfigError(f"{what}: {name}() has no parameter without a default, so it reads no data variable; "
                          f"say which it reads: sc.Function({name}, of=('variable', ...))")
    if 'frame' in names:
        raise ConfigError(f"{what}: {name}() has a parameter named 'frame'. If it reads the built-in frame index, "
                          f"say so: sc.Function({name}, of=({', '.join(map(repr, names))})); a function of the "
                          f"whole frame (its data, coordinates and header) is a variable source: "
                          f"sc.FrameFunction({name})")
    return tuple(names)


@dataclass(frozen=True, init=False)
class Function(Config):
    """A known function of data variables: ``fn(*variables, **params)``.

    ``of`` names the data variables, in the order of ``fn``'s positional parameters; without
    it they are ``fn``'s parameters that have no default and that ``params`` do not bind.
    ``params`` are checked against ``fn``'s signature. ``fn`` must be importable by the worker
    processes (:func:`~selfcal.config.functions.function_ref`)::

        sc.Function(np.cos, of="phase")
        sc.Function(above, of="temperature", t0=80.0)
        sc.Function(functools.partial(above, t0=80.0))     # reads `temperature`
    """
    fn: Callable
    of: tuple[str, ...]
    params: dict[str, Any]

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, fn, *, of=None, **params):
        what = f"Function({getattr(fn, '__name__', fn)})"
        fn, params = _unwrap(fn, params, what)
        if of is None:
            of = _infer_variables(fn, params, what)
        elif isinstance(of, str):
            of = (of,)
        of = tuple(of)
        _check_params(fn, params, len(of), what)
        _check(fn, what)
        object.__setattr__(self, 'fn', fn)
        object.__setattr__(self, 'of', of)
        object.__setattr__(self, 'params', FrozenDict(params))
        self.__post_init__()

    def __repr__(self):
        of = self.of[0] if len(self.of) == 1 else self.of
        return _call('Function', (_Named(_fn_name(self.fn)),), (('of', of),) + tuple(self.params.items()))

    @classmethod
    def _from_fields(cls, fn, of, params):
        return cls(fn, of=tuple(of), **dict(params))

    def lower(self, n=None) -> dict:
        """The coefficient's data form (``{variable, function, params[, n]}``)."""
        out = {'variable': self.of[0] if len(self.of) == 1 else list(self.of),
               'function': _ref(self.fn)}
        if self.params:
            out['params'] = dict(self.params)
        if n is not None:
            out['n'] = int(n)
        return out


@dataclass(frozen=True)
class Shape(Config):
    """A built-in function of one data variable: a tabulated curve, a Gaussian, a line, or a
    named coefficient of the instrument. Built by :func:`template`, :func:`gaussian`,
    :func:`linear` and :func:`catalog`."""
    kind: Literal['template', 'gaussian', 'linear', 'catalog']
    _: KW_ONLY
    of: str | None = 'wavelength'
    params: dict[str, Any] = field(default_factory=FrozenDict)

    def __repr__(self):
        params = ', '.join(f"{k}={v!r}" for k, v in self.params.items() if not isinstance(v, np.ndarray))
        of = '' if self.kind == 'catalog' or self.of == 'wavelength' else f", of={self.of!r}"
        if self.kind == 'catalog':
            rest = {k: v for k, v in self.params.items() if k != 'catalog'}
            return _call('catalog', (self.params['catalog'],), tuple(rest.items()))
        return f"{self.kind}({params}{of})"

    def lower(self, n=None) -> dict:
        """The coefficient's data form."""
        if self.kind == 'catalog':
            return dict(self.params)
        out = {'variable': self.of, 'function': self.kind}
        out.update({k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in self.params.items()})
        return out


def template(file=None, *, x=None, y=None, of='wavelength', norm='peak', x_key=None, y_key=None) -> Shape:
    """A tabulated function of ``of`` (linear interpolation, zero outside the table): arrays ``x``
    and ``y``, or a ``.npz`` file holding them (``x`` / ``y``, or the keys ``x_key`` / ``y_key``;
    a SPHEREx line template's ``center_um`` and ``G_peaknorm``, or ``G`` with ``norm="area"``)."""
    if (file is None) == (x is None or y is None):
        raise ConfigError("template(): give a file, or the arrays x= and y=")
    if file is not None:
        params = {'file': str(file), 'norm': norm}
        if x_key is not None:
            params['x_key'] = x_key
        if y_key is not None:
            params['y_key'] = y_key
    else:
        params = {'x': np.asarray(x, dtype=float), 'y': np.asarray(y, dtype=float)}
    return Shape('template', of=of, params=params)


def gaussian(center, *, sigma=None, width=None, intrinsic_var=None, fwhm_to_sigma=None, of='wavelength') -> Shape:
    """``exp(-(v - center)² / 2σ²)`` of the variable ``of``: a fixed ``sigma``, or one per
    observation from the width variable ``width`` (a FWHM, e.g. ``"bandwidth"``) added in
    quadrature to ``intrinsic_var``."""
    if (sigma is None) == (width is None):
        raise ConfigError("gaussian(): give sigma= (a fixed width) or width= (a data variable), not both")
    params = {'center': float(center)}
    if sigma is not None:
        params['sigma'] = float(sigma)
    else:
        params['width'] = str(width)
        if intrinsic_var is not None:
            params['intrinsic_var'] = float(intrinsic_var)
        if fwhm_to_sigma is not None:
            params['fwhm_to_sigma'] = float(fwhm_to_sigma)
    return Shape('gaussian', of=of, params=params)


def linear(center, halfwidth, *, of='wavelength') -> Shape:
    """``(v - center) / halfwidth`` of the variable ``of``."""
    return Shape('linear', of=of, params={'center': float(center), 'halfwidth': float(halfwidth)})


def catalog(name, **overrides) -> Shape:
    """The instrument's named coefficient ``name`` (SPHEREx: ``"pah_3p29"``), with its factory's
    parameters overridden by ``overrides`` (one given as None keeps the factory's default, so it
    is left out)."""
    return Shape('catalog', of=None, params={'catalog': str(name),
                                             **{k: v for k, v in overrides.items() if v is not None}})


def _as_function(value, what):
    """A ``times`` / ``basis`` / ``weight`` value: None, a variable name, a Function, a Shape, or a
    bare function (wrapped in :class:`Function`)."""
    if value is None or isinstance(value, (str, Function, Shape)):
        return value
    if callable(value):
        return Function(value) if not isinstance(value, functools.partial) else Function(value)
    raise ConfigError(f"{what}: expected a function of data variables, a variable name or a built-in shape, "
                      f"got {value!r}")


def _lower_function(value, n=None):
    if value is None:
        return None
    if isinstance(value, str):
        out = {'variable': value}
        if n is not None:
            out['n'] = int(n)
        return out
    return value.lower(n)


# =============================================================================== terms
@dataclass(frozen=True)
class Poly(Config):
    """A polynomial along an axis of a chunk map.

    As a soft prior (``Offsets(poly_prior=...)``), the offsets along ``along`` are pulled toward
    a degree-``degree`` polynomial with weight ``weight`` (default 1), over the ``window`` of
    axis values if one is given; ``along=None`` means the first axis the term is smoothed along.
    As a hard basis (``Offsets(polynomial=...)``), the offset IS a degree-``degree`` polynomial in
    ``along`` (default: the map's spectral axis), one per value of ``each`` (default: the map's
    group axis), over ``window`` (required), optionally piecewise on ``segments``
    (``[(lo, hi), ...]``).
    """
    degree: int
    _: KW_ONLY
    along: str | None = None
    window: range | None = None
    weight: float | None = None
    each: str | None = None
    segments: tuple[tuple[int, int], ...] | None = None

    def _validate(self):
        if self.degree < 0:
            raise ConfigError(f"Poly(degree={self.degree}): the degree is at least 0")
        if self.window is not None and (self.window.step != 1 or len(self.window) < 1):
            raise ConfigError(f"Poly(window={self.window!r}): a range of consecutive axis values, "
                              f"e.g. range(200, 321)")
        if self.weight is not None and self.weight < 0:
            raise ConfigError(f"Poly(weight={self.weight}): at least 0")

    def lower_soft(self, what):
        if self.each is not None or self.segments is not None:
            raise ConfigError(f"{what}: Poly(each=..., segments=...) describes a hard polynomial basis; "
                              f"give it as Offsets(polynomial=...)")
        out = {'axis': self.along, 'degree': int(self.degree),
               'weight': 1.0 if self.weight is None else float(self.weight)}
        if self.window is not None:
            out.update(lo=int(self.window.start), hi=int(self.window.stop - 1))
        return out


@dataclass(frozen=True)
class Sky(Config):
    """A sky term: a map on the reference grid times a known function of data variables.

    ``times``: the function (a Python function of data variables, :class:`Function`, a
    built-in :func:`template` / :func:`gaussian` / :func:`linear` / :func:`catalog`, or the name
    of a variable, which then multiplies the map itself); None: a constant sky. ``damping``: a
    coverage-weighted pull of the map toward zero (default: 0.1 for the model's first sky term,
    0.3 for the others; 0 for none).
    """
    name: str = 'continuum'
    _: KW_ONLY
    times: Function | Shape | str | Callable | None = None
    damping: float | None = None

    def _validate(self):
        object.__setattr__(self, 'times', _as_function(self.times, f"Sky({self.name!r}).times"))
        if self.damping is not None and self.damping < 0:
            raise ConfigError(f"Sky({self.name!r}, damping={self.damping}): at least 0")

    def lower(self, damping) -> dict:
        out = {'name': self.name, 'damp_weight': float(damping)}
        c = _lower_function(self.times)
        if c is not None:
            out['coefficient'] = c
        return out


@dataclass(frozen=True)
class Offsets(Config):
    """An offset term: one offset per chunk of a chunk map, shared by groups of frames.

    ``on``: the chunk map (None: the instrument's primary map; ``"detector"``: the whole
    detector as one chunk). ``per``: who shares an offset: ``"frame"`` (each frame its own),
    ``"all"`` (every frame the same: a detector-fixed pattern), or a frame variable (frames with
    equal values: ``"detector"``, ``"exposure"``, a night, a filter).

    Known functions: ``times`` (one function of data variables multiplying the offset at every
    observation), or ``basis`` with ``n`` functions (the unknowns become one coefficient per
    chunk and function; the function returns ``n`` arrays). ``polynomial``: the offset IS the
    polynomial :class:`Poly` along a chunk axis (no smoothness or shape prior then).

    Priors: ``smooth`` pulls chunks that neighbour along ``smooth_along`` (default: the map's
    own axes; ``()``: none) together, only pairs ``smooth_step`` apart when given;
    ``poly_prior`` pulls the offsets toward soft polynomials; ``mean_zero`` anchors each group's
    mean over chunks at zero; ``damping`` pulls every offset toward zero (coverage weighted).
    ``exact_group_rows`` (shared terms): the anchor and smoothness rows once per group, not per
    frame (the same normal equations, far fewer rows, different last bits).
    ``render``: which of the instrument's renderers draws the term on the mosaic.
    ``name``: how priors refer to the term (default: the map's name).
    """
    name: str | None = None
    _: KW_ONLY
    on: str | None = None
    per: str = 'frame'
    times: Function | Shape | str | Callable | None = None
    basis: Function | str | Callable | None = None
    n: int | None = None
    smooth: float = 0.0
    smooth_along: tuple[str, ...] | None = None
    smooth_step: int | None = None
    poly_prior: tuple[Poly, ...] = ()
    polynomial: Poly | None = None
    mean_zero: bool = False
    damping: float = 0.0
    exact_group_rows: bool = False
    render: str | None = None

    def _validate(self):
        label = f"Offsets({self.name!r})" if self.name else "Offsets()"
        object.__setattr__(self, 'times', _as_function(self.times, f"{label}.times"))
        object.__setattr__(self, 'basis', _as_function(self.basis, f"{label}.basis"))
        if self.times is not None and self.basis is not None:
            raise ConfigError(f"{label}: give times= (one function) or basis= with n= (several), not both")
        if self.basis is not None and (self.n is None or self.n < 2):
            raise ConfigError(f"{label}: basis= needs n= (the number of functions it returns, at least 2; "
                              f"one function is times=)")
        if self.n is not None and self.basis is None:
            raise ConfigError(f"{label}: n= is the number of functions of basis=, which is not given")
        if self.polynomial is not None:
            bad = [k for k, v in (('smooth', self.smooth), ('poly_prior', self.poly_prior),
                                  ('basis', self.basis), ('mean_zero', self.mean_zero),
                                  ('exact_group_rows', self.exact_group_rows)) if v]
            if bad:
                raise ConfigError(f"{label}: polynomial= makes the offset a polynomial, which takes no "
                                  f"{', '.join(bad)}")
            if self.polynomial.window is None:
                raise ConfigError(f"{label}: polynomial= needs Poly(window=range(lo, hi + 1)), the axis values "
                                  f"it spans")
            if self.polynomial.weight is not None:
                raise ConfigError(f"{label}: polynomial= is exact; weight= belongs to a soft poly_prior")
            if self.per != 'frame':
                raise ConfigError(f"{label}: polynomial= is fitted per frame (per='frame')")
        for p in self.poly_prior:
            p.lower_soft(label)
        if self.exact_group_rows and self.per == 'frame':
            raise ConfigError(f"{label}: exact_group_rows applies to a shared term (per='all' or a frame "
                              f"variable)")
        for k in ('smooth', 'damping'):
            if getattr(self, k) < 0:
                raise ConfigError(f"{label}: {k} is at least 0")

    @property
    def term_name(self) -> str:
        """The name priors use: ``name``, else the map's name (``"primary"`` for the primary map)."""
        return self.name or self.on or 'primary'

    def lower(self) -> dict:
        out = {'map': self.on}
        if self.polynomial is not None:
            p = self.polynomial
            out.update(kind='polybasis', degree=int(p.degree), axis=p.along, group_axis=p.each,
                       lo=int(p.window.start), hi=int(p.window.stop - 1),
                       segments=None if p.segments is None else [list(s) for s in p.segments])
        else:
            kind = {'frame': 'free', 'all': 'fixed'}.get(self.per, 'grouped')
            out.update(kind=kind, reg_weight=float(self.smooth), mean_zero=bool(self.mean_zero),
                       exact_group_rows=bool(self.exact_group_rows),
                       poly=[p.lower_soft('Offsets') for p in self.poly_prior])
            if self.smooth_along is not None:
                out['adjacency'] = list(self.smooth_along)
            if self.smooth_step is not None:
                out['adjacency_step'] = int(self.smooth_step)
            if kind == 'grouped':
                out['groups'] = self.per
        out['damp'] = float(self.damping)
        if self.render is not None:
            out['render'] = self.render
        if self.name is not None:
            out['name'] = self.name
        if self.times is not None:
            out['coefficient'] = _lower_function(self.times)
        if self.basis is not None:
            out['basis'] = _lower_function(self.basis, n=self.n)
        return out


# =============================================================================== data variables
def _source_function(fn, params, what):
    fn, params = _unwrap(fn, params, what)
    _check(fn, what)
    return fn, FrozenDict(params)


class _Source(Config):
    """A source of a data variable (``Model(variables={name: source})``)."""

    def lower(self, name) -> dict:
        raise NotImplementedError


@dataclass(frozen=True)
class Header(_Source):
    """A frame variable: the keyword ``key`` of each frame's stored header (``default`` where
    a frame lacks it)."""
    key: str
    _: KW_ONLY
    default: float | int | str | None = None

    def lower(self, name):
        out = {'header': self.key}
        if self.default is not None:
            out['default'] = self.default
        return out


@dataclass(frozen=True, init=False)
class PerFrame(_Source):
    """A frame variable computed by ``fn``: ``fn(frames, **params)`` (the frame file list) or,
    with ``of``, ``fn(*frame variables, **params)``; one value per frame."""
    fn: Callable
    of: tuple[str, ...]
    params: dict[str, Any]

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, fn, *, of=(), **params):
        fn, params = _source_function(fn, params, f"PerFrame({getattr(fn, '__name__', fn)})")
        object.__setattr__(self, 'fn', fn)
        object.__setattr__(self, 'of', (of,) if isinstance(of, str) else tuple(of))
        object.__setattr__(self, 'params', params)
        self.__post_init__()

    def __repr__(self):
        return _call('PerFrame', (_Named(_fn_name(self.fn)),),
                     ((('of', self.of),) if self.of else ()) + tuple(self.params.items()))

    @classmethod
    def _from_fields(cls, fn, of, params):
        return cls(fn, of=tuple(of), **dict(params))

    def lower(self, name):
        out = {'per_frame': _ref(self.fn)}
        if self.of:
            out['inputs'] = list(self.of)
        if self.params:
            out['params'] = dict(self.params)
        return out


def _map_value(source, what):
    if isinstance(source, np.ndarray):
        return source
    if callable(source):
        _check(source, what)
        return source
    return str(source)


@dataclass(frozen=True, init=False)
class DetectorMap(_Source):
    """A detector map: an array of the detector's shape, a ``.npy`` / ``.fits`` file, or a
    function ``fn(geometry, **params)`` returning the array."""
    source: Any
    params: dict[str, Any]

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, source, **params):
        object.__setattr__(self, 'source', _map_value(source, 'DetectorMap'))
        object.__setattr__(self, 'params', FrozenDict(params))
        self.__post_init__()

    def __repr__(self):
        src = _Named(_fn_name(self.source)) if callable(self.source) else self.source
        return _call('DetectorMap', (src,), tuple(self.params.items()))

    @classmethod
    def _from_fields(cls, source, params):
        return cls(source, **dict(params))

    def lower(self, name):
        v = self.source
        out = {'detector': _ref(v) if callable(v) else v}
        if self.params:
            out['params'] = dict(self.params)
        return out


@dataclass(frozen=True, init=False)
class SkyMap(_Source):
    """A reference-grid map: an array of the grid's shape, a ``.npy`` / ``.fits`` file, or a
    function ``fn(ref_wcs, ref_shape, **params)`` returning the array."""
    source: Any
    params: dict[str, Any]

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, source, **params):
        object.__setattr__(self, 'source', _map_value(source, 'SkyMap'))
        object.__setattr__(self, 'params', FrozenDict(params))
        self.__post_init__()

    def __repr__(self):
        src = _Named(_fn_name(self.source)) if callable(self.source) else self.source
        return _call('SkyMap', (src,), tuple(self.params.items()))

    @classmethod
    def _from_fields(cls, source, params):
        return cls(source, **dict(params))

    def lower(self, name):
        v = self.source
        out = {'sky': _ref(v) if callable(v) else v}
        if self.params:
            out['params'] = dict(self.params)
        return out


@dataclass(frozen=True)
class SolvedSky(_Source):
    """A reference-grid map: a sky term solved before (``term``, default the first) of the cal
    file ``cal``."""
    cal: str
    _: KW_ONLY
    term: str | int | None = None

    def lower(self, name):
        out = {'sky_cal': self.cal}
        if self.term is not None:
            out['term'] = self.term
        return out


@dataclass(frozen=True)
class Layer(_Source):
    """A per-observation plane stored in each frame file (``layers/<name>``; default: the
    variable's own name)."""
    name: str | None = None

    def lower(self, name):
        return {'layer': self.name or True}


@dataclass(frozen=True, init=False)
class Derived(_Source):
    """A per-observation function of other data variables: ``fn(*of, **params)``."""
    fn: Callable
    of: tuple[str, ...]
    params: dict[str, Any]

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, fn, *, of=None, **params):
        what = f"Derived({getattr(fn, '__name__', fn)})"
        fn, params = _unwrap(fn, params, what)
        of = _infer_variables(fn, params, what) if of is None else ((of,) if isinstance(of, str) else tuple(of))
        _check_params(fn, params, len(of), what)
        _check(fn, what)
        object.__setattr__(self, 'fn', fn)
        object.__setattr__(self, 'of', of)
        object.__setattr__(self, 'params', FrozenDict(params))
        self.__post_init__()

    def __repr__(self):
        return _call('Derived', (_Named(_fn_name(self.fn)),), (('of', self.of),) + tuple(self.params.items()))

    @classmethod
    def _from_fields(cls, fn, of, params):
        return cls(fn, of=tuple(of), **dict(params))

    def lower(self, name):
        out = {'function': _ref(self.fn), 'inputs': list(self.of)}
        if self.params:
            out['params'] = dict(self.params)
        return out


@dataclass(frozen=True, init=False)
class FrameFunction(_Source):
    """A per-observation variable computed from the whole frame: ``fn(frame, **params)``, where
    ``frame`` holds the frame's data, coordinates and header."""
    fn: Callable
    params: dict[str, Any]

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, fn, **params):
        fn, params = _source_function(fn, params, f"FrameFunction({getattr(fn, '__name__', fn)})")
        object.__setattr__(self, 'fn', fn)
        object.__setattr__(self, 'params', params)
        self.__post_init__()

    def __repr__(self):
        return _call('FrameFunction', (_Named(_fn_name(self.fn)),), tuple(self.params.items()))

    @classmethod
    def _from_fields(cls, fn, params):
        return cls(fn, **dict(params))

    def lower(self, name):
        out = {'frame_function': _ref(self.fn)}
        if self.params:
            out['params'] = dict(self.params)
        return out


# =============================================================================== priors
@dataclass(frozen=True, init=False)
class Prior(Config):
    """Linear rows on the unknowns of ``terms`` (sky-term names, offset-term names,
    ``"scalar"``): ``fn(*term infos, **params)`` returns them (the contract of
    :mod:`selfcal.models.priors`), scaled by ``weight``. ``fn`` is a function, or the name of a
    ready-made prior of :mod:`selfcal.models.priors` (see :mod:`selfcal.priors`)."""
    fn: Callable | str
    terms: tuple[str, ...]
    weight: float
    name: str | None
    params: dict[str, Any]

    def __new__(cls, *args, **kwargs):
        return object.__new__(cls)

    def __init__(self, fn, terms, *, weight=1.0, name=None, **params):
        what = f"Prior({getattr(fn, '__name__', fn)})"
        if isinstance(fn, str):
            from . import priors as ready
            if ':' not in fn and not hasattr(ready, fn):
                raise ConfigError(f"{what}: no ready-made prior {fn!r} (have {ready.__all__[2:]})")
        else:
            fn, params = _unwrap(fn, params, what)
            _check_params(fn, params, len((terms,) if isinstance(terms, str) else terms), what)
            _check(fn, what)
        object.__setattr__(self, 'fn', fn)
        object.__setattr__(self, 'terms', (terms,) if isinstance(terms, str) else tuple(terms))
        object.__setattr__(self, 'weight', weight)
        object.__setattr__(self, 'name', name)
        object.__setattr__(self, 'params', FrozenDict(params))
        self.__post_init__()

    def __repr__(self):
        fn = self.fn if isinstance(self.fn, str) else _Named(_fn_name(self.fn))
        terms = self.terms[0] if len(self.terms) == 1 else self.terms
        kw = ((('weight', self.weight),) if self.weight != 1.0 else ()) + \
             ((('name', self.name),) if self.name is not None else ()) + tuple(self.params.items())
        return _call('Prior', (fn, terms), kw)

    @classmethod
    def _from_fields(cls, fn, terms, weight, name, params):
        return cls(fn, tuple(terms), weight=weight, name=name, **dict(params))

    def lower(self) -> dict:
        out = {'terms': list(self.terms), 'function': self.fn if isinstance(self.fn, str) else _ref(self.fn),
               'weight': float(self.weight)}
        if self.params:
            out['params'] = dict(self.params)
        if self.name is not None:
            out['name'] = self.name
        return out


# =============================================================================== the model
@dataclass(frozen=True)
class Model(Config):
    """The model: sky terms, offset terms, the per-frame scalar, data variables, an
    observation weight and priors.

    ``sky``: :class:`Sky` terms (at least one; default one constant sky). ``offsets``:
    :class:`Offsets` terms. ``scalar``: a per-frame scalar ``s(frame)``. ``variables``: the
    model's own data variables, ``{name: source}`` (:class:`Header`, :class:`PerFrame`,
    :class:`DetectorMap`, :class:`SkyMap`, :class:`SolvedSky`, :class:`Layer`,
    :class:`Derived`, :class:`FrameFunction`). ``weight``: a function of data variables that
    multiplies every observation's weight (the coadd uses its square). ``priors``:
    :class:`Prior` rows. The presets :func:`continuum`, :func:`spectral` and :func:`two_block`
    build the standard models.
    """
    _: KW_ONLY
    sky: tuple[Sky, ...] = (Sky(),)
    offsets: tuple[Offsets, ...] = ()
    scalar: bool = True
    variables: dict[str, _Source] = field(default_factory=FrozenDict)
    weight: Function | Shape | str | Callable | None = None
    priors: tuple[Prior, ...] = ()

    def _validate(self):
        object.__setattr__(self, 'weight', _as_function(self.weight, 'Model(weight=...)'))
        if not self.sky:
            raise ConfigError("Model(sky=...): at least one sky term")
        names = [t.name for t in self.sky]
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            raise ConfigError(f"Model: sky terms named twice: {dup}")
        off = [t.term_name for t in self.offsets]
        explicit = [t.name for t in self.offsets if t.name]
        dup = sorted({n for n in explicit if explicit.count(n) > 1} | (set(explicit) & set(names)))
        if dup:
            raise ConfigError(f"Model: term names used twice: {dup}")
        known = set(names) | set(off) | ({'scalar'} if self.scalar else set())
        for p in self.priors:
            bad = [t for t in p.terms if t not in known]
            if bad and not any(t.name is None for t in self.offsets):
                raise ConfigError(f"Model: prior {p.name or p.fn!r} names terms {bad}; the model's terms are "
                                  f"{sorted(known)}")

    # ---- lowering -------------------------------------------------------------------------
    def sky_dampings(self) -> list[float]:
        """Each sky term's damping, defaults resolved: 0.1 for the first term, 0.3 for the others."""
        return [t.damping if t.damping is not None else (0.1 if j == 0 else 0.3) for j, t in enumerate(self.sky)]

    def lower(self, mosaic='full') -> dict:
        """The ``[model]`` table (:meth:`~selfcal.models.spec.ModelSpec.from_config`) of this model."""
        table = {'scalar': bool(self.scalar), 'mosaic': mosaic,
                 'sky': [t.lower(d) for t, d in zip(self.sky, self.sky_dampings())]}
        if self.offsets:
            table['offset'] = [t.lower() for t in self.offsets]
        if self.variables:
            table['variables'] = {k: v.lower(k) for k, v in self.variables.items()}
        if self.weight is not None:
            table['weight'] = _lower_function(self.weight)
        if self.priors:
            table['prior'] = [p.lower() for p in self.priors]
        return table

    def spec(self, mosaic='full'):
        """The :class:`~selfcal.models.spec.ModelSpec` of this model."""
        from .spec import ModelSpec
        return ModelSpec.from_config(self.lower(mosaic))

    def spectral_window(self):
        """``(lo, hi)`` (inclusive) of the first polynomial window of the offset terms (the hard
        polynomial's first), or None. The N-pass refit reads it."""
        for t in self.offsets:
            if t.polynomial is not None:
                w = t.polynomial.window
                return int(w.start), int(w.stop - 1)
        for t in self.offsets:
            for p in t.poly_prior:
                if p.window is not None:
                    return int(p.window.start), int(p.window.stop - 1)
        return None


# =============================================================================== presets
def continuum(smooth=0.1, poly_prior=None, damping=None) -> Model:
    """A constant sky (``damping``: see :class:`Sky`), an offset per frame and chunk of the
    primary map smoothed along the map's own axes (``smooth``) and anchored at zero mean, optional
    soft polynomials (``poly_prior``: a :class:`Poly` or several), and the per-frame scalar."""
    poly = () if poly_prior is None else ((poly_prior,) if isinstance(poly_prior, Poly) else tuple(poly_prior))
    return Model(sky=(Sky(damping=damping),), offsets=(Offsets(smooth=smooth, poly_prior=poly, mean_zero=True),))


def spectral(lines, polynomial=None, smooth=None, poly_prior=None, damping=None) -> Model:
    """A constant sky (``damping``) plus the sky terms ``lines`` (:class:`Sky` terms with a
    function of the wavelength), and the per-frame scalar. The offsets: a hard polynomial along
    the spectral axis (``polynomial``, a :class:`Poly` with a window), or the :func:`continuum`
    offset term (``smooth``, default 0.1; ``poly_prior``)."""
    lines = (lines,) if isinstance(lines, Sky) else tuple(lines)
    if polynomial is not None:
        if smooth is not None or poly_prior is not None:
            raise ConfigError("spectral(polynomial=...): the offset is the polynomial itself, which takes no "
                              "smooth= or poly_prior=")
        offsets = (Offsets(polynomial=polynomial),)
    else:
        poly = () if poly_prior is None else ((poly_prior,) if isinstance(poly_prior, Poly) else tuple(poly_prior))
        offsets = (Offsets(smooth=0.1 if smooth is None else smooth, poly_prior=poly, mean_zero=True),)
    return Model(sky=(Sky(damping=damping),) + lines, offsets=offsets)


def two_block(second='readout', smooth=0.1, second_smooth=0.0, along='subchannel', damping=None) -> Model:
    """A constant sky, an offset per frame and chunk of the primary map smoothed between
    neighbours one step apart along ``along`` (no anchor), and a detector-fixed offset on the
    chunk map ``second`` shared by every frame, anchored at zero mean (``second_smooth``: its
    smoothness weight). No per-frame scalar."""
    return Model(sky=(Sky(damping=damping),),
                 offsets=(Offsets(smooth=smooth, smooth_along=(along,), smooth_step=1),
                          Offsets(on=second, per='all', smooth=second_smooth, smooth_along=(), mean_zero=True)),
                 scalar=False)
