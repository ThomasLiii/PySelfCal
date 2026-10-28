"""The self-calibration model as data: variables, sky terms, offset terms, priors.

The equation the solver fits, for every observation ``i`` (one value of one
frame on one reference pixel ``P``, seen at a detector position)::

    data_i = Σ_j S_j[P] · c_j(v_i)  +  Σ_m Σ_k O_m[g_m(frame), chunk_m(i), k] · φ_mk(v_i)

* **data variables** ``v`` (:mod:`selfcal.models.variables`): named
  per-observation quantities — the built-in coordinates, the instrument's
  detector maps and frame values, and the model's own (``[model.variables]``:
  header keywords, functions of the frame list, detector or sky maps, stored
  layers, functions of other variables or of the whole frame).
* **S terms** (:class:`SkyTerm`): a per-pixel map ``S_j`` times a known
  coefficient ``c_j(v)`` — any function of data variables (a spectral template
  of the wavelength, a sine of the time, cos 2ψ of a polariser angle, ...);
  none = ``c = 1``. Prior: Tikhonov ``damp_weight``.
* **O terms** (:class:`OffsetTerm`): an offset per chunk of a chunk map, shared
  by the frames of a group ``g_m`` (``free``: every frame its own; ``fixed``:
  all frames one; ``grouped``: frames with equal values of any frame variable),
  optionally times ``n`` known functions ``φ_k(v)`` of data variables
  (``coefficient`` = one function, ``basis`` = several: a pattern times the
  temperature, a per-frame gradient in detector coordinates, a gain times a
  previous sky), or a Chebyshev polynomial along a chunk axis (``polybasis``).
  Priors: smoothness along chunk axes, polynomial shape, Tikhonov ``damp``;
  hard constraints: no polynomial of a degree along a chunk axis (``exclude``).
  A per-frame scalar is the O term of one chunk (map ``"detector"``), one value
  per frame.
* **weight**: an optional function of data variables multiplying every
  observation's weight (an inverse-variance plane, a per-frame quality).
* **priors** (:class:`PriorSpec`): any linear rows on the unknowns of one or
  several terms, written as a function (:mod:`selfcal.models.priors`).

A :class:`ModelSpec` is built from the ``[model]`` table a
:class:`~selfcal.models.model.Model` lowers to (:meth:`ModelSpec.from_config`),
checked against an instrument (:meth:`ModelSpec.check`) and lowered into the
objects the solver consumes. Every function is a built-in or ANY importable
``"package.module:name"``; a new instrument, coefficient, offset set-up or
prior needs no change here.
"""
from __future__ import annotations

import dataclasses
import importlib
import os
from dataclasses import dataclass, field

import numpy as np

from selfcal.config.base import import_name

from .offset_model import Basis, OffsetBlock, OffsetModel
from .offset_structure import (
    ChunkAxes,
    adjacency_along,
    adjacency_union,
    poly_basis_along,
    poly_chains_along,
)
from .sky_model import Coefficient, ImportedFunction, NoiseVariance, SkyComponent, SkyModel
from .variables import BUILTIN_VARIABLES, Derived, FrameFunction, VariableSet

__all__ = ['SkyTerm', 'OffsetTerm', 'PolyConstraint', 'VariableSpec', 'PriorSpec', 'ChunkMapSpec', 'ModelSpec',
           'build_coefficient', 'resolve_variable', 'load_function', 'chunk_map_of']

BUILTIN_FUNCTIONS = ('template', 'gaussian', 'linear')
OFFSET_KINDS = ('free', 'polybasis', 'fixed', 'grouped')
VARIABLE_SOURCES = ('header', 'per_frame', 'detector', 'sky', 'sky_cal', 'layer', 'function', 'frame_function')
DETECTOR_MAP = 'detector'          # the built-in single-chunk map (the whole detector)


# ---------------------------------------------------------------------------
# functions by name
# ---------------------------------------------------------------------------
def load_function(ref):
    """A callable from ``"package.module:name"`` (or a callable, returned as is)."""
    if callable(ref):
        return ref
    module, sep, attr = str(ref).partition(':')
    if not sep or not module or not attr:
        raise ValueError(f"function {ref!r} must look like 'package.module:name'")
    return getattr(importlib.import_module(module), attr)


def _hashable(v):
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    return v


# ---------------------------------------------------------------------------
# variables
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class VariableSpec:
    """One ``[model.variables]`` entry: a named data variable and its source.

    ======================================  =====================================================
    ``header = "KEY"``                      frame: the keyword of each frame's stored header
                                            (``default`` when absent)
    ``per_frame = "pkg.mod:fn"``            frame: ``fn(frames, **params)`` -> one value per frame,
                                            or ``fn(*frame variables, **params)`` with ``inputs``
    ``detector = "pkg.mod:fn" | file``      detector map: ``fn(geom, **params)`` or a .npy/.fits
    ``sky = "pkg.mod:fn" | file``           reference-grid map: ``fn(ref_wcs, ref_shape, **params)``
                                            or a .npy/.fits
    ``sky_cal = "cal.h5"``, ``term``        reference-grid map: a solved sky term (default the first)
    ``layer = "name"``                      per observation: the frame file's ``layers/<name>``
    ``function = "pkg.mod:fn"``, ``inputs`` per observation: ``fn(*inputs, **params)``
    ``frame_function = "pkg.mod:fn"``       per observation: ``fn(frame, **params)`` with the
                                            frame's data, coordinates and header
    ======================================  =====================================================
    """
    name: str
    source: str
    value: object = None
    inputs: tuple = ()
    params: tuple = ()
    term: str | None = None
    default: object = None

    @property
    def scope(self) -> str:
        """What the variable's values are indexed by.

        ``'frame'`` (a ``header`` or ``per_frame`` source), ``'detector'``, ``'sky'``
        (also ``sky_cal``), ``'layer'``, ``'derived'`` (a ``function`` source) or
        ``'frame_function'``."""
        return {'header': 'frame', 'per_frame': 'frame', 'detector': 'detector', 'sky': 'sky',
                'sky_cal': 'sky', 'layer': 'layer', 'function': 'derived',
                'frame_function': 'frame_function'}[self.source]

    @classmethod
    def from_config(cls, name, d) -> VariableSpec:
        """The ``[model.variables]`` entry ``name = d`` as a :class:`VariableSpec`.

        ``d`` holds exactly one source key of :data:`VARIABLE_SOURCES`, whose value
        becomes ``value``, and optionally ``inputs`` (a string is one input),
        ``params`` (stored as a hashable tuple of pairs), ``term`` and ``default``.
        A ``layer`` source given as ``true`` (or empty) reads the layer named like
        the variable. A :class:`VariableSpec` is returned unchanged. Raises
        ``ValueError`` for no or several source keys, an unknown key, or a
        ``function`` source without ``inputs``.
        """
        if isinstance(d, VariableSpec):
            return d
        d = dict(d)
        srcs = [k for k in VARIABLE_SOURCES if k in d]
        if len(srcs) != 1:
            raise ValueError(f"variable {name!r} needs exactly one source key of {VARIABLE_SOURCES}, got {sorted(d)}")
        src = srcs[0]
        value = d.pop(src)
        inputs = d.pop('inputs', ())
        inputs = (inputs,) if isinstance(inputs, str) else tuple(inputs)
        params = dict(d.pop('params', None) or {})
        term = d.pop('term', None)
        default = d.pop('default', None)
        if d:
            raise ValueError(f"variable {name!r}: unknown keys {sorted(d)}")
        if src == 'function' and not inputs:
            raise ValueError(f"variable {name!r} = function of other variables needs `inputs`")
        if src == 'layer' and value in (True, None, ''):
            value = name
        return cls(name=str(name), source=src, value=value, inputs=inputs, params=_hashable(params),
                   term=term, default=default)


# ---------------------------------------------------------------------------
# terms
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SkyTerm:
    """One sky term: a per-pixel map times ``coefficient``.

    ``coefficient`` None: ``c = 1`` (a constant sky). Otherwise a
    :class:`~selfcal.models.sky_model.Coefficient`, or its config form, a dict::

        {variable = "wavelength",                  # data variable(s) the coefficient reads
         function = "template" | "gaussian" | "linear" | "package.module:name",
         <the function's parameters>}             # inline, or under params = {...}

    ``name`` (the product name of the map) defaults to ``continuum`` for a
    constant term; a term with a coefficient names itself.
    ``damp_weight``: the term's Tikhonov prior (None = the solver default). ``exclude``: a hard
    constraint on the map's shape, ``"package.module:function"`` (or the function itself) returning
    the excluded shapes ``(pixels, r)`` from the grid's pixel coordinates ``(u, v)`` (see
    :class:`selfcal.run.recipe.Sky`); None: none.
    """
    name: str | None = None
    coefficient: object = None
    damp_weight: float | None = None
    exclude: object = None

    def __post_init__(self):
        name = self.name
        c = self.coefficient
        if name is None:
            if c is None:
                name = 'continuum'
            else:
                raise ValueError(f"a sky term with a coefficient needs a name: {c!r}")
        object.__setattr__(self, 'name', str(name))


@dataclass(frozen=True)
class PolyConstraint:
    """A soft polynomial constraint: along ``axis`` the offset is pulled toward a
    degree-``degree`` polynomial (finite-difference rows of weight ``weight``),
    optionally only over the window ``[lo, hi]`` of that axis. ``axis=None``: the
    first axis the term is smoothed along (the standard term's default)."""
    axis: str | None = None
    degree: int = 1
    weight: float = 1.0
    lo: int | None = None
    hi: int | None = None


@dataclass(frozen=True)
class OffsetTerm:
    """One offset block on the chunk map ``map`` (None = the instrument's primary
    map; ``"detector"`` = the whole detector as one chunk unless the instrument
    has a map of that name).

    Sharing: ``kind='free'`` — an offset per (frame, chunk); ``'fixed'`` — one
    offset vector shared by every frame (a detector-fixed pattern);
    ``'grouped'`` — one per group of frames with equal values of ``groups``
    (any frame variable: ``detector``, ``exposure``, a night, a filter, ...);
    ``'polybasis'`` — the offset IS a degree-``degree`` Chebyshev in ``axis``
    (default: the map's spectral axis) per value of ``group_axis`` over
    ``[lo, hi]``, optionally piecewise on ``segments``.

    Known functions of data variables: ``coefficient`` (one function, the
    sky-term form ``{variable, function, params}``) multiplies the offset at
    every observation; ``basis`` (``{variable, function, params, n}``, ``n``
    functions) makes the unknowns one coefficient per (group, chunk, function).

    Priors: ``reg_weight`` pulls chunks that neighbour along the ``adjacency``
    axes together (None = the map's default axes; ``()`` = none;
    ``adjacency_step`` restricts pairs to that difference), ``poly`` adds soft
    polynomial constraints, ``damp`` is Tikhonov damping toward zero.
    ``exclude_degree`` / ``exclude_axis``: a hard constraint, the term has no
    polynomial of that degree along the axis (per line of the other axes, per
    group), imposed by reparametrisation in the solver
    (:meth:`ModelSpec.column_exclusions`); degree 0 with no axis: no constant
    over all of a group's chunks. ``signature_degree``: a hard constraint, the
    term carries no part of its response to the sky polynomials of degree 1 to
    that degree (``'all'``: to every pattern of the deflation but the
    constant), computed from the system by the deflation's setup
    (:meth:`ModelSpec.signature_exclusions`). ``exclude_overlap``: a hard
    constraint, the term carries no detector pattern that the named offset terms
    can make with the same coefficients in every frame (their fixed patterns), the
    intersection of the two pattern spaces over the job's pixels
    (:meth:`ModelSpec.column_exclusions`).
    ``exact_group_rows``: for fixed/grouped terms, emit the adjacency rows once
    per group instead of once per frame (exact; far fewer rows). ``render``:
    which of the chunk map's renderers draws the term on the mosaic (default
    the map's own). ``name``: how priors refer to the term (default: the
    map's name)."""
    map: str | None = None
    kind: str = 'free'
    reg_weight: float = 0.0
    adjacency: tuple | None = None
    adjacency_step: int | None = None
    poly: tuple = ()
    axis: str | None = None
    group_axis: str | None = None
    degree: int | None = None
    lo: int | None = None
    hi: int | None = None
    segments: tuple | None = None
    damp: float = 0.0
    groups: str | None = None
    exact_group_rows: bool = False
    exclude_degree: int | None = None
    exclude_axis: str | None = None
    signature_degree: int | str | None = None
    exclude_overlap: tuple | None = None
    render: str | None = None
    coefficient: object = None
    basis: object = None
    name: str | None = None

    def __post_init__(self):
        if self.kind not in OFFSET_KINDS:
            raise ValueError(f"offset term kind must be one of {OFFSET_KINDS}, got {self.kind!r}")
        object.__setattr__(self, 'poly', tuple(
            p if isinstance(p, PolyConstraint) else PolyConstraint(**p) for p in self.poly))
        if self.adjacency is not None:
            object.__setattr__(self, 'adjacency', tuple(self.adjacency))
        if self.kind == 'polybasis' and self.degree is None:
            raise ValueError("a polybasis offset term needs a degree")
        if self.kind == 'grouped' and not self.groups:
            raise ValueError("a grouped offset term needs `groups` (a frame variable, e.g. 'detector')")
        if self.exact_group_rows and self.kind not in ('fixed', 'grouped'):
            raise ValueError("exact_group_rows applies to fixed / grouped terms")
        if self.coefficient is not None and self.basis is not None:
            raise ValueError("an offset term takes `coefficient` (one function) or `basis` (several), not both")
        if self.basis is not None and self.kind == 'polybasis':
            raise ValueError("a polybasis term takes a `coefficient`, not a `basis`")
        if self.exclude_degree is not None:
            if int(self.exclude_degree) < 0:
                raise ValueError("exclude_degree is at least 0")
            if self.kind == 'polybasis' or self.n_basis != 1:
                raise ValueError("exclude applies to a free / fixed / grouped term with one unknown per chunk")
        if self.exclude_overlap is not None:
            object.__setattr__(self, 'exclude_overlap', tuple(self.exclude_overlap))
            if self.kind == 'polybasis' or self.n_basis != 1:
                raise ValueError("exclude_overlap applies to a free / fixed / grouped term with one unknown per chunk")
            if self.exclude_degree is not None:
                raise ValueError("a term takes one exclusion: exclude_degree or exclude_overlap")
        if self.signature_degree is not None and self.signature_degree != 'all' \
                and not 1 <= int(self.signature_degree) <= 2:
            raise ValueError("signature_degree is 1, 2 (the deflation's polynomial patterns go to degree 2) or 'all'")

    @property
    def n_basis(self) -> int:
        """The number of known functions per chunk: the ``n`` of ``basis``, 1 without one."""
        if self.basis is None:
            return 1
        b = self.basis
        return int(b.n) if isinstance(b, Basis) else int(dict(b).get('n', 1))


@dataclass(frozen=True)
class PriorSpec:
    """One ``[[model.prior]]`` entry: ``function(*terms, **params)`` returns linear
    rows on the unknowns of the named ``terms`` (sky-term names, offset-term
    names), scaled by ``weight``. ``function`` is a ready-made prior
    of :mod:`selfcal.models.priors` by its bare name or any
    ``"package.module:name"``."""
    terms: tuple
    function: object
    params: tuple = ()
    weight: float = 1.0
    name: str | None = None

    @classmethod
    def from_config(cls, d) -> PriorSpec:
        """One ``[[model.prior]]`` table as a :class:`PriorSpec`.

        ``term`` (or ``terms``; a string is one term) and ``function`` are required;
        ``params``, ``weight`` (default 1.0) and ``name`` are optional, and any other
        key is an inline parameter of the function, merged into ``params``. A
        :class:`PriorSpec` is returned unchanged; a table without a term or a
        function raises ``ValueError``.
        """
        if isinstance(d, PriorSpec):
            return d
        d = dict(d)
        terms = d.pop('terms', None) or d.pop('term', None)
        if terms is None:
            raise ValueError(f"a prior needs `term` (or `terms`): {d}")
        terms = (terms,) if isinstance(terms, str) else tuple(terms)
        if 'function' not in d:
            raise ValueError(f"a prior needs `function`: {d}")
        function = d.pop('function')
        params = dict(d.pop('params', None) or {})
        weight = float(d.pop('weight', 1.0))
        name = d.pop('name', None)
        params.update(d)
        return cls(terms=terms, function=function, params=_hashable(params), weight=weight, name=name)

    def callable(self):
        """The prior function itself, resolved from ``function``.

        A bare name (no ``:``) is taken from :mod:`selfcal.models.priors`, the
        module of the ready-made priors (``ValueError`` when it has no such name);
        an import path ``"package.module:name"`` is imported
        (:func:`load_function`); a callable is returned as is.
        """
        f = self.function
        if isinstance(f, str) and ':' not in f:
            from . import priors
            if not hasattr(priors, f):
                raise ValueError(f"unknown prior {f!r}: a ready-made one of selfcal.models.priors "
                                 f"({priors.__all__[2:]}) or 'package.module:name'")
            return getattr(priors, f)
        return load_function(f)


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkMapSpec:
    """One chunk map of the model: ``function`` (an import path ``package.module:name``, or the
    function itself) called as ``function(geometry, **params)`` on the instrument's detector
    geometry returns the partition, a :class:`~selfcal.instruments.base.Chunks`; ``tag`` joins the
    product name when the map is the primary one."""
    name: str
    function: object
    tag: str = ''
    params: dict = field(default_factory=dict)

    def evaluate(self, geom):
        """The map on ``geom``: ``function(geom, **params)``."""
        return load_function(self.function)(geom, **dict(self.params))

    def key(self):
        """The map as plain data (what the run engine keeps an evaluated geometry under)."""
        fn = self.function
        ref = fn if isinstance(fn, str) else f"{getattr(fn, '__module__', '')}:{getattr(fn, '__qualname__', repr(fn))}"
        return {'name': self.name, 'function': ref, 'tag': self.tag, 'params': _hashable(self.params)}


@dataclass(frozen=True)
class ModelSpec:
    """A self-calibration model: chunk maps, sky and offset terms, variables, weight, priors.

    ``chunk_maps`` holds the model's chunk maps (:class:`ChunkMapSpec`, in order, the first
    primary; none: the instrument's defaults, which the run engine supplies).

    ``sky`` holds the sky terms (:class:`SkyTerm`; at least one, by default one
    constant ``continuum`` term) and ``offset`` the offset terms
    (:class:`OffsetTerm`; none by default). ``variables`` are the model's own data variables
    (:class:`VariableSpec`), ``weight`` an optional function of data variables
    that multiplies every observation's weight (a
    :class:`~selfcal.models.sky_model.Coefficient` or its config form), and
    ``priors`` the prior functions (:class:`PriorSpec`). ``mosaic`` records the
    coadd the model was lowered for: ``'full'`` (the mosaic and the instrument's
    auxiliary coadds, such as wavelength maps), ``'no_wav'`` (the mosaic only) or
    ``'none'``. Terms, variables and priors may
    be given in their config forms (dicts; ``variables`` as ``{name: table}``),
    which are converted on construction. The constructor raises ``ValueError``
    when there is no sky term, or when a variable is defined twice or named like
    a built-in.

    Build one from a ``[model]`` table (:meth:`from_config`) or in Python,
    validate it against an instrument (:meth:`check`)
    and lower it into the solver's inputs with :meth:`build_sky_model`,
    :meth:`build_offset_model`, :meth:`build_variables`, :meth:`build_weight`,
    :meth:`build_priors` and :meth:`setup_kwargs`.
    """
    sky: tuple = (SkyTerm(),)
    offset: tuple = ()
    mosaic: str = 'full'                 # full | no_wav | none
    variables: tuple = ()
    weight: object = None
    priors: tuple = ()
    chunk_maps: tuple = ()

    def __post_init__(self):
        maps = self.chunk_maps
        if isinstance(maps, dict):
            maps = tuple(ChunkMapSpec(name=str(k), **dict(v)) for k, v in maps.items())
        object.__setattr__(self, 'chunk_maps', tuple(maps))
        object.__setattr__(self, 'sky', tuple(
            t if isinstance(t, SkyTerm) else _sky_term(t) for t in self.sky))
        object.__setattr__(self, 'offset', tuple(
            t if isinstance(t, OffsetTerm) else OffsetTerm(**t) for t in self.offset))
        vs = self.variables
        if isinstance(vs, dict):
            vs = tuple(VariableSpec.from_config(k, v) for k, v in vs.items())
        object.__setattr__(self, 'variables', tuple(vs))
        object.__setattr__(self, 'priors', tuple(PriorSpec.from_config(p) for p in self.priors))
        if not self.sky:
            raise ValueError("the model needs at least one sky term")
        names = [v.name for v in self.variables]
        dup = sorted({n for n in names if names.count(n) > 1} | (set(names) & set(BUILTIN_VARIABLES)))
        if dup:
            raise ValueError(f"data variables defined twice (or shadowing a built-in): {dup}")

    # ---- from a config table -------------------------------------------------------
    @classmethod
    def from_config(cls, table: dict) -> ModelSpec:
        """The ``[model]`` table: ``chunk_maps = {name = {function=..., tag=..., params={...}}}``,
        ``sky = [{name=..., coefficient={...}, damp_weight=...}, ...]``,
        ``offset = [{map=..., kind=..., coefficient/basis={...}, poly=[...]}, ...]``,
        ``variables = {name = {<source> = ..., ...}}``, ``weight = {variable, function, ...}``,
        ``prior = [{term(s)=..., function=..., weight=..., <params>}, ...]``, ``mosaic``."""
        t = dict(table)
        chunk_maps = t.pop('chunk_maps', None) or {}
        if 'scalar' in t:
            raise ValueError("[model] scalar: the per-frame scalar is an offset term, "
                             "offset = [{map = 'detector', name = 'scalar'}] (sc.Offsets('scalar', on='detector'))")
        sky = t.pop('sky', None) or [{}]
        offset = t.pop('offset', None) or []
        mosaic = str(t.pop('mosaic', 'full'))
        variables = t.pop('variables', None) or {}
        weight = t.pop('weight', None)
        priors = t.pop('prior', None) or t.pop('priors', None) or []
        if t:
            raise ValueError(f"unknown [model] keys {sorted(t)}; expected chunk_maps / sky / offset / variables / "
                             f"weight / prior / mosaic")
        return cls(sky=tuple(_sky_term(dict(s)) for s in sky),
                   offset=tuple(OffsetTerm(**dict(o)) for o in offset), mosaic=mosaic,
                   variables=tuple(VariableSpec.from_config(k, v) for k, v in dict(variables).items()),
                   weight=None if weight is None else dict(weight),
                   priors=tuple(PriorSpec.from_config(p) for p in priors),
                   chunk_maps=dict(chunk_maps))

    # ---- properties ------------------------------------------------------------------------
    @property
    def has_coefficients(self) -> bool:
        """Whether any sky term has a coefficient (reads data variables)."""
        return any(t.coefficient is not None for t in self.sky)

    @property
    def needs_variables(self) -> bool:
        """Whether anything in the model reads data variables."""
        return (self.has_coefficients or bool(self.variables) or self.weight is not None
                or any(t.coefficient is not None or t.basis is not None for t in self.offset))

    def variable_names(self, geom=None, frame_variables=()) -> list[str]:
        """Every data-variable name the model may read: built-ins, the
        instrument's detector maps (and their aliases), its frame variables, the
        model's own."""
        names = list(BUILTIN_VARIABLES) + [v.name for v in self.variables] + list(frame_variables)
        if geom is not None:
            names += list(getattr(geom, 'aux', {}) or {})
            if getattr(geom, 'wavelength_key', None):
                names.append('wavelength')
            if getattr(geom, 'width_key', None):
                names.append('bandwidth')
        return names

    def referenced_variables(self) -> set:
        """The data-variable names the model's functions read by name: sky and
        offset coefficients and bases, the weight, derived variables' inputs."""
        out = set()

        def add(c):
            if c is None:
                return
            if isinstance(c, (Coefficient, Basis)):
                out.update(c.variables)
                return
            d = dict(c)
            v = d.get('variable')
            if v is not None:
                out.update([v] if isinstance(v, str) else list(v))
            if d.get('width'):
                out.add(d['width'])

        for t in self.sky:
            add(t.coefficient)
        for t in self.offset:
            add(t.coefficient)
            add(t.basis)
        add(self.weight)
        for v in self.variables:
            out.update(v.inputs)
        return out

    def _known(self, frame_variables):
        return tuple(BUILTIN_VARIABLES) + tuple(v.name for v in self.variables) + tuple(frame_variables)

    def offset_names(self, geom=None) -> list[str]:
        """How priors refer to the offset terms: ``name``, else the map's name
        (numbered when several terms share a map)."""
        raw = []
        for term in self.offset:
            if term.name:
                raw.append(term.name)
            elif term.map:
                raw.append(term.map)
            else:
                raw.append(geom.primary if geom is not None else 'primary')
        out = []
        for i, n in enumerate(raw):
            out.append(n if raw.count(n) == 1 or self.offset[i].name else f'{n}{raw[:i].count(n)}')
        return out

    def check(self, geom, frame_variables=()):
        """Raise a clear error when the model refers to something the instrument
        or the model does not provide: a data variable, a chunk map, a chunk
        axis, a prior's term or function."""
        known = self._known(frame_variables)
        self.build_sky_model(geom, log=lambda *a, **k: None, frame_variables=frame_variables)
        for v in self.variables:
            if v.scope == 'derived':
                for i in v.inputs:
                    resolve_variable(i, geom, known)
        for term in self.offset:
            cm = _chunk_map(term, geom)
            names = list(cm.axes.names) if cm.axes is not None else []
            smooth_axes = cm.adjacency_axes if term.adjacency is None else tuple(term.adjacency)
            if any(p.axis is None for p in term.poly) and not smooth_axes:
                raise ValueError(f"offset term on {cm.name!r}: a polynomial constraint without an axis follows "
                                 f"the term's first smoothing axis, and the term is smoothed along none")
            axes = list(term.adjacency or ()) + [p.axis for p in term.poly if p.axis is not None]
            if term.kind == 'polybasis':
                axes += [term.axis or cm.spectral_axis, term.group_axis or cm.group_axis]
            bad = [a for a in axes if a not in names]
            if bad:
                raise ValueError(f"offset term on {cm.name!r} names axes {bad}; the map's axes are {names}")
            if term.kind == 'grouped' and term.groups not in known + ('exposure', 'detector'):
                raise ValueError(f"offset term on {cm.name!r} groups frames by {term.groups!r}, which is not a "
                                 f"frame variable (known: {sorted(set(known) - set(BUILTIN_VARIABLES))})")
            self._offset_function(term, geom, known, log=lambda *a, **k: None)
        if self.weight is not None:
            build_coefficient(self.weight, geom, name='weight', log=lambda *a, **k: None, known=known)
        terms = set(t.name for t in self.sky) | set(self.offset_names(geom))
        for p in self.priors:
            bad = [t for t in p.terms if t not in terms]
            if bad:
                raise ValueError(f"prior {p.name or p.function!r} names terms {bad}; the model's terms are "
                                 f"{sorted(terms)}")
            p.callable()

    # ---- lowering: sky ---------------------------------------------------------------------
    def build_sky_model(self, geom, log=print, frame_variables=()) -> SkyModel:
        """The solver's SkyModel: one component per term, coefficients resolved
        against the data variables."""
        known = self._known(frame_variables)
        comps = []
        for term in self.sky:
            coeff = None if term.coefficient is None else build_coefficient(
                term.coefficient, geom, name=term.name, log=log, known=known)
            comps.append(SkyComponent(name=term.name, coefficient=coeff,
                                      damp_weight=None if term.damp_weight is None else float(term.damp_weight)))
        return SkyModel(tuple(comps))

    def build_weight(self, geom, frame_variables=(), log=print):
        """The observation weight (a Coefficient of data variables), or None."""
        if self.weight is None:
            return None
        return build_coefficient(self.weight, geom, name='weight', log=log, known=self._known(frame_variables))

    # ---- lowering: offsets -------------------------------------------------------------------
    def build_offset_model(self, geom, n_frames, frame_groups=None, log=print, frame_variables=()) -> OffsetModel:
        """``frame_groups``: ``{name: per-frame values}`` the grouped terms group by
        (the instrument's groupings and the solve's frame variables)."""
        known = self._known(tuple(frame_variables) + tuple(frame_groups or ()))
        blocks = [self._offset_block(term, geom, n_frames, frame_groups or {}, log, known)
                  for term in self.offset]
        return OffsetModel(blocks)

    def setup_kwargs(self) -> dict:
        """The solver options the terms' priors imply beyond the OffsetModel:
        per-map damping and the exact grouped rows. Empty when no term asks for
        them (the historical recipes)."""
        kw = {}
        if any(t.damp for t in self.offset):
            kw['damp_offset_maps'] = [float(t.damp) for t in self.offset]
        exact = [m for m, t in enumerate(self.offset) if t.exact_group_rows]
        if exact:
            kw['group_adjacency_maps'] = exact
        return kw

    def column_exclusions(self, geom, layout, window=None, log=print) -> list:
        """The terms' hard constraints as the solver takes them (``apply_lsqr(exclusions=...)``):
        ``[(cols, P), ...]`` with ``cols`` ``(G, n)`` the full-layout columns of each offset group's
        chunks along one line of the excluded axis and ``P`` ``(n, k + 1)`` an orthonormal basis of
        the polynomials of degree <= k in the chunks' coordinate along that axis (degree 0 with no
        axis: one line of all the map's chunks, ``P`` the constant). ``layout`` is the
        :class:`~selfcal.core.layout.SystemLayout` of the assembled system. Empty without ``exclude``
        terms (the historical recipes). A sky term's ``exclude`` gives one group over the term's pixel
        columns with the shapes it returns (orthonormalised), evaluated at the grid's pixel coordinates.
        An offset term's ``exclude_overlap`` gives, per group, the term's chunks over the pixels the job's
        observations reach (``window``, the job's per-pixel valid weight on the detector grid, ``> 0``, and the
        one pixel around it that bilinear sampling touches; None: every pixel) and an orthonormal basis of the
        patterns it shares with the named terms (:meth:`_overlap`)."""
        out = []
        H, W = (int(v) for v in layout.ref_shape)
        for j, term in enumerate(self.sky):
            if term.exclude is None:
                continue
            fn = import_name(term.exclude) if isinstance(term.exclude, str) else term.exclude
            vv, uu = np.mgrid[0:H, 0:W]
            u = (uu.ravel() / max(W - 1, 1)).astype(np.float64)
            v = (vv.ravel() / max(H - 1, 1)).astype(np.float64)
            S = np.asarray(fn(u, v), dtype=np.float64)
            if S.ndim == 1:
                S = S[:, None]
            if S.shape[0] != H * W or S.ndim != 2:
                raise ValueError(f"sky term {term.name!r}: exclude's shapes must be (pixels, r) over the {H}x{W} grid, got {S.shape}")
            P = np.linalg.qr(S)[0]
            cols = (j * H * W + np.arange(H * W, dtype=np.int64))[None, :]
            out.append((cols, P))
        for m, term in enumerate(self.offset):
            if term.exclude_overlap is not None:
                if layout.det_template_arr_list[m] is not None:
                    raise ValueError(f"offset term {self.offset_names(geom)[m]!r}: exclude does not apply to a template term")
                chunks, P = self._overlap(m, geom, window, log)
                if P.shape[1]:
                    n_chunks, n_groups = int(layout.num_chunks_list[m]), int(layout.num_offset_groups_list[m])
                    cols = int(layout.col_bases[m]) + (np.arange(n_groups, dtype=np.int64)[:, None] * n_chunks) + chunks[None, :]
                    out.append((cols, P))
                continue
            if term.exclude_degree is None:
                continue
            cm = _chunk_map(term, geom)
            k = int(term.exclude_degree)
            axes = cm.axes
            if axes is None or len(axes) == 0:
                raise ValueError(f"offset term on map {cm.name!r}: exclude needs a chunk map with axes")
            axis = term.exclude_axis
            if axis is None and len(axes) == 1:
                axis = axes.names[0]
            if axis is None and k != 0:
                raise ValueError(f"offset term on map {cm.name!r}: exclude=Poly({k}) needs along= (axes: "
                                 f"{axes.names}); only degree 0, the constant, takes every chunk at once")
            if axis is not None and axis not in axes:
                raise ValueError(f"offset term on map {cm.name!r}: no chunk axis {axis!r} (axes: {axes.names})")
            if layout.det_template_arr_list[m] is not None:
                raise ValueError(f"offset term on map {cm.name!r}: exclude does not apply to a template term")
            n_chunks = int(layout.num_chunks_list[m])
            n_groups = int(layout.num_offset_groups_list[m])
            base = int(layout.col_bases[m])
            if axis is None:                                    # the constant over all of a group's chunks
                ids = np.arange(n_chunks)
                coord = np.zeros(n_chunks)
                key = np.zeros(n_chunks, dtype=np.int64)
            else:
                coord = np.asarray(axes[axis].of_chunk, dtype=np.float64)
                others = [np.asarray(a.of_chunk) for a in axes.others(axis)]
                ids = np.arange(min(n_chunks, coord.size))
                key = np.zeros(ids.size, dtype=np.int64)
                for o in others:
                    key = key * (int(o.max()) + 1) + o[ids]
            for line in np.unique(key):
                chunks = ids[key == line]
                if chunks.size < k + 2:
                    continue                                    # the line would be constrained to zero: left free
                t = coord[chunks]
                t = (t - t.mean()) / max(float(np.ptp(t)) / 2.0, 1e-300)
                V = np.vander(t, k + 1, increasing=True)
                P = np.linalg.qr(V)[0]
                cols = base + (np.arange(n_groups, dtype=np.int64)[:, None] * n_chunks) + chunks[None, :]
                out.append((cols, P))
        return out

    def _overlap(self, m, geom, window, log):
        """``(chunks, P)``: the patterns offset term ``m`` shares with the terms its ``exclude_overlap`` names,
        as directions over its chunks. ``chunks`` are the term's chunk ids on the pixels the job's
        observations reach (``window > 0`` and the one pixel around it; None: every pixel) and ``P``
        ``(len(chunks), k)`` an orthonormal basis of the term's coefficients that make a fixed pattern of the
        named terms there.

        Over those pixels, the term makes the patterns ``a[chunk(p)] f(p)`` (``f`` its function, 1 without
        one); a named term with ``n`` functions ``g_k`` on its own chunk map makes, with the same coefficients
        in every frame, ``sum b[chunk'(p), k] g_k(p)``: the span of ``1[chunk'(p) = c] g_k(p)``. The shared
        patterns are the intersection of the two spans: the combinations ``w`` of the named terms' patterns
        that equal their projection on the term's span (``a[chunk] = sum_p f w / sum_p f^2`` per chunk), found
        from the null space of ``w - projection`` over an orthonormal basis of the named terms' patterns.
        Every function is evaluated in float64 at the pixel centres (``det_x``, ``det_y`` = column, row)."""
        names = self.offset_names(geom)
        term = self.offset[m]
        index = {n: j for j, n in enumerate(names)}
        bad = [n for n in term.exclude_overlap if n not in index or index[n] == m]
        if bad:
            raise ValueError(f"offset term {names[m]!r}: exclude_overlap names {bad}, not another offset term "
                             f"of the model (the offset terms: {names})")
        cm = _chunk_map(term, geom)
        det = np.asarray(cm.det).ravel()
        inside = det >= 0
        if window is not None:
            # the window and the pixels around it that the job's observations reach: an observation is solved when
            # its bilinear sample of the window is above zero, so its four detector neighbours lie within one pixel of it
            from scipy.ndimage import binary_dilation
            reach = binary_dilation(np.asarray(window) > 0, structure=np.ones((3, 3), dtype=bool))
            inside &= reach.ravel()
        pix = np.flatnonzero(inside)
        chunk_ids, cid = np.unique(det[pix], return_inverse=True)
        known = self._known(())

        def values(j):                  # (pixels, n) float64: term j's functions at the pixel centres (ones: none)
            basis = self._offset_function(self.offset[j], geom, known, log=lambda *a, **k: None)
            if basis is None:
                return np.ones((pix.size, 1))
            return basis.evaluate(self._detector_values(basis.variables, geom, pix, names[j]), dtype=np.float64)

        f = values(m)[:, 0]
        den = np.bincount(cid, f * f, minlength=chunk_ids.size)
        if np.any(den <= 0):
            raise ValueError(f"offset term {names[m]!r}: its function is zero on every pixel of a chunk")

        def project(w):                 # the term's chunk coefficients closest to each column of w, and their pattern
            a = np.stack([np.bincount(cid, f * w[:, k], minlength=chunk_ids.size) for k in range(w.shape[1])], axis=1)
            a /= den[:, None]
            return a, a[cid] * f[:, None]

        cols = []
        for n in term.exclude_overlap:
            t = self.offset[index[n]]
            if t.kind == 'polybasis':
                raise ValueError(f"offset term {names[m]!r}: exclude_overlap names {n!r}, a polynomial-basis term")
            g = values(index[n])
            c_det = np.asarray(_chunk_map(t, geom).det).ravel()[pix]
            for c in np.unique(c_det[c_det >= 0]):
                on = (c_det == c)[:, None]
                cols.append(np.where(on, g, 0.0))
        if not cols:
            return chunk_ids.astype(np.int64), np.zeros((chunk_ids.size, 0))
        W = np.concatenate(cols, axis=1)
        U, sv, _ = np.linalg.svd(W, full_matrices=False)
        Q = U[:, sv > _OVERLAP_TOL * sv.max()] if sv.size and sv.max() > 0 else U[:, :0]
        _, rest = project(Q)
        _, s2, Vt = np.linalg.svd(Q - rest, full_matrices=False)
        shared = Q @ Vt[s2 <= _OVERLAP_TOL].T                          # the named terms' patterns the term can make
        a, _ = project(shared)
        P = np.linalg.qr(a)[0] if a.shape[1] else a
        log(f"[model] {names[m]}: exclude=Overlap{tuple(term.exclude_overlap)}: {P.shape[1]} shared pattern(s) "
            f"over {pix.size} pixels ({chunk_ids.size} chunks; the named terms make {Q.shape[1]} fixed patterns there)",
            flush=True)
        return chunk_ids.astype(np.int64), P

    def _detector_values(self, names, geom, pix, term_name):
        """The data variables ``names`` at the detector pixels ``pix`` (flat indices), float64: the built-in
        ``det_x`` / ``det_y`` (column / row), the model's detector maps, the instrument's detector maps
        (``geom.aux``) and variables derived from them. Raises ``ValueError`` for any other variable: it has
        no value fixed on the detector."""
        H, W = geom.shape
        own = {v.name: v for v in self.variables}
        out = {}

        def get(name):
            if name in out:
                return out[name]
            if name == 'det_x':
                val = (pix % W).astype(np.float64)
            elif name == 'det_y':
                val = (pix // W).astype(np.float64)
            elif name in own and own[name].scope == 'detector':
                v = own[name]
                val = _map_source(v, lambda fn: fn(geom, **dict(v.params)), geom.shape,
                                  'detector grid').ravel()[pix].astype(np.float64)
            elif name in own and own[name].scope == 'derived':
                v = own[name]
                fn = ImportedFunction(str(v.value), v.params) if isinstance(v.value, str) else _bind(v.value, dict(v.params))
                val = np.asarray(fn(*[get(i) for i in v.inputs]), dtype=np.float64)
            elif name in (getattr(geom, 'aux', None) or {}):
                val = np.asarray(geom.aux[name], dtype=np.float64).ravel()[pix]
            else:
                raise ValueError(f"exclude_overlap: offset term {term_name!r} reads {name!r}, which has no value fixed "
                                 f"on the detector (only det_x, det_y, detector maps and variables derived from them "
                                 f"make a fixed pattern)")
            out[name] = np.broadcast_to(val, pix.shape)
            return out[name]

        for n in names:
            get(n)
        return out

    def signature_exclusions(self, layout) -> list:
        """The terms whose response to smooth sky fields is excluded (``Offsets(exclude=SkySignature(k))``), as the
        solver takes them (``apply_lsqr(signatures=...)``): ``[(cols, k, name, chunk), ...]`` with ``cols`` the
        term's full-layout columns (every group, chunk and function), ``k`` the largest degree of the sky
        polynomials whose response the term may not carry (``'all'``: every pattern but the constant), and ``chunk``
        each column's place within its group (the columns of one ``chunk`` value, one per group, hold the term's
        detector-fixed configurations). Empty without such terms."""
        def deg(k):
            return k if k == 'all' else int(k)

        out = []
        for m, term in enumerate(self.offset):
            k = term.signature_degree
            if k is None:
                continue
            base, nc = int(layout.col_bases[m]), int(layout.num_chunks_list[m])
            cols = np.arange(base, int(layout.col_bases[m + 1]), dtype=np.int64)
            out.append((cols, deg(k), term.name or term.map or 'primary', (cols - base) % nc))
        return out

    def _offset_function(self, term, geom, known, log):
        """The term's known functions of data variables as a Basis (None: none)."""
        spec = term.coefficient if term.coefficient is not None else term.basis
        if spec is None:
            return None
        if isinstance(spec, Basis):
            return spec
        n = 1
        if term.basis is not None:
            spec = dict(spec)
            n = int(spec.pop('n', 1))
        coeff = build_coefficient(spec, geom, name=term.name or term.map or 'offset', log=log, known=known)
        return Basis(coeff, n)

    def _offset_block(self, term, geom, n_frames, frame_groups, log, known=()):
        cm = _chunk_map(term, geom)
        basis = self._offset_function(term, geom, known, log)
        if term.kind == 'polybasis':
            axis = term.axis or cm.spectral_axis
            group = term.group_axis or cm.group_axis
            if axis is None or group is None or term.lo is None or term.hi is None:
                raise ValueError(f"polybasis term on map {cm.name!r} needs axis, group_axis, lo, hi")
            pb = poly_basis_along(cm.axes, axis, group, int(term.degree), int(term.lo), int(term.hi),
                                  segments=term.segments)
            log(f"[model] {cm.name}: polynomial-basis offset, degree {term.degree} in {axis!r} per "
                f"{group!r} over [{term.lo}, {term.hi}]"
                + (f", {len(term.segments)} segments" if term.segments else ""), flush=True)
            return OffsetBlock(chunk_map=cm.det, poly_basis=pb, basis=basis)
        # free / fixed / grouped: adjacency + soft polynomials
        axes = cm.adjacency_axes if term.adjacency is None else tuple(term.adjacency)
        if term.adjacency_step is not None:
            if len(axes) != 1:
                raise ValueError("adjacency_step needs exactly one adjacency axis")
            adj = adjacency_along(cm.det, cm.axes, axes[0], step=int(term.adjacency_step))
        else:
            adj = adjacency_union(cm.det, cm.axes, axes) if axes else None
        groups = []
        for pc in term.poly:
            axis = pc.axis if pc.axis is not None else (axes[0] if axes else None)
            if axis is None:
                raise ValueError(f"offset term on {cm.name!r}: a polynomial constraint without an axis follows "
                                 f"the term's first smoothing axis, and the term is smoothed along none")
            size = cm.axes[axis].size
            if size < pc.degree + 2:
                log(f"[model] axis {axis!r} has {size} values < degree+2={pc.degree + 2}: "
                    f"skipping the (vacuous) polynomial constraint along it.")
                continue
            chains, stencil = poly_chains_along(cm.axes, axis, int(pc.degree), pc.lo, pc.hi)
            groups.append({'chains': chains, 'stencil': stencil, 'weight': pc.weight})
        nb = 1 if basis is None else basis.n
        if nb > 1:
            # Unknowns are (chunk, function): chunk pairs / chains act per function.
            if adj is not None:
                adj = tuple(np.concatenate([np.asarray(a) * nb + k for k in range(nb)]) for a in adj)
            groups = [{'chains': np.concatenate([np.asarray(g['chains']) * nb + k for k in range(nb)]),
                       'stencil': g['stencil'], 'weight': g['weight']} for g in groups]
        det_groups = None
        if term.kind == 'fixed':
            det_groups = np.zeros(n_frames, dtype=int)
        elif term.kind == 'grouped':
            if term.groups not in frame_groups:
                raise ValueError(f"offset term on {cm.name!r} groups frames by {term.groups!r}, which is "
                                 f"not a frame variable of this solve (has {sorted(frame_groups)})")
            det_groups = np.asarray(frame_groups[term.groups])
            if det_groups.shape != (n_frames,):
                raise ValueError(f"frame grouping {term.groups!r} has shape {det_groups.shape}, "
                                 f"expected ({n_frames},)")
        if basis is not None:
            log(f"[model] {cm.name}: offset times {basis.n} function(s) "
                f"{basis.coefficient.describe()}", flush=True)
        return OffsetBlock(
            chunk_map=cm.det, adj_info=adj, reg_weight=term.reg_weight,
            poly_constraints=groups or None, det_groups=det_groups, basis=basis)

    # ---- lowering: variables -------------------------------------------------------------------
    def build_variables(self, geom, frames, *, ref_shape=None, ref_wcs=None, frame_variables=None,
                        log=print) -> VariableSet:
        """Resolve ``[model.variables]`` for one solve's frame list into a
        :class:`~selfcal.models.variables.VariableSet`, together with the
        instrument's frame variables (``frame_variables``: ``{name: values}``)."""
        frame = dict(frame_variables or {})
        detector, sky, derived, functions, layers = {}, {}, {}, {}, []
        headers = [v for v in self.variables if v.source == 'header']
        if headers:
            from ..io.frames import frame_header_values
            vals = frame_header_values(frames, [v.value for v in headers])
            for v in headers:
                arr = vals[v.value]
                if v.default is not None:
                    missing = (np.array([x is None for x in arr]) if arr.dtype == object else ~np.isfinite(arr))
                    arr = arr.copy()
                    arr[missing] = v.default
                frame[v.name] = arr
                log(f"[model] variable {v.name!r}: header keyword {v.value!r} of {len(frames)} frames", flush=True)
        for v in self.variables:
            p = dict(v.params)
            if v.source == 'per_frame':
                fn = load_function(v.value)
                if v.inputs:
                    missing = [i for i in v.inputs if i not in frame]
                    if missing:
                        raise ValueError(f"per-frame variable {v.name!r} reads {missing}, which are not frame "
                                         f"variables (define them first; have {sorted(frame)})")
                    out = fn(*[frame[i] for i in v.inputs], **p)
                else:
                    out = fn(list(frames), **p)
                out = np.asarray(out)
                if out.shape != (len(frames),):
                    raise ValueError(f"per-frame variable {v.name!r}: {v.value} returned shape {out.shape}, "
                                     f"expected ({len(frames)},)")
                frame[v.name] = out
            elif v.source == 'detector':
                detector[v.name] = _map_source(v, lambda fn: fn(geom, **p), geom.shape, 'detector grid')
            elif v.source == 'sky':
                sky[v.name] = _map_source(v, lambda fn: fn(ref_wcs, ref_shape, **p), ref_shape, 'reference grid')
            elif v.source == 'sky_cal':
                from ..io.calfile import CalFile
                with CalFile(v.value) as cal:
                    m = cal.sky(v.term if v.term is not None else 0)
                if ref_shape is not None and tuple(m.shape) != tuple(ref_shape):
                    raise ValueError(f"sky variable {v.name!r}: {v.value} is on a {m.shape} grid, the solve's "
                                     f"reference grid is {tuple(ref_shape)}")
                sky[v.name] = np.nan_to_num(np.asarray(m, dtype=np.float32))
            elif v.source == 'layer':
                layers.append(str(v.value))
                if str(v.value) != v.name:
                    derived[v.name] = Derived((str(v.value),), _identity)
            elif v.source == 'function':
                derived[v.name] = Derived(v.inputs, ImportedFunction(str(v.value), v.params)
                                          if isinstance(v.value, str) else _bind(v.value, p))
            elif v.source == 'frame_function':
                functions[v.name] = FrameFunction(ImportedFunction(str(v.value), v.params)
                                                  if isinstance(v.value, str) else _bind(v.value, p))
        return VariableSet(detector=detector, frame=frame, sky=sky, layers=tuple(layers), derived=derived,
                           frame_functions=functions)

    # ---- lowering: priors -----------------------------------------------------------------------
    def build_priors(self, geom, variables=None, n_frames=0):
        """The ``[[model.prior]]`` entries as ``setup_lsqr(priors=...)`` callables."""
        if not self.priors:
            return []
        from .priors import ModelPrior, TermInfo
        sky_index = {t.name: j for j, t in enumerate(self.sky)}
        off_index = {n: m for m, n in enumerate(self.offset_names(geom))}
        axes = [getattr(_chunk_map(t, geom), 'axes', None) for t in self.offset]
        nbs = [t.n_basis for t in self.offset]

        def describe(info, name):
            L = info.layout
            pc = info.pixel_counts
            if name in sky_index:
                j = sky_index[name]
                n_sky = L.num_sky
                return TermInfo(name=name, kind='sky', shape=tuple(L.ref_shape), col_base=j * n_sky,
                                coverage=pc[j * n_sky:(j + 1) * n_sky].reshape(L.ref_shape),
                                variables=variables, n_frames=L.num_frames)
            m = off_index[name]
            nb = nbs[m]
            shape = (L.num_offset_groups_list[m], L.num_chunks_list[m] // nb, nb)
            sl = L.offset_slice(m)
            return TermInfo(name=name, kind='offset', shape=shape, col_base=sl.start,
                            coverage=pc[sl].reshape(shape), frame_group=np.asarray(L.frame_to_group_list[m]),
                            variables=variables, axes=axes[m], n_frames=L.num_frames)

        return [ModelPrior(p.name or f"{getattr(p.function, '__name__', p.function)} on {', '.join(p.terms)}",
                           p.terms, p.callable(), dict(p.params), p.weight, describe) for p in self.priors]


def _identity(x):
    return x


class _bind:
    """A plain callable with bound keyword parameters (picklable when ``fn`` is)."""

    def __init__(self, fn, params):
        self.fn = fn
        self.params = dict(params)

    def __call__(self, *args):
        return self.fn(*args, **self.params)


def _map_source(v, call, shape, where):
    """A detector / sky map: an array, a function (``call(fn)``) or a .npy / .fits file."""
    val = v.value
    if isinstance(val, np.ndarray):
        m = np.asarray(val, dtype=np.float32)
    elif callable(val) or (isinstance(val, str) and ':' in val and not os.path.exists(val)):
        m = np.asarray(call(load_function(val)), dtype=np.float32)
    elif str(val).endswith('.npy'):
        m = np.load(val).astype(np.float32)
    else:
        from astropy.io import fits
        m = np.asarray(fits.getdata(val), dtype=np.float32)
    if shape is not None and tuple(m.shape) != tuple(shape):
        raise ValueError(f"variable {v.name!r}: map of shape {m.shape}, expected the {where} {tuple(shape)}")
    return m


#: A combination of the named terms' patterns whose part outside the term's own patterns is below this fraction of
#: its size (or a pattern below this fraction of the largest) is rounding, not structure (float64 evaluation).
_OVERLAP_TOL = 1e-8


def chunk_map_of(term, geom):
    """The chunk map an offset term lives on (``"detector"``: one chunk, unless
    the instrument names a map so)."""
    return _chunk_map(term, geom)


def _chunk_map(term, geom):
    """See :func:`chunk_map_of`."""
    if term.map is None:
        return geom.chunk_map
    if term.map in geom.chunk_maps:
        return geom.chunk_maps[term.map]
    if term.map == DETECTOR_MAP:
        cm = geom.chunk_map
        return dataclasses.replace(
            cm, name=DETECTOR_MAP, det=np.zeros(cm.det.shape, dtype=np.int32),
            grid=np.zeros(cm.grid.shape, dtype=np.int32),
            axes=ChunkAxes.row_major((DETECTOR_MAP,), (1,), ('both',)), adjacency_axes=(),
            spectral_axis=None, group_axis=None, renderers={}, render=None, extra={})
    raise ValueError(f"offset term map {term.map!r} is not a chunk map of the instrument "
                     f"({sorted(geom.chunk_maps)}; or {DETECTOR_MAP!r} for the whole detector)")


def _sky_term(d) -> SkyTerm:
    d = dict(d)
    if 'type' in d:
        raise ValueError("sky terms have no `type`: give `name` and, for a term that is not constant, "
                         "`coefficient = { variable = ..., function = ..., <parameters> }` "
                         f"(got {d})")
    return SkyTerm(**d)


def resolve_variable(name, geom, known=()):
    """The key of a data variable: a name in ``known`` (built-ins, the model's
    variables, the instrument's frame variables), a key of the instrument's
    detector maps ``geom.aux``, or an alias of one — ``wavelength`` (the
    instrument's ``wavelength_key``), ``bandwidth`` (its ``width_key``)."""
    if name in known:
        return name
    aux = getattr(geom, 'aux', None) or {}
    aliases = {}
    if getattr(geom, 'wavelength_key', None):
        aliases['wavelength'] = geom.wavelength_key
    if getattr(geom, 'width_key', None):
        aliases['bandwidth'] = geom.width_key
    key = aliases.get(name, name)
    if key not in aux:
        avail = sorted(set(known)) + sorted(aux) + [f'{a} (= {k})' for a, k in aliases.items()]
        raise ValueError(f"data variable {name!r} is not provided by the instrument or the model "
                         f"(variables: {avail or 'none'})")
    return key


def build_coefficient(spec, geom, name=None, log=print, known=()) -> Coefficient:
    """A :class:`~selfcal.models.sky_model.Coefficient` from its config form (see
    :class:`SkyTerm`), with variable names resolved against the instrument and
    the ``known`` variables."""
    if isinstance(spec, Coefficient):
        keys = tuple(resolve_variable(v, geom, known) for v in spec.main_variables)
        return type(spec)(keys[0] if isinstance(spec.variable, str) else keys, spec.function)
    spec = dict(spec)
    kind = spec.pop('kind', None)           # 'variance': the coefficient is a noise variance (sc.Variance)
    if kind not in (None, 'variance'):
        raise ValueError(f"unknown coefficient kind {kind!r} (None or 'variance')")
    coef = _build_coefficient(spec, geom, name, log, known)
    return NoiseVariance(coef.variable, coef.function) if kind == 'variance' else coef


def _build_coefficient(spec, geom, name, log, known):
    if 'variable' not in spec:
        raise ValueError(f"a coefficient needs `variable` (and a `function`; none = the variable "
                         f"itself): {spec}")
    variable = spec.pop('variable')
    function = spec.pop('function', 'selfcal.models.variables:identity')
    params = dict(spec.pop('params', None) or {})
    params.update(spec)
    many = isinstance(variable, (list, tuple))
    keys = (tuple(resolve_variable(v, geom, known) for v in variable) if many
            else resolve_variable(variable, geom, known))
    if callable(function):
        return Coefficient(keys, _bind(function, params) if params else function)
    function = str(function)
    if ':' in function:
        return Coefficient(keys, ImportedFunction(function, _hashable(params)))
    if function not in BUILTIN_FUNCTIONS:
        raise ValueError(f"unknown coefficient function {function!r}: a built-in {BUILTIN_FUNCTIONS} "
                         f"or a Python function 'package.module:name'")
    if many:
        raise ValueError(f"the built-in {function!r} takes one variable; got {list(variable)}")
    return Coefficient(keys, _builtin(function, params, geom, name, log, known))


def _builtin(function, p, geom, name, log, known=()):
    from .profiles import GaussianProfile, LinearProfile, QuadratureSigma, TemplateProfile

    def need(*keys):
        missing = [k for k in keys if k not in p]
        if missing:
            raise ValueError(f"coefficient {function!r} of term {name!r} needs {missing}")

    if function == 'template':
        # A tabulated function (linear interpolation, zero outside): inline x / y, or an
        # npz file with arrays x / y (keys x_key / y_key), or the SPHEREx template layout
        # (center_um + G_peaknorm; norm = "area" -> G).
        if 'file' in p:
            d = np.load(p['file'])
            x_key = p.get('x_key') or ('x' if 'x' in d else 'center_um')
            y_key = p.get('y_key') or ('y' if 'y' in d else
                                       ('G_peaknorm' if p.get('norm', 'peak') == 'peak' else 'G'))
            x, y = d[x_key], d[y_key]
            log(f"[model] {name!r}: tabulated coefficient {os.path.basename(p['file'])} [{y_key}] over "
                f"{float(x[0]):.4g}..{float(x[-1]):.4g}", flush=True)
        else:
            need('x', 'y')
            x, y = p['x'], p['y']
        return TemplateProfile(x=np.asarray(x, float), values=np.asarray(y, float))
    if function == 'gaussian':
        # exp(-(v - center)^2 / 2 sigma^2); sigma = sqrt((w / fwhm_to_sigma)^2 + intrinsic_var)
        # per observation when `width` names a data variable w, else the scalar `sigma`.
        need('center')
        source = None
        if p.get('width') is not None:
            source = QuadratureSigma(fwhm_key=resolve_variable(p['width'], geom, known),
                                     fwhm_to_sigma=float(p.get('fwhm_to_sigma', 2.355)),
                                     intrinsic_var=float(p.get('intrinsic_var') or 0.0))
        elif p.get('sigma') is None:
            raise ValueError(f"gaussian coefficient of term {name!r} needs `sigma` or `width`")
        return GaussianProfile(center=float(p['center']),
                               sigma=None if p.get('sigma') is None else float(p['sigma']),
                               sigma_source=source)
    need('center', 'halfwidth')                   # linear: (v - center) / halfwidth
    return LinearProfile(center=float(p['center']), halfwidth=float(p['halfwidth']))
