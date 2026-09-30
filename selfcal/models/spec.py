"""The self-calibration model as data: sky terms + offset terms + their priors.

The equation the solver fits is::

    data(frame, pixel) = Σ_j S_j(pixel) · c_j(v)   +   Σ_m O_m(frame, chunk_m(pixel))   +   s(frame)

* **S terms** (:class:`SkyTerm`, one per sky block): a per-pixel map ``S_j``
  times a known multiplicative coefficient ``c_j(v)`` — any function of data
  variables ``v`` the instrument provides for every observation (SPHEREx: its
  wavelength and band-width maps). No coefficient means ``c = 1`` (a constant
  sky). The coefficient is a built-in shape (tabulated, Gaussian, linear), a
  named entry of the instrument's coefficient catalogue, or ANY importable
  Python function. Prior: Tikhonov damping ``damp_weight``.
* **O terms** (:class:`OffsetTerm`, one per chunk map): how the per-frame
  offset is parameterised on that map and what pulls it —
  ``kind='free'`` (one unknown per frame and chunk) with *smoothness*
  (``reg_weight`` along the ``adjacency`` axes), *shape* (soft ``poly``
  constraints along an axis) and an *anchor* (``mean_zero``: the per-frame
  mean over chunks is 0); ``kind='polybasis'`` (the offset IS a Chebyshev
  polynomial along ``axis``, one per value of ``group_axis``; no weight
  knob); ``kind='fixed'`` (one offset vector shared by every frame — a
  detector-fixed pattern).
* the per-frame **scalar** ``s`` (``scalar=True``), the DC of the offset.

A :class:`ModelSpec` is built from the ``[model]`` table of a run config
(:meth:`ModelSpec.from_config`) or by a mode (the named recipes are presets
that produce a spec from their shorter ``[params]``), and lowered onto an
instrument's geometry into the objects the solver consumes
(:class:`~selfcal.models.sky_model.SkyModel`,
:class:`~selfcal.models.offset_model.OffsetModel`). A new coefficient needs no
change here — name any function ``"package.module:name"``; adding a new way to
parameterise an offset is a new ``kind`` in :meth:`ModelSpec.build_offset_model`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .offset_model import OffsetBlock, OffsetModel
from .offset_structure import adjacency_along, adjacency_union, poly_basis_along, poly_chains_along
from .sky_model import Coefficient, ImportedFunction, SkyComponent, SkyModel

__all__ = ['SkyTerm', 'OffsetTerm', 'PolyConstraint', 'ModelSpec', 'build_coefficient', 'resolve_variable']

BUILTIN_FUNCTIONS = ('template', 'gaussian', 'linear')
OFFSET_KINDS = ('free', 'polybasis', 'fixed', 'grouped')


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
        {catalog = "pah_3p29", <overrides>}        # a named coefficient of the instrument

    ``name`` (the product name of the map) defaults to ``continuum`` for a
    constant term and to the catalogue entry for a catalogue coefficient.
    ``damp_weight``: the term's Tikhonov prior (None = the solver default).
    """
    name: str | None = None
    coefficient: object = None
    damp_weight: float | None = None

    def __post_init__(self):
        name = self.name
        c = self.coefficient
        if name is None:
            if c is None:
                name = 'continuum'
            elif isinstance(c, dict) and 'catalog' in c:
                name = str(c['catalog'])
            else:
                raise ValueError(f"a sky term with a coefficient needs a name: {c!r}")
        object.__setattr__(self, 'name', str(name))


@dataclass(frozen=True)
class PolyConstraint:
    """A soft polynomial constraint: along ``axis`` the offset is pulled toward a
    degree-``degree`` polynomial (finite-difference rows of weight ``weight``),
    optionally only over the window ``[lo, hi]`` of that axis."""
    axis: str
    degree: int = 1
    weight: float = 1.0
    lo: int | None = None
    hi: int | None = None


@dataclass(frozen=True)
class OffsetTerm:
    """One offset block on the chunk map ``map`` (None = the instrument's primary map).

    ``kind='free'``: unknown per (frame, chunk); ``reg_weight`` pulls chunks
    that neighbour along the ``adjacency`` axes together (None = the map's
    default adjacency axes; ``()`` = none; ``adjacency_step`` restricts pairs to
    that difference along the axis), ``poly`` adds soft polynomial constraints,
    ``mean_zero`` anchors the per-frame mean over chunks at zero.
    ``kind='polybasis'``: the offset is a degree-``degree`` Chebyshev in
    ``axis`` (default: the map's spectral axis) per value of ``group_axis``
    (default: the map's group axis) over ``[lo, hi]``, optionally piecewise on
    ``segments``.
    ``kind='fixed'``: one offset vector shared by every frame (with the same
    smoothness / anchor knobs as ``free``); ``kind='grouped'``: one offset
    vector per frame group ``groups`` (a grouping the instrument provides,
    e.g. ``'detector'`` = the frame's detector index, so a detector-fixed
    pattern for a multi-detector camera).

    Priors shared by every kind: ``damp`` (Tikhonov damping of this term's
    offsets toward zero, 0 = none). ``exact_group_rows``: for fixed/grouped
    terms, emit the mean-zero anchor and the adjacency rows once per group
    instead of once per frame (exact, and far fewer rows; changes the matrix
    structure, so it is a per-term choice). ``render``: which of the
    instrument's offset renderers draws this term on the mosaic (None = the
    instrument's default for the map)."""
    map: str | None = None
    kind: str = 'free'
    reg_weight: float = 0.0
    adjacency: tuple | None = None
    adjacency_step: int | None = None
    poly: tuple = ()
    mean_zero: bool = False
    axis: str | None = None
    group_axis: str | None = None
    degree: int | None = None
    lo: int | None = None
    hi: int | None = None
    segments: tuple | None = None
    damp: float = 0.0
    groups: str | None = None
    exact_group_rows: bool = False
    render: str | None = None

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
            raise ValueError("a grouped offset term needs `groups` (a frame grouping, e.g. 'detector')")
        if self.exact_group_rows and self.kind not in ('fixed', 'grouped'):
            raise ValueError("exact_group_rows applies to fixed / grouped terms")


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    sky: tuple = (SkyTerm(),)
    offset: tuple = ()
    scalar: bool = True
    mosaic: str = 'full'                 # full | no_wav | none

    def __post_init__(self):
        object.__setattr__(self, 'sky', tuple(
            t if isinstance(t, SkyTerm) else _sky_term(t) for t in self.sky))
        object.__setattr__(self, 'offset', tuple(
            t if isinstance(t, OffsetTerm) else OffsetTerm(**t) for t in self.offset))
        if not self.sky:
            raise ValueError("the model needs at least one sky term")

    # ---- from a config table -------------------------------------------------------
    @classmethod
    def from_config(cls, table: dict) -> ModelSpec:
        """The ``[model]`` table: ``sky = [{name=..., coefficient={...}, damp_weight=...}, ...]``,
        ``offset = [{map=..., kind=..., poly=[{axis=..., degree=..., weight=...}]}, ...]``,
        ``scalar``, ``mosaic``."""
        t = dict(table)
        sky = t.pop('sky', None) or [{}]
        offset = t.pop('offset', None) or []
        scalar = bool(t.pop('scalar', True))
        mosaic = str(t.pop('mosaic', 'full'))
        if t:
            raise ValueError(f"unknown [model] keys {sorted(t)}; expected sky / offset / scalar / mosaic")
        return cls(sky=tuple(_sky_term(dict(s)) for s in sky),
                   offset=tuple(OffsetTerm(**dict(o)) for o in offset), scalar=scalar, mosaic=mosaic)

    # ---- properties ------------------------------------------------------------------------
    @property
    def has_coefficients(self) -> bool:
        """Whether any sky term has a coefficient (reads data variables)."""
        return any(t.coefficient is not None for t in self.sky)

    @property
    def x0_kind(self) -> str:
        """How the solve is initialised: from the per-frame scalars when the model
        has them, else from the normal equations' diagonal."""
        return 'scalar_only' if self.scalar else 'from_Ab'

    def check(self, geom, catalog=None):
        """Raise a clear error when a term refers to something the instrument does
        not provide: a data variable, a chunk map, a chunk axis, a catalogue entry."""
        self.build_sky_model(geom, catalog, log=lambda *a, **k: None)
        for term in self.offset:
            cm = geom.chunk_maps.get(term.map) if term.map else geom.chunk_map
            if cm is None:
                raise ValueError(f"offset term map {term.map!r} is not a chunk map of the instrument "
                                 f"({sorted(geom.chunk_maps)})")
            names = list(cm.axes.names) if cm.axes is not None else []
            axes = list(term.adjacency or ()) + [p.axis for p in term.poly]
            if term.kind == 'polybasis':
                axes += [term.axis or cm.spectral_axis, term.group_axis or cm.group_axis]
            bad = [a for a in axes if a not in names]
            if bad:
                raise ValueError(f"offset term on {cm.name!r} names axes {bad}; the map's axes are {names}")

    # ---- lowering: sky ---------------------------------------------------------------------
    def build_sky_model(self, geom, catalog=None, log=print) -> SkyModel:
        """The solver's SkyModel: one component per term, coefficients resolved
        against the instrument's data variables (``geom``) and its coefficient
        catalogue."""
        comps = []
        for term in self.sky:
            coeff = None if term.coefficient is None else build_coefficient(
                term.coefficient, geom, catalog=catalog, name=term.name, log=log)
            comps.append(SkyComponent(name=term.name, coefficient=coeff,
                                      damp_weight=None if term.damp_weight is None else float(term.damp_weight)))
        return SkyModel(tuple(comps))

    # ---- lowering: offsets -------------------------------------------------------------------
    def build_offset_model(self, geom, n_frames, frame_groups=None, log=print) -> OffsetModel:
        """``frame_groups``: ``{name: per-frame group id array}`` for the grouped
        terms (the instrument's ``frame_groups(frames)``)."""
        blocks = [self._offset_block(term, geom, n_frames, frame_groups or {}, log) for term in self.offset]
        return OffsetModel(blocks, use_per_frame_scalar=self.scalar)

    def setup_kwargs(self) -> dict:
        """The solver options the terms' priors imply beyond the OffsetModel:
        per-map damping and the exact grouped rows. Empty when no term asks for
        them (the historical recipes)."""
        kw = {}
        if any(t.damp for t in self.offset):
            kw['damp_offset_maps'] = [float(t.damp) for t in self.offset]
        exact = [m for m, t in enumerate(self.offset) if t.exact_group_rows]
        if exact:
            kw['mean_offset_group_rows'] = True
            kw['group_adjacency_maps'] = exact
        return kw

    def _offset_block(self, term, geom, n_frames, frame_groups, log):
        cm = geom.chunk_maps[term.map] if term.map else geom.chunk_map
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
            return OffsetBlock(chunk_map=cm.det, poly_basis=pb)
        # free / fixed: adjacency + soft polynomials + anchor
        axes = cm.adjacency_axes if term.adjacency is None else tuple(term.adjacency)
        if term.adjacency_step is not None:
            if len(axes) != 1:
                raise ValueError("adjacency_step needs exactly one adjacency axis")
            adj = adjacency_along(cm.det, cm.axes, axes[0], step=int(term.adjacency_step))
        else:
            adj = adjacency_union(cm.det, cm.axes, axes) if axes else None
        groups = []
        for pc in term.poly:
            size = cm.axes[pc.axis].size
            if size < pc.degree + 2:
                log(f"[model] axis {pc.axis!r} has {size} values < degree+2={pc.degree + 2}: "
                    f"skipping the (vacuous) polynomial constraint along it.")
                continue
            chains, stencil = poly_chains_along(cm.axes, pc.axis, int(pc.degree), pc.lo, pc.hi)
            groups.append({'chains': chains, 'stencil': stencil, 'weight': pc.weight})
        det_groups = None
        if term.kind == 'fixed':
            det_groups = np.zeros(n_frames, dtype=int)
        elif term.kind == 'grouped':
            if term.groups not in frame_groups:
                raise ValueError(f"offset term on {cm.name!r} groups frames by {term.groups!r}, which the "
                                 f"instrument does not provide (has {sorted(frame_groups)})")
            det_groups = np.asarray(frame_groups[term.groups])
            if det_groups.shape != (n_frames,):
                raise ValueError(f"frame grouping {term.groups!r} has shape {det_groups.shape}, "
                                 f"expected ({n_frames},)")
        return OffsetBlock(
            chunk_map=cm.det, adj_info=adj, reg_weight=term.reg_weight,
            poly_constraints=groups or None,
            mean_offset=np.zeros(n_frames) if term.mean_zero else None,
            det_groups=det_groups)


def _sky_term(d) -> SkyTerm:
    d = dict(d)
    if 'type' in d:
        raise ValueError("sky terms have no `type`: give `name` and, for a term that is not constant, "
                         "`coefficient = { variable = ..., function = ..., <parameters> }` "
                         f"(got {d})")
    return SkyTerm(**d)


def resolve_variable(name, geom):
    """The key of a data variable the instrument provides for every observation:
    a key of ``geom.aux``, or an alias — ``wavelength`` (the instrument's
    ``wavelength_key``), ``bandwidth`` (its ``width_key``)."""
    aux = getattr(geom, 'aux', None) or {}
    aliases = {}
    if getattr(geom, 'wavelength_key', None):
        aliases['wavelength'] = geom.wavelength_key
    if getattr(geom, 'width_key', None):
        aliases['bandwidth'] = geom.width_key
    key = aliases.get(name, name)
    if key not in aux:
        avail = sorted(aux) + [f'{a} (= {k})' for a, k in aliases.items()]
        raise ValueError(f"data variable {name!r} is not provided by the instrument "
                         f"(variables: {avail or 'none'})")
    return key


def _hashable(v):
    if isinstance(v, (list, tuple)):
        return tuple(_hashable(x) for x in v)
    if isinstance(v, dict):
        return tuple(sorted((k, _hashable(x)) for k, x in v.items()))
    return v


def build_coefficient(spec, geom, catalog=None, name=None, log=print) -> Coefficient:
    """A :class:`~selfcal.models.sky_model.Coefficient` from its config form (see
    :class:`SkyTerm`), with variable names resolved against the instrument."""
    if isinstance(spec, Coefficient):
        keys = tuple(resolve_variable(v, geom) for v in spec.main_variables)
        return Coefficient(keys[0] if isinstance(spec.variable, str) else keys, spec.function)
    spec = dict(spec)
    if 'catalog' in spec:
        entry = spec.pop('catalog')
        cat = catalog or {}
        if entry not in cat:
            raise ValueError(f"coefficient catalogue entry {entry!r} unknown (the instrument has {sorted(cat)})")
        return cat[entry](**spec)
    if 'variable' not in spec or 'function' not in spec:
        raise ValueError(f"a coefficient needs `variable` and `function` (or `catalog`): {spec}")
    variable = spec.pop('variable')
    function = str(spec.pop('function'))
    params = dict(spec.pop('params', None) or {})
    params.update(spec)
    many = isinstance(variable, (list, tuple))
    keys = tuple(resolve_variable(v, geom) for v in variable) if many else resolve_variable(variable, geom)
    if ':' in function:
        return Coefficient(keys, ImportedFunction(function, _hashable(params)))
    if function not in BUILTIN_FUNCTIONS:
        raise ValueError(f"unknown coefficient function {function!r}: a built-in {BUILTIN_FUNCTIONS} "
                         f"or a Python function 'package.module:name'")
    if many:
        raise ValueError(f"the built-in {function!r} takes one variable; got {list(variable)}")
    return Coefficient(keys, _builtin(function, params, geom, name, log))


def _builtin(function, p, geom, name, log):
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
        return TemplateProfile(wave_um=np.asarray(x, float), values=np.asarray(y, float))
    if function == 'gaussian':
        # exp(-(v - center)^2 / 2 sigma^2); sigma = sqrt((w / fwhm_to_sigma)^2 + intrinsic_var)
        # per observation when `width` names a data variable w, else the scalar `sigma`.
        need('center')
        source = None
        if p.get('width') is not None:
            source = QuadratureSigma(fwhm_key=resolve_variable(p['width'], geom),
                                     fwhm_to_sigma=float(p.get('fwhm_to_sigma', 2.355)),
                                     intrinsic_var_um2=float(p.get('intrinsic_var') or 0.0))
        elif p.get('sigma') is None:
            raise ValueError(f"gaussian coefficient of term {name!r} needs `sigma` or `width`")
        return GaussianProfile(center_um=float(p['center']),
                               sigma_um=None if p.get('sigma') is None else float(p['sigma']),
                               sigma_source=source)
    need('center', 'halfwidth')                   # linear: (v - center) / halfwidth
    return LinearProfile(center_um=float(p['center']), halfwidth_um=float(p['halfwidth']))
