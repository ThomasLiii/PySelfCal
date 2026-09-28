"""The self-calibration model as data: sky terms + offset terms + their priors.

The equation the solver fits is::

    data(frame, pixel) = Σ_j S_j(pixel) · c_j(λ_pixel)   +   Σ_m O_m(frame, chunk_m(pixel))   +   s(frame)

* **S terms** (:class:`SkyTerm`, one per sky block): a *continuum* (constant
  per pixel, coefficient 1) or a *line* (per-pixel amplitude of a spectral
  profile evaluated at the instrument's wavelength map). Prior: Tikhonov
  damping ``damp_weight``.
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
:class:`~selfcal.models.offset_model.OffsetModel`). Adding a new profile shape
is a new :class:`~selfcal.models.profiles.SpectralProfile`; adding a new way to
parameterise an offset is a new ``kind`` in :meth:`ModelSpec.build_offset_model`.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

from .offset_model import OffsetBlock, OffsetModel
from .offset_structure import adjacency_along, adjacency_union, poly_basis_along, poly_chains_along
from .sky_model import ContinuumComponent, SkyModel, SpectralComponent

__all__ = ['SkyTerm', 'OffsetTerm', 'PolyConstraint', 'ModelSpec']

SKY_TYPES = ('continuum', 'line')
OFFSET_KINDS = ('free', 'polybasis', 'fixed')


# ---------------------------------------------------------------------------
# terms
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SkyTerm:
    """One sky block. ``type='continuum'``: constant per pixel. ``type='line'``:
    per-pixel amplitude of a profile — a tabulated ``template`` (npz with
    ``center_um`` and ``G`` / ``G_peaknorm``), a Gaussian ``center_um`` +
    ``sigma_um``, a Gaussian with a per-pixel width from the instrument's
    band-width map (``center_um`` + ``intrinsic_var_um2``), or a ``catalog``
    entry of the instrument's line catalogue (optionally with ``line_center``
    / ``line_sigma`` overrides). ``damp_weight``: Tikhonov prior of the block
    (None = the run's default)."""
    type: str = 'continuum'
    name: str | None = None
    damp_weight: float | None = None
    template: str | None = None
    template_norm: str = 'peak'
    center_um: float | None = None
    sigma_um: float | None = None
    intrinsic_var_um2: float | None = None
    fwhm_to_sigma: float = 2.355
    catalog: str | None = None
    line_center: float | None = None
    line_sigma: float | None = None

    def __post_init__(self):
        if self.type not in SKY_TYPES:
            raise ValueError(f"sky term type must be one of {SKY_TYPES}, got {self.type!r}")
        if self.type == 'line' and not (self.template or self.center_um is not None or self.catalog):
            raise ValueError(f"line term {self.name!r} needs a template, a center_um or a catalog entry")


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
    smoothness / anchor knobs as ``free``)."""
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

    def __post_init__(self):
        if self.kind not in OFFSET_KINDS:
            raise ValueError(f"offset term kind must be one of {OFFSET_KINDS}, got {self.kind!r}")
        object.__setattr__(self, 'poly', tuple(
            p if isinstance(p, PolyConstraint) else PolyConstraint(**p) for p in self.poly))
        if self.adjacency is not None:
            object.__setattr__(self, 'adjacency', tuple(self.adjacency))
        if self.kind == 'polybasis' and self.degree is None:
            raise ValueError("a polybasis offset term needs a degree")


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelSpec:
    sky: tuple = (SkyTerm('continuum'),)
    offset: tuple = ()
    scalar: bool = True
    mosaic: str = 'full'                 # full | no_wav | none

    def __post_init__(self):
        object.__setattr__(self, 'sky', tuple(
            t if isinstance(t, SkyTerm) else SkyTerm(**t) for t in self.sky))
        object.__setattr__(self, 'offset', tuple(
            t if isinstance(t, OffsetTerm) else OffsetTerm(**t) for t in self.offset))
        if not self.sky or self.sky[0].type != 'continuum':
            raise ValueError("the first sky term must be the continuum")

    # ---- from a config table -------------------------------------------------------
    @classmethod
    def from_config(cls, table: dict) -> ModelSpec:
        """The ``[model]`` table: ``sky = [{type=...}, ...]``, ``offset = [{map=..., kind=...,
        poly=[{axis=..., degree=..., weight=...}]}, ...]``, ``scalar``, ``mosaic``."""
        t = dict(table)
        sky = t.pop('sky', None) or [{'type': 'continuum'}]
        offset = t.pop('offset', None) or []
        scalar = bool(t.pop('scalar', True))
        mosaic = str(t.pop('mosaic', 'full'))
        if t:
            raise ValueError(f"unknown [model] keys {sorted(t)}; expected sky / offset / scalar / mosaic")
        return cls(sky=tuple(SkyTerm(**dict(s)) for s in sky),
                   offset=tuple(OffsetTerm(**dict(o)) for o in offset), scalar=scalar, mosaic=mosaic)

    # ---- properties ------------------------------------------------------------------------
    @property
    def has_lines(self) -> bool:
        return any(t.type == 'line' for t in self.sky)

    @property
    def x0_kind(self) -> str:
        """How the solve is initialised: from the per-frame scalars when the model
        has them, else from the normal equations' diagonal."""
        return 'scalar_only' if self.scalar else 'from_Ab'

    def requires(self, geom) -> list[str]:
        """Capability tags this model needs from the instrument."""
        req = []
        if self.has_lines:
            req.append('wavelength')
        if any(t.kind == 'polybasis' or any(p.axis == _spectral_axis(geom, t) for p in t.poly)
               for t in self.offset if t.kind != 'fixed'):
            req.append('spectral_axis')
        return req

    # ---- lowering: sky ---------------------------------------------------------------------
    def build_sky_model(self, geom, line_catalog=None, log=print) -> SkyModel:
        comps = []
        for i, term in enumerate(self.sky):
            if term.type == 'continuum':
                comps.append(ContinuumComponent(damp_weight=term.damp_weight)
                             if term.damp_weight is not None else ContinuumComponent())
                continue
            if geom.wavelength_key is None:
                raise ValueError(f"sky term {term.name or i!r} is a line but the instrument has no "
                                 f"wavelength map")
            if term.catalog:
                cat = (line_catalog or {})
                if term.catalog not in cat:
                    raise ValueError(f"line catalogue entry {term.catalog!r} unknown (have {sorted(cat)})")
                model = cat[term.catalog](term.line_center, term.line_sigma)
                for c in model.components[1:]:
                    comps.append(c if term.damp_weight is None else
                                 SpectralComponent(name=c.name, profile=c.profile,
                                                   wavelength_key=c.wavelength_key,
                                                   damp_weight=float(term.damp_weight)))
                continue
            name = term.name or f'line_{i}'
            comps.append(SpectralComponent(name=name, profile=_profile(term, geom, name, log),
                                           wavelength_key=geom.wavelength_key,
                                           damp_weight=None if term.damp_weight is None
                                           else float(term.damp_weight)))
        return SkyModel(tuple(comps))

    # ---- lowering: offsets -------------------------------------------------------------------
    def build_offset_model(self, geom, n_frames, log=print) -> OffsetModel:
        blocks = [self._offset_block(term, geom, n_frames, log) for term in self.offset]
        return OffsetModel(blocks, use_per_frame_scalar=self.scalar)

    def _offset_block(self, term, geom, n_frames, log):
        cm = geom.chunk_maps[term.map] if term.map else geom.chunk_map
        if term.kind == 'polybasis':
            axis = term.axis or cm.spectral_axis
            group = term.group_axis or cm.group_axis
            if axis is None or group is None or term.lo is None or term.hi is None:
                raise ValueError(f"polybasis term on map {cm.name!r} needs axis, group_axis, lo, hi")
            pb = poly_basis_along(cm.axes, axis, group, int(term.degree), int(term.lo), int(term.hi),
                                  segments=term.segments)
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
        return OffsetBlock(
            chunk_map=cm.det, adj_info=adj, reg_weight=term.reg_weight,
            poly_constraints=groups or None,
            mean_offset=np.zeros(n_frames) if term.mean_zero else None,
            det_groups=np.zeros(n_frames, dtype=int) if term.kind == 'fixed' else None)


def _spectral_axis(geom, term):
    cm = geom.chunk_maps[term.map] if term.map else geom.chunk_map
    return cm.spectral_axis


def _profile(term, geom, name, log):
    from .profiles import GaussianProfile, QuadratureSigma, TemplateProfile
    if term.template:
        d = np.load(term.template)
        key = 'G_peaknorm' if term.template_norm == 'peak' else 'G'
        log(f"[model] line {name!r}: template {os.path.basename(term.template)} [{key}]: "
            f"{d['center_um'][0]:.3f}-{d['center_um'][-1]:.3f} um", flush=True)
        return TemplateProfile(wave_um=np.asarray(d['center_um'], float), values=np.asarray(d[key], float))
    if term.sigma_um is not None:
        return GaussianProfile(center_um=float(term.center_um), sigma_um=float(term.sigma_um))
    if geom.width_key is None:
        raise ValueError(f"line term {name!r}: a per-pixel width needs the instrument's band-width map; "
                         f"give sigma_um instead")
    return GaussianProfile(center_um=float(term.center_um),
                           sigma_source=QuadratureSigma(fwhm_key=geom.width_key,
                                                        fwhm_to_sigma=float(term.fwhm_to_sigma),
                                                        intrinsic_var_um2=float(term.intrinsic_var_um2 or 0.0)))
