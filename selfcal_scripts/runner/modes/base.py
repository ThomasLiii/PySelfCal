"""Calibration-mode contract + registry.

A *mode* is the calibration recipe: how to assemble the offset model (including
its regularisation), the sky model, the x0 init and the mosaic geometry from a
run config + an instrument's geometry. The generic engine talks only to this
interface and resolves modes by name through ``get_mode``; it never references
a specific mode. Adding a calibration variant is a new module here with an
``@register_mode`` class; nothing else in the runner changes.

Modes are written against the instrument-neutral geometry objects
(:class:`~selfcal.instruments.base.DetectorGeometry` / ``JobGeometry``): the
offset structure is expressed in the *axes* of the chunk grid ("adjacency along
``column``", "degree-2 polynomial along the spectral axis per group") through
:mod:`selfcal.models.offset_structure`, so every recipe runs on any instrument
that declares the axes it needs. Modes read ``[params]`` only, never the
``[instrument]`` table.

Tiling and the N-pass schedule are NOT mode properties: any mode runs tiled
when the config has a ``[tiling]`` table, and any mode with the two N-pass hooks
(``clip_group_edges``, ``refit_poly_basis``) runs as task ``npass``.

Names: the registered name of a mode is structural (``continuum``,
``spectral``, ``spectral_softpoly``, ``spectral_polybasis``,
``two_block_fixed``); the historical SPHEREx names (``pahfit``,
``pahfit_subch``, ``pahfit_lvf``, ``pahfit_lvf_polybasis``, ``multiline``,
``tiled``, ``k2_readout``) are registered presets of those recipes and keep
their exact behaviour.
"""
import numpy as np

_MODE_REGISTRY = {}


def register_mode(name, *aliases):
    """Register ``cls`` as ``name``; ``aliases`` select the same class."""
    def deco(cls):
        cls.name = name
        _MODE_REGISTRY[name] = cls
        for a in aliases:
            _MODE_REGISTRY[a] = cls
        return cls
    return deco


def get_mode(name):
    if name not in _MODE_REGISTRY:
        raise ValueError(
            f"unknown mode {name!r}; available: {sorted(_MODE_REGISTRY)}")
    mode = _MODE_REGISTRY[name]()
    mode.requested_name = name
    return mode


def available_modes():
    return sorted(_MODE_REGISTRY)


def param(params, *keys, default=None):
    """The first of ``keys`` present in ``[params]`` (generic name first, then the
    historical SPHEREx spelling), else ``default``."""
    for k in keys:
        if k in params:
            return params[k]
    return default


def spectral_window(params):
    """``(lo, hi)`` of the spectral polynomial window (inclusive chunk-axis values)."""
    lo = param(params, 'spectral_poly_lo', 'subch_poly_lo')
    hi = param(params, 'spectral_poly_hi', 'subch_poly_hi')
    if lo is None or hi is None:
        raise ValueError("[params] needs spectral_poly_lo / spectral_poly_hi "
                         "(historical spelling subch_poly_lo / subch_poly_hi)")
    return int(lo), int(hi)


class CalMode:
    """Base class with the defaults the simplest (continuum) mode needs.

    Subclass + ``@register_mode("name")``; override only what differs. Class attrs:
      mosaic_mode "full" (mosaic + the instrument's aux coadds, e.g. wavelength
                  maps) | "no_wav" (mosaic only) | "none" (skip mosaic)
      requires    capability tags the instrument must provide (e.g. "wavelength").
    """

    name = None
    requested_name = None
    mosaic_mode = "full"
    requires = ()

    def build_offset_model(self, cfg, inst, geom, jobgeom, job, n_frames):
        raise NotImplementedError

    def build_sky_model(self, cfg, inst, geom):
        from selfcal.models.sky_model import SkyModel
        return SkyModel.continuum_only()

    def aux_maps(self, cfg, inst, geom):
        """Named per-pixel maps the solve needs (``{}`` for a continuum sky)."""
        return {}

    def x0(self, cfg, cc):
        from selfcal.core.solution import compute_x0_scalar_only
        return compute_x0_scalar_only(
            cc.A, cc.b, cc.ref_shape,
            scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
            num_sky_blocks=cc.num_sky_blocks,
            active_mask=getattr(cc, "active_mask", None))

    def configure(self, cfg, cc):
        pass

    def mosaic_geometry(self, cfg, inst, geom, jobgeom):
        """(chunk_maps, offset renderers) for make_mosaic. Default: the primary
        chunk map on the reference grid, rendered by the instrument's smooth
        offset renderer (None = block-constant)."""
        return ([geom.chunk_map.grid],
                [inst.offset_renderer(cfg.instrument_cfg, geom, jobgeom)])

    # ---- N-pass hooks (task 'npass', passes >= 2) ---------------------------
    def clip_group_edges(self, cfg, inst, geom):
        """Wavelength bin edges of the grouped outlier clip (``subch_clip``):
        one group per value of the primary map's spectral axis."""
        from selfcal.models.offset_structure import group_edges_along
        cm = geom.chunk_map
        if cm.spectral_axis is None or geom.wavelength_key is None:
            raise ValueError("the grouped outlier clip needs a chunk map with a spectral axis "
                             "and an instrument wavelength map")
        return group_edges_along(geom.aux[geom.wavelength_key], cm.det, cm.axes, cm.spectral_axis)

    def refit_poly_basis(self, cfg, inst, geom, degree, segments=None):
        """The per-frame polynomial offset basis of the OFFSET refit: a
        degree-``degree`` polynomial in the primary map's spectral axis, one per
        value of its group axis, over the ``[params]`` spectral window
        (optionally piecewise on ``segments``)."""
        from selfcal.models.offset_structure import poly_basis_along
        cm = geom.chunk_map
        lo, hi = spectral_window(cfg.params)
        return poly_basis_along(cm.axes, cm.spectral_axis, cm.group_axis, int(degree), lo, hi,
                                segments=segments)


def standard_block(cfg, geom, n_frames, *, extra_poly_groups=(), column_poly_default_weight=None):
    """The standard single offset block: adjacency along the chunk map's
    adjacency axes + an optional soft polynomial along ``[params].poly_axis``
    (default: the first adjacency axis) + a per-frame mean-zero anchor.

    The polynomial constraint is applied iff ``[params].poly_weight`` is set
    (or ``column_poly_default_weight`` is given by the mode) and the axis has at
    least ``degree + 2`` values; a soft polynomial along a shorter axis would
    be vacuous (and the chain builder would raise), so it is skipped.
    ``extra_poly_groups`` are appended after it (the spectral polynomial of the
    soft-poly modes). Returns ``(OffsetBlock, poly_groups)``.
    """
    from selfcal.models.offset_model import OffsetBlock
    from selfcal.models.offset_structure import adjacency_union, poly_chains_along
    p = cfg.params
    cm = geom.chunk_map
    adj_axes = tuple(param(p, 'adjacency_axes', default=cm.adjacency_axes))
    adj = adjacency_union(cm.det, cm.axes, adj_axes) if adj_axes else None
    poly_groups = []
    weight = param(p, 'poly_weight', default=column_poly_default_weight)
    if weight is not None:
        deg = int(param(p, 'poly_degree', default=1))
        axis = param(p, 'poly_axis', default=adj_axes[0] if adj_axes else None)
        if axis is not None and cm.axes[axis].size >= deg + 2:
            chains, stencil = poly_chains_along(cm.axes, axis, deg)
            poly_groups.append({'chains': chains, 'stencil': stencil, 'weight': weight})
        elif axis is not None:
            print(f"[{cfg.mode}] axis {axis!r} has {cm.axes[axis].size} values < degree+2={deg + 2}: "
                  f"skipping the (vacuous) polynomial constraint along it.")
    poly_groups.extend(extra_poly_groups)
    block = OffsetBlock(chunk_map=cm.det, adj_info=adj, reg_weight=p.get('reg_weight', 0.1),
                        poly_constraints=poly_groups or None, mean_offset=np.zeros(n_frames))
    return block
