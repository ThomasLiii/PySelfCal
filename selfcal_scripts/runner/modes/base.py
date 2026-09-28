"""Calibration-mode contract + registry.

A *mode* is the calibration recipe. Since the model became data
(:class:`selfcal.models.spec.ModelSpec`: sky terms + offset terms + their
priors), a mode is simply the function that produces a spec from a run config
and an instrument's geometry — :meth:`CalMode.model_spec`. Everything else
(lowering the spec to the solver's objects, the aux maps, x0, the mosaic
geometry) is shared here. The ``model`` mode reads the spec verbatim from the
``[model]`` table; the named recipes are presets that build one from their
shorter ``[params]``.

The generic engine talks only to this interface and resolves modes by name
through ``get_mode``; it never references a specific mode. Adding a recipe is
a new module here with an ``@register_mode`` class overriding ``model_spec``;
nothing else in the runner changes.

Tiling and the N-pass schedule are NOT mode properties: any mode runs tiled
when the config has a ``[tiling]`` table, and any mode with the two N-pass hooks
(``clip_group_edges``, ``refit_poly_basis``) runs as task ``npass``.

Names: the registered name of a mode is structural (``continuum``,
``spectral``, ``spectral_softpoly``, ``spectral_polybasis``,
``two_block_fixed``, ``model``); the historical SPHEREx names (``pahfit``,
``pahfit_subch``, ``pahfit_lvf``, ``pahfit_lvf_polybasis``, ``multiline``,
``tiled``, ``k2_readout``) are registered presets of those recipes and keep
their exact behaviour.
"""
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
    """Base class: a recipe = a :class:`ModelSpec` producer.

    Subclass + ``@register_mode("name")``; override ``model_spec`` (and, rarely,
    ``configure`` / ``mosaic_geometry``). Class attrs:
      mosaic_mode "full" (mosaic + the instrument's aux coadds, e.g. wavelength
                  maps) | "no_wav" (mosaic only) | "none" (skip mosaic)
      requires    capability tags the instrument must provide (e.g. "wavelength").
    """

    name = None
    requested_name = None
    mosaic_mode = "full"
    requires = ()

    # ---- the recipe -----------------------------------------------------------------
    def model_spec(self, cfg, inst, geom):
        raise NotImplementedError

    def spec(self, cfg, inst, geom):
        """The spec, built once per (mode instance, config)."""
        cached = getattr(self, '_spec', None)
        if cached is None or cached[0] is not cfg:
            self._spec = (cfg, self.model_spec(cfg, inst, geom))
        return self._spec[1]

    # ---- shared lowering ------------------------------------------------------------------
    def build_offset_model(self, cfg, inst, geom, jobgeom, job, n_frames):
        return self.spec(cfg, inst, geom).build_offset_model(geom, n_frames, log=self._log)

    def build_sky_model(self, cfg, inst, geom):
        return self.spec(cfg, inst, geom).build_sky_model(geom, inst.line_catalog(), log=self._log)

    def aux_maps(self, cfg, inst, geom):
        """Named per-pixel maps the solve needs: all of the instrument's when the
        sky has spectral terms, none for a continuum-only sky."""
        return dict(geom.aux) if self.spec(cfg, inst, geom).has_lines else {}

    def x0_kind(self, cfg, inst, geom):
        return self.spec(cfg, inst, geom).x0_kind

    def x0(self, cfg, cc):
        from selfcal.core.solution import compute_x0_from_Ab, compute_x0_scalar_only
        if getattr(self, '_spec', None) is not None and self._spec[1].x0_kind == 'from_Ab':
            return compute_x0_from_Ab(cc.A, cc.b, cc.ref_shape,
                                      active_mask=getattr(cc, "active_mask", None))
        return compute_x0_scalar_only(
            cc.A, cc.b, cc.ref_shape,
            scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
            num_sky_blocks=cc.num_sky_blocks,
            active_mask=getattr(cc, "active_mask", None))

    def configure(self, cfg, cc):
        """Post-solve settings recorded on the cal (default: the line-Fisher
        threshold when the sky has spectral terms)."""
        if getattr(self, '_spec', None) is not None and self._spec[1].has_lines:
            cc.line_fisher_threshold = cfg.params.get('line_fisher_threshold', 10.0)

    def mosaic_geometry(self, cfg, inst, geom, jobgeom):
        """(chunk_maps, offset renderers) for make_mosaic: every offset term's map
        on the reference grid; the primary map rendered by the instrument's
        smooth offset renderer, the others block-constant."""
        spec = self.spec(cfg, inst, geom)
        maps, funcs = [], []
        for term in spec.offset:
            cm = geom.chunk_maps[term.map] if term.map else geom.chunk_map
            maps.append(cm.grid)
            funcs.append(inst.offset_renderer(cfg.instrument_cfg, geom, jobgeom)
                         if cm is geom.chunk_map else None)
        return maps, funcs

    def _log(self, *args, **kw):
        print(*args, **kw)

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


# ---------------------------------------------------------------------------
# building blocks the presets share
# ---------------------------------------------------------------------------
def standard_offset_term(cfg, geom, *, extra_poly=(), column_poly_default_weight=None):
    """The standard offset term: free per (frame, chunk) on the primary map,
    smoothness along the map's adjacency axes, an optional soft polynomial
    along ``[params].poly_axis`` (default: the first adjacency axis) when
    ``poly_weight`` is set (or the mode gives a default weight), a mean-zero
    anchor. ``extra_poly`` constraints are appended (the spectral polynomial
    of the soft-poly recipes)."""
    from selfcal.models.spec import OffsetTerm, PolyConstraint
    p = cfg.params
    cm = geom.chunk_map
    adj_axes = tuple(param(p, 'adjacency_axes', default=cm.adjacency_axes))
    poly = []
    weight = param(p, 'poly_weight', default=column_poly_default_weight)
    if weight is not None:
        axis = param(p, 'poly_axis', default=adj_axes[0] if adj_axes else None)
        if axis is not None:
            poly.append(PolyConstraint(axis=axis, degree=int(param(p, 'poly_degree', default=1)), weight=weight))
    poly.extend(extra_poly)
    return OffsetTerm(map=None, kind='free', reg_weight=p.get('reg_weight', 0.1), adjacency=adj_axes,
                      poly=tuple(poly), mean_zero=True)


def spectral_sky_terms(cfg, geom):
    """The sky terms of the spectral recipes from ``[params]``: continuum +
    ``[[params.lines]]`` entries, or a single ``line_template_npz`` line, or the
    catalogue line ``[params].line`` (default ``pah_3p29``) with ``line_center``
    / ``line_sigma`` overrides."""
    from selfcal.models.spec import SkyTerm
    p = cfg.params
    terms = [SkyTerm('continuum')]
    lines = p.get('lines')
    if lines:
        for spec in lines:
            kw = dict(type='line', name=spec['name'], damp_weight=spec.get('damp_weight'))
            if 'template_npz' in spec:
                kw.update(template=spec['template_npz'], template_norm=spec.get('template_norm', 'peak'))
            elif 'center_um' in spec:
                kw.update(center_um=float(spec['center_um']))
                if spec.get('sigma_um') is not None:
                    kw['sigma_um'] = float(spec['sigma_um'])
                else:
                    kw['intrinsic_var_um2'] = float(spec.get('intrinsic_var_um2', 0.0))
            else:
                raise ValueError(f"line spec needs 'template_npz' or 'center_um': {spec}")
            terms.append(SkyTerm(**kw))
        return terms
    npz = p.get('line_template_npz')
    if npz:
        terms.append(SkyTerm(type='line', name='pah_3p29', template=npz,
                             template_norm=p.get('line_template_norm', 'peak')))
        return terms
    terms.append(SkyTerm(type='line', catalog=p.get('line', 'pah_3p29'),
                         line_center=p.get('line_center'), line_sigma=p.get('line_sigma')))
    return terms
