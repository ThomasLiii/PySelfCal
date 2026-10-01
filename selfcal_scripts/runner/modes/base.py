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
    @staticmethod
    def frame_variable_names(cfg, inst):
        """The instrument's frame-variable names (time, filter, ... ; default
        ``exposure``, ``detector``)."""
        return tuple(inst.frame_variable_names(cfg.instrument_cfg))

    def build_offset_model(self, cfg, inst, geom, jobgeom, job, n_frames, frames=None, variables=None):
        """``frames``: the frame list (needed by grouped terms, which share an
        offset over the frames with equal values of a frame variable);
        ``variables``: the solve's data variables (their frame variables are
        groupings too)."""
        spec = self.spec(cfg, inst, geom)
        groups = None
        if frames is not None and any(t.kind == 'grouped' for t in spec.offset):
            groups = dict(inst.frame_groups(frames))
            for k, v in inst.frame_variables(frames, cfg.instrument_cfg).items():
                groups.setdefault(k, v)
            if variables is not None:
                for k, v in variables.frame.items():
                    groups.setdefault(k, v)
        return spec.build_offset_model(geom, n_frames, frame_groups=groups, log=self._log,
                                       catalog=inst.coefficient_catalog(),
                                       frame_variables=self.frame_variable_names(cfg, inst))

    def setup_kwargs(self, cfg, inst, geom):
        """Extra ``setup_lsqr`` options the model's priors imply (per-map damping,
        exact grouped rows); empty for the historical recipes."""
        return self.spec(cfg, inst, geom).setup_kwargs()

    def build_sky_model(self, cfg, inst, geom):
        return self.spec(cfg, inst, geom).build_sky_model(
            geom, inst.coefficient_catalog(), log=self._log,
            frame_variables=self.frame_variable_names(cfg, inst))

    def aux_maps(self, cfg, inst, geom):
        """The detector maps the solve samples at every observation: all of the
        instrument's when anything in the model reads data variables, none
        otherwise."""
        return dict(geom.aux) if self.spec(cfg, inst, geom).needs_variables else {}

    def build_variables(self, cfg, inst, geom, frames, ref_shape=None, ref_wcs=None):
        """The solve's data variables beyond the instrument's detector maps (a
        :class:`~selfcal.models.variables.VariableSet`): the model's
        ``[model.variables]`` and the instrument's frame variables. ``None`` when
        the model reads none of them (the historical recipes)."""
        spec = self.spec(cfg, inst, geom)
        fnames = set(self.frame_variable_names(cfg, inst))
        if not spec.variables and not (spec.referenced_variables() & fnames):
            return None
        fv = inst.frame_variables(frames, cfg.instrument_cfg)
        return spec.build_variables(geom, frames, ref_shape=ref_shape, ref_wcs=ref_wcs,
                                    frame_variables=fv, log=self._log)

    def build_weight(self, cfg, inst, geom):
        """The observation weight of the model (a function of data variables) or None."""
        return self.spec(cfg, inst, geom).build_weight(
            geom, inst.coefficient_catalog(), frame_variables=self.frame_variable_names(cfg, inst),
            log=self._log)

    def build_priors(self, cfg, inst, geom, variables, n_frames):
        """The model's ``[[model.prior]]`` rows as ``setup_lsqr(priors=...)`` callables."""
        return self.spec(cfg, inst, geom).build_priors(geom, variables=variables, n_frames=n_frames)

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
        """Post-solve settings recorded on the cal (default: the read-time Fisher
        threshold for the terms after the first, when a sky term has a coefficient)."""
        if getattr(self, '_spec', None) is not None and self._spec[1].has_coefficients:
            cc.line_fisher_threshold = cfg.params.get('line_fisher_threshold', 10.0)

    def mosaic_geometry(self, cfg, inst, geom, jobgeom):
        """(chunk_maps, offset renderers) for make_mosaic: every offset term's map
        on the reference grid; the primary map rendered by the instrument's
        smooth offset renderer, the others block-constant."""
        from selfcal.models.spec import chunk_map_of
        spec = self.spec(cfg, inst, geom)
        maps, funcs = [], []
        for term in spec.offset:
            cm = chunk_map_of(term, geom)
            maps.append(cm.grid)
            funcs.append(inst.offset_renderer(cfg.instrument_cfg, geom, jobgeom,
                                              map_name=cm.name, render=term.render))
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
    """The sky terms of the spectral presets from their ``[params]``: a constant
    term, then one term per ``[[params.lines]]`` entry — a tabulated coefficient
    of the wavelength (``template_npz``), a Gaussian of it (``center_um`` +
    ``sigma_um``, or ``center_um`` + ``intrinsic_var_um2`` for a per-observation
    width from the band-width variable) — or a single tabulated
    ``line_template_npz`` term, or the catalogue coefficient ``[params].line``
    (default ``pah_3p29``) with ``line_center`` / ``line_sigma`` overrides."""
    from selfcal.models.spec import SkyTerm
    p = cfg.params
    terms = [SkyTerm('continuum')]
    lines = p.get('lines')
    if lines:
        for spec in lines:
            if 'template_npz' in spec:
                coeff = {'variable': 'wavelength', 'function': 'template', 'file': spec['template_npz'],
                         'norm': spec.get('template_norm', 'peak')}
            elif 'center_um' in spec:
                coeff = {'variable': 'wavelength', 'function': 'gaussian', 'center': float(spec['center_um'])}
                if spec.get('sigma_um') is not None:
                    coeff['sigma'] = float(spec['sigma_um'])
                else:
                    coeff.update(width='bandwidth', intrinsic_var=float(spec.get('intrinsic_var_um2', 0.0)))
            else:
                raise ValueError(f"line spec needs 'template_npz' or 'center_um': {spec}")
            terms.append(SkyTerm(name=spec['name'], coefficient=coeff, damp_weight=spec.get('damp_weight')))
        return terms
    npz = p.get('line_template_npz')
    if npz:
        terms.append(SkyTerm(name='pah_3p29', coefficient={
            'variable': 'wavelength', 'function': 'template', 'file': npz,
            'norm': p.get('line_template_norm', 'peak')}))
        return terms
    entry = p.get('line', 'pah_3p29')
    terms.append(SkyTerm(name=entry, coefficient={'catalog': entry, 'center': p.get('line_center'),
                                                  'sigma': p.get('line_sigma')}))
    return terms
