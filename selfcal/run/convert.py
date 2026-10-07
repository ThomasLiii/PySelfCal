"""From a TOML run config to the Python API's objects.

:func:`from_runconfig` reads a :class:`~selfcal.run.config.RunConfig` (as
:func:`~selfcal.run.config.load_config` makes it from a TOML file) into a field, a recipe, the
jobs and the action's options; :meth:`Converted.lower` lowers them back onto the engine, and
``selfcal_scripts/gates/config_equivalence.py`` checks that the result runs identically.
:meth:`Converted.to_python` writes the equivalent script. Every value is written out
explicitly (the TOML-era defaults of the library calls differ from the Python API's production
defaults), so the conversion never depends on a default.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from dataclasses import field as dc_field

from ..config.base import ConfigError
from ..config.functions import load_callable
from ..models import model as M
from .compute import Compute
from .recipe import ChunkGroups, Clip, Coadd, Fit, Numerics, Recipe
from .schedule import Passes, Refit, Tiles

__all__ = ['Converted', 'from_runconfig', 'convert_file']

# library defaults of the keywords a TOML table may leave out (setup_lsqr, apply_lsqr, make_mosaic)
_CAL = dict(apply_mask=True, apply_weight=True, outlier_thresh=3.0, ignore_list=None, batch_size=10, max_workers=20,
            offset_regularization=False, weighted_damping=False, damp_weight=0.1, damp_weight_line=None,
            damp_offset=0.0)
_LSQR = dict(atol=1e-6, btol=1e-6, damp=0.01, iter_lim=300, precondition=True, solver='lsmr', use_float32=True)
_MOS = dict(apply_mask=True, apply_weight=True, max_workers=20, make_std_map=False, apply_sigma_clipping=False,
            sigma=2.0, normalize_offset=False, apply_offset=True, ignore_list=None, cache_batch_size=10,
            coadd_batch_size=10, cache_intermediate=False, valid_chunk_thresh=0.01)
_SPECTRAL_AXIS = {'spherex': 'subchannel'}       # the primary chunk map's spectral axis, by instrument


@dataclass
class Converted:
    """A run config as the Python API's objects: ``action`` (``"reproject"``, ``"calibrate"``,
    ``"mosaic"`` or ``"precompute"``) and its arguments, plus ``notes`` on what the conversion
    dropped or changed (keys no code reads)."""
    action: str
    field: object = None
    recipe: Recipe | None = None
    jobs: tuple | None = None
    tiles: Tiles | None = None
    passes: Passes | None = None
    frames: object = None
    cal: str | None = None
    compute: Compute | None = None
    reproject: dict | None = None
    precompute: dict | None = None
    zodi: dict | None = None
    notes: list = dc_field(default_factory=list)
    frames_expr: str | None = None     # how to_python writes ``frames`` (``frames_in(dir)[:n]``)

    def lower(self):
        """The run configs these objects lower to (:func:`~selfcal.run.lower.lower`)."""
        from .lower import lower
        if self.action in ('calibrate', 'mosaic'):
            return [low.cfg for low in lower(self.field, self.recipe, task='mosaic' if self.action == 'mosaic' else 'cal',
                                             jobs=self.jobs, tiles=self.tiles, passes=self.passes, frames=self.frames,
                                             cal=self.cal, compute=self.compute)]
        raise ConfigError(f"Converted.lower(): the {self.action} action is not lowered through a recipe")

    def to_python(self, source=None) -> str:
        """The equivalent run script."""
        from ..config.base import python_repr
        refs = set()
        _collect_functions(self.recipe, refs)
        head = [f'"""{("Converted from " + source) if source else "A selfcal run"} (selfcal.run.convert)."""',
                'import selfcal as sc', 'from selfcal import *  # noqa: F401,F403']
        if self.field is not None and type(self.field.instrument).__name__ == 'SPHEREx':
            head.append('from selfcal.instruments import spherex')
        for r in sorted(refs):
            module, _, name = r.partition(':')
            head.append(f'from {module} import {name}')
        lines = head + ['']
        if self.action == 'precompute':
            kw = {k: v for k, v in self.precompute.items() if k != 'detectors'}
            if 'from selfcal.instruments import spherex' not in lines:
                lines.insert(3, 'from selfcal.instruments import spherex')
            lines += [f'PRECOMPUTE = dict(detectors={python_repr(self.precompute["detectors"])}'
                      + ''.join(f', {k}={python_repr(v)}' for k, v in kw.items()) + ')', '',
                      'if __name__ == "__main__":', '    spherex.precompute_lvf(**PRECOMPUTE)']
            return '\n'.join(lines) + '\n'
        from .compute import Compute
        lines += [f'COMPUTE = {self.compute!r}',
                  f'FIELD = {self.field.replace(compute=Compute())!r}'[:-1] + ', compute=COMPUTE)']
        if self.recipe is not None:
            lines.append(f'RECIPE = {self.recipe!r}')
        if self.action == 'reproject':
            args = [f'    {k}={python_repr(v)},' for k, v in self.reproject.items()]
            lines += ['REPROJECT = dict(', *args, ')', '', 'if __name__ == "__main__":',
                      '    FIELD.reproject(**REPROJECT)']
        else:
            opts = [f'    jobs={python_repr(self.jobs)},']
            for k in ('tiles', 'passes', 'frames', 'cal'):
                v = getattr(self, k)
                if v is not None:
                    text = self.frames_expr if (k == 'frames' and self.frames_expr) else python_repr(v)
                    opts.append(f'    {k}={text},')
            lines += ['RUN = dict(', *opts, ')', '', 'if __name__ == "__main__":',
                      f'    result = FIELD.{self.action}(RECIPE, **RUN)']
            if self.zodi:
                z = ', '.join(f'{k}={python_repr(v)}' for k, v in self.zodi.items())
                lines.append(f'    spherex.zodi_anchor(result, {z})')
            lines.append('    print(result)')
        text = '\n'.join(lines) + '\n'
        if '<array' in text:
            raise ConfigError("to_python(): a setting holds an array, which has no Python form here; write the "
                              "script by hand")
        return text


def _collect_functions(obj, out):
    import dataclasses

    from ..config.base import Config
    from ..config.functions import function_ref
    if isinstance(obj, Config):
        for f in dataclasses.fields(obj):
            _collect_functions(getattr(obj, f.name), out)
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            _collect_functions(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            _collect_functions(x, out)
    elif callable(obj) and hasattr(obj, '__code__'):
        out.add(function_ref(obj))


# =============================================================================== instruments
def _instrument(cfg, notes):
    from ..instruments.camera import Camera
    from ..instruments.euclid.settings import Euclid
    from ..instruments.spherex import settings as sx
    t = dict(cfg.instrument_cfg)
    name = t.pop('name')
    if name == 'spherex':
        inst = sx.SPHEREx(int(t.pop('detector')), num_col=int(t.pop('num_col', 3)), num_sub=int(t.pop('num_sub', 10)),
                          num_ch=int(t.pop('num_ch', 34)), calib_dir=t.pop('calib_dir', None),
                          lvf_dir=t.pop('lvf_dir', None))
        jobs = _spherex_jobs(t, sx)
    elif name == 'grid':
        chunks = t.pop('chunks', [4])
        chunks = [chunks] if isinstance(chunks, int) else list(chunks)
        dq = t.pop('dq_ext', -1)
        inst = Camera(tuple(t.pop('detector_shape')), chunks=tuple(chunks * 2 if len(chunks) == 1 else chunks),
                      sci_ext=int(t.pop('sci_ext', 1)), dq_ext=None if dq is None or int(dq) < 0 else int(dq),
                      reference_ext=None if 'ref_use_ext' not in t else tuple(t.pop('ref_use_ext')),
                      tag=t.pop('tag', None), unit=t.pop('unit', ''), job=t.pop('job_name', 'All'))
        jobs = inst.default_jobs()
    elif name == 'euclid':
        kw = {k: t.pop(k) for k in ('band', 'chunks', 'strips', 'tilt_strips', 'edge_zero_px', 'edge_ramp_px',
                                    'detectors', 'tag') if k in t}
        if 'det_shape' in t:
            kw['det_shape'] = tuple(t.pop('det_shape'))
        if 'ref_use_ext' in t:
            kw['reference_ext'] = tuple(t.pop('ref_use_ext'))
        inst = Euclid(**kw)
        jobs = inst.default_jobs()
    else:
        raise ConfigError(f"no Python form of the instrument {name!r} (a registered plugin): keep its TOML config")
    if t:
        raise ConfigError(f"[instrument] keys {sorted(t)} have no Python form for {name!r}")
    return inst, tuple(jobs)


def _spherex_jobs(t, sx):
    if 'detectors' in t:                            # task precompute
        return ()
    defs = t.pop('window_defs', {}) or {}
    if 'windows' in t:
        return tuple(sx.window(w, subchannels=range(*defs[w]) if w in defs else None) for w in t.pop('windows'))
    if 'subch_window' in t:
        lo, hi = t.pop('subch_window')
        return (sx.window(t.pop('window_name', f'subch{lo}_{hi}'), subchannels=range(int(lo), int(hi))),)
    if 'channel_range' in t:
        a, b = t.pop('channel_range')
        return sx.channels(int(a), int(b) - 1)
    if 'channels' in t:
        return tuple(sx.group(*([c] if isinstance(c, int) else c)) for c in t.pop('channels'))
    return ()                                       # a reprojection: no jobs


# =============================================================================== the model
def _coefficient(c):
    """A coefficient's config form as a ``times`` / ``basis`` value."""
    if c is None:
        return None, None
    c = dict(c)
    if 'catalog' in c:
        name = c.pop('catalog')
        return M.catalog(name, **c), None
    n = c.pop('n', None)
    variable = c.pop('variable')
    function = c.pop('function', None)
    params = dict(c.pop('params', None) or {})
    params.update(c)
    if function is None:
        if params:
            raise ConfigError(f"a coefficient of the variable {variable!r} without a function takes no parameters")
        return variable, n
    of = variable if isinstance(variable, str) else tuple(variable)
    if function in ('template', 'gaussian', 'linear'):
        if function == 'template':
            if 'file' in params:
                return M.template(params.pop('file'), of=of, **params), n
            return M.template(x=params.pop('x'), y=params.pop('y'), of=of, **params), n
        if function == 'gaussian':
            kw = {k: params.pop(k) for k in ('sigma', 'width', 'intrinsic_var', 'fwhm_to_sigma') if k in params}
            return M.gaussian(params.pop('center'), of=of, **kw), n
        return M.linear(params.pop('center'), params.pop('halfwidth'), of=of), n
    return M.Function(load_callable(function), of=of, **params), n


def _offsets_from_term(term):
    """An :class:`~selfcal.models.spec.OffsetTerm` as :class:`~selfcal.models.model.Offsets`."""
    times, _ = _coefficient(term.coefficient)
    basis, n = _coefficient(term.basis)
    common = dict(on=term.map, times=times, damping=float(term.damp), render=term.render, name=term.name)
    if term.kind == 'polybasis':
        return M.Offsets(polynomial=M.Poly(int(term.degree), along=term.axis, each=term.group_axis,
                                           window=range(int(term.lo), int(term.hi) + 1),
                                           segments=None if term.segments is None else tuple(map(tuple, term.segments))),
                         **common)
    per = {'free': 'frame', 'fixed': 'all'}.get(term.kind, term.groups)
    poly = tuple(M.Poly(int(p.degree), along=p.axis, weight=float(p.weight),
                        window=None if p.lo is None else range(int(p.lo), int(p.hi) + 1)) for p in term.poly)
    return M.Offsets(per=per, basis=basis, n=n, smooth=float(term.reg_weight),
                     smooth_along=None if term.adjacency is None else tuple(term.adjacency),
                     smooth_step=term.adjacency_step, poly_prior=poly, mean_zero=bool(term.mean_zero),
                     exact_group_rows=bool(term.exact_group_rows), **common)


def _source(v):
    p = dict(v.params)
    if v.source == 'header':
        return M.Header(v.value, default=v.default)
    if v.source == 'per_frame':
        return M.PerFrame(load_callable(v.value), of=tuple(v.inputs), **p)
    if v.source == 'detector':
        return M.DetectorMap(load_callable(v.value) if isinstance(v.value, str) and ':' in v.value
                             and not os.path.exists(v.value) else v.value, **p)
    if v.source == 'sky':
        return M.SkyMap(load_callable(v.value) if isinstance(v.value, str) and ':' in v.value
                        and not os.path.exists(v.value) else v.value, **p)
    if v.source == 'sky_cal':
        return M.SolvedSky(v.value, term=v.term)
    if v.source == 'layer':
        return M.Layer(None if v.value == v.name else v.value)
    if v.source == 'function':
        return M.Derived(load_callable(v.value), of=tuple(v.inputs), **p)
    return M.FrameFunction(load_callable(v.value), **p)


def _model_from_table(table, dampings):
    from ..models.spec import ModelSpec
    spec = ModelSpec.from_config(table)
    sky = tuple(M.Sky(t.name, times=_coefficient(t.coefficient)[0], damping=d) for t, d in zip(spec.sky, dampings))
    weight = _coefficient(spec.weight)[0] if spec.weight is not None else None
    priors = tuple(M.Prior(p.function if isinstance(p.function, str) and ':' not in p.function
                           else load_callable(p.function), p.terms, weight=p.weight, name=p.name, **dict(p.params))
                   for p in spec.priors)
    return M.Model(sky=sky, offsets=tuple(_offsets_from_term(t) for t in spec.offset), scalar=spec.scalar,
                   variables={v.name: _source(v) for v in spec.variables}, weight=weight, priors=priors), spec.mosaic


def _param(p, *keys, default=None):
    for k in keys:
        if k in p:
            return p[k]
    return default


def _preset_model(cfg, inst_name, notes):
    """The model of a named recipe (``mode`` + ``[params]``), term by term as the mode builds it."""
    from .modes import get_mode
    mode = get_mode(cfg.mode)
    p = dict(cfg.params)
    used = {'line_fisher_threshold'}
    spectral_axis = _SPECTRAL_AXIS.get(inst_name)

    def take(*keys, default=None):
        used.update(keys)
        return _param(p, *keys, default=default)

    def standard_offsets(extra_poly=(), column_default=None):
        adj = take('adjacency_axes')
        poly = []
        weight = take('poly_weight', default=column_default)
        if weight is not None:
            poly.append(M.Poly(int(take('poly_degree', default=1)), along=take('poly_axis'), weight=float(weight)))
        else:
            take('poly_degree', 'poly_axis')
        poly.extend(extra_poly)
        return M.Offsets(smooth=float(take('reg_weight', default=0.1)),
                         smooth_along=None if adj is None else tuple(adj), poly_prior=tuple(poly), mean_zero=True)

    def window():
        lo, hi = take('spectral_poly_lo', 'subch_poly_lo'), take('spectral_poly_hi', 'subch_poly_hi')
        if lo is None or hi is None:
            raise ConfigError("[params] needs spectral_poly_lo / spectral_poly_hi")
        return range(int(lo), int(hi) + 1)

    def spectral_sky():
        lines = take('lines')
        out = []
        if lines:
            for spec in lines:
                if 'template_npz' in spec:
                    times = M.template(spec['template_npz'], norm=spec.get('template_norm', 'peak'))
                elif 'center_um' in spec:
                    if spec.get('sigma_um') is not None:
                        times = M.gaussian(spec['center_um'], sigma=spec['sigma_um'])
                    else:
                        times = M.gaussian(spec['center_um'], width='bandwidth',
                                           intrinsic_var=float(spec.get('intrinsic_var_um2', 0.0)))
                else:
                    raise ConfigError(f"line spec needs 'template_npz' or 'center_um': {spec}")
                out.append((spec['name'], times, spec.get('damp_weight')))
            return out
        npz = take('line_template_npz')
        if npz:
            return [('pah_3p29', M.template(npz, norm=take('line_template_norm', default='peak')), None)]
        entry = take('line', default='pah_3p29')
        return [(entry, M.catalog(entry, center=take('line_center'), sigma=take('line_sigma')), None)]

    name = mode.name
    if name == 'continuum':
        sky, offsets, scalar = [], (standard_offsets(),), True
    elif name in ('spectral', 'spectral_softpoly', 'tiled'):
        sky = spectral_sky()
        extra = ()
        weight = take('spectral_poly_weight', 'subch_poly_weight')
        required = getattr(mode, 'spectral_poly_required', False)
        if weight is not None or required:
            if weight is None:
                raise ConfigError("[params] needs spectral_poly_weight")
            if spectral_axis is None:
                raise ConfigError(f"the spectral polynomial needs a spectral axis ({inst_name!r} has none)")
            extra = (M.Poly(int(take('spectral_poly_degree', 'subch_poly_degree')), along=spectral_axis,
                            weight=float(weight), window=window()),)
        elif name == 'spectral' and (_param(p, 'spectral_poly_lo', 'subch_poly_lo') is not None):
            window()                                 # read by the Gram check only
        offsets = (standard_offsets(extra, getattr(mode, 'column_poly_default_weight', None)),)
        scalar = True
    elif name == 'spectral_polybasis':
        sky = spectral_sky()
        segs = take('spectral_poly_segments', 'subch_poly_segments')
        offsets = (M.Offsets(polynomial=M.Poly(int(take('spectral_poly_degree', 'subch_poly_degree')), window=window(),
                                               segments=None if segs is None else tuple(map(tuple, segs)))),)
        scalar = True
        dead = sorted(set(p) & {'reg_weight', 'poly_weight', 'poly_degree', 'poly_axis', 'adjacency_axes'})
        if dead:
            notes.append(f"[params] {dead}: not read by the polynomial-basis recipe (dropped)")
            used.update(dead)
    elif name == 'two_block_fixed':
        if spectral_axis is None:
            raise ConfigError(f"two_block_fixed smooths along the spectral axis ({inst_name!r} has none)")
        sky = []
        offsets = (M.Offsets(smooth=float(take('reg_weight', default=0.1)), smooth_along=(spectral_axis,),
                             smooth_step=1),
                   M.Offsets(on=take('second_map', default='readout'), per='all',
                             smooth=float(take('second_reg_weight', 'readout_reg_weight', default=0.0)),
                             smooth_along=(), mean_zero=True))
        scalar = False
    else:
        raise ConfigError(f"no Python form of the mode {cfg.mode!r} (defined outside selfcal): keep its TOML config")
    unused = sorted(set(p) - used)
    if unused:
        notes.append(f"[params] {unused}: read by no code path (dropped)")
    return sky, offsets, scalar, mode.mosaic_mode


def _sky_dampings(cal, n_terms, own, passes):
    """Each sky term's damping as the joint solve resolves it; refuses a config whose N-pass SKY
    passes resolve one differently (``damp_weight_line`` unset with line terms)."""
    if not cal['weighted_damping']:
        return [0.0] * n_terms
    dw = float(cal['damp_weight'])
    dwl = cal['damp_weight_line']
    joint, sky_pass = [], []
    for j in range(n_terms):
        o = own[j] if j < len(own) else None
        if o is not None:
            joint.append(float(o))
            sky_pass.append(float(o))
        elif j == 0:
            joint.append(dw)
            sky_pass.append(dw)
        else:
            joint.append(float(dwl) if dwl is not None else 3.0 * dw)
            sky_pass.append(float(dwl) if dwl is not None else 0.0)
    if passes and joint != sky_pass:
        raise ConfigError("the config's N-pass SKY passes damp the line terms differently from its joint solve "
                          "(damp_weight_line unset): no Python form keeps both; set damp_weight_line")
    return joint


# =============================================================================== the run config
def _clip_from(thresh, edges=None, variable=None):
    if thresh is None:
        return None
    if edges is not None:
        return Clip(float(thresh), edges=tuple(edges), variable=variable)
    return Clip(float(thresh))


def _hooks(cfg, inst_name):
    from .config import get_postprocess
    from .engine import resolve_instrument
    out = {}
    for which, spec in (cfg.hooks or {}).items():
        if not spec:
            continue
        spec = {'name': spec} if isinstance(spec, str) else dict(spec)
        name = spec.pop('name')
        factories = dict(resolve_instrument(inst_name).hooks())
        out[which] = factories[name](**spec) if name in factories else get_postprocess(name)
    return out


def from_runconfig(cfg) -> Converted:
    """The Python API's objects of the run config ``cfg`` (see the module docstring)."""
    from ..run.field import Field
    notes = []
    inst_name = cfg.instrument
    if not isinstance(inst_name, str):
        raise ConfigError("from_runconfig(): a run config read from TOML (the instrument by name)")
    if cfg.task == 'precompute':
        t = cfg.instrument_cfg
        kw = {'detectors': list(t['detectors'])}
        for k, name in (('lvf_output_dir', 'output_dir'), ('calib_dir', 'calib_dir'), ('num_sub', 'num_sub'),
                        ('num_ch', 'num_ch')):
            if k in t:
                kw[name] = t[k]
        return Converted('precompute', precompute=kw, notes=notes)
    inst, jobs = _instrument(cfg, notes)
    path = os.path.join(cfg.output_dir, cfg.resolved_run_name()) if cfg.output_dir else None
    cal = {**_CAL, **cfg.calibration}
    mos = {**_MOS, **cfg.mosaic}
    lsq = {**_LSQR, **cfg.lsqr}
    tiling = cfg.tiling or {}
    compute = Compute(None if cfg.cache_dir is None else cfg.cache_dir.rstrip('/'),
                      workers=int(cal['max_workers']), coadd_workers=int(mos['max_workers']),
                      stage=cfg.staging, keep_staged=cfg.keep_nvme, io_limit=cfg.hdd_io_limit,
                      cache_frames=bool(mos['cache_intermediate']),
                      memory_guard=bool(tiling.get('rss_guardrail', True)) if tiling else None,
                      stage_dir=tiling.get('nvme_subdir') if tiling else None)
    field = Field(path, inst, cfg.resolution_arcsec, compute=compute)
    if cfg.task == 'reproject':
        r = dict(cfg.reproject)
        from .engine import resolve_instrument
        engine_inst, table = inst.engine(())
        layout = resolve_instrument(engine_inst).exposure_layout(table)
        for k, default in (('inner_parallel', 1), ('header_filter_workers', 16)):
            if k in r and r.pop(k) != default:
                notes.append(f"[reproject] {k}: dropped (the default is {default})")
        for k, have in (('use_ext', list(layout.ref_use_ext)), ('sci_ext_list', list(layout.sci_ext)),
                        ('dq_ext_list', None if layout.dq_ext is None else list(layout.dq_ext))):
            if k in r and list(r.pop(k) or []) != list(have or []):
                raise ConfigError(f"[reproject] {k} differs from the instrument's layout; give the instrument "
                                  f"those extensions (sc.Camera(sci_ext=, dq_ext=, reference_ext=))")
        file_pattern = r.pop('file_pattern').format(**cfg.instrument_cfg)
        kw = {'exposures': [d + file_pattern for d in r.pop('input_dirs')],
              'reference': r.pop('source_ref_path', None), 'method': r.pop('reproj_func', 'exact'),
              'padding': r.pop('padding_pixels', 100), 'padding_fraction': r.pop('padding_percentage', 0.05),
              'replace': r.pop('replace_existing', False), 'verify': r.pop('check', False)}
        compute = compute.replace(workers=int(r.pop('max_workers', 50)))
        if r:
            raise ConfigError(f"[reproject] keys {sorted(r)} have no Python form")
        return Converted('reproject', field=field.replace(compute=compute), compute=compute, reproject=kw, notes=notes)

    # ---- the model -------------------------------------------------------------------------
    own = []
    if cfg.mode == 'model':
        from ..models.spec import ModelSpec
        spec = ModelSpec.from_config(cfg.model)
        own = [t.damp_weight for t in spec.sky]
        dampings = _sky_dampings(cal, len(spec.sky), own, bool(cfg.passes))
        model, mosaic_mode = _model_from_table(cfg.model, dampings)
    else:
        sky, offsets, scalar, mosaic_mode = _preset_model(cfg, inst_name, notes)
        names = ['continuum'] + [s[0] for s in sky]
        own = [None] + [s[2] for s in sky]
        dampings = _sky_dampings(cal, len(names), own, bool(cfg.passes))
        terms = (M.Sky(damping=dampings[0]),) + tuple(M.Sky(n, times=t, damping=d)
                                                      for (n, t, _), d in zip(sky, dampings[1:]))
        model = M.Model(sky=terms, offsets=offsets, scalar=scalar)
    if not cal['offset_regularization']:
        model = model.replace(offsets=tuple(o if o.polynomial is not None else
                                            o.replace(smooth=0.0, poly_prior=()) for o in model.offsets))
        notes.append("offset_regularization = false: no smoothness or polynomial rows")
    if cal['damp_offset']:
        model = model.replace(offsets=tuple(o.replace(damping=float(cal['damp_offset'])) for o in model.offsets))
        notes.append(f"damp_offset = {cal['damp_offset']}: written as each offset term's damping")

    # ---- fit, coadd, numerics -------------------------------------------------------------
    hooks = _hooks(cfg, inst_name)
    post = hooks.get('post_cal')
    if cfg.postprocess:
        from .config import get_postprocess
        post = get_postprocess(cfg.postprocess)
    group_axis = None
    if cal.get('outlier_groups'):
        group_axis = cal['outlier_groups'].get('along')
    clip = _clip_from(cal['outlier_thresh'], cal.get('outlier_group_edges'),
                      cal.get('outlier_group_variable') or cal.get('outlier_aux_key'))
    if group_axis is not None:
        clip = Clip(clip.sigma, per=ChunkGroups.along(group_axis))
    tol = (float(lsq['atol']), float(lsq['btol']))
    fit = Fit(int(lsq['iter_lim']), clip=clip, use_mask=bool(cal['apply_mask']),
              ignore_flags=tuple(cal['ignore_list'] or ()), shot_noise_weights=bool(cal['apply_weight']),
              tolerance=tol[0] if tol[0] == tol[1] else tol, method=lsq['solver'], damp=float(lsq['damp']),
              precondition=bool(lsq['precondition']), float32=bool(lsq['use_float32']),
              frame_hook=post, raw_frame_hook=hooks.get('pre_cal'),
              line_fisher_threshold=float(cfg.params.get('line_fisher_threshold', 10.0)))
    makes_mosaic = cfg.task == 'mosaic' or (not cfg.skip_mosaic and mosaic_mode != 'none')
    coadd = None
    if makes_mosaic:
        coadd = Coadd(float(mos['sigma']) if mos['apply_sigma_clipping'] else None, std=bool(mos['make_std_map']),
                      use_mask=bool(mos['apply_mask']), ignore_flags=tuple(mos['ignore_list'] or ()),
                      shot_noise_weights=bool(mos['apply_weight']), oversample=int(cfg.oversample),
                      instrument_maps=bool(cfg.wavelength_coadd) and mosaic_mode == 'full',
                      min_chunk_coverage=float(mos['valid_chunk_thresh']), subtract_offsets=bool(mos['apply_offset']),
                      normalize_offsets=bool(mos['normalize_offset']), frame_hook=hooks.get('post_mosaic'))
    numerics = Numerics(int(cfg.apply_n_threads), batch=int(cal['batch_size']),
                        mosaic_batch=int(mos['cache_batch_size']), coadd_batch=int(mos['coadd_batch_size']))
    suffix = cfg.suffix
    tiles = None
    if tiling:
        stitched = tiling['stitched_suffix']
        name = stitched.lstrip('_')
        if stitched and not stitched.startswith('_') or suffix and not suffix.startswith('_'):
            raise ConfigError(f"suffixes {suffix!r} / {stitched!r} must start with '_'")
        tiles = _tiles(tiling, suffix, name, notes)
    else:
        if suffix and not suffix.startswith('_'):
            raise ConfigError(f"suffix {suffix!r}: a recipe's name is joined with '_'; the suffix must start with '_'")
        name = suffix[1:] if suffix else ''
    recipe = Recipe(model, fit=fit, coadd=coadd, numerics=numerics, name=name)

    # ---- frames, passes ---------------------------------------------------------------------
    frames = None
    frames_expr = None
    reproj_dir = os.path.join(field.path, 'reprojected')
    if tiling:
        if os.path.normpath(tiling['full_reproj_dir']) != os.path.normpath(reproj_dir):
            frames = tiling['full_reproj_dir']
    elif cfg.reproj_override:
        if os.path.normpath(cfg.reproj_override) == os.path.normpath(reproj_dir):
            field = field.replace(compute=compute.replace(stage=None))
            compute = field.compute
        else:
            frames = cfg.reproj_override
    if cfg.n_frames:
        if frames is not None:                     # the first n frames of another directory
            import glob
            source = frames
            frames = sorted(glob.glob(os.path.join(frames, '*.h5')))[:int(cfg.n_frames)]
            if len(frames) < int(cfg.n_frames):
                raise ConfigError(f"n_frames = {cfg.n_frames}: {cfg.reproj_override} holds {len(frames)} frames")
            frames_expr = f"frames_in({source!r})[:{int(cfg.n_frames)}]"
        else:
            frames = int(cfg.n_frames)
    passes = _passes(cfg.passes, cal, inst_name) if cfg.task == 'npass' else None
    zodi = None
    if cfg.zodi.get('pred_dir'):
        z = dict(cfg.zodi)
        zodi = {'predictions': z.pop('pred_dir'), **z}
        notes.append("[zodi]: the anchor is fitted by spherex.zodi_anchor(result, ...) after the calibration")
    action = 'mosaic' if cfg.task == 'mosaic' else 'calibrate'
    return Converted(action, field=field, recipe=recipe, jobs=jobs, tiles=tiles, passes=passes, frames=frames,
                     cal=cfg.cal_override, compute=compute, zodi=zodi, notes=notes, frames_expr=frames_expr)


def _tiles(t, suffix, name, notes):
    t = dict(t)
    kw = dict(assign=t.pop('frame_filter', 'center'), halo=int(t.pop('halo', 0)),
              tile_name=suffix.lstrip('_'), stitched_name='{name}')
    if t.get('only_tiles') is not None:
        kw['only'] = tuple(t.pop('only_tiles'))
    if 'tiles' in t:
        kw['boxes'] = {x['name']: tuple(x['bbox']) for x in t.pop('tiles')}
    else:
        kw['grid'] = tuple(t.pop('grid'))
        kw['overlap'] = int(t.pop('overlap_px', 0))
        names = t.pop('tile_names', None)
        kw['names'] = None if names is None else tuple(names)
    glob = t.get('frame_glob')
    if glob not in (None, 'exp_*_det_00.h5', 'exp_*_det_*.h5'):
        raise ConfigError(f"[tiling].frame_glob = {glob!r}: no Python form (a tiled run takes every frame)")
    if glob == 'exp_*_det_00.h5':
        notes.append("[tiling].frame_glob = 'exp_*_det_00.h5': every frame is taken (the same frames for a "
                     "one-detector instrument)")
    return Tiles(**kw)


def _per(subch, inst_name):
    if not subch:
        return 'frame'
    axis = _SPECTRAL_AXIS.get(inst_name)
    if axis is None:
        raise ConfigError(f"subch_clip groups along the spectral axis ({inst_name!r} has none)")
    return ChunkGroups.along(axis)


def _passes(p, cal, inst_name):
    from .npass import _OFFSET_DEFAULTS, _SKY_DEFAULTS
    sky = {**_SKY_DEFAULTS, **p.get('sky', {})}
    off = {**_OFFSET_DEFAULTS, **p.get('offset', {})}
    init = p.get('init')
    init_clip = None
    if init:
        thresh = init.get('outlier_thresh', cal['outlier_thresh'])
        flags = init.get('ignore_list')
        init_clip = Clip(float(thresh), per=_per(init.get('subch_clip'), inst_name),
                         ignore_flags=None if flags is None else tuple(flags))
    return Passes(int(p.get('n', 4)), order=p.get('order', 'sky_first'), init_clip=init_clip,
                  sky_clip=Clip(float(sky['outlier_thresh']), per=_per(sky['subch_clip'], inst_name)),
                  offset=Refit(int(off['poly_degree']),
                               clip=Clip(float(off['outlier_thresh']), per=_per(off['subch_clip'], inst_name)),
                               bright_cut=off['bright_cut'], min_pixels=int(off['min_pix']),
                               segments=None if off.get('segments') is None else tuple(map(tuple, off['segments'])),
                               ridge=float(off.get('ridge', 0.0))),
                  stop_tol=float(p.get('stop_tol', 0.0)), sky_merge=p.get('sky_merge', 'combine'),
                  keep_moments=bool(p.get('keep_moments', False)),
                  ends_on_offset=_ends_on_offset(int(p.get('n', 4)), p.get('order', 'sky_first')))


def _ends_on_offset(n, order):
    from .schedule import schedule
    return n > 1 and schedule(n, order)[-1] == 'offset'


def _import_script(path):
    """The module of the script ``path``, imported without running its ``__main__`` block."""
    import importlib.util
    name = '_selfcal_converted_' + os.path.splitext(os.path.basename(path))[0].replace('-', '_')
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def convert_file(config_path, out_path, *, check=True, force=False) -> list[str]:
    """Write the Python form of the TOML run config ``config_path`` to ``out_path`` and return the
    conversion's notes. With ``check``, the script is first written beside ``out_path``, imported
    and its objects lowered again: unless the run engine would do exactly what the TOML makes it do
    (:func:`selfcal.run.equivalence.differences`), nothing is written and
    :class:`~selfcal.config.base.ConfigError` names the differences. An existing ``out_path`` is
    replaced only with ``force``."""
    from .config import load_config
    if os.path.exists(out_path) and not force:
        raise ConfigError(f"{out_path} exists; give another output (-o) or pass force=True (--force) to replace it")
    cfg = load_config(config_path)
    conv = from_runconfig(cfg)
    text = conv.to_python(os.path.relpath(config_path))
    directory, name = os.path.split(os.path.abspath(out_path))
    draft = os.path.join(directory, f'_selfcal_convert_{os.getpid()}_{name}')
    with open(draft, 'w') as f:
        f.write(text)
    try:
        notes = _check_converted(conv, cfg, config_path, draft) if check else conv.notes
        os.replace(draft, out_path)
        return notes
    finally:
        if os.path.exists(draft):
            os.remove(draft)


def _check_converted(conv, cfg, config_path, out_path) -> list[str]:
    """Raise unless the script at ``out_path`` makes the run engine do what ``cfg`` does."""
    if conv.action == 'precompute':
        import inspect

        from ..instruments.spherex.settings import precompute_lvf
        module = _import_script(out_path)
        try:
            inspect.signature(precompute_lvf).bind(**module.PRECOMPUTE)
        except TypeError as e:
            raise ConfigError(f"{config_path}: the converted script would not run ({e}); nothing written") from None
        return conv.notes
    from .equivalence import differences
    from .lower import lower
    module = _import_script(out_path)
    if conv.action == 'reproject':
        mine = [module.FIELD.reprojection_config(**module.REPROJECT)]
    else:
        mine = [low.cfg for low in lower(module.FIELD, module.RECIPE,
                                         task='mosaic' if conv.action == 'mosaic' else 'cal', **module.RUN)]
    problems = [f'{len(mine)} engine runs for one TOML run'] if len(mine) != 1 else differences(cfg, mine[0])
    if problems:
        raise ConfigError(f"{config_path}: the converted script would not run identically ("
                          + '; '.join(problems[:6]) + "); nothing written")
    return conv.notes
