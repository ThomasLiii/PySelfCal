"""Whether two run configs make the run engine do the same thing.

:func:`engine_view` is what the engine does with a :class:`~selfcal.run.config.RunConfig`,
independent of how the config spells it: the library calls' keywords with their defaults filled,
each sky term's resolved damping, the offset rows, the sky coefficients evaluated on the
instrument's real maps, the jobs and every product path, the frames (by name), staging, tiles and
the N-pass schedule. :func:`differences` compares two configs that way. The TOML converter
(:func:`selfcal.run.convert.convert_file`) checks its scripts with it, and
``selfcal_scripts/gates/config_equivalence.py typed`` checks every shipped config.
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import inspect
import io
import json
import os

import numpy as np

__all__ = ['engine_view', 'differences', 'cached_geometry', 'diff']

N_FRAMES = 137          # frames of the probe frame list the offset model is lowered over


def _h(a):
    a = np.ascontiguousarray(a)
    return (a.shape, str(a.dtype), hashlib.sha1(a.tobytes()).hexdigest())


def _ser(x):
    if isinstance(x, np.ndarray):
        return ('ndarray',) + _h(x)
    if isinstance(x, dict):
        return {k: _ser(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return type(x).__name__, [_ser(v) for v in x]
    if callable(x):
        return ('callable', getattr(x, '__name__', type(x).__name__))
    if hasattr(x, '__dataclass_fields__'):
        return (type(x).__name__, {k: _ser(getattr(x, k)) for k in x.__dataclass_fields__})
    return x


def _sky(model, geom):
    """Each sky term by what it COMPUTES, independent of how the component class is laid
    out: name, damping, the variables it reads, and the hash of its coefficients evaluated
    on a fixed probe of the instrument's real per-pixel maps and the built-in detector
    coordinates ``det_x`` / ``det_y`` (every 97th detector pixel)."""
    aux = getattr(geom, 'aux', None) or {}
    probe = {k: np.ascontiguousarray(np.asarray(v).ravel()[::97]) for k, v in aux.items()}
    if getattr(geom, 'shape', None) is not None:
        rows, cols = (int(n) for n in geom.shape)
        pixel = np.arange(0, rows * cols, 97)
        probe.setdefault('det_x', (pixel % cols).astype(np.float32))
        probe.setdefault('det_y', (pixel // cols).astype(np.float32))
    out = []
    for c in model.components:
        coeff = c.coefficients(probe)
        reads = getattr(c, 'aux_requirements', None) or ()
        out.append({'name': c.name, 'damp_weight': getattr(c, 'damp_weight', None),
                    'reads': tuple(sorted(reads)),
                    'coeff': None if coeff is None else _h(np.asarray(coeff))})
    return out


def _sig_defaults(fn):
    return {k: p.default for k, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty and k != 'self'}


def _norm(v):
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _norm(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        return ['ndarray', list(v.shape), str(v.dtype), hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest()]
    if callable(v):
        return ['callable', f"{getattr(v, '__module__', '?')}:{getattr(v, '__qualname__', type(v).__name__)}"]
    return v


def diff(a, b, path=''):
    """The differences between two normalised structures, as 'path: a != b' lines."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k, '<absent>'), b.get(k, '<absent>'), f'{path}.{k}' if path else k)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f'{path}[{i}]')
        return out
    return [] if a == b else [f'{path}: {str(a)[:80]} != {str(b)[:80]}']


_GEOMETRY = {}


def _subclasses(root):
    seen, stack = [], list(root.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls not in seen:
            seen.append(cls)
            stack.extend(cls.__subclasses__())
    return seen


@contextlib.contextmanager
def cached_geometry():
    """Within the block, build each instrument's detector geometry once per (settings, oversample):
    many configs share a detector, and a SPHEREx geometry takes seconds. Two levels: the engine
    instrument's ``detector_geometry`` by its whole ``[instrument]`` table, and the settings
    object's ``geometry`` (``sc.SPHEREx``, ``sc.Euclid``, ``sc.Camera``, ...; the registry
    adapters delegate to it) by the settings, which leave out the job selection, so configs of
    one detector that select other channels or windows share one geometry."""
    from ..instruments.base import Instrument
    from ..instruments.contract import Instrument as Settings
    patched = []
    for cls in _subclasses(Instrument):
        if 'detector_geometry' not in cls.__dict__:
            continue
        orig = cls.__dict__['detector_geometry']

        def cached(self, inst_cfg, oversample, _orig=orig, _cls=cls):
            key = (_cls.__qualname__, json.dumps(dict(inst_cfg), sort_keys=True, default=str), oversample)
            if key not in _GEOMETRY:
                _GEOMETRY[key] = _orig(self, inst_cfg, oversample)
            return _GEOMETRY[key]
        patched.append((cls, 'detector_geometry', orig, cached))
    for cls in _subclasses(Settings):
        if 'geometry' not in cls.__dict__:
            continue
        orig = cls.__dict__['geometry']

        def cached_settings(self, oversample=1, _orig=orig):
            try:
                key = ('settings', type(self), self, oversample)
                hash(key)
            except TypeError:                   # settings holding arrays: not shared
                return _orig(self, oversample)
            if key not in _GEOMETRY:
                _GEOMETRY[key] = _orig(self, oversample)
            return _GEOMETRY[key]
        patched.append((cls, 'geometry', orig, cached_settings))
    for cls, name, _, wrapper in patched:
        setattr(cls, name, wrapper)
    try:
        yield
    finally:
        for cls, name, orig, _ in patched:
            setattr(cls, name, orig)


def _canon(x):
    """A comparable form: arrays by shape, type and hash; tuples and lists alike; dataclasses by
    their fields; functions by module:qualname; other objects by class and state."""
    if isinstance(x, np.ndarray):
        return ['ndarray', list(x.shape), str(x.dtype), hashlib.sha1(np.ascontiguousarray(x).tobytes()).hexdigest()]
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, dict):
        return {str(k): _canon(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_canon(v) for v in x]
    if isinstance(x, range):
        return ['range', x.start, x.stop, x.step]
    if hasattr(x, '__dataclass_fields__'):
        return [type(x).__name__, {k: _canon(getattr(x, k)) for k in x.__dataclass_fields__}]
    if callable(x) and hasattr(x, '__code__'):
        return ['function', f"{x.__module__}:{x.__qualname__}"]
    if callable(x) or hasattr(x, '__dict__'):
        state = getattr(x, '__getstate__', lambda: vars(x))() if hasattr(x, '__dict__') else None
        return [f"{type(x).__module__}:{type(x).__qualname__}", _canon(state) if isinstance(state, dict) else None]
    return x


_SETUP_DEFAULTS = None


def _frames_sha(files):
    names = [os.path.basename(f) for f in files]
    return [len(names), hashlib.sha1('\n'.join(names).encode()).hexdigest()]


def _built_offset_rows(offsets, regularize):
    """The offset model's setup keywords (``OffsetModel.to_setup_kwargs()``) with each map's
    smoothness and soft-polynomial rows described as the solver builds them
    (``core/assembly.py`` ``_prep_lsqr``; ``core/system.py``, the grouped adjacency): smoothness
    rows only with ``offset_regularization``, a positive weight and at least one adjacent pair,
    a polynomial group only with ``offset_regularization``, a nonzero weight and at least one
    chain, and neither on a template or polynomial-basis map. Rows that are not built read
    alike however they were left out (the switch off, a zero weight, no pairs, no polynomial):
    weight 0.0, no adjacency, no polynomial groups."""
    out = dict(offsets)
    reg_weights, adj_infos, polys = [], [], []
    for m in range(len(offsets['chunk_maps'])):
        free = (regularize and offsets['det_templates'][m] is None
                and offsets['poly_basis_list'][m] is None)
        weight, adj = offsets['reg_weights'][m], offsets['adj_infos'][m]
        smooth = (free and weight > 0 and adj is not None
                  and not all(np.asarray(a).size == 0 for a in adj))
        reg_weights.append(weight if smooth else 0.0)
        adj_infos.append(adj if smooth else None)
        groups = offsets['poly_constraints_list'][m]
        built = [g for g in groups or () if free and float(g['weight']) != 0 and np.shape(g['chains'])[0] > 0]
        polys.append(groups if groups and len(built) == len(groups) else (built or None))
    out.update(reg_weights=reg_weights, adj_infos=adj_infos, poly_constraints_list=polys)
    return out


def engine_view(cfg):
    """What the run engine does with ``cfg``, independent of how the config spells it: the
    library calls' keywords (defaults filled), each sky term's resolved damping, the offset rows,
    the sky coefficients evaluated on the real geometry, the jobs and every product path, the
    frames (by name), staging, tiles and the N-pass schedule."""
    from .. import _state
    from ..pipeline.pipeline_wrapper import Calibrator, Mosaicker
    from . import engine as E
    from .npass import _OFFSET_DEFAULTS, _SKY_DEFAULTS, schedule
    _state.set_progress(False)
    view = {'task': cfg.task}
    if cfg.task in ('reproject', 'precompute'):
        inst = E.resolve_instrument(cfg.instrument)
        if cfg.task == 'precompute':
            view['precompute'] = _canon(dict(cfg.instrument_cfg))
            return view
        r = dict(cfg.reproject)
        layout = inst.exposure_layout(cfg.instrument_cfg)
        pattern = r['file_pattern'].format(**cfg.instrument_cfg)
        files = sorted(sum((glob.glob(d + pattern) for d in r['input_dirs']), []))
        view['exposures'] = _frames_sha(files) if files else ['none found', [d + pattern for d in r['input_dirs']]]
        view['layout'] = _canon({'use_ext': r.get('use_ext', list(layout.ref_use_ext)),
                                 'sci_ext': r.get('sci_ext_list', list(layout.sci_ext)),
                                 'dq_ext': r.get('dq_ext_list', layout.dq_ext), 'detector_ids': layout.detector_ids,
                                 'reader': layout.reader, 'header_keys': layout.header_keys,
                                 'cache_tag': layout.cache_tag})
        view['reproject'] = _canon({'padding_pixels': r.get('padding_pixels', 100), 'max_workers': r.get('max_workers', 50),
                                    'reproj_func': r.get('reproj_func', 'exact'),
                                    'padding_percentage': r.get('padding_percentage', 0.05),
                                    'replace_existing': r.get('replace_existing', False), 'check': r.get('check', False),
                                    'source_ref_path': r.get('source_ref_path'),
                                    'output': os.path.join(cfg.output_dir, cfg.resolved_run_name()),
                                    'resolution_arcsec': cfg.resolution_arcsec})
        return view
    with contextlib.redirect_stdout(io.StringIO()):
        ctx = E.RunContext.build(cfg)
        inst, mode, geom = ctx.inst, ctx.mode, ctx.geom
        jobs = ctx.jobs()
        spec = mode.spec(cfg, inst, geom)
        sky_model = mode.build_sky_model(cfg, inst, geom)
        job0 = jobs[0]
        jg = ctx.job_geometry(job0)
        frames = [f'/probe/exp_{i:04d}_det_{i % 16:02d}.h5' for i in range(N_FRAMES)]
        om = mode.build_offset_model(cfg, inst, geom, jg, job0, N_FRAMES, frames=frames)
    view['jobs'] = _canon([(j.name, j.kind, j.value) for j in jobs])
    view['frame_tag'] = ctx.frame_tag
    view['unit'] = inst.data_unit(cfg.instrument_cfg)
    view['run'] = [cfg.output_dir and os.path.join(cfg.output_dir, cfg.resolved_run_name()),
                   cfg.resolution_arcsec]

    # setup_lsqr: the keywords solve_job passes (models compared below), defaults filled
    calk = dict(ctx.cal_kwargs)
    calk.update(mode.setup_kwargs(cfg, inst, geom))
    groups = calk.pop('outlier_groups', None)
    if groups is not None:
        calk.update(E.clip_groups(ctx, groups))
    if not (calk.get('outlier_group_variable') or calk.get('outlier_aux_key')):
        calk['outlier_group_variable'] = geom.wavelength_key
    pre, post = E.resolve_hook(cfg, inst, 'pre_cal'), E.resolve_hook(cfg, inst, 'post_cal')
    if pre is not None:
        calk['preprocess_func'] = pre
    if post is not None:
        calk['postprocess_func'] = post
    setup = {**_sig_defaults(Calibrator.setup_lsqr), 'ignore_list': [], **calk}
    if setup['ignore_list'] is None:
        setup['ignore_list'] = []
    regularize, weighted = bool(setup.pop('offset_regularization')), bool(setup.pop('weighted_damping'))
    dw, dwl = float(setup.pop('damp_weight')), setup.pop('damp_weight_line')
    if dwl is None and len(sky_model.components) > 1:
        dwl = 3.0 * dw
    damping = sky_model.damp_weights(dw, dwl) if weighted else [0.0] * len(sky_model.components)
    for k in ('max_workers', 'batch_spill_dir'):
        setup.pop(k, None)
    view['setup_lsqr'] = _canon(setup)
    view['workers'] = calk.get('max_workers', 20)
    view['sky_damping'] = damping
    sky = _sky(sky_model, geom)
    for t in sky:
        t.pop('damp_weight', None)
    view['sky'] = _canon(sky)
    view['offsets'] = _canon(_built_offset_rows(om.to_setup_kwargs(), regularize))
    view['x0'] = mode.x0_kind(cfg, inst, geom)
    aux = mode.aux_maps(cfg, inst, geom)
    view['aux'] = None if not aux else _canon(list(aux.values()))
    view['line_fisher_threshold'] = cfg.params.get('line_fisher_threshold', 10.0) if spec.has_coefficients else None
    view['apply_lsqr'] = _canon({**_sig_defaults(Calibrator.apply_lsqr), 'use_float32': True,
                                 'n_threads': cfg.apply_n_threads, **cfg.lsqr})

    # the mosaic, when one is made
    makes_mosaic = cfg.task == 'mosaic' or (cfg.task == 'cal' and not cfg.tiling and not cfg.skip_mosaic
                                            and mode.mosaic_mode != 'none')
    if makes_mosaic:
        mos = {**_sig_defaults(Mosaicker.make_mosaic), 'ignore_list': [], **cfg.mosaic}
        if mos['ignore_list'] is None:
            mos['ignore_list'] = []
        mos['postprocess_func'] = E.resolve_hook(cfg, inst, 'post_mosaic')
        mos['oversample'] = cfg.oversample
        mos['instrument_maps'] = (mode.mosaic_mode == 'full' and cfg.wavelength_coadd
                                  and inst.aux_coadds(geom) is not None)
        # sigma is read by the sigma-clip pass (it runs with the std map only) and by the instrument's
        # aux coadds (engine.mosaic_job, core/coadd.run_coadd_schedule); elsewhere it is unused, and
        # is described as make_mosaic's default whatever the config says
        if not ((mos['make_std_map'] and mos['apply_sigma_clipping']) or mos['instrument_maps']):
            mos['sigma'] = _sig_defaults(Mosaicker.make_mosaic)['sigma']
        mos['coadd_workers'] = mos.pop('max_workers')
        with contextlib.redirect_stdout(io.StringIO()):
            cms, funcs = mode.mosaic_geometry(cfg, inst, geom, jg)
        mos['geometry'] = [_canon(c) for c in cms]
        mos['renderers'] = [_canon(f) for f in funcs]
        view['mosaic'] = _canon(mos)
    else:
        view['mosaic'] = None

    # products, frames, staging
    base = cfg.tiling['stitched_suffix'] if cfg.tiling else cfg.suffix
    view['products'] = {j.name: {'cal': ctx.cal_path(j), 'mosaic': ctx.mosaic_path(j) if makes_mosaic else None,
                                 'npass_stem': 'cal_' + ctx.stem(j, base) if cfg.task == 'npass' else None}
                        for j in jobs}
    reproj = ctx.pipeline_config.reproj_dir
    if cfg.tiling:
        t = cfg.tiling
        tiles, only = E.resolve_tiles(t, tuple(t['ref_shape']))
        view['tiling'] = _canon({'tiles': [(x.name, x.bbox) for x in tiles], 'only': only,
                                 'ref_shape': t['ref_shape'], 'frame_filter': t.get('frame_filter', 'center'),
                                 'halo': t.get('halo', 0), 'line': t.get('line', True),
                                 'stage_dir': ctx.tiling_nvme_dir(), 'memory_guard': t.get('rss_guardrail', True),
                                 'tile_cals': {x.name: ctx.tile_cal_file(job0, x) for x in tiles},
                                 'stitched': ctx.stitched_cal_path(job0)})
        files = E.tiling_frames(t) if os.path.isdir(t['full_reproj_dir']) else None
        view['frames'] = ['tiled', t['full_reproj_dir'], _frames_sha(files) if files else 'not on this machine']
    else:
        where = cfg.reproj_override or reproj
        if cfg.frame_files:
            names = [os.path.basename(f) for f in cfg.frame_files]
        elif os.path.isdir(where):
            names = [os.path.basename(f) for f in E.frame_list(where, cfg.n_frames)]
        else:
            names = None
        view['frames'] = [where, _frames_sha(names) if names else 'not on this machine']
        view['staging'] = None if cfg.reproj_override else _canon({
            'how': cfg.staging, 'dir': getattr(cfg, 'stage_dir', None) or os.path.join(cfg.cache_dir, f'reproj_nvme_{ctx.pipeline_config.run_name}'),
            'keep': cfg.keep_nvme, 'io_limit': cfg.hdd_io_limit})
    view['cache_dir'] = cfg.cache_dir if (makes_mosaic and cfg.mosaic.get('cache_intermediate')) or cfg.tiling \
        or cfg.task == 'npass' else 'unused'
    view['cal_override'] = cfg.cal_override

    # the N-pass schedule
    if cfg.task == 'npass':
        p = cfg.passes
        n, order = int(p.get('n', 4)), p.get('order', 'sky_first')
        init = dict(p.get('init', {}))
        init_clip = {'outlier_thresh': init.get('outlier_thresh', calk.get('outlier_thresh')),
                     'grouped': bool(init.get('subch_clip')) if 'subch_clip' in init
                     else calk.get('outlier_group_edges') is not None,
                     'ignore_list': init.get('ignore_list', calk.get('ignore_list') or [])}
        view['passes'] = _canon({'schedule': schedule(n, order), 'stop_tol': p.get('stop_tol', 0.0),
                                 'sky_merge': p.get('sky_merge', 'combine'), 'keep_moments': p.get('keep_moments', False),
                                 'init': init_clip, 'sky': {**_SKY_DEFAULTS, **p.get('sky', {})},
                                 'offset': {**_OFFSET_DEFAULTS, **p.get('offset', {})}})
        with contextlib.redirect_stdout(io.StringIO()):
            view['npass_edges'] = _canon(mode.clip_group_edges(cfg, inst, geom))
            off = {**_OFFSET_DEFAULTS, **p.get('offset', {})}
            view['npass_refit_basis'] = _canon(mode.refit_poly_basis(cfg, inst, geom, int(off['poly_degree']),
                                                                     segments=off.get('segments')))
    return view


def differences(a, b) -> list[str]:
    """The differences between what the engine does with the run configs ``a`` and ``b`` (empty:
    the same run)."""
    with cached_geometry():
        va = _norm(json.loads(json.dumps(engine_view(a), default=str)))
        vb = _norm(json.loads(json.dumps(engine_view(b), default=str)))
    return diff(va, vb)
