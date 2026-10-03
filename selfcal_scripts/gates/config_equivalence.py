"""What the run engine consumes from a run config, normalised so that two configs that run
identically compare equal: the evidence that a new front end (the Python API, a converter) is a
faithful translation of a TOML config.

Two levels:

* ``effective(cfg)``: every value the engine reads, with each table merged with the defaults of
  the code that reads it (``Calibrator.setup_lsqr`` / ``apply_lsqr`` / ``Mosaicker.make_mosaic`` by
  signature; the dict tables by the engine's own ``.get(key, default)`` values) and the mode's
  ``[params]`` / the ``[model]`` table kept as given. Cheap: no geometry, no data.
* ``snapshot(cfg)``: what the mode lowers to on the instrument's real geometry (the offset model's
  setup kwargs, the sky model evaluated on the detector maps, the aux maps, the mosaic geometry,
  x0, the N-pass clip edges and refit basis), as ``mode_lowering_snapshot.py`` computes it. Needs the
  instrument's calibration files (seconds per SPHEREx detector).

A third check, ``typed``, runs each config through the TOML path and through the Python API
(:mod:`selfcal.run.convert`, then lowered) and compares what the engine does with each
(``engine_view``: the library calls' keywords with defaults filled, resolved per-term damping,
offset rows, sky coefficients, jobs, product paths, frames, staging, tiles, passes).

Usage:
  config_equivalence.py baseline <out_dir> [config.toml ...]   # default: every shipped config
  config_equivalence.py compare <dir_a> <dir_b>
  config_equivalence.py typed [config.toml ...]
"""
import contextlib
import glob
import hashlib
import inspect
import io
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHIPPED = (sorted(glob.glob(os.path.join(REPO, 'selfcal_scripts/configs/*.toml')))
           + sorted(glob.glob(os.path.join(REPO, 'selfcal_scripts/gates/configs/*.toml')))
           + sorted(glob.glob(os.path.join(REPO, 'selfcal_scripts/transfer_function/*.toml')))
           + sorted(glob.glob(os.path.join(REPO, 'examples/quickstart/*.toml'))))


def _sig_defaults(fn):
    return {k: p.default for k, p in inspect.signature(fn).parameters.items()
            if p.default is not inspect.Parameter.empty and k != 'self'}


def _defaults():
    from selfcal.pipeline.pipeline_wrapper import Calibrator, Mosaicker
    from selfcal.run import npass as rnp
    return {
        'calibration': _sig_defaults(Calibrator.setup_lsqr),
        # the engine passes use_float32=True and n_threads=apply_n_threads unless [lsqr] overrides them
        'lsqr': {**_sig_defaults(Calibrator.apply_lsqr), 'use_float32': True},
        'mosaic': _sig_defaults(Mosaicker.make_mosaic),
        # selfcal/run/pipelines.py run_reprojection
        'reproject': dict(padding_pixels=100, max_workers=50, inner_parallel=1, reproj_func='exact',
                          padding_percentage=0.05, replace_existing=False, check=False,
                          header_filter_workers=16, source_ref_path=None),
        # engine.py tiling_frames / tile_assignment, pipelines.py run_tiled
        'tiling': dict(frame_glob='exp_*_det_*.h5', frame_filter='center', halo=0, rss_guardrail=True,
                       line=True, only_tiles=None),
        # npass.py
        'passes': dict(n=4, order='sky_first', stop_tol=0.0, sky_merge='combine', keep_moments=False,
                       sky=dict(rnp._SKY_DEFAULTS), offset=dict(rnp._OFFSET_DEFAULTS)),
    }


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


def _merge(defaults, given):
    out = dict(defaults)
    for k, v in (given or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = {**out[k], **v}
        else:
            out[k] = v
    return _norm(out)


def effective(cfg):
    """Everything the engine reads from ``cfg`` (a RunConfig), defaults filled, normalised."""
    from selfcal.run.config import RunConfig
    D = _defaults()
    top = {}
    for name, f in RunConfig.__dataclass_fields__.items():
        if f.type in ('dict',) or name in ('instrument_cfg', 'params', 'calibration', 'lsqr', 'mosaic', 'zodi',
                                            'reproject', 'tiling', 'passes', 'model', 'hooks'):
            continue
        top[name] = _norm(getattr(cfg, name))
    out = {'top': top, 'instrument_cfg': _norm(dict(cfg.instrument_cfg))}
    if cfg.instrument == 'grid':                       # chunks = [n] means [n, n]
        c = out['instrument_cfg'].get('chunks', [4])
        out['instrument_cfg']['chunks'] = c * 2 if len(c) == 1 else c
    out['params'] = _norm(dict(cfg.params))
    out['model'] = _norm(dict(cfg.model))
    lsqr = dict(D['lsqr'])
    lsqr['n_threads'] = cfg.apply_n_threads
    out['calibration'] = _merge(D['calibration'], cfg.calibration)
    out['lsqr'] = _merge(lsqr, cfg.lsqr)
    out['mosaic'] = _merge(D['mosaic'], cfg.mosaic)
    for t in ('reproject', 'tiling', 'passes'):
        given = getattr(cfg, t)
        out[t] = _merge(D[t], given) if given else None
    out['zodi'] = _norm(dict(cfg.zodi)) or None
    out['hooks'] = _norm(dict(cfg.hooks)) or None
    return out


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


def _memoize_geometry():
    """Build each instrument geometry once per (instrument settings, oversample): many configs share
    a detector, and a SPHEREx geometry takes seconds. The cache only shortcuts identical calls."""
    from selfcal.instruments.base import Instrument
    if getattr(Instrument, '_equivalence_memo', False):
        return
    seen, stack = set(), list(Instrument.__subclasses__())
    while stack:
        cls = stack.pop()
        if cls in seen:
            continue
        seen.add(cls)
        stack.extend(cls.__subclasses__())
        if 'detector_geometry' not in cls.__dict__:
            continue
        orig = cls.__dict__['detector_geometry']

        def cached(self, inst_cfg, oversample, _orig=orig, _cls=cls):
            key = (_cls.__qualname__, json.dumps(dict(inst_cfg), sort_keys=True, default=str), oversample)
            if key not in _GEOMETRY:
                _GEOMETRY[key] = _orig(self, inst_cfg, oversample)
            return _GEOMETRY[key]
        cls.detector_geometry = cached
    Instrument._equivalence_memo = True


def snapshot(cfg):
    """What the mode lowers to on the real geometry (None for reproject / precompute)."""
    if cfg.task not in ('cal', 'mosaic', 'npass'):
        return None
    _memoize_geometry()
    from selfcal import _state
    from selfcal.run.engine import RunContext
    from selfcal_scripts.gates import mode_lowering_snapshot as mls
    _state.set_progress(False)
    with contextlib.redirect_stdout(io.StringIO()):
        ctx = RunContext.build(cfg)
        inst, mode, geom = ctx.inst, ctx.mode, ctx.geom
        job = ctx.jobs()[0]
        jg = ctx.job_geometry(job)
        frames = [f'/probe/exp_{i:04d}_det_{i % 16:02d}.h5' for i in range(mls.N_FRAMES)]
        om = mode.build_offset_model(cfg, inst, geom, jg, job, mls.N_FRAMES, frames=frames)
        snap = {'jobs': [j.name for j in ctx.jobs()], 'mosaic_mode': mode.mosaic_mode,
                'offset': mls._ser(om.to_setup_kwargs()), 'sky': mls._sky(mode.build_sky_model(cfg, inst, geom), geom),
                'setup_kwargs': mls._ser(mode.setup_kwargs(cfg, inst, geom)), 'x0': mode.x0_kind(cfg, inst, geom)}
        aux = mode.aux_maps(cfg, inst, geom)
        snap['aux'] = None if not aux else [mls._h(a) for a in aux.values()]
        cms, funcs = mode.mosaic_geometry(cfg, inst, geom, jg)
        snap['mosaic'] = ([mls._h(c) for c in cms], [f is not None for f in funcs])
        if cfg.task == 'npass':
            for name, call in (('edges', lambda: mode.clip_group_edges(cfg, inst, geom)),
                               ('refit_basis', lambda: mode.refit_poly_basis(cfg, inst, geom, degree=2))):
                try:
                    v = call()
                    snap[name] = mls._h(v) if isinstance(v, np.ndarray) else mls._ser(v)
                except Exception as e:
                    snap[name] = repr(e)[:120]
    return _norm(snap)


def _stem(path):
    rel = os.path.relpath(path, REPO)
    return rel.replace('/', '__').replace('.toml', '')


def baseline(out_dir, paths):
    from selfcal.run.config import load_config
    os.makedirs(out_dir, exist_ok=True)
    ok = 0
    for p in paths:
        stem = _stem(p)
        try:
            cfg = load_config(p)
            rec = {'config': os.path.relpath(p, REPO), 'effective': effective(cfg)}
            try:
                rec['snapshot'] = snapshot(cfg)
            except Exception as e:                    # e.g. geometry inputs missing on this machine
                rec['snapshot'] = f'unavailable: {type(e).__name__}: {str(e)[:120]}'
            with open(os.path.join(out_dir, stem + '.json'), 'w') as f:
                json.dump(rec, f, indent=1, sort_keys=True)
            ok += 1
            s = rec['snapshot']
            print(f"ok   {rec['config']}  snapshot: {'yes' if isinstance(s, dict) else ('n/a' if s is None else s)}")
        except Exception as e:
            print(f"FAIL {os.path.relpath(p, REPO)}: {type(e).__name__}: {str(e)[:160]}")
    print(f'{ok}/{len(paths)} configs written to {out_dir}')


def compare_dirs(a, b):
    names = sorted(set(os.listdir(a)) | set(os.listdir(b)))
    bad = 0
    for n in names:
        pa, pb = os.path.join(a, n), os.path.join(b, n)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f'MISSING {n}')
            bad += 1
            continue
        ra, rb = json.load(open(pa)), json.load(open(pb))
        d = diff(ra['effective'], rb['effective']) + diff(ra.get('snapshot'), rb.get('snapshot'), 'snapshot')
        print(('EQUAL   ' if not d else 'DIFFERS ') + n)
        for line in d[:12]:
            print('    ', line)
        bad += bool(d)
    print('ALL EQUAL' if not bad else f'{bad} DIFFER')
    return bad


# --------------------------------------------------------------------------- the engine's view
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


def engine_view(cfg):
    """What the run engine does with ``cfg``, independent of how the config spells it: the
    library calls' keywords (defaults filled), each sky term's resolved damping, the offset rows,
    the sky coefficients evaluated on the real geometry, the jobs and every product path, the
    frames (by name), staging, tiles and the N-pass schedule."""
    from selfcal import _state
    from selfcal.pipeline.pipeline_wrapper import Calibrator, Mosaicker
    from selfcal.run import engine as E
    from selfcal.run.npass import _OFFSET_DEFAULTS, _SKY_DEFAULTS, schedule
    from selfcal_scripts.gates import mode_lowering_snapshot as mls
    _state.set_progress(False)
    _memoize_geometry()
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
        frames = [f'/probe/exp_{i:04d}_det_{i % 16:02d}.h5' for i in range(mls.N_FRAMES)]
        om = mode.build_offset_model(cfg, inst, geom, jg, job0, mls.N_FRAMES, frames=frames)
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
    sky = mls._sky(sky_model, geom)
    for t in sky:
        t.pop('damp_weight', None)
    view['sky'] = _canon(sky)
    offsets = om.to_setup_kwargs()
    if not regularize:                       # no smoothness or polynomial rows are built
        for k in ('reg_weights', 'poly_constraints_list'):
            offsets.pop(k, None)
    view['offsets'] = _canon(offsets)
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


def typed(paths):
    """Each config through the TOML path and through the Python API (converted, then lowered):
    the engine's views must be equal."""
    from selfcal.run.config import load_config
    from selfcal.run.convert import from_runconfig
    bad = 0
    for p in paths:
        rel = os.path.relpath(p, REPO)
        try:
            cfg = load_config(p)
            conv = from_runconfig(cfg)
            if conv.action in ('calibrate', 'mosaic'):
                lowered = conv.lower()
                if len(lowered) != 1:
                    raise AssertionError(f"{len(lowered)} engine runs for one TOML run")
                a, b = engine_view(cfg), engine_view(lowered[0])
            elif conv.action == 'reproject':
                from selfcal.run.config import RunConfig
                f, kw = conv.field, conv.reproject
                engine_inst, table = f.instrument.engine(())
                low = RunConfig(task='reproject', output_dir=os.path.dirname(f.path), run_name=f.name,
                                resolution_arcsec=f.pixel_scale, instrument_cfg=table,
                                reproject={'input_dirs': kw['exposures'], 'file_pattern': '',
                                           'reproj_func': kw['method'], 'padding_pixels': kw['padding'],
                                           'padding_percentage': kw['padding_fraction'],
                                           'replace_existing': kw['replace'], 'check': kw['verify'],
                                           'source_ref_path': kw['reference'],
                                           'max_workers': conv.compute.resolved_workers})
                low.instrument = engine_inst
                a, b = engine_view(cfg), engine_view(low)
            else:
                cfg2 = load_config(p)
                a, b = engine_view(cfg), engine_view(cfg2)
            d = diff(_norm(json.loads(json.dumps(a, default=str))), _norm(json.loads(json.dumps(b, default=str))))
        except Exception as e:
            d = [f'ERROR {type(e).__name__}: {str(e)[:200]}']
        print(('EQUAL   ' if not d else 'DIFFERS ') + rel + (f"  ({len(conv.notes)} notes)" if not d and conv.notes else ''))
        for line in d[:12]:
            print('    ', line)
        bad += bool(d)
    print(f'{len(paths) - bad}/{len(paths)} equal through the Python API' if bad else
          f'ALL {len(paths)} EQUAL through the Python API')
    return bad


if __name__ == '__main__':
    sys.path.insert(0, REPO)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'baseline':
        baseline(sys.argv[2], sys.argv[3:] or SHIPPED)
    elif cmd == 'compare':
        sys.exit(1 if compare_dirs(sys.argv[2], sys.argv[3]) else 0)
    elif cmd == 'typed':
        sys.exit(1 if typed(sys.argv[2:] or SHIPPED) else 0)
    else:
        print(__doc__)
