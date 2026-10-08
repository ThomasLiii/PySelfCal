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
(:func:`selfcal.run.equivalence.engine_view`: the library calls' keywords with defaults filled,
resolved per-term damping, offset rows, sky coefficients, jobs, product paths, frames, staging,
tiles, passes).

``views`` records those engine views, so that a rewrite of what the engine reads can be shown to
leave what it does unchanged (``compare-views`` of the directories made before and after): one
JSON file per shipped TOML config, ``toml__<its path in the repository, '/' as '__', no
.toml>.json``, holding its view, and one per run script, ``script__<name>.json``, holding a list,
one view per engine run of the script's action, in order (a campaign's, FIELDS:
``{'<i> <field name>': that list}``, per field in order). A view that cannot be made is stored as
``{"ERROR": "<type>: <message>"}``. The views read this machine's files (frame lists, reference
grids, calibration data) and its CPU count (the default number of workers): compare views made
on one machine.

Usage:
  config_equivalence.py baseline <out_dir> [config.toml ...]   # default: every shipped config
  config_equivalence.py compare <dir_a> <dir_b>
  config_equivalence.py typed [config.toml ...]
  config_equivalence.py runs [name ...]        # selfcal_scripts/runs/<name>.py vs configs/<name>.toml
  config_equivalence.py views <out_dir> [name ...]   # default: every shipped config and run script
  config_equivalence.py compare-views <dir_a> <dir_b>
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


def typed(paths):
    """Each config through the TOML path and through the Python API (converted, then lowered):
    the engine's views must be equal (:func:`selfcal.run.equivalence.engine_view`)."""
    from selfcal.run.equivalence import engine_view
    _memoize_geometry()
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


def runs(names=None):
    """Each run script ``selfcal_scripts/runs/<name>.py`` against the TOML config it replaces,
    ``selfcal_scripts/configs/<name>.toml``: the run engine must do the same with both."""
    import importlib

    from selfcal.run.config import load_config
    from selfcal.run.equivalence import differences
    from selfcal.run.lower import lower
    folder = os.path.join(REPO, 'selfcal_scripts', 'runs')
    names = names or sorted(f[:-3] for f in os.listdir(folder) if f.endswith('.py') and not f.startswith('_'))
    bad = 0
    _memoize_geometry()
    for name in names:
        toml = os.path.join(REPO, 'selfcal_scripts', 'configs', f'{name}.toml')
        if not os.path.exists(toml):
            print(f'NO TOML {name} (a run script of its own)')
            continue
        try:
            cfg = load_config(toml)
            module = importlib.import_module(f'selfcal_scripts.runs.{name}')
            if cfg.task == 'precompute':
                d = diff(_norm(dict(cfg.instrument_cfg, name=None)),
                         _norm({**module.PRECOMPUTE, 'lvf_output_dir': None, 'name': None}))
                d = [x for x in d if 'lvf_output_dir' not in x]
            elif cfg.task == 'reproject':
                d = differences(cfg, module.FIELD.reprojection_config(**module.REPROJECT))
            else:
                lowered = lower(module.FIELD, module.RECIPE, task='mosaic' if cfg.task == 'mosaic' else 'cal',
                                **module.RUN)
                d = (differences(cfg, lowered[0].cfg) if len(lowered) == 1
                     else [f'{len(lowered)} engine runs for one TOML run'])
        except Exception as e:
            d = [f'ERROR {type(e).__name__}: {str(e)[:200]}']
        print(('EQUAL   ' if not d else 'DIFFERS ') + name)
        for line in d[:12]:
            print('    ', line)
        bad += bool(d)
    print(f'ALL {len(names)} RUN SCRIPTS EQUAL THEIR CONFIGS' if not bad else f'{bad} of {len(names)} DIFFER')
    return bad


# The two producers of the stored views: what they return is the file format (see the module
# docstring), however the engine's input is built.
def _views_of_toml(path):
    """The engine view of a TOML run config (one engine run)."""
    from selfcal.run.config import load_config
    from selfcal.run.equivalence import engine_view
    return engine_view(load_config(path))


def _views_of_script(path):
    """The engine views of a run script of the repository: a list, one per engine run of its action,
    in order. The action is ``FIELD.calibrate(RECIPE, **RUN)``, as ``selfcal plan`` reads a script; a
    reproject script's is its reprojection, a precompute script's its settings; a campaign (FIELDS)
    gives ``{'<i> <field name>': that list}`` per field."""
    import importlib

    from selfcal.run.equivalence import engine_view
    from selfcal.run.lower import lower
    module = importlib.import_module(os.path.relpath(path, REPO)[:-len('.py')].replace(os.sep, '.'))
    if hasattr(module, 'PRECOMPUTE'):
        return [{'precompute': module.PRECOMPUTE}]
    if hasattr(module, 'REPROJECT'):
        return [engine_view(module.FIELD.reprojection_config(**module.REPROJECT))]
    # overwrite decides only whether existing products are made again
    run = {k: v for k, v in getattr(module, 'RUN', {}).items() if k != 'overwrite'}

    def action(field):
        return [engine_view(low.cfg) for low in lower(field, getattr(module, 'RECIPE', None), task='cal', **run)]
    if hasattr(module, 'FIELDS'):
        return {f'{i:02d} {field.name}': action(field) for i, field in enumerate(module.FIELDS)}
    return action(module.FIELD)


def views(out_dir, names=None):
    """Write the normalised engine views of every shipped TOML config and every run script (or of
    those whose file stem or view-file stem is in ``names``) to ``out_dir``, one JSON file each.
    Returns the number of views that could not be made (stored as ERROR)."""
    from selfcal.run.compute import pin_threads
    pin_threads()               # one BLAS thread, as the engine runs: the views must not depend on the shell
    _memoize_geometry()

    def name(p):
        return os.path.splitext(os.path.basename(p))[0]
    folder = os.path.join(REPO, 'selfcal_scripts', 'runs')
    scripts = sorted(os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.py') and not f.startswith('_'))
    items = ([('toml__' + _stem(p), _views_of_toml, p) for p in SHIPPED]
             + [('script__' + name(p), _views_of_script, p) for p in scripts])
    if names:
        for n in names:
            if not any(n in (stem, name(p)) for stem, _, p in items):
                print(f'NO MATCH {n}')
        items = [(stem, produce, p) for stem, produce, p in items if stem in names or name(p) in names]
    os.makedirs(out_dir, exist_ok=True)
    errors = 0
    for stem, produce, path in items:
        try:
            out = _norm(json.loads(json.dumps(produce(path), default=str)))
        except Exception as e:
            out = {'ERROR': f'{type(e).__name__}: {e}'}
            errors += 1
        with open(os.path.join(out_dir, stem + '.json'), 'w') as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write('\n')
        failed = isinstance(out, dict) and 'ERROR' in out
        print(f"{'ERROR' if failed else 'ok':<6}  {stem}" + (f"  {out['ERROR'][:160]}" if failed else ''))
    print(f'{len(items)} view files written to {out_dir}' + (f', {errors} of them ERROR' if errors else ''))
    return errors


def _typed(v):
    """Numbers tagged with their type, so that ``diff`` tells 0 from 0.0 and true from 1."""
    if isinstance(v, dict):
        return {k: _typed(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_typed(x) for x in v]
    return f'{type(v).__name__}:{v!r}' if isinstance(v, (bool, int, float)) else v


def compare_views(a, b):
    """The view files of two ``views`` directories, file by file: EQUAL (identical files), DIFFERS
    (with the first differing paths; values compared, then their types) or MISSING (in one
    directory only). Returns the number not equal."""
    names = sorted(n for n in set(os.listdir(a)) | set(os.listdir(b)) if n.endswith('.json'))
    if not names:
        print(f'no view files in {a} or {b}')
        return 1
    bad = 0
    for n in names:
        pa, pb = os.path.join(a, n), os.path.join(b, n)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f'MISSING {n}  (only in {a if os.path.exists(pa) else b})')
            bad += 1
            continue
        with open(pa) as fa, open(pb) as fb:
            ta, tb = fa.read(), fb.read()
        va, vb = json.loads(ta), json.loads(tb)
        d = [] if ta == tb else (diff(va, vb) or diff(_typed(va), _typed(vb)) or ['(formatting only)'])
        both_error = not d and isinstance(va, dict) and 'ERROR' in va
        print(('EQUAL   ' if not d else 'DIFFERS ') + n + ('  (ERROR in both)' if both_error else ''))
        for line in d[:12]:
            print('    ', line)
        bad += bool(d)
    print(f'ALL {len(names)} EQUAL' if not bad else f'{bad} of {len(names)} NOT EQUAL')
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
    elif cmd == 'runs':
        sys.exit(1 if runs(sys.argv[2:] or None) else 0)
    elif cmd == 'views':
        views(sys.argv[2], sys.argv[3:] or None)
    elif cmd == 'compare-views':
        sys.exit(1 if compare_views(sys.argv[2], sys.argv[3]) else 0)
    else:
        print(__doc__)
