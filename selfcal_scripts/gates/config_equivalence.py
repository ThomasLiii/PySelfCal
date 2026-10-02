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

Usage:
  config_equivalence.py baseline <out_dir> [config.toml ...]   # default: every shipped config
  config_equivalence.py compare <dir_a> <dir_b>
"""
import glob
import hashlib
import inspect
import io
import contextlib
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
    from selfcal_scripts.runner import npass as rnp
    return {
        'calibration': _sig_defaults(Calibrator.setup_lsqr),
        # the engine passes use_float32=True and n_threads=apply_n_threads unless [lsqr] overrides them
        'lsqr': {**_sig_defaults(Calibrator.apply_lsqr), 'use_float32': True},
        'mosaic': _sig_defaults(Mosaicker.make_mosaic),
        # selfcal_scripts/runner/pipelines.py run_reprojection
        'reproject': dict(padding_pixels=100, max_workers=50, inner_parallel=1, reproj_func='exact',
                          padding_percentage=0.05, replace_existing=False, check=False,
                          header_filter_workers=16, source_ref_path=None),
        # engine.py tiling_frames / tile_assignment, pipelines.py run_tiled
        'tiling': dict(frame_glob='exp_*_det_00.h5', frame_filter='center', halo=0, rss_guardrail=True,
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
    from selfcal_scripts.runner.config import RunConfig
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
    from selfcal_scripts.gates import mode_lowering_snapshot as mls
    from selfcal_scripts.runner.engine import RunContext
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
    from selfcal_scripts.runner.config import load_config
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
            print(f'MISSING {n}'); bad += 1; continue
        ra, rb = json.load(open(pa)), json.load(open(pb))
        d = diff(ra['effective'], rb['effective']) + diff(ra.get('snapshot'), rb.get('snapshot'), 'snapshot')
        print(('EQUAL   ' if not d else 'DIFFERS ') + n)
        for line in d[:12]:
            print('    ', line)
        bad += bool(d)
    print('ALL EQUAL' if not bad else f'{bad} DIFFER')
    return bad


if __name__ == '__main__':
    sys.path.insert(0, REPO)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'baseline':
        baseline(sys.argv[2], sys.argv[3:] or SHIPPED)
    elif cmd == 'compare':
        sys.exit(1 if compare_dirs(sys.argv[2], sys.argv[3]) else 0)
    else:
        print(__doc__)
