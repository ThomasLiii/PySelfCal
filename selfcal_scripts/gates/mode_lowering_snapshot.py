"""Snapshot what every calibration MODE lowers to for a given config: the offset model's
setup kwargs, the sky model (component classes, names, keys, damp weights, profile fields),
the aux maps (hashes), the mosaic geometry (chunk-map hashes, renderer presence), x0 kind,
the N-pass hooks. Byte-strict comparison before/after a refactor of the modes/instrument.

Usage: mode_lowering_snapshot.py dump <out.pkl> <config.toml> [--mode NAME]
       mode_lowering_snapshot.py compare <a.pkl> <b.pkl>
Works with the S1 (dict geometry) and S2 (typed geometry) runner APIs."""
import hashlib
import pickle
import sys

import numpy as np

N_FRAMES = 137


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
    on a fixed probe of the instrument's real per-pixel maps (every 97th detector pixel)."""
    aux = getattr(geom, 'aux', None) or {}
    probe = {k: np.ascontiguousarray(np.asarray(v).ravel()[::97]) for k, v in aux.items()}
    out = []
    for c in model.components:
        coeff = c.coefficients(probe)
        reads = getattr(c, 'aux_requirements', None) or ()
        out.append({'name': c.name, 'damp_weight': getattr(c, 'damp_weight', None),
                    'reads': tuple(sorted(reads)),
                    'coeff': None if coeff is None else _h(np.asarray(coeff))})
    return out


def dump(out, config_path, mode_name=None):
    from selfcal import _state
    _state.set_progress(False)
    from selfcal_scripts.runner.config import load_config
    from selfcal_scripts.runner.modes import get_mode
    from selfcal_scripts.runner.engine import RunContext
    cfg = load_config(config_path)
    if mode_name:
        cfg.mode = mode_name
    ctx = RunContext.build(cfg)
    inst, mode = ctx.inst, ctx.mode
    job = ctx.jobs()[0]
    s2 = hasattr(ctx, 'geom')                     # S2: typed geometry; S1: dicts
    jg = ctx.job_geometry(job) if s2 else ctx.channel_inputs(job)
    geom = ctx.geom if s2 else ctx.det_inputs
    snap = {'mode': mode.name, 'mosaic_mode': mode.mosaic_mode,
            'requires': tuple('spectral_axis' if r == 'subchannel' else r for r in mode.requires)}
    import inspect
    frames = [f'/probe/exp_{i:04d}_det_{i % 16:02d}.h5' for i in range(N_FRAMES)]   # for detector-grouped terms
    if 'frames' in inspect.signature(mode.build_offset_model).parameters:
        om = mode.build_offset_model(cfg, inst, geom, jg, job, N_FRAMES, frames=frames)
    else:
        om = mode.build_offset_model(cfg, inst, geom, jg, job, N_FRAMES)
    snap['offset'] = _ser(om.to_setup_kwargs())
    snap['sky'] = _sky(mode.build_sky_model(cfg, inst, geom), geom)
    if s2:
        aux = mode.aux_maps(cfg, inst, geom)
        snap['aux'] = None if not aux else [_h(a) for a in aux.values()]
    else:
        aux = mode.det_aux(cfg, inst, geom)
        snap['aux'] = None if aux is None else [_h(a) for a in aux]
    cms, funcs = mode.mosaic_geometry(cfg, inst, geom, jg)
    snap['mosaic'] = ([_h(c) for c in cms], [f is not None for f in funcs])
    if hasattr(mode, 'x0_kind'):
        snap['x0'] = mode.x0_kind(cfg, inst, geom)
    else:
        import inspect
        snap['x0'] = 'from_Ab' if 'compute_x0_from_Ab' in inspect.getsource(mode.x0) else 'scalar_only'
    p = cfg.params
    if 'subch_poly_lo' in p or 'spectral_poly_lo' in p:
        try:
            snap['edges'] = _h(mode.clip_group_edges(cfg, inst, geom))
        except Exception as e:
            snap['edges'] = repr(e)[:80]
        try:
            snap['refit_basis'] = _ser(mode.refit_poly_basis(cfg, inst, geom, degree=2))
        except Exception as e:
            snap['refit_basis'] = repr(e)[:80]
    with open(out, 'wb') as f:
        pickle.dump(snap, f)
    print(f"snapshot {out}: mode={snap['mode']} offset keys={sorted(om.to_setup_kwargs())} sky={[c['name'] for c in snap['sky']]}")


def _eq(a, b, path, fails):
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b):
            fails.append(f"{path}: keys {sorted(set(a) ^ set(b))}"); return
        for k in a:
            _eq(a[k], b[k], f"{path}.{k}", fails)
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            fails.append(f"{path}: len {len(a)} != {len(b)}"); return
        for i, (x, y) in enumerate(zip(a, b)):
            _eq(x, y, f"{path}[{i}]", fails)
    else:
        try:
            same = (a == b) if not isinstance(a, float) else (a == b or (np.isnan(a) and np.isnan(b)))
        except Exception:
            same = False
        if not same:
            fails.append(f"{path}: {str(a)[:60]!s} != {str(b)[:60]!s}")


def compare(pa, pb):
    a, b = pickle.load(open(pa, 'rb')), pickle.load(open(pb, 'rb'))
    a.pop('mode', None); b.pop('mode', None)     # the registered name may differ (alias vs structural)
    for d in (a, b):                              # pre-S2 snapshots: qualname x0, 'subchannel' capability tag
        d['requires'] = tuple('spectral_axis' if r == 'subchannel' else r for r in d.get('requires', ()))
        if isinstance(d.get('x0'), str) and d['x0'].endswith('.x0'):
            d['x0'] = 'from_Ab' if d['x0'].startswith('K2') else 'scalar_only'
        off = d.get('offset')                     # pre-S6 snapshots: no per-map offset bases
        if isinstance(off, dict) and 'basis_list' not in off and 'chunk_maps' in off:
            off['basis_list'] = ('list', [None] * len(off['chunk_maps'][1]))
    fails = []
    _eq(a, b, '', fails)
    for f in fails[:20]:
        print('  ', f)
    print('LOWERING EQUAL' if not fails else f'LOWERING DIFFERS ({len(fails)})')
    return 0 if not fails else 1


if __name__ == '__main__':
    if sys.argv[1] == 'dump':
        mode = sys.argv[sys.argv.index('--mode') + 1] if '--mode' in sys.argv else None
        dump(sys.argv[2], sys.argv[3], mode)
    else:
        sys.exit(compare(sys.argv[2], sys.argv[3]))
