"""What the run engine does with an engine run, independent of how it was spelled.

:func:`engine_view` describes an engine run (:class:`~selfcal.run.runspec.RunSpec`) by what the
engine does with it: the library calls' keywords with their defaults filled, each sky term's
resolved damping, the offset rows, the sky coefficients evaluated on the instrument's real maps,
the jobs and every product path, the frames (by name), staging, tiles and the N-pass schedule.
``selfcal_scripts/gates/config_equivalence.py views`` records it for every run script, so that a
change to the engine can be shown to leave what it does unchanged.
"""
import contextlib
import glob
import hashlib
import inspect
import io
import os

import numpy as np

__all__ = ['engine_view', 'diff']

N_FRAMES = 137          # frames of the probe frame list the offset model is lowered over


def _h(a):
    a = np.ascontiguousarray(a)
    return (a.shape, str(a.dtype), hashlib.sha1(a.tobytes()).hexdigest())


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


# Options the library's calls no longer have, at the values every run used then and uses now (the
# views recorded before they were removed name them).
_FIXED_SETUP = dict(compact_zero_columns=True, damp_offset=0.0, outlier_subchannel_edges=None, outlier_aux_key=None,
                    spectral_fit=False, line_center=None, line_sigma=None)
_FIXED_SOLVER = dict(resume=False, keep_state=False)
_FIXED_COADD = dict(preprocess_func=None)


def engine_view(spec):
    """What the run engine does with ``spec`` (a :class:`~selfcal.run.runspec.RunSpec`): the
    library calls' keywords (defaults filled), each sky term's resolved damping, the offset rows,
    the sky coefficients evaluated on the real geometry, the jobs and every product path, the
    frames (by name), staging, tiles and the N-pass schedule."""
    from .. import _state
    from ..pipeline.pipeline_wrapper import Calibrator, Mosaicker
    from . import engine as E
    from .schedule import schedule
    _state.set_progress(False)
    view = {'task': spec.task}
    if spec.task == 'reproject':
        r, layout = spec.reproject, spec.instrument.layout()
        files = sorted(sum((glob.glob(p) for p in r.exposures), []))
        view['exposures'] = _frames_sha(files) if files else ['none found', list(r.exposures)]
        view['layout'] = _canon({'use_ext': list(layout.ref_use_ext), 'sci_ext': list(layout.sci_ext),
                                 'dq_ext': layout.dq_ext, 'detector_ids': layout.detector_ids,
                                 'reader': layout.reader, 'header_keys': layout.header_keys,
                                 'cache_tag': layout.cache_tag})
        view['reproject'] = _canon({'padding_pixels': r.padding, 'max_workers': r.workers, 'reproj_func': r.method,
                                    'padding_percentage': r.padding_fraction, 'replace_existing': r.replace,
                                    'check': r.verify, 'source_ref_path': r.reference,
                                    'output': os.path.join(spec.output_dir, spec.run_name),
                                    'resolution_arcsec': spec.resolution_arcsec})
        return view
    with contextlib.redirect_stdout(io.StringIO()):
        ctx = E.RunContext.build(spec)
        geom, model = ctx.geom, ctx.model
        jobs = ctx.jobs()
        sky_model = ctx.sky_model()
        job0 = jobs[0]
        jg = ctx.job_geometry(job0)
        frames = [f'/probe/exp_{i:04d}_det_{i % 16:02d}.h5' for i in range(N_FRAMES)]
        om = ctx.offset_model(frames)
    view['jobs'] = _canon([(j.name, j.kind, j.value) for j in jobs])
    view['frame_tag'] = ctx.frame_tag
    view['unit'] = ctx.unit
    view['run'] = [spec.output_dir and os.path.join(spec.output_dir, spec.run_name), spec.resolution_arcsec]

    # setup_lsqr: the keywords solve_job passes (models compared below), defaults filled
    calk = dict(spec.setup)
    calk.update(model.setup_kwargs())
    groups = calk.pop('outlier_groups', None)
    if groups is not None:
        calk.update(E.clip_groups(ctx, groups))
    if not calk.get('outlier_group_variable'):
        calk['outlier_group_variable'] = geom.wavelength_key
    if spec.pre_cal is not None:
        calk['preprocess_func'] = spec.pre_cal
    if spec.post_cal is not None:
        calk['postprocess_func'] = spec.post_cal
    setup = {**_sig_defaults(Calibrator.setup_lsqr), **_FIXED_SETUP, 'ignore_list': [], **calk}
    if setup['ignore_list'] is None:
        setup['ignore_list'] = []
    regularize, weighted = bool(setup.pop('offset_regularization')), bool(setup.pop('weighted_damping'))
    for k in ('damp_weight', 'damp_weight_line', 'max_workers', 'batch_spill_dir'):
        setup.pop(k, None)
    view['setup_lsqr'] = _canon(setup)
    view['workers'] = calk.get('max_workers', 20)
    view['sky_damping'] = list(ctx.sky_damping) if weighted else [0.0] * len(sky_model.components)
    sky = _sky(sky_model, geom)
    for t in sky:
        t.pop('damp_weight', None)
    view['sky'] = _canon(sky)
    view['offsets'] = _canon(_built_offset_rows(om.to_setup_kwargs(), regularize))
    view['x0'] = model.x0_kind
    det_aux, _ = ctx.aux_maps()
    view['aux'] = None if det_aux is None else _canon(det_aux)
    view['line_fisher_threshold'] = spec.line_fisher_threshold if model.has_coefficients else None
    view['apply_lsqr'] = _canon({**_sig_defaults(Calibrator.apply_lsqr), **_FIXED_SOLVER, **spec.lsqr})

    # the mosaic, when one is made
    makes_mosaic = spec.task == 'mosaic' or (spec.task == 'cal' and spec.tiling is None and spec.make_mosaic)
    if makes_mosaic:
        mos = {**_sig_defaults(Mosaicker.make_mosaic), **_FIXED_COADD, 'ignore_list': [], **spec.mosaic}
        if mos['ignore_list'] is None:
            mos['ignore_list'] = []
        mos['postprocess_func'] = spec.post_mosaic
        mos['oversample'] = spec.oversample
        mos['instrument_maps'] = spec.instrument_maps and ctx.inst.aux_coadds(geom) is not None
        # sigma is read by the sigma-clip pass (it runs with the std map only) and by the instrument's
        # aux coadds (engine.mosaic_job, core/coadd.run_coadd_schedule); elsewhere it is unused, and
        # is described as make_mosaic's default whatever the run says
        if not ((mos['make_std_map'] and mos['apply_sigma_clipping']) or mos['instrument_maps']):
            mos['sigma'] = _sig_defaults(Mosaicker.make_mosaic)['sigma']
        mos['coadd_workers'] = mos.pop('max_workers')
        with contextlib.redirect_stdout(io.StringIO()):
            cms, funcs = ctx.mosaic_geometry(jg)
        mos['geometry'] = [_canon(c) for c in cms]
        mos['renderers'] = [_canon(f) for f in funcs]
        view['mosaic'] = _canon(mos)
    else:
        view['mosaic'] = None

    # products, frames, staging
    view['products'] = {j.name: {'cal': ctx.cal_path(j), 'mosaic': ctx.mosaic_path(j) if makes_mosaic else None,
                                 'npass_stem': ctx.pass_stem(j) if spec.task == 'npass' else None}
                        for j in jobs}
    reproj = ctx.pipeline_config.reproj_dir
    if spec.tiling is not None:
        t = spec.tiling
        tiles, only = E.resolve_tiles(t)
        view['tiling'] = _canon({'tiles': [(x.name, x.bbox) for x in tiles], 'only': only, 'ref_shape': t.ref_shape,
                                 'frame_filter': t.assign, 'halo': t.halo, 'line': t.stitch_line,
                                 'stage_dir': t.stage_dir, 'memory_guard': t.memory_guard,
                                 'tile_cals': {x.name: ctx.tile_cal_file(job0, x) for x in tiles},
                                 'stitched': ctx.stitched_cal_path(job0)})
        files = E.tiling_frames(t.frames_dir) if os.path.isdir(t.frames_dir) else None
        view['frames'] = ['tiled', t.frames_dir, _frames_sha(files) if files else 'not on this machine']
    else:
        source = spec.frames
        where = source.in_place or reproj
        if source.files:
            names = [os.path.basename(f) for f in source.files]
        elif os.path.isdir(where):
            names = [os.path.basename(f) for f in E.frame_list(where, source.first_n)]
        else:
            names = None
        view['frames'] = [where, _frames_sha(names) if names else 'not on this machine']
        view['staging'] = None if source.in_place else _canon({'how': source.stage, 'dir': source.stage_dir,
                                                                'keep': source.keep, 'io_limit': source.io_limit})
    view['cache_dir'] = spec.scratch if (makes_mosaic and spec.mosaic.get('cache_intermediate')) \
        or spec.tiling is not None or spec.task == 'npass' else 'unused'
    view['cal_override'] = spec.cal_override

    # the N-pass schedule
    if spec.task == 'npass':
        p = spec.passes
        init = p.init_clip
        init_clip = {'outlier_thresh': init.sigma if init is not None else calk.get('outlier_thresh'),
                     'grouped': init.grouped if init is not None else calk.get('outlier_group_edges') is not None,
                     'ignore_list': (list(init.ignore_flags) if init is not None and init.ignore_flags is not None
                                     else calk.get('ignore_list') or [])}
        view['passes'] = _canon({'schedule': schedule(p.n, p.order), 'stop_tol': p.stop_tol, 'sky_merge': p.sky_merge,
                                 'keep_moments': p.keep_moments, 'init': init_clip,
                                 'sky': {'outlier_thresh': p.sky_clip.sigma, 'subch_clip': p.sky_clip.grouped},
                                 'offset': {'poly_degree': p.refit_degree, 'outlier_thresh': p.refit_clip.sigma,
                                            'subch_clip': p.refit_clip.grouped, 'bright_cut': p.bright_cut,
                                            'min_pix': p.min_pixels, 'segments': p.segments, 'ridge': p.ridge}})
        with contextlib.redirect_stdout(io.StringIO()):
            view['npass_edges'] = _canon(ctx.clip_group_edges())
            view['npass_refit_basis'] = _canon(ctx.refit_poly_basis(p.refit_degree, segments=p.segments))
    return view
