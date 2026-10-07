"""The plan of an action: everything checked and resolved before any work starts.

``field.plan(recipe, jobs=...)`` (and every action, first) lowers the settings onto the engine,
builds the instrument's geometry, checks the model against it (data variables, chunk maps and
axes, catalogue entries, prior terms), finds the frames and the existing products, and checks
that the worker processes can import every function the run sends them. It takes seconds and
computes nothing; printing the plan shows what the action will do.
"""
from __future__ import annotations

import ast
import contextlib
import glob
import io
import multiprocessing
import os
import pickle
import sys

from ..config.base import Config, ConfigError
from ..config.functions import _is_main_guard, function_ref
from .lower import Lowered, as_recipe, lower
from .products import Book, check, expected_products, refusal, remedy

__all__ = ['Plan', 'make_plan']

#: The states of an existing product an action refuses (see :func:`selfcal.run.products.check`).
REFUSED = ('different', 'unrecorded', 'changed')


class Plan:
    """What an action will do (see the module docstring): the engine runs (:attr:`lowered`), the
    jobs and their products, the frames, the pass schedule, and notes. ``str(plan)`` prints it."""

    def __init__(self, field, action, recipe, lowered, compute, *, tiles=None, passes=None, notes=()):
        self.field, self.action, self.recipe = field, action, recipe
        self.lowered: list[Lowered] = lowered
        self.compute = compute
        self.tiles, self.passes = tiles, passes
        self.notes = list(notes)
        self.contexts = []
        self.products = []           # selfcal.run.products.Product, in the order the engine makes them
        self.book = None             # the products' inputs (selfcal.run.products.Book)
        self.frames = None
        self.frame_list = []         # the frame files the action uses

    @property
    def jobs(self):
        return tuple(j for low in self.lowered for j in low.jobs)

    @property
    def refused(self):
        """The existing products the action would refuse: made by other inputs (``"different"``),
        without a record (``"unrecorded"``: adopt them, :meth:`~selfcal.run.field.Field.adopt`), or
        changed since they were recorded (``"changed"``)."""
        return [p for p in self.products if p.state in REFUSED]

    def __str__(self):
        f, r = self.field, self.recipe
        lines = [f"Plan: {self.action} {f.name}  ({f.path})",
                 f"  instrument  {f.instrument!r}"]
        if r is not None:
            lines.append(f"  recipe      {r!r}")
            if r.model is not None:
                for t, d in zip(r.model.sky, r.model.sky_dampings()):
                    lines.append(f"  sky term    {t.name}: damping {d:g}" + (f", times {t.times!r}" if t.times else ''))
                for t in r.model.offsets:
                    lines.append(f"  offsets     {t!r}")
        if self.frames is not None:
            n, where, how = self.frames
            lines.append(f"  frames      {n} in {where}{' (' + how + ')' if how else ''}")
        shown = {'missing': 'made', 'current': 'reused', 'replace': 'made again (overwrite)',
                 'unrecorded': 'refused: exists, unrecorded', 'different': 'refused: exists, other inputs',
                 'changed': 'refused: changed since recorded'}
        for prod in self.products:
            label = f"{prod.kind} {prod.job.name}" + (f" [{prod.tile}]" if prod.tile else '') + \
                (f" pass {prod.index}" if prod.index else '')
            lines.append(f"  {label:<22} {os.path.basename(prod.path)}  ({shown.get(prod.state, prod.state)})")
        if self.tiles is not None:
            lines.append(f"  tiles       {self.tiles!r}")
        if self.passes is not None:
            from .schedule import schedule
            lines.append(f"  passes      {', '.join(t.upper() for t in schedule(self.passes.n, self.passes.order))}")
        for n in self.notes:
            lines.append('  note        ' + n.replace('\n', '\n' + ' ' * 14))
        return '\n'.join(lines)

    __repr__ = __str__


# --------------------------------------------------------------------------- worker checks
def _function_refs(obj, out):
    """The import paths of every function in a settings object (recursively)."""
    import dataclasses
    if isinstance(obj, Config):
        for f in dataclasses.fields(obj):
            _function_refs(getattr(obj, f.name), out)
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            _function_refs(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            _function_refs(x, out)
    elif callable(obj) and not isinstance(obj, type) and hasattr(obj, '__code__'):
        out.add(function_ref(obj))
    return out


def _by_values(obj, out):
    """The functions sent by value in a settings object (recursively)."""
    import dataclasses

    from ..config.functions import by_value
    if isinstance(obj, by_value):
        out.append(obj)
    elif isinstance(obj, Config):
        for f in dataclasses.fields(obj):
            _by_values(getattr(obj, f.name), out)
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            _by_values(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            _by_values(x, out)
    return out


def _probe(refs, blobs):
    """Runs in a fresh worker process: import every function, unpickle every object."""
    from selfcal.config.functions import load_callable
    errors = []
    for ref in refs:
        try:
            load_callable(ref)
        except Exception as e:
            errors.append(f"the function {ref}: {type(e).__name__}: {e}")
    for name, blob in blobs:
        try:
            pickle.loads(blob)
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
    return errors


def check_in_worker(refs, objects):
    """Import ``refs`` and unpickle ``objects`` (``{name: object}``) in one fresh worker process,
    as the run's workers will; :class:`~selfcal.config.base.ConfigError` listing what fails."""
    if not refs and not objects:
        return
    from concurrent.futures import ProcessPoolExecutor
    blobs = [(name, pickle.dumps(o)) for name, o in objects.items()]
    try:
        # A fresh interpreter importing the script and the functions by name, as the pipeline's
        # forkserver workers do; 'spawn', so the shared forkserver starts later, inside the action
        # (its workers then write to the action's log).
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context('spawn')) as ex:
            errors = ex.submit(_probe, sorted(refs), blobs).result(timeout=300)
    except Exception as e:
        raise ConfigError(f"a worker process could not start ({type(e).__name__}: {e}). The run script is "
                          f"imported by every worker: its top level must only declare (functions, settings); "
                          f"put the run under `if __name__ == \"__main__\":`") from None
    if errors:
        raise ConfigError("the worker processes cannot use: " + '; '.join(errors))


def check_main_guard():
    """Refuse an action (or a plan) started from the top level of a run script outside its
    ``__main__`` guard (every worker would import the script and start it again), and one started
    while a worker process imports the script."""
    if multiprocessing.parent_process() is not None:
        main = sys.modules.get('__mp_main__') or sys.modules.get('__main__')
        raise ConfigError(f"an action was started while a worker process imported the run script "
                          f"{getattr(main, '__file__', '')}; put the script's work under "
                          f"`if __name__ == \"__main__\":`")
    main = sys.modules.get('__main__')
    path = getattr(main, '__file__', None)
    if not path:
        return                                           # an interactive session or a notebook
    frame = sys._getframe(1)
    while frame is not None and (frame.f_globals.get('__name__') != '__main__'
                                 or frame.f_globals.get('__file__') != path):
        frame = frame.f_back
    if frame is None or frame.f_code.co_name != '<module>':
        return                                           # called from a function: the caller guards it
    try:
        with open(path) as f:
            tree = ast.parse(f.read(), filename=path)
    except (OSError, SyntaxError):
        return
    line = frame.f_lineno
    for node in tree.body:
        if isinstance(node, ast.If) and _is_main_guard(node.test) and node.lineno <= line <= node.end_lineno:
            return
    raise ConfigError(f"{os.path.basename(path)}, line {line}: start the run under `if __name__ == \"__main__\":`. "
                      f"Every worker process imports the script, and would start the run again")


# --------------------------------------------------------------------------- the plan
def _frames_in(path):
    return sorted(glob.glob(os.path.join(path, '*.h5')))


def make_plan(field, action, recipe=None, *, jobs=None, tiles=None, passes=None, frames=None, cal=None,
              compute=None, overwrite=False, check_workers=True, check_products=True, allow_no_frames=False) -> Plan:
    """The :class:`Plan` of ``action`` (``"calibrate"`` or ``"mosaic"``) on ``field``; raises
    :class:`~selfcal.config.base.ConfigError` for anything that would fail later, including an
    existing product that was not made by the same inputs (see :mod:`selfcal.run.products`;
    ``overwrite`` marks those to be made again, ``check_products=False`` only lists them)."""
    from .compute import pin_threads
    from .engine import RunContext
    check_main_guard()
    pin_threads()                  # before the first worker (the worker check below) starts the forkserver
    recipe = as_recipe(recipe)
    compute = compute or field.compute
    if action == 'mosaic':
        if recipe.coadd is None:
            raise ConfigError("mosaic: the recipe makes no mosaic (coadd=None); give it a Coadd")
        if tiles is not None or passes is not None:
            raise ConfigError("mosaic: a mosaic coadds each job's cal; it takes no tiles or passes")
    elif (tiles is not None or passes is not None) and recipe.coadd is not None:
        raise ConfigError("calibrate(tiles=..., passes=...): a tiled or N-pass calibration makes no mosaic; "
                          "give a recipe without one: recipe.replace(coadd=None)")
    lowered = lower(field, recipe, task='mosaic' if action == 'mosaic' else 'cal', jobs=jobs, tiles=tiles,
                    passes=passes, frames=frames, cal=cal, compute=compute)
    plan = Plan(field, action, recipe, lowered, compute, tiles=tiles, passes=passes)

    # frames
    first = lowered[0].cfg
    where = (first.tiling['full_reproj_dir'] if first.tiling else
             first.reproj_override or os.path.join(field.path, 'reprojected'))
    found = _frames_in(where) if first.frame_files is None else [
        f for f in first.frame_files if os.path.exists(os.path.join(where, os.path.basename(f)))]
    if first.frame_files is not None and len(found) != len(first.frame_files):
        raise ConfigError(f"frames=: {len(first.frame_files) - len(found)} of the {len(first.frame_files)} frames "
                          f"are not in {where}")
    if not found and action != 'mosaic':
        missing = f"no frames in {where} (reproject the exposures first: field.reproject(...))"
        if not allow_no_frames:                   # only field.plan() shows a plan without frames
            raise ConfigError(missing)
        plan.notes.append(f"{missing}; the model was checked against the instrument without them")
    n = len(found) if first.n_frames is None else min(first.n_frames, len(found))
    how = None
    if first.reproj_override is None:
        how = f"staged ({first.staging}) to " + (first.stage_dir or os.path.join(
            first.cache_dir, f'reproj_nvme_{field.name}')) if tiles is None else 'staged per tile'
    plan.frames = (n, where, how)

    # the engine's own resolution: geometry, the model checked against the instrument
    from .. import _state
    for low in lowered:
        cfg = low.cfg
        progress = _state.progress_enabled
        _state.set_progress(False)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ctx = RunContext.build(cfg)
        finally:
            _state.set_progress(progress)
        plan.contexts.append(ctx)
        groups = cfg.calibration.get('outlier_groups')
        if groups is not None:
            from .engine import clip_groups
            try:
                clip_groups(ctx, groups)
            except ValueError as e:
                raise ConfigError(f"Fit(clip=...): {e}") from None
        if passes is not None:
            _check_passes(ctx, recipe, passes, low)
        _check_instrument_maps(ctx, recipe, compute)
        _check_smoothing(ctx, cfg)
    used = found if first.n_frames is None else found[:n]
    plan.frame_list = list(used)
    plan.book = Book(field, recipe, passes=passes, tiles=tiles)
    plan.products = expected_products(plan, plan.book, used)
    refusals = []
    by_path = {prod.path: prod for prod in plan.products}
    for prod in plan.products:                  # engine order: what a product is made from comes first
        input_cal = action == 'mosaic' and prod.kind == 'cal'
        if not os.path.exists(prod.path):
            if input_cal:
                raise ConfigError(f"mosaic: {os.path.basename(prod.path)} does not exist (calibrate first)")
            continue
        made_again = [d for d in prod.depends if d in by_path and by_path[d].state in ('missing', 'replace')]
        if made_again and not input_cal:          # made again from what is made again
            prod.state = 'replace'
            prod.details = [f"made from {os.path.basename(made_again[0])}, which is made again"]
            continue
        prod.state, prod.details = check(prod.path, prod.inputs())
        if overwrite and not input_cal:
            prod.state = 'replace'                # made again, whatever made it
            continue
        if prod.state == 'current' or not check_products:
            continue
        refusals.append(refusal(prod.path, prod.state, prod.details, input_cal=input_cal, tile=prod.tile is not None))
    if refusals:
        raise refusals[0] if len(refusals) == 1 else ConfigError(_summary(plan.products, action=action))
    if plan.refused:
        plan.notes.append(_summary(plan.refused, would=True, action=action))
    for low in lowered:
        low.cfg.reuse_mosaics = True          # every existing mosaic is current, or is made again

    if check_workers:
        refs = _function_refs(recipe, set())
        objects = {k: v for k, v in (('Fit(frame_hook)', recipe.fit.frame_hook),
                                      ('Fit(raw_frame_hook)', recipe.fit.raw_frame_hook),
                                      ('Coadd(frame_hook)', recipe.coadd.frame_hook if recipe.coadd else None))
                   if v is not None}
        for fn in _by_values(recipe, []):
            objects[f'sc.by_value({fn.__qualname__})'] = fn
        check_in_worker(refs, objects)
    return plan


def _summary(products, would=False, action='calibrate'):
    """One message for many products an action refuses, by why, with what to do about each kind."""
    groups = {}
    for prod in products:
        if prod.state in REFUSED:
            input_cal = action == 'mosaic' and prod.kind == 'cal'
            groups.setdefault((prod.state, input_cal, prod.tile is not None), []).append(prod)
    why = {'unrecorded': 'have no record of how they were made (made before records existed, by a TOML run, or '
                         'interrupted)',
           'different': 'were made with different inputs',
           'changed': 'were written again after they were recorded'}
    lines = []
    for (state, input_cal, tile), prods in groups.items():
        names = ', '.join(os.path.basename(p.path) for p in prods[:3]) + (f' and {len(prods) - 3} more'
                                                                              if len(prods) > 3 else '')
        first = f" (the first: {'; '.join(prods[0].details[:3])})" if prods[0].details else ''
        lines.append(f"{len(prods)} {why[state]}{first}: {names}. To go on: "
                     + remedy(state, input_cal=input_cal, tile=tile))
    head = 'the run would refuse existing products' if would else 'existing products refused'
    return f"{head}:\n  " + '\n  '.join(lines)


def _check_smoothing(ctx, cfg):
    """An offset term smoothed (``smooth`` > 0) on a chunk map with no axis to smooth along would add
    no smoothness rows at all: say so instead."""
    from ..models.spec import DETECTOR_MAP
    for term in (cfg.model or {}).get('offset', ()):
        if not term.get('reg_weight') or term.get('adjacency') is not None:
            continue
        name = term.get('map')
        if name == DETECTOR_MAP:
            continue                               # one chunk: nothing to smooth
        cm = ctx.geom.chunk_map if name is None else ctx.geom.chunk_maps.get(name)
        if cm is not None and not tuple(cm.adjacency_axes or ()):
            raise ConfigError(f"Offsets({term.get('name') or ''!r}, smooth={term['reg_weight']}): the chunk map "
                              f"{cm.name!r} declares no axes to smooth along; give smooth_along=(...) (its axes: "
                              f"{list(cm.axes.names) if cm.axes is not None else []}), or the map adjacency_axes")


def _check_instrument_maps(ctx, recipe, compute):
    """An instrument's per-pixel maps (SPHEREx: the wavelength maps) are coadded against the std map,
    in the sigma-clip pass or over the frame cache; say so now rather than after the solve."""
    c = recipe.coadd
    if c is None or not c.instrument_maps or ctx.inst.aux_coadds(ctx.geom) is None:
        return
    if not c.std:
        raise ConfigError(f"Coadd(instrument_maps=True): {ctx.inst.name}'s instrument maps are coadded against the std "
                          f"map; give Coadd(std=True), or instrument_maps=False")
    if c.clip is None and not compute.cache_frames:
        raise ConfigError("Coadd(instrument_maps=True): the instrument maps are coadded in the sigma-clip pass or over "
                          "the frame cache; give Coadd(clip=...) or Compute(cache_frames=True), or instrument_maps=False")


def _check_passes(ctx, recipe, passes, low):
    cm = ctx.geom.chunk_map
    if len(low.jobs) != 1:
        raise ConfigError(f"calibrate(passes=...): the N-pass solve runs one job; got {[j.name for j in low.jobs]}")
    if recipe.model.spectral_window() is None:
        raise ConfigError("calibrate(passes=...): the OFFSET pass refits a polynomial along the spectral axis over "
                          "the model's window: give the model a polynomial with a window (Offsets(polynomial="
                          "sc.Poly(2, window=range(lo, hi + 1))))")
    if any(t.times is not None or t.basis is not None for t in recipe.model.offsets):
        raise ConfigError("calibrate(passes=...): the OFFSET pass refits a polynomial per frame; offset terms with "
                          "times= or basis= run without passes")
    for name, clip in (('init_clip', passes.init_clip), ('sky_clip', passes.sky_clip),
                       ('offset.clip', passes.offset.clip)):
        per = getattr(clip, 'per', 'frame')
        axis = getattr(per, 'axis', None)
        if axis is not None and axis != cm.spectral_axis:
            raise ConfigError(f"Passes({name}=...): the N-pass clips group along the primary chunk map's spectral "
                              f"axis {cm.spectral_axis!r}, not {axis!r}")
