"""The plan of an action: everything checked and resolved before any work starts.

``field.plan(recipe, jobs=...)`` (and every action, first) lowers the settings onto the engine
(:class:`~selfcal.run.runspec.RunSpec`), resolves each engine run (the instrument's geometry, the
model checked against it: data variables, chunk maps and axes, catalogue entries, prior terms),
finds the frames and the existing products, and checks that the worker processes can import
every function the run sends them. It takes seconds and computes nothing; printing the plan shows
what the action will do. The action then runs on the plan's resolved runs (:attr:`Plan.contexts`).
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
from .lower import as_recipe, lower
from .products import Book, check, expected_products, refusal, remedy

__all__ = ['Plan', 'make_plan']

#: The states of an existing product an action refuses (see :func:`selfcal.run.products.check`).
REFUSED = ('different', 'unrecorded', 'changed')


class Plan:
    """What an action will do (see the module docstring): the engine runs (:attr:`lowered`, each
    resolved in :attr:`contexts`), the jobs and their products, the frames, the pass schedule, and
    notes. ``str(plan)`` prints it."""

    def __init__(self, field, action, recipe, lowered, compute, *, tiles=None, passes=None, notes=()):
        self.field, self.action, self.recipe = field, action, recipe
        self.lowered = lowered          # selfcal.run.runspec.RunSpec, one per engine run
        self.compute = compute
        self.tiles, self.passes = tiles, passes
        self.notes = list(notes)
        self.contexts = []           # selfcal.run.engine.RunContext of each engine run
        self.products = []           # selfcal.run.products.Product, in the order the engine makes them
        self.book = None             # the products' inputs (selfcal.run.products.Book)
        self.frames = None
        self.frame_list = []         # the frame files the action uses
        self.start = None            # {job name: the cal its solve starts from} (calibrate(start=...))
        self.snapshots = None        # selfcal.run.schedule.Snapshots (calibrate(snapshots=...))
        self.monitor = None          # selfcal.run.schedule.Monitor (calibrate(monitor=...))

    @property
    def jobs(self):
        return tuple(j for spec in self.lowered for j in spec.jobs)

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
        for job, path in (self.start or {}).items():
            lines.append(f"  start       {job}: {path}")
        if self.snapshots is not None:
            sn = self.snapshots
            its = r.fit.iterations if r is not None else None
            dirs = sorted({os.path.dirname(p.path) for p in self.products if p.kind == 'cal'}) or \
                [os.path.join(f.path, 'calibration')]
            lines.append(f"  snapshots   every {sn.every} iterations, "
                         + ('all kept' if sn.keep is None else f'the last {sn.keep} kept')
                         + f", in {', '.join(os.path.join(d, 'snapshots') for d in dirs)}"
                         + (f" (none: the solve runs {its} iterations at most)" if its is not None and sn.every >= its
                            else ''))
        if r is not None and r.fit.stop is not None:
            lines.append(f"  stop        {describe_stop(r.fit.stop)}")
        if self.monitor is not None:
            lines.append(f"  monitor     {describe_monitor(self.monitor, r.fit.stop if r is not None else None)}")
        if self.tiles is not None:
            lines.append(f"  tiles       {self.tiles!r}")
        if self.passes is not None:
            from .schedule import schedule
            lines.append(f"  passes      {', '.join(t.upper() for t in schedule(self.passes.n, self.passes.order))}")
        for n in self.notes:
            lines.append('  note        ' + n.replace('\n', '\n' + ' ' * 14))
        return '\n'.join(lines)

    __repr__ = __str__


def describe_stop(stop) -> str:
    """A :class:`~selfcal.run.recipe.Stop` in words (the plan's ``stop`` line)."""
    parts = []
    if stop.residual is not None:
        r = stop.residual
        parts.append(f"|r| falls less than {r.below:g} (relative) over {r.window} iterations"
                     + (f" (the true residual, every {r.every})" if r.every else ''))
    if stop.gradient is not None:
        g = stop.gradient
        parts.append(f"the largest |A^T r| over {g.window} iterations at most {g.below:g} of its start"
                     + (f" (the true gradient, every {g.every})" if g.every else ''))
    if stop.large_scale is not None:
        ls = stop.large_scale
        parts.append(f"the sky terms' smooth fit (degree {ls.degree}) changes less than {ls.below:g} at two checks, "
                     f"every {ls.every}")
    if stop.lsqr_tests:
        parts.append(f"the solver's tests {', '.join(stop.tests)}")
    joined = (' and ' if stop.combine == 'all' else ' or ').join(parts)
    return joined + (f"; not before iteration {stop.min_iterations}" if stop.min_iterations else '')


def describe_monitor(monitor, stop=None) -> str:
    """A :class:`~selfcal.run.schedule.Monitor` in words (the plan's ``monitor`` line)."""
    from selfcal.core.monitor import resolve_large_scale
    what = [w for w, on in (('|b - A x|', monitor.residual), ('|A^T r|', monitor.gradient)) if on]
    degree, step = resolve_large_scale(stop, monitor)
    if degree is not None and monitor.large_scale is not False:
        what.append(f"the sky terms' smooth fit (degree {degree}, every {step}th pixel)")
    return f"every {monitor.every} iterations: {', '.join(what)}"


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
              compute=None, overwrite=False, check_workers=True, check_products=True, allow_no_frames=False,
              start=None, snapshots=None, monitor=None) -> Plan:
    """The :class:`Plan` of ``action`` (``"calibrate"`` or ``"mosaic"``) on ``field``; raises
    :class:`~selfcal.config.base.ConfigError` for anything that would fail later, including an
    existing product that was not made by the same inputs (see :mod:`selfcal.run.products`;
    ``overwrite`` marks those to be made again, ``check_products=False`` only lists them). ``start``:
    the cal each job's solve starts from (:func:`~selfcal.run.lower.start_paths`), checked against
    the frames and the model now (the whole system when the solve is set up:
    :mod:`selfcal.core.warm_start`). ``snapshots``: the solution every ``k`` iterations as a cal
    file (:class:`~selfcal.run.schedule.Snapshots`, or a number of iterations; plain calibrations
    only). ``monitor``: checks of every solve every ``m`` iterations
    (:class:`~selfcal.run.schedule.Monitor`, or a number of iterations; a calibration)."""
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
    if start is not None and action != 'calibrate':
        raise ConfigError(f"{action}(start=...): only a calibration starts from a cal")
    from .schedule import as_snapshots
    snapshots = as_snapshots(snapshots)
    if snapshots is not None and action != 'calibrate':
        raise ConfigError(f"{action}(snapshots=...): only a calibration's solve writes snapshots")
    from .schedule import as_monitor
    monitor = as_monitor(monitor)
    if monitor is not None:
        if action != 'calibrate':
            raise ConfigError(f"{action}(monitor=...): only a calibration's solve is monitored")
        from selfcal.core.monitor import resolve_large_scale
        try:
            resolve_large_scale(as_recipe(recipe).fit.stop, monitor)
        except ValueError as e:
            raise ConfigError(str(e)) from None
    lowered = lower(field, recipe, task='mosaic' if action == 'mosaic' else 'cal', jobs=jobs, tiles=tiles,
                    passes=passes, frames=frames, cal=cal, compute=compute, start=start, snapshots=snapshots,
                    monitor=monitor)
    plan = Plan(field, action, recipe, lowered, compute, tiles=tiles, passes=passes)
    if start is not None:
        plan.start = {k: v for spec in lowered for k, v in spec.start.items()}
    plan.snapshots = snapshots
    plan.monitor = monitor

    # frames
    first = lowered[0]
    where = (first.tiling.frames_dir if first.tiling is not None else
             first.frames.in_place or os.path.join(field.path, 'reprojected'))
    files = first.frames.files
    found = _frames_in(where) if files is None else [f for f in files if os.path.exists(os.path.join(where, os.path.basename(f)))]
    if files is not None and len(found) != len(files):
        raise ConfigError(f"frames=: {len(files) - len(found)} of the {len(files)} frames are not in {where}")
    if not found and action != 'mosaic':
        missing = f"no frames in {where} (reproject the exposures first: field.reproject(...))"
        if not allow_no_frames:                   # only field.plan() shows a plan without frames
            raise ConfigError(missing)
        plan.notes.append(f"{missing}; the model was checked against the instrument without them")
    n = len(found) if first.frames.first_n is None else min(first.frames.first_n, len(found))
    how = None
    if first.frames.in_place is None:
        how = f"staged ({first.frames.stage}) to {first.frames.stage_dir}" if tiles is None else 'staged per tile'
    plan.frames = (n, where, how)

    # the engine's own resolution: geometry (one for the action's runs), the model checked against the
    # instrument
    from .. import _state
    for spec in lowered:
        progress = _state.progress_enabled
        _state.set_progress(False)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                ctx = RunContext.build(spec, geom=plan.contexts[0].geom if plan.contexts else None)
        finally:
            _state.set_progress(progress)
        plan.contexts.append(ctx)
        groups = spec.setup.get('outlier_groups')
        if groups is not None:
            from .engine import clip_groups
            try:
                clip_groups(ctx, groups)
            except ValueError as e:
                raise ConfigError(f"Fit(clip=...): {e}") from None
        if passes is not None:
            _check_passes(ctx, recipe, passes)
        _check_instrument_maps(ctx, recipe, compute)
        _check_smoothing(ctx)
    used = found if first.frames.first_n is None else found[:n]
    plan.frame_list = list(used)
    plan.book = Book(field, recipe, passes=passes, tiles=tiles, start=plan.start)
    plan.products = expected_products(plan, plan.book, used)
    if plan.start is not None:
        _check_starts(plan)
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
    for spec in lowered:
        spec.reuse_mosaics = True                 # every existing mosaic is current, or is made again

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


def _check_starts(plan):
    """What can be checked of a warm start before the solve is set up: each start is a cal of the
    action's frames (in its order), sky terms on its reference grid, offset terms on their chunk
    maps and job, and none is a cal the action writes; the model's offsets can be read back (no
    polynomial basis)."""
    from ..core.warm_start import contents_problems
    from ..models.spec import chunk_map_of
    from .lower import reference_shape
    from .products import job_key
    model = plan.recipe.model
    if any(getattr(t, 'polynomial', None) is not None for t in model.offsets):
        raise ConfigError("calibrate(start=...): an offset term with a polynomial basis (Offsets(polynomial=...)) "
                          "cannot be continued: its cal holds the offsets the polynomial expands to, not the "
                          "polynomial's coefficients")
    made = {os.path.abspath(p.path) for p in plan.products if p.kind == 'cal'}
    frames = [os.path.basename(f) for f in plan.frame_list] or None
    try:
        ref_shape = reference_shape(plan.field)
    except ConfigError:
        ref_shape = None                     # no grid yet: the plan says so
    for spec, ctx in zip(plan.lowered, plan.contexts):
        maps = [chunk_map_of(term, ctx.geom).det for term in ctx.model.offset]
        for job in ctx.jobs():
            path = spec.start[job.name]
            if not os.path.isfile(path):
                raise ConfigError(f"calibrate(start=...): the cal {path} (job {job.name}) does not exist")
            if path in made:
                raise ConfigError(f"calibrate(start=...): {os.path.basename(path)} is a cal this calibration "
                                  f"writes; give the recipe its own name (recipe.replace(name=...)), so the "
                                  f"continued solve is a product of its own")
            problems = contents_problems(path, frames=frames, sky_names=[t.name for t in model.sky],
                                         n_maps=len(model.offsets), ref_shape=ref_shape, chunk_maps=maps,
                                         job=job_key(job))
            if problems:
                raise ConfigError(f"start={path} (job {job.name}) is not a solution of this calibration's system: "
                                  f"{'; '.join(problems)}")


def _check_smoothing(ctx):
    """An offset term smoothed (``smooth`` > 0) on a chunk map with no axis to smooth along would add
    no smoothness rows at all: say so instead."""
    from ..models.spec import DETECTOR_MAP
    for term in ctx.model.offset:
        if not term.reg_weight or term.adjacency is not None:
            continue
        if term.map == DETECTOR_MAP:
            continue                               # one chunk: nothing to smooth
        cm = ctx.geom.chunk_map if term.map is None else ctx.geom.chunk_maps.get(term.map)
        if cm is not None and not tuple(cm.adjacency_axes or ()):
            raise ConfigError(f"Offsets({term.name or ''!r}, smooth={term.reg_weight}): the chunk map "
                              f"{cm.name!r} declares no axes to smooth along; give smooth_along=(...) (its axes: "
                              f"{list(cm.axes.names) if cm.axes is not None else []}), or the map adjacency_axes")


def _check_instrument_maps(ctx, recipe, compute):
    """An instrument's per-pixel maps (SPHEREx: the wavelength maps) are coadded against the std map,
    in the sigma-clip pass or over the frame cache; say so now rather than after the solve."""
    c = recipe.coadd
    if c is None or not c.instrument_maps or ctx.inst.aux_coadds(ctx.geom) is None:
        return
    if not c.std:
        raise ConfigError(f"Coadd(instrument_maps=True): {type(ctx.inst).__name__}'s instrument maps are coadded against the std "
                          f"map; give Coadd(std=True), or instrument_maps=False")
    if c.clip is None and not compute.cache_frames:
        raise ConfigError("Coadd(instrument_maps=True): the instrument maps are coadded in the sigma-clip pass or over "
                          "the frame cache; give Coadd(clip=...) or Compute(cache_frames=True), or instrument_maps=False")


def _check_passes(ctx, recipe, passes):
    cm = ctx.geom.chunk_map
    jobs = ctx.jobs()
    if len(jobs) != 1:
        raise ConfigError(f"calibrate(passes=...): the N-pass solve runs one job; got {[j.name for j in jobs]}")
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
