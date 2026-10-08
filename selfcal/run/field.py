"""The field: one data set, its folder, and the actions that run on it.

::

    field = sc.Field("outputs/nep_d3", sc.SPHEREx(3), pixel_scale=6.2, compute=ORCA)
    if __name__ == "__main__":
        field.reproject("exposures/*.fits")
        print(field.plan(RECIPE, jobs=spherex.channel(17)))
        result = field.calibrate(RECIPE, jobs=spherex.channels(1, 34))

The folder ``path`` holds ``ref.fits`` (the reference grid), ``reprojected/`` (the frames),
``calibration/`` and ``mosaic/`` (the products), ``logs/`` and ``records/`` (one log and one
JSON record per action). Every action plans first (:meth:`Field.plan`: checks, seconds),
pins the BLAS / OpenMP threads, then runs on the run engine (:mod:`selfcal.run`).
"""
from __future__ import annotations

import contextlib
import glob
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import KW_ONLY, dataclass, is_dataclass
from dataclasses import field as dc_field

from ..config.base import Config, ConfigError
from ..instruments.contract import Instrument
from .compute import Compute, Tuning, environment, pin_threads
from .lower import as_recipe, lower
from .plan import check_main_guard, make_plan
from .products import (
    clear_stale_intermediates,
    fingerprint,
    frames_digest,
    remove_product,
    verify,
    write_sidecar,
)
from .records import Record
from .result import Result

__all__ = ['Field', 'frames_in', 'Submitted']


@dataclass
class Submitted:
    """A calibration started in the background by :meth:`Field.submit`: its ``request`` (the JSON
    the detached process reruns), the ``console`` file of its output and its process id ``pid``.
    The run writes its own record and log under ``records/`` and ``logs/``."""
    request: str
    console: str
    pid: int
    process: object = dc_field(default=None, repr=False, compare=False)

    def running(self) -> bool:
        """Whether the process is still running."""
        if self.process is not None:
            return self.process.poll() is None
        try:
            with open(f'/proc/{self.pid}/stat') as f:
                return f.read().rsplit(')', 1)[1].split()[0] != 'Z'     # a zombie has finished
        except OSError:
            return False

    def wait(self, timeout=None) -> int:
        """Wait for the run to finish; its exit code."""
        if self.process is None:
            raise RuntimeError("wait(): only in the process that submitted the run")
        return self.process.wait(timeout)


def frames_in(directory) -> list[str]:
    """The frame files (``*.h5``) of ``directory``, sorted: ``frames=sc.frames_in(d)[:300]``."""
    return sorted(glob.glob(os.path.join(os.fspath(directory), '*.h5')))


@contextlib.contextmanager
def _console_logging():
    """Show the library's progress messages on stdout during an action, unless the application
    has configured logging itself."""
    lib = logging.getLogger('selfcal')
    added = None
    if not logging.getLogger().handlers and not any(not isinstance(h, logging.NullHandler) for h in lib.handlers):
        added = logging.StreamHandler(sys.stdout)
        added.setFormatter(logging.Formatter('%(message)s'))
        lib.addHandler(added)
        old_level = lib.level
        lib.setLevel(logging.INFO)
    try:
        yield
    finally:
        if added is not None:
            lib.removeHandler(added)
            lib.setLevel(old_level)


def process_log(field):
    """The run log of this process, started by its first action (scripts only; None in an
    interactive session or a notebook): ``<field>/logs/<script>_<time>_<pid>.log``, holding the
    script's text and everything this process and its workers print. Every later action of the
    process appends to it (a worker pool, once started, keeps writing where it started), after a
    line naming the action and its record."""
    import datetime

    from . import runlog
    from .records import script_path
    if runlog._active is not None:
        return runlog._active
    script = script_path()
    if not script:
        return None
    stem = os.path.splitext(os.path.basename(script))[0]
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    path = os.path.join(field.path, 'logs', f'{stem}_{stamp}_{os.getpid()}.log')
    return runlog.start_run_log(path, config_path=script,
                                repo=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


@contextlib.contextmanager
def _action(field, name, settings, lowered, compute=None, numerics=None, frames=None):
    """Pin threads, set the environment knobs (``Compute.tuning``, ``Numerics.rmatvec_threads``),
    log (:func:`process_log`) and record the action; finish the record however the action ends."""
    pin_threads()
    record = None
    knobs = environment(compute or field.compute, numerics)
    knobs.__enter__()
    try:
        log = process_log(field)
        record = Record(field, name, settings, lowered, frames=frames)
        record.data['tuning'] = Tuning.resolved()
        if log is not None:
            record.data['log'] = log.path
            print(f"[selfcal] {name} {field.name}: record {record.path}", flush=True)
        record.write()
        with _console_logging():
            yield record
    except BaseException as e:
        if record is not None:
            record.finish(error=e)
        raise
    finally:
        knobs.__exit__(None, None, None)


def _prepare(plan):
    """Before a calibration runs: delete the products it makes again (``overwrite``), and the N-pass
    intermediates that were made from other inputs (an interrupted run with the same inputs resumes)."""
    from ..io.atomic import sweep_partials
    from .schedule import schedule
    for prod in plan.products:                    # what interrupted writes left behind
        sweep_partials(prod.path)
    replaced = {prod.path for prod in plan.products if prod.state == 'replace'}
    for path in replaced:
        remove_product(path)
    if plan.passes is None:
        return
    passes, book = plan.passes, plan.book
    for spec, ctx in zip(plan.lowered, plan.contexts):
        cal_dir = ctx.pipeline_config.cal_dir
        for job in ctx.jobs():
            stem = ctx.pass_stem(job)
            work = os.path.join(spec.scratch, f'npass_{stem}')
            if replaced and os.path.isdir(work):
                shutil.rmtree(work)
            expected, previous = {}, book.init_path[job.name]
            for i, kind in enumerate(schedule(passes.n, passes.order)[1:], start=2):
                path = os.path.join(cal_dir, f'{stem}_pass{i}{"sky" if kind == "sky" else "off"}.h5')
                if kind == 'sky':
                    expected[f'pass{i}'] = fingerprint(book.passed(job, i, 'sky'))
                else:
                    expected[f'sky{i - 1}'] = book.fp_of(previous)
                previous = path
            clear_stale_intermediates(work, cal_dir, stem, expected)


@dataclass(frozen=True)
class Field(Config):
    """One data set: its folder ``path`` (``<output_dir>/<run_name>``; a relative path is relative
    to the working directory of each action), the ``instrument`` that took it, the reference
    grid's ``pixel_scale`` (arcsec, needed to make the grid), and the machine its actions use by
    default (``compute``)."""
    path: str
    instrument: Instrument
    pixel_scale: float | None = None
    _: KW_ONLY
    compute: Compute = dc_field(default_factory=Compute)

    def _validate(self):
        inst = type(self.instrument)
        if not (is_dataclass(inst) and inst.__dataclass_params__.frozen):
            raise ConfigError(f"Field(instrument={inst.__name__}(...)): an sc.Instrument subclass is a frozen "
                              f"dataclass; put @dataclass(frozen=True) above class {inst.__name__}")
        # Kept as written (only ~ expanded): the cal file records its frames under this path.
        path = os.path.expanduser(self.path)
        object.__setattr__(self, 'path', path.rstrip('/') or path)
        if self.pixel_scale is not None and self.pixel_scale <= 0:
            raise ConfigError(f"Field(pixel_scale={self.pixel_scale}): positive (arcsec per pixel)")

    def __repr__(self):
        args = [repr(self.path), repr(self.instrument)]
        if self.pixel_scale is not None:
            args.append(repr(self.pixel_scale))
        if self.compute != Compute():
            args.append(f'compute={self.compute!r}')
        return f"Field({', '.join(args)})"

    @property
    def name(self) -> str:
        """The run name: the folder's name."""
        return os.path.basename(self.path)

    @property
    def frames(self) -> list[str]:
        """The reprojected frame files, sorted."""
        return sorted(glob.glob(os.path.join(self.path, 'reprojected', '*.h5')))

    def reference(self):
        """``(wcs, shape)`` of the reference grid (``ref.fits``)."""
        from ..geometry import wcs_helper
        return wcs_helper.load_from_fits(os.path.join(self.path, 'ref.fits'))

    # ---- actions ----------------------------------------------------------------------------
    def plan(self, recipe=None, *, jobs=None, tiles=None, passes=None, frames=None, overwrite=False, compute=None,
             action='calibrate'):
        """What :meth:`calibrate` (or ``action="mosaic"``) would do, checked: a printable
        :class:`~selfcal.run.plan.Plan`, listing each product and whether it would be made, reused
        or refused (:attr:`~selfcal.run.plan.Plan.refused`). Computes nothing; raises
        :class:`~selfcal.config.base.ConfigError` for anything else that would fail."""
        return make_plan(self, action, recipe, jobs=jobs, tiles=tiles, passes=passes, frames=frames, compute=compute,
                         overwrite=overwrite, check_products=False, allow_no_frames=True)

    def reproject(self, exposures, *, reference=None, method='exact', padding=100, padding_fraction=0.05,
                  replace=False, verify=False, compute=None):
        """Reproject raw exposure files onto the field's reference grid, one frame file per
        (exposure, detector) in ``reprojected/``. ``exposures``: a glob pattern, or a list of
        patterns or files. The grid is ``ref.fits`` when it exists, else made from ``reference``
        (a FITS file whose WCS it takes) or fitted to the exposures, ``padding`` pixels and
        ``padding_fraction`` wider, at ``pixel_scale``. ``method``: ``"exact"`` (flux-conserving),
        ``"interp"`` (bilinear) or ``"adaptive"``. Existing frames are kept unless ``replace``; ``verify`` load-tests every
        frame afterwards. Returns the frame directory."""
        from .pipelines import run
        check_main_guard()
        compute = compute or self.compute
        if self.pixel_scale is None and not os.path.exists(os.path.join(self.path, 'ref.fits')):
            raise ConfigError(f"{self!r}: reprojection makes the reference grid at pixel_scale=, which is not given")
        spec = self.reprojection_spec(exposures, reference=reference, method=method, padding=padding,
                                      padding_fraction=padding_fraction, replace=replace, verify=verify,
                                      compute=compute)
        settings = {'exposures': list(spec.reproject.exposures), 'reference': reference, 'method': method,
                    'padding': padding, 'padding_fraction': padding_fraction, 'replace': replace, 'verify': verify,
                    'compute': compute}
        with _action(self, 'reproject', settings, [spec], compute) as record:
            out = run(spec)
            record.finish(products={'reprojected': out})
        return out

    def reprojection_spec(self, exposures, *, reference=None, method='exact', padding=100, padding_fraction=0.05,
                          replace=False, verify=False, compute=None):
        """The engine run :meth:`reproject` runs (a :class:`~selfcal.run.runspec.RunSpec`; nothing is
        read or run)."""
        from .runspec import ReprojectSpec, RunSpec
        compute = compute or self.compute
        if method not in ('exact', 'interp', 'adaptive'):
            raise ConfigError(f"reproject(method={method!r}): 'exact', 'interp' or 'adaptive'")
        patterns = [exposures] if isinstance(exposures, (str, os.PathLike)) else list(exposures)
        reproject = ReprojectSpec(exposures=tuple(os.fspath(p) for p in patterns),
                                  reference=None if reference is None else os.fspath(reference), method=method,
                                  padding=int(padding), padding_fraction=float(padding_fraction),
                                  replace=bool(replace), verify=bool(verify), workers=compute.resolved_workers)
        return RunSpec(task='reproject', field=self, instrument=self.instrument, output_dir=os.path.dirname(self.path),
                       run_name=self.name, resolution_arcsec=self.pixel_scale, reproject=reproject)

    def calibrate(self, recipe=None, *, jobs=None, tiles=None, passes=None, frames=None, overwrite=False,
                  compute=None) -> Result:
        """Solve ``recipe`` (a :class:`~selfcal.run.recipe.Recipe`, or a bare model) for each job,
        and coadd each job's mosaic when the recipe has a coadd.

        ``jobs``: the instrument's jobs (default: its own; SPHEREx names them). ``tiles``: solve
        tile by tile and stitch (:class:`~selfcal.run.schedule.Tiles`); ``passes``: the N-pass
        alternating solve (:class:`~selfcal.run.schedule.Passes`). ``frames``: None (every
        frame), a number (the first ``n``), a directory (its frames, read in place) or a list of
        frame files. An existing cal is reused when it was made by the same inputs; ``overwrite``
        makes the jobs' cal and mosaic again. ``compute`` overrides the field's.
        """
        check_main_guard()
        recipe = as_recipe(recipe)
        plan = make_plan(self, 'calibrate', recipe, jobs=jobs, tiles=tiles, passes=passes, frames=frames,
                         compute=compute, overwrite=overwrite)
        settings = {'recipe': recipe, 'jobs': plan.jobs, 'tiles': tiles, 'passes': passes, 'frames': frames,
                    'compute': plan.compute, 'overwrite': overwrite}
        return self._run('calibrate', plan, recipe, settings)

    def mosaic(self, recipe=None, *, jobs=None, cal=None, frames=None, overwrite=False, compute=None) -> Result:
        """Coadd each job's mosaic from its existing cal file, or from ``cal`` (a cal file or a
        :class:`~selfcal.run.result.Result`, e.g. a solution on another grid; its frames without a
        frame file here are dropped). A current mosaic is reused unless ``overwrite``."""
        check_main_guard()
        recipe = as_recipe(recipe)
        if isinstance(cal, Result):
            cal = cal.final or (cal.cal_paths[0] if len(cal.cal_paths) == 1 else None)
            if cal is None:
                raise ConfigError("mosaic(cal=...): a result with one cal")
        plan = make_plan(self, 'mosaic', recipe, jobs=jobs, frames=frames, cal=cal and os.fspath(cal),
                         compute=compute, overwrite=overwrite)
        settings = {'recipe': recipe, 'jobs': plan.jobs, 'cal': cal and os.fspath(cal), 'frames': frames,
                    'compute': plan.compute, 'overwrite': overwrite}
        return self._run('mosaic', plan, recipe, settings)

    def adopt(self, recipe=None, *, jobs=None, tiles=None, passes=None, frames=None, compute=None) -> list[str]:
        """Record existing products as made by ``recipe`` (products made before records existed, or
        by a TOML run): each product the matching :meth:`calibrate` would make that exists without a
        sidecar is checked (its frames, sky terms, offset maps and detector shape; a mosaic's maps)
        and given one. What a product does not show (the fit's and the coadd's settings) is taken
        on trust: adopt only products this recipe made. A product is adopted only when what it is
        made from (a mosaic's cal) is current or adopted too. Returns the adopted paths; raises
        :class:`~selfcal.config.base.ConfigError` listing the products that do not match (the
        others are adopted)."""
        from .products import _cal_frames
        recipe = as_recipe(recipe)
        plan = make_plan(self, 'calibrate', recipe, jobs=jobs, tiles=tiles, passes=passes, frames=frames,
                         compute=compute, check_workers=False, check_products=False)
        geom = plan.contexts[0].geom if plan.contexts else None
        adopted, problems = [], []
        sound = {prod.path for prod in plan.products if prod.state == 'current'}
        for prod in plan.products:                 # engine order: dependencies are adopted first
            if prod.state != 'unrecorded':
                continue
            unsound = [d for d in prod.depends if d not in sound]
            if unsound:
                problems.append(f"{os.path.basename(prod.path)}: made from {os.path.basename(unsound[0])}, which is "
                                f"not current or adopted")
                continue
            inputs = prod.inputs()
            issues = verify(prod, recipe, geom)
            if prod.kind == 'cal' and not issues:
                got = frames_digest(_cal_frames(prod.path))
                if got != inputs['frames']:
                    issues.append(f"its {got['n']} frames are not the {inputs['frames']['n']} expected")
            if issues:
                problems.append(f"{os.path.basename(prod.path)}: {'; '.join(issues)}")
                continue
            write_sidecar(prod.path, inputs, adopted=True)
            adopted.append(prod.path)
            sound.add(prod.path)
        if problems:
            raise ConfigError(f"adopted {len(adopted)} products; these do not match the recipe:\n  "
                              + '\n  '.join(problems))
        return adopted

    def submit(self, recipe=None, *, jobs=None, tiles=None, passes=None, frames=None, overwrite=False,
               compute=None) -> Submitted:
        """Plan :meth:`calibrate` now (every check), then run it detached in its own session, so it
        outlives this process and its terminal. The run is the rerun of a request written to
        ``records/`` (``selfcal rerun``), so every function it uses must be importable (no
        :class:`~selfcal.config.functions.by_value`). Returns a :class:`Submitted`."""
        from .records import write_request
        check_main_guard()
        recipe = as_recipe(recipe)
        plan = make_plan(self, 'calibrate', recipe, jobs=jobs, tiles=tiles, passes=passes, frames=frames,
                         compute=compute, overwrite=overwrite)
        if _by_value_in(recipe):
            raise ConfigError("submit(): the recipe sends a function by value (sc.by_value), which a detached run "
                              "cannot import; write the function to a module")
        settings = {'recipe': recipe, 'jobs': plan.jobs, 'tiles': tiles, 'passes': passes, 'frames': frames,
                    'compute': plan.compute, 'overwrite': overwrite}
        request = write_request(self, 'calibrate', settings)
        # the detached run rebuilds the settings from the request: check now that it can (arrays are
        # recorded by hash only; a hook class must be importable)
        probe = subprocess.run([sys.executable, '-c', 'import sys; from selfcal.run.records import load_action; '
                                'load_action(sys.argv[1])', request], capture_output=True, text=True, timeout=600,
                               cwd=os.getcwd())
        if probe.returncode != 0:
            os.remove(request)
            last = (probe.stderr.strip().splitlines() or ['?'])[-1]
            raise ConfigError(f"submit(): a detached run could not rebuild these settings from the request ({last}); "
                              f"keep arrays in files (sc.DetectorMap('map.npy')) and hooks in importable modules, "
                              f"or run the action here")
        console = os.path.join(self.path, 'logs', os.path.basename(request)[:-len('.json')] + '.console')
        os.makedirs(os.path.dirname(console), exist_ok=True)
        command = [sys.executable, '-m', 'selfcal', 'rerun', request] + (['--overwrite'] if overwrite else [])
        with open(console, 'ab') as out:
            proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                                    start_new_session=True, cwd=os.getcwd())
        return Submitted(request=request, console=console, pid=proc.pid, process=proc)

    def result(self, recipe=None, *, jobs=None, tiles=None, passes=None) -> Result:
        """The products ``recipe`` made (or would make) for ``jobs``, as a
        :class:`~selfcal.run.result.Result`, without running anything."""
        recipe = as_recipe(recipe)
        lowered = lower(self, recipe, jobs=jobs, tiles=tiles, passes=passes)
        from .engine import RunContext
        cals, mosaics, jobs_out = [], [], []
        final = None
        for spec in lowered:
            ctx = RunContext.build(spec, need_geometry=False)
            for job in ctx.jobs():
                path = ctx.cal_path(job) if tiles is None else ctx.stitched_cal_path(job)
                cals.append(path)
                if recipe.coadd is not None and tiles is None and passes is None:
                    mosaics.append(ctx.mosaic_path(job))
                jobs_out.append(job)
        if passes is not None and cals:
            stem = os.path.basename(cals[0])[:-len('.h5')]
            from .schedule import schedule
            sched = schedule(passes.n, passes.order)
            for i in range(len(sched), 0, -1):
                if sched[i - 1] == 'sky':
                    final = os.path.join(os.path.dirname(cals[0]), f'{stem}_pass{i}sky.h5')
                    break
        return Result(self, recipe, tuple(jobs_out), cals, mosaics, final=final or (cals[0] if len(cals) == 1 else None))

    # ---- running -------------------------------------------------------------------------------
    def _run(self, action, plan, recipe, settings) -> Result:
        from .pipelines import run
        jobs, cals, mosaics, tiles_out, passes_out, final = [], [], [], {}, None, None
        from .products import frames_digest as digest
        from .records import check_rerun_frames
        frames = digest(getattr(plan, 'frame_list', []))
        check_rerun_frames(frames)
        with _action(self, action, settings, plan.lowered, plan.compute, recipe.numerics, frames=frames) as record:
            _prepare(plan)
            plan.book.record = record.path
            for spec, ctx in zip(plan.lowered, plan.contexts):
                spec.on_product = plan.book
                out = run(spec, ctx)
                if isinstance(out, dict):                      # the N-pass scheduler
                    passes_out = {int(k): v for k, v in out['products'].items()}
                    final = out['final']
                    cals.extend(passes_out[1] if isinstance(passes_out[1], list) else [passes_out[1]])
                else:
                    cals.extend([out.stitched] if out.stitched else out.cal_paths)
                    mosaics.extend(out.mosaic_paths)
                    if out.tiles:
                        tiles_out.update(out.tiles)
                        final = out.stitched
                jobs.extend(ctx.jobs())
            if final is None and len(cals) == 1:
                final = cals[0]
            record.finish(products={'cal': cals, 'mosaic': mosaics, 'tiles': tiles_out or None,
                                    'passes': passes_out, 'final': final, 'sidecars_written': plan.book.written})
        return Result(self, recipe, tuple(jobs), cals, mosaics, tile_cals=tiles_out or None, passes=passes_out,
                      final=final, record=record.path)


def _by_value_in(obj) -> bool:
    """Whether a settings object holds a function sent by value."""
    import dataclasses

    from ..config.functions import by_value
    if isinstance(obj, by_value):
        return True
    if isinstance(obj, Config):
        return any(_by_value_in(getattr(obj, f.name)) for f in dataclasses.fields(obj))
    if isinstance(obj, (tuple, list)):
        return any(_by_value_in(x) for x in obj)
    if isinstance(obj, dict):
        return any(_by_value_in(x) for x in obj.values())
    return False
