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
import sys
from dataclasses import KW_ONLY, dataclass
from dataclasses import field as dc_field

from ..config.base import Config, ConfigError
from ..instruments.contract import Instrument
from .compute import Compute, pin_threads
from .lower import as_recipe, lower
from .plan import check_main_guard, make_plan
from .records import Record
from .result import Result

__all__ = ['Field', 'frames_in']


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


@contextlib.contextmanager
def _action(field, name, settings, lowered):
    """Pin threads, start the log (scripts only) and the record; finish them however the action ends."""
    from . import runlog
    pin_threads()
    main = sys.modules.get('__main__')
    script = getattr(main, '__file__', None)
    record = None
    log = None
    try:
        record = Record(field, name, settings, lowered)
        if script:
            path = os.path.join(field.path, 'logs', f'{record.stem}.log')
            log = runlog.start_run_log(path, config_path=script,
                                       repo=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            if log is not None:
                record.data['log'] = log.path
                record.write()
        with _console_logging():
            yield record
    except BaseException as e:
        if record is not None:
            record.finish(error=e)
        raise
    finally:
        if log is not None:
            log.stop()
            runlog._active = None


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
        # Kept as written (only ~ expanded): the cal file records its frames under this path, as
        # a TOML run records them under <output_dir>/<run_name>.
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
    def plan(self, recipe=None, *, jobs=None, tiles=None, passes=None, frames=None, compute=None, action='calibrate'):
        """What :meth:`calibrate` (or ``action="mosaic"``) would do, checked: a printable
        :class:`~selfcal.run.plan.Plan`. Computes nothing; raises
        :class:`~selfcal.config.base.ConfigError` for what would fail."""
        return make_plan(self, action, recipe, jobs=jobs, tiles=tiles, passes=passes, frames=frames, compute=compute)

    def reproject(self, exposures, *, reference=None, method='exact', padding=100, padding_fraction=0.05,
                  replace=False, verify=False, compute=None):
        """Reproject raw exposure files onto the field's reference grid, one frame file per
        (exposure, detector) in ``reprojected/``. ``exposures``: a glob pattern, or a list of
        patterns or files. The grid is ``ref.fits`` when it exists, else made from ``reference``
        (a FITS file whose WCS it takes) or fitted to the exposures, ``padding`` pixels and
        ``padding_fraction`` wider, at ``pixel_scale``. ``method``: ``"exact"`` (flux-conserving),
        ``"interp"`` (bilinear) or ``"adaptive"``. Existing frames are kept unless ``replace``; ``verify`` load-tests every
        frame afterwards. Returns the frame directory."""
        from .config import RunConfig
        from .pipelines import run
        check_main_guard()
        compute = compute or self.compute
        if self.pixel_scale is None and not os.path.exists(os.path.join(self.path, 'ref.fits')):
            raise ConfigError(f"{self!r}: reprojection makes the reference grid at pixel_scale=, which is not given")
        if method not in ('exact', 'interp', 'adaptive'):
            raise ConfigError(f"reproject(method={method!r}): 'exact', 'interp' or 'adaptive'")
        patterns = [exposures] if isinstance(exposures, (str, os.PathLike)) else list(exposures)
        patterns = [os.fspath(p) for p in patterns]
        inst, table = self.instrument.engine(())
        cfg = RunConfig(task='reproject', output_dir=os.path.dirname(self.path), run_name=self.name,
                        resolution_arcsec=self.pixel_scale, cache_dir=compute.scratch, instrument_cfg=table,
                        reproject={'input_dirs': patterns, 'file_pattern': '', 'reproj_func': method,
                                   'padding_pixels': int(padding), 'padding_percentage': float(padding_fraction),
                                   'replace_existing': bool(replace), 'check': bool(verify),
                                   'source_ref_path': None if reference is None else os.fspath(reference),
                                   'max_workers': compute.resolved_workers})
        cfg.instrument = inst
        settings = {'exposures': patterns, 'reference': reference, 'method': method, 'padding': padding,
                    'padding_fraction': padding_fraction, 'replace': replace, 'verify': verify, 'compute': compute}
        from .lower import Lowered
        with _action(self, 'reproject', settings, [Lowered(cfg, ())]) as record:
            out = run(cfg)
            record.finish(products={'reprojected': out})
        return out

    def calibrate(self, recipe=None, *, jobs=None, tiles=None, passes=None, frames=None, overwrite=False,
                  compute=None) -> Result:
        """Solve ``recipe`` (a :class:`~selfcal.run.recipe.Recipe`, or a bare model) for each job,
        and coadd each job's mosaic when the recipe has a coadd.

        ``jobs``: the instrument's jobs (default: its own; SPHEREx names them). ``tiles``: solve
        tile by tile and stitch (:class:`~selfcal.run.schedule.Tiles`); ``passes``: the N-pass
        alternating solve (:class:`~selfcal.run.schedule.Passes`). ``frames``: None (every
        frame), a number (the first ``n``), a directory (its frames, read in place) or a list of
        frame files. An existing cal is reused, as by a TOML run, unless ``overwrite`` (which
        deletes the jobs' cal and mosaic first). ``compute`` overrides the field's.
        """
        check_main_guard()
        recipe = as_recipe(recipe)
        plan = self.plan(recipe, jobs=jobs, tiles=tiles, passes=passes, frames=frames, compute=compute)
        if overwrite:
            if tiles is not None or passes is not None:
                raise ConfigError("calibrate(overwrite=True) deletes a plain run's products; remove a tiled or "
                                  "N-pass run's products by hand")
            for _, cal, mos, _ in plan.products:
                for p in (cal, mos):
                    if p and os.path.exists(p):
                        os.remove(p)
        settings = {'recipe': recipe, 'jobs': plan.jobs, 'tiles': tiles, 'passes': passes, 'frames': frames,
                    'compute': plan.compute}
        return self._run('calibrate', plan, recipe, settings)

    def mosaic(self, recipe=None, *, jobs=None, cal=None, frames=None, compute=None) -> Result:
        """Coadd each job's mosaic from its existing cal file, or from ``cal`` (a cal file or a
        :class:`~selfcal.run.result.Result`, e.g. a solution on another grid; its frames without a
        frame file here are dropped)."""
        check_main_guard()
        recipe = as_recipe(recipe)
        if isinstance(cal, Result):
            cal = cal.final or (cal.cal_paths[0] if len(cal.cal_paths) == 1 else None)
            if cal is None:
                raise ConfigError("mosaic(cal=...): a result with one cal")
        plan = make_plan(self, 'mosaic', recipe, jobs=jobs, frames=frames, cal=cal and os.fspath(cal),
                         compute=compute)
        settings = {'recipe': recipe, 'jobs': plan.jobs, 'cal': cal, 'frames': frames, 'compute': plan.compute}
        return self._run('mosaic', plan, recipe, settings)

    def result(self, recipe=None, *, jobs=None, tiles=None, passes=None) -> Result:
        """The products ``recipe`` made (or would make) for ``jobs``, as a
        :class:`~selfcal.run.result.Result`, without running anything."""
        recipe = as_recipe(recipe)
        lowered = lower(self, recipe, jobs=jobs, tiles=tiles, passes=passes)
        from .engine import RunContext
        cals, mosaics, jobs_out = [], [], []
        final = None
        for low in lowered:
            ctx = RunContext.build(low.cfg, need_geometry=False, need_mode=False)
            ctx.frame_tag = ctx.inst.frame_tag(low.cfg.instrument_cfg)
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
        with _action(self, action, settings, plan.lowered) as record:
            for low, ctx in zip(plan.lowered, plan.contexts):
                out = run(low.cfg)
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
                                    'passes': passes_out, 'final': final})
        return Result(self, recipe, tuple(jobs), cals, mosaics, tile_cals=tiles_out or None, passes=passes_out,
                      final=final, record=record.path)
