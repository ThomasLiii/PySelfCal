"""Lowering: the Python API's objects to the run engine's :class:`~selfcal.run.runspec.RunSpec`.

A field, a recipe, jobs and the options of an action become the engine runs of the action, one
per group of jobs the instrument runs together, with every value resolved: the library keywords
of the solve, the solver and the coadd, the frames and their staging, the tiles and the passes.
Nothing here reads data; a tiled run reads the reference grid's shape from ``ref.fits``.
"""
from __future__ import annotations

import os

from ..config.base import ConfigError
from ..models.model import Model
from .recipe import ChunkGroups, Recipe
from .runspec import FrameSource, PassClip, PassesSpec, RunSpec, TilingSpec

__all__ = ['lower', 'setup_options', 'solver_options', 'coadd_options', 'passes_spec', 'tiling_spec', 'start_paths']


def as_recipe(recipe) -> Recipe:
    """A :class:`~selfcal.run.recipe.Recipe` from a recipe, a bare model, or None (the default recipe)."""
    if recipe is None:
        return Recipe()
    if isinstance(recipe, Model):
        return Recipe(recipe)
    if isinstance(recipe, Recipe):
        return recipe
    raise ConfigError(f"expected a Recipe or a Model, got {recipe!r}")


# --------------------------------------------------------------------------- the library's keywords
def _ignore(flags, instrument):
    return list(instrument.default_ignore_flags() if flags is None else flags)


def clip_keys(clip) -> dict:
    """The ``setup_lsqr`` keywords of a fit's clip (chunk groups as ``outlier_groups``, which the
    engine resolves on the instrument's primary chunk map)."""
    if clip is None:
        return {'outlier_thresh': None}
    out = {'outlier_thresh': float(clip.sigma)}
    if clip.edges is not None:
        out['outlier_group_edges'] = list(clip.edges)
        if clip.variable is not None:
            out['outlier_group_variable'] = clip.variable
    elif clip.per == 'chunk':
        out['outlier_groups'] = {'chunk': True}
    elif isinstance(clip.per, ChunkGroups):
        g = clip.per
        out['outlier_groups'] = ({'along': g.axis, 'map': g.map} if g.axis is not None
                                 else {'mapping': list(g.groups), 'map': g.map})
    return out


def setup_options(recipe, compute, instrument) -> dict:
    """The solve's ``setup_lsqr`` keywords (the model's own options aside)."""
    fit, model = recipe.fit, recipe.model
    damp = model.sky_dampings()
    options = {'apply_mask': fit.use_mask, 'apply_weight': fit.shot_noise_weights,
               'ignore_list': _ignore(fit.ignore_flags, instrument),
               'offset_regularization': True, 'weighted_damping': True,
               'damp_weight': float(damp[0]), 'batch_size': recipe.numerics.batch,
               'max_workers': compute.resolved_workers}
    options.update(clip_keys(fit.clip))
    return options


def solver_options(recipe) -> dict:
    """The solver's ``apply_lsqr`` keywords: with ``Fit(tolerance=0)`` also ``conlim=0``, so the
    solve runs exactly ``Fit.iterations`` iterations (otherwise the solver's own ``conlim``); with
    ``Fit(stop=...)`` the stop rules (``stop``)."""
    fit = recipe.fit
    atol, btol = fit.atol_btol
    out = {'solver': fit.method, 'iter_lim': fit.iterations, 'atol': atol, 'btol': btol, 'damp': float(fit.damp),
           'precondition': fit.precondition, 'use_float32': fit.float32, 'n_threads': recipe.numerics.threads}
    if fit.exact_iterations:
        out['conlim'] = 0.0
    if fit.stop is not None:
        out['stop'] = fit.stop
    return out


def coadd_options(recipe, compute, instrument) -> dict:
    """The coadd's ``make_mosaic`` keywords, or ``{}`` when the recipe has no coadd."""
    c = recipe.coadd
    if c is None:
        return {}
    return {'apply_mask': c.use_mask, 'apply_weight': c.shot_noise_weights,
            'ignore_list': _ignore(c.ignore_flags, instrument), 'make_std_map': c.std,
            'apply_sigma_clipping': c.clip is not None, 'sigma': 2.0 if c.clip is None else float(c.clip),
            'apply_offset': c.subtract_offsets, 'normalize_offset': c.normalize_offsets,
            'valid_chunk_thresh': float(c.min_chunk_coverage),
            'cache_batch_size': recipe.numerics.mosaic_batch, 'coadd_batch_size': recipe.numerics.coadd_batch,
            'cache_intermediate': compute.cache_frames, 'max_workers': compute.resolved_coadd_workers}


def _pass_clip(clip, ignore_flags=None) -> PassClip:
    """An N-pass clip: grouped by the primary map's spectral axis (chunk groups along an axis), or
    per frame."""
    if clip.edges is not None or clip.per == 'chunk' or (isinstance(clip.per, ChunkGroups) and clip.per.axis is None):
        raise ConfigError("the N-pass clips group by an axis of the primary chunk map (ChunkGroups.along(...)) "
                          "or not at all (per='frame')")
    return PassClip(float(clip.sigma), isinstance(clip.per, ChunkGroups), ignore_flags)


def passes_spec(passes) -> PassesSpec:
    """The N-pass schedule of :class:`~selfcal.run.schedule.Passes`."""
    p, refit = passes, passes.offset
    init = None
    if p.init_clip is not None:
        flags = p.init_clip.ignore_flags
        init = _pass_clip(p.init_clip, None if flags is None else tuple(flags))
    return PassesSpec(n=p.n, order=p.order, init_clip=init, sky_clip=_pass_clip(p.sky_clip),
                      refit_degree=refit.degree, refit_clip=_pass_clip(refit.clip), bright_cut=refit.bright_cut,
                      min_pixels=refit.min_pixels,
                      segments=None if refit.segments is None else tuple(tuple(s) for s in refit.segments),
                      ridge=float(refit.ridge), stop_tol=float(p.stop_tol), sky_merge=p.sky_merge,
                      keep_moments=p.keep_moments)


def _suffix(template, name, tile='{tile}'):
    text = template.format(name=name, tile=tile).strip('_')
    return f'_{text}' if text else ''


def reference_shape(field):
    """``(rows, cols)`` of the field's reference grid, from its ``ref.fits``."""
    path = os.path.join(field.path, 'ref.fits')
    if not os.path.exists(path):
        raise ConfigError(f"{field}: no reference grid {path} (reproject the exposures first)")
    from astropy.io import fits
    h = fits.getheader(path)
    return int(h['NAXIS2']), int(h['NAXIS1'])


def tiling_spec(tiles, field, recipe, compute, scratch, frames_dir) -> TilingSpec:
    """The tiles of a tiled calibration (:class:`~selfcal.run.schedule.Tiles`)."""
    t = tiles
    boxes = None if t.boxes is None else tuple((name, tuple(int(v) for v in box)) for name, box in t.boxes.items())
    return TilingSpec(ref_shape=reference_shape(field), frames_dir=frames_dir,
                      stage_dir=os.path.join(scratch, compute.stage_dir or f'reproj_nvme_{field.name}'),
                      stitched_suffix=_suffix(t.stitched_name, recipe.name), tiles=boxes,
                      grid=None if boxes is not None else tuple(t.grid), overlap=t.overlap,
                      names=None if t.names is None else tuple(t.names), only=t.only, assign=t.assign, halo=t.halo,
                      stitch_line=len(recipe.model.sky) > 1,
                      memory_guard=True if compute.memory_guard is None else bool(compute.memory_guard))


# --------------------------------------------------------------------------- a warm start
def start_paths(start, jobs) -> dict | None:
    """``{job name: cal path}`` of ``calibrate(start=...)`` for ``jobs`` (None without a start).

    ``start``: a cal file (a path or an open :class:`~selfcal.io.calfile.CalFile`), for an action of
    one job; an earlier :class:`~selfcal.run.result.Result`, whose cal of each job (by name) starts
    the job of the same name; or a mapping ``{job (or its name): cal}``. Every job needs its start;
    a result's or a mapping's other jobs are left out. The paths are made absolute (they are
    recorded)."""
    if start is None:
        return None
    from ..io.calfile import CalFile
    from .result import Result

    def path(p):
        if isinstance(p, CalFile):
            p = p.path
        if not isinstance(p, (str, os.PathLike)):
            raise ConfigError(f"calibrate(start=...): a cal file, a Result or {{job: cal}}; got {p!r}")
        return os.path.abspath(os.fspath(p))
    names = [j.name for j in jobs]
    if isinstance(start, Result):
        if start.tile_cals or start.passes:
            raise ConfigError("calibrate(start=...): a tiled or an N-pass result is no start (its cals are a "
                              "stitched sky or the passes'); give a plain calibration's result or cal")
        given = {j.name: p for j, p in zip(start.jobs, start.cal_paths)}
    elif isinstance(start, dict):
        given = {getattr(k, 'name', k): v for k, v in start.items()}
    else:
        if len(names) != 1:
            raise ConfigError(f"calibrate(start=...): one cal for {len(names)} jobs; give a Result of these jobs "
                              f"or {{job: cal}}")
        given = {names[0]: start}
    missing = [n for n in names if n not in given]
    if missing:
        raise ConfigError(f"calibrate(start=...): no start for the job(s) {missing} (the start has "
                          f"{sorted(given)})")
    return {n: path(given[n]) for n in names}


# --------------------------------------------------------------------------- the engine runs
def _scratch(field, recipe, compute, tiles, passes):
    """The engine's scratch area (with its trailing ``/``): ``Compute.scratch``; without one,
    ``<field>/scratch/`` when the run needs a scratch area (the coadd's frame cache, tiles, passes),
    else none."""
    if compute.scratch is not None:
        return compute.scratch.rstrip('/') + '/'
    needs = tiles is not None or passes is not None or (recipe.coadd is not None and compute.cache_frames)
    return os.path.join(field.path, 'scratch') + '/' if needs else None


def _frame_source(field, frames, compute, scratch, tiles):
    """``(FrameSource, frame directory)`` of ``frames=``: None (the field's frames), a number (the
    first ``n``), a directory (its frames, read in place) or a list of frame files."""
    reproj_dir = os.path.join(field.path, 'reprojected')
    frames_dir, first_n, files, in_place = reproj_dir, None, None, None
    if isinstance(frames, int) and not isinstance(frames, bool):
        first_n = frames
    elif isinstance(frames, (str, os.PathLike)):
        frames_dir = in_place = os.fspath(frames)
    elif frames is not None:
        files = tuple(os.fspath(f) for f in frames)
        dirs = {os.path.dirname(f) for f in files if os.path.dirname(f)}
        if len(dirs) > 1:
            raise ConfigError(f"frames=: the frame files lie in {len(dirs)} directories; give frames of one")
        if dirs and os.path.normpath(dirs.pop()) != os.path.normpath(reproj_dir):
            frames_dir = in_place = os.path.dirname(files[0])
    if not compute.stages and in_place is None:
        in_place = reproj_dir
    stage_dir = None
    if tiles is not None:
        if files is not None or first_n is not None:
            raise ConfigError("a tiled calibration takes every frame of its frame directory; frames= may name a "
                              "directory only")
        in_place = None                        # a tiled run stages each tile's frames from frames_dir
    else:
        stage_dir = compute.stage_dir or (os.path.join(scratch, f'reproj_nvme_{field.name}') if scratch else None)
    source = FrameSource(in_place=in_place, files=files, first_n=first_n, stage=compute.stage or 'copy',
                         stage_dir=stage_dir, keep=compute.keep_staged, io_limit=compute.io_limit)
    return source, frames_dir


def lower(field, recipe=None, *, task='cal', jobs=None, tiles=None, passes=None, frames=None, cal=None,
          compute=None, start=None, snapshots=None, monitor=None) -> list[RunSpec]:
    """The engine runs of one action: a :class:`~selfcal.run.runspec.RunSpec` per group of jobs the
    instrument runs together (SPHEREx: its channel jobs, then its window jobs).

    ``task``: ``"cal"`` (with ``tiles``: tiled; with ``passes``: the N-pass solve) or ``"mosaic"``
    (of the cal ``cal``, default each job's own). ``frames``: None (the field's frames), a number
    (the first ``n``), a directory (its frames, read in place) or a list of frame files.
    ``start``: the cal each job's solve starts from (:func:`start_paths`; a plain calibration only).
    ``snapshots``: the solution every ``k`` iterations as a cal file
    (:class:`~selfcal.run.schedule.Snapshots`; a plain calibration only). ``monitor``: checks of
    every solve every ``m`` iterations (:class:`~selfcal.run.schedule.Monitor`; a calibration).
    """
    recipe = as_recipe(recipe)
    compute = compute or field.compute
    inst = field.instrument
    jobs = tuple(inst.default_jobs()) if jobs is None else ((jobs,) if not isinstance(jobs, (list, tuple))
                                                            else tuple(jobs))
    inst.check_jobs(jobs)
    if start is not None and (task != 'cal' or tiles is not None or passes is not None):
        raise ConfigError("calibrate(start=...): a warm start continues a plain calibration, one solve per job; "
                          "a tiled or an N-pass calibration takes no start")
    starts = start_paths(start, jobs)
    if monitor is not None and task == 'mosaic':
        raise ConfigError("mosaic(monitor=...): only a calibration's solve is monitored")
    if snapshots is not None and (task != 'cal' or tiles is not None or passes is not None):
        raise ConfigError("calibrate(snapshots=...): snapshots are written by a plain calibration, one solve per job; "
                          "a tiled or an N-pass calibration takes none")
    groups = {}
    for j in jobs:
        groups.setdefault(j.kind, []).append(j)
    model, coadd = recipe.model, recipe.coadd
    mosaic = 'none' if coadd is None else ('full' if coadd.instrument_maps else 'no_wav')
    scratch = _scratch(field, recipe, compute, tiles, passes)
    source, frames_dir = _frame_source(field, frames, compute, scratch, tiles)
    out = []
    for group in groups.values():
        spec = RunSpec(
            task='npass' if passes is not None else task, field=field, instrument=inst,
            output_dir=os.path.dirname(field.path), run_name=field.name, resolution_arcsec=field.pixel_scale,
            recipe=recipe, jobs=tuple(group), scratch=scratch, suffix=recipe.suffix,
            oversample=coadd.oversample if coadd else 1, frames=source, model=model.lower(mosaic),
            line_fisher_threshold=float(recipe.fit.line_fisher_threshold), spectral_window=model.spectral_window(),
            setup=setup_options(recipe, compute, inst), lsqr=solver_options(recipe),
            mosaic=coadd_options(recipe, compute, inst), pre_cal=recipe.fit.raw_frame_hook,
            post_cal=recipe.fit.frame_hook, post_mosaic=coadd.frame_hook if coadd is not None else None,
            make_mosaic=coadd is not None, instrument_maps=coadd is not None and coadd.instrument_maps,
            cal_override=cal, passes=None if passes is None else passes_spec(passes),
            start=None if starts is None else {j.name: starts[j.name] for j in group}, snapshots=snapshots,
            monitor=monitor)
        if tiles is not None:
            spec.suffix = _suffix(tiles.tile_name, recipe.name)
            spec.tiling = tiling_spec(tiles, field, recipe, compute, scratch, frames_dir)
        out.append(spec)
    return out
