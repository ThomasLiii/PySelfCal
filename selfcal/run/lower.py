"""Lowering: the Python API's objects to the run engine's :class:`~selfcal.run.config.RunConfig`.

A field, a recipe, jobs and the options of an action become the same ``RunConfig`` a TOML run
config makes, so the engine runs a Python-configured calibration exactly as it runs a TOML one
(the equivalence is checked over every shipped config by
``selfcal_scripts/gates/config_equivalence.py``). Nothing here reads data; a tiled run reads
the reference grid's shape from ``ref.fits``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

from ..config.base import ConfigError
from ..models.model import Model
from .config import RunConfig
from .recipe import ChunkGroups, Recipe

__all__ = ['Lowered', 'lower', 'calibration_table', 'lsqr_table', 'mosaic_table', 'passes_table', 'tiling_table']


@dataclass
class Lowered:
    """One engine run of an action: its :class:`~selfcal.run.config.RunConfig` and the jobs it makes."""
    cfg: RunConfig
    jobs: tuple


def as_recipe(recipe) -> Recipe:
    """A :class:`~selfcal.run.recipe.Recipe` from a recipe, a bare model, or None (the default recipe)."""
    if recipe is None:
        return Recipe()
    if isinstance(recipe, Model):
        return Recipe(recipe)
    if isinstance(recipe, Recipe):
        return recipe
    raise ConfigError(f"expected a Recipe or a Model, got {recipe!r}")


# --------------------------------------------------------------------------- tables
def _ignore(flags, instrument):
    return list(instrument.default_ignore_flags() if flags is None else flags)


def clip_keys(clip) -> dict:
    """The ``[calibration]`` keys of a fit's clip."""
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


def calibration_table(recipe, compute, instrument) -> dict:
    """``[calibration]``: the solve's ``setup_lsqr`` keywords."""
    fit, model = recipe.fit, recipe.model
    damp = model.sky_dampings()
    table = {'apply_mask': fit.use_mask, 'apply_weight': fit.shot_noise_weights,
             'ignore_list': _ignore(fit.ignore_flags, instrument),
             'offset_regularization': True, 'weighted_damping': True,
             'damp_weight': float(damp[0]), 'batch_size': recipe.numerics.batch,
             'max_workers': compute.resolved_workers}
    table.update(clip_keys(fit.clip))
    return table


def lsqr_table(recipe) -> dict:
    """``[lsqr]``: the solver's ``apply_lsqr`` keywords."""
    fit = recipe.fit
    atol, btol = fit.atol_btol
    return {'solver': fit.method, 'iter_lim': fit.iterations, 'atol': atol, 'btol': btol, 'damp': float(fit.damp),
            'precondition': fit.precondition, 'use_float32': fit.float32}


def mosaic_table(recipe, compute, instrument) -> dict:
    """``[mosaic]``: the coadd's ``make_mosaic`` keywords."""
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


def _grouped(clip) -> bool:
    """Whether an N-pass clip is grouped (by the primary map's spectral axis: ``subch_clip``)."""
    if clip.edges is not None or clip.per == 'chunk' or (isinstance(clip.per, ChunkGroups) and clip.per.axis is None):
        raise ConfigError("the N-pass clips group by an axis of the primary chunk map (ChunkGroups.along(...)) "
                          "or not at all (per='frame')")
    return isinstance(clip.per, ChunkGroups)


def passes_table(passes) -> dict:
    """``[passes]``: the N-pass schedule."""
    p = passes
    table = {'n': p.n, 'order': p.order, 'stop_tol': float(p.stop_tol), 'sky_merge': p.sky_merge,
             'keep_moments': p.keep_moments,
             'sky': {'outlier_thresh': float(p.sky_clip.sigma), 'subch_clip': _grouped(p.sky_clip)},
             'offset': {'poly_degree': p.offset.degree, 'outlier_thresh': float(p.offset.clip.sigma),
                        'subch_clip': _grouped(p.offset.clip), 'bright_cut': p.offset.bright_cut,
                        'min_pix': p.offset.min_pixels, 'ridge': float(p.offset.ridge)}}
    if p.offset.segments is not None:
        table['offset']['segments'] = [list(s) for s in p.offset.segments]
    if p.init_clip is not None:
        init = {'outlier_thresh': float(p.init_clip.sigma), 'subch_clip': _grouped(p.init_clip)}
        if p.init_clip.ignore_flags is not None:
            init['ignore_list'] = list(p.init_clip.ignore_flags)
        table['init'] = init
    return table


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


def tiling_table(tiles, field, recipe, compute, frames_dir, stage_dir) -> dict:
    """``[tiling]``: the tiles of a tiled calibration."""
    t = tiles
    table = {'ref_shape': list(reference_shape(field)), 'full_reproj_dir': frames_dir,
             'frame_glob': 'exp_*_det_*.h5', 'frame_filter': t.assign, 'halo': t.halo,
             'nvme_subdir': stage_dir, 'stitched_suffix': _suffix(t.stitched_name, recipe.name),
             'line': len(recipe.model.sky) > 1,
             'rss_guardrail': True if compute.memory_guard is None else bool(compute.memory_guard)}
    if t.boxes is not None:
        table['tiles'] = [{'name': k, 'bbox': list(v)} for k, v in t.boxes.items()]
    else:
        table['grid'] = list(t.grid)
        table['overlap_px'] = t.overlap
        table['tile_names'] = None if t.names is None else list(t.names)
    if t.only is not None:
        table['only_tiles'] = list(t.only)
    return table


# --------------------------------------------------------------------------- the run config
def _cache_dir(field, recipe, compute, tiles, passes):
    """The engine's scratch area (``cache_dir``, with its trailing ``/``): ``Compute.scratch``; without
    one, ``<field>/scratch/`` when the run needs a scratch area (the coadd's frame cache, tiles, passes),
    else none."""
    if compute.scratch is not None:
        return compute.scratch.rstrip('/') + '/'
    needs = tiles is not None or passes is not None or (recipe.coadd is not None and compute.cache_frames)
    return os.path.join(field.path, 'scratch') + '/' if needs else None


def _hooks(recipe):
    hooks = {}
    if recipe.fit.raw_frame_hook is not None:
        hooks['pre_cal'] = recipe.fit.raw_frame_hook
    if recipe.fit.frame_hook is not None:
        hooks['post_cal'] = recipe.fit.frame_hook
    if recipe.coadd is not None and recipe.coadd.frame_hook is not None:
        hooks['post_mosaic'] = recipe.coadd.frame_hook
    return hooks


def lower(field, recipe=None, *, task='cal', jobs=None, tiles=None, passes=None, frames=None, cal=None,
          compute=None) -> list[Lowered]:
    """The engine runs of one action: a :class:`Lowered` per group of jobs the instrument runs
    together (SPHEREx: its channel jobs, then its window jobs).

    ``task``: ``"cal"`` (with ``tiles``: tiled; with ``passes``: the N-pass solve) or ``"mosaic"``
    (of the cal ``cal``, default each job's own). ``frames``: None (the field's frames), a number
    (the first ``n``), a directory (its frames, read in place) or a list of frame files.
    """
    recipe = as_recipe(recipe)
    compute = compute or field.compute
    inst = field.instrument
    jobs = tuple(inst.default_jobs()) if jobs is None else ((jobs,) if not isinstance(jobs, (list, tuple))
                                                            else tuple(jobs))
    inst.check_jobs(jobs)
    groups = {}
    for j in jobs:
        groups.setdefault(j.kind, []).append(j)
    run_task = 'npass' if passes is not None else task
    model = recipe.model
    mosaic = 'none' if recipe.coadd is None else ('full' if recipe.coadd.instrument_maps else 'no_wav')
    params = {'line_fisher_threshold': float(recipe.fit.line_fisher_threshold)}
    window = model.spectral_window()
    if window is not None:
        params['spectral_poly_lo'], params['spectral_poly_hi'] = window
    out = []
    for group in groups.values():
        engine_inst, table = inst.engine(group)
        reproj_dir = os.path.join(field.path, 'reprojected')
        frames_dir, n_frames, frame_files = reproj_dir, None, None
        reproj_override = None
        if isinstance(frames, int) and not isinstance(frames, bool):
            n_frames = frames
        elif isinstance(frames, (str, os.PathLike)):
            frames_dir = reproj_override = os.fspath(frames)
        elif frames is not None:
            frame_files = [os.fspath(f) for f in frames]
            dirs = {os.path.dirname(f) for f in frame_files if os.path.dirname(f)}
            if len(dirs) > 1:
                raise ConfigError(f"frames=: the frame files lie in {len(dirs)} directories; give frames of one")
            if dirs and os.path.normpath(dirs.pop()) != os.path.normpath(reproj_dir):
                frames_dir = reproj_override = os.path.dirname(frame_files[0])
        if not compute.stages and reproj_override is None:
            reproj_override = reproj_dir
        if tiles is not None:
            reproj_override = None             # a tiled run stages each tile's frames from frames_dir
        stage_dir = compute.stage_dir
        cfg = RunConfig(
            task=run_task, mode='model', output_dir=os.path.dirname(field.path.rstrip('/')),
            run_name=os.path.basename(field.path.rstrip('/')), resolution_arcsec=field.pixel_scale,
            cache_dir=_cache_dir(field, recipe, compute, tiles, passes),
            suffix=recipe.suffix, oversample=recipe.coadd.oversample if recipe.coadd else 1,
            staging=compute.stage or 'copy', keep_nvme=compute.keep_staged, hdd_io_limit=compute.io_limit,
            apply_n_threads=recipe.numerics.threads, n_frames=n_frames,
            skip_mosaic=recipe.coadd is None,
            wavelength_coadd=recipe.coadd.instrument_maps if recipe.coadd else True,
            reproj_override=reproj_override, cal_override=cal,
            instrument_cfg=table, params=params, calibration=calibration_table(recipe, compute, inst),
            lsqr=lsqr_table(recipe), mosaic=mosaic_table(recipe, compute, inst), model=model.lower(mosaic),
            hooks=_hooks(recipe))
        cfg.instrument = engine_inst
        cfg.frame_files = frame_files
        if stage_dir is not None and tiles is None:
            cfg.stage_dir = stage_dir
        if tiles is not None:
            if frame_files is not None or n_frames is not None:
                raise ConfigError("a tiled calibration takes every frame of its frame directory; frames= may "
                                  "name a directory only")
            cfg.suffix = _suffix(tiles.tile_name, recipe.name)
            cfg.tiling = tiling_table(tiles, field, recipe, compute, frames_dir,
                                      stage_dir or f'reproj_nvme_{os.path.basename(field.path.rstrip("/"))}')
            if stage_dir is not None:
                cfg.tiling['nvme_subdir'] = stage_dir
        if passes is not None:
            cfg.passes = passes_table(passes)
        out.append(Lowered(cfg, tuple(group)))
    return out
