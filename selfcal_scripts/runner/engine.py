"""The run engine's primitives — mode- and instrument-agnostic.

A run resolves its config ONCE into a :class:`RunContext` (instrument, mode,
output layout, detector geometry, product names) and then applies two
primitives to it:

* :func:`solve_job` — one joint LSQR solve over a frame list → one cal file
  (the block every task shares: plain cal, each tile of a tiled cal, the INIT
  pass of an N-pass run);
* :func:`mosaic_job` — one coadd of a cal file → one mosaic FITS.

Tiling (``[tiling]``) and passes (``[passes]``) are options layered on top of
these in :mod:`.pipelines` / :mod:`.npass`; they never re-implement the solve.
Every product name (cal, mosaic, mosaic cache, tile cals, stitched cal, N-pass
products) comes from :class:`RunContext`, so the pieces of a run can never
disagree about a file name.

The engine reads no ``[instrument]`` key: the instrument turns that table into
typed geometry (:class:`~selfcal.instruments.base.DetectorGeometry`,
``JobGeometry``), the mode turns the geometry into the offset/sky recipe.

Edits here must keep calibration output byte-identical — run
``workspace/unify/scripts/run_gates.sh`` (or the equivalent gate set) before
committing. All numeric choices live in the TOML config and the mode/instrument
objects; this module only sequences them.
"""
from __future__ import annotations

import gc
import glob as glob_module
import os
import shutil
from dataclasses import dataclass, field

from selfcal.instruments import get_instrument
from selfcal.pipeline import pipeline_wrapper

from . import staging
from .config import get_postprocess
from .modes import get_mode


def check_requires(mode, inst):
    """A mode declares the instrument capabilities it needs; fail early."""
    missing = [c for c in mode.requires if c not in inst.capabilities]
    if missing:
        raise ValueError(
            f"mode {mode.name!r} requires instrument capabilities {missing} "
            f"that {inst.name!r} does not provide (has {sorted(inst.capabilities)})")


def calibration_kwargs(cfg):
    """The [calibration] table + the resolved (named) postprocess func."""
    kw = dict(cfg.calibration)
    kw['postprocess_func'] = get_postprocess(cfg.postprocess)
    return kw


# ---------------------------------------------------------------------------
# Run context: resolved once per run; owns the product names
# ---------------------------------------------------------------------------
@dataclass
class RunContext:
    """Everything a task resolves once from its config.

    ``geom`` is the instrument's detector-level geometry (chunk maps with their
    axes, aux maps) built once per run; per-job geometry comes from
    :meth:`job_geometry`. The ``stem`` / ``cal_*`` / ``mosaic_*`` methods are
    the ONE place product names are formed: ``<frame_tag>_<job.name><suffix>``.
    """
    cfg: object
    inst: object
    mode: object
    pipeline_config: pipeline_wrapper.PipelineConfig
    geom: object
    cal_kwargs: dict
    frame_tag: str

    @classmethod
    def build(cls, cfg, *, need_mode=True, need_geometry=True):
        """Resolve the config. ``need_geometry=False`` (reprojection) skips the
        detector geometry and the frame tag, which need calibration data the
        reprojection stage does not."""
        inst = get_instrument(cfg.instrument)
        mode = None
        if need_mode:
            mode = get_mode(cfg.mode)
            check_requires(mode, inst)
        pc = pipeline_wrapper.PipelineConfig(
            output_dir=cfg.output_dir,
            run_name=cfg.resolved_run_name(),
            resolution_arcsec=cfg.resolution_arcsec)
        geom = frame_tag = None
        if need_geometry:
            geom = inst.detector_geometry(cfg.instrument_cfg, cfg.oversample)
            frame_tag = inst.frame_tag(cfg.instrument_cfg)
        return cls(cfg=cfg, inst=inst, mode=mode, pipeline_config=pc,
                   geom=geom, cal_kwargs=calibration_kwargs(cfg), frame_tag=frame_tag)

    # ---- geometry ---------------------------------------------------------
    def jobs(self):
        return self.inst.jobs(self.cfg.instrument_cfg)

    def single_job(self, what):
        jobs = self.jobs()
        if len(jobs) != 1:
            raise ValueError(f"{what} runs one job per config; [instrument] resolves to "
                             f"{len(jobs)} jobs: {[j.name for j in jobs]}")
        return jobs[0]

    def job_geometry(self, job):
        return self.inst.job_geometry(self.cfg.instrument_cfg, self.geom, job)

    def aux_maps(self):
        """``(det_aux list, aux_keys)`` the mode asks the solve to carry (``(None, None)`` if none)."""
        aux = self.mode.aux_maps(self.cfg, self.inst, self.geom)
        if not aux:
            return None, None
        return [aux[k] for k in aux], list(aux)

    # ---- naming (the one place) --------------------------------------------
    def stem(self, job, suffix=None):
        suffix = self.cfg.suffix if suffix is None else suffix
        return f'{self.frame_tag}_{job.name}{suffix}'

    def cal_file(self, job, suffix=None):
        return f'cal_{self.stem(job, suffix)}.h5'

    def cal_path(self, job, suffix=None):
        return os.path.join(self.pipeline_config.cal_dir, self.cal_file(job, suffix))

    def mosaic_file(self, job, suffix=None):
        return f'mosaic_{self.stem(job, suffix)}.fits'

    def mosaic_path(self, job, suffix=None):
        return os.path.join(self.pipeline_config.mos_dir, self.mosaic_file(job, suffix))

    def mosaic_cache_dir(self, job, suffix=None):
        return f'{self.cfg.cache_dir}cache_{self.stem(job, suffix)}'

    def tile_cal_file(self, job, tile):
        return self.cal_file(job, self.cfg.suffix.format(tile=tile.name))

    def stitched_cal_path(self, job):
        return self.cal_path(job, self.cfg.tiling['stitched_suffix'])

    def tiling_nvme_dir(self):
        return os.path.join(self.cfg.cache_dir, self.cfg.tiling['nvme_subdir'])


@dataclass
class CalResult:
    """What a calibration task returns. ``cal_paths`` are the per-job cals of a
    plain run or the per-tile cals of a tiled run; ``stitched`` is the stitched
    cal of a full tiled run; ``sky_path`` is the one cal that holds the field's
    sky (the stitched cal, or the single cal)."""
    cal_paths: list
    mosaic_paths: list = field(default_factory=list)
    tiles: dict | None = None        # {tile_name: cal_path} (tiled runs)
    stitched: str | None = None      # stitched cal (full tiled runs)
    assignment: dict | None = None   # {tile_name: (files, idx)} (tiled runs)

    @property
    def sky_path(self):
        if self.stitched:
            return self.stitched
        return self.cal_paths[0] if len(self.cal_paths) == 1 else None


# ---------------------------------------------------------------------------
# Staging of the whole run's frames (plain cal / mosaic)
# ---------------------------------------------------------------------------
def stage_run(ctx):
    """The frame directory a plain run reads from: the ``reproj_override`` dir as
    is (regression configs, manual re-runs; no staging, no cleanup) or the
    per-run NVMe copy of the reprojected dir."""
    cfg = ctx.cfg
    if cfg.reproj_override:
        staging.set_hdd_io_limit(None)
        return cfg.reproj_override
    if not cfg.cache_dir:
        raise ValueError("cache_dir is required (the staging area for the reprojected frames), "
                         "or give reproj_override")
    return staging.prepare_nvme(cfg, ctx.pipeline_config.reproj_dir, ctx.pipeline_config.run_name)


def unstage_run(ctx, frame_dir):
    if not ctx.cfg.reproj_override:
        staging.cleanup_nvme(ctx.cfg, frame_dir)


def frame_list(frame_dir, n_frames=None):
    """The sorted ``*.h5`` frames of a directory, optionally the first ``n``."""
    files = sorted(glob_module.glob(os.path.join(frame_dir, '*.h5')))
    return files[:n_frames] if n_frames else files


# ---------------------------------------------------------------------------
# Primitive 1: one joint solve -> one cal file
# ---------------------------------------------------------------------------
def solve_job(ctx, job, jobgeom, *, frame_dir, cal_file, hdd_reproj_dir,
              frames=None, checkpoint=None):
    """setup_lsqr + apply_lsqr + save for one job over one frame list.

    ``frames`` (paths under ``frame_dir``) overrides the directory glob. The cal
    records the frames under ``hdd_reproj_dir`` — their permanent location — so
    it stays valid after the staged copy is cleaned up. ``checkpoint(label)`` is
    an optional progress/RSS hook called around the two heavy steps.
    """
    cfg, inst, mode, geom = ctx.cfg, ctx.inst, ctx.mode, ctx.geom
    checkpoint = checkpoint or (lambda label: None)
    cc = pipeline_wrapper.Calibrator(ctx.pipeline_config, reproj_dir=frame_dir)
    if frames is not None:
        cc.reproj_list = list(frames)
    n_frames = len(cc.reproj_list)
    offset_model = mode.build_offset_model(cfg, inst, geom, jobgeom, job, n_frames)
    sky_model = mode.build_sky_model(cfg, inst, geom)
    det_aux, aux_keys = ctx.aux_maps()
    checkpoint('pre-setup_lsqr')
    cc.setup_lsqr(
        offset_model=offset_model,
        grid_valid_weight=jobgeom.det_valid_weight,
        oversample_factor=1,
        sky_model=sky_model,
        det_aux=det_aux,
        aux_keys=aux_keys,
        outlier_aux_key=geom.wavelength_key,
        batch_spill_dir=cfg.cache_dir,
        **ctx.cal_kwargs)
    checkpoint('post-setup_lsqr')
    # List-pop hand-off: keeping a plain `x0` local would pin the full-layout
    # f64 vector for the entire solve (see Calibrator.apply_lsqr).
    _x0_owned = [mode.x0(cfg, cc)]
    checkpoint('pre-apply_lsqr')
    cc.apply_lsqr(x0=_x0_owned.pop(), use_float32=True, n_threads=cfg.apply_n_threads, **cfg.lsqr)
    checkpoint('post-apply_lsqr')
    mode.configure(cfg, cc)
    # Save with the permanent (HDD) paths so the cal stays valid after cleanup.
    staged_list = cc.reproj_list
    cc.reproj_list = staging.remap_to_nvme(staged_list, hdd_reproj_dir)
    cal_path = cc.save_calibration(cal_file=cal_file)
    cc.reproj_list = staged_list
    del cc
    gc.collect()
    return cal_path


# ---------------------------------------------------------------------------
# Primitive 2: one cal -> one mosaic
# ---------------------------------------------------------------------------
def mosaic_job(ctx, job, jobgeom, *, cal_path, frame_dir, mos_file, cache_dir):
    """Coadd the frames of ``cal_path`` (read from ``frame_dir``) into
    ``mos_file``; the instrument's aux coadds (SPHEREx: the wavelength maps)
    ride along when the mode asks for a full mosaic and the instrument has them."""
    cfg, inst, mode, geom = ctx.cfg, ctx.inst, ctx.mode, ctx.geom
    chunk_maps, det_offset_funcs = mode.mosaic_geometry(cfg, inst, geom, jobgeom)
    mm = pipeline_wrapper.Mosaicker(ctx.pipeline_config, reproj_dir=frame_dir,
                                    unit=inst.data_unit(cfg.instrument_cfg))
    mm.load_calibration(cal_path=cal_path)
    mm.reproj_list = staging.remap_to_nvme(mm.reproj_list, frame_dir)
    # `wavelength_coadd` (default true) selects the instrument's aux coadds (the
    # wav_mean/wav_std maps). They are sigma-clipped against the std map, so
    # they need make_std_map; with apply_sigma_clipping they are coadded inside
    # the sigma-clip pass (no extra pass, no cache needed), otherwise the
    # standalone coadd runs over the intermediate cache. Say so here rather
    # than fail deep inside the coadd.
    wav_maps = None
    aux_coadds = inst.aux_coadds(geom) if mode.mosaic_mode == 'full' and cfg.wavelength_coadd else None
    want_wav = aux_coadds is not None
    if want_wav:
        if not cfg.mosaic.get('make_std_map', False):
            raise ValueError(
                "wavelength_coadd = true needs [mosaic] make_std_map = true "
                "(the wavelength coaddition sigma-clips against the std map). Set it "
                "true, or set wavelength_coadd = false to build the mosaic "
                "without the wav_mean/wav_std maps.")
        if cfg.mosaic.get('apply_sigma_clipping', False):
            wav_maps = aux_coadds
        elif not cfg.mosaic.get('cache_intermediate', False):
            raise ValueError(
                "wavelength_coadd = true needs [mosaic] apply_sigma_clipping = "
                "true (coadded in the sigma-clip pass) or cache_intermediate = "
                "true (standalone coadd over the cache). Set one, or set "
                "wavelength_coadd = false.")
    maps = mm.make_mosaic(
        chunk_maps=chunk_maps,
        grid_valid_weight=jobgeom.grid_valid_weight,
        oversample_factor=cfg.oversample,
        det_offset_funcs=det_offset_funcs,
        cache_dir=cache_dir,
        wav_maps=wav_maps,
        **cfg.mosaic)
    if want_wav:
        inst.finalize_mosaic(geom, mm, maps, cfg.mosaic['sigma'])
    mos_path = mm.save_mosaic(mos_file=mos_file, overwrite=True)
    del mm, maps
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    return mos_path


# ---------------------------------------------------------------------------
# Tiling helpers shared by the tiled cal and the N-pass scheduler
# ---------------------------------------------------------------------------
def resolve_tiles(tiling, ref_shape):
    """The tile list of a ``[tiling]`` table: an explicit ``tiles`` list (arbitrary,
    possibly overlapping bboxes) or a uniform ``grid`` with ``overlap_px``; then
    the optional ``only_tiles`` restriction (the full grid is built first so every
    bbox is right). Returns ``(tiles, only_tiles)``."""
    from selfcal.pipeline.tiled import make_tile_grid, TileSpec
    if tiling.get('tiles'):
        tiles = [TileSpec(name=spec['name'], bbox=tuple(spec['bbox'])) for spec in tiling['tiles']]
    else:
        tiles = make_tile_grid(ref_shape, tiling['grid'][0], tiling['grid'][1],
                               overlap_px=tiling['overlap_px'], names=tiling['tile_names'])
    only_tiles = tiling.get('only_tiles')
    if only_tiles:
        all_names = [tile.name for tile in tiles]
        tiles = [tile for tile in tiles if tile.name in only_tiles]
        if not tiles:
            raise ValueError(f"only_tiles={only_tiles} matched no tile in {all_names}")
    return tiles, only_tiles


def tiling_frames(tiling):
    """Every frame of the field, in exposure order, from ``full_reproj_dir``."""
    files = glob_module.glob(os.path.join(tiling['full_reproj_dir'],
                                          tiling.get('frame_glob', 'exp_*_det_00.h5')))
    return sorted(files, key=lambda p: int(os.path.basename(p).split('_')[1]))


def tile_assignment(tiling, ref_shape):
    """``(TiledCalibration, tiles, only_tiles, assignment)`` for a ``[tiling]`` table."""
    from selfcal.pipeline.tiled import TiledCalibration
    tiles, only_tiles = resolve_tiles(tiling, ref_shape)
    files = tiling_frames(tiling)
    tiled = TiledCalibration(files, tiles, frame_filter=tiling.get('frame_filter', 'center'),
                             halo=tiling.get('halo', 0))
    return tiled, tiles, only_tiles, tiled.assign_frames()
