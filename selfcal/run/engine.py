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

Edits here must keep calibration output byte-identical — run the gates
(``selfcal_scripts/gates/run_gates.sh`` and ``run_m13_gate.sh``) before
committing. All numeric choices live in the TOML config and the mode/instrument
objects; this module only sequences them.
"""
from __future__ import annotations

import gc
import glob as glob_module
import os
import shutil
from dataclasses import dataclass, field

import numpy as np

from selfcal.instruments import get_instrument
from selfcal.instruments.base import Instrument
from selfcal.pipeline import pipeline_wrapper

from . import staging
from .config import get_postprocess
from .modes import get_mode


def resolve_instrument(instrument):
    """The engine instrument of a run config: a registered name, or an instrument object as is."""
    return instrument if isinstance(instrument, Instrument) else get_instrument(instrument)


def announce(cfg, kind, path, **info):
    """Tell the run's product book (``cfg.on_product``, set by the Python API) that the product
    ``path`` was written; a TOML run has none."""
    book = getattr(cfg, 'on_product', None)
    if book is not None:
        book(kind, path, **info)


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


def resolve_hook(cfg, inst, which):
    """The per-frame hook ``[hooks].<which>`` (``pre_cal`` / ``post_cal`` /
    ``post_mosaic``): a table ``{name = ..., <params>}`` naming one of the
    instrument's hook factories (``inst.hooks()``) or the runner's named hooks;
    ``None`` when the config has none."""
    spec = (cfg.hooks or {}).get(which)
    if not spec:
        return None
    if callable(spec):                     # a hook object (the Python API)
        return spec
    if isinstance(spec, str):
        spec = {'name': spec}
    params = dict(spec)
    name = params.pop('name', None)
    if not name:
        raise ValueError(f"[hooks].{which} needs a 'name'")
    factories = dict(inst.hooks())
    if name in factories:
        return factories[name](**params)
    fn = get_postprocess(name)
    if params:
        raise ValueError(f"hook {name!r} takes no parameters")
    return fn


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
        inst = resolve_instrument(cfg.instrument)
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
            if mode is not None:
                mode.spec(cfg, inst, geom)        # the recipe, resolved once (fails early on a bad [model])
        return cls(cfg=cfg, inst=inst, mode=mode, pipeline_config=pc,
                   geom=geom, cal_kwargs=calibration_kwargs(cfg), frame_tag=frame_tag)

    # ---- geometry ---------------------------------------------------------
    def jobs(self):
        """The run's jobs: the instrument's expansion of the ``[instrument]`` table.

        A list of :class:`~selfcal.instruments.base.Job`. SPHEREx gives one per selected channel,
        channel group or subchannel window; ``grid`` and ``euclid`` give one. The ``cal`` and
        ``mosaic`` tasks loop over it; ``npass`` uses :meth:`single_job`.
        """
        return self.inst.jobs(self.cfg.instrument_cfg)

    def single_job(self, what):
        """The run's single job, for a task that handles one job only (``npass``).

        Raises ``ValueError`` when the ``[instrument]`` table resolves to any other number of
        jobs; ``what`` names the task in that message (e.g. ``"task 'npass'"``).
        """
        jobs = self.jobs()
        if len(jobs) != 1:
            raise ValueError(f"{what} runs one job per config; [instrument] resolves to "
                             f"{len(jobs)} jobs: {[j.name for j in jobs]}")
        return jobs[0]

    def job_geometry(self, job):
        """The :class:`~selfcal.instruments.base.JobGeometry` of ``job``: valid pixels and weights.

        The instrument builds it from the ``[instrument]`` table and the detector geometry
        ``geom``. :func:`solve_job` uses its ``det_valid_weight``, :func:`mosaic_job` its
        ``grid_valid_weight``.
        """
        return self.inst.job_geometry(self.cfg.instrument_cfg, self.geom, job)

    def aux_maps(self):
        """``(det_aux list, aux_keys)`` the mode asks the solve to carry (``(None, None)`` if none)."""
        aux = self.mode.aux_maps(self.cfg, self.inst, self.geom)
        if not aux:
            return None, None
        return [aux[k] for k in aux], list(aux)

    # ---- naming (the one place) --------------------------------------------
    def stem(self, job, suffix=None):
        """The stem of every product name of ``job``: ``<frame_tag>_<job.name><suffix>``.

        ``suffix`` defaults to the config's top-level ``suffix``. :meth:`cal_file`,
        :meth:`mosaic_file` and :meth:`mosaic_cache_dir` wrap the stem, the N-pass products extend
        it, and the SPHEREx zodi hook reads ``zodi_pred_<stem>.npz``. Example (SPHEREx, detector 4,
        channel 17): ``Detector4_NumSub10_NumCh34_NumCol3_Ch17<suffix>``.
        """
        suffix = self.cfg.suffix if suffix is None else suffix
        return f'{self.frame_tag}_{job.name}{suffix}'

    def cal_file(self, job, suffix=None):
        """The cal file name of ``job``, ``cal_<stem>.h5`` (see :meth:`cal_path`)."""
        return f'cal_{self.stem(job, suffix)}.h5'

    def cal_path(self, job, suffix=None):
        """The full path of ``job``'s cal file: :meth:`cal_file` in the run's cal directory.

        That directory is ``pipeline_config.cal_dir``, ``<output_dir>/<run_name>/calibration``.
        The untiled ``cal`` task skips the solve when this file exists; the ``mosaic`` task reads
        it unless ``cal_override`` is set.
        """
        return os.path.join(self.pipeline_config.cal_dir, self.cal_file(job, suffix))

    def mosaic_file(self, job, suffix=None):
        """The mosaic file name of ``job``, ``mosaic_<stem>.fits`` (see :meth:`mosaic_path`)."""
        return f'mosaic_{self.stem(job, suffix)}.fits'

    def mosaic_path(self, job, suffix=None):
        """The full path of ``job``'s mosaic: :meth:`mosaic_file` in the run's mosaic directory.

        That directory is ``pipeline_config.mos_dir``, ``<output_dir>/<run_name>/mosaic``, where
        :func:`mosaic_job` saves the mosaic.
        """
        return os.path.join(self.pipeline_config.mos_dir, self.mosaic_file(job, suffix))

    def mosaic_cache_dir(self, job, suffix=None):
        """The directory of ``job``'s intermediate mosaic cache: ``<cache_dir>cache_<stem>``.

        The coadd writes it only when ``[mosaic].cache_intermediate`` is true, and
        :func:`mosaic_job` deletes it after saving the mosaic. The two parts are joined as
        strings, so ``cache_dir`` needs its trailing ``/`` (as in the shipped configs) for the
        directory to land inside it.
        """
        return f'{self.cfg.cache_dir}cache_{self.stem(job, suffix)}'

    def tile_cal_file(self, job, tile):
        """The cal file name of a tile: :meth:`cal_file` with ``suffix.format(tile=tile.name)``.

        ``tile`` is a :class:`~selfcal.pipeline.tiled.TileSpec`. The top-level ``suffix`` of a
        tiled config therefore carries a ``{tile}`` placeholder; without one, every tile of the
        job gets the same file name.
        """
        return self.cal_file(job, self.cfg.suffix.format(tile=tile.name))

    def stitched_cal_path(self, job):
        """The path of ``job``'s stitched cal: :meth:`cal_path` with ``[tiling].stitched_suffix``.

        The tiled ``cal`` task writes the Fisher stitch of its tile cals there, and skips the
        stitch when that file exists.
        """
        return self.cal_path(job, self.cfg.tiling['stitched_suffix'])

    def tiling_nvme_dir(self):
        """The staging directory of a tiled run: ``[tiling].nvme_subdir`` under ``cache_dir``.

        The tiled ``cal`` task copies each tile's frames there before solving the tile, and the
        N-pass SKY passes of a tiled run stage into it too. The engine never deletes it, and
        staging keeps complete copies already present, so frames copied once are reused. It must
        be a directory the pipeline made (:func:`~selfcal.run.staging.claim`).
        """
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
        """The one cal that holds the run's sky: the stitched cal, else the only cal, else None."""
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
    """Undo :func:`stage_run`: delete the staged copy of the frames unless the config keeps it.

    With ``reproj_override`` it does nothing (the frames were read in place). Otherwise
    :func:`~selfcal.run.staging.cleanup_nvme` keeps ``frame_dir`` when
    ``staging = "reuse"`` (another run staged it) or ``keep_nvme = true``, and deletes it
    otherwise. The untiled ``cal`` task and the ``mosaic`` task call it after their last job.
    """
    if not ctx.cfg.reproj_override:
        staging.cleanup_nvme(ctx.cfg, frame_dir)


def frame_list(frame_dir, n_frames=None):
    """The sorted ``*.h5`` frames of a directory, optionally the first ``n``."""
    files = sorted(glob_module.glob(os.path.join(frame_dir, '*.h5')))
    return files[:n_frames] if n_frames else files


def clip_groups(ctx, groups):
    """The ``setup_lsqr`` keywords of an outlier clip over chunk groups (the Python API's
    ``Clip(per=...)``): ``{"along": axis}``, ``{"chunk": True}`` or ``{"mapping": [...]}``, on the
    primary chunk map. Along the primary map's spectral axis of an instrument with a wavelength map,
    the clip bins the wavelength between the axis values' wavelengths (the N-pass passes' clip,
    ``subch_clip``); otherwise each pixel is judged within the group of its dominant chunk."""
    cm, geom = ctx.geom.chunk_map, ctx.geom
    if groups.get('map') not in (None, cm.name):
        raise ValueError(f"clip groups {groups}: groups of the primary chunk map ({cm.name!r}) only")
    axis = groups.get('along')
    if axis is not None and axis == cm.spectral_axis and geom.wavelength_key:
        return {'outlier_group_edges': ctx.mode.clip_group_edges(ctx.cfg, ctx.inst, geom)}
    n = cm.n_chunks
    if axis is not None:
        names = list(cm.axes.names) if cm.axes is not None else []
        if axis not in names:
            raise ValueError(f"clip groups along {axis!r}: the primary chunk map's axes are {names}")
        mapping = np.asarray(cm.axes[axis].of_chunk)[:n]
    elif groups.get('chunk'):
        mapping = np.arange(n)
    else:
        mapping = np.asarray(groups['mapping'])
        if mapping.shape != (n,):
            raise ValueError(f"clip groups: a mapping of {mapping.shape[0]} chunks; the primary chunk map has {n}")
    return {'outlier_chunk_groups': mapping.astype(np.int64)}


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
    # The model's data variables beyond the instrument's detector maps (frame
    # values, sky maps, stored layers, functions) — None for the historical recipes.
    variables = mode.build_variables(cfg, inst, geom, cc.reproj_list, ref_shape=cc.ref_shape,
                                     ref_wcs=cc.ref_wcs)
    offset_model = mode.build_offset_model(cfg, inst, geom, jobgeom, job, n_frames, frames=cc.reproj_list,
                                           variables=variables)
    sky_model = mode.build_sky_model(cfg, inst, geom)
    det_aux, aux_keys = ctx.aux_maps()
    cal_kwargs = dict(ctx.cal_kwargs)
    cal_kwargs.update(mode.setup_kwargs(cfg, inst, geom))     # the model's extra solver options (if any)
    if variables is not None:
        cal_kwargs['variables'] = variables
    weight_function = mode.build_weight(cfg, inst, geom)
    if weight_function is not None:
        cal_kwargs['weight_function'] = weight_function
    priors = mode.build_priors(cfg, inst, geom, variables, n_frames)
    if priors:
        cal_kwargs['priors'] = priors
    groups = cal_kwargs.pop('outlier_groups', None)
    if groups is not None:
        cal_kwargs.update(clip_groups(ctx, groups))
    # The grouped clip bins the instrument's wavelength map unless the config
    # names another data variable ([calibration].outlier_group_variable).
    if not (cal_kwargs.get('outlier_group_variable') or cal_kwargs.get('outlier_aux_key')):
        cal_kwargs['outlier_group_variable'] = geom.wavelength_key
    pre = resolve_hook(cfg, inst, 'pre_cal')
    post = resolve_hook(cfg, inst, 'post_cal')
    if pre is not None:
        cal_kwargs['preprocess_func'] = pre
    if post is not None:
        if cal_kwargs.get('postprocess_func') is not None:
            raise ValueError("give either `postprocess` or [hooks].post_cal, not both")
        cal_kwargs['postprocess_func'] = post
    checkpoint('pre-setup_lsqr')
    cc.setup_lsqr(
        offset_model=offset_model,
        grid_valid_weight=jobgeom.det_valid_weight,
        oversample_factor=1,
        sky_model=sky_model,
        det_aux=det_aux,
        aux_keys=aux_keys,
        batch_spill_dir=cfg.cache_dir,
        **cal_kwargs)
    checkpoint('post-setup_lsqr')
    # List-pop hand-off: keeping a plain `x0` local would pin the full-layout
    # f64 vector for the entire solve (see Calibrator.apply_lsqr).
    _x0_owned = [mode.x0(cfg, cc)]
    checkpoint('pre-apply_lsqr')
    # [lsqr] may override the float32 solve and the thread count.
    lsqr_kwargs = dict(use_float32=True, n_threads=cfg.apply_n_threads)
    lsqr_kwargs.update(cfg.lsqr)
    cc.apply_lsqr(x0=_x0_owned.pop(), **lsqr_kwargs)
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
    # A cal solved on another grid (cal_override): frames it lists that have no
    # reprojected file HERE are dropped, with their rows of every per-frame array
    # (the solution is per frame / detector-plane, so it transfers).
    keep = np.array([os.path.exists(f) for f in mm.reproj_list])
    if not keep.all():
        n_all = len(mm.reproj_list)
        mm.reproj_list = [f for f, k in zip(mm.reproj_list, keep) if k]
        for attr in ('offsets', 'offset_coverages', 'offset_coverage_fracs'):
            arrs = getattr(mm, attr, None)
            if arrs:
                setattr(mm, attr, [a[keep] if (hasattr(a, 'shape') and len(a) == n_all) else a for a in arrs])
        print(f"Dropped {int((~keep).sum())} cal frames with no reprojected file in {frame_dir} "
              f"({int(keep.sum())} remain)")
    mosaic_kwargs = dict(cfg.mosaic)
    post = resolve_hook(cfg, inst, 'post_mosaic')
    # Offset terms whose offsets are coefficients of known functions of data
    # variables are not constant per chunk: the mosaic subtracts them at every
    # observation (BasisOffsetSubtractor), and hands the chunk path zeros for
    # them (plus the per-frame scalar on map 0, which the cal folds in there).
    basis_hook = _basis_offset_hook(ctx, mm, cal_path, chunk_maps)
    # The model's observation weight (a row weight w) weights the coadd by w², as
    # in the solve.
    weight_hook = _observation_weight_hook(ctx, mm)
    hooks = [h for h in (basis_hook, post, weight_hook) if h is not None]
    if len(hooks) > 1:
        from selfcal.pipeline.model_eval import ComposedHook
        post = ComposedHook(hooks)
    elif hooks:
        post = hooks[0]
    if post is not None:
        mosaic_kwargs['postprocess_func'] = post
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
        **mosaic_kwargs)
    if want_wav:
        inst.finalize_mosaic(geom, mm, maps, cfg.mosaic['sigma'])
    mos_path = mm.save_mosaic(mos_file=mos_file, overwrite=True)
    del mm, maps
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    return mos_path


def _basis_offset_hook(ctx, mm, cal_path, chunk_maps):
    """The mosaic's per-observation subtraction of the offset terms that carry
    known functions of data variables (``coefficient`` / ``basis``), or None.
    Rewrites those maps' entries of ``mm.offsets`` so the chunk path subtracts
    only the per-frame scalar there."""
    cfg, inst, mode, geom = ctx.cfg, ctx.inst, ctx.mode, ctx.geom
    spec = mode.spec(cfg, inst, geom)
    offset_model = spec.build_offset_model(geom, len(mm.reproj_list), log=lambda *a, **k: None,
                                           catalog=inst.coefficient_catalog(),
                                           frame_groups=_all_frame_groups(inst, cfg, mm.reproj_list),
                                           frame_variables=mode.frame_variable_names(cfg, inst))
    terms = [(m, b.basis, b.chunk_map) for m, b in enumerate(offset_model.blocks) if b.basis is not None]
    if not terms:
        return None
    from selfcal.io.calfile import CalFile
    from selfcal.pipeline.model_eval import BasisOffsetSubtractor
    keep = [os.path.basename(p) for p in mm.reproj_list]
    with CalFile(cal_path) as cal:
        scalar = cal.frame_scalar
        order = {os.path.basename(p): i for i, p in enumerate(cal.reproj_list)}
    rows = np.array([order[k] for k in keep], dtype=np.int64)
    for m, _, cm in terms:
        n_chunks = int(np.asarray(chunk_maps[m]).max()) + 1
        zeros = np.zeros((len(rows), n_chunks), dtype=np.float64)
        if m == 0 and scalar is not None:
            zeros += np.asarray(scalar)[rows][:, None]
        mm.offsets[m] = zeros
        mm.offset_coverage_fracs[m] = np.ones_like(zeros)
    # The data variables over the mosaic's frames (the cal's, minus any without
    # a reprojected file here); coefficients are looked up by the cal's order.
    return BasisOffsetSubtractor(cal_path, terms, variables=_mosaic_variables(ctx, mm), oversample_factor=1,
                                 frame_names=list(mm.reproj_list))


def _mosaic_variables(ctx, mm):
    """The data variables over the mosaic's frames, with the instrument's detector
    maps (the solve passes those separately), for the mosaic-side evaluators."""
    from selfcal.models.variables import VariableSet
    cfg, inst, mode, geom = ctx.cfg, ctx.inst, ctx.mode, ctx.geom
    vs = mode.build_variables(cfg, inst, geom, list(mm.reproj_list), ref_shape=mm.ref_shape,
                              ref_wcs=mm.ref_wcs) or VariableSet()
    extra = {k: v for k, v in (geom.aux or {}).items() if k not in vs.detector}
    return vs.merged(detector=extra) if extra else vs


def _observation_weight_hook(ctx, mm):
    """The mosaic's per-observation weight from the model's ``weight``, or None."""
    cfg, inst, mode, geom = ctx.cfg, ctx.inst, ctx.mode, ctx.geom
    weight = mode.build_weight(cfg, inst, geom)
    if weight is None:
        return None
    from selfcal.pipeline.model_eval import ObservationWeight
    return ObservationWeight(weight, variables=_mosaic_variables(ctx, mm), frame_names=list(mm.reproj_list),
                             det_shape=geom.shape)


def _all_frame_groups(inst, cfg, frames):
    groups = dict(inst.frame_groups(frames))
    for k, v in inst.frame_variables(frames, cfg.instrument_cfg).items():
        groups.setdefault(k, v)
    return groups


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
    """Every frame of the field (every detector), in (exposure, detector) order, from
    ``full_reproj_dir``; ``[tiling].frame_glob`` (default ``exp_*_det_*.h5``) narrows it."""
    files = glob_module.glob(os.path.join(tiling['full_reproj_dir'],
                                          tiling.get('frame_glob', 'exp_*_det_*.h5')))
    from selfcal.io.reproj import parse_reproj_basename
    return sorted(files, key=parse_reproj_basename)


def tile_assignment(tiling, ref_shape):
    """``(TiledCalibration, tiles, only_tiles, assignment)`` for a ``[tiling]`` table."""
    from selfcal.pipeline.tiled import TiledCalibration
    tiles, only_tiles = resolve_tiles(tiling, ref_shape)
    files = tiling_frames(tiling)
    tiled = TiledCalibration(files, tiles, frame_filter=tiling.get('frame_filter', 'center'),
                             halo=tiling.get('halo', 0))
    return tiled, tiles, only_tiles, tiled.assign_frames()
