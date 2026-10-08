"""The run engine's primitives — instrument-agnostic.

An engine run (a :class:`~selfcal.run.runspec.RunSpec`) is resolved ONCE into a
:class:`RunContext` (the instrument, the output layout, the detector geometry, the model and
every product name) and then applies two primitives to it:

* :func:`solve_job` — one joint LSQR solve over a frame list → one cal file (the block every
  task shares: plain cal, each tile of a tiled cal, the INIT pass of an N-pass run);
* :func:`mosaic_job` — one coadd of a cal file → one mosaic FITS.

Tiling and passes are options layered on top of these in :mod:`.pipelines` / :mod:`.npass`;
they never re-implement the solve. Every product name (cal, mosaic, mosaic cache, tile cals,
stitched cal, N-pass products) comes from :class:`RunContext`, so the pieces of a run can never
disagree about a file name.

The engine talks to the instrument through its contract only
(:class:`~selfcal.instruments.contract.Instrument`: the geometry, a job's geometry, the product
tag, the frame variables and the optional renderers, maps and catalogue) and lowers the model
(:class:`~selfcal.models.spec.ModelSpec`) to the solver's objects; it never names a telescope.

Edits here must keep calibration output byte-identical — run the gates
(``selfcal_scripts/gates/run_gates.sh`` and ``run_m13_gate.sh``) before committing.
"""
from __future__ import annotations

import collections
import copy
import gc
import glob as glob_module
import os
import shutil
from dataclasses import dataclass, field, replace

import numpy as np

from selfcal.pipeline import pipeline_wrapper

from . import staging


def announce(spec, kind, path, **info):
    """Tell the run's product book (``spec.on_product``, set by the action) that the product
    ``path`` was written."""
    if spec.on_product is not None:
        spec.on_product(kind, path, **info)


def _quiet(*args, **kwargs):
    """A log that says nothing."""


# ---------------------------------------------------------------------------
# The detector geometry, built once per action (and kept when the instrument says it may be)
# ---------------------------------------------------------------------------
#: How many detector geometries the process keeps between actions (:func:`instrument_geometry`).
GEOMETRIES_KEPT = 2
#: The environment variables an instrument's data files are found through.
GEOMETRY_VARIABLES = ('SELFCAL_SPHEREX_CALIB_DIR', 'SELFCAL_LVF_PARAMS_DIR', 'SELFCAL_SPHEREX_CHANNEL_FILE')
_GEOMETRIES = collections.OrderedDict()


def _file_state(path):
    try:
        st = os.stat(path)
    except OSError:
        return path, None, None
    return path, st.st_size, st.st_mtime_ns


def geometry_is_pure(inst):
    """Whether the geometry of ``inst`` may be kept between actions: its
    :attr:`~selfcal.instruments.contract.Instrument.geometry_is_pure` is true and is declared by the
    class whose ``geometry()`` it runs, or a subclass of it (a subclass that overrides the geometry of
    a pure instrument is not pure unless it declares so again)."""
    declared = defined = None
    for i, klass in enumerate(type(inst).__mro__):
        if declared is None and 'geometry_is_pure' in vars(klass):
            declared = i
        if defined is None and 'geometry' in vars(klass):
            defined = i
    return bool(getattr(inst, 'geometry_is_pure', False)) and None not in (declared, defined) and declared <= defined


def instrument_geometry(inst, oversample):
    """The detector geometry of ``inst`` sampled ``oversample`` times per pixel
    (:meth:`~selfcal.instruments.contract.Instrument.geometry`).

    An action builds it once: its plan, its runs and an N-pass INIT share it through their
    contexts. The geometry of a pure instrument (:func:`geometry_is_pure`: the built-in ones) is
    also kept between actions: the last :data:`GEOMETRIES_KEPT`, each under the instrument's
    settings, the oversampling, the path, size and time of each file its geometry reads
    (``geometry_files()``) and the :data:`GEOMETRY_VARIABLES`; it is built again when any of them
    changed, and each call returns a copy of its own, so no run sees another's changes. Any other
    instrument's geometry is built at each call and returned as built (never kept, never copied)."""
    if not geometry_is_pure(inst):
        return inst.geometry(oversample)
    from ..config.base import encode
    from .products import fingerprint
    key = (type(inst), fingerprint(encode(inst)), int(oversample),
           tuple(_file_state(p) for p in inst.geometry_files()),
           tuple(os.environ.get(v) for v in GEOMETRY_VARIABLES))
    if key in _GEOMETRIES:
        _GEOMETRIES.move_to_end(key)
    else:
        _GEOMETRIES[key] = inst.geometry(oversample)
        while len(_GEOMETRIES) > GEOMETRIES_KEPT:
            _GEOMETRIES.popitem(last=False)
    return copy.deepcopy(_GEOMETRIES[key])


# ---------------------------------------------------------------------------
# Run context: resolved once per run; lowers the model; owns the product names
# ---------------------------------------------------------------------------
@dataclass
class RunContext:
    """Everything an engine run resolves once from its :class:`~selfcal.run.runspec.RunSpec`.

    ``geom`` is the instrument's detector geometry (chunk maps with their axes, detector maps),
    ``model`` the run's :class:`~selfcal.models.spec.ModelSpec`, checked against them, and
    ``sky_damping`` each sky term's damping (every pass of the run uses it); per-job geometry
    comes from :meth:`job_geometry`. The methods below lower the model to the solver's objects,
    and the ``stem`` / ``cal_*`` / ``mosaic_*`` methods are the ONE place product names are
    formed: ``<frame_tag>_<job.name><suffix>``.
    """
    spec: object
    inst: object
    pipeline_config: pipeline_wrapper.PipelineConfig
    frame_tag: str
    geom: object = None
    model: object = None
    sky_damping: list = field(default_factory=list)

    @classmethod
    def build(cls, spec, *, need_geometry=True, geom=None):
        """Resolve ``spec``. ``need_geometry=False`` (a reprojection; the product names) skips the
        detector geometry and the model, which need calibration data the reprojection does not.
        ``geom``: the detector geometry when the action has built it already (for another of its
        job groups, which share the instrument and the oversampling)."""
        from selfcal.models.spec import ModelSpec
        inst = spec.instrument
        pc = pipeline_wrapper.PipelineConfig(output_dir=spec.output_dir, run_name=spec.run_name,
                                             resolution_arcsec=spec.resolution_arcsec)
        ctx = cls(spec=spec, inst=inst, pipeline_config=pc, frame_tag=inst.product_tag)
        if need_geometry:
            ctx.geom = geom if geom is not None else instrument_geometry(inst, spec.oversample)
            ctx.model = ModelSpec.from_config(spec.model)
            # every data variable, chunk map and axis, catalogue entry and prior term exists
            ctx.model.check(ctx.geom, ctx.catalog(), frame_variables=ctx.frame_variable_names())
            ctx.sky_damping = ctx.sky_model(log=_quiet).damp_weights(spec.setup['damp_weight'])
        return ctx

    def derive(self, spec):
        """The context of ``spec``, a run of the same instrument and model whose own settings differ
        (the N-pass INIT): the geometry and the model are this context's."""
        return replace(self, spec=spec)

    # ---- the jobs and their geometry -----------------------------------------------------
    def jobs(self):
        """The run's jobs (:class:`~selfcal.instruments.contract.Job`): SPHEREx a channel, a group of
        channels or a subchannel window each; a camera and Euclid one."""
        return tuple(self.spec.jobs)

    def single_job(self, what):
        """The run's single job, for a task that handles one job only (the N-pass solve); raises
        ``ValueError`` (``what`` names the task) when the run has another number of jobs."""
        jobs = self.jobs()
        if len(jobs) != 1:
            raise ValueError(f"{what} runs one job; the run has {len(jobs)}: {[j.name for j in jobs]}")
        return jobs[0]

    def job_geometry(self, job):
        """The :class:`~selfcal.instruments.base.JobGeometry` of ``job``: its valid pixels and
        weights (:func:`solve_job` uses ``det_valid_weight``, :func:`mosaic_job`
        ``grid_valid_weight``)."""
        return self.inst.job_geometry(self.geom, job)

    @property
    def unit(self):
        """The unit of the calibrated data (the mosaic's ``BUNIT``)."""
        return str(getattr(self.inst, 'unit', '') or '')

    # ---- the model, lowered to the solver's objects --------------------------------------
    def catalog(self):
        """The instrument's named sky coefficients (``sc.catalog(name)``)."""
        return dict(self.inst.coefficient_catalog())

    def frame_variable_names(self):
        """The names of the instrument's frame variables (``exposure``, ``detector``, ...)."""
        return tuple(self.inst.frame_variable_names())

    def frame_groups(self, frames, variables=None):
        """``{name: per-frame values}`` a grouped offset term can share an offset over: the
        instrument's frame groups and frame variables, then the model's per-frame variables
        (``variables``, a :class:`~selfcal.models.variables.VariableSet` over ``frames``)."""
        groups = dict(self.inst.frame_groups(frames))
        for k, v in self.inst.frame_variables(frames).items():
            groups.setdefault(k, v)
        if variables is not None:
            for k, v in variables.frame.items():
                groups.setdefault(k, v)
        return groups

    def offset_model(self, frames, variables=None, log=print):
        """The solver's :class:`~selfcal.models.offset_model.OffsetModel` over ``frames``, for the
        solve and the mosaic alike: a grouped term groups the frames by :meth:`frame_groups`."""
        grouped = any(t.kind == 'grouped' for t in self.model.offset)
        return self.model.build_offset_model(self.geom, len(frames),
                                             frame_groups=self.frame_groups(frames, variables) if grouped else None,
                                             log=log, catalog=self.catalog(),
                                             frame_variables=self.frame_variable_names())

    def sky_model(self, log=print):
        """The solver's :class:`~selfcal.models.sky_model.SkyModel`: one component per sky term,
        coefficients resolved against the data variables and the instrument's catalogue."""
        return self.model.build_sky_model(self.geom, self.catalog(), log=log,
                                          frame_variables=self.frame_variable_names())

    def aux_maps(self):
        """``(det_aux list, aux_keys)``: the instrument's detector maps when the model reads data
        variables, else ``(None, None)``."""
        aux = dict(self.geom.aux) if self.model.needs_variables else {}
        if not aux:
            return None, None
        return [aux[k] for k in aux], list(aux)

    def variables(self, frames, ref_shape=None, ref_wcs=None):
        """The data variables over ``frames`` beyond the instrument's detector maps (a
        :class:`~selfcal.models.variables.VariableSet`: the model's variables and the instrument's
        frame variables); None when the model reads none of them."""
        if not self.model.variables and not (self.model.referenced_variables() & set(self.frame_variable_names())):
            return None
        return self.model.build_variables(self.geom, frames, ref_shape=ref_shape, ref_wcs=ref_wcs,
                                          frame_variables=dict(self.inst.frame_variables(frames)), log=print)

    def weight(self):
        """The model's observation weight (a function of data variables), or None."""
        return self.model.build_weight(self.geom, self.catalog(), frame_variables=self.frame_variable_names(),
                                       log=print)

    def priors(self, variables, n_frames):
        """The model's priors as ``setup_lsqr(priors=...)`` callables."""
        return self.model.build_priors(self.geom, variables=variables, n_frames=n_frames)

    def system(self, cc, job):
        """The identity of the system ``cc`` (a set-up
        :class:`~selfcal.pipeline.pipeline_wrapper.Calibrator`) solves for ``job``: a
        :class:`~selfcal.core.warm_start.System` (its frames in order, the model's unknowns, the
        reference grid, the job), recorded on the cal so that a later solve can start from it."""
        from selfcal.core.warm_start import System

        from .products import job_key
        return System.of(cc.layout, frames=cc.reproj_list, sky_model=cc.sky_model, chunk_maps=cc.chunk_maps,
                         basis_list=getattr(cc, 'basis_list', None), wcs=cc.ref_wcs, extra={'job': job_key(job)})

    def start(self, job, system, cal_path):
        """The warm start of ``job``'s solve (``calibrate(start=...)``): the
        :class:`~selfcal.core.warm_start.WarmStart` of the cal it starts from, checked against
        ``system`` (refused with :class:`~selfcal.config.base.ConfigError` when it is not a solution
        of it), or None. ``cal_path``: the cal the solve writes, which is never its start."""
        starts = self.spec.start
        if not starts or job.name not in starts:
            return None
        if self.spec.tiling is not None or self.spec.passes is not None:
            raise ValueError("a warm start continues a plain calibration; a tiled or an N-pass run takes no start")
        from selfcal.core.warm_start import WarmStart

        from .products import start_identity, start_label
        path = starts[job.name]
        if os.path.abspath(path) == os.path.abspath(cal_path):
            raise ValueError(f"start={path}: the cal this solve writes cannot be its start")
        start = WarmStart(path, identity=start_label(start_identity(path)))
        start.check(system)
        return start

    def snapshot_writer(self, cal_path, *, frames, system, start):
        """The :class:`~selfcal.core.snapshots.SnapshotWriter` of the solve that writes ``cal_path``
        (``calibrate(snapshots=...)``), or None without snapshots: every ``spec.snapshots.every``
        iterations, ``snapshots/<cal stem>_it<NNNN>.h5`` beside the cal, recording ``frames`` (the
        frames' permanent paths), the ``system`` solved and the ``start`` it continues."""
        snaps = self.spec.snapshots
        if snaps is None:
            return None
        if self.spec.tiling is not None or self.spec.passes is not None:
            raise ValueError("snapshots are written by a plain calibration; a tiled or an N-pass run takes none")
        from selfcal.core.snapshots import SnapshotWriter
        return SnapshotWriter(cal_path, snaps.every, snaps.keep, frames=frames, system=system, start=start)

    def x0(self, cc, start=None):
        """The LSQR starting vector of the system ``cc`` (a set-up
        :class:`~selfcal.pipeline.pipeline_wrapper.Calibrator`), float64 in the full column layout.

        With a warm start (``start``, a checked :class:`~selfcal.core.warm_start.WarmStart`): the
        solution of the cal it reads (:meth:`~selfcal.core.warm_start.WarmStart.vector`).
        Otherwise, with the per-frame scalar, each frame's scalar starts at the weighted mean of its
        data and everything else at zero (:func:`~selfcal.core.solution.compute_x0_scalar_only`);
        without it, the first sky term starts at zero and every later column at its own diagonal
        least-squares estimate (:func:`~selfcal.core.solution.compute_x0_from_Ab`, given one sky
        block whatever the model's number)."""
        if start is not None:
            return start.vector(active_mask=getattr(cc, 'active_mask', None))
        from selfcal.core.solution import compute_x0_from_Ab, compute_x0_scalar_only
        if self.model.x0_kind == 'from_Ab':
            return compute_x0_from_Ab(cc.A, cc.b, cc.ref_shape, active_mask=getattr(cc, 'active_mask', None))
        return compute_x0_scalar_only(cc.A, cc.b, cc.ref_shape, scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
                                      num_sky_blocks=cc.num_sky_blocks, active_mask=getattr(cc, 'active_mask', None))

    def solve_options(self):
        """The solver's ``apply_lsqr`` keywords."""
        return dict(self.spec.lsqr)

    def history_path(self, cal_file):
        """Where the history per iteration of the solve that makes ``cal_file`` goes:
        ``<field>/records/<cal stem>_history.npz``, next to the action records (a tile's cal, a tile's
        history)."""
        from selfcal.core.solve_record import history_path
        return history_path(os.path.join(self.spec.field.path, 'records'), cal_file)

    def configure(self, cc):
        """Settings recorded on the cal after the solve: the read-time Fisher threshold of the sky
        terms after the first, when a sky term has a coefficient."""
        if self.model.has_coefficients:
            cc.line_fisher_threshold = self.spec.line_fisher_threshold

    def mosaic_geometry(self, jobgeom):
        """``(chunk maps, offset renderers)`` of ``make_mosaic``: each offset term's chunk map on the
        reference grid and the instrument's renderer of its offsets (None: constant over each
        chunk)."""
        from selfcal.models.spec import chunk_map_of
        maps, funcs = [], []
        for term in self.model.offset:
            cm = chunk_map_of(term, self.geom)
            maps.append(cm.grid)
            funcs.append(self.inst.offset_renderer(self.geom, jobgeom, map_name=cm.name, render=term.render))
        return maps, funcs

    def clip_group_edges(self):
        """Wavelength bin edges of the grouped outlier clip: one group per value of the primary
        map's spectral axis."""
        from selfcal.models.offset_structure import group_edges_along
        cm, geom = self.geom.chunk_map, self.geom
        if cm.spectral_axis is None or geom.wavelength_key is None:
            raise ValueError("the grouped outlier clip needs a chunk map with a spectral axis and an instrument "
                             "wavelength map")
        return group_edges_along(geom.aux[geom.wavelength_key], cm.det, cm.axes, cm.spectral_axis)

    def refit_poly_basis(self, degree, segments=None):
        """The per-frame polynomial offset basis of the N-pass OFFSET refit: degree ``degree`` along
        the primary map's spectral axis, one per value of its group axis, over the model's
        polynomial window (optionally piecewise on ``segments``)."""
        from selfcal.models.offset_structure import poly_basis_along
        if self.spec.spectral_window is None:
            raise ValueError("the OFFSET refit's polynomial spans the model's polynomial window, and the model "
                             "has none")
        cm = self.geom.chunk_map
        lo, hi = self.spec.spectral_window
        return poly_basis_along(cm.axes, cm.spectral_axis, cm.group_axis, int(degree), lo, hi, segments=segments)

    # ---- naming (the one place) ------------------------------------------------------------
    def stem(self, job, suffix=None):
        """The stem of every product name of ``job``: ``<frame_tag>_<job.name><suffix>``.

        ``suffix`` defaults to the run's. :meth:`cal_file`, :meth:`mosaic_file` and
        :meth:`mosaic_cache_dir` wrap the stem and the N-pass products extend it. Example (SPHEREx,
        detector 4, channel 17): ``Detector4_NumSub10_NumCh34_NumCol3_Ch17<suffix>``.
        """
        suffix = self.spec.suffix if suffix is None else suffix
        return f'{self.frame_tag}_{job.name}{suffix}'

    def cal_file(self, job, suffix=None):
        """The cal file name of ``job``, ``cal_<stem>.h5`` (see :meth:`cal_path`)."""
        return f'cal_{self.stem(job, suffix)}.h5'

    def cal_path(self, job, suffix=None):
        """The full path of ``job``'s cal file: :meth:`cal_file` in the run's ``calibration/``
        directory. A plain calibration skips the solve when this file exists; a mosaic task reads
        it unless the run names another cal."""
        return os.path.join(self.pipeline_config.cal_dir, self.cal_file(job, suffix))

    def mosaic_file(self, job, suffix=None):
        """The mosaic file name of ``job``, ``mosaic_<stem>.fits`` (see :meth:`mosaic_path`)."""
        return f'mosaic_{self.stem(job, suffix)}.fits'

    def mosaic_path(self, job, suffix=None):
        """The full path of ``job``'s mosaic: :meth:`mosaic_file` in the run's ``mosaic/`` directory."""
        return os.path.join(self.pipeline_config.mos_dir, self.mosaic_file(job, suffix))

    def mosaic_cache_dir(self, job, suffix=None):
        """The directory of ``job``'s intermediate mosaic cache, ``<scratch>cache_<stem>``: written
        when the coadd caches its frames, deleted after the mosaic is saved."""
        return f'{self.spec.scratch}cache_{self.stem(job, suffix)}'

    def tile_cal_file(self, job, tile):
        """The cal file name of a tile (a :class:`~selfcal.pipeline.tiled.TileSpec`): :meth:`cal_file`
        with the run's suffix, whose ``{tile}`` is the tile's name."""
        return self.cal_file(job, self.spec.suffix.format(tile=tile.name))

    def stitched_cal_path(self, job):
        """The path of ``job``'s stitched cal, :meth:`cal_path` with the tiling's stitched suffix: the
        tiled calibration writes the Fisher stitch of its tile cals there (and skips the stitch
        when that file exists)."""
        return self.cal_path(job, self.spec.tiling.stitched_suffix)

    def pass_stem(self, job):
        """The stem of ``job``'s N-pass products, ``cal_<stem>`` with the stitched suffix of a tiled
        run (the run's suffix otherwise): ``<pass stem>_pass<i>sky.h5`` / ``_pass<i>off.h5``,
        ``<pass stem>_npass_monitor.json``, the work directory ``<scratch>npass_<pass stem>``."""
        suffix = self.spec.tiling.stitched_suffix if self.spec.tiling is not None else self.spec.suffix
        return 'cal_' + self.stem(job, suffix)


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
    """The frame directory a plain run reads from: the frames' own directory when they are read in
    place (no staging, no cleanup), else the staged copy of the field's frames."""
    frames = ctx.spec.frames
    if frames.in_place:
        staging.set_hdd_io_limit(None)
        return frames.in_place
    if not frames.stage_dir:
        raise ValueError("staging the frames needs a staging directory: Compute(scratch=...) or stage_dir=")
    return staging.prepare_nvme(frames, ctx.pipeline_config.reproj_dir)


def unstage_run(ctx, frame_dir):
    """Undo :func:`stage_run`: delete the staged copy of the frames unless the run keeps it (see
    :func:`~selfcal.run.staging.cleanup_nvme`); frames read in place are left alone."""
    if not ctx.spec.frames.in_place:
        staging.cleanup_nvme(ctx.spec.frames, frame_dir)


def frame_list(frame_dir, n_frames=None):
    """The sorted ``*.h5`` frames of a directory, optionally the first ``n``."""
    files = sorted(glob_module.glob(os.path.join(frame_dir, '*.h5')))
    return files[:n_frames] if n_frames else files


def clip_groups(ctx, groups):
    """The ``setup_lsqr`` keywords of an outlier clip over chunk groups of the primary chunk map
    (``Clip(per=...)``): ``{"along": axis}``, ``{"chunk": True}`` or ``{"mapping": [...]}``. Along the
    primary map's spectral axis of an instrument with a wavelength map, the clip bins the wavelength
    between the axis values' wavelengths (the N-pass passes' grouped clip); otherwise each pixel is
    judged within the group of its dominant chunk."""
    cm, geom = ctx.geom.chunk_map, ctx.geom
    if groups.get('map') not in (None, cm.name):
        raise ValueError(f"clip groups {groups}: groups of the primary chunk map ({cm.name!r}) only")
    axis = groups.get('along')
    if axis is not None and axis == cm.spectral_axis and geom.wavelength_key:
        return {'outlier_group_edges': ctx.clip_group_edges()}
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
def solve_job(ctx, job, jobgeom, *, frame_dir, cal_file, hdd_reproj_dir, frames=None, checkpoint=None,
              tile=None):
    """The engine's one solve: ``setup_lsqr`` + ``apply_lsqr`` + save for one job over one frame
    list, for every task (a plain cal, each tile of a tiled cal, the INIT pass of an N-pass run).

    ``frames`` (paths under ``frame_dir``) overrides the directory glob. The system is the model
    lowered by ``ctx`` with the run's ``setup_lsqr`` keywords; the starting vector
    (:meth:`RunContext.x0`) and the solver's options (:meth:`RunContext.solve_options`) come from the
    context, the one place a change to the solve goes. The cal records the frames under
    ``hdd_reproj_dir``, their permanent location, so it stays valid after the staged copy is
    cleaned up. ``checkpoint(label)`` is an optional progress/RSS hook called around the two heavy
    steps. A warm start (``spec.start``, a plain calibration's) starts the solve from the solution
    of an earlier cal of the same system (:meth:`RunContext.start`); snapshots (``spec.snapshots``,
    a plain calibration's) write the solution every ``k`` iterations as a cal file beside the cal
    (:meth:`RunContext.snapshot_writer`; the action's record lists them under the solve). A monitor
    (``spec.monitor``) checks the solve every ``m`` iterations and the fit's stop rules
    (``Fit(stop=...)``, in the solver's options) may end it early (:mod:`selfcal.core.monitor`):
    the checks go to the history file, the stop record to the three places below.

    The record of the solve (:class:`~selfcal.core.solve_record.SolveRecord`), with the identity
    of the system solved and the solve's start, goes to three places: the cal's ``solve`` group,
    the history per iteration to :meth:`RunContext.history_path`, and the action's record
    (``solves``, through the product book: ``spec.on_product('solve', ...)``, with ``tile``, the
    tile's name in a tiled run).
    """
    spec = ctx.spec
    checkpoint = checkpoint or (lambda label: None)
    cc = pipeline_wrapper.Calibrator(ctx.pipeline_config, reproj_dir=frame_dir)
    if frames is not None:
        cc.reproj_list = list(frames)
    n_frames = len(cc.reproj_list)
    # The model's data variables beyond the instrument's detector maps (frame values, sky maps,
    # stored layers, functions): None when the model reads none.
    variables = ctx.variables(cc.reproj_list, ref_shape=cc.ref_shape, ref_wcs=cc.ref_wcs)
    offset_model = ctx.offset_model(cc.reproj_list, variables)
    sky_model = ctx.sky_model()
    det_aux, aux_keys = ctx.aux_maps()
    cal_kwargs = dict(spec.setup)
    cal_kwargs.update(ctx.model.setup_kwargs())       # the model's own solver options (if any)
    if variables is not None:
        cal_kwargs['variables'] = variables
    weight_function = ctx.weight()
    if weight_function is not None:
        cal_kwargs['weight_function'] = weight_function
    priors = ctx.priors(variables, n_frames)
    if priors:
        cal_kwargs['priors'] = priors
    groups = cal_kwargs.pop('outlier_groups', None)
    if groups is not None:
        cal_kwargs.update(clip_groups(ctx, groups))
    # The grouped clip bins the instrument's wavelength map unless the clip names another variable.
    if not cal_kwargs.get('outlier_group_variable'):
        cal_kwargs['outlier_group_variable'] = ctx.geom.wavelength_key
    if spec.pre_cal is not None:
        cal_kwargs['preprocess_func'] = spec.pre_cal
    if spec.post_cal is not None:
        cal_kwargs['postprocess_func'] = spec.post_cal
    checkpoint('pre-setup_lsqr')
    cc.setup_lsqr(
        offset_model=offset_model,
        grid_valid_weight=jobgeom.det_valid_weight,
        oversample_factor=1,
        sky_model=sky_model,
        det_aux=det_aux,
        aux_keys=aux_keys,
        batch_spill_dir=spec.scratch,
        **cal_kwargs)
    checkpoint('post-setup_lsqr')
    # The identity of the system set up (recorded on the cal) and the solve's start: the solution
    # of an earlier cal of this system (checked against it), or the default guess.
    system = ctx.system(cc, job)
    start = ctx.start(job, system, os.path.join(ctx.pipeline_config.cal_dir, cal_file))
    # Snapshots record the frames' permanent paths, as the cal does, and the cal's settings (configure).
    snapshots = ctx.snapshot_writer(os.path.join(ctx.pipeline_config.cal_dir, cal_file), system=system, start=start,
                                    frames=staging.remap_to_nvme(cc.reproj_list, hdd_reproj_dir))
    if snapshots is not None:
        ctx.configure(cc)
    # List-pop hand-off: keeping a plain `x0` local would pin the full-layout
    # f64 vector for the entire solve (see Calibrator.apply_lsqr).
    _x0_owned = [ctx.x0(cc, start=start)]
    checkpoint('pre-apply_lsqr')
    watch = {} if spec.monitor is None else {'monitor': spec.monitor}
    if snapshots is not None:
        watch['snapshots'] = snapshots
    cc.apply_lsqr(x0=_x0_owned.pop(), **ctx.solve_options(), **watch)
    checkpoint('post-apply_lsqr')
    ctx.configure(cc)
    record = cc.solve_record
    record.set_system(system)
    if start is not None:
        record.continues(start)
    # The history first, so the cal names a file that exists.
    record.save_history(ctx.history_path(cal_file))
    # Save with the permanent (HDD) paths so the cal stays valid after cleanup.
    staged_list = cc.reproj_list
    cc.reproj_list = staging.remap_to_nvme(staged_list, hdd_reproj_dir)
    cal_path = cc.save_calibration(cal_file=cal_file)
    cc.reproj_list = staged_list
    del cc
    gc.collect()
    entry = record.entry()
    if snapshots is not None:
        entry['snapshots'] = snapshots.summary()
        print(f"{len(snapshots.written)} snapshots kept in {snapshots.directory}"
              + (f" ({len(snapshots.removed)} deleted, keep={snapshots.keep})" if snapshots.removed else '')
              + (f"; {len(snapshots.failed)} could not be written" if snapshots.failed else ''))
    announce(spec, 'solve', cal_path, job=job, tile=tile, solve=entry)
    return cal_path


# ---------------------------------------------------------------------------
# Primitive 2: one cal -> one mosaic
# ---------------------------------------------------------------------------
def mosaic_job(ctx, job, jobgeom, *, cal_path, frame_dir, mos_file, cache_dir):
    """Coadd the frames of ``cal_path`` (read from ``frame_dir``) into ``mos_file``; the instrument's
    maps (SPHEREx: the wavelength maps) ride along when the run asks for them and the instrument
    has them."""
    spec, inst, geom = ctx.spec, ctx.inst, ctx.geom
    chunk_maps, det_offset_funcs = ctx.mosaic_geometry(jobgeom)
    mm = pipeline_wrapper.Mosaicker(ctx.pipeline_config, reproj_dir=frame_dir, unit=ctx.unit)
    mm.load_calibration(cal_path=cal_path)
    mm.reproj_list = staging.remap_to_nvme(mm.reproj_list, frame_dir)
    # A cal solved on another grid (another run's cal): frames it lists that have no
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
    mosaic_kwargs = dict(spec.mosaic)
    # Offset terms whose offsets are coefficients of known functions of data
    # variables are not constant per chunk: the mosaic subtracts them at every
    # observation (BasisOffsetSubtractor), and hands the chunk path zeros for
    # them (plus the per-frame scalar on map 0, which the cal folds in there).
    basis_hook = _basis_offset_hook(ctx, mm, cal_path)
    # The model's observation weight (a row weight w) weights the coadd by w², as
    # in the solve.
    weight_hook = _observation_weight_hook(ctx, mm)
    hooks = [h for h in (basis_hook, spec.post_mosaic, weight_hook) if h is not None]
    post = None
    if len(hooks) > 1:
        from selfcal.pipeline.model_eval import ComposedHook
        post = ComposedHook(hooks)
    elif hooks:
        post = hooks[0]
    if post is not None:
        mosaic_kwargs['postprocess_func'] = post
    # The instrument's maps (the wav_mean / wav_std maps) are sigma-clipped against the std map:
    # with the sigma clip they are coadded inside its pass (no extra pass, no cache needed),
    # otherwise the standalone coadd runs over the frame cache.
    wav_maps = None
    aux_coadds = inst.aux_coadds(geom) if spec.instrument_maps else None
    want_wav = aux_coadds is not None
    if want_wav:
        if not spec.mosaic.get('make_std_map', False):
            raise ValueError("Coadd(instrument_maps=True): the instrument's maps are coadded against the std map; "
                             "give Coadd(std=True), or instrument_maps=False")
        if spec.mosaic.get('apply_sigma_clipping', False):
            wav_maps = aux_coadds
        elif not spec.mosaic.get('cache_intermediate', False):
            raise ValueError("Coadd(instrument_maps=True): the instrument's maps are coadded in the sigma-clip pass "
                             "or over the frame cache; give Coadd(clip=...) or Compute(cache_frames=True), or "
                             "instrument_maps=False")
    maps = mm.make_mosaic(
        chunk_maps=chunk_maps,
        grid_valid_weight=jobgeom.grid_valid_weight,
        oversample_factor=spec.oversample,
        det_offset_funcs=det_offset_funcs,
        cache_dir=cache_dir,
        wav_maps=wav_maps,
        **mosaic_kwargs)
    if want_wav:
        inst.finalize_mosaic(geom, mm, maps, spec.mosaic['sigma'])
    mos_path = mm.save_mosaic(mos_file=mos_file, overwrite=True)
    del mm, maps
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
    return mos_path


def _basis_offset_hook(ctx, mm, cal_path):
    """The mosaic's per-observation subtraction of the offset terms that carry
    known functions of data variables (``coefficient`` / ``basis``), or None.
    Rewrites those maps' entries of ``mm.offsets`` so the chunk path subtracts
    only the per-frame scalar there."""
    if not any(t.coefficient is not None or t.basis is not None for t in ctx.model.offset):
        return None
    from selfcal.io.calfile import CalFile
    from selfcal.pipeline.model_eval import BasisOffsetSubtractor
    # The data variables over the mosaic's frames (the cal's, minus any without
    # a reprojected file here); coefficients are looked up by the cal's order.
    frames = list(mm.reproj_list)
    variables = _mosaic_variables(ctx, mm)
    offset_model = ctx.offset_model(frames, variables, log=_quiet)
    terms = [(m, b.basis, b.chunk_map) for m, b in enumerate(offset_model.blocks) if b.basis is not None]
    keep = [os.path.basename(p) for p in frames]
    with CalFile(cal_path) as cal:
        scalar = cal.frame_scalar
        order = {os.path.basename(p): i for i, p in enumerate(cal.reproj_list)}
    rows = np.array([order[k] for k in keep], dtype=np.int64)
    for m, _, cm in terms:
        n_chunks = int(np.asarray(cm).max()) + 1
        zeros = np.zeros((len(rows), n_chunks), dtype=np.float64)
        if m == 0 and scalar is not None:
            zeros += np.asarray(scalar)[rows][:, None]
        mm.offsets[m] = zeros
        mm.offset_coverage_fracs[m] = np.ones_like(zeros)
    return BasisOffsetSubtractor(cal_path, terms, variables=variables, oversample_factor=1, frame_names=frames)


def _mosaic_variables(ctx, mm):
    """The data variables over the mosaic's frames, with the instrument's detector
    maps (the solve passes those separately), for the mosaic-side evaluators."""
    from selfcal.models.variables import VariableSet
    vs = ctx.variables(list(mm.reproj_list), ref_shape=mm.ref_shape, ref_wcs=mm.ref_wcs) or VariableSet()
    extra = {k: v for k, v in (ctx.geom.aux or {}).items() if k not in vs.detector}
    return vs.merged(detector=extra) if extra else vs


def _observation_weight_hook(ctx, mm):
    """The mosaic's per-observation weight from the model's ``weight``, or None."""
    weight = ctx.weight()
    if weight is None:
        return None
    from selfcal.pipeline.model_eval import ObservationWeight
    return ObservationWeight(weight, variables=_mosaic_variables(ctx, mm), frame_names=list(mm.reproj_list),
                             det_shape=ctx.geom.shape)


# ---------------------------------------------------------------------------
# Tiling helpers shared by the tiled cal and the N-pass scheduler
# ---------------------------------------------------------------------------
def resolve_tiles(tiling):
    """The tiles of a :class:`~selfcal.run.runspec.TilingSpec`: its explicit tiles (arbitrary,
    possibly overlapping bboxes) or its uniform grid; then the optional ``only`` restriction (the
    full grid is built first so every bbox is right). Returns ``(tiles, only)``."""
    from selfcal.pipeline.tiled import TileSpec, make_tile_grid
    if tiling.tiles is not None:
        tiles = [TileSpec(name=name, bbox=tuple(bbox)) for name, bbox in tiling.tiles]
    else:
        tiles = make_tile_grid(tiling.ref_shape, tiling.grid[0], tiling.grid[1], overlap_px=tiling.overlap,
                               names=None if tiling.names is None else list(tiling.names))
    only = tiling.only
    if only:
        all_names = [tile.name for tile in tiles]
        tiles = [tile for tile in tiles if tile.name in only]
        if not tiles:
            raise ValueError(f"only={list(only)} matched no tile in {all_names}")
    return tiles, only


def tiling_frames(frames_dir):
    """Every frame of a directory (every detector), in (exposure, detector) order."""
    from selfcal.io.reproj import parse_reproj_basename
    files = glob_module.glob(os.path.join(frames_dir, 'exp_*_det_*.h5'))
    return sorted(files, key=parse_reproj_basename)


def tile_assignment(tiling):
    """``(TiledCalibration, tiles, only, assignment)`` of a :class:`~selfcal.run.runspec.TilingSpec`."""
    from selfcal.pipeline.tiled import TiledCalibration
    tiles, only = resolve_tiles(tiling)
    tiled = TiledCalibration(tiling_frames(tiling.frames_dir), tiles, frame_filter=tiling.assign, halo=tiling.halo)
    return tiled, tiles, only, tiled.assign_frames()
