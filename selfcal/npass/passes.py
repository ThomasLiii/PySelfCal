"""One SKY or OFFSET pass of an N-pass run, over the run's tiles (:class:`PassRunner`), in the run
engine's context: its model, its staging and its products. The schedule of the passes is
:mod:`selfcal.run.npass`'s; what a pass computes is :mod:`selfcal.npass.sky`'s and
:mod:`selfcal.npass.offsets`'."""
from __future__ import annotations

import gc
import json
import os
import time

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from ..models.offset_model import OffsetModel
from ..pipeline import pipeline_wrapper
from ..pipeline.model_eval import ComposedHook
from ..pipeline.tiled import stitch
from ..run import staging
from ..run.engine import setup_job, tile_assignment
from .offsets import SkySubtractor, merge_offset_cals, normal_equations, solve_frames
from .sky import OffsetModelSubtractor, combine_moments, dump_moments, solve_pixels

__all__ = ["PassRunner", "offset_spec", "append_monitor"]


def _quiet(*args, **kwargs):
    """A log that says nothing."""


def offset_spec(spec, edges_fn, hook):
    """An OFFSET pass's run: ``spec`` with the sky held fixed (``fixed_sky``: no sky unknowns; ``hook``
    subtracts the sky after the run's own frame hook) and the OFFSET pass's clip. Everything else --
    the model, its constraints and priors, the weights -- is the run's."""
    clip = spec.passes.refit_clip
    setup = dict(spec.setup)
    setup['fixed_sky'] = True
    setup['outlier_thresh'] = clip.sigma
    setup.pop('outlier_groups', None)                # the pass's clip alone decides the grouping
    if clip.grouped:
        setup['outlier_group_edges'] = edges_fn()
    else:
        setup.pop('outlier_group_edges', None)
    return spec.replace(setup=setup, post_cal=hook)


class PassRunner:
    """One N-pass run's SKY and OFFSET passes over its tiles: everything they share, resolved once in the
    run's context (the job, its geometry and model, the tiles' frames, the products' names)."""

    def __init__(self, spec, ctx):
        self.spec, self.ctx = spec, ctx
        self.p = spec.passes
        self.geom = ctx.geom
        self.job = ctx.single_job("calibrate(passes=...)")
        self.jobgeom = ctx.job_geometry(self.job)
        self.sky_model = ctx.sky_model()
        self.det_aux, self.aux_keys = ctx.aux_maps()
        self.cm = self.geom.chunk_map.det
        self.grid_valid = self.jobgeom.det_valid_weight
        self.stem = ctx.pass_stem(self.job)
        self.cal_dir = ctx.pipeline_config.cal_dir
        self.work_dir = os.path.join(spec.scratch, f"npass_{self.stem}")
        os.makedirs(self.work_dir, exist_ok=True)
        self.monitor_path = os.path.join(self.cal_dir, f"{self.stem}_npass_monitor.json")
        self._edges = None
        self.max_workers = int(spec.setup.get("max_workers", 48))
        self.lft = spec.line_fisher_threshold
        self.tile_frames = None       # {tile_name: [hdd paths]}
        self.all_frames = None        # [hdd paths] of the whole field

    def variables_for(self, frames):
        """The model's data variables over ``frames`` beyond the detector maps
        (frame values, sky maps, layers, functions); None for a model that reads
        detector maps only."""
        model = self.ctx.model
        if not model.variables and not (model.referenced_variables() & set(self.ctx.frame_variable_names())):
            return None
        from selfcal.geometry import wcs_helper
        ref_wcs, ref_shape = wcs_helper.load_from_fits(self.ctx.pipeline_config.ref_path)
        return self.ctx.variables(list(frames), ref_shape=ref_shape, ref_wcs=ref_wcs)

    def edges(self):
        if self._edges is None:
            self._edges = self.ctx.clip_group_edges()
        return self._edges

    def product(self, i, kind):
        return os.path.join(self.cal_dir, f"{self.stem}_pass{i}{kind}.h5")

    # ---- frames -------------------------------------------------------------
    def resolve_frames(self, init_result):
        spec = self.spec
        if spec.tiling is not None:
            assignment = init_result.assignment if init_result else None
            if assignment is None:
                _, _, _, assignment = tile_assignment(spec.tiling)
            # Moments are additive only over DISJOINT frame sets: with
            # overlapping tile bboxes (or halo > 0) a frame can be centre-assigned
            # to several tiles -> keep it in the FIRST tile listed.
            self.tile_frames = {}
            seen = set()
            n_dup = 0
            for name, (files, _) in assignment.items():
                keep = []
                for f in files:
                    b = os.path.basename(f)
                    if b in seen:
                        n_dup += 1
                        continue
                    seen.add(b)
                    keep.append(f)
                self.tile_frames[name] = keep
            if n_dup:
                print(f"[npass] {n_dup} frames assigned to more than one tile -> kept in the "
                      f"first tile listed (moments must not double-count)", flush=True)
            self.all_frames = sorted({f for fs in self.tile_frames.values() for f in fs})
        else:
            cal = init_result.cal_paths[0]
            with h5py.File(cal, "r") as f:
                reproj = [r.decode() if isinstance(r, bytes) else str(r) for r in f["reproj_list"][:]]
            src = spec.frames.in_place or os.path.dirname(reproj[0])
            frames = [os.path.join(src, os.path.basename(r)) for r in reproj]
            self.tile_frames = {"all": frames}
            self.all_frames = list(frames)

    # ---- SKY pass -----------------------------------------------------------
    def offset_terms(self, frames, variables):
        """``[(m, basis or None, det_chunk_map)]``: the model's offset terms, as the solve assembles them."""
        om = self.ctx.offset_model(list(frames), variables, log=_quiet)
        return [(m, b.basis, b.chunk_map) for m, b in enumerate(om.blocks)]

    def frame_hook(self, hook):
        """``hook`` after the run's own frame hook (``Fit(frame_hook=)``), which every pass applies as the INIT does."""
        if self.spec.post_cal is None:
            return hook
        return ComposedHook([self.spec.post_cal, hook])

    def staged(self, files):
        """The frame directory and the frame list of one tile's pass (staged on fast disk when the run stages)."""
        spec = self.spec
        if spec.tiling is not None:
            nvme = staging.claim(spec.tiling.stage_dir, spec.tiling.frames_dir)
            staging.stage_files(files, nvme, spec.frames.io_limit)
            return nvme, sorted(os.path.join(nvme, os.path.basename(f)) for f in files)
        return spec.frames.in_place or os.path.dirname(self.all_frames[0]), list(files)

    def sky_pass(self, i, offsets_cals, out):
        """The sky given the offsets of ``offsets_cals``: per tile, the per-pixel sums of the data minus
        the offsets (no matrix), dumped; the dumps summed and every pixel solved once into ``out``
        (``sky_merge="combine"``), or each tile solved and the tiles stitched."""
        spec, ctx, p = self.spec, self.ctx, self.p
        clip = p.sky_clip
        edges = self.edges() if clip.grouped else None
        calk = dict(spec.setup)
        for k in ("outlier_thresh", "outlier_group_edges", "outlier_groups"):
            calk.pop(k, None)
        if not calk.get("outlier_group_variable"):
            calk["outlier_group_variable"] = self.geom.wavelength_key
        dws = ctx.sky_damping
        io_limit = spec.frames.io_limit
        if spec.tiling is not None:
            nvme = staging.claim(spec.tiling.stage_dir, spec.tiling.frames_dir)
        else:
            nvme = spec.frames.in_place or os.path.dirname(self.all_frames[0])
        mom_dir = os.path.join(self.work_dir, "moments")
        os.makedirs(mom_dir, exist_ok=True)
        pieces = []
        for name, files in self.tile_frames.items():
            piece = (os.path.join(mom_dir, f"pass{i}_{name}.npz") if p.sky_merge == "combine"
                     else os.path.join(self.cal_dir, f"{self.stem}_pass{i}sky_{name}.h5"))
            if os.path.exists(piece):
                print(f"[npass] pass {i} SKY [{name}]: exists, skipping ({os.path.basename(piece)})",
                      flush=True)
                pieces.append(piece)
                continue
            t0 = time.time()
            if spec.tiling is not None:
                staging.stage_files(files, nvme, io_limit)
                frame_list = sorted(os.path.join(nvme, os.path.basename(f)) for f in files)
            else:
                frame_list = list(files)
            print(f"\n[npass] pass {i} SKY [{name}]: {len(frame_list)} frames, closed form given the offsets, "
                  f"clip {clip.sigma}{' per-subch' if edges is not None else ''}",
                  flush=True)
            cc = pipeline_wrapper.Calibrator(ctx.pipeline_config, reproj_dir=nvme)
            cc.reproj_list = frame_list
            variables = self.variables_for(frame_list)
            if variables is not None:
                calk['variables'] = variables
            # The offsets of every term of the model, as the solve's rows define them.
            subtract = self.frame_hook(OffsetModelSubtractor(
                offsets_cals, self.offset_terms(frame_list, variables), variables=variables,
                frame_names=frame_list))
            cc.setup_lsqr(offset_model=OffsetModel.sky_only(), grid_valid_weight=self.grid_valid,
                          oversample_factor=1,
                          sky_model=self.sky_model, det_aux=self.det_aux, aux_keys=self.aux_keys,
                          postprocess_func=subtract, outlier_thresh=float(clip.sigma),
                          outlier_group_edges=edges,
                          sky_rhs_moments=True, moments_only=True, batch_spill_dir=spec.scratch, **calk)
            if p.sky_merge == "combine":
                dump_moments(cc, piece)
            else:
                cc.x = solve_pixels(cc.pixel_fisher, cc.pixel_cross, cc.pixel_rhs, cc.pixel_counts,
                                    cc.ref_shape[0] * cc.ref_shape[1], cc.num_sky_blocks, damp_weights=dws)
                ctx.configure(cc)
                cc.reproj_list = staging.remap_to_nvme(cc.reproj_list, ctx.pipeline_config.reproj_dir)
                cc.save_calibration(cal_file=os.path.basename(piece))
            del cc
            gc.collect()
            print(f"[npass] pass {i} SKY [{name}] done ({time.time()-t0:.0f}s)", flush=True)
            pieces.append(piece)
        if p.sky_merge == "combine":
            combine_moments(pieces, out, sky_names=list(self.sky_model.names), damp_weights=dws,
                            line_fisher_threshold=self.lft, attrs={"npass_pass": i})
            # The dumps are pure intermediates and they are BIG (a J=4 full-NEP
            # tile dump is ~23 GB, so one n=8 run is ~0.5 TB and a 16-tile one
            # ~1.4 TB). Once the combine has written the product they have no
            # further use -- a resumed run skips the whole pass on the product's
            # existence, not on theirs. Keeping them filled the 3.6 TB root disk
            # mid-run on 2026-09-10 and killed pass 4 with ENOSPC.
            if not p.keep_moments:
                freed = 0
                for q in pieces:
                    try:
                        freed += os.path.getsize(q)
                        os.remove(q)
                    except OSError:
                        pass
                print(f"[npass] pass {i} SKY: removed {len(pieces)} moment dumps "
                      f"({freed/1e9:.0f} GB freed); Passes(keep_moments=True) keeps them", flush=True)
        else:
            ref_shape = spec.tiling.ref_shape if spec.tiling is not None else None
            stitch(pieces, out, ref_shape=ref_shape, line=self.sky_model.n_blocks >= 2)
        return out

    # ---- OFFSET pass --------------------------------------------------------
    def offset_pass(self, i, sky_cal, out):
        """The model's offsets given the sky ``sky_cal``, frame by frame: per tile, the job's rows set up
        as the engine sets them up (:func:`~selfcal.run.engine.setup_job`), on the data minus the sky
        and with no sky columns, reduced in the workers to each frame's normal equations (no matrix)
        and solved directly (:func:`~selfcal.npass.offsets.solve_frames`); the tiles' solutions joined
        into ``out``. Returns ``(out, monitor)``."""
        spec, ctx, p = self.spec, self.ctx, self.p
        variables = self.variables_for(self.all_frames)
        sky = SkySubtractor(sky_cal, self.sky_model, export_dir=os.path.join(self.work_dir, f"sky_pass{i-1}"),
                            aux_keys=tuple(self.aux_keys or ()), variables=variables,
                            frames=self.all_frames if variables is not None else None,
                            exclude_off_map=True, bright_cut=p.bright_cut, min_pixels=p.min_pixels)
        off_ctx = ctx.derive(offset_spec(spec, self.edges, self.frame_hook(sky)))
        opts = ctx.solve_options()
        pieces, solves = [], []
        for name, files in self.tile_frames.items():
            piece_file = f"{self.stem}_pass{i}off_{name}.h5"
            piece = os.path.join(self.cal_dir, piece_file)
            if os.path.exists(piece):
                print(f"[npass] pass {i} OFFSET [{name}]: exists, skipping ({piece_file})", flush=True)
                with h5py.File(piece, "r") as f:
                    summary = json.loads(f.attrs["solve"]) if "solve" in f.attrs else {}
            else:
                t0 = time.time()
                frame_dir, frame_list = self.staged(files)
                print(f"\n[npass] pass {i} OFFSET [{name}]: {len(frame_list)} frames, the model's offsets given "
                      f"{os.path.basename(sky_cal)}", flush=True)
                cc = setup_job(off_ctx, self.jobgeom, frame_dir=frame_dir, frames=frame_list,
                               reduce=normal_equations)
                cc.x, summary = solve_frames(
                    cc.reduced, cc.constraint_blocks, cc.layout.total_cols, start=ctx.seed,
                    exclusions=ctx.model.column_exclusions(ctx.geom, cc.layout) or None,
                    precondition=opts.get("precondition", True),
                    damp=opts.get("damp", 0.0), dtype=np.float32 if opts.get("use_float32") else np.float64)
                cc.reduced = cc.constraint_blocks = None
                print(f"[npass] pass {i} OFFSET [{name}]: {summary['blocks']} blocks of {summary['columns']} columns "
                      f"solved directly ({summary['joined']} joined, {summary['left_out']} direction(s) left out, "
                      f"largest condition {summary['largest_condition']:.3g}); |b - A x| = "
                      f"{summary['true_residual']:.6e}, |b| = {summary['bnorm']:.6e}", flush=True)
                ctx.configure(cc)
                cc.reproj_list = staging.remap_to_nvme(cc.reproj_list, ctx.pipeline_config.reproj_dir)
                cc.save_calibration(cal_file=piece_file)
                with h5py.File(piece, "a") as f:
                    f.attrs["solve"] = json.dumps(summary)
                del cc
                gc.collect()
                print(f"[npass] pass {i} OFFSET [{name}] done ({time.time()-t0:.0f}s)", flush=True)
            solves.append({"tile": name, **{k: summary[k] for k in ("method", "blocks", "columns", "left_out",
                                                                    "true_residual", "bnorm") if k in summary}})
            pieces.append(piece)
        merge_offset_cals(pieces, out, attrs={"npass_pass": i, "sky_cal": os.fspath(sky_cal)})
        return out, {"solves": solves}


def append_monitor(path, record):
    """Append one pass record to the JSON monitor file (list of dicts)."""
    data = []
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data = [r for r in data if r.get("pass") != record.get("pass")] + [record]
    data.sort(key=lambda r: r.get("pass", 0))
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=float)
    return data
