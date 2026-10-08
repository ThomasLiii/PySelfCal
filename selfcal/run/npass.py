"""N-pass alternating solve (task ``npass``) — a scheduler over the engine.

One formalism for the spectral calibrations: J sky blocks per pixel (continuum +
N line amplitudes), a per-frame polynomial offset in the chunk row coordinate
per column plus a per-frame scalar. The joint problem is solved by
**alternating least squares**, each half solved exactly, in a schedule of three
pass types::

    INIT    pass 1      S, a, s jointly   — joint LSQR (a plain or tiled ``cal`` run)
    SKY     SKY passes  S | a, s          — per-tile moment dumps -> one closed-form solve
    OFFSET  the others  a, s | S          — per-frame dense least squares, one global sky

The schedule is INIT, then SKY and OFFSET passes alternately, ``order`` saying which comes first
(:func:`~selfcal.run.schedule.schedule`). With ``n = 1`` this task IS the single solve of task
``cal`` — same model, same tiling, same clip — and reproduces it byte for byte (regression gate).

Every SKY/OFFSET pass is an exact block minimization, so the joint objective is
non-increasing in N. More passes are not automatically better: along the exact
null spaces (uniform line floor <-> static detector pattern; uniform sky <->
per-frame scalars) the objective is flat and the iterate can drift, so the
scheduler records per-pass monitors (residual RMS, step norms, gauge indicators)
in ``<stem>_npass_monitor.json`` and ``stop_tol`` can stop early.
Each pass writes a product (``<stem>_pass{i}sky.h5`` / ``_pass{i}off.h5``);
a re-run skips passes whose product exists.

The settings are a :class:`~selfcal.run.runspec.PassesSpec` (from
:class:`~selfcal.run.schedule.Passes`). Pass 1 is the ``cal`` task on the same run (tiled when the
run has tiles; its tiles are then also the memory tiling of every SKY pass); the INIT clip
replaces only the clip of the run's ``setup_lsqr`` keywords for that pass. Frames for the SKY
passes are re-staged per tile; the OFFSET passes read every frame of the field from the tiled
run's frame directory (or the pass-1 frame list). Every pass damps the sky terms as the INIT does
(:attr:`~selfcal.run.engine.RunContext.sky_damping`).

The passes use two methods of the run's context: :meth:`~selfcal.run.engine.RunContext.clip_group_edges`
(the grouped outlier clip) and :meth:`~selfcal.run.engine.RunContext.refit_poly_basis` (the OFFSET
refit's per-frame polynomial).
"""
from __future__ import annotations

import gc
import os
import time

from . import staging
from .engine import RunContext, announce, tile_assignment
from .schedule import schedule

__all__ = ["describe_schedule", "run_npass"]


def _init_spec(spec, edges_fn):
    """The pass-1 run: ``spec`` with the INIT clip applied to its ``setup_lsqr`` keywords. Nothing
    else changes, so ``n = 1`` without an INIT clip is exactly the ``cal`` task."""
    clip = spec.passes.init_clip
    setup = dict(spec.setup)
    if clip is not None:
        setup['outlier_thresh'] = clip.sigma
        if clip.ignore_flags is not None:
            setup['ignore_list'] = list(clip.ignore_flags)
        if clip.grouped:
            setup['outlier_group_edges'] = edges_fn()
        else:
            setup.pop('outlier_groups', None)        # the first pass clips per frame
    return spec.replace(setup=setup)


def _clip(clip):
    return f"{clip.sigma:g} {'grouped' if clip.grouped else 'per frame'}"


def describe_schedule(spec) -> list[str]:
    """The schedule of an N-pass run, one line per pass."""
    p = spec.passes
    sched = schedule(p.n, p.order)
    how1 = ("tiled (" + ("explicit tiles" if spec.tiling.tiles is not None else "grid") + ")"
            if spec.tiling is not None else "single cal")
    lines = [f"npass: n={p.n}, order={p.order}, sky_merge={p.sky_merge}, stop_tol={p.stop_tol}"]
    for i, t in enumerate(sched, start=1):
        if t == "init":
            over = f"clip {_clip(p.init_clip)}" if p.init_clip is not None else "the fit's clip"
            lines.append(f"  pass {i}: INIT   joint LSQR iter_lim={spec.lsqr.get('iter_lim')} "
                         f"via {how1}; {over}" + ("  -> product = the plain cal" if p.n == 1 else ""))
        elif t == "sky":
            lines.append(f"  pass {i}: SKY    closed form given pass-{i-1} offsets, clip {_clip(p.sky_clip)}")
        else:
            src = "the stitched INIT sky" if i == 2 else f"pass-{i-1} sky"
            lines.append(f"  pass {i}: OFFSET per-frame refit given {src}: degree {p.refit_degree}, clip "
                         f"{_clip(p.refit_clip)}, bright_cut {p.bright_cut}, min_pixels {p.min_pixels}, "
                         f"segments {p.segments}, ridge {p.ridge}")
    return lines


# --------------------------------------------------------------------------- #
class _Run:
    """Everything the SKY/OFFSET passes share (resolved once, in the run's context)."""

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
        if any(t.coefficient is not None or t.basis is not None for t in model.offset):
            raise ValueError("the N-pass OFFSET pass refits a polynomial basis per frame; a model whose "
                             "offset terms carry a coefficient or a basis runs without passes")
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
            import h5py
            import hdf5plugin  # noqa: F401
            cal = init_result.cal_paths[0]
            with h5py.File(cal, "r") as f:
                reproj = [r.decode() if isinstance(r, bytes) else str(r) for r in f["reproj_list"][:]]
            src = spec.frames.in_place or os.path.dirname(reproj[0])
            frames = [os.path.join(src, os.path.basename(r)) for r in reproj]
            self.tile_frames = {"all": frames}
            self.all_frames = list(frames)

    # ---- SKY pass -----------------------------------------------------------
    def sky_pass(self, i, offsets_cals, out):
        from selfcal.models.offset_model import OffsetModel
        from selfcal.pipeline import pipeline_wrapper
        from selfcal.pipeline.npass import OffsetSubtractor, combine_moments, dump_moments
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
        subtract = OffsetSubtractor(offsets_cals)
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
            print(f"\n[npass] pass {i} SKY [{name}]: {len(frame_list)} frames, K=0 closed form, "
                  f"clip {clip.sigma}{' per-subch' if edges is not None else ''}",
                  flush=True)
            cc = pipeline_wrapper.Calibrator(ctx.pipeline_config, reproj_dir=nvme)
            cc.reproj_list = frame_list
            variables = self.variables_for(frame_list)
            if variables is not None:
                calk['variables'] = variables
            cc.setup_lsqr(offset_model=OffsetModel.sky_only(), grid_valid_weight=self.grid_valid,
                          oversample_factor=1,
                          sky_model=self.sky_model, det_aux=self.det_aux, aux_keys=self.aux_keys,
                          postprocess_func=subtract, outlier_thresh=float(clip.sigma),
                          outlier_group_edges=edges,
                          sky_rhs_moments=True, batch_spill_dir=spec.scratch, **calk)
            if p.sky_merge == "combine":
                dump_moments(cc, piece)
            else:
                cc.solve_sky_closed_form(dws)
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
            from selfcal.pipeline.tiled import stitch
            ref_shape = spec.tiling.ref_shape if spec.tiling is not None else None
            stitch(pieces, out, ref_shape=ref_shape, line=self.sky_model.n_blocks >= 2)
        return out

    # ---- OFFSET pass --------------------------------------------------------
    def offset_pass(self, i, sky_cal, out):
        from selfcal.pipeline.npass import SkySubtractor, refit_offsets_per_frame
        p = self.p
        pb = self.ctx.refit_poly_basis(p.refit_degree, segments=p.segments)
        variables = self.variables_for(self.all_frames)
        sky = SkySubtractor(sky_cal, self.sky_model,
                            export_dir=os.path.join(self.work_dir, f"sky_pass{i-1}"),
                            aux_keys=tuple(self.aux_keys or ()), variables=variables,
                            frames=self.all_frames if variables is not None else None)
        edges = self.edges() if p.refit_clip.grouped else None
        _, mon = refit_offsets_per_frame(
            self.all_frames, sky, det_chunk_map=self.cm, grid_valid=self.grid_valid,
            det_aux=self.det_aux, poly_basis=pb, edges=edges, edges_key=self.geom.wavelength_key,
            ignore_list=self.spec.setup.get("ignore_list", []),
            thresh=float(p.refit_clip.sigma), bright_cut=p.bright_cut,
            min_pix=int(p.min_pixels), out_h5=out, max_workers=self.max_workers,
            ridge=float(p.ridge or 0.0),
            attrs={"npass_pass": i})
        return out, mon


# --------------------------------------------------------------------------- #
def run_npass(spec, ctx=None, *, run_calibration):
    """Run the schedule. ``run_calibration`` is the ``cal`` task (plain or tiled
    per the run), used for the INIT pass."""
    from selfcal.pipeline.npass import append_monitor, offset_monitors, sky_monitors
    p = spec.passes
    sched = schedule(p.n, p.order)
    for line in describe_schedule(spec):
        print(f"[npass] {line}", flush=True)
    if sched[-1] == "offset":
        print(f"[npass] NOTE: the schedule ends on an OFFSET pass, which writes offsets but no "
              f"sky map -- the last sky product will be from pass {len(sched) - 1}. Use an odd n "
              f"with order='offset_first' (or an even n with 'sky_first') to end on a sky.",
              flush=True)
    ctx = ctx or RunContext.build(spec)
    run = _Run(spec, ctx)

    # ---- pass 1: INIT = the cal task -------------------------------------------
    t0 = time.time()
    init = _init_spec(spec, edges_fn=run.edges)
    res = run_calibration(init, ctx=ctx.derive(init))
    init_cals = list(res.cal_paths)
    init_sky = res.sky_path
    append_monitor(run.monitor_path, {"pass": 1, "type": "init", "products": init_cals,
                                      "sky": init_sky, "wall_s": time.time() - t0})
    if p.n == 1:
        print("[npass] n=1: done (product = the pass-1 cal).", flush=True)
        return {"products": {1: init_cals}, "final": init_sky}
    if init_sky is None:
        raise ValueError("pass 1 produced no stitched sky (a partial tiled run?); "
                         "the N-pass solve needs the full pass-1 field for pass 2")

    # ---- passes >= 2 ----------------------------------------------------------
    run.resolve_frames(res)
    products = {1: init_cals}
    prev_offsets = init_cals          # list of cals holding offsets/map_0 + frame_scalar
    prev_sky = init_sky
    prev_sky_product = None           # last SKY-pass product (for step norms)
    prev_off_product = None
    final = init_sky
    for i, t in zip(range(2, p.n + 1), sched[1:]):
        t0 = time.time()
        if t == "sky":
            out = run.product(i, "sky")
            if os.path.exists(out):
                print(f"[npass] pass {i} SKY: product exists, skipping ({out})", flush=True)
            else:
                run.sky_pass(i, prev_offsets, out)
                announce(spec, 'pass', out, job=run.job, index=i, pass_type='sky')
            mon = sky_monitors(out, prev_sky_product or init_sky, fisher_min=run.lft)
            prev_sky = prev_sky_product = out
            final = out
        else:
            out = run.product(i, "off")
            if os.path.exists(out):
                print(f"[npass] pass {i} OFFSET: product exists, skipping ({out})", flush=True)
                mon = {}
            else:
                _, mon = run.offset_pass(i, prev_sky, out)
                announce(spec, 'pass', out, job=run.job, index=i, pass_type='off')
            mon = dict(mon, **offset_monitors(out, prev_off_product))
            prev_offsets = [out]
            prev_off_product = out
        products[i] = out
        rec = {"pass": i, "type": t, "product": out, "wall_s": time.time() - t0, "monitor": mon}
        append_monitor(run.monitor_path, rec)
        print(f"[npass] pass {i} {t.upper()} done ({time.time()-t0:.0f}s): {mon}", flush=True)
        if t == "sky" and p.stop_tol > 0 and prev_sky_product is not None:
            steps = [v.get("step_rms") for v in mon.values() if isinstance(v, dict)]
            steps = [s for s in steps if s is not None]
            if steps and max(steps) < p.stop_tol:
                print(f"[npass] stop_tol reached after pass {i} (max step {max(steps):.3e} < {p.stop_tol})",
                      flush=True)
                break
    print(f"[npass] DONE. final product: {final}", flush=True)
    return {"products": products, "final": final, "monitor": run.monitor_path}
