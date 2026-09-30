"""N-pass alternating solve (task = ``npass``) — a scheduler over the engine.

One formalism for the spectral calibrations: J sky blocks per pixel (continuum +
N line amplitudes), a per-frame polynomial offset in the chunk row coordinate
per column plus a per-frame scalar. The joint problem is solved by
**alternating least squares**, each half solved exactly, in a schedule of three
pass types::

    INIT    pass 1      S, a, s jointly   — joint LSQR (a plain or tiled ``cal`` run)
    SKY     even passes S | a, s          — per-tile moment dumps -> one closed-form solve
    OFFSET  odd passes  a, s | S          — per-frame dense least squares, one global sky

Schedule for ``[passes].n = N``: INIT, SKY, OFFSET, SKY, OFFSET, ...  With
``n = 1`` this task IS the single solve of task ``cal`` — same mode, same
tiling, same clip — and reproduces it byte for byte (regression gate). ``n = 2``
is the "two-pass", ``n = 4`` the SEP recipe.

Every SKY/OFFSET pass is an exact block minimization, so the joint objective is
non-increasing in N. More passes are not automatically better: along the exact
null spaces (uniform line floor <-> static detector pattern; uniform sky <->
per-frame scalars) the objective is flat and the iterate can drift, so the
scheduler records per-pass monitors (residual RMS, step norms, gauge indicators)
in ``<stem>_npass_monitor.json`` and ``[passes].stop_tol`` can stop early.
Each pass writes a product (``<stem>_pass{i}sky.h5`` / ``_pass{i}off.h5``);
a re-run skips passes whose product exists.

Config (``[passes]``)::

    n         = 4          # number of passes
    order     = "sky_first"  # 'sky_first' (INIT,SKY,OFFSET,...) | 'offset_first' (INIT,OFFSET,SKY,...)
    stop_tol  = 0.0        # > 0: stop after a SKY pass whose step RMS (all blocks) is below it
    sky_merge = "combine"  # 'combine' (exact, additive moments) | 'stitch' (Fisher; legacy)
    keep_moments = false   # true keeps each SKY pass's per-tile dumps (~23 GB each at J=4)
    init   = { outlier_thresh = 2.5, subch_clip = true, ignore_list = [21] }   # pass-1 clip
    sky    = { outlier_thresh = 5.0, subch_clip = true }
    offset = { poly_degree = 4, outlier_thresh = 2.5, subch_clip = true,
               bright_cut = 0.05, min_pix = 5000 }

Pass 1 is the ``cal`` task on the same config (tiled when ``[tiling]`` is
present; its tiles are then also the memory tiling of every SKY pass);
``[passes].init`` overrides only the clip-related ``[calibration]`` keys for
that pass (the iteration count stays in ``[lsqr]``). Frames for the SKY passes
are re-staged per tile; the OFFSET passes read every frame of the field from
``[tiling].full_reproj_dir`` (or the pass-1 frame list).

The two mode hooks the passes use — ``clip_group_edges`` (the grouped outlier
clip of ``subch_clip``) and ``refit_poly_basis`` (the OFFSET refit's per-frame
polynomial) — are defined on :class:`~.modes.base.CalMode`.
"""
from __future__ import annotations

import dataclasses
import gc
import os
import time

from . import staging
from .engine import RunContext, tile_assignment

__all__ = ["schedule", "describe_schedule", "run_npass"]

PASS_TYPES = ("init", "sky", "offset")
_SKY_DEFAULTS = dict(outlier_thresh=5.0, subch_clip=True)
_OFFSET_DEFAULTS = dict(poly_degree=4, outlier_thresh=2.5, subch_clip=True,
                        bright_cut=0.05, min_pix=5000, segments=None, ridge=0.0)


def schedule(n: int, order: str = "sky_first") -> list[str]:
    """Pass types for an ``n``-pass run: INIT, then alternating SKY / OFFSET.

    ``order`` picks which half runs first after INIT:

    ``"sky_first"`` (INIT, SKY, OFFSET, SKY, ...) goes straight to the exact
        sky. Its pass-2 product inherits pass 1's per-tile offsets, which were
        solved jointly with a per-tile sky: within a tile each frame's offset
        error was compensated by that tile's own sky, and the Fisher stitch
        blended the boundaries. Freezing those offsets and deriving ONE sky
        removes the compensation, so pass-1 offset error prints into the sky
        as footprint arcs, concentrated where the tiles' gauges disagree (NEP
        D4, 2026-09-10: 4-5x excess within ~1000 px of a tile edge).

    ``"offset_first"`` (INIT, OFFSET, SKY, OFFSET, ...) re-levels every frame
        against the single stitched INIT sky BEFORE any exact sky solve, so
        the first SKY pass already sees globally consistent offsets. Costs one
        extra OFFSET pass; makes a cheap low-n run usable.
    """
    n = int(n)
    if n < 1:
        raise ValueError(f"[passes].n must be >= 1, got {n}")
    if order not in ("sky_first", "offset_first"):
        raise ValueError(f"[passes].order must be 'sky_first' or 'offset_first', got {order!r}")
    first, second = ("offset", "sky") if order == "offset_first" else ("sky", "offset")
    return ["init"] + [first if i % 2 == 0 else second for i in range(2, n + 1)]


def _init_cfg(cfg, edges_fn=None):
    """The pass-1 config: the run config with ``[passes].init`` clip overrides
    applied to ``[calibration]``. Nothing else changes, so ``n = 1`` with no
    overrides is exactly the ``cal`` task."""
    over = dict(cfg.passes.get("init", {}))
    cal = dict(cfg.calibration)
    for k in ("outlier_thresh", "ignore_list"):
        if k in over:
            cal[k] = over[k]
    if over.get("subch_clip"):
        if edges_fn is None:
            raise ValueError("[passes].init.subch_clip needs the mode's clip-group edges")
        cal["outlier_group_edges"] = edges_fn()
    return dataclasses.replace(cfg, calibration=cal)


def describe_schedule(cfg) -> list[str]:
    """Human-readable schedule for ``--dry-run``."""
    p = cfg.passes
    n = int(p.get("n", 4))
    order = p.get("order", "sky_first")
    sched = schedule(n, order)
    how1 = "tiled (" + ("explicit tiles" if cfg.tiling.get("tiles") else "grid") + ")" \
        if cfg.tiling else "single cal"
    sky = dict(_SKY_DEFAULTS, **p.get("sky", {}))
    off = dict(_OFFSET_DEFAULTS, **p.get("offset", {}))
    lines = [f"npass: n={n}, order={order}, sky_merge={p.get('sky_merge', 'combine')}, "
             f"stop_tol={p.get('stop_tol', 0.0)}"]
    for i, t in enumerate(sched, start=1):
        if t == "init":
            over = p.get("init", {})
            lines.append(f"  pass {i}: INIT   joint LSQR iter_lim={cfg.lsqr.get('iter_lim')} "
                         f"via {how1}; clip overrides {over or '(none: legacy clip)'}"
                         + ("  -> product = the legacy cal" if n == 1 else ""))
        elif t == "sky":
            lines.append(f"  pass {i}: SKY    closed form given pass-{i-1} offsets {sky}")
        else:
            src = "the stitched INIT sky" if i == 2 else f"pass-{i-1} sky"
            lines.append(f"  pass {i}: OFFSET per-frame refit given {src} {off}")
    return lines


# --------------------------------------------------------------------------- #
class _Run:
    """Everything the SKY/OFFSET passes share (resolved once via the RunContext)."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.p = cfg.passes
        self.ctx = ctx = RunContext.build(cfg)
        self.inst, self.mode = ctx.inst, ctx.mode
        self.geom = ctx.geom
        self.job = ctx.single_job("task 'npass'")
        self.jobgeom = ctx.job_geometry(self.job)
        self.sky_model = self.mode.build_sky_model(cfg, self.inst, self.geom)
        self.det_aux, self.aux_keys = ctx.aux_maps()
        self.cm = self.geom.chunk_map.det
        self.grid_valid = self.jobgeom.det_valid_weight
        base = cfg.tiling["stitched_suffix"] if cfg.tiling else cfg.suffix
        self.stem = "cal_" + ctx.stem(self.job, base)
        self.cal_dir = ctx.pipeline_config.cal_dir
        self.work_dir = os.path.join(cfg.cache_dir, f"npass_{self.stem}")
        os.makedirs(self.work_dir, exist_ok=True)
        self.monitor_path = os.path.join(self.cal_dir, f"{self.stem}_npass_monitor.json")
        self._edges = None
        calk = ctx.cal_kwargs
        self.damp_weight = float(calk.get("damp_weight", 0.0))
        self.damp_weight_line = calk.get("damp_weight_line", None)
        self.max_workers = int(calk.get("max_workers", 48))
        self.lft = cfg.params.get("line_fisher_threshold", 10.0)
        self.tile_frames = None       # {tile_name: [hdd paths]}
        self.all_frames = None        # [hdd paths] of the whole field

    def variables_for(self, frames):
        """The model's data variables over ``frames`` beyond the detector maps
        (frame values, sky maps, layers, functions); None for the historical
        recipes, which read detector maps only."""
        spec = self.mode.spec(self.cfg, self.inst, self.geom)
        if any(t.coefficient is not None or t.basis is not None for t in spec.offset):
            raise ValueError("the N-pass OFFSET pass refits a polynomial basis per frame; a model whose "
                             "offset terms carry a coefficient or a basis runs as task 'cal'")
        if not spec.variables and not (spec.referenced_variables()
                                       & set(self.mode.frame_variable_names(self.cfg, self.inst))):
            return None
        from selfcal.geometry import wcs_helper
        ref_wcs, ref_shape = wcs_helper.load_from_fits(self.ctx.pipeline_config.ref_path)
        return self.mode.build_variables(self.cfg, self.inst, self.geom, list(frames), ref_shape=ref_shape,
                                         ref_wcs=ref_wcs)

    def edges(self):
        if self._edges is None:
            self._edges = self.mode.clip_group_edges(self.cfg, self.inst, self.geom)
        return self._edges

    def product(self, i, kind):
        return os.path.join(self.cal_dir, f"{self.stem}_pass{i}{kind}.h5")

    # ---- frames -------------------------------------------------------------
    def resolve_frames(self, init_result):
        cfg = self.cfg
        if cfg.tiling:
            assignment = init_result.assignment if init_result else None
            if assignment is None:
                _, _, _, assignment = tile_assignment(cfg.tiling, tuple(cfg.tiling["ref_shape"]))
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
            import h5py, hdf5plugin  # noqa: F401
            cal = init_result.cal_paths[0]
            with h5py.File(cal, "r") as f:
                reproj = [r.decode() if isinstance(r, bytes) else str(r) for r in f["reproj_list"][:]]
            src = cfg.reproj_override or os.path.dirname(reproj[0])
            frames = [os.path.join(src, os.path.basename(r)) for r in reproj]
            self.tile_frames = {"all": frames}
            self.all_frames = list(frames)

    # ---- SKY pass -----------------------------------------------------------
    def sky_pass(self, i, offsets_cals, out):
        from selfcal.pipeline import pipeline_wrapper
        from selfcal.models.offset_model import OffsetModel
        from selfcal.pipeline.npass import (OffsetSubtractor, dump_moments, combine_moments,
                                           sky_damp_weights)
        cfg, ctx = self.cfg, self.ctx
        opts = dict(_SKY_DEFAULTS, **self.p.get("sky", {}))
        merge = self.p.get("sky_merge", "combine")
        edges = self.edges() if opts.get("subch_clip") else None
        calk = dict(ctx.cal_kwargs)
        for k in ("outlier_thresh", "outlier_group_edges", "outlier_subchannel_edges", "postprocess_func"):
            calk.pop(k, None)
        if not (calk.get("outlier_group_variable") or calk.get("outlier_aux_key")):
            calk["outlier_group_variable"] = self.geom.wavelength_key
        dws = sky_damp_weights(self.sky_model, self.damp_weight, self.damp_weight_line)
        nvme = (ctx.tiling_nvme_dir() if cfg.tiling
                else (cfg.reproj_override or os.path.dirname(self.all_frames[0])))
        os.makedirs(nvme, exist_ok=True)
        mom_dir = os.path.join(self.work_dir, "moments"); os.makedirs(mom_dir, exist_ok=True)
        subtract = OffsetSubtractor(offsets_cals)
        pieces = []
        for name, files in self.tile_frames.items():
            piece = (os.path.join(mom_dir, f"pass{i}_{name}.npz") if merge == "combine"
                     else os.path.join(self.cal_dir, f"{self.stem}_pass{i}sky_{name}.h5"))
            if os.path.exists(piece):
                print(f"[npass] pass {i} SKY [{name}]: exists, skipping ({os.path.basename(piece)})",
                      flush=True)
                pieces.append(piece); continue
            t0 = time.time()
            if cfg.tiling:
                staging.stage_files(files, nvme, cfg.hdd_io_limit)
                frame_list = sorted(os.path.join(nvme, os.path.basename(f)) for f in files)
            else:
                frame_list = list(files)
            print(f"\n[npass] pass {i} SKY [{name}]: {len(frame_list)} frames, K=0 closed form, "
                  f"clip {opts['outlier_thresh']}{' per-subch' if edges is not None else ''}",
                  flush=True)
            cc = pipeline_wrapper.Calibrator(ctx.pipeline_config, reproj_dir=nvme)
            cc.reproj_list = frame_list
            variables = self.variables_for(frame_list)
            if variables is not None:
                calk['variables'] = variables
            cc.setup_lsqr(offset_model=OffsetModel.sky_only(), grid_valid_weight=self.grid_valid,
                          oversample_factor=1,
                          sky_model=self.sky_model, det_aux=self.det_aux, aux_keys=self.aux_keys,
                          postprocess_func=subtract, outlier_thresh=float(opts["outlier_thresh"]),
                          outlier_group_edges=edges,
                          sky_rhs_moments=True, batch_spill_dir=cfg.cache_dir, **calk)
            if merge == "combine":
                dump_moments(cc, piece)
            else:
                cc.solve_sky_closed_form(damp_weight=self.damp_weight,
                                         damp_weight_line=self.damp_weight_line)
                self.mode.configure(cfg, cc)
                cc.reproj_list = staging.remap_to_nvme(cc.reproj_list, ctx.pipeline_config.reproj_dir)
                cc.save_calibration(cal_file=os.path.basename(piece))
            del cc
            gc.collect()
            print(f"[npass] pass {i} SKY [{name}] done ({time.time()-t0:.0f}s)", flush=True)
            pieces.append(piece)
        if merge == "combine":
            combine_moments(pieces, out, sky_names=list(self.sky_model.names), damp_weights=dws,
                            line_fisher_threshold=self.lft, attrs={"npass_pass": i})
            # The dumps are pure intermediates and they are BIG (a J=4 full-NEP
            # tile dump is ~23 GB, so one n=8 run is ~0.5 TB and a 16-tile one
            # ~1.4 TB). Once the combine has written the product they have no
            # further use -- a resumed run skips the whole pass on the product's
            # existence, not on theirs. Keeping them filled the 3.6 TB root disk
            # mid-run on 2026-09-10 and killed pass 4 with ENOSPC.
            if not self.p.get("keep_moments", False):
                freed = 0
                for q in pieces:
                    try:
                        freed += os.path.getsize(q)
                        os.remove(q)
                    except OSError:
                        pass
                print(f"[npass] pass {i} SKY: removed {len(pieces)} moment dumps "
                      f"({freed/1e9:.0f} GB freed); set [passes].keep_moments = true "
                      f"to retain them", flush=True)
        else:
            from selfcal.pipeline.tiled import stitch
            ref_shape = tuple(cfg.tiling["ref_shape"]) if cfg.tiling else None
            stitch(pieces, out, ref_shape=ref_shape, line=self.sky_model.n_blocks >= 2)
        return out

    # ---- OFFSET pass --------------------------------------------------------
    def offset_pass(self, i, sky_cal, out):
        from selfcal.pipeline.npass import SkySubtractor, refit_offsets_per_frame
        opts = dict(_OFFSET_DEFAULTS, **self.p.get("offset", {}))
        pb = self.mode.refit_poly_basis(self.cfg, self.inst, self.geom,
                                        degree=int(opts["poly_degree"]),
                                        segments=opts.get("segments"))
        variables = self.variables_for(self.all_frames)
        sky = SkySubtractor(sky_cal, self.sky_model,
                            export_dir=os.path.join(self.work_dir, f"sky_pass{i-1}"),
                            aux_keys=tuple(self.aux_keys or ()), variables=variables,
                            frames=self.all_frames if variables is not None else None)
        edges = self.edges() if opts.get("subch_clip") else None
        _, mon = refit_offsets_per_frame(
            self.all_frames, sky, det_chunk_map=self.cm, grid_valid=self.grid_valid,
            det_aux=self.det_aux, poly_basis=pb, edges=edges, edges_key=self.geom.wavelength_key,
            ignore_list=self.cfg.calibration.get("ignore_list", []),
            thresh=float(opts["outlier_thresh"]), bright_cut=opts.get("bright_cut"),
            min_pix=int(opts["min_pix"]), out_h5=out, max_workers=self.max_workers,
            ridge=float(opts.get("ridge") or 0.0),
            attrs={"npass_pass": i})
        return out, mon


# --------------------------------------------------------------------------- #
def run_npass(cfg, *, run_calibration):
    """Run the schedule. ``run_calibration`` is the ``cal`` task (plain or tiled
    per the config), used for the INIT pass."""
    from selfcal.pipeline.npass import sky_monitors, offset_monitors, append_monitor
    p = cfg.passes
    n = int(p.get("n", 4))
    sched = schedule(n, p.get("order", "sky_first"))
    for line in describe_schedule(cfg):
        print(f"[npass] {line}", flush=True)
    if sched[-1] == "offset":
        print(f"[npass] NOTE: the schedule ends on an OFFSET pass, which writes offsets but no "
              f"sky map -- the last sky product will be from pass {len(sched) - 1}. Use an odd n "
              f"with order='offset_first' (or an even n with 'sky_first') to end on a sky.",
              flush=True)
    run = _Run(cfg)

    # ---- pass 1: INIT = the cal task -------------------------------------------
    t0 = time.time()
    cfg1 = _init_cfg(cfg, edges_fn=run.edges)
    res = run_calibration(cfg1)
    init_cals = list(res.cal_paths)
    init_sky = res.sky_path
    append_monitor(run.monitor_path, {"pass": 1, "type": "init", "products": init_cals,
                                      "sky": init_sky, "wall_s": time.time() - t0})
    if n == 1:
        print("[npass] n=1: done (product = the pass-1 cal).", flush=True)
        return {"products": {1: init_cals}, "final": init_sky}
    if init_sky is None:
        raise ValueError("pass 1 produced no stitched sky (partial tiled run?); "
                         "npass needs the full pass-1 field for pass 2")

    # ---- passes >= 2 ----------------------------------------------------------
    run.resolve_frames(res)
    products = {1: init_cals}
    prev_offsets = init_cals          # list of cals holding offsets/map_0 + frame_scalar
    prev_sky = init_sky
    prev_sky_product = None           # last SKY-pass product (for step norms)
    prev_off_product = None
    final = init_sky
    for i, t in zip(range(2, n + 1), sched[1:]):
        t0 = time.time()
        if t == "sky":
            out = run.product(i, "sky")
            if os.path.exists(out):
                print(f"[npass] pass {i} SKY: product exists, skipping ({out})", flush=True)
            else:
                run.sky_pass(i, prev_offsets, out)
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
            mon = dict(mon, **offset_monitors(out, prev_off_product))
            prev_offsets = [out]
            prev_off_product = out
        products[i] = out
        rec = {"pass": i, "type": t, "product": out, "wall_s": time.time() - t0, "monitor": mon}
        append_monitor(run.monitor_path, rec)
        print(f"[npass] pass {i} {t.upper()} done ({time.time()-t0:.0f}s): {mon}", flush=True)
        tol = float(p.get("stop_tol", 0.0))
        if t == "sky" and tol > 0 and prev_sky_product is not None:
            steps = [v.get("step_rms") for v in mon.values() if isinstance(v, dict)]
            steps = [s for s in steps if s is not None]
            if steps and max(steps) < tol:
                print(f"[npass] stop_tol reached after pass {i} (max step {max(steps):.3e} < {tol})",
                      flush=True)
                break
    print(f"[npass] DONE. final product: {final}", flush=True)
    return {"products": products, "final": final, "monitor": run.monitor_path}
