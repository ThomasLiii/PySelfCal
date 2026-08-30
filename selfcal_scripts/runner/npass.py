"""N-pass alternating solve (task = ``npass``).

One formalism for the spectral calibrations: J sky blocks per pixel (continuum +
N line amplitudes), a per-frame polynomial offset in subchannel per column plus
a per-frame scalar. The joint problem is solved by **alternating least
squares**, each half solved exactly, in a schedule of three pass types::

    INIT    pass 1      S, a, s jointly   — joint LSQR (the legacy 'cal'/'tiled' task)
    SKY     even passes S | a, s          — per-tile moment dumps -> one closed-form solve
    OFFSET  odd passes  a, s | S          — per-frame dense least squares, one global sky

Schedule for ``[passes].n = N``: INIT, SKY, OFFSET, SKY, OFFSET, ...  With
``n = 1`` this task IS the legacy single solve — same mode, same tiling, same
clip — and reproduces it byte for byte (regression gate). ``n = 2`` is the
"two-pass", ``n = 4`` the SEP recipe.

Every SKY/OFFSET pass is an exact block minimization, so the joint objective is
non-increasing in N. More passes are not automatically better: along the exact
null spaces (uniform line floor <-> static detector pattern; uniform sky <->
per-frame scalars) the objective is flat and the iterate can drift, so the
runner records per-pass monitors (residual RMS, step norms, gauge indicators)
in ``<stem>_npass_monitor.json`` and ``[passes].stop_tol`` can stop early.
Each pass writes a product (``<stem>_pass{i}sky.h5`` / ``_pass{i}off.h5``);
a re-run skips passes whose product exists.

Config (``[passes]``)::

    n         = 4          # number of passes
    stop_tol  = 0.0        # > 0: stop after a SKY pass whose step RMS (all blocks) is below it
    sky_merge = "combine"  # 'combine' (exact, additive moments) | 'stitch' (Fisher; legacy)
    init   = { outlier_thresh = 2.5, subch_clip = true, ignore_list = [21] }   # pass-1 clip
    sky    = { outlier_thresh = 5.0, subch_clip = true }
    offset = { poly_degree = 4, outlier_thresh = 2.5, subch_clip = true,
               bright_cut = 0.05, min_pix = 5000 }

Pass 1 delegates to ``run_tiled`` when ``[tiled]`` is present (its
``[tiled].tiles`` / grid is then also the memory tiling of every SKY pass),
else to ``run_calibration``; ``[passes].init`` overrides only the clip-related
``[calibration]`` keys for that pass (the iteration count stays in ``[lsqr]``).
Frames for the SKY passes are re-staged per tile; the OFFSET passes read every
frame of the field from ``[tiled].full_reproj_dir`` (or the pass-1 frame list).
"""
from __future__ import annotations

import dataclasses
import gc
import os
import time

import numpy as np

from selfcal_scripts.runner.config import get_instrument
from selfcal_scripts.runner.modes.base import get_mode
from selfcal_scripts.runner import staging

__all__ = ["schedule", "describe_schedule", "run_npass"]

PASS_TYPES = ("init", "sky", "offset")
_SKY_DEFAULTS = dict(outlier_thresh=5.0, subch_clip=True)
_OFFSET_DEFAULTS = dict(poly_degree=4, outlier_thresh=2.5, subch_clip=True,
                        bright_cut=0.05, min_pix=5000)


def schedule(n: int) -> list[str]:
    """Pass types for an ``n``-pass run: INIT, then alternating SKY / OFFSET."""
    n = int(n)
    if n < 1:
        raise ValueError(f"[passes].n must be >= 1, got {n}")
    return ["init"] + ["sky" if i % 2 == 0 else "offset" for i in range(2, n + 1)]


def _init_cfg(cfg, edges_fn=None):
    """The pass-1 config: the run config with ``[passes].init`` clip overrides
    applied to ``[calibration]``. Nothing else changes, so ``n = 1`` with no
    overrides is exactly the legacy task."""
    over = dict(cfg.passes.get("init", {}))
    cal = dict(cfg.calibration)
    for k in ("outlier_thresh", "ignore_list"):
        if k in over:
            cal[k] = over[k]
    if over.get("subch_clip"):
        if edges_fn is None:
            raise ValueError("[passes].init.subch_clip needs the instrument's subchannel edges")
        cal["outlier_subchannel_edges"] = edges_fn()
    return dataclasses.replace(cfg, calibration=cal)


def describe_schedule(cfg) -> list[str]:
    """Human-readable schedule for ``--dry-run``."""
    p = cfg.passes
    n = int(p.get("n", 4))
    sched = schedule(n)
    how1 = "tiled (" + ("explicit tiles" if cfg.tiled.get("tiles") else "grid") + ")" \
        if cfg.tiled else "single cal"
    sky = dict(_SKY_DEFAULTS, **p.get("sky", {}))
    off = dict(_OFFSET_DEFAULTS, **p.get("offset", {}))
    lines = [f"npass: n={n}, sky_merge={p.get('sky_merge', 'combine')}, "
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
            lines.append(f"  pass {i}: OFFSET per-frame refit given pass-{i-1} sky {off}")
    return lines


# --------------------------------------------------------------------------- #
class _Run:
    """Everything the SKY/OFFSET passes share (resolved once)."""

    def __init__(self, cfg):
        from selfcal_scripts.runner.pipelines import _make_config, _calibration_kwargs
        self.cfg = cfg
        self.p = cfg.passes
        self.inst = get_instrument(cfg.instrument)
        self.mode = get_mode(cfg.mode)
        self.selfcal_config = _make_config(cfg)
        self.cal_dir = self.selfcal_config.cal_dir
        self.det_inputs = self.inst.detector_inputs(cfg.instrument_cfg, cfg.oversample)
        self.job = self.inst.jobs(cfg.instrument_cfg)[0]
        self.ch_inputs = self.inst.channel_inputs(cfg.instrument_cfg, self.det_inputs, self.job)
        self.frame_tag = self.inst.frame_tag(cfg.instrument_cfg)
        self.sky_model = self.mode.build_sky_model(cfg, self.inst, self.det_inputs)
        self.det_aux = self.mode.det_aux(cfg, self.inst, self.det_inputs)
        self.cm = self.det_inputs["det_chunk_map"]
        self.ncol = int(cfg.instrument_cfg["num_col"])
        self.grid_valid = self.ch_inputs["det_valid_mask_padded"]
        base = cfg.tiled["stitched_suffix"] if cfg.tiled else cfg.suffix
        self.stem = f"cal_{self.frame_tag}_{self.job.name}{base}"
        self.work_dir = os.path.join(cfg.cache_dir, f"npass_{self.stem}")
        os.makedirs(self.work_dir, exist_ok=True)
        self.monitor_path = os.path.join(self.cal_dir, f"{self.stem}_npass_monitor.json")
        self._edges = None
        calk = _calibration_kwargs(cfg)
        self.damp_weight = float(calk.get("damp_weight", 0.0))
        self.damp_weight_line = calk.get("damp_weight_line", None)
        self.max_workers = int(calk.get("max_workers", 48))
        self.lft = cfg.params.get("line_fisher_threshold", 10.0)
        self.tile_frames = None       # {tile_name: [hdd paths]}
        self.all_frames = None        # [hdd paths] of the whole field

    def edges(self):
        if self._edges is None:
            self._edges = self.inst.subchannel_bc_edges(self.det_inputs, self.cm, self.ncol)
        return self._edges

    def product(self, i, kind):
        return os.path.join(self.cal_dir, f"{self.stem}_pass{i}{kind}.h5")

    # ---- frames -------------------------------------------------------------
    def resolve_frames(self, init_result):
        cfg = self.cfg
        if cfg.tiled:
            assignment = init_result.get("assignment") if init_result else None
            if assignment is None:
                from selfcal.pipeline.tiled import TiledCalibration, TileSpec, make_tile_grid
                import glob as glob_module
                t = cfg.tiled
                files = sorted(glob_module.glob(os.path.join(t["full_reproj_dir"],
                                                             t.get("frame_glob", "exp_*_det_00.h5"))),
                               key=lambda q: int(os.path.basename(q).split("_")[1]))
                if t.get("tiles"):
                    tiles = [TileSpec(name=s["name"], bbox=tuple(s["bbox"])) for s in t["tiles"]]
                else:
                    tiles = make_tile_grid(tuple(t["ref_shape"]), t["grid"][0], t["grid"][1],
                                           overlap_px=t["overlap_px"], names=t["tile_names"])
                only = t.get("only_tiles")
                if only:
                    tiles = [x for x in tiles if x.name in only]
                assignment = TiledCalibration(files, tiles, frame_filter=t.get("frame_filter", "center"),
                                              halo=t.get("halo", 0)).assign_frames()
            self.tile_frames = {name: list(files) for name, (files, _) in assignment.items()}
            seen = set()
            for name, files in self.tile_frames.items():
                dup = seen.intersection(os.path.basename(f) for f in files)
                if dup:
                    raise ValueError(f"tile {name}: {len(dup)} frames also in another tile — "
                                     "npass needs disjoint tiles (frame_filter='center', halo=0)")
                seen.update(os.path.basename(f) for f in files)
            self.all_frames = sorted({f for fs in self.tile_frames.values() for f in fs})
        else:
            import h5py, hdf5plugin  # noqa: F401
            cal = init_result[0]
            with h5py.File(cal, "r") as f:
                reproj = [r.decode() if isinstance(r, bytes) else str(r) for r in f["reproj_list"][:]]
            src = cfg.reproj_override or os.path.dirname(reproj[0])
            frames = [os.path.join(src, os.path.basename(r)) for r in reproj]
            self.tile_frames = {"all": frames}
            self.all_frames = list(frames)

    # ---- SKY pass -----------------------------------------------------------
    def sky_pass(self, i, offsets_cals, out):
        from selfcal.pipeline import pipeline_wrapper
        from selfcal.pipeline.npass import (OffsetSubtractor, dump_moments, combine_moments,
                                           sky_damp_weights)
        from selfcal_scripts.runner.pipelines import _calibration_kwargs
        cfg = self.cfg
        opts = dict(_SKY_DEFAULTS, **self.p.get("sky", {}))
        merge = self.p.get("sky_merge", "combine")
        edges = self.edges() if opts.get("subch_clip") else None
        calk = _calibration_kwargs(cfg)
        for k in ("outlier_thresh", "outlier_subchannel_edges", "postprocess_func"):
            calk.pop(k, None)
        dws = sky_damp_weights(self.sky_model, self.damp_weight, self.damp_weight_line)
        nvme = (os.path.join(cfg.cache_dir, cfg.tiled["nvme_subdir"]) if cfg.tiled
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
            if cfg.tiled:
                staging.stage_files(files, nvme, cfg.hdd_io_limit)
                frame_list = sorted(os.path.join(nvme, os.path.basename(f)) for f in files)
            else:
                frame_list = list(files)
            print(f"\n[npass] pass {i} SKY [{name}]: {len(frame_list)} frames, K=0 closed form, "
                  f"clip {opts['outlier_thresh']}{' per-subch' if edges is not None else ''}",
                  flush=True)
            cc = pipeline_wrapper.Calibrator(self.selfcal_config, reproj_dir=nvme)
            cc.reproj_list = frame_list
            cc.setup_lsqr(chunk_maps=[], grid_valid_weight=self.grid_valid, oversample_factor=1,
                          sky_model=self.sky_model, det_aux=self.det_aux,
                          postprocess_func=subtract, outlier_thresh=float(opts["outlier_thresh"]),
                          outlier_subchannel_edges=edges, use_per_frame_scalar=False,
                          sky_rhs_moments=True, batch_spill_dir=cfg.cache_dir, **calk)
            if merge == "combine":
                dump_moments(cc, piece)
            else:
                cc.solve_sky_closed_form(damp_weight=self.damp_weight,
                                         damp_weight_line=self.damp_weight_line)
                self.mode.configure(cfg, cc)
                cc.reproj_list = staging.remap_to_nvme(cc.reproj_list, self.selfcal_config.reproj_dir)
                cc.save_calibration(cal_file=os.path.basename(piece))
            del cc
            gc.collect()
            print(f"[npass] pass {i} SKY [{name}] done ({time.time()-t0:.0f}s)", flush=True)
            pieces.append(piece)
        if merge == "combine":
            combine_moments(pieces, out, sky_names=list(self.sky_model.names), damp_weights=dws,
                            line_fisher_threshold=self.lft, attrs={"npass_pass": i})
        else:
            from selfcal.pipeline.tiled import stitch
            ref_shape = tuple(cfg.tiled["ref_shape"]) if cfg.tiled else None
            stitch(pieces, out, ref_shape=ref_shape, line=self.sky_model.n_blocks >= 2)
        return out

    # ---- OFFSET pass --------------------------------------------------------
    def offset_pass(self, i, sky_cal, out):
        from selfcal.pipeline.npass import SkySubtractor, refit_offsets_per_frame
        opts = dict(_OFFSET_DEFAULTS, **self.p.get("offset", {}))
        p = self.cfg.params
        pb = self.inst.subchannel_poly_basis(self.cm, self.ncol, degree=int(opts["poly_degree"]),
                                             lo=int(p["subch_poly_lo"]), hi=int(p["subch_poly_hi"]))
        sky = SkySubtractor(sky_cal, self.sky_model,
                            export_dir=os.path.join(self.work_dir, f"sky_pass{i-1}"),
                            aux_keys=getattr(self.inst, "aux_keys", ("BC", "BW")))
        edges = self.edges() if opts.get("subch_clip") else None
        _, mon = refit_offsets_per_frame(
            self.all_frames, sky, det_chunk_map=self.cm, grid_valid=self.grid_valid,
            det_aux=self.det_aux, poly_basis=pb, edges=edges,
            ignore_list=self.cfg.calibration.get("ignore_list", []),
            thresh=float(opts["outlier_thresh"]), bright_cut=opts.get("bright_cut"),
            min_pix=int(opts["min_pix"]), out_h5=out, max_workers=self.max_workers,
            attrs={"npass_pass": i})
        return out, mon


# --------------------------------------------------------------------------- #
def run_npass(cfg, *, run_calibration, run_tiled):
    from selfcal.pipeline.npass import sky_monitors, offset_monitors, append_monitor
    p = cfg.passes
    n = int(p.get("n", 4))
    sched = schedule(n)
    for line in describe_schedule(cfg):
        print(f"[npass] {line}", flush=True)
    run = _Run(cfg)

    # ---- pass 1: INIT = the legacy task --------------------------------------
    t0 = time.time()
    cfg1 = _init_cfg(cfg, edges_fn=run.edges)
    if cfg.tiled:
        res = run_tiled(cfg1)
        init_cals = list(res["tiles"].values()) if isinstance(res["tiles"], dict) else list(res["tiles"])
        init_sky = res["stitched"] or (init_cals[0] if len(init_cals) == 1 else None)
    else:
        res = run_calibration(cfg1)
        init_cals = list(res)
        init_sky = init_cals[0]
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
