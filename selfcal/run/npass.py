"""N-pass alternating solve (task ``npass``) — a scheduler over the engine.

The joint problem -- the model's sky terms ``S`` (one map per term) and its offsets ``a`` (whatever
offset terms the model has, and the per-frame scalar) -- is solved by **alternating least squares**,
each half solved exactly, in a schedule of three pass types::

    INIT    pass 1      S, a jointly   — the joint solve (a plain or tiled ``cal`` run)
    SKY     SKY passes  S | a          — per-tile moment dumps summed -> one closed-form solve per pixel
    OFFSET  the others  a | S          — the model's offset system with the one global sky held fixed

Both kinds of pass give the full-field solution; tiles only bound their memory, and neither builds
the matrix. A SKY pass sums the tiles' per-pixel moments before it solves anything. Given the sky, the
offsets of different frames are independent (a term of one frame sees only that frame's data), so an
OFFSET pass solves every frame on its own, from the frame's normal equations. A term shared across
frames (``per="all"``, a frame group) would tie the frames together; the plan refuses one.

What a pass computes is :mod:`selfcal.npass` (:mod:`~selfcal.npass.sky`, :mod:`~selfcal.npass.offsets`)
and :class:`~selfcal.npass.passes.PassRunner` runs it over the tiles; this module is the schedule.
Nothing in the passes knows the form of the offsets: they read the model's rows from the engine's row
assembly (:func:`~selfcal.run.engine.setup_job`) and subtract the offsets with the solve's own chunk
weights and functions. The offset terms are chosen in the run script (the recipe's model) and nowhere
else.

The schedule is INIT, then SKY and OFFSET passes alternately, ``order`` saying which comes first
(:func:`~selfcal.run.schedule.schedule`). With ``n = 1`` this task IS the single solve of task
``cal`` — same model, same tiling, same clip — and reproduces it byte for byte (regression gate).

Every SKY/OFFSET pass is an exact block minimization of the joint objective, so the objective is
non-increasing in N. More passes are not automatically
better: along the exact null spaces (uniform line floor <-> static detector pattern; uniform sky
<-> per-frame scalars) the objective is flat and the iterate can drift, so the scheduler records
per-pass monitors (residual RMS, step norms, gauge indicators) in ``<stem>_npass_monitor.json`` and
``stop_tol`` can stop early. Each pass writes a product (``<stem>_pass{i}sky.h5`` /
``_pass{i}off.h5``); a re-run skips passes whose product exists. An OFFSET pass also keeps its
tiles' solutions (``<stem>_pass{i}off_<tile>.h5``, each a cal of the offsets alone), which a
resumed pass reuses.

The settings are a :class:`~selfcal.run.runspec.PassesSpec` (from
:class:`~selfcal.run.schedule.Passes`). Pass 1 is the ``cal`` task on the same run (tiled when the
run has tiles; its tiles are then also the memory tiling of every SKY and OFFSET pass); the INIT
clip replaces only the clip of the run's ``setup_lsqr`` keywords for that pass. Frames are
re-staged per tile. Every pass damps the sky terms as the INIT does
(:attr:`~selfcal.run.engine.RunContext.sky_damping`). The grouped clips use
:meth:`~selfcal.run.engine.RunContext.clip_group_edges`.
"""
from __future__ import annotations

import os
import time

from .engine import RunContext, announce
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
            lines.append(f"  pass {i}: OFFSET the model's offset terms given {src} (sky held fixed), "
                         f"every frame solved directly; "
                         f"clip {_clip(p.refit_clip)}, bright_cut {p.bright_cut}, min_pixels {p.min_pixels}")
    return lines


# --------------------------------------------------------------------------- #
def run_npass(spec, ctx=None, *, run_calibration):
    """Run the schedule. ``run_calibration`` is the ``cal`` task (plain or tiled
    per the run), used for the INIT pass."""
    from selfcal.npass.offsets import offset_monitors
    from selfcal.npass.passes import PassRunner, append_monitor
    from selfcal.npass.sky import sky_monitors
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
    run = PassRunner(spec, ctx)

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
