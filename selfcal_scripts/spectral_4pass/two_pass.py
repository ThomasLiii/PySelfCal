"""Passes 1+2: joint offset solve, then closed-form sky given the offsets.

  PASS 1 (offset): ``pahfit_lvf_polybasis`` cal (measured template, hard
    deg-D subchannel poly-basis offset + per-frame scalar) with a STRICT
    PER-SUBCHANNEL sigma clip (``pass1_thresh``) and the full bitmask
    (``pass1_ignore_list``), so the per-frame offset is fit on clean sky.
    This is the only coupled solve in the chain and the only step that needs
    the field split into (memory-sized) staged halves.

  PASS 2 (sky): freeze pass-1's offsets (subtracted via ``OffsetSubtractor``,
    a POSTprocess hook) and solve the K=0 sky-only system — continuum + line,
    no offset columns — in per-pixel CLOSED FORM with the loose clip
    (``pass2_thresh``, per subchannel when ``pass2_subch_clip``). The K=0
    system is block-diagonal per pixel, so the closed form is the exact
    answer; LSQR at a finite ``iter_lim`` semi-converges on low-diversity
    pixels (the southern cont<->line collapse). ``--lsqr`` keeps the old path.

    python -m selfcal_scripts.spectral_4pass.two_pass <config.toml> [--maxframes N] [--lsqr]

Writes two cals under the config's suffix: ``..._p1off.h5`` and ``..._p2line.h5``.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import glob

import numpy as np
import hdf5plugin  # noqa: F401
import h5py

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from selfcal_scripts.spectral_4pass.hooks import subchannel_bc_edges, OffsetSubtractor  # noqa: E402
from selfcal_scripts.runner.config import load_config, get_instrument  # noqa: E402
from selfcal_scripts.runner.modes.base import get_mode  # noqa: E402
from selfcal_scripts.runner.pipelines import run_calibration, _make_config, _calibration_kwargs  # noqa: E402
from selfcal.pipeline import pipeline_wrapper  # noqa: E402


def cal_path_for(cfg, suffix):
    """Path of the cal file a run with this config + suffix writes."""
    inst = get_instrument(cfg.instrument)
    frame_tag = inst.frame_tag(cfg.instrument_cfg)
    job = inst.jobs(cfg.instrument_cfg)[0]
    return os.path.join(_make_config(cfg).cal_dir, f"cal_{frame_tag}_{job.name}{suffix}.h5")


def run_pass1(cfg, edges, thresh1, suffix1):
    """Joint offset+sky cal, strict per-subchannel clip, full bitmask."""
    cfg.calibration = dict(cfg.calibration)
    cfg.calibration["outlier_thresh"] = float(thresh1)
    cfg.calibration["outlier_subchannel_edges"] = edges
    p1_ignore = list(cfg.params.get("pass1_ignore_list", []))
    cfg.calibration["ignore_list"] = p1_ignore
    cfg.suffix = suffix1
    cfg.skip_mosaic = True
    print(f"\n===== PASS 1 (offset): thresh={thresh1} PER-SUBCHANNEL, "
          f"ignore_list={p1_ignore}, suffix={suffix1} =====", flush=True)
    run_calibration(cfg)
    return cal_path_for(cfg, suffix1)


def setup_sky_only(cfg, offsets_cal, thresh, edges, frame_filter=None, sky_rhs_moments=True):
    """Assemble the K=0 sky-only system with the offsets pre-subtracted.

    Returns ``(cc, mode)``. ``frame_filter(basenames) -> basenames`` restricts
    the frame set (used by the moment-dump path so each frame belongs to
    exactly one tile).
    """
    inst = get_instrument(cfg.instrument)
    mode = get_mode(cfg.mode)
    di = inst.detector_inputs(cfg.instrument_cfg, cfg.oversample)
    job = inst.jobs(cfg.instrument_cfg)[0]
    ci = inst.channel_inputs(cfg.instrument_cfg, di, job)
    sky_model = mode.build_sky_model(cfg, inst, di)
    det_aux = mode.det_aux(cfg, inst, di)
    # NATIVE window mask to match det_aux + oversample_factor=1 (as run_calibration).
    grid_valid = ci["det_valid_mask_padded"]

    cc = pipeline_wrapper.Calibrator(_make_config(cfg), reproj_dir=cfg.reproj_override)
    with h5py.File(offsets_cal, "r") as f:
        reproj = [r.decode() if isinstance(r, bytes) else str(r) for r in f["reproj_list"][:]]
    # The offsets cal may cover the WHOLE field (global pass 3) while this run
    # is one staged tile: keep only frames actually present here.
    cand = [os.path.join(cfg.reproj_override, os.path.basename(r)) for r in reproj]
    cc.reproj_list = [q for q in cand if os.path.exists(q)]
    if len(cc.reproj_list) < len(cand):
        print(f"[4pass] offsets cover {len(cand)} frames; {len(cc.reproj_list)} staged here",
              flush=True)
    if frame_filter is not None:
        keep = set(frame_filter([os.path.basename(q) for q in cc.reproj_list]))
        cc.reproj_list = [q for q in cc.reproj_list if os.path.basename(q) in keep]
        print(f"[4pass] frame filter: {len(cc.reproj_list)} frames", flush=True)

    calk = _calibration_kwargs(cfg)
    for drop in ("outlier_thresh", "outlier_subchannel_edges", "postprocess_func"):
        calk.pop(drop, None)
    cc.setup_lsqr(
        chunk_maps=[],                      # K=0 -> no offset columns
        grid_valid_weight=grid_valid,
        oversample_factor=1,
        sky_model=sky_model,
        det_aux=det_aux,
        postprocess_func=OffsetSubtractor(offsets_cal),   # AFTER weights
        outlier_thresh=float(thresh),
        outlier_subchannel_edges=edges,
        use_per_frame_scalar=False,
        sky_rhs_moments=sky_rhs_moments,
        **calk)
    return cc, mode


def run_pass2(cfg, offsets_cal, thresh2, suffix2, orig_calibration=None,
              subch_edges=None, closed_form=True):
    """K=0 sky-only solve on offset-subtracted data (closed form by default)."""
    if orig_calibration is not None:
        cfg.calibration = dict(orig_calibration)
    inst = get_instrument(cfg.instrument)
    calk = _calibration_kwargs(cfg)
    dw = float(calk.get("damp_weight", 0.1))
    dwl = calk.get("damp_weight_line", None)
    print(f"\n===== PASS 2 (sky): K=0 sky-only, {'CLOSED FORM' if closed_form else 'LSQR'}, "
          f"thresh={thresh2}, damp {dw}/{dwl}, suffix={suffix2} =====", flush=True)
    cc, mode = setup_sky_only(cfg, offsets_cal, thresh2, subch_edges,
                              sky_rhs_moments=closed_form)
    if closed_form:
        cc.solve_sky_closed_form(damp_weight=dw, damp_weight_line=dwl)
    else:
        from selfcal.core.solution import compute_x0_scalar_only
        x0 = compute_x0_scalar_only(
            cc.A, cc.b, cc.ref_shape,
            scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
            num_sky_blocks=cc.num_sky_blocks,
            active_mask=getattr(cc, "active_mask", None))
        cc.apply_lsqr(x0=x0, use_float32=True, n_threads=cfg.apply_n_threads, **cfg.lsqr)
    mode.configure(cfg, cc)
    frame_tag = inst.frame_tag(cfg.instrument_cfg)
    job = inst.jobs(cfg.instrument_cfg)[0]
    cal_file = f"cal_{frame_tag}_{job.name}{suffix2}.h5"
    cc.save_calibration(cal_file=cal_file)
    path = os.path.join(_make_config(cfg).cal_dir, cal_file)
    print(f"[4pass] PASS 2 saved: {path}", flush=True)
    return path


def main(config_path, maxframes=None, closed_form=True):
    cfg = load_config(os.path.abspath(config_path))
    base = cfg.suffix
    thresh1 = float(cfg.params.get("pass1_thresh", 2.5))
    thresh2 = float(cfg.params.get("pass2_thresh", 5.0))

    if maxframes is not None:       # quick smoke: temp symlink dir with a subset
        src = cfg.reproj_override
        files = sorted(glob.glob(os.path.join(src, "*.h5")))
        take = files[:: max(1, len(files) // maxframes)][:maxframes]
        sub = src.rstrip("/") + f"_smoke{maxframes}"
        os.makedirs(sub, exist_ok=True)
        for fp in take:
            d = os.path.join(sub, os.path.basename(fp))
            if not os.path.islink(d):
                os.symlink(os.path.realpath(fp), d)
        cfg.reproj_override = sub
        print(f"[4pass] SMOKE: {len(take)} frames -> {sub}", flush=True)

    inst = get_instrument(cfg.instrument)
    orig_calibration = dict(cfg.calibration)
    edges = subchannel_bc_edges(inst, cfg)
    p1 = run_pass1(cfg, edges, thresh1, base + "_p1off")
    with h5py.File(p1, "r") as f:
        oc = f["offset_coverage/map_0"][:]
    per_frame = oc.sum(axis=1)
    thin = int((per_frame < np.median(per_frame) * 0.05).sum())
    print(f"[4pass] pass-1 per-frame offset obs: median {np.median(per_frame):,.0f}, "
          f"min {per_frame.min():,}, frames <5% of median: {thin}", flush=True)
    p2 = run_pass2(cfg, p1, thresh2, base + "_p2line", orig_calibration=orig_calibration,
                   subch_edges=edges if cfg.params.get("pass2_subch_clip", True) else None,
                   closed_form=closed_form)
    print(f"\n[4pass] passes 1+2 DONE.\n  pass1 (offset): {p1}\n  pass2 (sky):    {p2}", flush=True)
    return p1, p2


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("--maxframes", type=int, default=None, help="smoke test on a frame subset")
    ap.add_argument("--lsqr", action="store_true", help="pass 2 with LSQR instead of the closed form")
    a = ap.parse_args()
    main(a.config, maxframes=a.maxframes, closed_form=not a.lsqr)
