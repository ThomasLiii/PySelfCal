"""Orchestrate the full 4-pass chain over staged tiles (halves).

    python -m selfcal_scripts.spectral_4pass.run_4pass --tag WE \\
        --reproj-dir /path/to/reprojected --work-dir workspace/4pass_WE \\
        configs/sep_d4_half_WEST.toml configs/sep_d4_half_EAST.toml

Steps (each pass runs in its own subprocess so peak memory is released between
steps; a step whose output already exists is skipped, so a failed chain can be
re-launched with the same command):

  1. per tile:  two_pass          -> <cal>_p1off.h5, <cal>_p2line.h5
  2. stitch p2 tiles              -> cal_<frame>_<job>_<TAG>_p2_STITCHED.h5
  3. global_pass3 (ALL frames)    -> cal_<frame>_<job>_<TAG>_GLOBALP3_offsets.h5
  4. make_frame_lists (disjoint)  -> <work>/framelists/<tile>.txt
  5. per tile:  closed_form_pass4 --dump-moments -> <work>/moments/<TAG>_<tile>.npz
  6. combine_moments              -> cal_<frame>_<job>_<TAG>_p4line_SEAMFREE.h5  (the product)

All tile configs must share instrument / mode / output settings and differ only
in ``reproj_override`` (the staged frames) and ``suffix``. The global pass 3
uses the first config for its parameters.
"""
import os
import sys
import time
import subprocess

import h5py
import hdf5plugin  # noqa: F401

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from selfcal_scripts.runner.config import load_config, get_instrument  # noqa: E402
from selfcal_scripts.runner.pipelines import _make_config  # noqa: E402

PKG = "selfcal_scripts.spectral_4pass"


def _log(summary, msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(summary, "a") as f:
        f.write(line + "\n")


def _run(summary, log_path, args):
    """Run a module step in a subprocess, stdout+stderr to log_path."""
    _log(summary, f"run: {' '.join(args)}  (log {os.path.basename(log_path)})")
    with open(log_path, "w") as lf:
        rc = subprocess.call([sys.executable, "-m", *args], stdout=lf, stderr=subprocess.STDOUT,
                             cwd=_ROOT)
    _log(summary, f"rc={rc}")
    if rc != 0:
        raise SystemExit(f"step failed (rc={rc}); see {log_path}")


def cal_naming(cfg_path):
    cfg = load_config(cfg_path)
    inst = get_instrument(cfg.instrument)
    frame_tag = inst.frame_tag(cfg.instrument_cfg)
    job = inst.jobs(cfg.instrument_cfg)[0].name
    return cfg, _make_config(cfg).cal_dir, f"cal_{frame_tag}_{job}"


def main(tag, configs, reproj_dir, work_dir, max_workers=48,
         damp_weight=0.1, damp_weight_line=0.005):
    os.makedirs(work_dir, exist_ok=True)
    logs = os.path.join(work_dir, "logs"); os.makedirs(logs, exist_ok=True)
    summary = os.path.join(work_dir, f"{tag}_summary.log")
    _log(summary, f"=== 4-pass chain '{tag}' over {len(configs)} tiles ===")

    tiles = []
    for c in configs:
        cfg, cal_dir, stem = cal_naming(c)
        name = os.path.splitext(os.path.basename(c))[0]
        tiles.append(dict(name=name, cfg_path=os.path.abspath(c), cfg=cfg, cal_dir=cal_dir,
                          stem=stem, staged=cfg.reproj_override,
                          p2=os.path.join(cal_dir, f"{stem}{cfg.suffix}_p2line.h5")))
    cal_dir, stem = tiles[0]["cal_dir"], tiles[0]["stem"]
    p2_stitched = os.path.join(cal_dir, f"{stem}_{tag}_p2_STITCHED.h5")
    gp3 = os.path.join(cal_dir, f"{stem}_{tag}_GLOBALP3_offsets.h5")
    final = os.path.join(cal_dir, f"{stem}_{tag}_p4line_SEAMFREE.h5")

    # 1. passes 1+2 per tile
    for t in tiles:
        if os.path.exists(t["p2"]):
            _log(summary, f"{t['name']}: p2 exists, skipping passes 1+2"); continue
        _run(summary, os.path.join(logs, f"{t['name']}_p12.log"),
             [f"{PKG}.two_pass", t["cfg_path"]])

    # 2. stitch the pass-2 skies
    if not os.path.exists(p2_stitched):
        from selfcal.pipeline.tiled import stitch
        with h5py.File(tiles[0]["p2"], "r") as f:
            ref_shape = tuple(f["skymap"].shape)
        _log(summary, f"stitch {len(tiles)} p2 skies -> {os.path.basename(p2_stitched)}")
        stitch([t["p2"] for t in tiles], p2_stitched, ref_shape=ref_shape, line=True, verbose=True)
    else:
        _log(summary, "stitched p2 exists, skipping")

    # 3. global pass 3 over ALL frames
    if not os.path.exists(gp3):
        _run(summary, os.path.join(logs, "gp3.log"),
             [f"{PKG}.global_pass3", tiles[0]["cfg_path"], p2_stitched, os.path.abspath(reproj_dir),
              gp3, "--max-workers", str(max_workers)])
    else:
        _log(summary, "global pass 3 exists, skipping")

    # 4. disjoint frame lists
    from selfcal_scripts.spectral_4pass.make_frame_lists import main as make_lists
    fl = make_lists(os.path.join(work_dir, "framelists"), [(t["name"], t["staged"]) for t in tiles])

    # 5. moment dumps per tile
    mom_dir = os.path.join(work_dir, "moments"); os.makedirs(mom_dir, exist_ok=True)
    dumps = []
    for t in tiles:
        out = os.path.join(mom_dir, f"{tag}_{t['name']}.npz")
        dumps.append(out)
        if os.path.exists(out):
            _log(summary, f"{t['name']}: moments exist, skipping"); continue
        _run(summary, os.path.join(logs, f"{t['name']}_moments.log"),
             [f"{PKG}.closed_form_pass4", t["cfg_path"], gp3, f"_MOM_{tag}_{t['name']}",
              "--frame-list", fl[t["name"]], "--dump-moments", out])

    # 6. combine
    _run(summary, os.path.join(logs, "combine.log"),
         [f"{PKG}.combine_moments", final, tiles[0]["p2"], *dumps,
          "--damp-weight", str(damp_weight), "--damp-weight-line", str(damp_weight_line)])
    _log(summary, f"DONE -> {final}")
    return final


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="+", help="one TOML per staged tile")
    ap.add_argument("--tag", required=True, help="name of this tiling, e.g. WE or NS")
    ap.add_argument("--reproj-dir", required=True, help="directory with ALL reprojected frames")
    ap.add_argument("--work-dir", required=True, help="where logs / framelists / moments go")
    ap.add_argument("--max-workers", type=int, default=48)
    ap.add_argument("--damp-weight", type=float, default=0.1)
    ap.add_argument("--damp-weight-line", type=float, default=0.005)
    a = ap.parse_args()
    main(a.tag, a.configs, a.reproj_dir, a.work_dir, a.max_workers, a.damp_weight, a.damp_weight_line)
