"""Pass 4 (or a stand-alone pass 2): K=0 sky-only solve in closed form, with
optional per-tile MOMENT DUMPS for a seam-free full-field combination.

Same system as ``two_pass.run_pass2`` — the given offsets (pass 1 or pass 3)
are subtracted through the ``OffsetSubtractor`` POSTprocess hook, per-subchannel
clip at ``pass2_thresh``, same damping — solved per pixel in closed form.

``--dump-moments out.npz`` saves this tile's per-pixel normal-equation moments
instead of solving. They are ADDITIVE across tiles (sums over observations of
w^2, w^2 G, w^2 G^2, w^2 v, w^2 G v), so ``combine_moments`` can sum them and
solve once — mathematically identical to a single full-field solve, hence no
tile seam can survive. That requires each frame to belong to exactly ONE tile:
pass ``--frame-list`` (from ``make_frame_lists``) since staged tiles overlap.

    python -m selfcal_scripts.spectral_4pass.closed_form_pass4 <config.toml> \\
        <offsets_cal.h5> <suffix> [--frame-list f.txt] [--dump-moments out.npz]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys

import numpy as np
import hdf5plugin  # noqa: F401

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from selfcal_scripts.spectral_4pass.hooks import subchannel_bc_edges  # noqa: E402
from selfcal_scripts.spectral_4pass.two_pass import setup_sky_only, cal_path_for  # noqa: E402
from selfcal_scripts.runner.config import load_config, get_instrument  # noqa: E402
from selfcal_scripts.runner.pipelines import _calibration_kwargs  # noqa: E402


def main(config_path, offsets_cal, suffix, frame_list=None, dump_moments=None):
    cfg = load_config(config_path)
    inst = get_instrument(cfg.instrument)
    edges = subchannel_bc_edges(inst, cfg)
    thresh = float(cfg.params.get("pass2_thresh", 5.0))
    calk = _calibration_kwargs(cfg)
    dw = float(calk.get("damp_weight", 0.1))
    dwl = calk.get("damp_weight_line", None)

    frame_filter = None
    if frame_list is not None:
        keep = {ln.strip() for ln in open(frame_list) if ln.strip()}
        frame_filter = lambda names: [n for n in names if n in keep]  # noqa: E731
    print(f"===== CLOSED-FORM SKY PASS: K=0 sky-only, clip {thresh} per-subch, "
          f"damp {dw}/{dwl}, suffix={suffix} =====", flush=True)
    cc, mode = setup_sky_only(cfg, offsets_cal, thresh, edges, frame_filter=frame_filter,
                              sky_rhs_moments=True)

    if dump_moments is not None:
        cc._materialize_pixel_state()
        pc = cc.pixel_cross
        cross = pc if not isinstance(pc, dict) else pc[(0, 1)]
        np.savez(dump_moments,
                 pixel_counts=np.asarray(cc.pixel_counts, dtype=np.float64),
                 pixel_fisher=np.asarray(cc.pixel_fisher, dtype=np.float64),
                 pixel_cross=np.asarray(cross, dtype=np.float64),
                 pixel_rhs=np.asarray(cc.pixel_rhs, dtype=np.float64),
                 num_sky_blocks=cc.num_sky_blocks,
                 ref_shape=np.asarray(cc.ref_shape),
                 n_frames=len(cc.reproj_list))
        print(f"[4pass] moments dumped -> {dump_moments}", flush=True)
        return dump_moments

    cc.solve_sky_closed_form(damp_weight=dw, damp_weight_line=dwl)
    mode.configure(cfg, cc)
    path = cal_path_for(cfg, suffix)
    cc.save_calibration(cal_file=os.path.basename(path))
    print(f"[4pass] saved: {path}", flush=True)
    return path


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("offsets_cal", help="cal (fragment) holding offsets/map_0 + frame_scalar")
    ap.add_argument("suffix")
    ap.add_argument("--frame-list", default=None, help="basenames to use (one per line)")
    ap.add_argument("--dump-moments", default=None, help="save moments to this .npz instead of solving")
    a = ap.parse_args()
    main(os.path.abspath(a.config), os.path.abspath(a.offsets_cal), a.suffix,
         frame_list=a.frame_list, dump_moments=a.dump_moments)
