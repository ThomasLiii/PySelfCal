"""Pass 3: per-frame offset refit against ONE fixed global sky.

With the sky fixed (the stitched pass-2 sky is subtracted through the
``SkySubtractor`` POSTprocess hook), each frame's offset unknowns — a degree-D
Chebyshev shape per column plus the DC scalar, ``num_col*D + 1`` per frame —
are constrained only by that frame's own pixels. The problem is exactly
independent per frame, so it is solved for the WHOLE field at once: a dense
per-frame least squares (``np.linalg.lstsq``) in a process pool, no tiles, no
LSQR, and no per-tile offset<->sky gauge to disagree at seams (every frame is
levelled against the same sky).

Data prep matches the other passes: Poisson weights on the raw data, then the
sky subtraction, then the per-subchannel clip at ``pass3_thresh``. Pixels whose
modelled sky exceeds ``BRIGHT_CUT`` are excluded from the offset fit (bright
regions have large absolute sky error, which each frame would otherwise absorb
into its offset and pass 4 would print as per-frame fringes); frames left with
fewer than ``MIN_PIX`` pixels fall back to all of them (flagged).

    python -m selfcal_scripts.spectral_4pass.global_pass3 <config.toml> \\
        <stitched_p2_sky.h5> <reproj_dir> <out_offsets.h5> [--max-workers N]

Output: an ``OffsetSubtractor``-compatible cal fragment (``offsets/map_0`` per
frame per chunk, ``frame_scalar``, ``chunk_maps/map_0``, ``reproj_list``,
``fit_ok``) covering every frame in ``reproj_dir``.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import glob
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import hdf5plugin  # noqa: F401
import h5py

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from selfcal_scripts.spectral_4pass.hooks import subchannel_bc_edges, SkySubtractor  # noqa: E402
from selfcal_scripts.runner.config import load_config, get_instrument  # noqa: E402
from selfcal_scripts.runner.modes.base import get_mode  # noqa: E402
from selfcal.core.subframe import _prep_subframe  # noqa: E402
from selfcal.geometry.map_helper import find_outliers_grouped  # noqa: E402
from selfcal.models.offset_basis import cheb_shape_basis  # noqa: E402

BRIGHT_CUT = 0.05          # MJy/sr of modelled sky (cont + line*G) above which a pixel is dropped
MIN_PIX = 5000             # below this many faint pixels, use all pixels instead
_G = {}                    # per-worker state (filled by _init)


def _init(cfg_path, sky_cal, tpl_path):
    cfg = load_config(cfg_path)
    inst = get_instrument(cfg.instrument)
    mode = get_mode(cfg.mode)
    di = inst.detector_inputs(cfg.instrument_cfg, cfg.oversample)
    job = inst.jobs(cfg.instrument_cfg)[0]
    ci = inst.channel_inputs(cfg.instrument_cfg, di, job)
    ncol = int(cfg.instrument_cfg["num_col"])
    _G["cm"] = mode.build_offset_model(cfg, inst, di, ci, job, 1).chunk_maps[0]
    _G["grid_valid"] = ci["det_valid_mask_padded"]
    _G["det_aux"] = mode.det_aux(cfg, inst, di)
    _G["edges"] = subchannel_bc_edges(inst, cfg)
    _G["ignore"] = list(cfg.calibration.get("ignore_list", []))
    _G["thresh"] = float(cfg.params.get("pass3_thresh", 2.5))
    lo, hi = int(cfg.params["subch_poly_lo"]), int(cfg.params["subch_poly_hi"])
    deg = int(cfg.params.get("pass3_poly_degree", 4))
    _G["lo"], _G["hi"], _G["deg"], _G["ncol"] = lo, hi, deg, ncol
    n_chunks = int(_G["cm"].max()) + 1
    sub = np.arange(n_chunks) // ncol
    col = np.arange(n_chunks) % ncol
    _G["n_chunks"], _G["sub"], _G["col"] = n_chunks, sub, col
    _G["Bc"] = cheb_shape_basis(sub.astype(float), deg, lo, hi)     # (n_chunks, deg)
    _G["subtract"] = SkySubtractor(sky_cal, tpl_path)                # lazy load per worker


def _fit_frame(path):
    """Min-norm LSQ of (data - global sky) on [per-column deg-D shape, DC].

    Returns ``(basename, (offsets, scalar) | None, resid_rms, n_used)`` with
    ``n_used < 0`` flagging the all-pixels fallback.
    """
    try:
        g = _G
        ref_coords, sub_data, sub_weight, contribs, sub_aux = _prep_subframe(
            file=path, chunk_offsets=None, for_lsqr=True, det_offset_funcs=None,
            det_aux=g["det_aux"], chunk_maps=[g["cm"]],
            apply_weight=True, apply_mask=True, ignore_list=g["ignore"],
            grid_valid_weight=g["grid_valid"], oversample_factor=1,
            valid_threshold=0.5, postprocess_func=g["subtract"], preprocess_func=None)
        sub_h, sub_w = sub_data.shape
        valid = sub_weight > 0
        masked = np.where(valid, sub_data, np.nan)
        groups = np.digitize(sub_aux[0], g["edges"])
        valid &= ~find_outliers_grouped(masked, groups, threshold=g["thresh"])
        # bright-sky exclusion, with the same edge-safe window the subtraction
        # used; pixels off the map have no sky to subtract -> excluded
        sub = g["subtract"]
        cw, lw, on_map = sub.window(ref_coords, sub_data.shape)
        valid &= on_map
        sky_pred = cw + lw * sub.line_coeff(sub_aux)
        faint = valid & (sky_pred < BRIGHT_CUT)
        n_faint = int(faint.sum())
        used_all = n_faint < MIN_PIX
        if not used_all:
            valid = faint
        vc = np.nonzero(valid)
        pix = vc[0] * sub_w + vc[1]
        v = sub_data[vc]
        w = sub_weight[vc]
        n = v.size
        if n < 200:
            return os.path.basename(path), None, np.nan, n
        cc = contribs[0][:, pix].tocoo()
        nz = cc.data != 0
        chunk_i, obs_i, cval = cc.row[nz], cc.col[nz], cc.data[nz]
        deg, ncol = g["deg"], g["ncol"]
        D = np.zeros((n, ncol * deg + 1))
        wc = w[obs_i] * cval
        for d in range(deg):
            np.add.at(D, (obs_i, g["col"][chunk_i] * deg + d), wc * g["Bc"][chunk_i, d])
        D[:, -1] = w                                           # DC scalar
        a, *_ = np.linalg.lstsq(D, w * v, rcond=None)
        off = np.zeros(g["n_chunks"], dtype=np.float32)
        for d in range(deg):
            off += (a[g["col"] * deg + d] * g["Bc"][:, d]).astype(np.float32)
        resid = w * v - D @ a
        return (os.path.basename(path), (off, np.float32(a[-1])), float(np.std(resid)),
                n if not used_all else -n)
    except Exception:
        return os.path.basename(path), None, np.nan, -1


def main(cfg_path, sky_cal, reproj_dir, out_h5, max_workers=48):
    cfg = load_config(cfg_path)
    tpl = cfg.params["line_template_npz"]
    frames = sorted(glob.glob(os.path.join(reproj_dir, "*.h5")))
    print(f"[gp3] {len(frames)} frames | sky {os.path.basename(sky_cal)}", flush=True)
    _init(cfg_path, sky_cal, tpl)          # parent copy for cm / n_chunks
    n_chunks = _G["n_chunks"]
    offsets = np.zeros((len(frames), n_chunks), dtype=np.float32)
    scalars = np.zeros(len(frames), dtype=np.float32)
    ok = np.zeros(len(frames), dtype=bool)
    fell_back = 0
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_init,
                             initargs=(cfg_path, sky_cal, tpl)) as ex:
        for i, (name, fit, rms, n) in enumerate(ex.map(_fit_frame, frames, chunksize=8)):
            if fit is not None:
                offsets[i], scalars[i] = fit
                ok[i] = True
                if n < 0:
                    fell_back += 1
            if (i + 1) % 2000 == 0:
                print(f"[gp3] {i+1}/{len(frames)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[gp3] done in {time.time()-t0:.0f}s; fitted {ok.sum()}/{len(frames)}; "
          f"{fell_back} frames fell back to all pixels (too few faint px)", flush=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("offsets/map_0", data=offsets, **hdf5plugin.Blosc())
        f.create_dataset("frame_scalar", data=scalars)
        f.create_dataset("chunk_maps/map_0", data=_G["cm"], **hdf5plugin.Blosc())
        f.create_dataset("reproj_list",
                         data=np.array([os.path.join(reproj_dir, os.path.basename(p)).encode()
                                        for p in frames]))
        f.create_dataset("fit_ok", data=ok)
        f.attrs["model"] = ("GLOBAL pass 3: per-frame min-norm LSQ of (data - stitched global "
                            "p2 sky) on deg-%d Chebyshev/column + DC scalar" % _G["deg"])
        f.attrs["sky_cal"] = sky_cal
        f.attrs["bright_cut"] = BRIGHT_CUT
    print(f"[gp3] saved {out_h5}", flush=True)
    return out_h5


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config")
    ap.add_argument("sky_cal", help="stitched pass-2 sky cal")
    ap.add_argument("reproj_dir", help="ALL frames of the field (not a staged tile)")
    ap.add_argument("out_h5")
    ap.add_argument("--max-workers", type=int, default=48)
    a = ap.parse_args()
    main(os.path.abspath(a.config), os.path.abspath(a.sky_cal), os.path.abspath(a.reproj_dir),
         os.path.abspath(a.out_h5), max_workers=a.max_workers)
