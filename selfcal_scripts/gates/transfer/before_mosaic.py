"""The weighted coadd as the workspace ran it before the library had pixel weights: the transfer gate's "before".

A verbatim copy of the production path of ``workspace/lsqr-convergence/scripts/scene_d3/mosaic_from_cal.py``
(2026-10-13, ``MOSAIC_CLIP=2 MOSAIC_WEIGHTS=channel_median``): the coadd of a cal's frames (data minus the cal's
offsets) with the mean, std and 2-sigma clipped mean maps, flag 21 ignored, every frame pixel weighted
``1/max(|raw value|, m)`` -- the weighted solve's least-squares weight -- with ``m`` the median of the frame's raw
values over the pixels the solve used (inside the job window, no flag set), applied by wrapping the coadd's subframe
preparation.
    DET=d SCENE_FIELD=<field dir> SCENE_SCRATCH=<scratch dir> [MOSAIC_WORKERS=32] python before_mosaic.py FRAMES_DIR CAL.h5 NAME"""
import os
import sys
import time

SCR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCR)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("NUMPY_MADVISE_HUGEPAGE", "0")

import selfcal as sc  # noqa: E402
from selfcal.instruments import spherex  # noqa: E402
from xbasis import xtilde  # noqa: E402

FIELD = os.environ['SCENE_FIELD']
SCRATCH = os.environ['SCENE_SCRATCH']

# --- the workspace patch, verbatim (mosaic_from_cal.py, MOSAIC_WEIGHTS=channel_median) -------------------------------
# coadd weights matching the weighted solve (run_scene.py SNW_CEIL=channel_median): each frame pixel weighs
# 1/max(|raw value|, m) -- the solve's least-squares weight w^2 -- with m the median of the frame's raw values over
# the pixels the solve used (inside the job window, no flag set). The library's coadd weight would be computed on
# the offset-subtracted data, so the raw values are read from the frame here. Module level: the coadd workers
# re-import this script.
import h5py as _h5py  # noqa: E402
import hdf5plugin as _hdf5plugin  # noqa: E402,F401
import numpy as _np2  # noqa: E402
import selfcal.core.coadd as _coadd  # noqa: E402
_prep_coadd_library = _coadd._prep_subframe


def _prep_coadd_weighted(*args, **kwargs):
    ref, d, w, cc, aux = _prep_coadd_library(*args, **kwargs)
    path = kwargs.get('file', args[0] if args else None)
    with _h5py.File(path, 'r') as f:
        raw = f['sub_data'][:].astype(_np2.float64); bm = f['sub_bitmask'][:]
    solve_px = (w > 0) & (bm == 0) & _np2.isfinite(raw)
    if solve_px.any():
        m = float(_np2.median(raw[solve_px]))
        with _np2.errstate(all='ignore'):
            f_w = 1.0 / _np2.maximum(_np2.abs(raw), m)
        w = (w * _np2.nan_to_num(f_w, nan=0.0)).astype(w.dtype)
    return ref, d, w, cc, aux


_coadd._prep_subframe = _prep_coadd_weighted
# ---------------------------------------------------------------------------------------------------------------------


def main():
    frames_dir, cal, name = sys.argv[1], sys.argv[2], sys.argv[3]
    nw = int(os.environ.get('MOSAIC_WORKERS', '32'))     # the maps do not depend on it
    compute = sc.Compute(os.path.join(SCRATCH, name), workers=nw, stage=None)
    inst = sc.SPHEREx(int(os.environ.get('DET', '3')), num_col=1, num_ch=17, calib_dir='/mnt/md124/SPHEREx/SpecCal_202509/ParameterFiles')
    model = sc.Model(sky=(sc.Sky(damping=0.0),),
                     offsets=(sc.Offsets(on='subchannel', mean_zero=True),
                              sc.Offsets('xs_term', on='subchannel', per='frame', times=sc.Function(xtilde, of='det_x'))))
    field = sc.Field(FIELD, inst, 6.19999999999992, compute=compute)
    recipe = sc.Recipe(model, fit=sc.Fit(600, clip=sc.Clip(2.0), ignore_flags=(), tolerance=0.0),
                       coadd=sc.Coadd(clip=2.0, std=True, ignore_flags=(21,), instrument_maps=False),
                       numerics=sc.Numerics(nw), name=name)
    t0 = time.time()
    res = field.mosaic(recipe, jobs=spherex.channel(9), cal=cal, frames=sc.frames_in(frames_dir))
    print(res, flush=True)
    print(f"[before_mosaic {name}] done: wall {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
