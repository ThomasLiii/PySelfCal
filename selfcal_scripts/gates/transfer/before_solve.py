"""The weighted solve as the workspace ran it before the library had pixel weights: the transfer gate's "before".

A verbatim copy of the production path of ``workspace/lsqr-convergence/scripts/scene_d3/run_scene.py`` (2026-10-13):
the subchannel-constants + per-frame x-slope model (``xbasis.xtilde``), no clip, deflated CGLS to the chi-squared stop,
and the pixel weights ``1/sqrt(max(|value|, m))`` -- ``m`` the median of the frame's values inside the job's window --
applied by wrapping the assembly's subframe preparation. Kept so that the golden it made can be remade from a
committed tree; the library's run of the same recipe (``python_gates.py transfer``) must reproduce it byte for byte.

Precision: the workspace patch computed the weights in float32 (``d`` is float32 and ``m`` a Python float, which NumPy
keeps at float32), while its coadd counterpart (``before_mosaic.py``) computed them in float64. The library computes a
noise variance in float64 everywhere, so this golden maker evaluates the same expression in float64; the production
arithmetic is kept behind ``TRANSFER_FLOAT32_PATCH=1`` (it changes every weight by up to 9e-8 relative, which moves
the chi-squared stop by 20 iterations on the fixture and the cal by the solve's convergence tolerance, ~2 % of the
sky's rms: see the gates README).
    DET=d SCENE_FIELD=<field dir> SCENE_SCRATCH=<scratch dir> python before_solve.py FRAMES_DIR NAME"""
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

# --- the workspace patch, verbatim (run_scene.py, SNW_CEIL=channel_median) -------------------------------------------
# pixel weights from the data the solve sees: each frame's rows are weighted 1/sqrt(max(|value|, m)), m = the
# median of that frame's pixels inside the job's window (the padded subchannel set; unflagged), so no pixel
# weighs more than the channel's median pixel and brighter ones are down-weighted as 1/value. Wraps the
# assembly's subframe preparation (its weight already holds the window and the flag mask) instead of the
# library's whole-subframe weight; module level so the assembly workers, which re-import this script, use it.
import numpy as _np  # noqa: E402
import selfcal.core.assembly as _assembly  # noqa: E402
_prep_library = _assembly._prep_subframe
_logged = []
PRODUCTION_FLOAT32 = os.environ.get('TRANSFER_FLOAT32_PATCH') == '1'


def _prep_channel_median_weight(*args, **kwargs):
    want = bool(kwargs.get('apply_weight')) and bool(kwargs.get('for_lsqr'))
    if want:
        kwargs['apply_weight'] = False
    ref, d, w, cc, aux = _prep_library(*args, **kwargs)
    if want:
        sel = w > 0
        if sel.any():
            if PRODUCTION_FLOAT32:          # the patch as the production solves ran it (NumPy keeps d's float32)
                m = float(_np.median(d[sel]))
                ww = _np.zeros_like(w)
                ww[sel] = 1.0 / _np.sqrt(_np.maximum(_np.abs(d[sel]), m))
            else:                           # precision-aligned: the variance in float64, as the library computes it
                v = d[sel].astype(_np.float64)
                m = float(_np.median(v))
                ww = _np.zeros(w.shape, dtype=_np.float64)
                ww[sel] = 1.0 / _np.sqrt(_np.maximum(_np.abs(v), m))
            w = (w * ww).astype(w.dtype)
            if not _logged:
                _logged.append(1)
                print(f'[weights] pid {os.getpid()}: first frame: {int(sel.sum())} window pixels, median {m:.4f} MJy/sr, '
                      f'weight ceiling {1 / _np.sqrt(m):.3f}, min weight {ww[sel].min():.3f}', flush=True)
    return ref, d, w, cc, aux


_assembly._prep_subframe = _prep_channel_median_weight
# ---------------------------------------------------------------------------------------------------------------------


def main():
    frames_dir, name = sys.argv[1], sys.argv[2]
    print('selfcal from', sc.__file__, flush=True)
    compute = sc.Compute(os.path.join(SCRATCH, name), workers=30, stage=None)
    inst = sc.SPHEREx(int(os.environ.get('DET', '3')), num_col=1, num_ch=17, calib_dir='/mnt/md124/SPHEREx/SpecCal_202509/ParameterFiles')
    terms = [sc.Offsets(on='subchannel', mean_zero=True),
             sc.Offsets('xs_term', on='subchannel', per='frame', times=sc.Function(xtilde, of='det_x'))]
    model = sc.Model(sky=(sc.Sky(damping=0.0),), offsets=tuple(terms), variables={}, priors=())
    field = sc.Field(FIELD, inst, 6.19999999999992, compute=compute)
    fit = sc.Fit(600, clip=None, ignore_flags=(), tolerance=0.0, deflate=sc.Deflate(4), shot_noise_weights=True,
                 stop=sc.Stop(chi2=sc.ChiSquaredRule(0.1, window=50, every=10)))
    recipe = sc.Recipe(model, fit=fit, coadd=None, numerics=sc.Numerics(30, batch=40), name=name)
    frames = sc.frames_in(frames_dir)
    print(f'{len(frames)} frames from {frames_dir}', flush=True)
    t0 = time.time()
    res = field.calibrate(recipe, jobs=spherex.channel(9), frames=frames, start=None,
                          monitor=sc.Monitor(every=10, residual=True, gradient=False, large_scale=2),
                          snapshots=sc.Snapshots(every=100))
    print(res, flush=True)
    print(f"[before_solve {name}] done: wall {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == '__main__':
    main()
