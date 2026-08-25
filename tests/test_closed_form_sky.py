"""Closed-form K=0 sky solve == converged LSQR (synthetic).

Builds a small 2-block (continuum + line) sky-only system with a spread of
per-pixel wavelength diversity, solves it with the per-pixel closed form and
with LSQR run to convergence, and checks they agree to float precision. Also
records the property that motivates the closed form: LSQR at a small
``iter_lim`` is far from converged on low-diversity pixels.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import h5py
import hdf5plugin  # noqa: F401

from selfcal.core.system import setup_lsqr
from selfcal.core.solve import apply_lsqr
from selfcal.core.solution import solve_sky_closed_form
from selfcal.models.sky_model import SkyModel

REF = (40, 40)
DET = (16, 16)
N_FRAMES = 80
LC, LS = 3.29, 0.03


def _make_frames(tmp_path, rng):
    cont_true = 1.0 + 0.3 * rng.standard_normal(REF)
    line_true = 0.2 * np.abs(rng.standard_normal(REF))
    det_y, det_x = np.mgrid[0:DET[0], 0:DET[1]].astype(np.float64)
    paths = []
    for k in range(N_FRAMES):
        y0 = int(rng.integers(0, REF[0] - DET[0]))
        x0 = int(rng.integers(0, REF[1] - DET[1]))
        # a smooth BC ramp whose centre jitters per frame: that spread is the
        # pixels' wavelength diversity; edge pixels see fewer centres
        bc = LC + 0.06 * (det_x / DET[1] - 0.5) + 0.02 * rng.standard_normal()
        G = np.exp(-0.5 * ((bc - LC) / LS) ** 2)
        obs = (cont_true[y0:y0 + DET[0], x0:x0 + DET[1]]
               + line_true[y0:y0 + DET[0], x0:x0 + DET[1]] * G
               + 2e-3 * rng.standard_normal(DET))
        p = os.path.join(tmp_path, f"sub_{k:04d}_det_1.h5")
        with h5py.File(p, "w") as f:
            f.create_dataset("sub_data", data=obs)
            f.create_dataset("sub_bitmask", data=np.zeros(DET, dtype=np.uint32))
            f.create_dataset("sub_mapping", data=np.stack([det_x, det_y]))
            f.create_dataset("sub_bc", data=bc)
            f.attrs["ref_coords"] = np.array([y0, y0 + DET[0], x0, x0 + DET[1]])
        paths.append(p)
    bc_det = LC + 0.06 * (det_x / DET[1] - 0.5)
    return paths, bc_det


def test_closed_form_matches_converged_lsqr(tmp_path):
    rng = np.random.default_rng(3)
    paths, bc_det = _make_frames(str(tmp_path), rng)
    sky_model = SkyModel.continuum_plus_pah_gaussian(LC, LS)
    r = setup_lsqr(paths, REF, sky_rhs_moments=True,
                   chunk_maps=[], sky_model=sky_model, det_aux=[bc_det],
                   max_workers=2, batch_size=20, outlier_thresh=100.0,
                   weighted_damping=True, damp_weight=1e-3, damp_weight_line=1e-3,
                   use_per_frame_scalar=False)
    num_sky = REF[0] * REF[1]
    x_cf = solve_sky_closed_form(r.pixel_fisher, r.pixel_cross, r.pixel_rhs, r.pixel_counts,
                                 num_sky, 2, damp_weights=[1e-3, 1e-3])
    am = r.active_mask

    def lsqr(iters):
        x = apply_lsqr(r.A, r.b, REF, damp=0.0, iter_lim=iters, n_threads=2,
                       active_mask=am, num_cols_full=int(am.size) if am is not None else None,
                       atol=1e-12, btol=1e-12)
        return x[:2 * num_sky]

    x_conv = lsqr(3000)
    x_few = lsqr(8)
    cov = np.r_[r.pixel_counts[:num_sky] > 0, r.pixel_counts[:num_sky] > 0]
    d_conv = np.abs(x_cf - x_conv)[cov]
    d_few = np.abs(x_cf - x_few)[cov]
    assert d_conv.max() < 1e-4, f"closed form != converged LSQR: max {d_conv.max():.2e}"
    # the motivating property: a short LSQR is NOT the answer
    assert d_few.max() > 1e-2
