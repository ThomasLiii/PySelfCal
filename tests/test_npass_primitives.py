"""Primitives of the N-pass alternating solve (selfcal.pipeline.npass), on a
small synthetic sky-only system:

* moment ADDITIVITY: dumps over two disjoint frame subsets, combined, equal the
  single full solve exactly (J = 2 and J = 4 — the seam-free full-field claim);
* SkySubtractor.window handles a subframe overhanging the map's low edge
  (negative ref_coords), the bug that produced the LMC streaks;
* refit_offsets_per_frame recovers known per-frame poly-basis offsets.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import h5py
import hdf5plugin  # noqa: F401
import pytest

from selfcal.core.system import setup_lsqr
from selfcal.core.solution import solve_sky_closed_form
from selfcal.models.sky_model import SkyModel, ContinuumComponent, SpectralComponent
from selfcal.models.profiles import GaussianProfile
from selfcal.pipeline import npass

REF = (40, 40)
DET = (16, 16)
LC = 3.29


def _sky_model(J):
    comps = [ContinuumComponent()]
    for j in range(1, J):
        comps.append(SpectralComponent(name=f"line{j}",
                                       profile=GaussianProfile(center_um=LC + 0.03 * (j - 1),
                                                               sigma_um=0.03),
                                       wavelength_key="BC"))
    return SkyModel(tuple(comps))


def _make_frames(tmp, rng, J, n_frames=80, offsets=None, bc_jitter=0.02):
    """Synthetic frames: sky (J blocks) + optional per-frame offset on the
    detector grid. ``bc_jitter`` shifts each frame's BC (wavelength diversity for
    the sky solve tests; 0 when the pipeline's detector-map aux must be exact).
    Returns (paths, bc_det, truth, sky_model)."""
    sky_model = _sky_model(J)
    truth = [1.0 + 0.3 * rng.standard_normal(REF)] + \
            [0.2 * np.abs(rng.standard_normal(REF)) for _ in range(J - 1)]
    det_y, det_x = np.mgrid[0:DET[0], 0:DET[1]].astype(np.float64)
    bc_det = LC + 0.06 * (det_x / DET[1] - 0.5)
    paths = []
    for k in range(n_frames):
        y0 = int(rng.integers(-3, REF[0] - DET[0] + 3))   # some frames overhang the edges
        x0 = int(rng.integers(-3, REF[1] - DET[1] + 3))
        bc = bc_det + bc_jitter * rng.standard_normal()
        aux = {"BC": bc, "BW": np.full(DET, 0.03)}
        obs = np.zeros(DET)
        ys, ye = max(0, -y0), min(DET[0], REF[0] - y0)
        xs, xe = max(0, -x0), min(DET[1], REF[1] - x0)
        on = np.zeros(DET, bool); on[ys:ye, xs:xe] = True
        for j, comp in enumerate(sky_model.components):
            c = np.ones(DET) if j == 0 else np.asarray(comp.coefficients(aux)).reshape(DET)
            sub = np.zeros(DET)
            sub[ys:ye, xs:xe] = truth[j][y0 + ys:y0 + ye, x0 + xs:x0 + xe]
            obs += sub * c
        if offsets is not None:
            obs += offsets[k]
        obs += 2e-3 * rng.standard_normal(DET)
        bitmask = np.where(on, 0, 1).astype(np.uint32)    # bit 0 = off-map
        p = os.path.join(tmp, f"sub_{k:04d}_det_1.h5")
        with h5py.File(p, "w") as f:
            f.create_dataset("sub_data", data=obs)
            f.create_dataset("sub_bitmask", data=bitmask)
            f.create_dataset("sub_mapping", data=np.stack([det_x, det_y]))
            f.create_dataset("sub_bc", data=bc)
            f.attrs["ref_coords"] = np.array([y0, y0 + DET[0], x0, x0 + DET[1]])
        paths.append(p)
    return paths, bc_det, truth, sky_model


def _setup(paths, sky_model, bc_det, postprocess=None):
    return setup_lsqr(paths, REF, sky_rhs_moments=True, chunk_maps=[], sky_model=sky_model,
                      det_aux=[bc_det, np.full(DET, 0.03)], max_workers=2, batch_size=20,
                      apply_mask=True, apply_weight=True,
                      outlier_thresh=100.0, weighted_damping=True, damp_weight=1e-3,
                      damp_weight_line=1e-3, use_per_frame_scalar=False,
                      postprocess_func=postprocess)


class _CC:
    """Minimal Calibrator stand-in for dump_moments."""
    def __init__(self, r, J, frames):
        self.pixel_counts, self.pixel_fisher, self.pixel_cross = r.pixel_counts, r.pixel_fisher, r.pixel_cross
        self.pixel_rhs, self.num_sky_blocks, self.ref_shape, self.reproj_list = r.pixel_rhs, J, REF, frames

    def _materialize_pixel_state(self):
        pass


@pytest.mark.parametrize("J", [2, 4])
def test_combine_moments_equals_full_solve(tmp_path, J):
    rng = np.random.default_rng(5 + J)
    paths, bc_det, truth, sky_model = _make_frames(str(tmp_path), rng, J)
    num_sky = REF[0] * REF[1]
    dws = [1e-3] * J
    r_full = _setup(paths, sky_model, bc_det)
    x_full = solve_sky_closed_form(r_full.pixel_fisher, r_full.pixel_cross, r_full.pixel_rhs,
                                   r_full.pixel_counts, num_sky, J, damp_weights=dws)
    dumps = []
    for k, sub in enumerate((paths[:37], paths[37:])):
        r = _setup(sub, sky_model, bc_det)
        d = str(tmp_path / f"m{k}.npz")
        npass.dump_moments(_CC(r, J, sub), d)
        dumps.append(d)
    out = str(tmp_path / "combined.h5")
    npass.combine_moments(dumps, out, sky_names=sky_model.names, damp_weights=dws,
                          line_fisher_threshold=0.0)
    with h5py.File(out, "r") as f:
        names = [n.decode() for n in f.attrs["sky_components"]]
        assert names == sky_model.names
        for j, n in enumerate(names):
            got = f["sky"][n][:].astype(np.float64).ravel()
            ref = x_full[j * num_sky:(j + 1) * num_sky]
            cov = r_full.pixel_counts[j * num_sky:(j + 1) * num_sky] > 0
            assert np.max(np.abs(got - ref)[cov]) < 1e-5, n
        assert f["skymap"].shape == REF and f["skymap_line"].shape == REF
        assert set(f["sky_separability"].keys()) == set(names[1:])
    # frames are double-counted -> refused
    with pytest.raises(ValueError):
        npass.combine_moments([dumps[0], dumps[0]], str(tmp_path / "dup.h5"),
                              sky_names=sky_model.names, damp_weights=dws)


def test_sky_subtractor_window_handles_low_edge_overhang(tmp_path):
    rng = np.random.default_rng(1)
    J = 2
    paths, bc_det, truth, sky_model = _make_frames(str(tmp_path), rng, J, n_frames=4)
    cal = str(tmp_path / "sky.h5")
    npass.write_sky_cal(cal, ref_shape=REF, sky_names=sky_model.names,
                        sky_maps=[t.astype(np.float32) for t in truth],
                        sky_counts=[np.ones(REF)] * J, sky_fishers=[np.ones(REF)] * J,
                        pixel_cross=np.zeros(REF[0] * REF[1]),
                        pixel_fisher=np.ones(J * REF[0] * REF[1]), reproj_list=[])
    sub = npass.SkySubtractor(cal, sky_model, export_dir=str(tmp_path / "exp"))
    rc = np.array([-5, DET[0] - 5, -2, DET[1] - 2])          # overhangs top-left
    maps, on = sub.window(rc, DET)
    assert not on[:5].any() and not on[:, :2].any() and on[5:, 2:].all()
    assert np.allclose(maps[0][5:, 2:], truth[0][:DET[0] - 5, :DET[1] - 2])
    assert np.all(maps[0][:5] == 0)
    rc = np.array([REF[0] - 6, REF[0] - 6 + DET[0], 3, 3 + DET[1]])   # overhangs bottom
    maps, on = sub.window(rc, DET)
    assert on[:6].all() and not on[6:].any()
    assert np.allclose(maps[1][:6], truth[1][REF[0] - 6:, 3:3 + DET[1]])


def test_refit_offsets_recovers_known_offsets(tmp_path):
    rng = np.random.default_rng(11)
    J = 2
    # chunk map on the detector: 4 "subchannels" (row bands) x 2 columns
    n_sub, ncol = 4, 2
    det_y, det_x = np.mgrid[0:DET[0], 0:DET[1]]
    cm = ((det_y * n_sub) // DET[0]) * ncol + (det_x * ncol) // DET[1]
    pb = {"degree": 1, "num_groups": ncol, "coord_lo": 0, "coord_hi": n_sub - 1,
          "chunk_coord": np.arange(n_sub * ncol) // ncol,
          "chunk_group": np.arange(n_sub * ncol) % ncol}
    from selfcal.models.offset_basis import cheb_shape_basis
    B = cheb_shape_basis(pb["chunk_coord"].astype(float), 1, 0, n_sub - 1)   # (chunks, 1)
    n_frames = 30
    a_true = 0.05 * rng.standard_normal((n_frames, ncol))
    s_true = 0.1 * rng.standard_normal(n_frames)
    from selfcal.geometry.map_helper import chunk_to_det
    offsets = []
    for k in range(n_frames):
        per_chunk = a_true[k][pb["chunk_group"]] * B[:, 0] + s_true[k]
        offsets.append(chunk_to_det(cm, chunk_data=per_chunk))
    paths, bc_det, truth, sky_model = _make_frames(str(tmp_path), rng, J, n_frames=n_frames,
                                                   offsets=offsets, bc_jitter=0.0)
    cal = str(tmp_path / "sky.h5")
    npass.write_sky_cal(cal, ref_shape=REF, sky_names=sky_model.names,
                        sky_maps=[t.astype(np.float32) for t in truth],
                        sky_counts=[np.ones(REF)] * J, sky_fishers=[np.ones(REF)] * J,
                        pixel_cross=np.zeros(REF[0] * REF[1]),
                        pixel_fisher=np.ones(J * REF[0] * REF[1]), reproj_list=[])
    sky = npass.SkySubtractor(cal, sky_model, export_dir=str(tmp_path / "exp"))
    out = str(tmp_path / "off.h5")
    _, mon = npass.refit_offsets_per_frame(
        paths, sky, det_chunk_map=cm, grid_valid=np.ones(DET, dtype=np.float32),
        det_aux=[bc_det, np.full(DET, 0.03)], poly_basis=pb, edges=None, ignore_list=[],
        thresh=100.0, bright_cut=None, min_pix=0, out_h5=out, max_workers=2)
    assert mon["n_fit"] == n_frames
    with h5py.File(out, "r") as f:
        off = f["offsets/map_0"][:]; sc = f["frame_scalar"][:]
    per_chunk_true = np.array([a_true[k][pb["chunk_group"]] * B[:, 0] + s_true[k]
                               for k in range(n_frames)])
    assert np.max(np.abs((off + sc[:, None]) - per_chunk_true)) < 5e-3
