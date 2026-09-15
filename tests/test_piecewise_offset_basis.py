"""Piecewise (segmented) Chebyshev offset basis.

Motivation (SEP J=4, 2026-09-15): a degree-4 polynomial fit over 121 subchannels
captured 3-5x less of the per-frame ~15-subchannel structure in the aromatic band
than the same degree over 60, and raising the degree to 8 extrapolated to
+-200 MJy/sr in frames with partial red-end coverage. Independent low-degree
shapes per segment keep the narrow-window resolution without either failure.

Contract pinned here:
  * one segment == the window reproduces cheb_shape_basis bit-for-bit, so every
    existing (unsegmented) spec is untouched: n_coef, eval_offset_basis, and the
    per-frame refit see identical numbers;
  * with segments the basis is block-structured (zero outside each segment),
    mean-zero per segment, n_coef = degree * n_segments;
  * the per-frame refit recovers a known piecewise offset through the public
    refit_offsets_per_frame path (the same synthetic frames as
    test_npass_primitives).
"""
import numpy as np

from selfcal.models.offset_basis import (cheb_shape_basis, piecewise_cheb_shape_basis,
                                         eval_offset_basis, n_coef)


def _spec(degree, lo, hi, segments=None, ncol=3, nsub=342):
    ids = np.arange(nsub * ncol)
    pb = dict(degree=degree, num_groups=ncol, coord_lo=lo, coord_hi=hi,
              chunk_coord=ids // ncol, chunk_group=ids % ncol)
    if segments:
        pb["segments"] = [tuple(s) for s in segments]
    return pb


def test_single_segment_is_bit_identical_to_the_global_basis():
    coord = np.arange(0, 342, dtype=float)
    ref = cheb_shape_basis(coord, 4, 200, 320)
    pw = piecewise_cheb_shape_basis(coord, 4, [(200, 320)])
    inside = (coord >= 200) & (coord <= 320)
    assert pw.shape == ref.shape
    assert np.array_equal(pw[inside], ref[inside])
    assert np.all(pw[~inside] == 0.0)
    # and the spec-level entry points agree with the unsegmented spec
    pb0 = _spec(4, 200, 320)
    assert n_coef(pb0) == 4
    assert np.array_equal(eval_offset_basis(pb0["chunk_coord"], pb0),
                          cheb_shape_basis(pb0["chunk_coord"], 4, 200, 320))


def test_two_segments_block_structure_and_mean_zero():
    coord = np.arange(0, 342, dtype=float)
    segs = [(200, 259), (260, 320)]
    B = piecewise_cheb_shape_basis(coord, 4, segs)
    assert B.shape == (342, 8)
    s1 = (coord >= 200) & (coord <= 259)
    s2 = (coord >= 260) & (coord <= 320)
    assert np.all(B[s1, 4:] == 0.0) and np.all(B[s2, :4] == 0.0)
    assert np.all(B[~(s1 | s2)] == 0.0)
    # each segment's columns are the narrow-window basis on that segment
    assert np.array_equal(B[s1, :4], cheb_shape_basis(coord[s1], 4, 200, 259))
    assert np.array_equal(B[s2, 4:], cheb_shape_basis(coord[s2], 4, 260, 320))
    assert np.allclose(B[s1, :4].mean(0), 0, atol=1e-12)
    assert np.allclose(B[s2, 4:].mean(0), 0, atol=1e-12)
    pb = _spec(4, 200, 320, segs)
    assert n_coef(pb) == 8
    assert np.array_equal(eval_offset_basis(pb["chunk_coord"], pb),
                          piecewise_cheb_shape_basis(pb["chunk_coord"], 4, segs))


def test_segments_must_be_increasing_and_inside_window():
    coord = np.arange(0, 342, dtype=float)
    for bad in ([(260, 320), (200, 259)], [(200, 270), (260, 320)], []):
        try:
            piecewise_cheb_shape_basis(coord, 4, bad)
        except ValueError:
            continue
        raise AssertionError(f"segments {bad} should have been rejected")


def test_refit_recovers_a_piecewise_offset(tmp_path):
    """Through the public per-frame refit (same synthetic frames and chunk-map
    construction as test_npass_primitives' refit test): a frame whose true offset
    is an independent quadratic on each of two segments is recovered by the
    piecewise basis, and NOT by the single global quadratic of the same degree."""
    import sys
    import types
    if "pytest" not in sys.modules:                       # the env has no pytest; the helper module imports it
        try:
            import pytest  # noqa: F401
        except ModuleNotFoundError:
            mark = types.SimpleNamespace(parametrize=lambda *a, **k: (lambda f: f))
            sys.modules["pytest"] = types.SimpleNamespace(mark=mark)
    import h5py, hdf5plugin  # noqa: F401
    from tests.test_npass_primitives import _make_frames, DET, REF
    from selfcal.geometry.map_helper import chunk_to_det
    from selfcal.pipeline import npass

    rng = np.random.default_rng(11)
    J = 2
    n_sub, ncol, deg = 16, 2, 2                            # one "subchannel" per detector row
    segs = [(0, 7), (8, 15)]
    det_y, det_x = np.mgrid[0:DET[0], 0:DET[1]]
    cm = ((det_y * n_sub) // DET[0]) * ncol + (det_x * ncol) // DET[1]
    pw = _spec(deg, 0, n_sub - 1, segs, ncol, n_sub)
    gl = _spec(deg, 0, n_sub - 1, None, ncol, n_sub)
    Bpw = eval_offset_basis(pw["chunk_coord"].astype(float), pw)        # (chunks, 4)
    n_frames = 30
    a_true = 0.05 * rng.standard_normal((n_frames, ncol, Bpw.shape[1]))
    s_true = 0.1 * rng.standard_normal(n_frames)
    grp = pw["chunk_group"]
    per_chunk_true = np.array([(a_true[k][grp] * Bpw).sum(axis=1) + s_true[k] for k in range(n_frames)])
    offsets = [chunk_to_det(cm, chunk_data=per_chunk_true[k]) for k in range(n_frames)]
    paths, bc_det, truth, sky_model = _make_frames(str(tmp_path), rng, J, n_frames=n_frames,
                                                   offsets=offsets, bc_jitter=0.0)
    cal = str(tmp_path / "sky.h5")
    npass.write_sky_cal(cal, ref_shape=REF, sky_names=sky_model.names,
                        sky_maps=[t.astype(np.float32) for t in truth],
                        sky_counts=[np.ones(REF)] * J, sky_fishers=[np.ones(REF)] * J,
                        pixel_cross=np.zeros(REF[0] * REF[1]),
                        pixel_fisher=np.ones(J * REF[0] * REF[1]), reproj_list=[])
    sky = npass.SkySubtractor(cal, sky_model, export_dir=str(tmp_path / "exp"))
    errs = {}
    for label, pb in (("piecewise", pw), ("global", gl)):
        out = str(tmp_path / f"off_{label}.h5")
        _, mon = npass.refit_offsets_per_frame(
            paths, sky, det_chunk_map=cm, grid_valid=np.ones(DET, dtype=np.float32),
            det_aux=[bc_det, np.full(DET, 0.03)], poly_basis=pb, edges=None, ignore_list=[],
            thresh=100.0, bright_cut=None, min_pix=0, out_h5=out, max_workers=2)
        assert mon["n_fit"] == n_frames
        with h5py.File(out, "r") as f:
            off = f["offsets/map_0"][:]; sc = f["frame_scalar"][:]
            model = f.attrs["model"]
        errs[label] = float(np.max(np.abs((off + sc[:, None]) - per_chunk_true)))
        if label == "piecewise":
            assert "2 segments" in model and list(f"[{a}, {b}]" for a, b in segs)[0] in model
    assert errs["piecewise"] < 5e-3, errs
    assert errs["global"] > 5 * errs["piecewise"], errs   # one quadratic cannot follow two


if __name__ == "__main__":
    import pathlib, sys, tempfile
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                if "tmp_path" in fn.__code__.co_varnames:
                    with tempfile.TemporaryDirectory() as d:
                        fn(pathlib.Path(d))
                else:
                    fn()
                print(f"  PASS {name}")
            except Exception as exc:                      # noqa: BLE001
                failed += 1
                print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failed else 0)
