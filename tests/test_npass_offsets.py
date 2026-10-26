"""The N-pass passes without the matrix, against the matrix of the same system.

The OFFSET pass (:mod:`selfcal.npass.offsets`): ``setup_lsqr(fixed_sky=True, reduce=normal_equations)``
reduces each frame's rows to its normal equations and :func:`solve_frames` solves them, on the OFFSET
pass's system of ``test_npass_primitives`` (the sky held fixed and subtracted, offsets per frame and a
per-frame scalar, the offset term of one chunk):

* the frames' normal equations and the constraint rows are the matrix's normal equations;
* the direct solution is the least-squares solution of the matrix (a polynomial-basis offset);
* where a frame's columns are not all determined (free chunk offsets beside the scalar), it is the
  limit of the column-scaled iterative solve from the engine's start, ``x0 + D (A D)^+ (b - A x0)``,
  computed here on the whole dense matrix;
* a hard constraint (``exclude``: no constant over a frame's chunks) holds exactly, and the solution is
  the least-squares solution under it;
* constraint rows or offsets shared by a few frames join their frames; a set over the cap is refused.

The SKY pass: ``setup_lsqr(moments_only=True)`` returns the matrix build's pixel state, byte for byte.
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from selfcal.core.solution import compute_x0_from_Ab  # noqa: E402
from selfcal.core.system import setup_lsqr  # noqa: E402
from selfcal.geometry.map_helper import chunk_to_det  # noqa: E402
from selfcal.models.offset_basis import cheb_shape_basis  # noqa: E402
from selfcal.npass import offsets as npass_offsets  # noqa: E402
from selfcal.npass.offsets import SkySubtractor, normal_equations, solve_frames  # noqa: E402
from tests.test_npass_primitives import DET, REF, _make_frames, _setup, sky_cal_of  # noqa: E402

N_SUB, N_COL = 4, 2


def chunk_map():
    det_y, det_x = np.mgrid[0:DET[0], 0:DET[1]]
    return ((det_y * N_SUB) // DET[0]) * N_COL + (det_x * N_COL) // DET[1]


def poly_basis():
    return {"degree": 1, "num_groups": N_COL, "coord_lo": 0, "coord_hi": N_SUB - 1,
            "chunk_coord": np.arange(N_SUB * N_COL) // N_COL,
            "chunk_group": np.arange(N_SUB * N_COL) % N_COL}


@pytest.fixture(scope="module")
def field(tmp_path_factory):
    """30 frames of a 2-term sky plus a per-frame polynomial offset and level, and the sky subtracter."""
    tmp = str(tmp_path_factory.mktemp("npass_offsets"))
    rng = np.random.default_rng(11)
    cm, pb = chunk_map(), poly_basis()
    B = cheb_shape_basis(pb["chunk_coord"].astype(float), 1, 0, N_SUB - 1)
    n_frames = 30
    a = 0.05 * rng.standard_normal((n_frames, N_COL))
    s = 0.1 * rng.standard_normal(n_frames)
    frame_offsets = [chunk_to_det(cm, chunk_data=a[k][pb["chunk_group"]] * B[:, 0] + s[k]) for k in range(n_frames)]
    paths, bc_det, truth, sky_model = _make_frames(tmp, rng, 2, n_frames=n_frames, offsets=frame_offsets,
                                                   bc_jitter=0.0)
    subtract = SkySubtractor(sky_cal_of(tmp, truth, sky_model), sky_model, export_dir=os.path.join(tmp, "exp"),
                             exclude_off_map=True)
    return dict(paths=paths, bc_det=bc_det, sky_model=sky_model, subtract=subtract)


def fixed_sky(field, **kw):
    """The OFFSET pass's system of ``field`` (``kw``: the offset model, and ``reduce``)."""
    return setup_lsqr(field["paths"], REF, sky_model=field["sky_model"],
                      det_aux=[field["bc_det"], np.full(DET, 0.03)], max_workers=2, batch_size=7,
                      apply_mask=True, apply_weight=True, outlier_thresh=100.0, postprocess_func=field["subtract"],
                      fixed_sky=True, **kw)


def dense(r):
    """A set-up system's matrix (dense), its right-hand side, its columns in the full layout and the
    full layout's size."""
    A = r.A.toarray().astype(np.float64)
    if r.active_mask is not None:
        return A, np.asarray(r.b, dtype=np.float64), np.flatnonzero(r.active_mask), r.active_mask.size
    return A, np.asarray(r.b, dtype=np.float64), np.arange(A.shape[1]), A.shape[1]


def normal_of(r, cols):
    """The full normal equations ``(N, h)`` of a reduced system over ``cols``."""
    loc = {int(c): i for i, c in enumerate(cols)}
    N, h = np.zeros((cols.size, cols.size)), np.zeros(cols.size)
    for f in r.reduced:
        if f is None:
            continue
        i = np.array([loc[int(c)] for c in f.cols])
        N[np.ix_(i, i)] += f.G
        h[i] += f.h
    C, d = npass_offsets._constraint_rows(r.constraint_blocks, r.pixel_counts.size)
    if C is not None:
        Cd = C[:, cols].toarray()
        N += Cd.T @ Cd
        h += Cd.T @ d
    return N, h


def one_chunk():
    """The one-chunk map of a per-frame scalar."""
    return np.zeros(DET, dtype=np.int64)


POLY = dict(chunk_maps=[chunk_map(), one_chunk()], poly_basis_list=[poly_basis(), None])
FREE = dict(chunk_maps=[chunk_map(), one_chunk()])


def exclusions_map0(n_groups, base):
    nc = N_SUB * N_COL
    cols = base + np.arange(n_groups)[:, None] * nc + np.arange(nc)[None, :]
    return [(cols, np.full((nc, 1), 1.0 / np.sqrt(nc)))]


def constrained_lstsq(A, b, cols, n_full, exclusions):
    """The least-squares solution of ``A`` (columns ``cols`` of the full layout) under the hard constraints
    (a group's present columns only: an absent column has no unknown), by a basis of their null space."""
    loc = {int(c): i for i, c in enumerate(cols)}
    rows = []
    for gcols, P in exclusions:
        for g in range(gcols.shape[0]):
            for j in range(P.shape[1]):
                e = np.zeros(cols.size)
                for c, v in zip(gcols[g], P[:, j]):
                    if int(c) in loc:
                        e[loc[int(c)]] = v
                if np.any(e):
                    rows.append(e)
    E = np.array(rows)
    _, sv, Vt = np.linalg.svd(E)
    Nb = Vt[(sv > 1e-12).sum():].T                                         # the null space of E
    z = np.linalg.lstsq(A @ Nb, b, rcond=None)[0]
    return Nb @ z, E


@pytest.mark.parametrize("model", ["poly", "free"])
def test_reduced_frames_are_the_normal_equations_of_the_matrix(field, model):
    kw = dict(POLY) if model == "poly" else dict(FREE)
    r = fixed_sky(field, **kw)
    rr = fixed_sky(field, reduce=normal_equations, **kw)
    assert rr.A is None and len(rr.reduced) == len(field["paths"])
    assert np.array_equal(rr.pixel_counts, r.pixel_counts)                 # the coverage the cal records
    A, b, cols, _ = dense(r)
    N, h = normal_of(rr, cols)
    assert np.max(np.abs(N - A.T @ A)) < 1e-5 * np.abs(A.T @ A).max()
    assert np.max(np.abs(h - A.T @ b)) < 1e-5 * np.abs(A.T @ b).max()


def test_direct_solution_is_the_least_squares_solution(field):
    r = fixed_sky(field, **POLY)
    rr = fixed_sky(field, reduce=normal_equations, **POLY)
    A, b, cols, n_full = dense(r)
    x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
    x, summary = solve_frames(rr.reduced, rr.constraint_blocks, n_full, start="separate")
    assert x.shape == (n_full,)
    assert np.max(np.abs(x[cols] - x_ref)) < 1e-6 * np.abs(x_ref).max()
    assert summary["method"] == "direct" and summary["blocks"] == len(field["paths"]) and summary["left_out"] == 0
    assert summary["true_residual"] == pytest.approx(np.linalg.norm(b - A @ x_ref), rel=1e-6)
    assert summary["bnorm"] == pytest.approx(np.linalg.norm(b), rel=1e-9)
    assert summary["columns"] == cols.size and summary["rows"] == A.shape[0]


def test_undetermined_directions_keep_the_start(field):
    """Free chunk offsets beside the per-frame scalar (map 1, one chunk): the scalar and the chunks'
    common level are one direction, so every frame's block is singular. The direct solution is the
    limit the column-scaled iterative solve reaches from the engine's start (every column at its own
    fit), ``x0 + D (A D)^+ (b - A x0)``, which keeps the start along that direction. That start has no
    part along it, in the column-scaled unknowns (the scalar's column is the sum of the chunks', so its
    own fit is the sum of theirs): the split is the one a start at zero gives."""
    r = fixed_sky(field, **FREE)
    rr = fixed_sky(field, reduce=normal_equations, **FREE)
    A, b, cols, n_full = dense(r)
    lo = n_full - len(field["paths"])                                      # the scalar term's columns come last
    x, summary = solve_frames(rr.reduced, rr.constraint_blocks, n_full, start="separate")
    assert summary["left_out"] == len(field["paths"])                     # one direction per frame
    x0 = compute_x0_from_Ab(r.A, r.b, REF, num_sky_blocks=2, active_mask=r.active_mask)[cols]
    D = 1.0 / np.linalg.norm(A, axis=0)
    x_lim = x0 + D * (np.linalg.pinv(A * D, rcond=1e-10) @ (b - A @ x0))
    assert np.max(np.abs(x[cols] - x_lim)) < 1e-6 * np.abs(x_lim).max()
    x_zero, _ = solve_frames(rr.reduced, rr.constraint_blocks, n_full)    # from zero: the same split
    assert np.max(np.abs(x_zero[lo:] - x[lo:])) < 1e-9 * np.abs(x[lo:]).max()


def test_hard_constraint_holds_and_is_the_constrained_least_squares(field):
    """``exclude=Poly(0)`` on the chunk offsets beside the scalar: each frame's chunk offsets average to
    zero (over its chunks), exactly; the duplicate direction is gone (no direction left out), and the
    solution is the least-squares solution under the constraint, from any start."""
    r = fixed_sky(field, **FREE)
    rr = fixed_sky(field, reduce=normal_equations, **FREE)
    A, b, cols, n_full = dense(r)
    n = len(field["paths"])
    ex = exclusions_map0(n, int(2 * REF[0] * REF[1]))
    x_ref, E = constrained_lstsq(A, b, cols, n_full, ex)
    for start in ("separate", None):
        x, summary = solve_frames(rr.reduced, rr.constraint_blocks, n_full, start=start, exclusions=ex)
        assert summary["left_out"] == 0
        assert np.max(np.abs(E @ x[cols])) < 1e-9 * np.abs(x[cols]).max()          # the constraint, exactly
        assert np.max(np.abs(x[cols] - x_ref)) < 1e-6 * np.abs(x_ref).max()


def test_constraint_rows_join_frames(field, monkeypatch):
    """A prior tying frame 0's scalar to frame 1's joins their frames; the solution is still the
    least-squares solution of the matrix. A set of columns over the cap is refused."""
    def tie(info):
        s0 = int(info.layout.col_bases[1])                                 # the scalar term: frame 0, frame 1
        return (np.array([0, 0]), np.array([s0, s0 + 1]), np.array([3.0, -3.0]), np.array([0.0]))
    kw = dict(POLY, priors=[tie])
    r = fixed_sky(field, **kw)
    rr = fixed_sky(field, reduce=normal_equations, **kw)
    A, b, cols, n_full = dense(r)
    x_ref = np.linalg.lstsq(A, b, rcond=None)[0]
    x, summary = solve_frames(rr.reduced, rr.constraint_blocks, n_full)
    assert summary["joined"] == 1 and summary["blocks"] == len(field["paths"]) - 1
    assert np.max(np.abs(x[cols] - x_ref)) < 1e-6 * np.abs(x_ref).max()
    monkeypatch.setattr(npass_offsets, "MAX_BLOCK_COLUMNS", 4)
    with pytest.raises(ValueError, match="do not separate"):
        solve_frames(rr.reduced, rr.constraint_blocks, n_full)


def test_offsets_shared_by_a_few_frames_join_them(field):
    """Offsets shared by groups of frames (``det_groups``), with no constant over a group's chunks: the
    frames of a group are joined and solved together, exactly the constrained least-squares solution."""
    n = len(field["paths"])
    groups = np.arange(n) % 10                                             # 10 groups of 3 frames
    kw = dict(FREE, det_groups_list=[groups, None])
    r = fixed_sky(field, **kw)
    rr = fixed_sky(field, reduce=normal_equations, **kw)
    A, b, cols, n_full = dense(r)
    ex = exclusions_map0(10, int(2 * REF[0] * REF[1]))
    x_ref, _ = constrained_lstsq(A, b, cols, n_full, ex)
    x, summary = solve_frames(rr.reduced, rr.constraint_blocks, n_full, exclusions=ex)
    assert summary["blocks"] == 10 and summary["joined"] == 10
    assert np.max(np.abs(x[cols] - x_ref)) < 1e-6 * np.abs(x_ref).max()


@pytest.mark.parametrize("J", [2, 4])
def test_moments_only_is_the_matrix_builds_pixel_state(tmp_path, J):
    rng = np.random.default_rng(3 + J)
    paths, bc_det, _, sky_model = _make_frames(str(tmp_path), rng, J, n_frames=40)
    full = _setup(paths, sky_model, bc_det)
    only = setup_lsqr(paths, REF, sky_rhs_moments=True, chunk_maps=[], sky_model=sky_model,
                      det_aux=[bc_det, np.full(DET, 0.03)], max_workers=2, batch_size=20, apply_mask=True,
                      apply_weight=True, outlier_thresh=100.0, weighted_damping=True, damp_weight=1e-3,
                      damp_weight_line=1e-3, moments_only=True)
    assert only.A is None and full.A is not None
    for name in ("pixel_counts", "pixel_fisher", "pixel_rhs"):
        a, b = getattr(full, name), getattr(only, name)
        assert a.dtype == b.dtype and a.tobytes() == b.tobytes(), name
    if J == 2:
        assert full.pixel_cross.tobytes() == only.pixel_cross.tobytes()
    else:
        assert set(full.pixel_cross) == set(only.pixel_cross)
        for k in full.pixel_cross:
            assert full.pixel_cross[k].tobytes() == only.pixel_cross[k].tobytes(), k
