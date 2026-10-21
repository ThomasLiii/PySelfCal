"""Deflation of the smooth sky patterns (selfcal.core.deflation; ``sc.Fit(deflate=sc.Deflate(...))``).

A deflated solve takes each sky term's smooth patterns together with the offsets that absorb them best,
solves the least squares along them directly, and iterates on the rest. It must reach THE least-squares
solution (the same one a plain solve reaches, its gauge included when the gauge lies in the space; the
same fit to the data otherwise), reach it in fewer iterations when the large scales are what is slow,
work on compacted and uncompacted systems, in float32 and float64, with LSQR and LSMR, from a warm
start, with monitors and snapshots that see the solution, and stay put past convergence. Through the
API: the setting is part of the fingerprint only when given, the plan says what it does, and the cal,
the action's record and the log say what it did.
"""
import json
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import selfcal as sc  # noqa: E402
from selfcal import _state  # noqa: E402
from selfcal.config import ConfigError  # noqa: E402
from selfcal.core.deflation import Deflation, axis_functions  # noqa: E402
from selfcal.core.monitor import SmoothFit  # noqa: E402
from selfcal.core.snapshots import ActiveColumns, Iterate  # noqa: E402
from selfcal.core.solve import apply_lsqr  # noqa: E402
from selfcal.io.calfile import CalFile  # noqa: E402
from selfcal.run import products  # noqa: E402
from tests.synthetic_exposures import DET, N_CHUNK_SIDE, REF_ARCSEC, write_exposures  # noqa: E402

pytestmark = pytest.mark.filterwarnings('ignore:overflow encountered in cast:RuntimeWarning')


# =================================================================== systems
def _chunk_system(h=24, w=24, frames=60, chunks=4, scalar=True, seed=1):
    """A sky of h x w pixels seen by frames of 8 rows x 16 columns at random places; each frame has a
    constant and a slope in x per 4-column chunk, a mean-zero anchor on its constants and (``scalar``) a
    per-frame scalar. Exact gauges: the sky's mean against the scalars, and the sky's x-plane (absorbed
    exactly by the per-chunk slopes). Returns (A full layout csr, b, the number of sky columns)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    sky = (0.7 * (yy / h) + 0.4 * (xx / w) ** 2 + 0.05 * rng.normal(size=(h, w))).ravel()
    npix = h * w
    off0 = npix
    sc0 = off0 + frames * chunks * 2
    rows, cols, vals, b = [], [], [], []
    r = 0
    for f in range(frames):
        y0, x0 = rng.integers(0, h - 8), rng.integers(0, w - 16)
        for yy_ in range(8):
            for xx_ in range(16):
                p = (y0 + yy_) * w + (x0 + xx_)
                c = xx_ * chunks // 16                 # the frame's 16 columns in `chunks` chunks (4-px chunks by default)
                ci = off0 + (f * chunks + c) * 2
                rows += [r, r, r]
                cols += [p, ci, ci + 1]
                vals += [1.0, 1.0, (xx_ % 4 - 1.5) / 1.5]
                if scalar:
                    rows.append(r)
                    cols.append(sc0 + f)
                    vals.append(1.0)
                b.append(sky[p] + 0.2 * np.sin(f) + 0.01 * rng.normal())
                r += 1
        for c in range(chunks):
            rows.append(r)
            cols.append(off0 + (f * chunks + c) * 2)
            vals.append(1.0)
        b.append(0.0)
        r += 1
    n = sc0 + (frames if scalar else 0)
    return sp.csr_matrix((vals, (rows, cols)), shape=(r, n)), np.asarray(b), npix, (h, w)


def _near_null_system(H=16, W=24, weak=0.005, noise=0.05, seed=0):
    """Frames that each cover most of ONE pixel column, with an offset per frame: a sky varying along x is
    absorbed by the offsets and fixed only by weak cross-frames (one per row): near-null smooth patterns
    along x (as in test_monitor). Returns (A full layout csr, b, the number of sky columns, the grid)."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    xt, yt = (xx - (W - 1) / 2) / ((W - 1) / 2), (yy - (H - 1) / 2) / ((H - 1) / 2)
    sky = 0.8 * xt + 0.5 * yt + 0.3 * xt ** 2 + 0.05 * rng.normal(size=(H, W))
    frames = []
    for x in range(W):
        for _ in range(3):
            frames.append(([(y, x) for y in np.sort(rng.choice(H, size=H * 3 // 4, replace=False))], 1.0))
    for y in range(H):
        frames.append(([(y, x) for x in range(W)], weak))
    nf = len(frames)
    offsets = rng.normal(size=nf)
    rows, cols, vals, b = [], [], [], []
    r = 0
    for f, (pix, w) in enumerate(frames):
        for (y, x) in pix:
            rows += [r, r]
            cols += [y * W + x, H * W + f]
            vals += [w, w]
            b.append(w * (sky[y, x] + offsets[f] + noise * rng.normal()))
            r += 1
    for f in range(nf):
        rows.append(r)
        cols.append(H * W + f)
        vals.append(1.0 / np.sqrt(nf))
    b.append(0.0)
    r += 1
    return sp.csr_matrix((vals, (rows, cols)), shape=(r, H * W + nf)), np.asarray(b), H * W, (H, W)


def _shared_system(frames=500, shared=16, seed=3, **kw):
    """The chunk system plus ``shared`` offsets every frame uses: one per frame-local column ``xx_``, the
    same unknown in all frames (a detector-fixed column pattern), with a fixed pattern added to the
    data. Each shared unknown couples to every frame's columns: in the offsets' normal matrix it is a
    hub. Returns (A, b, number of sky columns, grid, number of shared columns)."""
    A, b, npix, grid = _chunk_system(frames=frames, seed=seed, **kw)
    pattern = 0.1 * np.sin(np.arange(shared) * 0.9)
    n = A.shape[1]
    coo = A.tocoo()
    rows, cols, vals = list(coo.row), list(coo.col), list(coo.data)
    b = b.copy()
    # the data rows come in the frame order of _chunk_system: 8 x 16 pixels then one anchor row per frame
    r = 0
    for _f in range(frames):
        for yy_ in range(8):
            for xx_ in range(16):
                rows.append(r)
                cols.append(n + xx_ % shared)
                vals.append(1.0)
                b[r] += pattern[xx_ % shared]
                r += 1
        r += 1
    A2 = sp.csr_matrix((vals, (rows, cols)), shape=(A.shape[0], n + shared))
    return A2, b, npix, grid, shared

CHUNK = _chunk_system()
NEAR_NULL = _near_null_system()


@pytest.fixture(scope='module', autouse=True)
def scratch(tmp_path_factory):
    global _SCRATCH
    _SCRATCH = str(tmp_path_factory.mktemp('scratch'))
    yield _SCRATCH
    assert os.listdir(_SCRATCH) == []                   # every parked copy was removed


_SCRATCH = None


def _solve(system, deflate=None, compact=True, iter_lim=400, solver='lsqr', n_threads=1, float32=False,
           x0=None, **kw):
    A, b, _, grid = system
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    Ause = A[:, active].tocsr() if compact else A.tocsr()
    opts = dict(ref_shape=grid, atol=0.0, btol=0.0, conlim=0.0, damp=0.0, solver=solver, n_threads=n_threads,
                use_float32=float32, iter_lim=iter_lim, active_mask=active if compact else None,
                num_cols_full=A.shape[1] if compact else None, return_record=True, scratch_dir=_SCRATCH,
                sky_names=('sky',))
    opts.update(kw)
    return apply_lsqr(Ause.copy(), b.copy(), x0=None if x0 is None else x0.copy(), deflate=deflate, **opts)


def _converged(system, solver='lsqr'):
    """The least-squares solution plain LSQR from zero converges to: the minimum-norm one in the solver's
    column-scaled unknowns (a dense SVD solve of the small system), in the full layout."""
    A, b, _, _ = system
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    Ac = A[:, active].toarray()
    cn = np.sqrt((Ac * Ac).sum(axis=0))
    y = np.linalg.lstsq(Ac / cn, b, rcond=1e-12)[0]
    x = np.zeros(A.shape[1])
    x[active] = y / cn
    return x


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# =================================================================== offsets many frames share (hub columns)
def test_offsets_shared_by_every_frame_are_eliminated_by_a_schur_complement(caplog):
    """A column pattern every frame shares turns the offsets' normal matrix into one connected block
    (500 frames x 9 columns + 16 shared: above the dense limit). The shared columns are hubs: eliminated
    by a Schur complement, the blocks stay per frame, the pairing is exact, and the deflated solve
    reaches plain least squares' solution (gauges included)."""
    import logging
    A, b, npix, grid, shared = _shared_system()
    system = (A, b, npix, grid)
    x_plain = _converged(system)
    with caplog.at_level(logging.INFO, logger='selfcal.core.deflation'):
        x, rec = _solve(system, deflate=sc.Deflate(3, basis='polynomial'), iter_lim=600)
    assert _rel(x, x_plain) < 1e-7
    d = json.loads(rec.deflation)
    assert d['pairing_residual'] < 1e-10 and d['resolve_change'] < 1e-6
    msg = [m for m in caplog.messages if 'offset response solved' in m]
    assert msg and f'{shared} columns shared by many frames eliminated by a Schur complement' in msg[0]
    assert 'on 2500 blocks (largest 5,' in msg[0]   # per frame: the 4 constants + the scalar; the 4 slopes alone (orthogonal)


def test_the_schur_path_solves_the_normal_equations_exactly(monkeypatch):
    """On a small shared system, forced through the Schur path by a low dense limit: G X = R to rounding,
    and the same as the dense pseudo-inverse's G X."""
    import selfcal.core.deflation as D
    A, b, npix, grid, shared = _shared_system(frames=40)
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    Ac = A[:, active].tocsr()
    off = Ac[:, npix:]
    G = (off.T @ off).tocsr()
    G.sum_duplicates()
    rng = np.random.default_rng(0)
    R = rng.normal(size=(G.shape[0], 3))
    R = G @ R                                             # a consistent right-hand side
    X_dense, info_dense = D._block_pinv_solve(G, R)
    assert 'hubs' not in info_dense                       # one block of 376 columns: dense, no hub
    monkeypatch.setattr(D, '_DENSE_BLOCK', 16)
    X_schur, info = D._block_pinv_solve(G, R)
    assert info['hubs'] == shared and info['largest'] <= 9 and info['blocks'] >= 40     # per frame: at most 9 columns
    assert np.linalg.norm(G @ X_schur - R) < 1e-9 * np.linalg.norm(R)
    assert np.linalg.norm(G @ X_schur - G @ X_dense) < 1e-9 * np.linalg.norm(R)


def test_the_schur_path_stays_put_past_convergence():
    """The shared-column system far past its convergence (~125 iterations), CGLS at a fixed count: the solution
    stays at plain least squares' (a response with a null-space component would let it drift: the Schur
    path returns the minimum-norm one)."""
    A, b, npix, grid, _ = _shared_system()
    system = (A, b, npix, grid)
    x_plain = _converged(system)
    x, _rec = _solve(system, deflate=sc.Deflate(3, basis='polynomial'), iter_lim=1000, solver='cgls')
    assert _rel(x, x_plain) < 1e-7


# =================================================================== the method
def test_the_deflated_solution_is_the_least_squares_solution_gauge_included():
    """Polynomial patterns contain both exact gauges of the chunk system (the sky's mean against the
    scalars, the x-plane against the slopes): the deflated solve reaches plain LSQR's solution itself."""
    x_plain = _converged(CHUNK)
    x, rec = _solve(CHUNK, deflate=sc.Deflate(3, basis='polynomial'))
    assert _rel(x, x_plain) < 1e-7
    d = json.loads(rec.deflation)
    assert d['patterns'] == 16 and d['gauges'] == 2 and d['kept'] == 14
    assert d['pairing_residual'] < 1e-10 and d['nullness_min'] < 1e-8 and d['resolve_change'] < 1e-6
    assert rec.deflation_s is not None and rec.deflation_s >= 0
    assert 'deflation' in rec.attrs() and 'deflation_s' in rec.attrs()


def test_the_cosine_patterns_recognise_a_plane_gauge_whole():
    """Cosines alone contain the x-plane gauge only in part (the solve would drift along it once
    converged); the low-degree polynomials every basis includes put it in the space whole: the same
    solution, gauge included, and it stays there far past convergence."""
    x_plain = _converged(CHUNK)
    for n in (150, 1500):
        x, rec = _solve(CHUNK, deflate=sc.Deflate(3), iter_lim=n)
        d = json.loads(rec.deflation)
        assert d['patterns'] == 16 + 5 and d['gauges'] == 2
        assert _rel(x, x_plain) < 1e-7, n


def _x_gradient(x, system=NEAR_NULL):
    """The coefficient of x in the degree-2 fit of the sky of a full-layout solution."""
    A, b, npix, grid = system
    fit = SmoothFit(grid, ('sky',), degree=2, step=1)
    fit.bind(None)
    return float(fit.fit(Iterate(0, x[:npix], num_cols=npix))[0][1])


@pytest.mark.parametrize('solver', ['lsqr', 'lsmr'])
def test_the_large_scales_converge_in_far_fewer_iterations(solver):
    """On a system whose large scales along x are near-null, plain iterations are still far from the
    field-wide gradient after 15 iterations; the deflated solve has it (setup included: 2 passes)."""
    converged = _x_gradient(_converged(NEAR_NULL, solver=solver))
    plain = _x_gradient(_solve(NEAR_NULL, iter_lim=15, solver=solver)[0])
    x, rec = _solve(NEAR_NULL, deflate=sc.Deflate(4), iter_lim=15, solver=solver)
    assert abs(plain / converged - 1) > 0.1
    assert abs(_x_gradient(x) / converged - 1) < 1e-3
    assert json.loads(rec.deflation)['nullness_min'] < 0.05          # the slow patterns are near-null


def test_the_same_solution_compacted_or_not_and_with_threads():
    A, b, npix, grid = CHUNK
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    xc, rc = _solve(CHUNK, deflate=sc.Deflate(3, basis='polynomial'), iter_lim=60)
    xu, ru = _solve(CHUNK, deflate=sc.Deflate(3, basis='polynomial'), iter_lim=60, compact=False)
    xt, rt = _solve(CHUNK, deflate=sc.Deflate(3, basis='polynomial'), iter_lim=60, n_threads=3)
    assert np.all(xu[~active] == 0)                       # uncovered columns stay empty
    assert _rel(xu, xc) < 1e-9 and _rel(xt, xc) < 1e-9
    assert json.loads(ru.deflation)['columns'] == json.loads(rc.deflation)['columns'] == int(active.sum())


def test_float32_solves_reach_float32_precision_sooner():
    """In float32 the deflated solve reaches the solution to float32 precision in a fraction of the plain
    iterations. (Run far past convergence, float32 LSQR loses it again, deflated or not: an LSQR matter.)"""
    x64 = _converged(CHUNK)
    x32, rec = _solve(CHUNK, deflate=sc.Deflate(3), iter_lim=60, float32=True)
    plain = _solve(CHUNK, iter_lim=60, float32=True)[0]
    assert _rel(x32, x64) < 1e-5 < 1e-3 < _rel(plain, x64)


def test_stable_past_convergence():
    """Run far past convergence (the Krylov space of this small system is exhausted): the solution stays."""
    x_plain = _converged(CHUNK)
    for n in (200, 1500):
        x, rec = _solve(CHUNK, deflate=sc.Deflate(3, basis='polynomial'), iter_lim=n)
        assert _rel(x, x_plain) < 1e-7, n


def test_a_warm_start_at_the_solution_stays_there():
    A, b, npix, grid = CHUNK
    x_plain = _converged(CHUNK)
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    x, rec = _solve(CHUNK, deflate=sc.Deflate(2, basis='polynomial'), x0=x_plain, iter_lim=5)
    assert _rel(x, x_plain) < 1e-7
    assert abs(rec.history.arrays()['r1norm'][0] / np.linalg.norm(b - A @ x_plain) - 1) < 1e-8
    assert active.sum() > 0


def test_monitors_and_snapshots_see_the_solution():
    """A snapshot after k iterations is the solution of a k-iteration deflated solve; the monitors'
    true residual at a check is that solution's."""
    A, b, npix, grid = NEAR_NULL
    seen = {}

    def snap(it):
        seen[it.itn] = it.read(0, it.num_cols).copy()

    x10, _ = _solve(NEAR_NULL, deflate=sc.Deflate(3), iter_lim=10)
    x, rec = _solve(NEAR_NULL, deflate=sc.Deflate(3), iter_lim=20, snapshot=snap, snapshot_every=10,
                    monitor=sc.Monitor(10, gradient=True))
    assert sorted(seen) == [10]
    assert _rel(seen[10], x10) < 1e-10
    arr = rec.watch.arrays()
    assert list(arr['check_itn']) == [0, 10, 20]
    assert abs(arr['true_residual'][1] / np.linalg.norm(b - A @ x10) - 1) < 1e-8
    np.testing.assert_allclose(arr['residual_ratio'], 1, rtol=1e-6)       # the estimates are the full system's


def test_only_the_terms_asked_and_two_sky_terms():
    """Two sky terms (a line term whose coefficient varies from observation to observation): deflating
    both or only one reaches the least-squares solution."""
    A1, b, npix, grid = CHUNK
    rng = np.random.default_rng(3)
    A1 = A1.tocoo()
    coef = 0.2 + 1.6 * rng.random(A1.shape[0])
    sky = A1.col < npix
    rows = np.concatenate([A1.row, A1.row[sky]])
    cols = np.concatenate([np.where(sky, A1.col, A1.col + npix), A1.col[sky] + npix])
    vals = np.concatenate([A1.data, A1.data[sky] * coef[A1.row[sky]]])
    A2 = sp.csr_matrix((vals, (rows, cols)), shape=(A1.shape[0], A1.shape[1] + npix))
    system = (A2, b + 0.1 * coef * (b != 0), npix, grid)
    x_plain = _converged(system)
    for deflate in (sc.Deflate(3), sc.Deflate(3, terms=('continuum',))):
        x, rec = _solve(system, deflate=deflate, iter_lim=2000, sky_names=('continuum', 'line'))
        assert np.linalg.norm(A2 @ (x - x_plain)) / np.linalg.norm(A2 @ x_plain) < 1e-8
        assert json.loads(rec.deflation)['patterns'] == 21 * (2 if deflate.terms is None else 1)


def test_what_deflation_refuses():
    with pytest.raises(ValueError, match='undamped'):
        _solve(CHUNK, deflate=sc.Deflate(2), damp=0.1)
    A, b, npix, grid = CHUNK
    with pytest.raises(ValueError, match='COO'):
        apply_lsqr(A.tocoo(), b.copy(), ref_shape=grid, deflate=sc.Deflate(2), damp=0.0)
    with pytest.raises(ValueError, match="no sky term"):
        _solve(CHUNK, deflate=sc.Deflate(2, terms=('line',)))
    with pytest.raises(ConfigError, match='undamped'):
        sc.Fit(deflate=sc.Deflate(), damp=0.01)
    with pytest.raises(ConfigError):
        sc.Deflate(-1)
    with pytest.raises(ConfigError):
        sc.Deflate(2, basis='fourier')
    assert sc.Deflate(2, terms='continuum').terms == ('continuum',)
    with pytest.raises(ConfigError):
        sc.Deflate(2, storage='float16')
    with pytest.raises(ValueError, match="needs float32 products"):
        _solve(CHUNK, deflate=sc.Deflate(2, storage='float32'), iter_lim=5)
    for store, f32 in (('float64', True), (None, True), (None, False), ('float32', True)):
        x, rec = _solve(CHUNK, deflate=sc.Deflate(2, storage=store), float32=f32, iter_lim=5)
        assert json.loads(rec.deflation)['storage'] == ('float32' if (store == 'float32' or (store is None and f32)) else 'float64')
    assert sc.Deflate(3).patterns_per_term == 21 and sc.Deflate(3, basis='polynomial').patterns_per_term == 16
    assert sc.Deflate(1, basis='polynomial').patterns_per_term == 6 and sc.Deflate(0).patterns_per_term == 6


def test_the_pattern_functions():
    c = axis_functions('cosine', 3, 8)
    p = axis_functions('polynomial', 3, 8)
    assert c.shape == p.shape == (4, 8)
    np.testing.assert_allclose(c[0], 1)
    np.testing.assert_allclose(c[1], np.cos(np.pi * (np.arange(8) + 0.5) / 8))
    np.testing.assert_allclose(p[1], 2 * (np.arange(8) + 0.5) / 8 - 1)


def test_the_space_is_orthonormal_and_diagonalises_the_normal_matrix():
    A, b, npix, grid = CHUNK
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    Ac = A[:, active].tocsr()
    cn = np.sqrt(np.asarray(Ac.multiply(Ac).sum(axis=0)).ravel())
    B = (Ac @ sp.diags(1 / cn)).tocsr()
    D = Deflation(B, b, scale=1 / cn, active_mask=active, ref_shape=grid, sky_names=('sky',),
                  settings=sc.Deflate(3, basis='polynomial'), threads=1)
    V = D.V.T                                   # one direction per row of D.V
    np.testing.assert_allclose(V.T @ V, np.eye(D.kept), atol=1e-12)
    np.testing.assert_allclose(V.T @ (B.T @ (B @ V)), np.diag(D.lam), atol=1e-12 * D.lam.max())
    np.testing.assert_allclose(D.W.T, B.T @ (B @ V), atol=1e-12 * np.abs(D.W).max())
    D.close()
    # the same space, threaded or not (the patterns' passes in threads: disjoint outputs, the same sums)
    D4 = Deflation(B, b, scale=1 / cn, active_mask=active, ref_shape=grid, sky_names=('sky',),
                   settings=sc.Deflate(3, basis='polynomial'), threads=4)
    assert np.array_equal(D4.V, D.V) and np.array_equal(D4.W, D.W) and np.array_equal(D4.lam, D.lam)
    D4.close()


# =================================================================== through the API
NUMERICS = sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4)


def _recipe(iterations=30, name='d', **fit):
    return sc.Recipe(sc.continuum(), fit=sc.Fit(iterations, tolerance=0, **fit), coadd=None, numerics=NUMERICS,
                     name=name)


@pytest.fixture(scope='module')
def field():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_deflate_')
    write_exposures(os.path.join(tmp, 'exposures'), 12, np.random.default_rng(5))
    f = sc.Field(os.path.join(tmp, 'out', 'toy'), sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy'),
                 REF_ARCSEC, compute=sc.Compute(os.path.join(tmp, 'cache'), workers=2, io_limit=4))
    f.reproject(os.path.join(tmp, 'exposures', 'toy_exp_*_D0.fits'), method='interp', padding=8)
    yield f
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.filterwarnings('ignore:.*recorded with code:UserWarning')
def test_a_deflated_calibration_is_recorded_and_fingerprinted(field, caplog):
    import logging
    (job,) = field.instrument.default_jobs()
    book = lambda r: field.plan(r).book.cal(job, field.frames)          # noqa: E731
    plain, deflated = _recipe(20, name='same'), _recipe(20, name='same', deflate=sc.Deflate(2))
    assert book(plain) != book(deflated)                    # the setting is part of the fingerprint when given
    assert book(plain) == book(_recipe(20, name='same', deflate=None))
    assert 'deflate     14 patterns (cosine, order 2) of every sky term' in str(field.plan(
        _recipe(20, name='d1', deflate=sc.Deflate(2))))
    with caplog.at_level(logging.INFO, logger='selfcal'):
        res = field.calibrate(_recipe(20, name='d1', deflate=sc.Deflate(2)), monitor=sc.Monitor(5))
    assert 'Deflation: ' in caplog.text and 'directions solved directly' in caplog.text and 'off |b|^2' in caplog.text
    cal = res.cal_paths[0]
    with h5py.File(cal, 'r') as f:
        attrs = dict(f['solve'].attrs)
    assert 'deflation_s' not in attrs                      # the setup time stays out of the cal
    d = json.loads(attrs['deflation'])
    assert d['basis'] == 'cosine' and d['order'] == 2 and d['patterns'] == 14 and d['kept'] >= 1
    (entry,) = json.load(open(res.record))['solves']
    assert entry['deflation']['patterns'] == 14 and entry['deflation_s'] >= 0
    # closer to the converged sky than plain iterations, and still there far past convergence
    def sky(recipe):
        with CalFile(field.calibrate(recipe).cal_paths[0]) as c:
            return np.nan_to_num(np.asarray(c.sky(0), dtype=np.float64))

    ref = sky(_recipe(300, name='ref64', float32=False))
    covered = ref != 0

    def err(s):
        d, r = (s - ref)[covered], ref[covered]
        return np.sqrt(np.mean((d - d.mean()) ** 2)) / np.sqrt(np.mean((r - r.mean()) ** 2))

    with CalFile(cal) as c:
        deflated20 = np.nan_to_num(np.asarray(c.sky(0), dtype=np.float64))
    assert err(deflated20) < err(sky(_recipe(20, name='plain20')))
    assert err(sky(_recipe(160, name='d160', deflate=sc.Deflate(2)))) < 1e-5
