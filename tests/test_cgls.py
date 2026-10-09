"""CGLS (selfcal.core.cgls; ``sc.Fit(method="cgls")``, the default) and the solver's vector precision
(``sc.Fit(vectors=...)``).

CGLS takes the same products as LSQR and reaches the same least-squares solution; it carries its residual,
recomputes it from the data every 50 iterations, and its steps vanish with the gradient, so a converged
solve stays put however many iterations follow. It takes every hook of the other solvers (history,
snapshots, monitors and stop rules, warm start, deflation, damping) and the same stopping tests on the same
quantities. ``vectors="float64"`` gives LSQR and LSMR float64 vectors on a float32 matrix; the default keeps
each method's precision, so LSQR runs are unchanged. Through the API: CGLS is the default method, LSQR
products keep their fingerprint when the method is named, and the solve record says which method ran.
"""
import json
import os
import shutil
import sys
import tempfile

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
from selfcal.core import cgls as cgls_module  # noqa: E402
from selfcal.core.cgls import cgls  # noqa: E402
from selfcal.core.lsqr_inplace import lsqr_inplace  # noqa: E402
from selfcal.core.solve import apply_lsqr  # noqa: E402
from selfcal.core.solve_record import SolveHistory  # noqa: E402
from selfcal.io.calfile import CalFile  # noqa: E402
from tests.synthetic_exposures import DET, N_CHUNK_SIDE, REF_ARCSEC, write_exposures  # noqa: E402
from tests.test_deflation import CHUNK, NEAR_NULL, _converged, _x_gradient  # noqa: E402

pytestmark = pytest.mark.filterwarnings('ignore:overflow encountered in cast:RuntimeWarning')


@pytest.fixture(scope='module', autouse=True)
def scratch(tmp_path_factory):
    global _SCRATCH
    _SCRATCH = str(tmp_path_factory.mktemp('scratch'))
    yield _SCRATCH
    assert os.listdir(_SCRATCH) == []


_SCRATCH = None


def _scaled(system, compact=True):
    """The column-scaled system the solver sees: (B, b, scale, active)."""
    A, b, _, _ = system
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    Ac = A[:, active].tocsr() if compact else A.tocsr()
    cn = np.sqrt(np.asarray(Ac.multiply(Ac).sum(axis=0)).ravel())
    cn[cn == 0] = 1.0
    return (Ac @ sp.diags(1 / cn)).tocsr(), b.copy(), 1 / cn, active


def _solve(system, solver='cgls', iter_lim=400, float32=False, vectors=None, x0=None, **kw):
    A, b, _, grid = system
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    opts = dict(ref_shape=grid, atol=0.0, btol=0.0, conlim=0.0, damp=0.0, solver=solver, n_threads=1,
                use_float32=float32, iter_lim=iter_lim, active_mask=active, num_cols_full=A.shape[1],
                return_record=True, scratch_dir=_SCRATCH, sky_names=('sky',), vectors=vectors)
    opts.update(kw)
    return apply_lsqr(A[:, active].tocsr(), b.copy(), x0=None if x0 is None else x0.copy(), **opts)


def _rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


# =================================================================== the method
def test_cgls_reaches_lsqrs_solution_with_the_same_iterates():
    """Iterate for iterate the same solve as LSQR while converging; the same solution at the end."""
    B, b, _, _ = _scaled(CHUNK)
    x_star = _converged(CHUNK)[_scaled(CHUNK)[3]] / _scaled(CHUNK)[2]
    for n in (10, 30, 60):
        xc = cgls(B, b.copy(), atol=0, btol=0, iter_lim=n)[0]
        xl = lsqr_inplace(B, b.copy(), atol=0, btol=0, conlim=0, iter_lim=n)[0]
        assert _rel(xc, xl) < 1e-6, n
    xc, istop, itn = cgls(B, b.copy(), atol=0, btol=0, iter_lim=400)[:3]
    assert _rel(xc, x_star) < 1e-9 and istop == 7 and itn == 400


def test_a_converged_cgls_solve_stays_put():
    """Far past convergence the iterate does not move (the step vanishes with the gradient; the residual
    is recomputed every REFRESH iterations), where float32 LSQR on this system drifts."""
    B, b, _, _ = _scaled(CHUNK)
    x_star = _converged(CHUNK)[_scaled(CHUNK)[3]] / _scaled(CHUNK)[2]
    B32 = B.astype(np.float32)
    for n in (200, 2000):
        x = cgls(B32, b.astype(np.float32), atol=0, btol=0, iter_lim=n)[0]
        assert _rel(x, x_star) < 1e-4, n
    assert cgls_module.REFRESH == 50


def test_the_stopping_tests_and_the_record():
    """LSQR's tests on LSQR's quantities: the gradient test stops a converging solve; the history carries
    the same columns; the record says cgls."""
    x, rec = _solve(CHUNK, atol=1e-8, btol=0.0, conlim=1e8, iter_lim=1000)
    assert rec.method == 'cgls' and rec.istop == 2 and rec.iterations < 1000
    assert rec.acond == 0.0 and rec.anorm > 0
    h = rec.history.arrays()
    assert list(h['itn']) == list(range(rec.iterations + 1))
    assert np.all(np.diff(h['r1norm']) <= 1e-12) and np.isnan(h['test2'][0])
    assert abs(rec.true_residual / rec.r1norm - 1) < 1e-6
    x_plain = _converged(CHUNK)
    assert _rel(x, x_plain) < 1e-6


def test_hooks_history_snapshots_monitors_warm_start_and_damping():
    A, b, npix, grid = CHUNK
    seen = []
    x20, _ = _solve(CHUNK, iter_lim=20)
    x, rec = _solve(CHUNK, iter_lim=40, snapshot=lambda it: seen.append((it.itn, it.read(0, it.num_cols).copy())),
                    snapshot_every=20, monitor=sc.Monitor(20, gradient=True))
    assert [it for it, _ in seen] == [20] and _rel(seen[0][1], x20) < 1e-12
    arr = rec.watch.arrays()
    assert list(arr['check_itn']) == [0, 20, 40]
    np.testing.assert_allclose(arr['residual_ratio'], 1, rtol=1e-8)
    np.testing.assert_allclose(arr['gradient_ratio'], 1, rtol=1e-6)      # the gradient CGLS carries is exact
    # a stop rule
    x_s, rec_s = _solve(CHUNK, iter_lim=400, stop=sc.Stop(gradient=1e-6))
    assert rec_s.istop == 8 and rec_s.stop_rule == 'gradient'
    # warm start at the solution: stays
    x_plain = _converged(CHUNK)
    x_w, rec_w = _solve(CHUNK, iter_lim=5, x0=x_plain)
    assert _rel(x_w, x_plain) < 1e-7
    # damping: the damped normal equations' solution
    Bs, bs, scale, active = _scaled(CHUNK)
    d = 0.3
    xd = cgls(Bs, bs.copy(), damp=d, atol=0, btol=0, iter_lim=600)[0]
    xl = lsqr_inplace(Bs, bs.copy(), damp=d, atol=0, btol=0, conlim=0, iter_lim=600)[0]
    assert _rel(xd, xl) < 1e-7


def test_cgls_with_deflation_and_on_the_near_null_system():
    """Deflated CGLS has the near-null system's field-wide gradient after 15 iterations; plain CGLS has not."""
    converged = _x_gradient(_converged(NEAR_NULL))
    plain = _x_gradient(_solve(NEAR_NULL, iter_lim=15)[0])
    x, rec = _solve(NEAR_NULL, iter_lim=15, deflate=sc.Deflate(4))
    assert abs(plain / converged - 1) > 0.1 and abs(_x_gradient(x) / converged - 1) < 1e-3
    assert json.loads(rec.deflation)['kept'] >= 1 and rec.method == 'cgls'
    # past the residual refresh (every 50 iterations) the solver's residual is the full system's true one
    x, rec = _solve(CHUNK, iter_lim=120, deflate=sc.Deflate(3, basis='polynomial'), monitor=sc.Monitor(20))
    assert abs(rec.r1norm / rec.true_residual - 1) < 1e-6
    np.testing.assert_allclose(rec.watch.arrays()['residual_ratio'], 1, rtol=1e-6)
    assert _rel(x, _converged(CHUNK)) < 1e-6


def test_vector_precision():
    """float32 products: CGLS keeps float64 vectors (default) or float32 ones; LSQR gets float64 vectors on
    the float32 matrix with vectors='float64' and keeps its float32 ones otherwise (byte-identical to
    before). Every run converges to the float64 solution to the products' precision, and returns it in that
    precision."""
    x_star = _converged(CHUNK)
    x64, _ = _solve(CHUNK, iter_lim=400, float32=True)
    x32, _ = _solve(CHUNK, iter_lim=400, float32=True, vectors='float32')
    l32a, _ = _solve(CHUNK, solver='lsqr', iter_lim=150, float32=True)
    l32b, _ = _solve(CHUNK, solver='lsqr', iter_lim=150, float32=True)
    l64, _ = _solve(CHUNK, solver='lsqr', iter_lim=150, float32=True, vectors='float64')
    m64, _ = _solve(CHUNK, solver='lsmr', iter_lim=150, float32=True, vectors='float64')
    # the solution comes back in the products' precision whatever the solver's vectors (the cal's dtype)
    assert all(x.dtype == np.float32 for x in (x64, x32, l32a, l64, m64))
    assert np.array_equal(l32a, l32b)
    for x in (x64, x32, l32a, l64, m64):
        assert _rel(x.astype(np.float64), x_star) < 1e-3
    with pytest.raises(ValueError, match="vectors='float32'"):
        _solve(CHUNK, solver='lsqr', iter_lim=5, float32=False, vectors='float32')


def test_the_settings():
    assert sc.Fit().method == 'cgls' and sc.Fit().vectors is None
    assert sc.Fit(method='lsqr').method == 'lsqr'
    with pytest.raises(ConfigError, match="vectors"):
        sc.Fit(vectors='float16')
    with pytest.raises(ConfigError, match="float32 vectors"):
        sc.Fit(vectors='float32', float32=False)
    with pytest.raises(ConfigError, match="condition"):
        sc.Fit(stop=sc.Stop(solver_tests=('condition',)))
    assert sc.Fit(method='lsqr', stop=sc.Stop(solver_tests=('condition',))).method == 'lsqr'
    # the added field keeps the encoding of a fit that does not set it
    assert 'vectors' not in sc.Fit().to_dict() and sc.Fit(vectors='float64').to_dict()['vectors'] == 'float64'


# =================================================================== through the API
NUMERICS = sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4)


@pytest.fixture(scope='module')
def field():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_cgls_')
    write_exposures(os.path.join(tmp, 'exposures'), 12, np.random.default_rng(5))
    f = sc.Field(os.path.join(tmp, 'out', 'toy'), sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy'),
                 REF_ARCSEC, compute=sc.Compute(os.path.join(tmp, 'cache'), workers=2, io_limit=4))
    f.reproject(os.path.join(tmp, 'exposures', 'toy_exp_*_D0.fits'), method='interp', padding=8)
    yield f
    shutil.rmtree(tmp, ignore_errors=True)


def _recipe(iterations=30, name='c', **fit):
    return sc.Recipe(sc.continuum(), fit=sc.Fit(iterations, tolerance=0, **fit), coadd=None, numerics=NUMERICS,
                     name=name)


@pytest.mark.filterwarnings('ignore:.*recorded with code:UserWarning')
def test_a_default_calibration_runs_cgls_and_matches_lsqr(field):
    (job,) = field.instrument.default_jobs()
    book = lambda r: field.plan(r).book.cal(job, field.frames)          # noqa: E731
    assert book(_recipe(30, name='same')) != book(_recipe(30, name='same', method='lsqr'))
    cal_c = field.calibrate(_recipe(60, name='c60')).cal_paths[0]
    cal_l = field.calibrate(_recipe(60, name='l60', method='lsqr')).cal_paths[0]
    with CalFile(cal_c) as c, CalFile(cal_l) as l:
        assert c.solve['method'] == 'cgls' and l.solve['method'] == 'lsqr'
        sc_, sl = np.nan_to_num(np.asarray(c.sky(0), dtype=np.float64)), np.nan_to_num(np.asarray(l.sky(0), dtype=np.float64))
    covered = sl != 0
    d = (sc_ - sl)[covered]
    assert np.sqrt(np.mean((d - d.mean()) ** 2)) < 1e-3 * np.std(sl[covered])
    # a long solve stays at the solution
    cal_long = field.calibrate(_recipe(300, name='c300')).cal_paths[0]
    with CalFile(cal_long) as c:
        s300 = np.nan_to_num(np.asarray(c.sky(0), dtype=np.float64))
    d = (s300 - sc_)[covered]
    assert np.sqrt(np.mean((d - d.mean()) ** 2)) < 1e-3 * np.std(sl[covered])
