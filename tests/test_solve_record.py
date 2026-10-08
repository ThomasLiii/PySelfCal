"""The record of a solve (selfcal.core.solve_record): a fixed iteration count, and what the solve leaves.

``Fit(iterations=N, tolerance=0)`` runs exactly N iterations of LSQR or LSMR (the tolerance tests and
the condition-estimate stop off) on a system where the default tolerances stop earlier; every solve
leaves its record (the method, the iterations, ``istop`` and its meaning, the final estimates, the
tolerances, the wall time, the true residual ``|b - A x|``) in the cal file's ``solve`` group and in
the action's record, and its history per iteration in ``<field>/records/<cal stem>_history.npz``.
The recording solvers are scipy's bit for bit.
"""
import json
import os
import sys

import h5py
import numpy as np
import pytest
import scipy.sparse as sp
from scipy.sparse.linalg import lsmr as scipy_lsmr
from scipy.sparse.linalg import lsqr as scipy_lsqr

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import selfcal as sc  # noqa: E402
from selfcal import _state  # noqa: E402
from selfcal.config import ConfigError  # noqa: E402
from selfcal.core.lsmr import lsmr  # noqa: E402
from selfcal.core.lsqr_inplace import lsqr_inplace  # noqa: E402
from selfcal.core.solve import apply_lsqr  # noqa: E402
from selfcal.core.solve_record import HISTORY_COLUMNS, STOP_REASONS, SolveHistory  # noqa: E402
from selfcal.core.spill import ParkedVector  # noqa: E402
from selfcal.io.calfile import CalFile  # noqa: E402
from selfcal.run.lower import solver_options  # noqa: E402
from tests.synthetic_exposures import DET, N_CHUNK_SIDE, REF_ARCSEC, write_exposures  # noqa: E402

N = 25            # the fixed iteration count of the toy solves


def _system(dtype=np.float64, m=400, n=60, seed=0):
    """A small inconsistent least-squares system on which the default tolerances (1e-6) stop LSQR and
    LSMR after about 12 iterations and machine precision is reached after about 30."""
    A = sp.random(m, n, density=0.2, random_state=seed + 1, format='csr', dtype=np.float64).astype(dtype)
    b = np.random.default_rng(seed).normal(size=m).astype(dtype)
    return A, b


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# =================================================================== the recording solvers
# scipy's LSMR overflows a float32 scalar once a float32 system has converged; ours does the same
@pytest.mark.filterwarnings('ignore:overflow encountered in cast:RuntimeWarning')
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
@pytest.mark.parametrize('with_x0', [False, True])
@pytest.mark.parametrize('damp', [0.0, 0.3])
@pytest.mark.parametrize('conlim', [1e8, 0.0])
def test_the_recording_solvers_are_scipys_bit_for_bit(dtype, with_x0, damp, conlim):
    A, b = _system(dtype)
    x0 = np.random.default_rng(3).normal(size=A.shape[1]).astype(dtype) if with_x0 else None
    kw = dict(damp=damp, atol=1e-9, btol=1e-9, conlim=conlim)
    for ours, theirs, limit in ((lsqr_inplace, scipy_lsqr, 'iter_lim'), (lsmr, scipy_lsmr, 'maxiter')):
        h = SolveHistory()
        got = ours(A, b.copy(), x0=None if x0 is None else x0.copy(), history=h, **{limit: 40}, **kw)
        want = theirs(A, b.copy(), x0=None if x0 is None else x0.copy(), **{limit: 40}, **kw)
        assert _same(got[0], want[0]), ours.__name__
        n_scalars = 9 if ours is lsqr_inplace else 8          # LSQR's var (not computed here) is left out
        for g, w in zip(got[1:n_scalars], want[1:n_scalars]):
            assert g == w and type(g) is type(w), ours.__name__
        rows = h.arrays()
        assert list(rows) == list(HISTORY_COLUMNS)
        assert np.array_equal(rows['itn'], np.arange(got[2] + 1))      # row 0: the starting state
        if ours is lsqr_inplace:
            _, istop, itn, r1, r2, an, ac, ar, xn = got[:9]
        else:
            _, istop, itn, r1, ar, an, ac, xn = got
            r2 = r1
        last = h.last()
        assert (last['r1norm'], last['r2norm'], last['arnorm'], last['anorm'], last['acond'], last['xnorm']) == \
            (float(r1), float(r2), float(ar), float(an), float(ac), float(xn))
        assert np.all(np.diff(rows['elapsed_s']) >= 0)


# =================================================================== exactly N iterations, and the record
@pytest.mark.parametrize('method', ['lsqr', 'lsmr'])
def test_tolerance_zero_runs_exactly_n_iterations(method):
    A, b = _system()
    kw = dict(damp=0.0, iter_lim=N, solver=method, n_threads=1, use_float32=False, return_record=True)
    _, early = apply_lsqr(A.copy(), b.copy(), (1, 1), atol=1e-6, btol=1e-6, **kw)
    assert early.iterations < N and early.istop in (1, 2)                # the default tolerances stop earlier
    x, rec = apply_lsqr(A.copy(), b.copy(), (1, 1), atol=0.0, btol=0.0, conlim=0.0, **kw)
    assert rec.iterations == rec.iterations_total == rec.iteration_limit == N
    assert rec.istop == 7 and rec.stop == STOP_REASONS[7]
    assert (rec.method, rec.atol, rec.btol, rec.conlim, rec.damp) == (method, 0.0, 0.0, 0.0, 0.0)
    assert (rec.rows, rec.columns) == A.shape and rec.wall_s >= 0
    h = rec.history.arrays()
    assert np.array_equal(h['itn'], np.arange(N + 1))
    assert h['r1norm'][-1] == rec.r1norm and h['arnorm'][-1] == rec.arnorm and h['xnorm'][-1] == rec.xnorm
    assert h['anorm'][-1] == rec.anorm and h['acond'][-1] == rec.acond
    # the true residual and |b|, against an independent computation with the unscaled matrix
    assert rec.true_residual == pytest.approx(np.linalg.norm(b - A @ x), rel=1e-12)
    assert rec.bnorm == pytest.approx(np.linalg.norm(b), rel=1e-14)
    assert rec.r1norm == pytest.approx(rec.true_residual, rel=1e-8)     # the estimate has not drifted here
    attrs = rec.attrs()
    assert attrs['version'] == 1 and 'history' not in attrs and 'history_file' not in attrs


def test_the_condition_stop_is_the_solvers_own_unless_turned_off():
    """conlim is passed explicitly: 1e8 (the solvers' default) gives scipy's result; a small conlim stops
    with istop 3."""
    A, b = _system()
    x, rec = apply_lsqr(A.copy(), b.copy(), (1, 1), damp=0.0, iter_lim=N, solver='lsqr', n_threads=1,
                        atol=0.0, btol=0.0, conlim=2.0, return_record=True)
    assert rec.istop == 3 and rec.iterations < N and rec.conlim == 2.0


def test_the_right_hand_side_kept_on_disk_gives_the_same_record(monkeypatch, tmp_path):
    A, b = _system(np.float32)
    kw = dict(damp=0.0, iter_lim=N, solver='lsqr', n_threads=2, use_float32=True, atol=0.0, btol=0.0, conlim=0.0,
              return_record=True)
    x_mem, mem = apply_lsqr(A.copy(), b.copy(), (1, 1), **kw)
    monkeypatch.setenv('SELFCAL_SPILL_MIN_GB', '0')
    monkeypatch.setenv('SELFCAL_SPILL_DIR', str(tmp_path))
    x_disk, disk = apply_lsqr(A.copy(), b.copy(), (1, 1), **kw)
    assert _same(x_mem, x_disk)
    assert disk.true_residual == mem.true_residual and disk.bnorm == mem.bnorm
    assert list(tmp_path.iterdir()) == []                                   # the parked copy is gone


def test_a_parked_vector_round_trips_exactly(tmp_path, monkeypatch):
    v = np.random.default_rng(2).normal(size=1000).astype(np.float32)
    small = ParkedVector(v)
    assert not small.on_disk and _same(small.array(), v)
    monkeypatch.setenv('SELFCAL_SPILL_DIR', str(tmp_path))
    big = ParkedVector(v, min_gb=0)
    assert big.on_disk and _same(np.asarray(big.array()), v) and len(list(tmp_path.iterdir())) == 1
    big.discard()
    small.discard()
    assert list(tmp_path.iterdir()) == []
    with pytest.raises(ValueError):
        small.array()


# =================================================================== the setting
def test_fit_tolerance_zero_means_exactly_the_iterations():
    exact = sc.Fit(30, tolerance=0)
    assert exact.tolerance == 0.0 and exact.exact_iterations
    opts = solver_options(sc.Recipe(fit=exact))
    assert (opts['atol'], opts['btol'], opts['conlim'], opts['iter_lim']) == (0.0, 0.0, 0.0, 30)
    assert sc.Fit(30, tolerance=(0, 0)).exact_iterations
    for fit in (sc.Fit(30), sc.Fit(30, tolerance=(0, 1e-6))):
        assert not fit.exact_iterations and 'conlim' not in solver_options(sc.Recipe(fit=fit))
    with pytest.raises(ConfigError, match='at least 0'):
        sc.Fit(30, tolerance=-1e-6)


# =================================================================== through the Python API
def _datasets(path):
    out = []
    with h5py.File(path, 'r') as f:
        f.visititems(lambda name, obj: out.append(name) if isinstance(obj, h5py.Dataset) else None)
    return out


@pytest.mark.parametrize('method', ['lsqr', 'lsmr'])
def test_a_calibration_records_its_solve(tmp_path, method):
    _state.set_progress(False)
    write_exposures(str(tmp_path / 'exposures'), 12, np.random.default_rng(5))
    field = sc.Field(tmp_path / 'out' / 'toy', sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy'),
                     REF_ARCSEC, compute=sc.Compute(str(tmp_path / 'cache'), workers=2, io_limit=4))
    field.reproject(str(tmp_path / 'exposures' / 'toy_exp_*_D0.fits'), method='interp', padding=8)
    numerics = sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4)
    recipe = sc.Recipe(sc.continuum(), fit=sc.Fit(N, tolerance=0, method=method), coadd=None, numerics=numerics,
                       name='exact')
    result = field.calibrate(recipe)
    cal = result.cal_paths[0]
    with CalFile(cal) as c:
        solve = c.solve
        assert 'solve:' in c.describe()
    assert solve['method'] == method and solve['iterations'] == solve['iterations_total'] == N
    assert solve['istop'] == 7 and solve['stop'] == STOP_REASONS[7]
    assert (solve['atol'], solve['btol'], solve['conlim']) == (0.0, 0.0, 0.0)
    assert solve['true_residual'] > 0 and solve['bnorm'] > solve['true_residual']
    assert 'wall_s' not in solve                                             # the cal stays byte-reproducible
    assert not any(name.startswith('solve') for name in _datasets(cal))     # attributes only, no dataset
    stem = os.path.splitext(os.path.basename(cal))[0]
    history = os.path.join(field.path, 'records', f'{stem}_history.npz')
    assert solve['history_file'] == os.path.abspath(history)
    with np.load(history, allow_pickle=False) as h:
        assert str(h['method']) == method
        assert np.array_equal(h['itn'], np.arange(N + 1))
        for k in ('r1norm', 'r2norm', 'arnorm', 'anorm', 'acond', 'xnorm'):
            assert h[k][-1] == solve[k], k
    with open(result.record) as f:
        entries = json.load(f)['solves']
    assert len(entries) == 1
    entry = entries[0]
    assert (entry['cal'], entry['job'], entry['tile']) == (cal, result.jobs[0].name, None)
    assert {k: entry[k] for k in solve} == solve and entry['wall_s'] >= 0

    # the default tolerance stops this solve before its 200 iterations
    loose = field.calibrate(recipe.replace(fit=sc.Fit(200, method=method), name='loose'))
    with CalFile(loose.cal_paths[0]) as c:
        assert c.solve['iterations'] < 200 and c.solve['istop'] in (1, 2) and c.solve['conlim'] == 1e8

    # a tiled solve: one entry and one history per tile; the stitched cal has no solve of its own
    tiles = sc.Tiles((1, 2), overlap=10, names=('W', 'E'))
    t = field.calibrate(recipe, tiles=tiles, compute=field.compute.replace(stage_dir='toy_tiles', memory_guard=False))
    with open(t.record) as f:
        entries = json.load(f)['solves']
    assert sorted(e['tile'] for e in entries) == ['E', 'W']
    for e in entries:
        assert e['cal'] == t.tile_cals[e['tile']] and e['iterations'] == N and os.path.exists(e['history_file'])
        assert os.path.basename(e['history_file']) == os.path.basename(e['cal'])[:-len('.h5')] + '_history.npz'
    with CalFile(t.final) as c:
        assert c.solve is None
