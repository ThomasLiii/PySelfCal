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
def test_tolerance_zero_runs_exactly_n_iterations(method, tmp_path):
    A, b = _system()
    kw = dict(damp=0.0, iter_lim=N, solver=method, n_threads=1, use_float32=False, return_record=True,
              scratch_dir=str(tmp_path))
    _, early = apply_lsqr(A.copy(), b.copy(), (1, 1), atol=1e-6, btol=1e-6, **kw)
    assert early.iterations < N and early.istop in (1, 2)                # the default tolerances stop earlier
    x, rec = apply_lsqr(A.copy(), b.copy(), (1, 1), atol=0.0, btol=0.0, conlim=0.0, **kw)
    assert rec.iterations == rec.iterations_total == rec.iteration_limit == N
    assert rec.istop == 7 and rec.stop_reason == STOP_REASONS[7]
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
    assert 'stop' not in attrs and attrs['stop_reason'] == STOP_REASONS[7] and 'solver_tests' not in attrs
    assert list(tmp_path.iterdir()) == []                                   # the parked copy of b is gone


def test_the_condition_stop_is_the_solvers_own_unless_turned_off():
    """conlim is passed explicitly: 1e8 (the solvers' default) gives scipy's result; a small conlim stops
    with istop 3."""
    A, b = _system()
    x, rec = apply_lsqr(A.copy(), b.copy(), (1, 1), damp=0.0, iter_lim=N, solver='lsqr', n_threads=1,
                        atol=0.0, btol=0.0, conlim=2.0, return_record=True)
    assert rec.istop == 3 and rec.iterations < N and rec.conlim == 2.0


@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_the_right_hand_side_is_parked_on_disk_whatever_its_size(monkeypatch, tmp_path, dtype):
    """LSQR's copy of b is a file in the scratch directory during the solve (never in memory, even far
    below the pixel state's spill threshold), read back for |b - A x| and removed after; LSMR keeps b."""
    A, b = _system(dtype)
    monkeypatch.setenv('SELFCAL_SPILL_MIN_GB', '1000')
    seen = []
    kw = dict(damp=0.0, iter_lim=N, n_threads=2, use_float32=dtype == np.float32, atol=0.0, btol=0.0, conlim=0.0,
              return_record=True, scratch_dir=str(tmp_path / 'scratch'), snapshot_every=5,
              snapshot=lambda it: seen.append(sorted(p.name for p in (tmp_path / 'scratch').iterdir())))
    x, rec = apply_lsqr(A.copy(), b.copy(), (1, 1), solver='lsqr', **kw)
    assert len(seen) == 4 and all(len(names) == 1 and names[0].startswith('selfcal_parked_') and
                                  names[0].endswith('.npy') for names in seen)
    assert list((tmp_path / 'scratch').iterdir()) == []                    # removed when the solve ends
    want = np.linalg.norm(b.astype(np.float64) - A.astype(np.float64) @ x.astype(np.float64))
    assert rec.true_residual == pytest.approx(want, rel=1e-5 if dtype == np.float32 else 1e-12)
    assert rec.bnorm == pytest.approx(np.linalg.norm(b.astype(np.float64)), rel=1e-12)
    seen.clear()
    x2, rec2 = apply_lsqr(A.copy(), b.copy(), (1, 1), solver='lsmr', **kw)  # LSMR keeps b: nothing parked
    assert seen and all(names == [] for names in seen) and np.isfinite(rec2.true_residual)


def test_a_right_hand_side_that_cannot_be_parked_leaves_the_solve_unchanged(tmp_path, monkeypatch, caplog):
    """No scratch directory, an unusable one, or a full disk: the solve goes on, its solution unchanged,
    and the true residual is NaN (a warning says why)."""
    import logging
    A, b = _system()
    kw = dict(damp=0.0, iter_lim=N, solver='lsqr', n_threads=1, atol=0.0, btol=0.0, conlim=0.0, return_record=True)
    x, rec = apply_lsqr(A.copy(), b.copy(), (1, 1), scratch_dir=str(tmp_path), **kw)
    assert np.isfinite(rec.true_residual)
    x_none, none = apply_lsqr(A.copy(), b.copy(), (1, 1), **kw)               # no scratch directory
    not_a_dir = tmp_path / 'a_file'
    not_a_dir.write_text('')
    with caplog.at_level(logging.WARNING, logger='selfcal.core.spill'):
        x_bad, bad = apply_lsqr(A.copy(), b.copy(), (1, 1), scratch_dir=str(not_a_dir / 'scratch'), **kw)
    assert any('Could not park' in r.getMessage() for r in caplog.records)
    real_save = np.save

    def full_disk(*a, **k):
        raise OSError(28, 'No space left on device')
    monkeypatch.setattr(np, 'save', full_disk)
    x_full, full = apply_lsqr(A.copy(), b.copy(), (1, 1), scratch_dir=str(tmp_path / 'full'), **kw)
    monkeypatch.setattr(np, 'save', real_save)
    for xi, ri in ((x_none, none), (x_bad, bad), (x_full, full)):
        assert _same(xi, x) and ri.iterations == rec.iterations and ri.r1norm == rec.r1norm
        assert np.isnan(ri.true_residual) and np.isnan(ri.bnorm)
    assert list((tmp_path / 'full').iterdir()) == []                        # no partial file left


def test_a_parked_vector_round_trips_exactly(tmp_path):
    v = np.random.default_rng(2).normal(size=1003).astype(np.float32)
    parked = ParkedVector(v, tmp_path)
    assert parked.available and len(list(tmp_path.iterdir())) == 1 and len(parked) == 1003
    assert parked.dtype == v.dtype and parked.shape == v.shape
    for a, z in ((0, 1003), (0, 0), (5, 17), (1000, 2000), (-3, None)):
        got = parked[a:z]
        assert _same(got, v[a:z]) and not isinstance(got, np.memmap)
    v[:] = 0                                                                 # a copy: the original may change
    assert parked[0:3].any()
    with pytest.raises(ValueError, match='step 1'):
        parked[::2]
    parked.discard()
    assert list(tmp_path.iterdir()) == [] and not parked.available
    with pytest.raises(ValueError, match='not available'):
        parked[0:3]
    assert not ParkedVector(v, None).available                              # no directory: not parked


def test_parked_vectors_of_dead_processes_are_swept(tmp_path):
    import socket
    import subprocess

    from selfcal.core.spill import sweep_parked
    dead = subprocess.Popen([sys.executable, '-c', 'pass'])
    dead.wait()
    host = socket.gethostname().replace('_', '-')
    names = {'dead': f'selfcal_parked_{host}_{dead.pid}_x1y2.npy', 'alive': f'selfcal_parked_{host}_{os.getpid()}_ab.npy',
             'other_host': f'selfcal_parked_elsewhere_{dead.pid}_cd.npy', 'other': 'unrelated.npy'}
    for n in names.values():
        (tmp_path / n).write_bytes(b'')
    ParkedVector(np.zeros(4), tmp_path).discard()                           # parking sweeps the directory
    assert sorted(p.name for p in tmp_path.iterdir()) == sorted(n for k, n in names.items() if k != 'dead')
    assert sweep_parked(tmp_path) == []


def test_the_vendored_givens_rotation_is_scipys():
    import itertools

    from scipy.sparse.linalg._isolve.lsqr import _sym_ortho as theirs

    from selfcal.core.lsmr import _sym_ortho as ours
    values = [0.0, -0.0, 1.0, -2.5, 3e-300, -7e200, 1e-20, np.float32(1.5), np.float32(-0.0), np.float64(-2.0)]
    with np.errstate(over='ignore'):
        for a, b in itertools.product(values, values):
            for p, q in zip(ours(a, b), theirs(a, b)):
                assert type(p) is type(q) and np.signbit(p) == np.signbit(q) and (p == q or (p != p and q != q))


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
    assert solve['istop'] == 7 and solve['stop_reason'] == STOP_REASONS[7] and 'stop' not in solve
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


def test_a_calibration_parks_the_right_hand_side_in_its_scratch_area(tmp_path, monkeypatch):
    """The run engine gives the solve its scratch area (``Compute(scratch=)``), else ``<field>/scratch/``;
    the parked copy is gone after the solve."""
    from selfcal.core import solve as solve_mod
    _state.set_progress(False)
    write_exposures(str(tmp_path / 'exposures'), 8, np.random.default_rng(5))
    used = []

    class Spy(ParkedVector):
        __slots__ = ()

        def __init__(self, vec, directory, label=''):
            used.append(directory)
            super().__init__(vec, directory, label)
    monkeypatch.setattr(solve_mod, 'ParkedVector', Spy)
    camera = sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy')
    recipe = sc.Recipe(sc.continuum(), fit=sc.Fit(5, tolerance=0, method='lsqr'), coadd=None,
                       numerics=sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4), name='park')
    for name, compute, where in (('with', sc.Compute(str(tmp_path / 'fast'), workers=2, io_limit=4), tmp_path / 'fast'),
                                 ('without', sc.Compute(workers=2, io_limit=4), None)):
        field = sc.Field(tmp_path / 'out' / name, camera, REF_ARCSEC, compute=compute)
        field.reproject(str(tmp_path / 'exposures' / 'toy_exp_*_D0.fits'), method='interp', padding=8)
        where = str(where or os.path.join(field.path, 'scratch'))
        assert field.plan(recipe).contexts[0].solve_scratch() == where
        cal = field.calibrate(recipe).cal_paths[0]
        assert used[-1] == where and not [p for p in os.listdir(where) if p.startswith('selfcal_parked_')]
        with CalFile(cal) as c:
            assert np.isfinite(c.solve['true_residual']) and c.solve['true_residual'] > 0
