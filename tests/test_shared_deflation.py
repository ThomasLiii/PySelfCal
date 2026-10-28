"""The deflation's iterative solve for offsets that many frames share (:class:`selfcal.core.deflation._SharedSolve`):
with more shared columns than a dense Schur complement holds, their couplings to the other offsets are applied through
the matrix and their part of the response solved by conjugate gradients. On the toy of :mod:`test_deflation` (a
column pattern every frame shares), forced through that path, the pairing, the deflation space and the solution are
those of the dense path, with and without a constraint on the shared columns, and when the shared columns' own normal matrix is singular."""
import json

import numpy as np
import pytest
import scipy.sparse as sp

import selfcal as sc
import selfcal.core.deflation as D

from test_deflation import _converged, _rel, _shared_system, _solve  # noqa: E402


@pytest.fixture(autouse=True)
def scratch(tmp_path, monkeypatch):
    import test_deflation
    monkeypatch.setattr(test_deflation, '_SCRATCH', str(tmp_path))
    yield


def _both(system, shared_cols, monkeypatch, **kw):
    """The deflated solve through the dense path and through the shared columns' solve (the dense limit at 0)."""
    x_dense, rec_dense = _solve(system, deflate=sc.Deflate(3, basis='polynomial'), iter_lim=600, **kw)
    monkeypatch.setattr(D, '_SCHUR_MAX', 0)
    x_iter, rec_iter = _solve(system, deflate=sc.Deflate(3, basis='polynomial'), iter_lim=600, shared=shared_cols, **kw)
    return (x_dense, json.loads(rec_dense.deflation)), (x_iter, json.loads(rec_iter.deflation))


def _fixed_patterns(shared):
    """The patterns of the shared columns that the frames' chunk constants and slopes can also make: a constant and a
    slope within each group of 4 columns (a frame's chunk), orthonormal."""
    V = np.zeros((shared, 2 * (shared // 4)))
    for g in range(shared // 4):
        V[4 * g:4 * g + 4, 2 * g] = 1.0
        V[4 * g:4 * g + 4, 2 * g + 1] = (np.arange(4) - 1.5) / 1.5
    return np.linalg.qr(V)[0]


def test_the_shared_columns_solve_fits_as_the_dense_path_and_keeps_an_unconstrained_gauge_apart(monkeypatch):
    """The toy's shared columns line up with the frames' chunks, so a constant and a slope within each chunk's 4 columns
    are directions the data do not see (the chunks' own constants and slopes make them). The dense path takes them out
    of the response (the response of minimum norm); the iterative one solves the same normal equations but cannot find
    them, so the solution fits the data the same and sets those gauges differently. A model removes them with a column
    constraint (``Offsets(exclude=Overlap(...))``: the next test)."""
    A, b, npix, grid, shared = _shared_system(frames=200)
    system = (A, b, npix, grid)
    cols = np.arange(A.shape[1] - shared, A.shape[1])
    (x_dense, d_dense), (x_iter, d_iter) = _both(system, cols, monkeypatch)
    assert 'shared_columns' not in d_dense
    s = d_iter['shared_columns']
    assert s['columns'] == shared and s['converged'] and s['relative_residuals'][-1] <= D._SHARED_RTOL
    assert d_iter['pairing_residual'] < 1e-10
    assert np.linalg.norm(A @ (x_iter - x_dense)) < 1e-9 * np.linalg.norm(A @ x_dense)
    assert _rel(x_dense, _converged(system)) < 1e-7


def test_with_the_gauges_excluded_both_paths_give_the_same_deflation_and_solution(monkeypatch):
    """The shared columns without the patterns the frames' chunks can make (a column constraint within them): the
    response and the solution honour it, and the two paths give the same deflation space and solution."""
    A, b, npix, grid, shared = _shared_system(frames=200)
    system = (A, b, npix, grid)
    cols = np.arange(A.shape[1] - shared, A.shape[1])
    P = _fixed_patterns(shared)
    excl = [(cols[None, :], P)]
    (x_dense, d_dense), (x_iter, d_iter) = _both(system, cols, monkeypatch, exclusions=excl)
    assert d_iter['shared_columns']['converged'] and d_iter['pairing_residual'] < 1e-10
    assert np.abs(P.T @ x_iter[cols]).max() < 1e-9 * np.abs(x_iter[cols]).max()
    assert np.allclose(d_iter['coefficients'], d_dense['coefficients'], rtol=1e-7, atol=1e-10)
    assert np.allclose(d_iter['eigenvalues'], d_dense['eigenvalues'], rtol=1e-8, atol=1e-12 * max(d_dense['eigenvalues']))
    assert _rel(x_iter, x_dense) < 1e-7


def test_few_shared_columns_keep_the_dense_path():
    """With no more shared columns than the dense limit, passing them changes nothing: the same bytes."""
    A, b, npix, grid, shared = _shared_system(frames=100)
    system = (A, b, npix, grid)
    cols = np.arange(A.shape[1] - shared, A.shape[1])
    x0, r0 = _solve(system, deflate=sc.Deflate(3, basis='polynomial'), iter_lim=300)
    x1, r1 = _solve(system, deflate=sc.Deflate(3, basis='polynomial'), iter_lim=300, shared=cols)
    assert np.array_equal(x0, x1)
    assert 'shared_columns' not in json.loads(r1.deflation)


def test_a_singular_normal_matrix_of_the_shared_columns_leaves_its_unseen_patterns_out(monkeypatch):
    """A shared column observed exactly as another (a copy): their difference is a pattern no observation sees, so the
    shared columns' normal matrix is singular and its LU fails. The iterative solve leaves that pattern out of the
    response (its minimum-norm choice there) and factorises the rest; the response is otherwise the dense path's, and
    the solution fits the data as the dense path's does. (The two differ along that pattern, which no observation sees:
    the deflation's coefficients are not compared.)"""
    A, b, npix, grid, shared = _shared_system(frames=200)
    n = A.shape[1]
    A = sp.hstack([A, A[:, [n - shared]]], format='csr')            # a copy of the first shared column
    system = (A, b, npix, grid)
    cols = np.arange(n - shared, n + 1)
    P = np.vstack([_fixed_patterns(shared), np.zeros((1, 2 * (shared // 4)))])
    (x_dense, d_dense), (x_iter, d_iter) = _both(system, cols, monkeypatch, exclusions=[(cols[None, :], P)])
    s = d_iter['shared_columns']
    assert s['unseen_patterns'] == 1 and s['converged']
    assert d_iter['pairing_residual'] < 1e-10
    assert np.linalg.norm(A @ (x_iter - x_dense)) < 1e-7 * np.linalg.norm(A @ x_dense)
