"""The solver's starting vector (``sc.Fit(seed=)``): ``'separate'`` (the default) starts every non-sky column at its
own least-squares fit to the data (:func:`selfcal.core.solution.compute_x0_from_Ab`), ``'offsets'`` at its frame's
joint least-squares fit (:func:`selfcal.core.solution.compute_x0_offsets_fit`); the sky at zero in both."""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import selfcal as sc  # noqa: E402
from selfcal.config import ConfigError  # noqa: E402
from selfcal.core.solution import compute_x0_from_Ab, compute_x0_offsets_fit  # noqa: E402
from tests.test_deflation import _chunk_system  # noqa: E402

CHUNK = _chunk_system()


def test_the_offsets_seed_is_each_frames_joint_least_squares_fit():
    A, b, npix, grid = CHUNK
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    Ac = A[:, active].tocsr()
    x0 = compute_x0_offsets_fit(Ac, b, grid, active_mask=active)
    assert x0.shape == (A.shape[1],) and np.all(x0[:npix] == 0) and np.any(x0[npix:] != 0)
    # the residual against the offset columns alone is orthogonal to them: the joint fit per frame
    A_off = A[:, npix:]
    r = b - A_off @ x0[npix:]
    g = A_off.T @ r
    assert np.abs(g).max() < 1e-9 * np.abs(A_off.T @ b).max()
    # the separate seed fits each column alone: its residual is orthogonal to each column on its own, not
    # to the columns together
    xs = compute_x0_from_Ab(Ac, b, grid, active_mask=active)
    assert np.all(xs[:npix] == 0)
    cols = np.flatnonzero(active[npix:]) + npix
    rs = b[:, None] - A[:, cols].multiply(xs[cols][None, :]).toarray()
    assert np.allclose(np.asarray(A[:, cols].multiply(rs).sum(axis=0)).ravel(), 0.0, atol=1e-9 * np.abs(b).max())
    assert np.abs(A_off.T @ (b - A_off @ xs[npix:])).max() > 1e-6 * np.abs(A_off.T @ b).max()
    with pytest.raises(TypeError, match='COO'):
        compute_x0_offsets_fit(A.tocoo(), b, grid)


def test_the_setting():
    assert sc.Fit().seed == 'separate' and 'seed' not in sc.Fit().to_dict()
    assert sc.Fit(seed='offsets').to_dict()['seed'] == 'offsets'
    assert sc.Fit(seed='zero').to_dict()['seed'] == 'zero'
    with pytest.raises(ConfigError):
        sc.Fit(seed='median')
