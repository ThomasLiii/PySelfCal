"""The solver's starting vector (``sc.Fit(seed=)``): ``'offsets'`` starts every non-sky column at its frame's joint
least-squares fit to the data (:func:`selfcal.core.solution.compute_x0_offsets_fit`), the sky at zero."""
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
from selfcal.core.solution import compute_x0_offsets_fit, compute_x0_scalar_only  # noqa: E402
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
    # the scalar-only seed leaves the chunk constants at zero; the offsets seed does not
    x_sc = compute_x0_scalar_only(Ac, b, grid, scalar_col_start=A.shape[1] - 60, active_mask=active)
    assert np.all(x_sc[npix:A.shape[1] - 60] == 0) and np.any(x0[npix:A.shape[1] - 60] != 0)
    with pytest.raises(TypeError, match='COO'):
        compute_x0_offsets_fit(A.tocoo(), b, grid)


def test_the_setting():
    assert sc.Fit().seed == 'scalar' and 'seed' not in sc.Fit().to_dict()
    assert sc.Fit(seed='offsets').to_dict()['seed'] == 'offsets'
    assert sc.Fit(seed='zero').to_dict()['seed'] == 'zero'
    with pytest.raises(ConfigError):
        sc.Fit(seed='median')
