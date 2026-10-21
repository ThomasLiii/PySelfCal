"""``Fit(min_observations=N)``: the sky pixels observed fewer than N times are dropped from every frame's rows
(``setup_lsqr(min_observations=)``), counted by the row builders themselves.

Synthetic frames written with :func:`selfcal.io.frames.write_frame` (no instrument): frame k sits on grid rows
k..k+H, so the first and last grid rows are observed once and the coverage grows inward; one non-finite value and one
outlier leave single observations inside the field too (the outlier only when the solve clips it, or when the
model's weight is not finite there).
"""
import numpy as np
import pytest
import scipy.sparse as sp

from selfcal import _state
from selfcal.core.system import setup_lsqr
from selfcal.io.frames import standard_frame_path, write_frame
from selfcal.models.sky_model import Coefficient

H, W, F = 8, 8, 6
GRID = (H + F - 1, W)
N_SKY = GRID[0] * GRID[1]


def _frames(tmp_path):
    """F frames, frame k on the grid rows k..k+H, a smooth value plus a per-frame offset; frame 1 has a non-finite
    value at grid (1, 2) and an outlier at grid (1, 5), both on a row frames 0 and 1 alone observe."""
    _state.set_progress(False)
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    paths = []
    for k in range(F):
        vals = (1.0 + 0.05 * k + 0.01 * (y + k) + 0.002 * x + 1e-3 * rng.standard_normal((H, W))).astype(np.float32)
        if k == 1:
            vals[0, 2] = np.nan
            vals[0, 5] = 1e3
        paths.append(write_frame(standard_frame_path(str(tmp_path), k, 0), vals, [k, k + H, 0, W], np.stack([x, y])))
    return paths


def nan_above_100(value):
    """A row weight of 1, not finite where the value is above 100 (the outlier): the solve drops that row."""
    return np.where(value > 100, np.nan, 1.0)


def _setup(paths, **kw):
    common = dict(chunk_maps=[np.zeros((H, W), np.int32)], grid_valid_weight=np.ones((H, W), np.float32),
                  apply_mask=True, ignore_list=[], max_workers=1, batch_size=4)
    return setup_lsqr(paths, GRID, **{**common, **kw})


def _sky_counts(result):
    return np.asarray(result.pixel_counts[:N_SKY]).reshape(GRID)


def _full_width(result):
    """A with its columns at their uncompacted positions."""
    A = sp.csr_matrix(result.A)
    if result.active_mask is None:
        return A
    cols = np.flatnonzero(np.asarray(result.active_mask))
    return sp.csr_matrix((A.data, cols[A.indices], A.indptr), shape=(A.shape[0], len(result.active_mask)))


@pytest.mark.parametrize('clip', [None, 3])
def test_the_floor_removes_the_pixels_the_solve_observes_once(tmp_path, clip):
    """The pixels the unfloored solve covers once (after the mask and, with a clip, the clip) are gone; every other
    pixel keeps its coverage: the count is the solve's own sky coverage."""
    paths = _frames(tmp_path)
    plain = _sky_counts(_setup(paths, outlier_thresh=clip))
    floored = _sky_counts(_setup(paths, outlier_thresh=clip, min_observations=2))
    once = plain == 1
    assert once[0].all() and once[-1].all() and once[1, 2]          # the edge rows and the non-finite value
    assert once[1, 5] == (clip is not None)                          # the outlier: once only after the clip
    assert floored[once].sum() == 0
    np.testing.assert_array_equal(floored[~once], plain[~once])
    floored3 = _sky_counts(_setup(paths, outlier_thresh=clip, min_observations=3))
    np.testing.assert_array_equal(floored3[plain >= 3], plain[plain >= 3])
    assert floored3[(plain > 0) & (plain < 3)].sum() == 0


def test_a_row_the_weights_remove_does_not_count(tmp_path):
    """An observation whose model weight is not finite has no row in the solve, so it does not count: with no clip,
    the outlier's pixel is observed once by the solve and the floor drops it."""
    paths = _frames(tmp_path)
    weight = Coefficient('value', nan_above_100)
    plain = _sky_counts(_setup(paths, outlier_thresh=None, weight_function=weight))
    floored = _sky_counts(_setup(paths, outlier_thresh=None, weight_function=weight, min_observations=2))
    assert plain[1, 5] == 1 and _sky_counts(_setup(paths, outlier_thresh=None))[1, 5] == 2
    once = plain == 1
    assert floored[once].sum() == 0
    np.testing.assert_array_equal(floored[~once], plain[~once])


def test_the_rows_are_the_unfloored_rows_without_the_dropped_pixels(tmp_path):
    """The floor removes exactly the rows of the dropped pixels: the floored system is the unfloored one with those
    rows deleted, byte for byte, in the same order."""
    paths = _frames(tmp_path)
    plain = _setup(paths, outlier_thresh=None)
    floored = _setup(paths, outlier_thresh=None, min_observations=2)
    A, Af = _full_width(plain), _full_width(floored)
    sky_col = np.array([row.indices[row.indices < N_SKY][0] for row in A])          # one sky entry per data row
    keep = np.asarray(plain.pixel_counts)[sky_col] >= 2
    assert (~keep).sum() > 0
    diff = (A[keep] - Af)
    assert diff.nnz == 0 or np.abs(diff.data).max() == 0
    np.testing.assert_array_equal(A[keep].indices, Af.indices)
    np.testing.assert_array_equal(A[keep].data, Af.data)
    np.testing.assert_array_equal(np.asarray(plain.b)[keep], np.asarray(floored.b))


def test_a_pixel_observed_once_informs_nothing_else(tmp_path):
    """In the unconstrained least-squares fit a single-observation pixel fits its one row exactly, and the fit of
    every other row is the floored system's: dropping those pixels changes nothing else."""
    paths = _frames(tmp_path)
    plain = _setup(paths, outlier_thresh=None)
    floored = _setup(paths, outlier_thresh=None, min_observations=2)
    A, Af = _full_width(plain).toarray(), _full_width(floored).toarray()
    b, bf = np.asarray(plain.b, float), np.asarray(floored.b, float)
    sky_col = np.array([np.flatnonzero(A[i, :N_SKY])[0] for i in range(A.shape[0])])
    keep = np.asarray(plain.pixel_counts)[sky_col] >= 2
    fit = A @ np.linalg.lstsq(A, b, rcond=None)[0]
    fit_f = Af @ np.linalg.lstsq(Af, bf, rcond=None)[0]
    np.testing.assert_allclose(fit[~keep], b[~keep], atol=1e-9)          # fitted exactly by its own pixel
    np.testing.assert_allclose(fit[keep], fit_f, atol=1e-9)


def test_the_setting_lowers_to_the_setup_keyword_and_validates():
    import selfcal as sc
    from selfcal.config.base import ConfigError
    from selfcal.run.lower import setup_options

    class _Inst:
        def default_ignore_flags(self):
            return []

    compute = sc.Compute(workers=2)
    assert setup_options(sc.Recipe(sc.continuum(), fit=sc.Fit(min_observations=2)), compute,
                         _Inst())['min_observations'] == 2
    assert 'min_observations' not in setup_options(sc.Recipe(sc.continuum()), compute, _Inst())
    for bad in (1, 0, 2.0, True, '2'):
        with pytest.raises(ConfigError):
            sc.Fit(min_observations=bad)
    with pytest.raises(ValueError):
        setup_lsqr([], GRID, min_observations=1)
