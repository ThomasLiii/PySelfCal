"""The chi-squared of a solution per sky pixel, made at the end of the solve (``apply_lsqr(chi_squared=)``,
``Fit(chi_squared=)``): the sum of the squared residuals ``(b - A x)^2`` over each pixel's observations.

The systems are those of test_min_observations (synthetic frames, no instrument): the map is compared with the same sum
made from the matrix's own indices.
"""
import numpy as np
import pytest
import scipy.sparse as sp

from selfcal.core.solve import apply_lsqr
from test_min_observations import GRID, N_SKY, _frames, _full_width, _setup


def _expected(A, b, x, data_rows):
    """Each data row's squared residual added to the pixel of its entry in the first sky term (rows without one, the
    coverage floor's, add nothing)."""
    res = (np.asarray(b, np.float64) - A @ np.asarray(x, np.float64))[:data_rows]
    out = np.zeros(N_SKY)
    for i in range(data_rows):
        row = A[i]
        sky = row.indices[(row.indices < N_SKY) & (row.data != 0)]
        if sky.size:
            out[sky[0]] += res[i] ** 2
    return out


def _solve(result, tmp_path, b=None, A=None, data_rows=None, **kw):
    A = result.A if A is None else A
    b = np.asarray(result.b if b is None else b, dtype=np.float64).copy()
    active = result.active_mask
    # 8 iterations: the system has an exact freedom (the sky's level against the offsets'), along which a longer
    # solve in float32 products drifts until the products round away the residual
    opts = dict(ref_shape=GRID, atol=0.0, btol=0.0, conlim=0.0, damp=0.0, solver='cgls', n_threads=1, iter_lim=8,
                active_mask=active, num_cols_full=None if active is None else len(active), scratch_dir=str(tmp_path),
                return_record=True, chi_squared=A.shape[0] if data_rows is None else data_rows)
    return apply_lsqr(A, b, **{**opts, **kw})


@pytest.mark.parametrize('storage', ['unified', 'partitioned'])
@pytest.mark.parametrize('floor', [None, 2])
@pytest.mark.parametrize('solver', ['cgls', 'lsqr', 'lsmr'])
def test_the_map_is_the_squared_residuals_summed_per_pixel(tmp_path, monkeypatch, storage, floor, solver):
    """Every storage of the build, with and without a coverage floor (its rows are zero: they add nothing), each
    solver, with the column scaling on: the map is the per-pixel sum, and its total the solve's own ``|b - A x|^2``."""
    extra = {}
    if storage == 'partitioned':
        for k, v in (('SELFCAL_SCATTER_WORKERS', '2'), ('SELFCAL_BLOCK_NNZ', '1'), ('SELFCAL_RMATVEC_SPLIT', '2')):
            monkeypatch.setenv(k, v)
        extra = {'batch_spill_dir': str(tmp_path / 'spill')}
    result = _setup(_frames(tmp_path), outlier_thresh=None, min_observations=floor, **extra)
    A = _full_width(result).astype(np.float64)                    # a copy: the solve scales the matrix in place
    b = np.asarray(result.b, np.float64).copy()
    x, record, chi2 = _solve(result, tmp_path, solver=solver, n_threads=1 if storage == 'unified' else 2)
    assert chi2.shape == (N_SKY,) and chi2.dtype == np.float32
    expect = _expected(A, b, x, A.shape[0])
    assert expect.sum() > 0
    np.testing.assert_allclose(chi2, expect, rtol=2e-4, atol=1e-9 * expect.max())
    assert chi2[np.asarray(result.pixel_counts[:N_SKY]) == 0].sum() == 0
    np.testing.assert_allclose(chi2.sum(dtype=np.float64), record.true_residual ** 2, rtol=2e-4)


def test_constraint_rows_add_nothing(tmp_path):
    """Rows after the observations (here two rows that hold sky pixels and an offset near zero) are left out of the
    sums, although they have entries in the first sky term's columns."""
    result = _setup(_frames(tmp_path), outlier_thresh=None)
    assert result.active_mask is None or np.asarray(result.active_mask).all()
    A = sp.csr_matrix(result.A).astype(np.float64)
    m, n = A.shape
    tail = sp.csr_matrix((np.array([0.3, 0.2, 0.5]), (np.array([0, 1, 1]), np.array([5, 17, n - 1]))), shape=(2, n))
    A2 = sp.vstack([A, tail], format='csr')
    b2 = np.concatenate([np.asarray(result.b, np.float64), [0.4, -0.1]])
    x, record, chi2 = _solve(result, tmp_path, A=A2.astype(np.float32), b=b2, data_rows=m)
    np.testing.assert_allclose(chi2, _expected(A2, b2, x, m), rtol=2e-4, atol=1e-9)
    tail_res = (b2 - A2 @ x)[m:]
    np.testing.assert_allclose(chi2.sum(dtype=np.float64) + (tail_res ** 2).sum(), record.true_residual ** 2, rtol=2e-4)


def test_without_the_request_the_return_is_unchanged(tmp_path):
    result = _setup(_frames(tmp_path), outlier_thresh=None)
    out = _solve(result, tmp_path, chi_squared=None)
    assert len(out) == 2


# ---------------------------------------------------------------------------------------------- through the API
def _toy_recipe(name, **fit):
    import selfcal as sc
    from test_any_telescope import COADD, FIT, NUMERICS
    return sc.Recipe(sc.Model(sky=[sc.Sky(damping=0.0)], offsets=[sc.Offsets(damping=0.0)]),
                     fit=FIT.replace(iterations=40, clip=None, **fit), coadd=COADD, numerics=NUMERICS, name=name)


@pytest.mark.parametrize('matrix', ['stored', 'compact'])
def test_a_calibration_writes_the_map_by_default(tmp_path, matrix):
    """``Fit(chi_squared=)`` is True by default: the cal holds ``sky_chi2`` on the sky map's grid, zero where the sky
    has no coverage, its total the solve's recorded ``|b - A x|^2`` (no constraint row in this model); both matrix
    storages. With ``chi_squared=False`` the cal has no such dataset and everything else in it is the same."""
    import h5py
    import hdf5plugin  # noqa: F401

    import selfcal as sc
    from test_min_observations import _toy
    field = _toy(tmp_path)
    compute = field.compute.replace(matrix=matrix) if matrix != 'stored' else None
    on = field.calibrate(_toy_recipe('chi2_on'), jobs=sc.Job('All'), compute=compute)
    off = field.calibrate(_toy_recipe('chi2_off', chi_squared=False), jobs=sc.Job('All'), compute=compute)
    with h5py.File(on.cal_paths[0]) as f, h5py.File(off.cal_paths[0]) as g:
        chi2 = f['sky_chi2'][:]
        cov = f['sky_coverage/continuum'][:]
        assert chi2.shape == f['sky/continuum'].shape and chi2.dtype == np.float32
        assert (chi2[cov == 0] == 0).all() and (chi2[cov > 0] > 0).mean() > 0.99
        np.testing.assert_allclose(chi2.sum(dtype=np.float64), float(f['solve'].attrs['true_residual']) ** 2, rtol=1e-4)
        assert 'sky_chi2' not in g
        np.testing.assert_array_equal(f['sky/continuum'][:], g['sky/continuum'][:])
        np.testing.assert_array_equal(f['offsets/map_0'][:], g['offsets/map_0'][:])


def test_the_setting_is_checked_and_keeps_the_default_fingerprint():
    import selfcal as sc
    from selfcal.config.base import ConfigError
    assert sc.Fit().chi_squared is True
    assert 'chi_squared' not in sc.Fit().to_dict() == sc.Fit(chi_squared=True).to_dict()    # existing fingerprints kept
    assert sc.Fit(chi_squared=False).to_dict()['chi_squared'] is False
    with pytest.raises(ConfigError):
        sc.Fit(chi_squared=1)


# ---------------------------------------------------------------------------------------------- the SKY pass
@pytest.mark.parametrize('J', [2, 4])
def test_a_sky_pass_has_the_map_from_its_sums(tmp_path, J):
    """A SKY pass keeps no matrix: its chi-squared per pixel comes from its per-pixel sums, the squared right-hand side
    among them (``pixel_rhs``, after its blocks). It is the per-pixel sum of the squared residuals of the same rows,
    here built as a matrix too (whose damping rows, after the observations, add nothing)."""
    from selfcal.npass.sky import pixel_chi_squared, solve_pixels
    from test_npass_primitives import REF, _make_frames
    from test_npass_primitives import _setup as sky_setup
    paths, bc_det, _, sky_model = _make_frames(str(tmp_path), np.random.default_rng(11 + J), J)
    r = sky_setup(paths, sky_model, bc_det)
    num_sky = REF[0] * REF[1]
    assert r.pixel_rhs.shape == ((J + 1) * num_sky,)
    x = solve_pixels(r.pixel_fisher, r.pixel_cross, r.pixel_rhs, r.pixel_counts, num_sky, J)
    chi2 = pixel_chi_squared(x, r.pixel_fisher, r.pixel_cross, r.pixel_rhs, r.pixel_counts, num_sky, J)
    A, b = _full_width(r).astype(np.float64), np.asarray(r.b, np.float64)
    assert r.data_rows < A.shape[0]                                 # the damping rows follow the observations
    res = (b - A @ x)[:r.data_rows]
    expect = np.zeros(num_sky)
    for i in range(r.data_rows):
        row = A[i]
        expect[row.indices[row.indices < num_sky][0]] += res[i] ** 2
    assert chi2.dtype == np.float32 and expect.sum() > 0
    np.testing.assert_allclose(chi2, expect, rtol=1e-5, atol=1e-7 * expect.max())
    assert (chi2[np.asarray(r.pixel_counts[:num_sky]) == 0] == 0).all()
    # sums made before the right-hand side's squares were kept: no map
    assert pixel_chi_squared(x, r.pixel_fisher, r.pixel_cross, r.pixel_rhs[:J * num_sky], r.pixel_counts, num_sky,
                             J) is None
