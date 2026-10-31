"""CGLS's gradient by recurrence (``sc.Numerics(gradient='recurrence')``): the gradient of each iteration carried
along from the transpose product of the matvec's own result, which the observation table forms in the same pass as
the matvec (``normal_product``), instead of a transpose product of the residual. On the synthetic frames of
``test_observation_table``: the table's fused product is the matvec, and its transpose product is the float64
transpose product of the stored matvec bit for bit; the recurrence solve on the table and on the stored matrix (which
forms the two products in turn) agree within rounding, as do the recurrence and the recomputed gradient, plain and
with the deflation, up to and at convergence; the setting is CGLS's.
"""
import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip('numba')

import selfcal as sc  # noqa: E402
from selfcal.config.base import ConfigError  # noqa: E402
from selfcal.core.solve import _make_table_operator, apply_lsqr  # noqa: E402
from selfcal.run.lower import solver_options  # noqa: E402
from tests.test_observation_table import GRID, _bits, _both, _frames, _same_solve  # noqa: E402


@pytest.mark.parametrize('threads', [1, 3])
def test_the_fused_product_is_the_two_products(tmp_path, threads):
    """``normal_product(p)``: the matvec's output, the same bytes as ``matvec`` gives, and the float64 transpose
    product of that output, the same bytes as ``rmatvec_promoted`` gives, at one thread (sequential) and several
    (the row-split buffers)."""
    _stored, compact = _both(_frames(tmp_path))
    op = _make_table_operator(compact.A, threads)
    rng = np.random.default_rng(1)
    p = rng.standard_normal(op.shape[1]).astype(np.float32)
    q, t = op.normal_product(p)
    q_ref = np.array(op.matvec(p), copy=True)
    np.testing.assert_array_equal(_bits(np.asarray(q)), _bits(q_ref))
    t = np.asarray(t)
    assert t.dtype == np.float64
    np.testing.assert_array_equal(_bits(t), _bits(np.asarray(op.rmatvec_promoted(q_ref))))
    np.testing.assert_allclose(t, np.asarray(op.rmatvec(q_ref.astype(np.float64))), rtol=1e-12, atol=0)


@pytest.mark.parametrize('threads', [1, 3])
def test_recurrence_on_the_table_agrees_with_the_recurrence_on_the_matrix(tmp_path, threads):
    """The recurrence solve through the table and through the stored matrix (its two products in turn, the
    transpose one in float64 by the sequential scatter) agree within rounding: the float64 sums differ in their
    order only."""
    stored, compact = _both(_frames(tmp_path))
    kw = dict(ref_shape=GRID, iter_lim=20, atol=0, btol=0, conlim=0, solver='cgls', use_float32=True,
              n_threads=threads, return_record=True, scratch_dir=str(tmp_path), gradient='recurrence')
    x_s, rec_s = apply_lsqr(stored.A, stored.b.copy(), active_mask=stored.active_mask,
                            num_cols_full=None if stored.active_mask is None else stored.active_mask.size, **kw)
    x_c, rec_c = apply_lsqr(compact.A, compact.b.copy(), active_mask=compact.active_mask,
                            num_cols_full=None if compact.active_mask is None else compact.active_mask.size, **kw)
    assert rec_s.iterations == rec_c.iterations
    np.testing.assert_allclose(x_c, x_s, rtol=0, atol=1e-6 * np.max(np.abs(x_s)))
    np.testing.assert_allclose(rec_c.true_residual, rec_s.true_residual, rtol=1e-5)


def test_recurrence_agrees_with_the_recomputed_gradient_within_rounding(tmp_path):
    """The two forms give the same solution within rounding up to and at convergence: 30 iterations plain, 20 with
    the deflation (which converges this toy in about ten; far past convergence, with every stopping test off as
    here, the recurrence's float64 gradient lets the iterate move along the system's near-null directions where
    the recomputed gradient has stalled on its float32 floor: a solve is ended by its stop rule or tests)."""
    paths = _frames(tmp_path)
    kw = dict(ref_shape=GRID, atol=0, btol=0, conlim=0, solver='cgls', use_float32=True, n_threads=3,
              return_record=True, scratch_dir=str(tmp_path))

    def solve(**extra):               # a fresh table each time: apply_lsqr scales the one it is given in place
        _stored, compact = _both(paths)
        return apply_lsqr(compact.A, compact.b.copy(), active_mask=compact.active_mask,
                          num_cols_full=None if compact.active_mask is None else compact.active_mask.size,
                          **kw, **extra)

    for extra in ({'iter_lim': 30}, {'iter_lim': 20, 'deflate': sc.Deflate(), 'sky_names': ['sky']}):
        x_two, rec_two = solve(**extra)
        x_one, rec_one = solve(gradient='recurrence', **extra)
        assert rec_one.iterations == rec_two.iterations
        scale = np.max(np.abs(x_two))
        np.testing.assert_allclose(x_one, x_two, rtol=0, atol=1e-5 * scale)
        np.testing.assert_allclose(rec_one.true_residual, rec_two.true_residual, rtol=1e-4)


def test_the_setting():
    """``Numerics(gradient=)`` is checked, reaches the solve only when it is not the default (every existing
    product's settings are unchanged) and is CGLS's."""
    assert sc.Numerics().gradient == 'recomputed'
    with pytest.raises(ConfigError):
        sc.Numerics(gradient='fused')
    recipe = sc.Recipe(sc.continuum(), fit=sc.Fit(method='cgls'), numerics=sc.Numerics(2, gradient='recurrence'))
    assert solver_options(recipe)['gradient'] == 'recurrence'
    assert 'gradient' not in solver_options(sc.Recipe(sc.continuum(), fit=sc.Fit(method='cgls')))
    with pytest.raises(ConfigError, match='CGLS'):
        solver_options(sc.Recipe(sc.continuum(), fit=sc.Fit(method='lsqr'), numerics=sc.Numerics(2, gradient='recurrence')))
    with pytest.raises(ValueError, match='CGLS'):
        apply_lsqr(sp.csr_matrix(np.eye(2, dtype=np.float32)), np.ones(2, np.float32), ref_shape=(1, 2), solver='lsqr',
                   gradient='recurrence')
