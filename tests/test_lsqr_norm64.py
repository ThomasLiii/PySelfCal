"""The float64-accumulated norm in ``lsqr_inplace``: exact where BLAS ``sdot`` in float32 is not."""
import numpy as np

from selfcal.core import lsqr_inplace as li


def _f64_norm(x):
    return float(np.sqrt(np.dot(x.astype(np.float64), x.astype(np.float64))))


def test_long_float32_vector_of_small_values_is_accumulated_exactly():
    rng = np.random.default_rng(0)
    # 2e8 entries of ~1e-5: the float32 running sum (~2e-2) outgrows its increments (~1e-10)
    x = (rng.standard_normal(200_000_000, dtype=np.float32) * np.float32(1e-5))
    exact = _f64_norm(x)
    ours = float(li._norm(x))
    blas = float(np.linalg.norm(x))
    assert abs(ours - exact) <= 1e-6 * exact
    assert abs(blas - exact) > 1e-3 * exact, "the BLAS float32 norm should be visibly biased on this vector"


def test_short_and_float64_vectors_take_scipys_call_bit_for_bit():
    rng = np.random.default_rng(1)
    short = rng.standard_normal(10_000, dtype=np.float32)
    assert li._norm(short) == np.linalg.norm(short)
    assert type(li._norm(short)) is type(np.linalg.norm(short))
    wide = rng.standard_normal(3_000_000)
    assert li._norm(wide) == np.linalg.norm(wide)


def test_result_scalar_type_matches_scipy_for_float32():
    x = np.ones(2_000_000, dtype=np.float32)
    assert isinstance(li._norm(x), np.float32)
    assert float(li._norm(x)) == float(np.sqrt(np.float64(2_000_000)).astype(np.float32))
