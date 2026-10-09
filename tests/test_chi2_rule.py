"""The chi-squared stop rule (``sc.Stop(chi2=sigma)``, :class:`selfcal.run.recipe.ChiSquaredRule`): the
scale-free convergence criterion, on the true residual, with the noise variance from the residual or given.
"""
import json
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
from selfcal.core.monitor import RULE_ISTOP  # noqa: E402
from selfcal.run.plan import describe_stop  # noqa: E402
from tests.test_cgls import CHUNK, _converged, _rel, _solve, scratch  # noqa: E402,F401

pytestmark = pytest.mark.filterwarnings('ignore:overflow encountered in cast:RuntimeWarning')


def test_the_rule_stops_a_converged_solve_and_records_its_gain():
    A, b, npix, grid = CHUNK
    x_star = _converged(CHUNK)
    x, rec = _solve(CHUNK, iter_lim=2000, stop=sc.Stop(chi2=sc.ChiSquaredRule(0.1, window=20, every=10)))
    assert rec.istop == RULE_ISTOP and rec.stop_rule == 'chi2' and rec.iterations < 2000
    st = json.loads(rec.stop_values)['rules']['chi2']
    assert st['holds'] and st['value'] < 0.01 and st['window'] == 20 and st['every'] == 10
    m, n = rec.rows, rec.columns
    assert st['rows'] == m and st['unknowns'] == n
    assert abs(st['noise_variance'] - st['to'] ** 2 / (m - n)) < 1e-12 * st['noise_variance']
    # the solution is converged: the chi-squared still to gain is below the rule's threshold
    r_end = np.linalg.norm(b - A @ x)
    r_star = np.linalg.norm(b - A @ x_star)
    assert (r_end ** 2 - r_star ** 2) / st['noise_variance'] < 0.01
    # the arrays and the words
    arr = rec.watch.arrays()
    assert 'stop_chi2' in arr and np.isfinite(arr['stop_chi2'][rec.iterations])
    assert 'the chi-squared rule holds' in rec.watch.describe_stop(rec)


def test_a_given_noise_level():
    """With a noise variance given (a noise-free simulation), the gain is in those units: a tiny noise makes
    every fall of the residual count, so the rule holds only once the residual is flat to the arithmetic's
    floor, later than with the noise estimated from the residual."""
    x1, rec1 = _solve(CHUNK, iter_lim=2000, stop=sc.Stop(chi2=sc.ChiSquaredRule(0.1, window=20, every=10)))
    x2, rec2 = _solve(CHUNK, iter_lim=2000, stop=sc.Stop(chi2=sc.ChiSquaredRule(0.1, window=20, every=10, noise=1e-30)))
    assert rec2.istop == RULE_ISTOP and rec2.iterations > rec1.iterations
    st = json.loads(rec2.stop_values)['rules']['chi2']
    assert st['noise'] == 'given' and st['holds'] and st['value'] <= 0.01 and st['noise_variance'] == 1e-30


def test_the_settings_and_the_plan():
    s = sc.Stop(chi2=0.1)
    assert s.rules == ('chi2',) and s.chi2 == sc.ChiSquaredRule(0.1) and s.first_iteration('chi2') == 50
    assert sc.Stop(chi2=sc.ChiSquaredRule(0.3, window=30, every=15)).chi2.window == 30
    with pytest.raises(ConfigError):
        sc.ChiSquaredRule(0.0)
    with pytest.raises(ConfigError):
        sc.ChiSquaredRule(0.1, window=25, every=10)
    with pytest.raises(ConfigError):
        sc.ChiSquaredRule(0.1, noise=0.0)
    assert 'chi-squared gained over the last 50 iterations below 0.1^2' in describe_stop(s)
    assert s.unreachable(40) and not s.unreachable(60)
    assert 'chi2' not in sc.Stop(gradient=1e-3).to_dict()              # the added field keeps old encodings
    assert sc.Stop(chi2=0.1, gradient=1e-3, combine='any').rules == ('gradient', 'chi2')
