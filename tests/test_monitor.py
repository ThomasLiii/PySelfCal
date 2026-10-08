"""Convergence monitors and opt-in stop rules (selfcal.core.monitor; ``sc.Monitor``, ``sc.Fit(stop=sc.Stop(...))``).

Monitors check a solve every ``m`` iterations (the true residual and gradient against the solver's
estimates, the large-scale fit of each sky term) and change nothing; the stop rules (residual plateau,
gradient, large-scale stability, the solver's own tests; ``all`` / ``any``; ``min_iterations``) end a
solve where they should and not elsewhere, and the stop is recorded in the cal, the action's record and
the log. The failure mode they exist for is reproduced at small scale: on a system with a near-null
smooth sky mode the residual and gradient rules stop while the field-wide gradient of the sky is far
from its least-squares value; the large-scale rule waits until it has converged. Through the API:
monitors write the same cal, are recorded and replayed but never fingerprinted, and continue across a
warm start; ``Fit(stop=...)`` enters the fingerprint only when given.
"""
import json
import logging
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import selfcal as sc  # noqa: E402
from selfcal import _state  # noqa: E402
from selfcal.config import ConfigError  # noqa: E402
from selfcal.core.monitor import RULE_ISTOP, SmoothFit, StopRules, Watch, monomials  # noqa: E402
from selfcal.core.snapshots import ActiveColumns, Iterate  # noqa: E402
from selfcal.core.solve import apply_lsqr  # noqa: E402
from selfcal.core.solve_record import STOP_REASONS, SolveHistory  # noqa: E402
from selfcal.io.calfile import CalFile  # noqa: E402
from selfcal.run import products  # noqa: E402
from tests.synthetic_exposures import DET, N_CHUNK_SIDE, REF_ARCSEC, write_exposures  # noqa: E402

# scipy's LSMR (and so ours) overflows a float32 scalar once a float32 system has converged
pytestmark = pytest.mark.filterwarnings('ignore:overflow encountered in cast:RuntimeWarning')


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# =================================================================== a system with a near-null smooth mode
H, W = 16, 24


def _near_null_system(weak=0.005, noise=0.05, seed=0, hole=3):
    """A sky of H x W pixels (a field-wide gradient and curvature plus small-scale structure) seen by
    frames that each cover most of ONE pixel column, with an offset per frame: any sky that varies
    along x only is absorbed by the offsets, so the large scales along x are fixed only by weak
    cross-frames (one per pixel row, weight ``weak``): a near-null smooth mode. A 3 x 3 corner is
    never observed (compacted away). Returns the compact CSR ``A``, ``b``, the active mask and the
    full column count."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W]
    xt, yt = (xx - (W - 1) / 2) / ((W - 1) / 2), (yy - (H - 1) / 2) / ((H - 1) / 2)
    sky = 0.8 * xt + 0.5 * yt + 0.3 * xt ** 2 + 0.05 * rng.normal(size=(H, W))
    hole = (yy < hole) & (xx < hole)
    frames = []
    for x in range(W):
        for _ in range(3):
            frames.append(([(y, x) for y in np.sort(rng.choice(H, size=H * 3 // 4, replace=False))], 1.0))
    for y in range(H):
        frames.append(([(y, x) for x in range(W)], weak))
    nf = len(frames)
    offsets = rng.normal(size=nf)
    rows, cols, vals, b = [], [], [], []
    r = 0
    for f, (pix, w) in enumerate(frames):
        for (y, x) in pix:
            if hole[y, x]:
                continue
            rows += [r, r]
            cols += [y * W + x, H * W + f]
            vals += [w, w]
            b.append(w * (sky[y, x] + offsets[f] + noise * rng.normal()))
            r += 1
    for f in range(nf):                      # the offsets' gauge: their mean is 0 (one row)
        rows.append(r)
        cols.append(H * W + f)
        vals.append(1.0 / np.sqrt(nf))
    b.append(0.0)
    r += 1
    A = sp.csr_matrix((vals, (rows, cols)), shape=(r, H * W + nf))
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    assert active.sum() == A.shape[1] - int(hole.sum())
    return A[:, active].tocsr(), np.asarray(b), active, A.shape[1]


NEAR_NULL = _near_null_system()


def _solve(stop=None, monitor=None, solver='lsqr', iter_lim=150, **kw):
    A, b, active, n_full = NEAR_NULL
    opts = dict(ref_shape=(H, W), atol=0.0, btol=0.0, conlim=0.0, damp=0.0, solver=solver, n_threads=1,
                use_float32=False, iter_lim=iter_lim, active_mask=active, num_cols_full=n_full, return_record=True)
    opts.update(kw)
    return apply_lsqr(A.copy(), b.copy(), stop=stop, monitor=monitor, **opts)


def _x_gradient(x):
    """The coefficient of x in the degree-2 fit of the sky of a full-layout solution ``x``."""
    fit = SmoothFit((H, W), ('sky',), degree=2, step=1)
    active = NEAR_NULL[2]
    fit.bind(ActiveColumns(active))
    return float(fit.fit(Iterate(0, x[active], active=ActiveColumns(active), num_cols=x.size))[0][1])


@pytest.mark.parametrize('solver', ['lsqr', 'lsmr'])
def test_residual_and_gradient_rules_stop_before_the_large_scales_converge(solver, caplog):
    """The failure mode: the residual and gradient rules end the solve while the sky's field-wide gradient
    is at about half of its least-squares value; the large-scale rule waits until it has converged."""
    converged = _x_gradient(_solve(solver=solver)[0])
    with caplog.at_level(logging.INFO, logger='selfcal.core.monitor'):
        early = {name: _solve(stop=stop, solver=solver) for name, stop in
                 (('gradient', sc.Stop(gradient=1e-3)), ('residual', sc.Stop(residual=1e-3)),
                  ('large_scale', sc.Stop(large_scale=sc.LargeScaleRule(1e-2, every=10))))}
    its = {name: rec.iterations for name, (x, rec) in early.items()}
    for name, (x, rec) in early.items():
        assert rec.istop == RULE_ISTOP and rec.stop == STOP_REASONS[RULE_ISTOP]
        assert rec.stop_rule == name and rec.stop_iteration == rec.iterations
    assert its['gradient'] <= 20 and its['residual'] <= 25 and its['large_scale'] >= 60
    for name in ('gradient', 'residual'):
        assert _x_gradient(early[name][0]) < 0.6 * converged, name        # far from converged
    assert abs(_x_gradient(early['large_scale'][0]) / converged - 1) < 1e-3
    text = caplog.text
    assert f"Stopped by the gradient rule at iteration {its['gradient']}" in text
    assert f"Stopped by the large-scale rule at iteration {its['large_scale']}" in text


def test_combine_all_any_and_min_iterations():
    both = dict(gradient=1e-3, large_scale=sc.LargeScaleRule(1e-2, every=10))
    g = _solve(stop=sc.Stop(gradient=1e-3))[1].iterations
    ls = _solve(stop=sc.Stop(large_scale=sc.LargeScaleRule(1e-2, every=10)))[1].iterations
    x_all, all_ = _solve(stop=sc.Stop(**both))
    x_any, any_ = _solve(stop=sc.Stop(**both, combine='any'))
    assert any_.iterations == g < ls <= all_.iterations
    assert any_.stop_rule == 'gradient' and all_.stop_rule == 'gradient, large_scale'
    values = json.loads(all_.stop_values)
    assert values['rules']['gradient']['holds'] and values['rules']['large_scale']['holds']
    # min_iterations holds the rules back, not the decision where they hold
    late = _solve(stop=sc.Stop(gradient=1e-3, min_iterations=30))[1]
    assert late.iterations == 30 and late.stop_rule == 'gradient'
    # a rule that never holds: the iteration limit (the hard cap), recorded as "none"
    x, cap = _solve(stop=sc.Stop(gradient=1e-30), iter_lim=40)
    assert cap.istop == 7 and cap.iterations == 40 and cap.stop_rule == 'none' and cap.stop_iteration == 40
    assert not json.loads(cap.stop_values)['rules']['gradient']['holds']
    assert _same(x, _solve(iter_lim=40)[0])                        # rules that do not fire change nothing


@pytest.mark.parametrize('solver', ['lsqr', 'lsmr'])
def test_monitors_change_nothing_and_check_at_their_iterations(solver):
    x0, plain = _solve(solver=solver, iter_lim=35)
    x1, rec = _solve(solver=solver, iter_lim=35, monitor=sc.Monitor(10, gradient=True))
    assert _same(x0, x1) and rec.iterations == plain.iterations == 35 and rec.istop == plain.istop
    assert all(_same(a, b) for k, a in plain.history.arrays().items() for b in [rec.history.arrays()[k]]
               if k != 'elapsed_s')
    assert rec.stop_rule is None and 'stop_rule' not in rec.attrs()       # nothing in the cal's attributes
    w = rec.watch                     # the record keeps nothing of the solve alive (the operator holds A)
    assert w._op is None and w._b is None and w._scale is None and w._active is None and w.smooth._active is None
    arr = rec.watch.arrays()
    assert list(arr['check_itn']) == [0, 10, 20, 30] and list(arr['check_iteration']) == [0, 10, 20, 30]
    np.testing.assert_allclose(arr['residual_ratio'], 1, rtol=1e-8)
    np.testing.assert_allclose(arr['gradient_ratio'], 1, rtol=1e-5)
    assert arr['large_scale_coefficients'].shape == (4, 1, 6) and list(arr['large_scale_basis']) == \
        ['1', 'x', 'y', 'x^2', 'xy', 'y^2']
    assert np.isnan(arr['large_scale_change'][0, 0]) and arr['large_scale_change'][1, 0] == 1.0
    assert arr['large_scale_samples'][0] == 4 * 6 - 1           # every 4th row and column, the corner out
    assert not any(k.startswith('stop_') for k in arr)
    # the true values are those of the solution after exactly that many iterations
    A, b, active, _ = NEAR_NULL
    x20 = _solve(solver=solver, iter_lim=20)[0][active]
    assert abs(arr['true_residual'][2] / np.linalg.norm(b - A @ x20) - 1) < 1e-10
    xs = x20 * np.sqrt(np.asarray(A.multiply(A).sum(axis=0)).ravel())          # the solver's scaled unknowns
    As = A @ sp.diags(1 / np.sqrt(np.asarray(A.multiply(A).sum(axis=0)).ravel()))
    assert abs(arr['true_gradient'][2] / np.linalg.norm(As.T @ (b - As @ xs)) - 1) < 1e-6


def test_true_residual_rule_and_gradient_rule_on_checks():
    """A rule on the true values checks at its own cadence (no monitor) and stops only at a check."""
    x, rec = _solve(stop=sc.Stop(residual=sc.ResidualRule(1e-3, window=10, every=5)))
    assert rec.iterations % 5 == 0 and rec.stop_rule == 'residual'
    st = json.loads(rec.stop_values)['rules']['residual']
    assert st['measured'].startswith('the true') and st['value'] < 1e-3
    arr = rec.watch.arrays()
    assert list(arr['check_itn']) == list(range(0, rec.iterations + 1, 5))
    assert 'true_gradient' not in arr and 'large_scale_coefficients' not in arr        # only what the rule needs
    x, rec = _solve(stop=sc.Stop(gradient=sc.GradientRule(1e-3, window=4, every=2)))
    assert rec.iterations % 2 == 0 and rec.stop_rule == 'gradient'
    arr = rec.watch.arrays()
    k = list(arr['check_itn'])
    last = arr['true_gradient'][k.index(rec.iterations - 2):]
    assert max(last) <= 1e-3 * arr['true_gradient'][0]


# =================================================================== the rules on their own
class _FakeWatch:
    """The estimates and checks a StopRules reads, from given series."""

    def __init__(self, r1=None, ar=None, checks=None, smooth=None):
        self.r1, self.ar, self.checks, self.smooth = r1, ar, checks or {}, smooth

    def estimate(self, itn, name):
        return (self.r1 if name == 'r1norm' else self.ar)[itn]

    def check_value(self, itn, name):
        return self.checks[itn][name]


def _run_rules(stop, watch, n, tests=None):
    rules = StopRules(stop)
    rules.skip()
    for itn in range(1, n + 1):
        if rules.update(itn, tests(itn) if tests else (False,) * 7, watch):
            return itn, rules
    return None, rules


def test_each_rule_fires_where_it_should_and_not_before():
    # residual: r falls by 10 % per iteration until 30, then by 1e-5 per iteration
    r = [1.0]
    for i in range(1, 80):
        r.append(r[-1] * (0.9 if i <= 30 else 1 - 1e-5))
    itn, rules = _run_rules(sc.Stop(residual=sc.ResidualRule(1e-3, window=10)), _FakeWatch(r1=r), 79)
    assert itn == 40            # the first window wholly in the plateau: (r[30] - r[40]) / r[30] = 1e-4
    assert all(v >= 1e-3 for v in rules.series['residual'][10:40])
    # gradient: a non-monotone gradient, one spike in the window keeps the rule from holding
    g = [1.0] + [1e-4] * 60
    g[25] = 1e-2
    itn, rules = _run_rules(sc.Stop(gradient=sc.GradientRule(1e-3, window=10)), _FakeWatch(ar=g), 60)
    assert itn == 10 and rules.fired == (10, ['gradient'])
    itn, rules = _run_rules(sc.Stop(gradient=sc.GradientRule(1e-3, window=10), min_iterations=25), _FakeWatch(ar=g),
                            60)
    assert itn == 35                    # every window ending at 25-34 holds the spike at 25
    assert all(v == 1e-2 for v in rules.series['gradient'][25:35]) and rules.series['gradient'][35] == 1e-4
    # large scale: fits that settle at check 50 (relative change 1e-4 from then on)
    fit = SmoothFit((8, 8), ('sky',), 1, 1)
    fit.bind()
    c = {k: {'coefficients': np.array([[0.0, 1.0 + k / 10 if k < 50 else 7.0 + 1e-4 * k / 10, 0.5]])}
         for k in range(0, 101, 10)}
    itn, rules = _run_rules(sc.Stop(large_scale=sc.LargeScaleRule(1e-3, every=10)),
                            _FakeWatch(checks=c, smooth=fit), 100)
    assert itn == 70                    # the changes at 60 and 70 are both small; at 50 the jump
    assert np.isnan(rules.series['large_scale'][15])                  # evaluated at checks only
    # the solver's tests, each on its own
    tests = lambda itn: (itn >= 30, itn >= 20, itn >= 10, False, False, False, False)  # noqa: E731
    for kept, at in ((True, 10), (('compatible',), 30), (('least_squares',), 20), (('condition',), 10),
                     (('compatible', 'least_squares'), 20)):
        itn, rules = _run_rules(sc.Stop(lsqr_tests=kept), _FakeWatch(), 50, tests)
        assert itn == at, kept


def test_a_watch_returns_the_hard_stops():
    w = Watch(stop=sc.Stop(gradient=1e-30), history=SolveHistory())
    w._op = object()                     # no checks are due: nothing is computed
    w.history.add(0, 1, 1, 1, 0, 0, 0, 1, float('nan'))
    assert w(0, None, 0, None) == 0
    for code in (4, 5, 6, 7):
        w.history.add(code, 1, 1, 1, 0, 0, 0, 1, 1)
        tests = tuple(k == code - 1 for k in range(7))
        assert w(code, None, 0, tests) == code
    # the solver's tolerance tests without lsqr_tests: ignored
    w.history.add(8, 1, 1, 1, 0, 0, 0, 1, 1)
    assert w(8, None, 1, (True, True, True, False, False, False, False)) == 0


def test_the_solver_tests_switchable_on_their_own():
    """On a well-conditioned system the default tolerances stop LSQR at k; Stop(lsqr_tests=True) stops at
    the same k with the same solution (istop 8, the test recorded); a Stop without them runs on."""
    A = sp.random(400, 60, density=0.2, random_state=1, format='csr')
    b = np.random.default_rng(0).normal(size=400)
    kw = dict(ref_shape=(1, 1), damp=0.0, n_threads=1, solver='lsqr', iter_lim=200, return_record=True,
              atol=1e-6, btol=1e-6)
    x, plain = apply_lsqr(A.copy(), b.copy(), **kw)
    assert plain.istop in (1, 2)
    x1, rec = apply_lsqr(A.copy(), b.copy(), stop=sc.Stop(lsqr_tests=True), **kw)
    assert _same(x, x1) and rec.iterations == plain.iterations and rec.istop == RULE_ISTOP
    st = json.loads(rec.stop_values)['rules']['lsqr_tests']
    assert rec.stop_rule == 'lsqr_tests' and st['holds'] and st['tests'][{1: 'compatible', 2: 'least_squares'}[
        plain.istop]]
    h = plain.history.arrays()
    first_ls = int(np.argmax(h['test2'][1:] <= 1e-6)) + 1
    x2, ls = apply_lsqr(A.copy(), b.copy(), stop=sc.Stop(lsqr_tests='least_squares'), **kw)
    assert ls.iterations == first_ls
    x3, off = apply_lsqr(A.copy(), b.copy(), stop=sc.Stop(gradient=1e-30), **kw)
    assert off.iterations > plain.iterations and off.istop in (4, 5, 6, 7) and off.stop_rule == 'none'


def test_monitors_and_rules_need_the_csr_system():
    A, b, _, _ = NEAR_NULL
    with pytest.raises(ValueError, match='COO path'):
        apply_lsqr(A.tocoo(), b, (H, W), monitor=sc.Monitor(5))
    with pytest.raises(ValueError, match='COO path'):
        apply_lsqr(A.tocoo(), b, (H, W), stop=sc.Stop(gradient=1e-3))


# =================================================================== the large-scale fit
def test_the_smooth_fit_reads_polynomials_exactly(monkeypatch):
    from selfcal.core import snapshots as snapshots_mod
    monkeypatch.setattr(snapshots_mod, '_PREFIX_BLOCK', 7)
    h, w = 13, 21
    yy, xx = np.mgrid[0:h, 0:w]
    xt, yt = (xx - (w - 1) / 2) / ((w - 1) / 2), (yy - (h - 1) / 2) / ((h - 1) / 2)
    powers = monomials(3)
    coef = np.random.default_rng(1).normal(size=(2, len(powers)))
    skies = [sum(c * xt ** a * yt ** bb for c, (a, bb) in zip(coef[j], powers)) for j in range(2)]
    n_extra = 5
    full = np.concatenate([skies[0].ravel(), skies[1].ravel(), np.arange(n_extra, dtype=float)])
    mask = np.ones(full.size, dtype=bool)
    mask[np.random.default_rng(2).choice(2 * h * w, 60, replace=False)] = False       # uncovered pixels
    scale = np.random.default_rng(3).uniform(0.5, 2, size=int(mask.sum()))
    act = ActiveColumns(mask)
    it = Iterate(5, full[mask] / scale, scale=scale, active=act, num_cols=full.size)
    fit = SmoothFit((h, w), ('a', 'b'), degree=3, step=2)
    fit.bind(act)
    got = fit.fit(it)
    np.testing.assert_allclose(got, coef, rtol=1e-9, atol=1e-12)
    # the change: the rms over the sampled covered pixels of the change, means removed, by brute force
    other = coef + np.random.default_rng(4).normal(scale=0.1, size=coef.shape)
    for j in range(2):
        keep = mask[j * h * w:(j + 1) * h * w].reshape(h, w)[::2, ::2]
        g = np.stack([(xt ** a * yt ** bb)[::2, ::2][keep] for a, bb in powers], 1)
        s_new, s_old = g @ other[j], g @ coef[j]
        want = np.std(s_new - s_old) / np.std(s_new)
        assert abs(fit.change(other, coef)[j] / want - 1) < 1e-9
    assert fit.samples == tuple(int(mask[j * h * w:(j + 1) * h * w].reshape(h, w)[::2, ::2].sum()) for j in range(2))


# =================================================================== the settings
def test_the_settings():
    s = sc.Stop(residual=1e-3, gradient=2e-3, large_scale=1e-2)
    assert s.residual == sc.ResidualRule(1e-3) and s.gradient == sc.GradientRule(2e-3)
    assert s.large_scale == sc.LargeScaleRule(1e-2) and s.combine == 'all' and s.rules == (
        'residual', 'gradient', 'large_scale')
    assert sc.Stop(lsqr_tests=['condition', 'compatible', 'least_squares']).lsqr_tests is True
    assert sc.Stop(lsqr_tests='condition').tests == ('condition',) and sc.Stop(lsqr_tests=True).tests == (
        'compatible', 'least_squares', 'condition')
    for bad, match in ((dict(), 'at least one rule'), (dict(lsqr_tests=()), 'at least one rule'),
                       (dict(residual=0), 'positive'), (dict(gradient=1e-3, combine='both'), 'expected one of'),
                       (dict(lsqr_tests='fast'), "one of 'compatible'"), (dict(gradient=1e-3, min_iterations=-1), '0')):
        with pytest.raises(ConfigError, match=match):
            sc.Stop(**bad)
    with pytest.raises(ConfigError, match='multiple of every'):
        sc.GradientRule(1e-3, window=10, every=3)
    with pytest.raises(ConfigError, match='degree'):
        sc.LargeScaleRule(1e-3, degree=0)
    with pytest.raises(ConfigError, match='tolerance=0 turns'):
        sc.Fit(100, tolerance=0, stop=sc.Stop(lsqr_tests=True))
    with pytest.raises(ConfigError, match='no rule could end'):
        sc.Fit(100, stop=sc.Stop(gradient=1e-3, min_iterations=101))
    assert sc.Monitor() == sc.Monitor(10) and sc.Monitor().large_scale is True
    for bad in (dict(every=0), dict(large_scale=9), dict(step=0),
                dict(residual=False, gradient=False, large_scale=False)):
        with pytest.raises(ConfigError):
            sc.Monitor(**bad)
    from selfcal.run.schedule import as_monitor
    assert as_monitor(5) == sc.Monitor(5) and as_monitor(True) == sc.Monitor() and as_monitor(None) is None
    with pytest.raises(ConfigError, match='a sc.Monitor'):
        as_monitor('5')
    # one large-scale fit per solve: a monitor asking for another than the rule's is refused
    with pytest.raises(ValueError, match='degree 3'):
        Watch(stop=sc.Stop(large_scale=1e-3), monitor=sc.Monitor(large_scale=3))
    w = Watch(stop=sc.Stop(large_scale=sc.LargeScaleRule(1e-3, degree=3, step=2)), monitor=sc.Monitor())
    assert (w.smooth.degree, w.smooth.step) == (3, 2)


def test_without_stop_the_fit_encodes_as_before():
    assert 'stop' not in sc.Fit(50).to_dict() and 'stop' not in repr(sc.Fit(50))
    assert sc.Fit(50).replace(stop=sc.Stop(gradient=1e-3)).to_dict()['stop']['gradient']['below'] == 1e-3
    from selfcal.config.base import decode
    f = sc.Fit(80, stop=sc.Stop(gradient=1e-3, large_scale=sc.LargeScaleRule(1e-3, every=5), min_iterations=10))
    assert decode(json.loads(json.dumps(f.to_dict()))) == f


# =================================================================== through the API
NUMERICS = sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4)


def _recipe(iterations=30, name='m', stop=None, **fit):
    return sc.Recipe(sc.continuum(), fit=sc.Fit(iterations, tolerance=0, stop=stop, **fit), coadd=None,
                     numerics=NUMERICS, name=name)


@pytest.fixture(scope='module')
def field():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_monitor_')
    write_exposures(os.path.join(tmp, 'exposures'), 12, np.random.default_rng(5))
    f = sc.Field(os.path.join(tmp, 'out', 'toy'), sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy'),
                 REF_ARCSEC, compute=sc.Compute(os.path.join(tmp, 'cache'), workers=2, io_limit=4))
    f.reproject(os.path.join(tmp, 'exposures', 'toy_exp_*_D0.fits'), method='interp', padding=8)
    yield f
    shutil.rmtree(tmp, ignore_errors=True)


def _history(cal):
    with CalFile(cal) as c:
        return dict(np.load(c.solve['history_file']))


@pytest.mark.filterwarnings('ignore:.*recorded with code:UserWarning')
def test_a_monitored_calibration_writes_the_same_cal_and_records_its_checks(field, caplog):
    recipe = _recipe(14, name='mon')
    with caplog.at_level(logging.INFO, logger='selfcal'):
        res = field.calibrate(recipe, monitor=sc.Monitor(4, gradient=True))
    cal = res.cal_paths[0]
    out = caplog.text
    assert 'monitor at iteration 12: |b - A x| = ' in out and 'large scale (degree 2): continuum rms' in out
    saved = open(cal, 'rb').read()
    side = products.read_sidecar(cal)['fingerprint']
    h = _history(cal)
    assert list(h['check_itn']) == [0, 4, 8, 12] and list(h['itn']) == list(range(15))
    np.testing.assert_allclose(h['residual_ratio'], 1, rtol=1e-3)            # float32 solve: the estimates hold
    assert h['large_scale_coefficients'].shape == (4, 1, 6) and list(h['large_scale_terms']) == ['continuum']
    record = json.load(open(res.record))
    assert record['settings']['monitor'] == sc.Monitor(4, gradient=True).to_dict()
    (entry,) = record['solves']
    assert entry['monitor']['every'] == 4 and entry['monitor']['checks'] == 4 and entry['monitor']['last']['itn'] == 12
    assert 'stop_rule' not in entry
    # the same cal, byte for byte, without the monitor; the same fingerprint; the plan's book agrees
    (job,) = field.instrument.default_jobs()
    assert field.plan(recipe, monitor=4).book.cal(job, field.frames) == field.plan(recipe).book.cal(job, field.frames)
    assert 'monitor     every 4 iterations: |b - A x|, the sky terms' in str(field.plan(recipe, monitor=4))
    again = field.calibrate(recipe, overwrite=True).cal_paths[0]
    assert open(again, 'rb').read() == saved and products.read_sidecar(again)['fingerprint'] == side
    assert 'check_itn' not in _history(again)
    # a rerun of the monitored action replays the monitor
    redone = sc.rerun(res.record, overwrite=True)
    assert list(_history(redone.cal_paths[0])['check_itn']) == [0, 4, 8, 12]


def test_monitors_continue_across_a_warm_start(field):
    first = field.calibrate(_recipe(12, name='ws_first'), monitor=sc.Monitor(4))
    h1 = _history(first.cal_paths[0])
    more = field.calibrate(_recipe(8, name='ws_more'), start=first, monitor=sc.Monitor(4))
    h2 = _history(more.cal_paths[0])
    assert list(h2['check_itn']) == [0, 4, 8] and list(h2['check_iteration']) == [12, 16, 20]
    # the continuation's first check is the start: the large scales of the first solve's last check
    np.testing.assert_allclose(h2['large_scale_coefficients'][0], h1['large_scale_coefficients'][-1], rtol=1e-5,
                               atol=1e-9)
    assert np.isfinite(h2['large_scale_change'][1]).all()            # its change is measured from the start


def test_the_stop_record_in_the_cal_the_record_and_the_log(field, caplog):
    stop = sc.Stop(gradient=sc.GradientRule(1e-3, window=5), min_iterations=8)
    with caplog.at_level(logging.INFO, logger='selfcal'):
        res = field.calibrate(_recipe(300, name='stopped', stop=stop), monitor=sc.Monitor(5))
    cal = res.cal_paths[0]
    out = caplog.text
    with CalFile(cal) as c:
        solve = c.solve
        assert 'stop rules: gradient (judged at iteration' in c.describe()
    n = solve['iterations']
    assert 8 <= n < 300 and solve['istop'] == RULE_ISTOP and solve['stop_rule'] == 'gradient'
    assert solve['stop_iteration'] == n
    values, policy = json.loads(solve['stop_values']), json.loads(solve['stop_policy'])
    g = values['rules']['gradient']
    assert values['iteration'] == n and g['holds'] and g['value'] <= 1e-3 and g['window'] == 5
    assert policy == stop.to_dict()
    assert f"Stopped by the gradient rule at iteration {n}" in out and 'the gradient rule holds' in out
    (entry,) = json.load(open(res.record))['solves']
    assert entry['stop_rule'] == 'gradient' and entry['stop_values'] == values and entry['stop_policy'] == policy
    h = _history(cal)
    assert h['stop_gradient'].shape == h['itn'].shape and h['stop_holds'][n] and not h['stop_holds'][:n].any()
    # the rules enter the cal's inputs (a cal without them is another product)
    (job,) = field.instrument.default_jobs()
    plain = _recipe(300, name='stopped')
    a = products.cal_inputs(field, _recipe(300, name='stopped', stop=stop), job, field.frames)
    b = products.cal_inputs(field, plain, job, field.frames)
    assert products.fingerprint(a) != products.fingerprint(b) and 'stop' not in json.dumps(b['fit'])
    assert 'stop        the largest |A^T r| over 5 iterations at most 0.001 of its start; not before iteration 8' \
        in str(field.plan(_recipe(300, name='stopped', stop=stop)))


def test_snapshots_monitors_and_a_stop_together(field):
    """Snapshots every 4 iterations until the rule ends the solve (none at or after it); the monitor's
    checks and the rule's values in one history; the solution is that of a plain solve of that length."""
    stop = sc.Stop(gradient=sc.GradientRule(1e-3, window=5), min_iterations=8)
    res = field.calibrate(_recipe(300, name='together', stop=stop), snapshots=sc.Snapshots(4), monitor=sc.Monitor(4))
    cal = res.cal_paths[0]
    with CalFile(cal) as c:
        n = c.solve['iterations']
    stem = os.path.basename(cal)[:-len('.h5')]
    snaps = sorted(p for p in os.listdir(os.path.join(os.path.dirname(cal), 'snapshots')) if p.startswith(stem))
    assert snaps == [f'{stem}_it{k:04d}.h5' for k in range(4, n, 4)]
    h = _history(cal)
    assert list(h['check_itn']) == list(range(0, n + 1, 4)) and h['stop_holds'][n]
    exact = field.calibrate(_recipe(n, name='together_exact')).cal_paths[0]
    with h5py.File(cal, 'r') as a, h5py.File(exact, 'r') as b:
        for name in ('sky/continuum', 'offsets/map_0', 'frame_scalar'):
            assert _same(a[name][()], b[name][()]), name


def test_tiles_are_monitored_one_history_each(field):
    tiles = sc.Tiles((1, 2), overlap=10, names=('W', 'E'))
    t = field.calibrate(_recipe(6, name='tmon'), tiles=tiles, monitor=sc.Monitor(3),
                        compute=field.compute.replace(stage_dir='toy_tiles_mon', memory_guard=False))
    for e in json.load(open(t.record))['solves']:
        h = np.load(e['history_file'])
        assert list(h['check_itn']) == [0, 3, 6] and e['monitor']['checks'] == 3


def test_a_mosaic_takes_no_monitor(field):
    from selfcal.run.plan import make_plan
    with pytest.raises(ConfigError, match='only a calibration'):
        make_plan(field, 'mosaic', _recipe(5).replace(coadd=sc.Coadd()), monitor=sc.Monitor(2))
    with pytest.raises(ConfigError, match='degree 3'):
        field.plan(_recipe(20, stop=sc.Stop(large_scale=1e-3)), monitor=sc.Monitor(large_scale=3))


def test_history_files_of_old_solves_read_as_before(tmp_path):
    """A solve without monitors or rules saves exactly feature 1's arrays."""
    A = sp.random(100, 20, density=0.3, random_state=1, format='csr')
    b = np.random.default_rng(0).normal(size=100)
    x, rec = apply_lsqr(A, b, (1, 1), n_threads=1, solver='lsqr', iter_lim=5, return_record=True, damp=0.0)
    path = rec.save_history(tmp_path / 'h.npz')
    with np.load(path) as h:
        assert sorted(h.files) == sorted(['method', 'itn', 'r1norm', 'r2norm', 'arnorm', 'anorm', 'acond', 'xnorm',
                                          'test1', 'test2', 'elapsed_s'])
    with h5py.File(tmp_path / 'c.h5', 'w') as f:
        rec.write(f.create_group('solve'))
        assert not any(k.startswith('stop_') for k in f['solve'].attrs)
