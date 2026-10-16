"""Null modes (selfcal.core.nullmodes; ``sc.Fit(null_modes=sc.NullModes(...))``).

After a solve, smooth local sky patterns (cubic B-splines) are paired with the offsets that absorb them, the
directions the data see least are found from the system (each with its nullness), and those below a ratio lose
their sky amplitude to the offsets. It must: find directions that are genuinely null or near-null, leave the exact
gauges and the well-seen directions alone, remove exactly the component it reports (the sky changes by the rendered
component, the residual by what it says at a least-squares solution), be recorded in the cal, the record and the
plan, and be part of the fingerprint only when given.
"""
import json
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import selfcal as sc  # noqa: E402
from selfcal import _state  # noqa: E402
from selfcal.config import ConfigError  # noqa: E402
from selfcal.core.nullmodes import GAUGE_RATIO, NullModes, bspline_weights, render  # noqa: E402
from selfcal.core.solve import apply_lsqr  # noqa: E402
from tests.synthetic_exposures import DET, N_CHUNK_SIDE, REF_ARCSEC, write_exposures  # noqa: E402
from tests.test_deflation import _chunk_system, _converged, _near_null_system  # noqa: E402

pytestmark = pytest.mark.filterwarnings('ignore:overflow encountered in cast:RuntimeWarning')

CHUNK = _chunk_system()
NEAR_NULL = _near_null_system()


@pytest.fixture(scope='module', autouse=True)
def scratch(tmp_path_factory):
    global _SCRATCH
    _SCRATCH = str(tmp_path_factory.mktemp('scratch'))
    yield _SCRATCH


_SCRATCH = None


def _solve(system, null_modes=None, iter_lim=400, solver='lsqr', **kw):
    A, b, _, grid = system
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    opts = dict(ref_shape=grid, atol=0.0, btol=0.0, conlim=0.0, damp=0.0, solver=solver, n_threads=1,
                use_float32=False, iter_lim=iter_lim, active_mask=active, num_cols_full=A.shape[1],
                return_record=True, scratch_dir=_SCRATCH, sky_names=('sky',))
    opts.update(kw)
    return apply_lsqr(A[:, active].tocsr(), b.copy(), null_modes=null_modes, **opts)


def _analysis(system, settings):
    """The analysis built directly on the column-scaled system (as the solver sees it), and the column
    scale: solver x = physical x * cn."""
    A, b, _, grid = system
    active = np.asarray((A != 0).sum(axis=0)).ravel() > 0
    Ac = A[:, active].tocsr()
    cn = np.sqrt(np.asarray(Ac.multiply(Ac).sum(axis=0)).ravel())
    B = Ac @ __import__('scipy.sparse', fromlist=['diags']).diags(1.0 / cn)
    return NullModes(B.tocsr(), scale=1.0 / cn, active_mask=active, ref_shape=grid, sky_names=('sky',),
                     settings=settings, threads=1), active, cn, B.tocsr()


# =================================================================== the patterns
def test_the_splines_partition_unity_and_render_a_constant():
    t = np.random.default_rng(0).random(1000) * 5
    i, w = bspline_weights(t)
    assert np.all(w >= 0) and np.allclose(w.sum(axis=1), 1.0) and np.all(i == np.floor(t))
    grid = (3, 2, 4, 5)                                     # x0, y0, nx, ny: cells of 6 px
    m = render((40, 40), grid, np.ones((4 + 3) * (5 + 3)), cell=6)
    inside = m[2:2 + 30, 3:3 + 24]
    assert np.allclose(inside, 1.0) and m[:2].sum() == 0 and m[:, :3].sum() == 0


# =================================================================== the setting
def test_the_setting_is_checked_and_fingerprint_neutral():
    nm = sc.NullModes()
    assert nm.cell == 512.0 and nm.ratio is None and nm.terms is None
    assert sc.NullModes(terms='continuum').terms == ('continuum',)
    for bad in (dict(cell=0), dict(cell=-3), dict(ratio=1.5), dict(ratio=0), dict(terms=('a', 'b'))):
        with pytest.raises(ConfigError):
            sc.NullModes(**bad)
    assert 'null_modes' not in sc.Fit().to_dict()
    assert sc.Fit(null_modes=sc.NullModes(256)).to_dict()['null_modes']['cell'] == 256
    with pytest.raises(ConfigError, match='undamped'):
        sc.Fit(null_modes=sc.NullModes(), damp=0.01)
    with pytest.raises(ConfigError):
        sc.Fit(null_modes=sc.Deflate())


# =================================================================== the spectrum
def test_the_spectrum_separates_the_near_null_directions():
    """Frames that each see one pixel column, tied by weak cross frames: smooth patterns along x are nearly
    null, patterns along y are seen; the spectrum spans orders of magnitude and the smallest directions vary
    along x."""
    x, rec = _solve(NEAR_NULL, null_modes=sc.NullModes(cell=4), iter_lim=50)
    s = json.loads(rec.null_modes)
    nm = rec.null_modes_analysis
    assert s['patterns'] == (6 + 3) * (4 + 3) and s['directions'] >= 20 and s['ratio'] is None and s['applied'] == 0
    nu = nm.nullness
    assert np.all(np.diff(nu) >= -1e-12)                    # ascending
    assert nu[0] < 1e-2 * nu[-1] and nu[-1] > 0.3            # near-null and well-seen directions both present
    assert 'null_modes' in rec.attrs() and rec.null_modes_s is not None
    # the sigma of a direction: noise / nu times its sky rms per unit amplitude
    sig = nm.sigma(rec.noise_variance)
    assert sig[0] > 10 * sig[-1]
    # the least-seen direction varies along x (the columns the frames cannot tie) far more than along y
    grid = NEAR_NULL[3]
    m = render(grid, nm.grid, nm.V[:, 0], nm.cell)
    assert np.abs(np.diff(m, axis=1)).mean() > 3 * np.abs(np.diff(m, axis=0)).mean()


def test_the_rule_removes_exactly_what_it_reports():
    A, b, npix, grid = NEAR_NULL
    x_none, rec_none = _solve(NEAR_NULL, null_modes=sc.NullModes(cell=4), iter_lim=600)
    x_rule, rec_rule = _solve(NEAR_NULL, null_modes=sc.NullModes(cell=4, ratio=0.02), iter_lim=600)
    s = json.loads(rec_rule.null_modes)
    nm = rec_rule.null_modes_analysis
    assert s['applied'] >= 1 and s['removed_rms'] > 0 and np.all(nm.nullness[nm.applied] < 0.02)
    assert np.all(nm.nullness[nm.applied] >= GAUGE_RATIO)
    # the sky changed by minus the rendered component, nothing else of the sky
    d_sky = (x_rule - x_none)[:npix].reshape(grid)
    np.testing.assert_allclose(d_sky, -nm.render_removed(grid), atol=1e-9 * max(1.0, np.abs(d_sky).max()))
    assert abs(np.sqrt(np.mean(d_sky ** 2)) - s['removed_rms']) < 1e-9
    # the fit moved by at most what it says (the solve is converged to rounding on this small system)
    assert abs(rec_rule.true_residual - rec_none.true_residual) <= s['residual_change'] + 1e-9
    # the removed directions hold no sky component any more; the component before equals the amplitude recorded
    nm2, active, cn, B = _analysis(NEAR_NULL, sc.NullModes(cell=4, ratio=0.02))
    xs = x_none[active] * cn
    before = nm2.component(xs)
    nm2.apply(xs, rec_none.noise_variance)
    sel = nm2.applied
    Vs = nm2.V[:, sel]
    r_after = Vs.T @ nm2.pattern_dots(nm2._sky_physical(xs))
    assert np.abs(r_after).max() < 1e-9 * max(1.0, np.abs(Vs.T @ nm2.pattern_dots(nm2._sky_physical(x_none[active] * cn))).max())
    np.testing.assert_allclose(nm2.amplitude, before)
    assert np.array_equal(nm2.applied, nm.applied)


def test_at_the_least_squares_solution_the_residual_moves_by_the_norm_of_the_removed_pair():
    """At the exact least-squares solution the residual is orthogonal to the range of the matrix, so removing
    ``c . z`` changes ``|r|^2`` by exactly ``|B c z|^2``, what the analysis records."""
    A, b, npix, grid = NEAR_NULL
    nm, active, cn, B = _analysis(NEAR_NULL, sc.NullModes(cell=4, ratio=0.05))
    x_star = _converged(NEAR_NULL)
    xs = x_star[active] * cn
    r0 = b - B @ xs
    nm.apply(xs, float(r0 @ r0) / max(B.shape[0] - B.shape[1], 1))
    r1 = b - B @ xs
    assert nm.applied.sum() >= 1
    np.testing.assert_allclose(float(r1 @ r1 - r0 @ r0), nm.residual_change ** 2, rtol=1e-6)


def test_exact_gauges_and_well_seen_directions_are_left_alone():
    """The chunk system's exact gauges (the sky's mean against the scalars, the x-plane against the slopes)
    keep the solve's convention; a ratio below every non-gauge nullness changes nothing at all."""
    x_plain, rec_plain = _solve(CHUNK, iter_lim=200)
    x, rec = _solve(CHUNK, null_modes=sc.NullModes(cell=6, ratio=2e-6), iter_lim=200)
    s = json.loads(rec.null_modes)
    assert s['exact_gauges'] >= 2 and s['applied'] == 0
    assert np.array_equal(x, x_plain) and rec.true_residual == rec_plain.true_residual
    x2, rec2 = _solve(CHUNK, null_modes=sc.NullModes(cell=6, ratio=0.5), iter_lim=200)
    nm = rec2.null_modes_analysis
    assert nm.applied.sum() >= 1 and not np.any(nm.applied & (nm.nullness < GAUGE_RATIO))
    assert abs(rec2.true_residual - rec_plain.true_residual) <= json.loads(rec2.null_modes)['residual_change'] + 1e-9


def test_what_the_analysis_refuses():
    A, b, npix, grid = CHUNK
    with pytest.raises(ValueError, match='undamped'):
        _solve(CHUNK, null_modes=sc.NullModes(cell=6), damp=0.1, iter_lim=5)
    with pytest.raises(ValueError, match='COO'):
        apply_lsqr(A.tocoo(), b.copy(), ref_shape=grid, null_modes=sc.NullModes(cell=6), damp=0.0)
    with pytest.raises(ValueError, match='no sky term'):
        _solve(CHUNK, null_modes=sc.NullModes(cell=6, terms=('line',)), iter_lim=5)


# =================================================================== through the API
NUMERICS = sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4)


def _recipe(iterations=30, name='n', **fit):
    return sc.Recipe(sc.continuum(), fit=sc.Fit(iterations, tolerance=0, **fit), coadd=None, numerics=NUMERICS,
                     name=name)


@pytest.fixture(scope='module')
def field():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_nullmodes_')
    write_exposures(os.path.join(tmp, 'exposures'), 12, np.random.default_rng(5))
    f = sc.Field(os.path.join(tmp, 'out', 'toy'), sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy'),
                 REF_ARCSEC, compute=sc.Compute(os.path.join(tmp, 'cache'), workers=2, io_limit=4))
    f.reproject(os.path.join(tmp, 'exposures', 'toy_exp_*_D0.fits'), method='interp', padding=8)
    yield f
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.mark.filterwarnings('ignore:.*recorded with code:UserWarning')
def test_a_calibration_with_null_modes_is_planned_recorded_and_fingerprinted(field, caplog):
    import logging
    (job,) = field.instrument.default_jobs()
    book = lambda r: field.plan(r).book.cal(job, field.frames)          # noqa: E731
    plain, with_nm = _recipe(20, name='same'), _recipe(20, name='same', null_modes=sc.NullModes(16, ratio=1e-2))
    assert book(plain) != book(with_nm) and book(plain) == book(_recipe(20, name='same', null_modes=None))
    plan = str(field.plan(_recipe(20, name='n1', null_modes=sc.NullModes(16, ratio=1e-2))))
    assert 'null modes  after the solve, cubic B-spline patterns with 16-px knots' in plan and 'below nullness 0.01' in plan
    with caplog.at_level(logging.INFO, logger='selfcal'):
        res = field.calibrate(_recipe(20, name='n1', null_modes=sc.NullModes(16, ratio=1e-2)))
    assert 'NullModes: ' in caplog.text and 'spline patterns' in caplog.text
    cal = res.cal_paths[0]
    with h5py.File(cal, 'r') as f:
        attrs = dict(f['solve'].attrs)
        grp = f['null_modes']
        names = set(grp.keys())
        g_attrs = dict(grp.attrs)
        nu = grp['nullness'][()]
        directions = grp['directions'][()]
    assert 'null_modes_s' not in attrs and 'null_modes' in attrs
    s = json.loads(attrs['null_modes'])
    assert s['cell'] == 16 and s['ratio'] == 1e-2 and s['directions'] == nu.size == directions.shape[1]
    assert {'nullness', 'directions', 'removed', 'applied', 'amplitude', 'sigma_sky_rms', 'sky_rms_unit'} <= names
    assert int(g_attrs['patterns']) == s['patterns']
    (entry,) = json.load(open(res.record))['solves']
    assert entry['null_modes']['patterns'] == s['patterns'] and entry['null_modes_s'] >= 0
