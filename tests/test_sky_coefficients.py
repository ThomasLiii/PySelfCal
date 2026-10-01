"""Sky terms are a map times ANY coefficient function of data variables.

Unit level: ``Coefficient`` with plain callables (one or several variables, scalar
results broadcast), functions referenced by import path (and pickled the way the
worker processes receive them), bit-identity of the historical spelling, the
config form and its errors, the per-term damping rule.

End to end: an instrument providing one data variable ``u`` (a detector-plane
ramp), synthetic exposures carrying a second sky term ``S1(p) * shape(u)`` with
``shape`` an arbitrary function defined in this file, and a ``[model]`` whose
second term names that function — the solve recovers ``S1``.
"""
import dataclasses
import os
import pickle
import shutil
import sys
import tempfile

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from selfcal import _state                                                   # noqa: E402
from selfcal.instruments import register_instrument, get_instrument          # noqa: E402
from selfcal.instruments.grid import GridInstrument                          # noqa: E402
from selfcal.io.calfile import CalFile                                       # noqa: E402
from selfcal.models.profiles import GaussianProfile, QuadratureSigma, TemplateProfile   # noqa: E402
from selfcal.models.sky_model import (Coefficient, ImportedFunction, SkyComponent,     # noqa: E402
                                      SkyModel, SpectralComponent, ContinuumComponent)
from selfcal.models.spec import ModelSpec, SkyTerm, build_coefficient                 # noqa: E402


# ---------------------------------------------------------------------------
# arbitrary coefficient functions (module level: importable by path in workers)
# ---------------------------------------------------------------------------
def shape(u, power=2.0, offset=0.0):
    """An arbitrary coefficient of one variable."""
    return offset + np.asarray(u, dtype=np.float64) ** power


def two_var(a, b, scale=1.0):
    return scale * a * b


def constant_half(u):
    return 0.5


# ---------------------------------------------------------------------------
# unit level
# ---------------------------------------------------------------------------
def test_callable_coefficients():
    obs = {'u': np.linspace(0, 1, 5), 'v': np.full(5, 2.0)}
    c = Coefficient('u', shape).evaluate(obs)
    assert c.dtype == np.float32 and np.allclose(c, np.linspace(0, 1, 5) ** 2)
    c2 = Coefficient(('u', 'v'), two_var).evaluate(obs)
    assert np.allclose(c2, 2 * np.linspace(0, 1, 5))
    c3 = Coefficient('u', constant_half).evaluate(obs)
    assert c3.shape == (5,) and np.all(c3 == 0.5)
    comp = SkyComponent('s', Coefficient(('u', 'v'), two_var))
    assert comp.variables == ('u', 'v') and comp.required_variables == ('u', 'v')
    assert SkyComponent('flat').coefficients(obs) is None


def test_imported_function_pickles():
    f = ImportedFunction('tests.test_sky_coefficients:shape', (('power', 3.0), ('offset', 1.0)))
    c = Coefficient('u', f)
    c2 = pickle.loads(pickle.dumps(c))
    obs = {'u': np.array([0.0, 0.5, 1.0])}
    assert np.allclose(c2.evaluate(obs), [1.0, 1.125, 2.0])
    try:
        ImportedFunction('not_a_path', ())(np.zeros(1))
        raise AssertionError('expected a ValueError')
    except ValueError:
        pass


def test_historical_spelling_is_bit_identical():
    rng = np.random.default_rng(0)
    obs = {'BC': rng.uniform(3.1, 3.5, 10000).astype(np.float32),
           'BW': rng.uniform(0.02, 0.06, 10000).astype(np.float32)}
    prof = GaussianProfile(center_um=3.29, sigma_um=0.02,
                           sigma_source=QuadratureSigma(fwhm_key='BW', fwhm_to_sigma=2.355, intrinsic_var_um2=2.89e-4))
    old = SpectralComponent(name='pah', profile=prof, wavelength_key='BC').coefficients(obs)
    new = SkyComponent('pah', Coefficient('BC', prof)).coefficients(obs)
    assert old.dtype == new.dtype and old.tobytes() == new.tobytes()
    tmpl = TemplateProfile(wave_um=np.linspace(3.0, 3.6, 61), values=np.exp(-np.linspace(-3, 3, 61) ** 2))
    old = SpectralComponent(name='t', profile=tmpl).coefficients(obs)
    new = SkyComponent('t', Coefficient('BC', tmpl)).coefficients(obs)
    assert old.tobytes() == new.tobytes()
    assert SpectralComponent(name='pah', profile=prof).variables == ('BC', 'BW')
    assert SpectralComponent(name='pah', profile=prof).required_variables == ('BC',)


def test_damp_weight_rule():
    m = SkyModel((ContinuumComponent(), SkyComponent('a', Coefficient('u', shape)),
                  SkyComponent('b', Coefficient('u', shape), damp_weight=0.02)))
    assert m.damp_weights(0.1, 0.3) == [0.1, 0.3, 0.02]
    assert m.damp_weights(0.1, None) == [0.1, 0.0, 0.02]
    m2 = SkyModel((SkyComponent('sky', damp_weight=0.5),))
    assert m2.damp_weights(0.1) == [0.5]                  # the first term honours its own prior too


class _Geom:
    aux = {'BC': np.zeros(4), 'BW': np.zeros(4), 'u': np.zeros(4)}
    wavelength_key = 'BC'
    width_key = 'BW'


def test_config_form():
    g = _Geom()
    c = build_coefficient({'variable': 'wavelength', 'function': 'gaussian', 'center': 3.29,
                           'width': 'bandwidth', 'intrinsic_var': 2.89e-4}, g)
    assert c.variable == 'BC' and c.function.sigma_source.fwhm_key == 'BW'
    c = build_coefficient({'variable': 'u', 'function': 'template', 'x': [0, 1], 'y': [0, 2]}, g)
    assert np.allclose(c.evaluate({'u': np.array([0.25, 0.5])}), [0.5, 1.0])
    c = build_coefficient({'variable': 'u', 'function': 'linear', 'center': 0.5, 'halfwidth': 0.5}, g)
    assert np.allclose(c.evaluate({'u': np.array([0.0, 1.0])}), [-1.0, 1.0])
    c = build_coefficient({'variable': ['u', 'wavelength'], 'function': 'tests.test_sky_coefficients:two_var',
                           'params': {'scale': 3.0}}, g)
    assert c.variable == ('u', 'BC') and np.allclose(c.evaluate({'u': np.ones(2), 'BC': np.full(2, 2.0)}), 6.0)
    cat = {'mine': lambda center=None: Coefficient('BC', GaussianProfile(center_um=center or 1.0, sigma_um=0.1))}
    assert build_coefficient({'catalog': 'mine', 'center': 2.0}, g, catalog=cat).function.center_um == 2.0
    assert SkyTerm(coefficient={'catalog': 'mine'}).name == 'mine' and SkyTerm().name == 'continuum'
    for bad, msg in (({'variable': 'nope', 'function': 'linear', 'center': 0, 'halfwidth': 1}, 'nope'),
                     ({'variable': 'u', 'function': 'mystery'}, 'mystery'),
                     ({'variable': ['u', 'BC'], 'function': 'linear', 'center': 0, 'halfwidth': 1}, 'one variable'),
                     ({'catalog': 'absent'}, 'absent')):
        try:
            build_coefficient(bad, g, catalog=cat)
            raise AssertionError(f'expected a ValueError for {bad}')
        except ValueError as e:
            assert msg in str(e), (msg, str(e))
    for bad in ({'sky': [{'type': 'continuum'}]}, {'sky': [{'coefficient': {'variable': 'u', 'function': 'linear'}}]}):
        try:
            ModelSpec.from_config(bad)
            raise AssertionError(f'expected a ValueError for {bad}')
        except ValueError:
            pass


# ---------------------------------------------------------------------------
# end to end: an arbitrary coefficient of a data variable, recovered by the solve
# ---------------------------------------------------------------------------
H, W = 48, 64


def _u_map():
    """The data variable: a ramp across the detector columns, 0 .. 1."""
    return np.broadcast_to(np.linspace(0.0, 1.0, W, dtype=np.float32)[None, :], (H, W)).copy()


@register_instrument('grid_with_u')
class GridWithVariable(GridInstrument):
    """The grid instrument plus one data variable ``u`` (a detector-plane map)."""

    def detector_geometry(self, inst_cfg, oversample):
        geom = super().detector_geometry(inst_cfg, oversample)
        return dataclasses.replace(geom, aux={'u': _u_map()})


def test_arbitrary_coefficient_end_to_end():
    from selfcal_scripts.runner import pipelines
    from tests.synthetic_exposures import write_exposures
    from tests.test_runner_e2e_toy import _write_config
    _state.set_progress(False)
    # (importing the coefficient function by path re-imports this module, which re-registers the class)
    assert type(get_instrument('grid_with_u')).__name__ == 'GridWithVariable'
    tmp = tempfile.mkdtemp(prefix='selfcal_coeff_')
    try:
        rng = np.random.default_rng(21)
        yy, xx = np.mgrid[0:160, 0:160].astype(np.float64)
        s1 = 1.5 * np.sin(xx / 6.0) * np.cos(yy / 7.0)                  # the second term's map (truth)
        c_det = shape(_u_map(), power=2.0, offset=0.2)                   # its coefficient per detector pixel
        exp_dir = os.path.join(tmp, 'exposures')
        write_exposures(exp_dir, 40, rng, det_shape=(H, W), chunks=(3, 4), noise=0.01, extra_term=(s1, c_det))
        out, cache = os.path.join(tmp, 'out'), os.path.join(tmp, 'cache')
        os.makedirs(cache, exist_ok=True)
        inst = {'name': 'grid_with_u', 'tag': 'Coeff', 'detector_shape': [H, W], 'chunks': [3, 4],
                'sci_ext': 1, 'dq_ext': 2}
        rcfg = _write_config(os.path.join(tmp, 'reproject.toml'), 'reproject', out, cache, instrument=inst,
                             reproject=dict(input_dirs=[exp_dir], file_pattern='/toy_exp_*_D0.fits',
                                            padding_pixels=8, max_workers=2, inner_parallel=1,
                                            reproj_func='interp', padding_percentage=0.05, replace_existing=True))
        pipelines.run(rcfg)
        model = {'scalar': True, 'mosaic': 'none',
                 'sky': [{'name': 'continuum'},
                         {'name': 'shaped', 'damp_weight': 1e-4,
                          'coefficient': {'variable': 'u', 'function': 'tests.test_sky_coefficients:shape',
                                          'params': {'power': 2.0, 'offset': 0.2}}}],
                 'offset': [{'kind': 'free', 'reg_weight': 0.1, 'mean_zero': True}]}
        # light sky damping: a strong prior on the constant term would push part of it into
        # the second term (its coefficient is positive everywhere)
        ccfg = _write_config(os.path.join(tmp, 'cal.toml'), 'cal', out, cache, scalars={'mode': 'model'},
                             instrument=inst, model=model,
                             calibration=dict(apply_mask=True, apply_weight=False, outlier_thresh=8.0,
                                              ignore_list=[], batch_size=4, offset_regularization=True,
                                              weighted_damping=True, damp_weight=1e-4, max_workers=2),
                             lsqr=dict(atol=1e-12, btol=1e-12, damp=0, iter_lim=400, precondition=True,
                                       solver='lsqr'))
        res = pipelines.run(ccfg)
        with CalFile(res.cal_paths[0]) as cal:
            assert cal.sky_names == ['continuum', 'shaped']
            got = cal.sky('shaped')
            cov = cal.sky_coverage('shaped')
        ok = (cov >= 6) & np.isfinite(got)
        assert ok.sum() > 2000, ok.sum()
        # compare on the truth grid: map each covered reference pixel back through the WCS
        from astropy.wcs import WCS
        from selfcal.geometry import wcs_helper
        ref_wcs, _ = wcs_helper.load_from_fits(os.path.join(out, 'toy_run', 'ref.fits'))
        truth = WCS(naxis=2)
        truth.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        truth.wcs.crpix = [W / 2 + 0.5, H / 2 + 0.5]
        truth.wcs.crval = [180.0, 30.0]
        truth.wcs.cdelt = [-20.0 / 3600, 20.0 / 3600]
        py, px = np.nonzero(ok)
        tx, ty = truth.world_to_pixel(ref_wcs.pixel_to_world(px, py))
        tx, ty = np.round(tx).astype(int), np.round(ty).astype(int)
        inside = (tx >= 0) & (tx < 160) & (ty >= 0) & (ty < 160)
        g_, t_ = got[py[inside], px[inside]], s1[ty[inside], tx[inside]]
        r = np.corrcoef(g_, t_)[0, 1]
        slope = np.polyfit(t_, g_, 1)[0]
        print(f"recovered vs injected map of the 'shaped' term: r = {r:.4f}, slope {slope:.3f}, "
              f"{inside.sum()} pixels")
        assert r > 0.99 and abs(slope - 1) < 0.05, (r, slope)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test_callable_coefficients()
    test_imported_function_pickles()
    test_historical_spelling_is_bit_identical()
    test_damp_weight_rule()
    test_config_form()
    test_arbitrary_coefficient_end_to_end()
    print('OK sky coefficients')
