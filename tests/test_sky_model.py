"""Phase 3a unit tests: SkyModel / LineProfile foundation.

Pure (no SPHEREx data); runnable as ``python tests/test_sky_model.py`` or under
pytest. The load-bearing tests assert the new GaussianProfile / QuadratureSigma
reproduce the legacy inline ``G(λ)`` expression from lsqr._prep_lsqr *bitwise*,
so the Phase 3b row-assembly rewire stays byte-identical.
"""
import numpy as np

from selfcal.models.profiles import GaussianProfile, TemplateProfile, QuadratureSigma
from selfcal.models.sky_model import (SkyModel, SkyComponent, ContinuumComponent,
                               LineComponent)
from selfcal.instruments.spherex.spherex_utility import PAH_LINE_CENTER_UM, LINE_SIGMA_UM


def _legacy_G_per_pixel(lam, bw, center):
    """The exact inline computation from lsqr._prep_lsqr (per-pixel σ branch)."""
    obs_sigma_per = bw / 2.355
    sigma_per = np.sqrt(obs_sigma_per * obs_sigma_per + 2.890e-4)
    return np.exp(-0.5 * ((lam - center) / sigma_per) ** 2).astype(np.float32)


def _legacy_G_scalar(lam, center, sigma_scalar):
    """The scalar-σ branch (no BW supplied)."""
    return np.exp(-0.5 * ((lam - center) / sigma_scalar) ** 2).astype(np.float32)


def test_quadrature_sigma_literals():
    qs = QuadratureSigma(fwhm_key='BW', fwhm_to_sigma=2.355, intrinsic_var_um2=2.890e-4)
    assert qs.fwhm_to_sigma == 2.355
    assert qs.intrinsic_var_um2 == 2.890e-4
    # The intrinsic var is deliberately NOT (FWHM/2.355)**2 of PAH_INTRINSIC; it
    # is the historical literal. Guard against an accidental "cleanup".
    assert qs.intrinsic_var_um2 != (0.040 / 2.355) ** 2


def test_gaussian_profile_per_pixel_bitwise():
    rng = np.random.default_rng(0)
    lam = (3.0 + 0.6 * rng.random(5000)).astype(np.float32)
    bw = (0.05 + 0.1 * rng.random(5000)).astype(np.float32)
    center = PAH_LINE_CENTER_UM
    prof = GaussianProfile(center_um=center, sigma_um=LINE_SIGMA_UM,
                           sigma_source=QuadratureSigma(intrinsic_var_um2=2.890e-4))
    got = prof.evaluate(lam, {'BC': lam, 'BW': bw})
    ref = _legacy_G_per_pixel(lam, bw, center)
    assert got.dtype == np.float32
    assert np.array_equal(got, ref), "per-pixel Gaussian not bitwise-equal to legacy"


def test_gaussian_profile_scalar_fallback_bitwise():
    rng = np.random.default_rng(1)
    lam = (3.0 + 0.6 * rng.random(4000)).astype(np.float32)
    center, sigma = PAH_LINE_CENTER_UM, LINE_SIGMA_UM
    # No BW in aux -> scalar σ path.
    prof = GaussianProfile(center_um=center, sigma_um=sigma,
                           sigma_source=QuadratureSigma(intrinsic_var_um2=2.890e-4))
    got = prof.evaluate(lam, {'BC': lam})  # 'BW' absent
    ref = _legacy_G_scalar(lam, center, sigma)
    assert np.array_equal(got, ref), "scalar-σ Gaussian not bitwise-equal to legacy"


def test_template_profile_matches_interp_and_gaussian():
    rng = np.random.default_rng(2)
    center, sigma = PAH_LINE_CENTER_UM, LINE_SIGMA_UM
    grid = np.linspace(2.9, 3.7, 4000)
    vals = np.exp(-0.5 * ((grid - center) / sigma) ** 2)
    tmpl = TemplateProfile(wave_um=grid, values=vals)
    lam = (3.0 + 0.6 * rng.random(3000)).astype(np.float32)
    got = tmpl.evaluate(lam, {})
    assert np.array_equal(got, np.interp(lam, grid, vals, left=0.0, right=0.0).astype(np.float32))
    # A finely-sampled Gaussian template ≈ the analytic Gaussian (scalar σ).
    gauss = GaussianProfile(center_um=center, sigma_um=sigma).evaluate(lam, {})
    assert np.max(np.abs(got - gauss)) < 1e-4
    # Outside the grid -> 0.
    assert tmpl.evaluate(np.array([1.0, 10.0], dtype=np.float32), {}).tolist() == [0.0, 0.0]


def test_skymodel_continuum_only():
    sm = SkyModel.continuum_only()
    assert sm.n_blocks == 1
    assert sm.names == ['continuum']
    assert sm.aux_requirements == ()
    assert sm.components[0].coefficients({'anything': 1}) is None  # identity


def test_skymodel_continuum_plus_pah():
    sm = SkyModel.continuum_plus_pah_gaussian()
    assert sm.n_blocks == 2
    assert sm.names == ['continuum', 'pah_3p29']
    assert sm.aux_requirements == ('BC', 'BW')
    # Line component reproduces the legacy per-pixel G.
    rng = np.random.default_rng(3)
    lam = (3.0 + 0.6 * rng.random(2000)).astype(np.float32)
    bw = (0.05 + 0.1 * rng.random(2000)).astype(np.float32)
    line = sm.components[1]
    got = line.coefficients({'BC': lam, 'BW': bw})
    ref = _legacy_G_per_pixel(lam, bw, PAH_LINE_CENTER_UM)
    assert np.array_equal(got, ref)
    # Default center/σ are the SPHEREx PAH constants.
    assert line.profile.center_um == PAH_LINE_CENTER_UM
    assert line.profile.sigma_um == LINE_SIGMA_UM


def test_skymodel_rejects_duplicate_names():
    try:
        SkyModel((ContinuumComponent(), ContinuumComponent()))
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError on duplicate component names")


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")


if __name__ == '__main__':
    _run_all()
