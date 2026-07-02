"""Line-profile models for the spectral sky components.

A :class:`LineProfile` maps per-observation wavelengths (and optional auxiliary
maps) to a dimensionless line-shape coefficient ``G`` (peak ≈ 1 at line center).
The LSQR row for a spectral component multiplies the pixel weight by ``G`` (see
:mod:`selfcal.sky_model` and the row assembly in ``selfcal.lsqr``).

Bit-identity contract (SPHEREx PAH 3.29 µm): the historical inline computation
in ``lsqr._prep_lsqr`` for the per-pixel Gaussian was::

    obs_sigma_per = sub_BW[valid] / 2.355
    sigma_per     = np.sqrt(obs_sigma_per * obs_sigma_per + 2.890e-4)
    G_per         = np.exp(-0.5 * ((lambda_per - line_center) / sigma_per) ** 2).astype(np.float32)

``QuadratureSigma`` and ``GaussianProfile`` reproduce that exact sequence
(same op order, same float32 cast timing). The two literals — ``2.355``
(FWHM→σ) and ``2.890e-4`` (PAH intrinsic σ², which is NOT exactly
``PAH_INTRINSIC_SIGMA_UM**2``) — are carried verbatim. Do not "fix" them
without re-baselining the regression goldens.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QuadratureSigma:
    """Per-pixel Gaussian σ from an instrument LSF width map ⊗ an intrinsic width.

    ``σ(pixel) = sqrt((aux[fwhm_key] / fwhm_to_sigma)**2 + intrinsic_var_um2)``.

    For SPHEREx PAH this is ``QuadratureSigma(fwhm_key='BW', fwhm_to_sigma=2.355,
    intrinsic_var_um2=2.890e-4)`` and reproduces the legacy expression exactly.
    """

    fwhm_key: str = 'BW'
    fwhm_to_sigma: float = 2.355
    intrinsic_var_um2: float = 0.0

    def evaluate(self, aux):
        obs = aux[self.fwhm_key] / self.fwhm_to_sigma
        return np.sqrt(obs * obs + self.intrinsic_var_um2)


class SpectralProfile:
    """Base class / duck-typed protocol for an arbitrary spectral shape.

    ``evaluate(lam_um, aux) -> float32 ndarray`` returns the dimensionless
    per-observation coefficient (template value) at each wavelength. The shape
    can be any analytical form or numerical template; it is NOT line-specific.
    The owning :class:`~selfcal.sky_model.SpectralComponent` solves the per-pixel
    amplitude that multiplies this shape.
    """

    #: aux-map keys this profile samples (besides the wavelength key the
    #: owning component supplies). Used to declare SHM aux requirements.
    aux_requirements: tuple = ()

    def evaluate(self, lam_um, aux):  # pragma: no cover - interface
        raise NotImplementedError


# Back-compat alias (the abstraction is general, not line-specific).
LineProfile = SpectralProfile


@dataclass(frozen=True)
class GaussianProfile(SpectralProfile):
    """Gaussian shape ``exp(-0.5 * ((λ - center)/σ)**2)`` (analytical template).

    ``σ`` is per-pixel from ``sigma_source`` (a :class:`QuadratureSigma`) when
    its width map is available in ``aux``; otherwise the scalar ``sigma_um``
    fallback is used. This mirrors the legacy ``len(sub_aux) >= 2`` branch
    (per-pixel σ from BW) vs the scalar-σ branch.
    """

    center_um: float
    sigma_um: float = None
    sigma_source: QuadratureSigma = None

    @property
    def aux_requirements(self):
        if self.sigma_source is not None:
            return (self.sigma_source.fwhm_key,)
        return ()

    def evaluate(self, lam_um, aux):
        if (self.sigma_source is not None
                and aux.get(self.sigma_source.fwhm_key) is not None):
            sigma = self.sigma_source.evaluate(aux)
        else:
            sigma = self.sigma_um
        return np.exp(-0.5 * ((lam_um - self.center_um) / sigma) ** 2).astype(np.float32)


@dataclass(frozen=True)
class LinearProfile(SpectralProfile):
    """Normalized linear ramp ``t = (λ − center_um) / halfwidth_um``.

    The coefficient of a spectrally-SLOPED continuum block: sky(P, λ) =
    cont0[P] + cont_slope[P]·t(λ). ``t`` spans [-1, 1] over the fit window when
    ``halfwidth_um`` is the window half-width, so cont_slope is in the same
    units as cont0 (amplitude at the window edge). Not clipped — mildly
    extrapolates outside the window.
    """

    center_um: float
    halfwidth_um: float

    aux_requirements: tuple = ()

    def evaluate(self, lam_um, aux):
        return ((np.asarray(lam_um, dtype=np.float64) - self.center_um)
                / self.halfwidth_um).astype(np.float32)


@dataclass(frozen=True)
class TemplateProfile(SpectralProfile):
    """Arbitrary numerical template: linear interpolation of tabulated
    ``(wave_um, values)``.

    Zero outside the tabulated range. Demonstrates that the profile interface is
    not Gaussian- or line-specific — any tabulated spectral shape works; a
    Gaussian-sampled template matches GaussianProfile to float32 precision
    (see tests).
    """

    wave_um: np.ndarray
    values: np.ndarray

    aux_requirements: tuple = ()

    def evaluate(self, lam_um, aux):
        return np.interp(lam_um, self.wave_um, self.values,
                         left=0.0, right=0.0).astype(np.float32)
