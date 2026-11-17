"""Ready-made coefficient functions for sky terms.

A sky term is a map times a coefficient ``c(v)`` of data variables
(:mod:`selfcal.models.sky_model`); any callable works as the function. These
classes are the common shapes: a Gaussian (optionally with a per-observation
width read from a second variable), a tabulated function, a linear ramp, each
evaluated as ``evaluate(x, aux)`` on the per-observation values ``x`` of its
variable, whatever the variable is (a wavelength in the spectral fits; any
other coordinate works the same). The LSQR row multiplies the pixel weight by
the coefficient (row assembly in :mod:`selfcal.core.assembly`).

Bit-identity contract (SPHEREx PAH 3.29 µm): the original inline computation
in ``selfcal.core.assembly._prep_lsqr`` for the per-pixel Gaussian was::

    obs_sigma_per = sub_BW[valid] / 2.355
    sigma_per     = np.sqrt(obs_sigma_per * obs_sigma_per + 2.890e-4)
    G_per         = np.exp(-0.5 * ((lambda_per - line_center) / sigma_per) ** 2).astype(np.float32)

``QuadratureSigma`` and ``GaussianProfile`` reproduce that exact sequence
(same op order, same float32 cast timing). The two literals, ``2.355``
(FWHM→σ) and ``2.890e-4`` (the PAH intrinsic σ², which is NOT exactly
``PAH_INTRINSIC_SIGMA_UM**2``), are carried verbatim by their SPHEREx callers.
Do not "fix" them without re-baselining the byte-equality regression goldens
(reference ``cal_*.h5`` outputs of the byte-equality gates
(``selfcal_scripts/gates/``) and compared with ``selfcal_scripts/gates/diff_cal_h5.py``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    'QuadratureSigma',
    'SpectralProfile',
    'LineProfile',
    'GaussianProfile',
    'LinearProfile',
    'TemplateProfile',
]


@dataclass(frozen=True)
class QuadratureSigma:
    """A per-observation Gaussian σ from a width variable in quadrature with an intrinsic width:
    ``σ = sqrt((aux[fwhm_key] / fwhm_to_sigma)**2 + intrinsic_var)``, in the variable's units.

    For the SPHEREx PAH line this is ``QuadratureSigma(fwhm_key='BW', fwhm_to_sigma=2.355,
    intrinsic_var=2.890e-4)`` and reproduces the legacy expression exactly.
    """

    fwhm_key: str = 'BW'
    fwhm_to_sigma: float = 2.355
    intrinsic_var: float = 0.0

    def evaluate(self, aux: dict[str, np.ndarray]) -> np.ndarray:
        """The per-observation σ from the width map ``aux[fwhm_key]`` and the intrinsic width:
        ``sqrt((aux[fwhm_key] / fwhm_to_sigma)**2 + intrinsic_var)``."""
        obs = aux[self.fwhm_key] / self.fwhm_to_sigma
        return np.sqrt(obs * obs + self.intrinsic_var)


class SpectralProfile:
    """Base class / duck-typed protocol for an arbitrary coefficient shape.

    ``evaluate(x, aux) -> float32 ndarray`` returns the dimensionless
    per-observation coefficient (template value) at each value ``x`` of the
    variable. The shape can be any analytical form or numerical template; it
    is NOT line-specific. The owning :class:`~selfcal.models.sky_model.SpectralComponent`
    solves the per-pixel amplitude that multiplies this shape.
    """

    #: aux-map keys this profile samples (besides the variable the owning
    #: component supplies). Used to declare SHM aux requirements.
    aux_requirements: tuple = ()

    def evaluate(self, x: np.ndarray, aux: dict[str, np.ndarray]) -> np.ndarray:  # pragma: no cover - interface
        """The dimensionless coefficient at each observation's value ``x`` of the variable; ``aux`` holds the
        auxiliary maps at the valid pixels, keyed by name (the keys of ``aux_requirements``)."""
        raise NotImplementedError


# Back-compat alias (the abstraction is general, not line-specific).
LineProfile = SpectralProfile


@dataclass(frozen=True)
class GaussianProfile(SpectralProfile):
    """Gaussian shape ``exp(-0.5 * ((x - center) / σ)**2)`` (an analytical template).

    ``σ`` is per observation from ``sigma_source`` (a :class:`QuadratureSigma`) when
    its width map is available in ``aux``; otherwise the scalar ``sigma`` is used.
    Preserving this fallback order (the per-observation width map when present,
    the scalar σ otherwise) is part of the bit-identity contract described in
    the module docstring.
    """

    center: float
    sigma: float = None
    sigma_source: QuadratureSigma = None

    @property
    def aux_requirements(self) -> tuple[str, ...]:
        """Aux-map keys sampled: the width-map key when ``sigma_source`` is set."""
        if self.sigma_source is not None:
            return (self.sigma_source.fwhm_key,)
        return ()

    def evaluate(self, x: np.ndarray, aux: dict[str, np.ndarray]) -> np.ndarray:
        """The Gaussian coefficient ``exp(-0.5 * ((x - center) / σ)**2)``, float32, with σ from
        ``sigma_source``'s width map when ``aux`` holds it, else the scalar ``sigma``."""
        if (self.sigma_source is not None
                and aux.get(self.sigma_source.fwhm_key) is not None):
            sigma = self.sigma_source.evaluate(aux)
        else:
            sigma = self.sigma
        return np.exp(-0.5 * ((x - self.center) / sigma) ** 2).astype(np.float32)


@dataclass(frozen=True)
class LinearProfile(SpectralProfile):
    """A normalized linear ramp ``t = (x - center) / halfwidth``.

    The coefficient of a sloped block: sky(P, x) = c0[P] + c1[P]·t(x). ``t`` spans
    [-1, 1] over a window when ``halfwidth`` is the window's half-width, so c1 is
    in the same units as c0 (the amplitude at the window's edge). Not clipped:
    it extrapolates outside the window.
    """

    center: float
    halfwidth: float

    aux_requirements: tuple = ()

    def evaluate(self, x: np.ndarray, aux: dict[str, np.ndarray]) -> np.ndarray:
        """The ramp ``(x - center) / halfwidth``, float32 (not clipped to [-1, 1])."""
        return ((np.asarray(x, dtype=np.float64) - self.center)
                / self.halfwidth).astype(np.float32)


@dataclass(frozen=True)
class TemplateProfile(SpectralProfile):
    """An arbitrary numerical template: linear interpolation of the tabulated
    ``(x, values)``, zero outside the tabulated range.

    Demonstrates that the profile interface is not Gaussian- or line-specific:
    any tabulated shape works; a Gaussian-sampled template matches
    GaussianProfile to float32 precision
    (see ``tests/test_sky_model.py::test_template_profile_matches_interp_and_gaussian``).
    """

    x: np.ndarray
    values: np.ndarray

    aux_requirements: tuple = ()

    def evaluate(self, x: np.ndarray, aux: dict[str, np.ndarray]) -> np.ndarray:
        """The interpolated template value at each ``x``, float32, zero outside the tabulated range."""
        return np.interp(x, self.x, self.values,
                         left=0.0, right=0.0).astype(np.float32)
