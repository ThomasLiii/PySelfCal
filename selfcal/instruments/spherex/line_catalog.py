"""SPHEREx named coefficients — entries a sky term can refer to by name
(``coefficient = { catalog = "pah_3p29" }``).

``pah_3p29``: a Gaussian of the wavelength map ``BC`` centred on the PAH 3.29 um
feature, whose per-observation sigma comes from the band-width map ``BW`` in
quadrature with the intrinsic PAH width, else the scalar ``sigma`` (this is the
historical ``pahfit`` / ``num_sky_blocks == 2`` model exactly). Tabulated
templates convolved with the measured LVF response live as package data under
``data/line_templates/`` and are used with ``function = "template"``.
"""
from __future__ import annotations

from ...models.profiles import GaussianProfile, QuadratureSigma
from ...models.sky_model import Coefficient, SkyModel
from .spherex_utility import PAH_LINE_CENTER_UM, LINE_SIGMA_UM

PAH_INTRINSIC_VAR_UM2 = 2.890e-4
FWHM_TO_SIGMA = 2.355


def pah_3p29_coefficient(center=None, sigma=None) -> Coefficient:
    """The PAH 3.29 um coefficient of the wavelength map (``BC``, width ``BW``)."""
    return Coefficient('BC', GaussianProfile(
        center_um=PAH_LINE_CENTER_UM if center is None else center,
        sigma_um=LINE_SIGMA_UM if sigma is None else sigma,
        sigma_source=QuadratureSigma(fwhm_key='BW', fwhm_to_sigma=FWHM_TO_SIGMA,
                                     intrinsic_var_um2=PAH_INTRINSIC_VAR_UM2)))


def pah_3p29(line_center=None, line_sigma=None) -> SkyModel:
    """A constant term + the ``pah_3p29`` term: the ``pahfit`` sky, as a SkyModel."""
    return SkyModel.continuum_plus_gaussian_line(
        name='pah_3p29',
        center_um=PAH_LINE_CENTER_UM if line_center is None else line_center,
        sigma_um=LINE_SIGMA_UM if line_sigma is None else line_sigma,
        width_key='BW', fwhm_to_sigma=FWHM_TO_SIGMA, intrinsic_var_um2=PAH_INTRINSIC_VAR_UM2,
        wavelength_key='BC')


# name -> factory(**overrides) -> Coefficient
CATALOG = {'pah_3p29': pah_3p29_coefficient}
