"""SPHEREx named sky models — the spectral lines a mode can refer to by name.

``pah_3p29`` is the continuum + PAH 3.29 um Gaussian model of the ``pahfit``
recipe: per-pixel sigma from the band-width map (``BW``) in quadrature with the
intrinsic PAH width, else the scalar ``line_sigma`` (this reproduces the
historical ``num_sky_blocks == 2`` / ``spectral_fit`` model exactly). Templates
convolved with the measured LVF response live as package data under
``data/line_templates/`` and are used through ``[[params.lines]]``
(``template_npz``), not through this catalogue.
"""
from __future__ import annotations

from ...models.sky_model import SkyModel
from .spherex_utility import PAH_LINE_CENTER_UM, LINE_SIGMA_UM

PAH_INTRINSIC_VAR_UM2 = 2.890e-4
FWHM_TO_SIGMA = 2.355


def pah_3p29(line_center=None, line_sigma=None) -> SkyModel:
    """Continuum + PAH 3.29 um Gaussian line (``pah_3p29``), the ``pahfit`` sky."""
    return SkyModel.continuum_plus_gaussian_line(
        name='pah_3p29',
        center_um=PAH_LINE_CENTER_UM if line_center is None else line_center,
        sigma_um=LINE_SIGMA_UM if line_sigma is None else line_sigma,
        width_key='BW', fwhm_to_sigma=FWHM_TO_SIGMA, intrinsic_var_um2=PAH_INTRINSIC_VAR_UM2,
        wavelength_key='BC')


CATALOG = {'pah_3p29': pah_3p29}
