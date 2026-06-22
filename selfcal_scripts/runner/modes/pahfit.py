"""PAHfit spectral mode — continuum + per-pixel PAH 3.29um line amplitude.

Same offset structure as continuum (column poly), but a 2-component sky
(continuum + Gaussian line) that needs per-pixel wavelength from the instrument
(BC/BW), so it declares ``requires=("wavelength",)``. The line-Fisher mask
threshold is recorded on the cal as an attribute (read-time, non-destructive).
"""
from .base import register_mode, _single_col_poly_block
from .continuum import Continuum


@register_mode("pahfit")
class PAHfit(Continuum):
    mosaic_mode = "full"
    requires = ("wavelength",)

    def build_sky_model(self, cfg, inst, det_inputs):
        from selfcal.models.sky_model import SkyModel
        p = cfg.params
        return SkyModel.continuum_plus_pah_gaussian(
            p.get('line_center'), p.get('line_sigma'))

    def det_aux(self, cfg, inst, det_inputs):
        return inst.aux(det_inputs)

    def configure(self, cfg, cc):
        cc.line_fisher_threshold = cfg.params.get('line_fisher_threshold', 10.0)
