"""Continuum mode — the baseline recipe for any instrument.

Model: one continuum sky term; one free offset term on the primary chunk map
(smoothness along the map's adjacency axes — SPHEREx: the column strips; a
camera grid: both axes — an optional soft polynomial along ``[params].poly_axis``
when ``poly_weight`` is set, a per-frame mean-zero anchor); a per-frame scalar.
Full mosaic (+ the instrument's aux coadds, e.g. wavelength maps, when it has them).
"""
from selfcal.models.spec import ModelSpec, SkyTerm

from .base import CalMode, register_mode, standard_offset_term


@register_mode("continuum")
class Continuum(CalMode):
    mosaic_mode = "full"

    def model_spec(self, cfg, inst, geom):
        return ModelSpec(sky=(SkyTerm('continuum'),), offset=(standard_offset_term(cfg, geom),), scalar=True)
