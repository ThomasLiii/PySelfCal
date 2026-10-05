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
    """The ``continuum`` mode fits a constant sky, the standard offset term and a per-frame scalar.

    It requires no instrument capability, so any instrument can run it, and its
    ``mosaic_mode`` is ``"full"``: the mosaic plus the instrument's aux coadds
    (such as the SPHEREx wavelength maps) when it has them. The spectral modes
    subclass it.
    """
    mosaic_mode = "full"

    def model_spec(self, cfg, inst, geom):
        """Return a spec: one ``continuum`` sky term, the standard offset term, the per-frame scalar.

        The offset term reads ``[params]`` ``reg_weight`` (default 0.1),
        ``adjacency_axes``, ``poly_weight``, ``poly_axis`` and ``poly_degree``
        (see :func:`standard_offset_term`).
        """
        return ModelSpec(sky=(SkyTerm('continuum'),), offset=(standard_offset_term(cfg, geom),), scalar=True)
