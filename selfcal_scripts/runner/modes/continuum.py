"""Continuum mode — the baseline recipe for any instrument.

Single offset block: adjacency along the chunk map's adjacency axes (SPHEREx:
the column strips; a camera grid: both axes) + an optional soft polynomial
along ``[params].poly_axis`` when ``poly_weight`` is set + a per-frame
mean-zero anchor + a per-frame scalar; continuum-only sky; full mosaic (+ the
instrument's aux coadds, e.g. wavelength maps, when it has them).
"""
from .base import CalMode, register_mode, standard_block


@register_mode("continuum")
class Continuum(CalMode):
    mosaic_mode = "full"

    def build_offset_model(self, cfg, inst, geom, jobgeom, job, n_frames):
        from selfcal.models.offset_model import OffsetModel
        return OffsetModel([standard_block(cfg, geom, n_frames)], use_per_frame_scalar=True)
