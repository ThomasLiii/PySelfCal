"""Continuum mode — the instrument-agnostic baseline (run_cal / d5 / damp* family).

Single offset block: column adjacency + linear column poly-constraint + per-frame
mean-zero anchor + per-frame scalar; continuum-only sky; full mosaic + (if the
instrument supports it) wavelength append. No spectral/LVF assumptions — the only
SPHEREx-specific call (column adjacency / chunk map) goes through the instrument.
"""
from .base import CalMode, register_mode, _single_col_poly_block


@register_mode("continuum")
class Continuum(CalMode):
    mosaic_mode = "full"

    def build_offset_model(self, cfg, inst, det_inputs, ch_inputs, job, n_frames):
        return _single_col_poly_block(cfg, inst, det_inputs, n_frames)
