"""Tiled mode — region-partitioned PAHfit (the chunked-NEP build). Same spectral
sky as pahfit, but a single offset block with TWO poly-constraint groups (linear
column + cubic subchannel), and ``pipeline="tiled"`` so the engine routes it
through selfcal.pipeline.tiled.TiledCalibration (per-tile cal + Fisher stitch,
no mosaic) instead of the standard per-job loop."""
import numpy as np

from .base import register_mode
from .pahfit import PAHfit


@register_mode("tiled")
class Tiled(PAHfit):
    pipeline = "tiled"
    mosaic_mode = "none"
    requires = ("wavelength", "subchannel")

    def build_offset_model(self, cfg, inst, det_inputs, ch_inputs, job, n_frames):
        from selfcal.models.offset_model import OffsetModel, OffsetBlock
        p = cfg.params
        ncol = cfg.instrument_cfg['num_col']
        cm = det_inputs['det_chunk_map']
        adj = inst.column_adjacency(cm, ncol)
        pc, ps = inst.column_poly_chains(cm, ncol, degree=p.get('poly_degree', 1))
        scn, sst = inst.subchannel_poly_chains(
            p['subch_tot'], ncol, p['subch_poly_degree'],
            p['subch_poly_lo'], p['subch_poly_hi'])
        poly_groups = [
            {'chains': pc, 'stencil': ps, 'weight': p.get('poly_weight', 0.5)},
            {'chains': scn, 'stencil': sst, 'weight': p['subch_poly_weight']},
        ]
        return OffsetModel([
            OffsetBlock(chunk_map=cm, adj_info=adj, reg_weight=p.get('reg_weight', 0.1),
                        poly_constraints=poly_groups, mean_offset=np.zeros(n_frames)),
        ], use_per_frame_scalar=True)
