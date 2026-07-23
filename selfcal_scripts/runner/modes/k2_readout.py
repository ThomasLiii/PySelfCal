"""Readout-channel mode ("k2" = two chunk-map offset blocks): fiducial
subchannel offsets + detector-fixed readout-channel offsets (SPHEREx H2RG).
The two blocks: a free-per-frame subchannel block (subchannel adjacency) and a
detector-fixed readout block (single shared offset via det_groups=0, mean-zero
anchored). Continuum sky, x0 from A/b, mean-only mosaic over both chunk maps.
SPHEREx-specific (uses the readout-channel map)."""
import numpy as np

from .base import CalMode, register_mode


@register_mode("k2_readout")
class K2Readout(CalMode):
    mosaic_mode = "no_wav"
    requires = ("subchannel",)

    def build_offset_model(self, cfg, inst, det_inputs, ch_inputs, job, n_frames):
        from selfcal.models.offset_model import OffsetModel, OffsetBlock
        p = cfg.params
        ncol = cfg.instrument_cfg['num_col']
        det_sub = det_inputs['det_chunk_map']
        det_ro, _ = inst.readout_chunk_map(det_sub.shape)
        adj_sub = inst.subchannel_adjacency(det_sub, ncol)
        return OffsetModel([
            OffsetBlock(chunk_map=det_sub, adj_info=adj_sub,
                        reg_weight=p.get('reg_weight', 0.1)),
            OffsetBlock(chunk_map=det_ro, det_groups=np.zeros(n_frames, dtype=int),
                        mean_offset=np.zeros(n_frames),
                        reg_weight=p.get('readout_reg_weight', 0.0)),
        ], use_per_frame_scalar=False)

    def x0(self, cfg, cc):
        from selfcal.core.solution import compute_x0_from_Ab
        return compute_x0_from_Ab(cc.A, cc.b, cc.ref_shape,
                                  active_mask=getattr(cc, "active_mask", None))

    def mosaic_geometry(self, cfg, inst, det_inputs, ch_inputs):
        det_ro, _ = inst.readout_chunk_map(det_inputs['det_chunk_map'].shape)
        grid_ro = inst.upsample_chunk_map(det_ro, cfg.oversample)
        sub_render = inst.offset_render(cfg.instrument_cfg, det_inputs, ch_inputs)
        return [det_inputs['grid_chunk_map'], grid_ro], [sub_render, None]
