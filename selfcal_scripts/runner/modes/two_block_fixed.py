"""Two-block mode with a detector-fixed second block (preset ``k2_readout``).

Block A: the primary chunk map, free per frame, regularised along its spectral
axis (neighbouring spectral values in the same group; SPHEREx: subchannel
adjacency). Block B: a second chunk map (``[params].second_map``, default
``readout`` — the SPHEREx H2RG readout channels), ONE offset shared by every
frame (detector-fixed, ``det_groups = 0``), mean-zero anchored, with its own
regularisation weight ``second_reg_weight`` (historical ``readout_reg_weight``).
Continuum sky, x0 from A/b, mean-only mosaic over both maps.
"""
import numpy as np

from .base import CalMode, register_mode, param


@register_mode("two_block_fixed", "k2_readout")
class TwoBlockFixed(CalMode):
    mosaic_mode = "no_wav"
    requires = ("spectral_axis",)

    def _second(self, cfg, geom):
        name = cfg.params.get('second_map', 'readout')
        if name not in geom.chunk_maps:
            raise ValueError(f"[params].second_map = {name!r} is not a chunk map of the instrument "
                             f"({sorted(geom.chunk_maps)})")
        return geom.chunk_maps[name]

    def build_offset_model(self, cfg, inst, geom, jobgeom, job, n_frames):
        from selfcal.models.offset_model import OffsetModel, OffsetBlock
        from selfcal.models.offset_structure import adjacency_along
        p = cfg.params
        cm = geom.chunk_map
        second = self._second(cfg, geom)
        adj = adjacency_along(cm.det, cm.axes, cm.spectral_axis, step=1)
        return OffsetModel([
            OffsetBlock(chunk_map=cm.det, adj_info=adj, reg_weight=p.get('reg_weight', 0.1)),
            OffsetBlock(chunk_map=second.det, det_groups=np.zeros(n_frames, dtype=int),
                        mean_offset=np.zeros(n_frames),
                        reg_weight=param(p, 'second_reg_weight', 'readout_reg_weight', default=0.0)),
        ], use_per_frame_scalar=False)

    def x0(self, cfg, cc):
        from selfcal.core.solution import compute_x0_from_Ab
        return compute_x0_from_Ab(cc.A, cc.b, cc.ref_shape,
                                  active_mask=getattr(cc, "active_mask", None))

    def mosaic_geometry(self, cfg, inst, geom, jobgeom):
        second = self._second(cfg, geom)
        return ([geom.chunk_map.grid, second.grid],
                [inst.offset_renderer(cfg.instrument_cfg, geom, jobgeom), None])
