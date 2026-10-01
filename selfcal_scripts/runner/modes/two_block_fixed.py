"""Two-block mode with a detector-fixed second term (preset ``k2_readout``).

Term A: the primary chunk map, free per frame, smoothed along its spectral
axis (neighbouring spectral values in the same group; SPHEREx: subchannel
adjacency). Term B: a second chunk map (``[params].second_map``, default
``readout`` — the SPHEREx H2RG readout channels), ONE offset shared by every
frame (detector-fixed), mean-zero anchored, with its own smoothness weight
``second_reg_weight`` (historical ``readout_reg_weight``). Continuum sky, no
per-frame scalar (x0 from the normal equations), mean-only mosaic over both maps.
"""
from selfcal.models.spec import ModelSpec, OffsetTerm, SkyTerm

from .base import CalMode, register_mode, param


@register_mode("two_block_fixed", "k2_readout")
class TwoBlockFixed(CalMode):
    mosaic_mode = "no_wav"
    requires = ("spectral_axis",)

    def model_spec(self, cfg, inst, geom):
        p = cfg.params
        cm = geom.chunk_map
        second = p.get('second_map', 'readout')
        if second not in geom.chunk_maps:
            raise ValueError(f"[params].second_map = {second!r} is not a chunk map of the instrument "
                             f"({sorted(geom.chunk_maps)})")
        if cm.spectral_axis is None:
            raise ValueError("two_block_fixed smooths the primary map along its spectral axis; "
                             "the instrument's chunk map declares none")
        return ModelSpec(
            sky=(SkyTerm('continuum'),),
            offset=(OffsetTerm(map=None, kind='free', reg_weight=p.get('reg_weight', 0.1),
                               adjacency=(cm.spectral_axis,), adjacency_step=1, mean_zero=False),
                    OffsetTerm(map=second, kind='fixed',
                               reg_weight=param(p, 'second_reg_weight', 'readout_reg_weight', default=0.0),
                               adjacency=(), mean_zero=True)),
            scalar=False, mosaic='no_wav')
