"""Calibration-mode contract + registry.

A *mode* is the calibration recipe: how to assemble the offset model (including
its adjacency), the sky model, the x0 init, and the mosaic geometry, from a run
config + an instrument's geometry. The generic engine talks only to this
interface and resolves modes by name through ``get_mode`` — it never references a
specific mode. Adding a calibration variant is a new module here with an
``@register_mode`` class; nothing else in the runner changes.
"""
import numpy as np

_MODE_REGISTRY = {}


def register_mode(name):
    def deco(cls):
        cls.name = name
        _MODE_REGISTRY[name] = cls
        return cls
    return deco


def get_mode(name):
    if name not in _MODE_REGISTRY:
        raise ValueError(
            f"unknown mode {name!r}; available: {sorted(_MODE_REGISTRY)}")
    return _MODE_REGISTRY[name]()


def available_modes():
    return sorted(_MODE_REGISTRY)


class CalMode:
    """Base class with the defaults the simplest (continuum) mode needs.

    Subclass + ``@register_mode("name")``; override only what differs. Class attrs:
      pipeline    "cal" (the standard per-job loop) | "tiled" (TiledCalibration)
      mosaic_mode "full" (mosaic + wavelength append) | "no_wav" (mosaic only) |
                  "none" (skip mosaic)
      requires    capability tags the instrument must provide (e.g. "wavelength").
    """

    name = None
    pipeline = "cal"
    mosaic_mode = "full"
    requires = ()

    def build_offset_model(self, cfg, inst, det_inputs, ch_inputs, job, n_frames):
        raise NotImplementedError

    def build_sky_model(self, cfg, inst, det_inputs):
        from selfcal.models.sky_model import SkyModel
        return SkyModel.continuum_only()

    def det_aux(self, cfg, inst, det_inputs):
        return None

    def x0(self, cfg, cc):
        from selfcal.core.solution import compute_x0_scalar_only
        return compute_x0_scalar_only(
            cc.A, cc.b, cc.ref_shape,
            scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
            num_sky_blocks=cc.num_sky_blocks,
            active_mask=getattr(cc, "active_mask", None))

    def configure(self, cfg, cc):
        pass

    def mosaic_geometry(self, cfg, inst, det_inputs, ch_inputs):
        """(chunk_maps, det_offset_funcs) for make_mosaic. Default: one grid chunk
        map rendered by the instrument's smooth offset renderer."""
        return ([det_inputs['grid_chunk_map']],
                [inst.offset_render(cfg.instrument_cfg, det_inputs, ch_inputs)])


def _single_col_poly_block(cfg, inst, det_inputs, n_frames):
    """The shared single-block offset model: column adjacency + (optional) linear
    column poly-constraint + per-frame mean-zero anchor + per-frame scalar. Used
    by the continuum and pahfit modes (identical offset structure).

    The column poly-constraint is applied iff ``params.poly_weight`` is set
    (production configs set it to 0.5); omit it for a plain adjacency-only block.
    """
    from selfcal.models.offset_model import OffsetModel, OffsetBlock
    p = cfg.params
    ncol = cfg.instrument_cfg['num_col']
    cm = det_inputs['det_chunk_map']
    adj = inst.column_adjacency(cm, ncol)
    poly_group = None
    if p.get('poly_weight') is not None:
        deg = p.get('poly_degree', 1)
        if ncol >= deg + 2:
            pc, ps = inst.column_poly_chains(cm, ncol, degree=deg)
            poly_group = [{'chains': pc, 'stencil': ps, 'weight': p['poly_weight']}]
        # else: column poly needs >= degree+2 columns; vacuous at this NumCol -> skip.
    return OffsetModel([
        OffsetBlock(chunk_map=cm, adj_info=adj, reg_weight=p.get('reg_weight', 0.1),
                    poly_constraints=poly_group, mean_offset=np.zeros(n_frames)),
    ], use_per_frame_scalar=True)
