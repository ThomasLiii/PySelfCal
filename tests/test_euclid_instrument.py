"""The Euclid instrument without data: exposure layout, chunk maps + axes, edge taper,
renderers, hooks, and the grouped / damped / exact-row terms of the model spec."""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from selfcal.instruments import get_instrument                          # noqa: E402
from selfcal.core.subframe import FrameContext                          # noqa: E402
from selfcal.models.spec import ModelSpec, SkyTerm, OffsetTerm          # noqa: E402

CFG = {'name': 'euclid', 'band': 'J', 'chunks': 4, 'strips': 6, 'tilt_strips': 5, 'det_shape': [40, 60],
       'edge_zero_px': 2, 'edge_ramp_px': 3}


def test_layout_and_geometry():
    inst = get_instrument('euclid')
    lay = inst.exposure_layout(CFG)
    assert lay.sci_ext[:3] == [1, 4, 7] and lay.dq_ext[:3] == [3, 6, 9] and len(lay.detector_ids) == 16
    assert lay.ref_use_ext == (1, 10, 37, 46) and lay.default_ignore_bits == (11, 15)
    assert inst.jobs(CFG)[0].name == 'J' and inst.frame_tag(CFG) == 'EDFN' and inst.data_unit(CFG) == 'electron'
    geom = inst.detector_geometry(CFG, 2)
    assert geom.shape == (40, 60) and geom.primary == 'grid' and not geom.aux
    g = geom.chunk_map
    assert g.n_chunks == 16 and g.axes.names == ['row', 'col'] and g.adjacency_axes == ('row', 'col')
    assert g.grid.shape == (80, 120) and g.det.dtype == np.int64
    cs, rs = geom.chunk_maps['col_strips'], geom.chunk_maps['row_strips']
    assert cs.n_chunks == 6 and rs.n_chunks == 6 and cs.axes['strip'].scan == 'x' and rs.axes['strip'].scan == 'y'
    assert (cs.det[0] == cs.det[-1]).all() and (rs.det[:, 0] == rs.det[:, -1]).all()
    t = geom.chunk_maps['col_tilt']
    assert t.n_chunks == 5 and t.spectral_axis == 'strip' and t.group_axis == 'all' and t.axes['all'].size == 1
    jg = inst.job_geometry(CFG, geom, inst.jobs(CFG)[0])
    assert jg.det_valid_weight.shape == (40, 60) and jg.grid_valid_weight.shape == (80, 120)
    assert jg.det_valid_weight[0, 0] == 0 and jg.det_valid_weight[20, 30] == 1 and 0 < jg.det_valid_weight[3, 30] < 1
    assert inst.frame_groups(['/a/exp_0000_det_03.h5', '/a/exp_0001_det_11.h5']).get('detector').tolist() == [3, 11]


def test_renderers():
    inst = get_instrument('euclid')
    cfg = dict(CFG, edge_zero_px=0)
    geom = inst.detector_geometry(cfg, 1)
    jg = inst.job_geometry(cfg, geom, inst.jobs(cfg)[0])
    rng = np.random.default_rng(0)
    r_grid = inst.offset_renderer(cfg, geom, jg, 'grid')
    off = rng.normal(size=16)
    img = r_grid(geom.chunk_map.det, off)
    assert img.shape == (40, 60) and np.isfinite(img).all()
    r_strip = inst.offset_renderer(cfg, geom, jg, 'col_strips')
    vals = np.arange(6, dtype=float)
    s = r_strip(geom.chunk_maps['col_strips'].det, vals)
    assert s.shape == (40, 60) and s[0, 0] == 0 and s[0, -1] == 5 and (s[0] == s[-1]).all()
    r_ramp = inst.offset_renderer(cfg, geom, jg, 'row_tilt')
    ramp = r_ramp(geom.chunk_maps['row_tilt'].det, np.linspace(-1, 1, 5))
    assert ramp.shape == (40, 60) and np.allclose(np.diff(ramp[:, 0]), np.diff(ramp[:, 0])[0])
    assert inst.offset_renderer(cfg, geom, jg, 'grid', render='constant') is None


def test_hooks():
    inst = get_instrument('euclid')
    hooks = inst.hooks()
    assert set(hooks) == {'star_position_mask', 'residual_mask'}
    h = hooks['star_position_mask'](positions=np.array([[5.0, 5.0]]), radius_px=2)
    ctx = FrameContext(stage='pre', file='/x/exp_0000_det_00.h5', exp_idx=0, det_idx=0,
                       ref_coords=np.array([0, 10, 0, 10]), sub_data=np.ones((10, 10)), sub_weight=np.ones((10, 10)),
                       sub_mapping=None)
    out = h(ctx)
    assert out is ctx.sub_data and ctx.sub_weight[5, 5] == 0 and ctx.sub_weight[5, 8] == 1 and ctx.sub_weight[0, 0] == 1


def test_grouped_damped_terms():
    inst = get_instrument('euclid')
    cfg = dict(CFG, edge_zero_px=0)
    geom = inst.detector_geometry(cfg, 1)
    spec = ModelSpec(sky=(SkyTerm('continuum'),), offset=(
        OffsetTerm(map='grid', kind='grouped', groups='detector', reg_weight=0.1, adjacency=('row', 'col'),
                   mean_zero=True, exact_group_rows=True),
        OffsetTerm(map='col_strips', kind='free', adjacency=(), damp=0.3),
        OffsetTerm(map='row_strips', kind='free', adjacency=(), damp=0.3)), scalar=True)
    frames = [f'/a/exp_{i:04d}_det_{i % 3:02d}.h5' for i in range(7)]
    om = spec.build_offset_model(geom, 7, frame_groups=inst.frame_groups(frames))
    assert om.num_maps == 3 and om.use_per_frame_scalar
    assert om.blocks[0].det_groups.tolist() == [0, 1, 2, 0, 1, 2, 0] and om.blocks[0].mean_offset.shape == (7,)
    assert om.blocks[1].adj_info is None and om.blocks[1].det_groups is None
    assert spec.setup_kwargs() == {'damp_offset_maps': [0.0, 0.3, 0.3], 'mean_offset_group_rows': True,
                                   'group_adjacency_maps': [0]}
    plain = ModelSpec(offset=(OffsetTerm(kind='free', reg_weight=0.1, mean_zero=True),))
    assert plain.setup_kwargs() == {}
    try:
        spec.build_offset_model(geom, 7, frame_groups={})
        raise AssertionError('expected a ValueError')
    except ValueError as e:
        assert 'detector' in str(e)
