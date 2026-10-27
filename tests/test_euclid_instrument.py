"""The Euclid instrument without data: exposure layout, its default chunk maps + axes, edge taper,
the maps' renderers, hooks, and the grouped / damped / exact-row terms of the model spec."""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import pytest  # noqa: E402

import selfcal as sc  # noqa: E402
from selfcal.config import ConfigError  # noqa: E402
from selfcal.core.subframe import FrameContext  # noqa: E402
from selfcal.instruments.euclid import Euclid, StarMask  # noqa: E402
from selfcal.models.spec import ModelSpec, OffsetTerm, SkyTerm  # noqa: E402
from selfcal.run.engine import RunContext, chunk_map_specs, with_chunk_maps  # noqa: E402
from selfcal.run.lower import lower  # noqa: E402

INST = Euclid(band='J', chunks=4, strips=6, tilt_strips=5, det_shape=(40, 60), edge_zero_px=2, edge_ramp_px=3)
JOB = sc.Job('J')                                   # a Euclid run's job: the band


def geometry(inst, oversample):
    """The detector geometry with Euclid's default chunk maps evaluated on it, as the run engine builds it."""
    return with_chunk_maps(inst.geometry(oversample), chunk_map_specs(inst, ModelSpec()))


def test_layout_and_geometry():
    inst = INST
    lay = inst.layout()
    assert lay.sci_ext[:3] == [1, 4, 7] and lay.dq_ext[:3] == [3, 6, 9] and len(lay.detector_ids) == 16
    assert lay.ref_use_ext == (1, 10, 37, 46) and lay.default_ignore_bits == (11, 15)
    assert inst.product_tag == 'EDFN' and inst.unit == 'electron'
    assert inst.geometry(2).chunk_maps == {} and all(not m.tag for m in inst.default_chunk_maps().values())
    geom = geometry(inst, 2)
    assert geom.shape == (40, 60) and geom.primary == 'grid' and not geom.aux
    assert list(geom.chunk_maps) == ['grid', 'col_strips', 'row_strips', 'col_tilt', 'row_tilt']
    g = geom.chunk_map
    assert g.n_chunks == 16 and g.axes.names == ['row', 'col'] and g.adjacency_axes == ('row', 'col')
    assert g.grid.shape == (80, 120) and g.det.dtype == np.int64
    cs, rs = geom.chunk_maps['col_strips'], geom.chunk_maps['row_strips']
    assert cs.n_chunks == 6 and rs.n_chunks == 6 and cs.axes['strip'].scan == 'x' and rs.axes['strip'].scan == 'y'
    assert (cs.det[0] == cs.det[-1]).all() and (rs.det[:, 0] == rs.det[:, -1]).all()
    t = geom.chunk_maps['col_tilt']
    assert t.n_chunks == 5 and t.spectral_axis == 'strip' and t.group_axis == 'all' and t.axes['all'].size == 1
    jg = inst.job_geometry(geom, JOB)
    assert jg.det_valid_weight.shape == (40, 60) and jg.grid_valid_weight.shape == (80, 120)
    assert jg.det_valid_weight[0, 0] == 0 and jg.det_valid_weight[20, 30] == 1 and 0 < jg.det_valid_weight[3, 30] < 1
    assert inst.frame_groups(['/a/exp_0000_det_03.h5', '/a/exp_0001_det_11.h5']).get('detector').tolist() == [3, 11]


def _renderer(geom, jg, name, render=None):
    """The mosaic's renderer of a map's offsets: the one ``render`` names, else the map's default."""
    cm = geom.chunk_maps[name]
    return cm.renderers[render or cm.render](cm, jg)


def test_renderers():
    inst = INST.replace(edge_zero_px=0)
    geom = geometry(inst, 1)
    jg = inst.job_geometry(geom, JOB)
    rng = np.random.default_rng(0)
    assert [(k, m.render, sorted(m.renderers)) for k, m in geom.chunk_maps.items()] == [
        ('grid', 'spline', ['spline']), ('col_strips', 'strip', ['ramp', 'strip']),
        ('row_strips', 'strip', ['ramp', 'strip']), ('col_tilt', 'ramp', ['ramp', 'strip']),
        ('row_tilt', 'ramp', ['ramp', 'strip'])]
    r_grid = _renderer(geom, jg, 'grid')
    assert r_grid.fills_invalid
    off = rng.normal(size=16)
    img = r_grid(geom.chunk_map.det, off)
    assert img.shape == (40, 60) and np.isfinite(img).all()
    r_strip = _renderer(geom, jg, 'col_strips')
    vals = np.arange(6, dtype=float)
    s = r_strip(geom.chunk_maps['col_strips'].det, vals)
    assert s.shape == (40, 60) and s[0, 0] == 0 and s[0, -1] == 5 and (s[0] == s[-1]).all()
    r_ramp = _renderer(geom, jg, 'row_tilt')
    ramp = r_ramp(geom.chunk_maps['row_tilt'].det, np.linspace(-1, 1, 5))
    assert ramp.shape == (40, 60) and np.allclose(np.diff(ramp[:, 0]), np.diff(ramp[:, 0])[0])
    # through the engine: the map's default, the one a term names, 'constant' (None: constant over each chunk)
    model = sc.Model(offsets=[sc.Offsets(), sc.Offsets(on='col_strips', render='ramp', smooth_along=()),
                              sc.Offsets(on='row_tilt', render='constant', smooth_along=())])
    (spec,) = lower(sc.Field('/tmp/euclid_field', inst, 20.0), sc.Recipe(model, coadd=None), jobs=JOB)
    ctx = RunContext.build(spec)
    _, funcs = ctx.mosaic_geometry(ctx.job_geometry(JOB))
    assert funcs[0].func.__name__ == 'make_grid_offset_map' and funcs[1].func.__name__ == 'make_strip_offset_map'
    assert funcs[2] is None
    (spec,) = lower(sc.Field('/tmp/euclid_field', inst, 20.0),
                    sc.Recipe(sc.Model(offsets=[sc.Offsets(render='ramp')]), coadd=None), jobs=JOB)
    ctx = RunContext.build(spec)
    with pytest.raises(ConfigError, match="has no renderer 'ramp'"):
        ctx.mosaic_geometry(ctx.job_geometry(JOB))


def test_hooks():
    h = StarMask(np.array([[5.0, 5.0]]), radius_px=2)
    ctx = FrameContext(stage='pre', file='/x/exp_0000_det_00.h5', exp_idx=0, det_idx=0,
                       ref_coords=np.array([0, 10, 0, 10]), sub_data=np.ones((10, 10)), sub_weight=np.ones((10, 10)),
                       sub_mapping=None)
    out = h(ctx)
    assert out is ctx.sub_data and ctx.sub_weight[5, 5] == 0 and ctx.sub_weight[5, 8] == 1 and ctx.sub_weight[0, 0] == 1


def test_grouped_damped_terms():
    inst = INST.replace(edge_zero_px=0)
    geom = geometry(inst, 1)
    spec = ModelSpec(sky=(SkyTerm('continuum'),), offset=(
        OffsetTerm(map='grid', kind='grouped', groups='detector', reg_weight=0.1, adjacency=('row', 'col'),
                   exclude_degree=0, exact_group_rows=True),
        OffsetTerm(map='col_strips', kind='free', adjacency=(), damp=0.3),
        OffsetTerm(map='row_strips', kind='free', adjacency=(), damp=0.3),
        OffsetTerm(map='detector', name='scalar')))
    frames = [f'/a/exp_{i:04d}_det_{i % 3:02d}.h5' for i in range(7)]
    om = spec.build_offset_model(geom, 7, frame_groups=inst.frame_groups(frames))
    assert om.num_maps == 4 and om.blocks[3].chunk_map.max() == 0
    assert om.blocks[0].det_groups.tolist() == [0, 1, 2, 0, 1, 2, 0]
    assert om.blocks[1].adj_info is None and om.blocks[1].det_groups is None
    assert spec.setup_kwargs() == {'damp_offset_maps': [0.0, 0.3, 0.3, 0.0], 'group_adjacency_maps': [0]}
    plain = ModelSpec(offset=(OffsetTerm(kind='free', reg_weight=0.1, exclude_degree=0),))
    assert plain.setup_kwargs() == {}
    try:
        spec.build_offset_model(geom, 7, frame_groups={})
        raise AssertionError('expected a ValueError')
    except ValueError as e:
        assert 'detector' in str(e)
