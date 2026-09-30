"""Unit tests of the general model machinery (fast, no solve).

Data variables from every source, offset bases and coefficients, the lowering of
a ``[model]`` with variables / weight / priors, prior rows, the mean-zero anchor
of a basis, constraint-block normalisation, chunk maps with pixels outside every
chunk, the frame-file writer and header values, and the N-pass sky subtractor
reading general variables.
"""
import dataclasses
import os
import sys
import tempfile

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from selfcal.core.constraint_builders import ConstraintBlock, as_constraint_block, mean_offset_block  # noqa: E402
from selfcal.core.layout import SystemLayout                                                   # noqa: E402
from selfcal.geometry.map_helper import _parse_chunk_map, chunk_to_det                           # noqa: E402
from selfcal.models.offset_model import Basis, OffsetBlock, OffsetModel, as_basis_values        # noqa: E402
from selfcal.models.offset_structure import ChunkAxes                                           # noqa: E402
from selfcal.models.priors import TermInfo, frame_smoothness, sky_smoothness, toward_variable   # noqa: E402
from selfcal.models.sky_model import Coefficient, ImportedFunction                              # noqa: E402
from selfcal.models.spec import ModelSpec, OffsetTerm, VariableSpec                              # noqa: E402
from selfcal.models.variables import (Derived, FrameFunction, FrameObservations,                # noqa: E402
                                      ObservationVariables, VariableSet)


# ---------------------------------------------------------------------------
# functions referenced by import path
# ---------------------------------------------------------------------------
def double(x):
    return 2.0 * np.asarray(x)


def add(a, b):
    return np.asarray(a) + np.asarray(b)


def plane(x, y):
    return [x, y]


def first_row_value(frame):
    return float(np.asarray(frame.sub_data)[0, 0])


# ---------------------------------------------------------------------------
# a tiny geometry (duck-typed like DetectorGeometry)
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class _CM:
    name: str
    det: np.ndarray
    grid: np.ndarray
    axes: object
    adjacency_axes: tuple = ('row', 'col')
    spectral_axis: str = None
    group_axis: str = 'row'

    @property
    def n_chunks(self):
        return int(self.det.max()) + 1


@dataclasses.dataclass(frozen=True)
class _Geom:
    shape: tuple
    chunk_maps: dict
    primary: str = 'grid'
    aux: dict = dataclasses.field(default_factory=dict)
    wavelength_key: str = None
    width_key: str = None

    @property
    def chunk_map(self):
        return self.chunk_maps[self.primary]


def _geom():
    det = np.repeat(np.repeat(np.arange(6).reshape(2, 3), 4, axis=0), 4, axis=1).astype(np.int32)  # 8 x 12
    axes = ChunkAxes.row_major(('row', 'col'), (2, 3), ('y', 'x'))
    return _Geom(shape=det.shape, chunk_maps={'grid': _CM('grid', det, det, axes)},
                 aux={'u': np.linspace(0, 1, det.size, dtype=np.float32).reshape(det.shape)})


# ---------------------------------------------------------------------------
# variables
# ---------------------------------------------------------------------------
def test_observation_variables_every_source():
    rows, cols = np.array([0, 1, 2]), np.array([3, 4, 5])
    sub = np.arange(40, dtype=np.float32).reshape(5, 8)
    sub_mapping = np.stack([np.full((5, 8), 7.5), np.full((5, 8), 2.25)])
    sky = np.arange(100, dtype=np.float32).reshape(10, 10)
    ctx = FrameObservations(file='f.h5', index=3, pixels=(rows, cols), ref_coords=[2, 7, 1, 9],
                            sub_data=sub, sub_mapping=sub_mapping)
    obs = ObservationVariables(
        (rows, cols), index=3, ref_coords=[2, 7, 1, 9], sub_mapping=sub_mapping,
        detector={'u': sub}, frame_values={'time': 5.0, 'filter': 'J'}, sky={'s0': sky},
        layers={'var': sub * 10}, derived={'twice_u': Derived(('u',), double),
                                           'u_plus_t': Derived(('u', 'time'), add)},
        frame_functions={'corner': FrameFunction(first_row_value)}, context=ctx)
    assert np.array_equal(obs['u'], sub[rows, cols])
    assert np.array_equal(obs['var'], (sub * 10)[rows, cols])
    assert np.array_equal(obs['time'], [5.0, 5.0, 5.0])
    assert list(obs['filter']) == ['J', 'J', 'J']
    assert np.array_equal(obs['s0'], sky[rows + 2, cols + 1])
    assert np.array_equal(obs['twice_u'], 2 * sub[rows, cols])
    assert np.array_equal(obs['u_plus_t'], sub[rows, cols] + 5.0)
    assert np.array_equal(obs['corner'], [0.0, 0.0, 0.0])
    assert np.array_equal(obs['det_x'], [7.5] * 3) and np.array_equal(obs['det_y'], [2.25] * 3)
    assert np.array_equal(obs['sky_y'], rows + 2) and np.array_equal(obs['sky_x'], cols + 1)
    assert np.array_equal(obs['frame'], [3.0] * 3)
    assert 'u' in obs and 'nope' not in obs and obs.get('nope') is None
    with pytest.raises(KeyError):
        obs['nope']
    # a sky variable off the map is NaN, not a wrapped index
    far = ObservationVariables((np.array([0]), np.array([0])), ref_coords=[-5, 0, -5, 0], sky={'s0': sky})
    assert np.isnan(far['s0'][0])


def test_variable_set_rejects_duplicates_and_builtins():
    with pytest.raises(ValueError):
        VariableSet(frame={'t': np.zeros(2)}, sky={'t': np.zeros((2, 2))})
    with pytest.raises(ValueError):
        VariableSet(frame={'det_x': np.zeros(2)})
    vs = VariableSet(frame={'t': np.zeros(2)}, layers=('var',))
    assert vs.scope('t') == 'frame' and vs.scope('var') == 'layer' and vs.scope('det_x') == 'built-in'
    assert vs.merged(sky={'m': np.zeros((2, 2))}).provides('m')


def test_derived_variable_cycle_is_caught():
    obs = ObservationVariables((np.array([0]), np.array([0])),
                               derived={'a': Derived(('b',), double), 'b': Derived(('a',), double)})
    with pytest.raises(ValueError, match='depends on itself'):
        obs['a']


# ---------------------------------------------------------------------------
# bases
# ---------------------------------------------------------------------------
def test_basis_shapes():
    assert as_basis_values([np.arange(3), 2.0], 3, 2).shape == (3, 2)
    assert as_basis_values(np.ones((3, 2)), 3, 2).dtype == np.float32
    assert as_basis_values(np.arange(3), 3, 1).shape == (3, 1)
    assert np.array_equal(as_basis_values(4.0, 3, 1), np.full((3, 1), 4.0, dtype=np.float32))
    with pytest.raises(ValueError):
        as_basis_values([np.arange(3)], 3, 2)
    b = Basis(Coefficient(('x', 'y'), ImportedFunction('tests.test_general_model:plane')), 2)
    v = b.evaluate({'x': np.array([1.0, 2.0]), 'y': np.array([3.0, 4.0])})
    assert np.array_equal(v, [[1, 3], [2, 4]])
    with pytest.raises(ValueError):
        OffsetBlock(chunk_map=np.zeros((2, 2), int), template=np.zeros((1, 1)), basis=b)


def test_layout_counts_basis_columns():
    cm = np.zeros((4, 4), dtype=np.int32)
    cm[:, 2:] = 1
    b2 = Basis(Coefficient(('x', 'y'), plane), 2)
    L = SystemLayout.build((5, 5), [cm, cm], num_sky_blocks=1, num_frames=3,
                           basis_list=[b2, Basis(Coefficient('x', double), 1)])
    assert L.num_chunks_list == [4, 2]
    assert L.col_bases == [25, 25 + 12, 25 + 12 + 6]


def test_mean_zero_anchor_per_basis_function():
    ftg = np.arange(2)
    blk = mean_offset_block(0, np.zeros(2), 2, 6, ftg, [10, 22], weight=1.0, n_basis=2)
    assert blk.num_rows == 4 and blk.nnz_per_row == 3
    cols = blk.cols.reshape(4, 3)
    assert np.array_equal(cols[0], [10, 12, 14]) and np.array_equal(cols[1], [11, 13, 15])
    assert np.array_equal(cols[2], [16, 18, 20]) and np.array_equal(cols[3], [17, 19, 21])


# ---------------------------------------------------------------------------
# the spec: variables, offset functions, groupings, weight, priors
# ---------------------------------------------------------------------------
def test_spec_lowers_general_model():
    geom = _geom()
    spec = ModelSpec.from_config({
        'variables': {'time': {'header': 'MJD'}, 'hwp': {'per_frame': 'tests.test_general_model:double',
                                                         'inputs': ['time']},
                      'var': {'layer': 'variance'},
                      'psi': {'function': 'tests.test_general_model:add', 'inputs': ['u', 'time']}},
        'weight': {'variable': 'var', 'function': 'tests.test_general_model:double'},
        'sky': [{'name': 'continuum'},
                {'name': 'mod', 'coefficient': {'variable': 'psi', 'function': 'tests.test_general_model:double'}}],
        'offset': [{'kind': 'free', 'reg_weight': 0.1, 'mean_zero': True, 'name': 'chunks',
                    'basis': {'variable': ['det_x', 'det_y'], 'function': 'tests.test_general_model:plane', 'n': 2}},
                   {'kind': 'fixed', 'coefficient': {'variable': 'time'}},
                   {'map': 'detector', 'kind': 'grouped', 'groups': 'hwp'}],
        'prior': [{'term': 'chunks', 'function': 'frame_smoothness', 'variable': 'time', 'weight': 2.0}],
    })
    spec.check(geom)
    assert spec.needs_variables and spec.referenced_variables() >= {'psi', 'det_x', 'time', 'var', 'u'}
    sky = spec.build_sky_model(geom)
    assert sky.names == ['continuum', 'mod'] and sky.components[1].coefficient.variable == 'psi'
    groups = {'hwp': np.array([0., 0., 45., 45.])}
    om = spec.build_offset_model(geom, 4, frame_groups=groups)
    b0, b1, b2 = om.blocks
    assert b0.basis.n == 2
    # adjacency pairs expanded over the two functions: (c*2 + k)
    base_pairs = len(b0.adj_info[0]) // 2
    assert np.array_equal(b0.adj_info[0][:base_pairs] * 1, b0.adj_info[0][base_pairs:] - 1)
    assert b1.basis.n == 1 and np.array_equal(b1.det_groups, np.zeros(4))
    assert b2.chunk_map.max() == 0 and np.array_equal(b2.det_groups, groups['hwp'])
    assert spec.build_weight(geom).variable == 'var'
    assert spec.offset_names(geom) == ['chunks', 'grid', 'detector']
    [prior] = spec.build_priors(geom, variables=VariableSet(frame={'time': np.array([3., 1., 2., 0.])}),
                                n_frames=4)
    assert prior.term_names == ('chunks',) and prior.weight == 2.0


def test_spec_rejects_unknown_names():
    geom = _geom()
    for bad in ({'sky': [{'name': 's', 'coefficient': {'variable': 'nope'}}]},
                {'offset': [{'kind': 'grouped', 'groups': 'nope'}]},
                {'offset': [{'map': 'nope'}]},
                {'prior': [{'term': 'nope', 'function': 'sky_smoothness'}]},
                {'prior': [{'term': 'continuum', 'function': 'no_such_prior'}]}):
        with pytest.raises(ValueError):
            ModelSpec.from_config(bad).check(geom)
    with pytest.raises(ValueError):
        VariableSpec.from_config('x', {'header': 'A', 'layer': 'b'})


def test_spec_builds_variable_set(tmp_path):
    from selfcal.io.frames import standard_frame_path, write_frame
    from astropy.io import fits
    geom = _geom()
    frames = []
    for k in range(3):
        h = fits.Header()
        h['MJD'] = 100.0 + k
        p = write_frame(standard_frame_path(str(tmp_path), k, 0), np.zeros((2, 2)), [0, 2, 0, 2],
                        np.zeros((2, 2, 2)), layers={'variance': np.ones((2, 2))}, header=h)
        frames.append(p)
    np.save(tmp_path / 'sky.npy', np.ones((6, 6)))
    spec = ModelSpec.from_config({'variables': {
        'time': {'header': 'MJD'},
        'twice': {'per_frame': 'tests.test_general_model:double', 'inputs': ['time']},
        'var': {'layer': 'variance'},
        'm': {'sky': str(tmp_path / 'sky.npy')},
        'd': {'detector': 'tests.test_general_model:_det_map'},
    }})
    vs = spec.build_variables(geom, frames, ref_shape=(6, 6), frame_variables={'detector': np.zeros(3)})
    assert np.array_equal(vs.frame['time'], [100., 101., 102.]) and np.array_equal(vs.frame['twice'], [200., 202., 204.])
    assert vs.layers == ('variance',) and 'var' in vs.derived
    assert vs.sky['m'].shape == (6, 6) and vs.detector['d'].shape == geom.shape
    assert 'detector' in vs.frame


def _det_map(geom):
    return np.ones(geom.shape)


# ---------------------------------------------------------------------------
# priors
# ---------------------------------------------------------------------------
def test_prior_rows():
    cov = np.ones((3, 2, 1))
    cov[1, 1, 0] = 0                                    # group 1, chunk 1 unobserved
    t = TermInfo(name='o', kind='offset', shape=(3, 2, 1), col_base=100, coverage=cov,
                 frame_group=np.array([0, 1, 2]),
                 variables=VariableSet(frame={'time': np.array([10., 30., 20.])}), n_frames=3)
    rows, cols, vals, rhs = frame_smoothness(t, variable='time')
    # neighbours in time: group 0 (t=10) - group 2 (t=20) - group 1 (t=30); chunk 1 of group 1 skipped
    pairs = sorted(zip(cols[vals > 0].tolist(), cols[vals < 0].tolist()))
    assert pairs == [(102, 104), (104, 100), (105, 101)]
    assert t.index(2, 1, 0) == 105
    s = TermInfo(name='s', kind='sky', shape=(2, 3), col_base=0, coverage=np.ones((2, 3)))
    rows, cols, vals, rhs = sky_smoothness(s)
    assert rows.max() + 1 == 7 and np.all(rhs == 0)             # 4 horizontal + 3 vertical pairs
    tv = TermInfo(name='s', kind='sky', shape=(2, 3), col_base=6, coverage=np.ones((2, 3)),
                  variables=VariableSet(sky={'ref': np.arange(6.).reshape(2, 3)}))
    rows, cols, vals, rhs = toward_variable(tv, 'ref')
    assert np.array_equal(cols, np.arange(6) + 6) and np.array_equal(rhs, np.arange(6.))


def test_constraint_block_normalisation():
    blk = as_constraint_block(([1, 0, 0, 1], [5, 3, 3, 4], [1.0, 2.0, 0.5, -1.0], [0.0, 7.0]), 10)
    assert isinstance(blk, ConstraintBlock) and blk.num_rows == 2
    assert list(blk.nnz_per_row) == [1, 2]                  # the duplicate (0, 3) merged
    assert np.allclose(blk.data[:1], 2.5)
    assert as_constraint_block(None, 10) is None
    for bad in (([0], [10], [1.0], [0.0]), ([0], [1], [np.nan], [0.0]), ([0, 1], [1, 2], [1.0, 1.0], [0.0])):
        with pytest.raises(ValueError):
            as_constraint_block(bad, 10)


# ---------------------------------------------------------------------------
# chunk maps with pixels outside every chunk
# ---------------------------------------------------------------------------
def test_chunk_map_with_gaps():
    cm = np.array([[0, 0, -1, 1], [0, 0, -1, 1]], dtype=np.int32)
    parsed = _parse_chunk_map(cm)
    assert parsed.shape == (8, 2) and parsed.nnz == 6
    assert np.array_equal(np.asarray(parsed.sum(axis=1)).ravel(), [1, 1, 0, 1, 1, 1, 0, 1])
    off = np.array([5.0, 7.0])
    assert np.array_equal(chunk_to_det(cm, off), [[5, 5, 0, 7], [5, 5, 0, 7]])
    assert np.array_equal(chunk_to_det(cm, off, needed=np.array([1, 2, 3])), [5, 0, 7])
    full = np.zeros((2, 4), dtype=np.int32)
    assert _parse_chunk_map(full).nnz == 8                  # no gaps: the historical structure


# ---------------------------------------------------------------------------
# frames and header values
# ---------------------------------------------------------------------------
def test_write_frame_round_trip(tmp_path):
    from astropy.io import fits
    from selfcal.io.frames import frame_header_values, standard_frame_path, write_frame
    from selfcal.io.reproj import load_reproj_file
    h = fits.Header()
    h['FILTER'] = 'J'
    h['TEMP'] = 80.5
    p = write_frame(standard_frame_path(str(tmp_path), 7, 2), np.arange(6.).reshape(2, 3), [10, 12, 20, 23],
                    np.zeros((2, 2, 3)), layers={'variance': np.ones((2, 3))}, header=h)
    assert os.path.basename(p) == 'exp_0007_det_02.h5'
    got = load_reproj_file(p, ['sub_data', 'ref_coords', 'layers/variance'])
    assert got['exp_idx'] == 7 and got['det_idx'] == 2
    assert np.array_equal(got['ref_coords'], [10, 12, 20, 23]) and got['layers/variance'].shape == (2, 3)
    vals = frame_header_values([p], ['FILTER', 'TEMP', 'MISSING'])
    assert vals['FILTER'][0] == 'J' and vals['TEMP'][0] == 80.5 and np.isnan(vals['MISSING'][0])
    with pytest.raises(ValueError):
        write_frame(str(tmp_path / 'x.h5'), np.zeros((2, 3)), [0, 3, 0, 3], np.zeros((2, 2, 3)))


# ---------------------------------------------------------------------------
# the N-pass sky subtractor with general variables
# ---------------------------------------------------------------------------
def test_sky_subtractor_reads_frame_variables(tmp_path):
    import h5py
    from selfcal.core.subframe import FrameContext
    from selfcal.models.sky_model import SkyComponent, SkyModel
    from selfcal.pipeline.npass import SkySubtractor
    ref = (6, 6)
    cal = str(tmp_path / 'sky.h5')
    with h5py.File(cal, 'w') as f:
        f.create_dataset('skymap', data=np.ones(ref, np.float32))
        g = f.create_group('sky')
        g.create_dataset('continuum', data=np.ones(ref, np.float32))
        g.create_dataset('annual', data=np.full(ref, 2.0, np.float32))
    model = SkyModel((SkyComponent('continuum'),
                      SkyComponent('annual', Coefficient('time', ImportedFunction('tests.test_general_model:double')))))
    frames = ['/x/exp_0000_det_00.h5', '/x/exp_0001_det_00.h5']
    sub = SkySubtractor(cal, model, export_dir=str(tmp_path / 'exp'), aux_keys=(),
                        variables=VariableSet(frame={'time': np.array([1.0, 3.0])}), frames=frames)
    for i, name in enumerate(frames):
        ctx = FrameContext(stage='post', file=name, exp_idx=i, det_idx=0, ref_coords=[1, 3, 1, 3],
                           sub_data=np.full((2, 2), 100.0), sub_weight=np.ones((2, 2)),
                           sub_mapping=np.zeros((2, 2, 2)), sub_aux=None)
        out = sub(ctx)
        t = [1.0, 3.0][i]
        assert np.allclose(out, 100.0 - (1.0 + 2.0 * 2.0 * t)), (i, out)
        pred, on_map = sub.predict(ctx.ref_coords, (2, 2), None)        # the refit's second ask: cached
        assert np.allclose(pred, 1.0 + 4.0 * t) and on_map.all()


# ---------------------------------------------------------------------------
# the grouped clip on a variable that is not a detector map
# ---------------------------------------------------------------------------
def test_grouped_clip_on_any_variable(tmp_path):
    """Two halves of every frame sit at very different levels. Judged against the
    whole frame, the real outliers hide in the two-level spread and survive;
    grouping the clip by the built-in detector coordinate ``det_x`` judges each
    half against itself and rejects exactly the outliers."""
    from selfcal import _state
    from selfcal.core.system import setup_lsqr
    from selfcal.io.frames import standard_frame_path, write_frame
    _state.set_progress(False)
    H, W = 12, 16
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    frames = []
    rng = np.random.default_rng(0)
    for k in range(4):
        vals = np.where(x < 8, 1.0, 50.0) + rng.normal(0, 0.01, (H, W))
        vals[3, 2] += 5.0                                       # one real outlier per frame
        frames.append(write_frame(standard_frame_path(str(tmp_path), k, 0), vals, [k, k + H, 0, W],
                                  np.stack([x, y])))
    cm = np.zeros((H, W), dtype=np.int32)
    common = dict(chunk_maps=[cm], grid_valid_weight=np.ones((H, W), np.float32), apply_mask=False,
                  outlier_thresh=5.0, max_workers=1, batch_size=4)
    wide = setup_lsqr(frames, (H + 4, W), **common)
    grouped = setup_lsqr(frames, (H + 4, W), outlier_group_edges=[8.0], outlier_group_variable='det_x', **common)
    n_obs = 4 * H * W
    kept_wide = int(wide.pixel_counts[:(H + 4) * W].sum())
    kept_grouped = int(grouped.pixel_counts[:(H + 4) * W].sum())
    assert kept_wide == n_obs, kept_wide                         # frame-wide: the outliers survive
    assert kept_grouped == n_obs - 4, kept_grouped               # grouped: exactly the four outliers go
    with pytest.raises(ValueError, match='outlier_group_variable'):
        setup_lsqr(frames, (H + 4, W), outlier_group_edges=[8.0], **common)

