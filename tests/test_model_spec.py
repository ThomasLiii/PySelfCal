"""The model spec: a ``[model]`` table lowers to the same solver objects as the
named presets, and ``mode = "model"`` runs end to end on the grid instrument."""
import os
import shutil
import sys
import tempfile

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from selfcal import _state                                      # noqa: E402
from selfcal.instruments import get_instrument                  # noqa: E402
from selfcal.models.spec import ModelSpec, SkyTerm, OffsetTerm  # noqa: E402
from selfcal.run.modes import get_mode               # noqa: E402
from selfcal.run import pipelines                    # noqa: E402
from tests.synthetic_exposures import write_exposures           # noqa: E402
from tests.test_runner_e2e_toy import _write_config             # noqa: E402

GRID = {'name': 'grid', 'tag': 'Spec', 'detector_shape': [40, 60], 'chunks': [4, 6], 'sci_ext': 1, 'dq_ext': 2}


class _Cfg:
    def __init__(self, params, model=None):
        self.params = params
        self.model = model or {}
        self.instrument_cfg = GRID


def _same_kwargs(a, b):
    ka, kb = a.to_setup_kwargs(), b.to_setup_kwargs()
    assert set(ka) == set(kb)
    for k in ka:
        x, y = ka[k], kb[k]
        if k == 'use_per_frame_scalar':
            assert x == y
            continue
        assert len(x) == len(y)
        for u, v in zip(x, y):
            _same_value(u, v, k)


def _same_value(u, v, where):
    if isinstance(u, np.ndarray) or isinstance(v, np.ndarray):
        assert np.array_equal(np.asarray(u), np.asarray(v)) and np.asarray(u).dtype == np.asarray(v).dtype, where
    elif isinstance(u, (list, tuple)):
        assert len(u) == len(v), where
        for a, b in zip(u, v):
            _same_value(a, b, where)
    elif isinstance(u, dict):
        assert set(u) == set(v), where
        for k in u:
            _same_value(u[k], v[k], f'{where}.{k}')
    else:
        assert u == v, (where, u, v)


def test_model_table_equals_presets():
    inst = get_instrument('grid')
    geom = inst.detector_geometry(GRID, 1)
    n = 9
    # continuum preset == [model] with a continuum term and one free offset term
    preset = get_mode('continuum')
    cfg = _Cfg({'reg_weight': 0.2, 'poly_weight': 0.5, 'poly_degree': 1})
    spec = ModelSpec.from_config({
        'sky': [{'name': 'continuum'}],
        'offset': [{'kind': 'free', 'reg_weight': 0.2, 'adjacency': ['row', 'col'], 'mean_zero': True,
                    'poly': [{'axis': 'row', 'degree': 1, 'weight': 0.5}]}],
        'scalar': True})
    _same_kwargs(preset.build_offset_model(cfg, inst, geom, None, None, n), spec.build_offset_model(geom, n))
    assert preset.build_sky_model(cfg, inst, geom) == spec.build_sky_model(geom, inst.coefficient_catalog())
    assert spec.x0_kind == 'scalar_only' and not spec.has_coefficients
    spec.check(geom)
    # a detector-fixed second term lowers to a shared (det_groups = 0) mean-zero block
    spec2 = ModelSpec(sky=(SkyTerm('continuum'),),
                      offset=(OffsetTerm(kind='free', reg_weight=0.1, mean_zero=True),
                              OffsetTerm(kind='fixed', reg_weight=0.0, adjacency=(), mean_zero=True)),
                      scalar=False)
    om = spec2.build_offset_model(geom, n)
    assert om.num_maps == 2 and not om.use_per_frame_scalar and spec2.x0_kind == 'from_Ab'
    b = om.blocks[1]
    assert b.adj_info is None and np.array_equal(b.det_groups, np.zeros(n, dtype=int)) and b.mean_offset.shape == (n,)
    # a coefficient reads a data variable the instrument must provide: the grid instrument has none
    spec3 = ModelSpec(sky=(SkyTerm(), SkyTerm('l', coefficient={'variable': 'wavelength', 'function': 'gaussian',
                                                                  'center': 1.0, 'sigma': 0.1})))
    assert spec3.has_coefficients
    for call in (lambda: spec3.check(geom), lambda: spec3.build_sky_model(geom)):
        try:
            call()
            raise AssertionError('expected a ValueError')
        except ValueError as e:
            assert 'wavelength' in str(e)


def test_model_mode_end_to_end():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_model_')
    try:
        rng = np.random.default_rng(5)
        exp_dir = os.path.join(tmp, 'exposures')
        write_exposures(exp_dir, 10, rng, det_shape=(40, 60), chunks=(4, 6))
        out, cache = os.path.join(tmp, 'out'), os.path.join(tmp, 'cache')
        os.makedirs(cache, exist_ok=True)
        rcfg = _write_config(os.path.join(tmp, 'reproject.toml'), 'reproject', out, cache, instrument=GRID,
                             reproject=dict(input_dirs=[exp_dir], file_pattern='/toy_exp_*_D0.fits',
                                            padding_pixels=8, max_workers=2, inner_parallel=1,
                                            reproj_func='interp', padding_percentage=0.05, replace_existing=True))
        pipelines.run(rcfg)
        model = {'scalar': True, 'mosaic': 'no_wav',
                 'sky': [{'name': 'continuum'}],
                 'offset': [{'kind': 'free', 'reg_weight': 0.1, 'mean_zero': True,
                             'poly': [{'axis': 'col', 'degree': 1, 'weight': 0.5}]}]}
        ccfg = _write_config(os.path.join(tmp, 'cal.toml'), 'cal', out, cache, scalars={'mode': 'model'},
                             instrument=GRID, model=model)
        assert ccfg.mode == 'model' and ccfg.model['offset'][0]['poly'][0]['axis'] == 'col'
        res = pipelines.run(ccfg)
        assert len(res.cal_paths) == 1 and os.path.exists(res.cal_paths[0]) and len(res.mosaic_paths) == 1
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test_model_table_equals_presets()
    test_model_mode_end_to_end()
    print('OK model spec')
