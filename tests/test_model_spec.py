"""The model spec: a ``[model]`` table (the form :meth:`~selfcal.models.model.Model.lower` gives the
engine) lowers to the solver objects of the presets it spells out, and a model with a soft
polynomial along a chunk axis runs end to end on the built-in camera."""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import selfcal as sc  # noqa: E402
from selfcal import _state  # noqa: E402
from selfcal.models.spec import ModelSpec, OffsetTerm, SkyTerm  # noqa: E402
from tests.synthetic_exposures import write_exposures  # noqa: E402

CAMERA = sc.Camera((40, 60), chunks=(4, 6), dq_ext=2, tag='Spec')


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
    geom = CAMERA.geometry(1)
    n = 9
    # the continuum preset == a [model] table with a continuum term and one free offset term
    preset = sc.continuum(smooth=0.2, poly_prior=sc.Poly(1, along='row', weight=0.5)).spec()
    spec = ModelSpec.from_config({
        'sky': [{'name': 'continuum'}],
        'offset': [{'kind': 'free', 'reg_weight': 0.2, 'adjacency': ['row', 'col'], 'mean_zero': True,
                    'poly': [{'axis': 'row', 'degree': 1, 'weight': 0.5}]}],
        'scalar': True})
    _same_kwargs(preset.build_offset_model(geom, n), spec.build_offset_model(geom, n))
    assert [c.name for c in preset.build_sky_model(geom).components] == \
        [c.name for c in spec.build_sky_model(geom).components]
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
    # a coefficient reads a data variable the instrument must provide: the camera has none
    spec3 = ModelSpec(sky=(SkyTerm(), SkyTerm('l', coefficient={'variable': 'wavelength', 'function': 'gaussian',
                                                                  'center': 1.0, 'sigma': 0.1})))
    assert spec3.has_coefficients
    for call in (lambda: spec3.check(geom), lambda: spec3.build_sky_model(geom)):
        try:
            call()
            raise AssertionError('expected a ValueError')
        except ValueError as e:
            assert 'wavelength' in str(e)


def test_a_model_with_a_soft_polynomial_runs(tmp_path):
    _state.set_progress(False)
    write_exposures(str(tmp_path / 'exposures'), 10, np.random.default_rng(5), det_shape=(40, 60), chunks=(4, 6))
    field = sc.Field(tmp_path / 'out' / 'model', CAMERA, 20.0, compute=sc.Compute(str(tmp_path / 'cache'), workers=2))
    field.reproject(str(tmp_path / 'exposures' / 'toy_exp_*_D0.fits'), method='interp', padding=8)
    model = sc.Model(offsets=[sc.Offsets(smooth=0.1, mean_zero=True, poly_prior=sc.Poly(1, along='col', weight=0.5))])
    result = field.calibrate(sc.Recipe(model, fit=sc.Fit(60, tolerance=1e-8), coadd=sc.Coadd(clip=3.0,
                                                                                          instrument_maps=False),
                                       numerics=sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4)))
    assert len(result.cal_paths) == 1 and os.path.exists(result.cal_paths[0]) and len(result.mosaic_paths) == 1


if __name__ == '__main__':
    import tempfile
    from pathlib import Path
    test_model_table_equals_presets()
    test_a_model_with_a_soft_polynomial_runs(Path(tempfile.mkdtemp()))
    print('OK model spec')
