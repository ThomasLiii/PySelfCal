"""``sc.Offsets(exclude=sc.Overlap(...))``: a term gives up the detector patterns that other offset terms can make with
the same coefficients in every frame. The shared patterns are exact: for column constants they are the constant and the
slope along x (the same constraint as ``exclude=sc.Poly(1)``), for one offset per pixel every fixed pattern of the named
terms; a solve keeps the term free of them and still recovers the rest of a detector-fixed pattern."""
import types

import numpy as np
import pytest

import selfcal as sc
from selfcal.config import ConfigError
from selfcal.config.base import decode, encode
from selfcal.models.offset_structure import ChunkAxes
from selfcal.run.engine import chunk_map_specs, with_chunk_maps

H, W = 24, 32
BASE = 100                                   # where the excluded term's columns start in the toy layout


def pixel_chunks(geometry):
    """One chunk per detector pixel (row-major: chunk = row * width + column)."""
    h, w = geometry.shape
    return sc.Chunks(np.arange(h * w, dtype=np.int32).reshape(h, w),
                     ChunkAxes.row_major(('row', 'column'), (h, w), ('y', 'x')))


def column_chunks(geometry):
    """One chunk per detector column."""
    h, w = geometry.shape
    return sc.Chunks(np.tile(np.arange(w, dtype=np.int32), (h, 1)), ChunkAxes.row_major(('column',), (w,), ('x',)))


def gradient(det_x, det_y):
    """A gradient along each detector direction (two functions)."""
    return [(np.asarray(det_x) - (W - 1) / 2) / (W / 2), (np.asarray(det_y) - (H - 1) / 2) / (H / 2)]


def centred(temperature):
    return np.asarray(temperature, dtype=np.float64) - 80.0


def ramp(geometry):
    """A detector map: 0 to 1 along x."""
    return np.tile(np.linspace(0.0, 1.0, geometry.shape[1], dtype=np.float32), (geometry.shape[0], 1))


def square(x):
    return np.asarray(x) ** 2


def negated(x):
    return -np.asarray(x)


def _model(shared_on, exclude, others=None, variables=None):
    others = others if others is not None else [sc.Offsets('gradient', on='detector', basis=gradient, n=2),
                                                sc.Offsets('scalar', on='detector')]
    return sc.Model(chunk_maps={'pixels': sc.ChunkMap(pixel_chunks), 'columns': sc.ChunkMap(column_chunks)},
                    offsets=[sc.Offsets('shared', on=shared_on, per='all', exclude=exclude), *others],
                    variables=variables or {})


def _exclusions(model, window=None):
    """The model's column exclusions on a toy layout: the shared term's columns start at BASE."""
    cam = sc.Camera((H, W))
    spec = model.spec()
    geom = with_chunk_maps(cam.geometry(1), chunk_map_specs(cam, spec))
    n = H * W if model.offsets[0].on == 'pixels' else W
    k = len(model.offsets)
    layout = types.SimpleNamespace(det_template_arr_list=[None] * k, num_chunks_list=[n] + [2] * (k - 1),
                                   num_offset_groups_list=[1] + [5] * (k - 1),
                                   col_bases=[BASE + 10 * j + n * (j > 0) for j in range(k + 1)], ref_shape=(8, 8))
    return spec.column_exclusions(geom, layout, window=window, log=lambda *a, **kw: None)


def _projector(exclusions, n):
    """The removed part ``sum Q Q^T`` over the shared term's n columns."""
    P = np.zeros((n, n))
    for cols, Q in exclusions:
        for row in cols:
            P[np.ix_(row - BASE, row - BASE)] += Q @ Q.T
    return P


def _span_projector(vectors):
    Q = np.linalg.qr(np.stack(vectors, axis=1))[0]
    return Q @ Q.T


def test_column_constants_give_up_their_constant_and_slope_as_poly1():
    overlap = _exclusions(_model('columns', sc.Overlap('gradient', 'scalar')))
    poly = _exclusions(_model('columns', sc.Poly(1)))
    assert sum(Q.shape[1] for _, Q in overlap) == 2
    assert np.abs(_projector(overlap, W) - _projector(poly, W)).max() < 1e-10


def test_one_offset_per_pixel_gives_up_every_fixed_pattern():
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    gx, gy = gradient(x, y)
    excl = _exclusions(_model('pixels', sc.Overlap('gradient', 'scalar')))
    expected = _span_projector([np.ones(H * W), gx.ravel(), gy.ravel()])
    assert np.abs(_projector(excl, H * W) - expected).max() < 1e-10
    # only the named terms' patterns: the gradient alone has no constant
    excl = _exclusions(_model('pixels', sc.Overlap('gradient')))
    assert np.abs(_projector(excl, H * W) - _span_projector([gx.ravel(), gy.ravel()])).max() < 1e-10


def test_the_window_limits_the_pixels():
    window = np.zeros((H, W))
    window[:, : W // 2] = 1.0
    excl = _exclusions(_model('pixels', sc.Overlap('gradient', 'scalar')), window=window)
    (cols, Q), = excl
    reach = np.zeros((H, W))
    reach[:, : W // 2 + 1] = 1.0                       # the window and the column its observations' bilinear samples reach
    inside = np.flatnonzero(reach.ravel() > 0)
    assert np.array_equal(cols[0] - BASE, inside)
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    gx, gy = gradient(x, y)
    expected = _span_projector([np.ones(inside.size), gx.ravel()[inside], gy.ravel()[inside]])
    assert np.abs(Q @ Q.T - expected).max() < 1e-10


def test_detector_maps_and_derived_variables_make_fixed_patterns():
    variables = {'ramp': sc.DetectorMap(ramp), 'ramp2': sc.Derived(square, of='ramp')}
    others = [sc.Offsets('curve', on='detector', times=sc.Function(negated, of='ramp2'))]
    excl = _exclusions(_model('pixels', sc.Overlap('curve'), others=others, variables=variables))
    r = np.tile(np.linspace(0.0, 1.0, W, dtype=np.float32).astype(np.float64), (H, 1)).ravel()
    assert np.abs(_projector(excl, H * W) - _span_projector([r ** 2])).max() < 1e-10


def test_a_term_reading_frame_data_has_no_fixed_pattern():
    others = [sc.Offsets('thermal', on='detector', times=centred)]
    model = _model('pixels', sc.Overlap('thermal'), others=others, variables={'temperature': sc.Header('TEMP')})
    with pytest.raises(ValueError, match='fixed'):
        _exclusions(model)


def test_the_option_lowers_and_is_validated():
    t = sc.Offsets('shared', on='pixels', per='all', exclude=sc.Overlap('gradient', 'scalar'))
    assert t.lower()['exclude_overlap'] == ['gradient', 'scalar']
    assert repr(sc.Overlap('gradient', 'scalar')) == "Overlap('gradient', 'scalar')"
    assert decode(encode(sc.Overlap('a', 'b'))) == sc.Overlap('a', 'b')
    for bad in ((), ('a', 'a'), ('',)):
        with pytest.raises(ConfigError):
            sc.Overlap(*bad)
    with pytest.raises(ConfigError):
        sc.Offsets(on='detector', basis=gradient, n=2, exclude=sc.Overlap('scalar'))
    with pytest.raises(ConfigError, match='not another offset term'):
        _model('pixels', sc.Overlap('gradient', 'nothing'))
    with pytest.raises(ConfigError, match='not another offset term'):
        _model('pixels', sc.Overlap('shared'))


def test_a_per_pixel_pattern_without_the_per_frame_terms_patterns_end_to_end(tmp_path):
    """The camera toy of :mod:`test_any_telescope` with a detector-fixed pattern of one offset per pixel, shared by
    every frame, and a free gradient and level per frame: with ``exclude=Overlap('gradient', 'scalar')`` the solved
    pattern carries no constant and no gradient, exactly, and the pattern's remaining structure is recovered."""
    from test_any_telescope import COADD, FIT, NUMERICS, PIX, REF, TWO_WORKERS, agreement, plane, true_sky, write_exposure
    from test_any_telescope import H as TH
    from test_any_telescope import W as TW
    rng = np.random.default_rng(7)
    sky = true_sky()
    y, x = np.mgrid[0:TH, 0:TW].astype(np.float64)
    gx, gy = plane(x, y)
    pattern = rng.normal(0, 0.05, (TH, TW)) + 0.4 * gx + 0.03          # plus a gradient and a level the frames' terms hold
    n_exp = 48
    for k in range(n_exp):
        oy, ox = rng.integers(0, REF - TH), rng.integers(0, REF - TW)
        g = rng.normal(0, 0.3, 2)
        img = sky[oy:oy + TH, ox:ox + TW] + pattern + g[0] * gx + g[1] * gy + rng.normal(0, 1.0) + rng.normal(0, 0.01, (TH, TW))
        write_exposure(tmp_path / 'exposures' / f'toy_exp_{k:03d}.fits', img, ox, oy, {})
    cam = sc.Camera((TH, TW), tag='Toy')
    field = sc.Field(tmp_path / 'toy_run', cam, PIX, compute=TWO_WORKERS)
    field.reproject(tmp_path / 'exposures' / 'toy_exp_*.fits', method='interp', padding=8)
    model = sc.Model(chunk_maps={'pixels': sc.ChunkMap(pixel_chunks)}, sky=[sc.Sky(damping=1e-4)],
                     offsets=[sc.Offsets('pattern', on='pixels', per='all', exclude=sc.Overlap('gradient', 'scalar')),
                              sc.Offsets('gradient', on='detector', basis=plane, n=2),
                              sc.Offsets('scalar', on='detector')])
    result = field.calibrate(sc.Recipe(model, fit=FIT, coadd=COADD, numerics=NUMERICS), jobs=sc.Job('All'))
    with result.cal() as cal:
        got = np.asarray(cal.offsets[0][0], dtype=np.float64)
    Q = np.linalg.qr(np.stack([np.ones(TH * TW), gx.ravel(), gy.ravel()], axis=1))[0]
    moments = Q.T @ got
    assert np.abs(moments).max() < 1e-6 * np.abs(got).max(), moments
    truth = pattern.ravel()
    r, slope = agreement(got, truth - Q @ (Q.T @ truth), demean=False)
    print(f"per-pixel pattern without the frames' patterns: r = {r:.4f}, slope {slope:.3f}")
    assert r > 0.99 and abs(slope - 1) < 0.05, (r, slope)
