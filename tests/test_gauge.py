"""The post-solve gauge (``Fit(gauge=sc.Gauge(...))``, :mod:`selfcal.core.gauge`): a model with per-chunk
constants and a per-frame slope along detector x cannot tell a sky plane from its image in those
offsets; the gauge fits the survey-mean plane (``degree=2``: and quadratic) the offsets absorbed to the
solved slopes and moves it into the sky -- on the map (the cal's sky term and every mosaic) or through
the offsets.
A toy camera whose exposures roll through many angles; the solves take seconds."""
import os
import sys

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS

_TESTS = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_TESTS)
for _p in (_REPO, _TESTS):
    if _p not in sys.path:
        sys.path.insert(0, _p)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

from test_any_telescope import NUMERICS, PIX, REF, TWO_WORKERS, H, W, true_sky  # noqa: E402

import selfcal as sc  # noqa: E402
from selfcal.core.gauge import (  # noqa: E402
    GAUGE_KEYS,
    chunk_row_offsets,
    fit_gauge,
    frame_geometry,
)
from selfcal.io.frames import frame_header_values  # noqa: E402

FIT = sc.Fit(400, clip=None, use_mask=False, tolerance=1e-12)
COADD = sc.Coadd(clip=None, use_mask=False)
BANDS = (H, 1)                       # one full-width band per detector row: a constant and a slope per row can
                                     # carry any plane exactly, so the gauge freedom is exact (SPHEREx's
                                     # subchannel bands are ~6 rows: nearly so)
CENTRE = ((W - 1) / 2.0, (H - 1) / 2.0)
HALF = W / 2.0


def xtilde(det_x):
    """The slope basis: -1 at the detector's left edge, +1 at its right."""
    return (np.asarray(det_x, dtype=np.float64) - CENTRE[0]) / HALF


def rolled_wcs(ox, oy, roll):
    """The WCS of a detector pointed at truth pixel ``(oy, ox)`` for its centre, rolled by ``roll`` radians."""
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [CENTRE[0] + 1, CENTRE[1] + 1]
    truth = WCS(naxis=2)
    truth.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    truth.wcs.crpix = [REF / 2 + 0.5, REF / 2 + 0.5]
    truth.wcs.crval = [180.0, 30.0]
    truth.wcs.cdelt = [-PIX / 3600.0, PIX / 3600.0]
    c = truth.pixel_to_world(ox + CENTRE[0], oy + CENTRE[1])
    w.wcs.crval = [c.ra.deg, c.dec.deg]
    s, co = np.sin(roll), np.cos(roll)
    w.wcs.pc = [[co, -s], [s, co]]
    w.wcs.cdelt = [-PIX / 3600.0, PIX / 3600.0]
    return w, truth


def toy_field(tmp_path, n_exp=36, seed=11):
    """``n_exp`` rolled pointings of the toy camera on a sky that is a plane + a bowl + structure; per
    frame a constant per band and a small slope along detector x (the data are the same whether the
    plane and the bowl belong to the sky or to the offsets: the gauge decides). Returns the field,
    the truth sky and the truth grid's WCS."""
    rng = np.random.default_rng(seed)
    from selfcal.instruments.grid import rect_grid_chunk_map
    yy, xx = np.mgrid[0:REF, 0:REF].astype(np.float64)
    plane = 0.004 * (xx - REF / 2) - 0.0025 * (yy - REF / 2)
    bowl = 2.5e-5 * ((xx - REF / 2) ** 2 + 0.5 * (yy - REF / 2) ** 2)
    sky = true_sky() + plane + bowl
    cm = rect_grid_chunk_map((H, W), *BANDS)
    truth = None
    for k in range(n_exp):
        oy, ox = int(rng.integers(4, REF - H - 4)), int(rng.integers(4, REF - W - 4))
        roll = rng.uniform(0, 2 * np.pi)
        w, truth = rolled_wcs(ox, oy, roll)
        # the frame's pixels on the truth grid (bilinear), through the two WCS
        dy_, dx_ = np.mgrid[0:H, 0:W]
        tx, ty = truth.world_to_pixel(w.pixel_to_world(dx_, dy_))
        from scipy.ndimage import map_coordinates
        img = map_coordinates(sky, [ty, tx], order=1, mode='nearest')
        off = rng.normal(0, 0.2, cm.max() + 1)
        img = img + (off - off.mean())[cm] + rng.normal(0, 0.02) * xtilde(dx_) + rng.normal(0, 0.02, (H, W))
        os.makedirs(tmp_path / 'exposures', exist_ok=True)
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(np.asarray(img, np.float32), header=w.to_header(), name='SCI')]
                     ).writeto(tmp_path / 'exposures' / f'toy_exp_{k:03d}.fits', overwrite=True)
    field = sc.Field(tmp_path / 'toy_run', sc.Camera((H, W), chunks=BANDS, tag='Toy'), PIX, compute=TWO_WORKERS)
    field.reproject(tmp_path / 'exposures' / 'toy_exp_*.fits', method='interp', padding=8)
    assert len(field.frames) == n_exp
    return field, sky, truth


def model():
    return sc.Model(sky=[sc.Sky(damping=0.0)],
                    offsets=[sc.Offsets(mean_zero=True), sc.Offsets('slope', times=sc.Function(xtilde, of='det_x'))])


def large_scale_error(field, got, ok, sky, truth_wcs, degree=2):
    """The rms of the degree-``degree`` polynomial fitted to ``got - sky`` over the reference pixels ``ok``."""
    ref_wcs, _ = field.reference()
    py, px = np.nonzero(ok)
    tx, ty = truth_wcs.world_to_pixel(ref_wcs.pixel_to_world(px, py))
    inside = (tx >= 1) & (tx < REF - 2) & (ty >= 1) & (ty < REF - 2)
    px, py, tx, ty = px[inside], py[inside], tx[inside], ty[inside]
    from scipy.ndimage import map_coordinates
    d = got[py, px] - map_coordinates(sky, [ty, tx], order=1)
    x, y = (px - px.mean()) / px.std(), (py - py.mean()) / py.std()
    cols = [np.ones_like(x), x, y] + ([x * x, x * y, y * y] if degree == 2 else [])
    A = np.stack(cols, 1)
    coef = np.linalg.lstsq(A, d - d.mean(), rcond=None)[0]
    return float(np.sqrt(np.mean((A @ coef) ** 2)))


# ---- unit tests ---------------------------------------------------------------------------------------
def test_fit_recovers_the_polynomial_from_synthetic_slopes():
    rng = np.random.default_rng(1)
    K, S, half, scale = 400, 12, 1020.0, 6000.0
    th = rng.uniform(0, 2 * np.pi, K)
    J = np.stack([np.stack([np.cos(th), np.sin(th)], -1), np.stack([-np.sin(th), np.cos(th)], -1)], -1)
    D = rng.uniform(-4000, 4000, (K, 2))
    dy = np.linspace(-800, 800, S)
    g, Hq = np.array([-8e-6, 4.5e-6]), np.array([[9e-10, -4e-10], [-4e-10, 4e-10]])
    h_s, hd = rng.normal(0, 2e-6, S), np.array([3e-6, -2e-6, 1e-6])
    xk, yk = J[:, :, 0], J[:, :, 1]
    b = np.empty((K, S))
    for k in range(K):
        for s in range(S):
            b[k, s] = half * (xk[k] @ (g + Hq @ D[k]) + (xk[k] @ Hq @ yk[k]) * dy[s] + h_s[s]
                              + hd[0] * D[k, 0] / scale + hd[1] * D[k, 1] / scale + hd[2] * (D[k, 0] / scale) ** 2)
    ocov = np.full((K, S), 500.0)
    g_fit, H_fit, stats = fit_gauge(b, ocov, J, D, dy, half, degree=2, position_degree=2, scale_D=scale)
    assert np.allclose(g_fit, g, rtol=1e-8) and np.allclose(H_fit, Hq, rtol=1e-6), (g_fit, H_fit)
    assert stats['residual_rms'] < 1e-12 * stats['slope_rms']


def test_the_plane_from_the_frames_mean_slopes():
    """``degree=1`` (the default): each frame's mean slope over its covered chunks is ``x_k . g + c``,
    unweighted least squares over the frames; a plane and a detector-fixed slope come back exactly, an
    uncovered chunk does not enter its frame's mean and a frame without coverage does not enter at all."""
    rng = np.random.default_rng(2)
    K, S, half = 300, 10, 1020.0
    th = rng.uniform(0, 2 * np.pi, K)
    J = np.stack([np.stack([np.cos(th), np.sin(th)], -1), np.stack([-np.sin(th), np.cos(th)], -1)], -1)
    g, c = np.array([-8e-6, 4.5e-6]), 3e-6
    b = np.repeat((half * (J[:, :, 0] @ g + c))[:, None], S, axis=1)
    ocov = np.full((K, S), 100.0)
    ocov[5] = 0                                       # a frame without coverage
    ocov[7, 3] = 0
    b[7, 3] = 1e9                                     # an uncovered chunk's slope: not data
    g_fit, H_fit, stats = fit_gauge(b, ocov, J, None, None, half)
    assert np.allclose(g_fit, g, rtol=1e-9, atol=0) and np.all(H_fit == 0), g_fit
    assert abs(stats['detector_slope'] - c) < 1e-9 * c and stats['residual_rms'] < 1e-9 * stats['slope_rms']
    assert stats['n_frames'] == K - 1 and stats['n_slopes'] == (K - 1) * S - 1 and stats['n_parameters'] == 3


def test_frame_geometry_from_the_header(tmp_path):
    """The tangent geometry and the full-header geometry agree on headers without distortion, and give
    the frame's axes on the grid: a frame rolled by 90 degrees has its x axis along the grid's y."""
    import h5py
    w, truth = rolled_wcs(30, 20, np.pi / 2)
    path = str(tmp_path / 'f.h5')
    with h5py.File(path, 'w') as f:                       # only the stored header matters to the geometry
        f.attrs['det_header'] = w.to_header().tostring().encode('utf-8')
    headers = frame_header_values([path], GAUGE_KEYS)
    J, centres = frame_geometry(headers, truth, CENTRE)
    from selfcal.core.gauge import _stored_header
    J2, centres2 = frame_geometry([_stored_header(path)], truth, CENTRE, geometry='header')
    assert np.allclose(J, J2, atol=1e-9) and np.allclose(centres, centres2, atol=1e-6)
    assert np.allclose(centres[0], [30 + CENTRE[0], 20 + CENTRE[1]], atol=1e-6)
    assert abs(J[0, 0, 0]) < 5e-3 and abs(abs(J[0, 1, 0]) - 1) < 5e-3         # det x -> grid y (meridian convergence aside)
    assert np.allclose(chunk_row_offsets(np.array([[0, 0], [1, 1], [1, 1]]), 2, 1.0), [-1.0, 0.5])


# ---- the solve -------------------------------------------------------------------------------------------
def test_gauge_moves_the_absorbed_polynomial_into_the_sky(tmp_path):
    """A known plane hidden in the offsets of a solved cal (its image in the slopes, the band constants and
    the frame scalars, taken out of the sky) is a solution the data cannot tell from the original; a
    solve warm-started there keeps it, and the gauge finds it in the slopes (on top of whatever the
    original solve had absorbed) and brings it back: the cal's sky term and the mosaic match the truth
    either way, the offsets stay as solved. (A quadratic cannot hide the same way: its curvature along
    x has no term in the offsets, so the solve corrects it; the fit's quadratic is tested above.)"""
    import shutil

    import h5py

    from selfcal.core.gauge import move_into_offsets, polynomial
    field, sky, truth = toy_field(tmp_path)
    plain = field.calibrate(sc.Recipe(model(), fit=FIT, coadd=None, numerics=NUMERICS, name='plain'))
    g_true, H_true = np.array([3e-3, -2e-3]), np.zeros((2, 2))
    hidden = str(tmp_path / 'hidden.h5')
    shutil.copyfile(plain.cal_paths[0], hidden)
    with plain.cal() as c0:
        frames = list(c0.reproj_list)
    ref_wcs, _ = field.reference()
    with h5py.File(hidden, 'r+') as f:
        sky0 = f['sky/continuum'][:].astype(np.float64)
        cov = f['sky_coverage/continuum'][:]
        c, b = f['offsets/map_0'][:].astype(np.float64), f['offsets/map_1'][:].astype(np.float64)
        ocov, scalar = f['offset_coverage/map_1'][:], f['frame_scalar'][:].astype(np.float64)
        cmap = f['chunk_maps/map_0'][:]
        J, centres = frame_geometry(frame_header_values(frames, GAUGE_KEYS), ref_wcs, CENTRE)
        m = cov > 0
        py, px = np.nonzero(m)
        p0 = np.array([px.mean(), py.mean()])
        P = polynomial(sky0.shape, g_true, H_true, p0)
        sky0[m] -= P[m] - P[m].mean()
        good = (ocov > 0) & np.isfinite(b)
        dy = chunk_row_offsets(cmap, b.shape[1], CENTRE[1])
        c_new, b_new, s_new = move_into_offsets(c, b, scalar, good, J, centres - p0, dy, -g_true, -H_true, HALF)
        f['sky/continuum'][...] = sky0.astype(np.float32)
        f['offsets/map_0'][...] = c_new.astype(f['offsets/map_0'].dtype)
        f['offsets/map_1'][...] = b_new.astype(f['offsets/map_1'].dtype)
        f['frame_scalar'][...] = s_new.astype(f['frame_scalar'].dtype)
    fit = sc.Fit(400, clip=None, use_mask=False, tolerance=1e-12, gauge=sc.Gauge('slope'))
    gauged0 = field.calibrate(sc.Recipe(model(), fit=fit, coadd=COADD, numerics=NUMERICS, name='gauged0'))
    gauged = field.calibrate(sc.Recipe(model(), fit=fit, coadd=COADD, numerics=NUMERICS, name='gauged'), start=hidden)
    with gauged0.cal() as c0g:
        gz0 = c0g.gauge
        s0g, cov0 = c0g.sky(0), c0g.sky_coverage(0)
    with gauged.cal() as c1:
        s1, cov1 = c1.sky(0), c1.sky_coverage(0)
        gz = c1.gauge
        off1 = c1.offsets[1]
    assert gz is not None and gz['applied_to'] == 'map' and gz['n_slopes'] > 0
    dg, dH = np.asarray(gz['g']) - np.asarray(gz0['g']), np.asarray(gz['H']) - np.asarray(gz0['H'])
    print(f"gauge found g = {gz['g']}, the original solve's {gz0['g']}: difference {dg} (hidden {g_true}); "
          f"H difference {np.abs(dH).max():.2e} (nothing hidden)")
    assert np.allclose(dg, g_true, rtol=0.1, atol=1e-4), (dg, g_true)
    assert np.abs(dH).max() < 1e-7, dH
    assert np.std(off1 - b_new) < 0.05 * np.std(b_new - b)                  # the warm start kept the hidden image
    ok = (cov1 >= 4) & np.isfinite(s1) & (cov0 >= 4) & np.isfinite(s0g)
    scale = float(np.std(P[m]))
    e0 = large_scale_error(field, s0g, ok, sky, truth)
    e1 = large_scale_error(field, s1, ok, sky, truth)
    print(f"large-scale error of the gauged sky term: from the original solve {e0:.4f}, from the hidden one {e1:.4f} "
          f"(the hidden polynomial's rms {scale:.4f})")
    assert e1 < 0.15 * scale and e0 < 0.15 * scale, (e0, e1, scale)
    with gauged.mosaic() as m1:
        a1 = np.asarray(m1['MEAN_MAP'], np.float64)
        w1 = np.asarray(m1.weight('MEAN_MAP'))
    f1 = large_scale_error(field, a1, (w1 > 0) & ok, sky, truth)
    print(f"large-scale error of the mosaic gauged on the map {f1:.4f}")
    assert f1 < 0.15 * scale, (f1, scale)


@pytest.mark.parametrize('degree', [1, 2])
def test_gauge_through_the_offsets_agrees_with_the_map(tmp_path, degree):
    """``apply='offsets'`` gives the same sky term and nearly the same mosaic as the map route, for the
    plane and for the plane + quadratic."""
    field, sky, truth = toy_field(tmp_path, n_exp=24, seed=5)
    on_map = field.calibrate(sc.Recipe(model(), fit=sc.Fit(400, clip=None, use_mask=False, tolerance=1e-12,
                                                             gauge=sc.Gauge('slope', degree=degree)), coadd=COADD,
                                       numerics=NUMERICS, name='map'))
    via = field.calibrate(sc.Recipe(model(), fit=sc.Fit(400, clip=None, use_mask=False, tolerance=1e-12,
                                                          gauge=sc.Gauge('slope', degree=degree, apply='offsets')),
                                    coadd=COADD, numerics=NUMERICS, name='off'))
    with on_map.cal() as c0, via.cal() as c1:
        assert np.allclose(c0.sky(0), c1.sky(0), equal_nan=True)
        assert not np.array_equal(c0.offsets[1], c1.offsets[1])          # the offsets moved
        assert c1.gauge['applied_to'] == 'offsets' and c0.gauge['degree'] == degree
        assert ('position_degree' in c0.gauge) == (degree == 2)
    with on_map.mosaic() as m0, via.mosaic() as m1:
        a0, a1 = np.asarray(m0['MEAN_MAP'], np.float64), np.asarray(m1['MEAN_MAP'], np.float64)
        w = np.asarray(m0.weight('MEAN_MAP'))
    d = (a1 - a0)[w > 0]
    print(f"mosaic via the offsets minus on the map: rms {d.std():.4e}, max {np.abs(d).max():.4e}")
    assert d.std() < 0.02 * a0[w > 0].std()


def test_a_mosaic_of_a_gauged_cal(tmp_path):
    """``field.mosaic`` of a gauged cal adds the gauge the cal recorded, with the recipe that made it (its fit
    gauges) or any other: the calibration's own mosaic, byte for byte. Only a tiled or an N-pass calibration
    refuses a gauge."""
    from selfcal.run.lower import lower
    field, _, _ = toy_field(tmp_path, n_exp=24, seed=5)
    recipe = sc.Recipe(model(), fit=sc.Fit(400, clip=None, use_mask=False, tolerance=1e-12, gauge=sc.Gauge('slope')),
                       coadd=COADD, numerics=NUMERICS, name='gauged')
    made = field.calibrate(recipe)
    again = field.mosaic(recipe.replace(name='again'), cal=made)
    ungauged_fit = field.mosaic(recipe.replace(fit=recipe.fit.replace(gauge=None), name='nogauge'), cal=made)
    with made.cal() as cal:
        assert np.any(np.asarray(cal.gauge['g']) != 0)                  # the maps differ without the gauge
    with made.mosaic() as m0, again.mosaic() as m1, ungauged_fit.mosaic() as m2:
        for name in ('MEAN_MAP', 'STD_MAP'):
            a0 = np.asarray(m0[name])
            assert np.array_equal(a0, np.asarray(m1[name])) and np.array_equal(a0, np.asarray(m2[name])), name
    for kw in (dict(tiles=sc.Tiles(grid=(2, 1))), dict(passes=sc.Passes(3))):
        with pytest.raises(sc.ConfigError, match='post-solve gauge'):
            lower(field, recipe, **kw)


def test_the_plan_names_the_gauge_and_the_weight():
    from selfcal.run.plan import describe_gauge, describe_weight
    assert describe_gauge(sc.Gauge('slope')).startswith("the sky plane that the slopes 'slope' absorbed")
    assert 'quadratic (position degree 2)' in describe_gauge(sc.Gauge('slope', degree=2))
    assert 'moved from the offsets' in describe_gauge(sc.Gauge('slope', apply='offsets'))
    assert '1/sqrt' in describe_weight(sc.shot_noise())


def test_gauge_settings_are_checked():
    with pytest.raises(sc.ConfigError):
        sc.Gauge('')
    with pytest.raises(sc.ConfigError):
        sc.Gauge('slope', degree=3)
    with pytest.raises(sc.ConfigError):
        sc.Gauge('slope', apply='sky')
    with pytest.raises(sc.ConfigError):
        sc.Gauge('slope', position_degree=1)              # the plane's fit has no position polynomial
    assert sc.Gauge('slope').degree == 1 and sc.Gauge('slope').position_degree is None
    g = sc.Gauge('slope', degree=2, position_degree=0, centre=(1, 2))
    assert g.centre == (1.0, 2.0)
    assert 'gauge' not in sc.Fit().to_dict() and 'gauge' in sc.Fit(gauge=g).to_dict()
    with pytest.raises(sc.ConfigError):
        sc.Fit(gauge='slope')
