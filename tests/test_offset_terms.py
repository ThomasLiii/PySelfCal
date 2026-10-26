"""A per-frame scalar is an ordinary offset term (the term of one chunk, ``sc.Offsets('scalar', on='detector')``),
and ``exclude=sc.Poly(0)`` is the hard constraint "no constant over a group's chunks":

* the layout has the sky and the offset terms' columns, nothing else;
* ``exclude=Poly(0)`` without ``along`` takes every chunk of a map with several axes (degree 1 needs an axis);
* "the scalar term + constants with no constant" and "free constants" are the same model: the same least-squares
  fit, and the constants of the first average to zero exactly;
* a chunk's coverage fraction is taken over its pixels inside the solve's window;
* the mosaic leaves out the pixels of a chunk whose offset the coverage threshold zeroed and nothing fills;
* a cal of the older layout, with a separate ``frame_scalar``, is refused where offsets are applied.
"""
import os
import sys

import h5py
import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import selfcal as sc  # noqa: E402
from selfcal import _state  # noqa: E402
from selfcal.core.layout import SystemLayout  # noqa: E402
from selfcal.core.subframe import _render_offset  # noqa: E402
from selfcal.core.system import parse_pixel_counts_sky  # noqa: E402
from selfcal.io.calfile import CalFile  # noqa: E402
from selfcal.models.spec import ModelSpec, OffsetTerm  # noqa: E402
from tests.synthetic_exposures import write_exposures  # noqa: E402

CAMERA = sc.Camera((40, 60), chunks=(4, 6), dq_ext=2, tag='Terms')
SCALAR = sc.Offsets('scalar', on='detector')
NUMERICS = sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4)


# =================================================================== the layout and the constraint
def test_the_layout_is_the_sky_and_the_offset_terms():
    cm = np.zeros((4, 6), dtype=np.int32)
    cm[:, 3:] = 1
    L = SystemLayout.build((5, 5), [cm, np.zeros_like(cm)], num_sky_blocks=1, num_frames=3,
                           det_groups_list=[np.zeros(3, dtype=int), None])
    assert L.col_bases == [25, 25 + 2, 25 + 2 + 3] and L.total_cols == L.col_bases[-1]


def test_no_constant_takes_every_chunk_of_a_map_with_several_axes():
    geom = CAMERA.geometry(1)
    n_frames = 3
    L = SystemLayout.build((10, 10), [geom.chunk_map.det], num_sky_blocks=1, num_frames=n_frames)
    [(cols, P)] = ModelSpec(offset=(OffsetTerm(kind='free', exclude_degree=0),)).column_exclusions(geom, L)
    assert cols.shape == (n_frames, 24) and P.shape == (24, 1)
    assert np.array_equal(cols[1], 100 + 24 + np.arange(24))
    assert np.allclose(np.abs(P[:, 0]), 1 / np.sqrt(24))
    # along one axis: per line of the other (6 columns, each a line of 4 rows)
    by_row = ModelSpec(offset=(OffsetTerm(kind='free', exclude_degree=0, exclude_axis='row'),)).column_exclusions(geom, L)
    assert len(by_row) == 6 and all(c.shape == (n_frames, 4) for c, _ in by_row)
    with pytest.raises(ValueError, match='needs along='):
        ModelSpec(offset=(OffsetTerm(kind='free', exclude_degree=1),)).column_exclusions(geom, L)


def test_the_old_settings_are_gone():
    with pytest.raises(TypeError):
        sc.Offsets(mean_zero=True)
    with pytest.raises(TypeError):
        sc.Model(scalar=True)
    with pytest.raises(ValueError, match="map = 'detector'"):
        ModelSpec.from_config({'sky': [{'name': 'continuum'}], 'scalar': True})
    assert sc.continuum().offsets[-1] == SCALAR and sc.two_block().offsets[-1] != SCALAR


# =================================================================== the same model, written two ways
@pytest.fixture(scope='module')
def field(tmp_path_factory):
    _state.set_progress(False)
    tmp = tmp_path_factory.mktemp('offset_terms')
    write_exposures(str(tmp / 'exposures'), 12, np.random.default_rng(5), det_shape=(40, 60), chunks=(4, 6))
    f = sc.Field(tmp / 'out' / 'terms', CAMERA, 20.0, compute=sc.Compute(str(tmp / 'cache'), workers=2))
    f.reproject(str(tmp / 'exposures' / 'toy_exp_*_D0.fits'), method='interp', padding=8)
    return f


def _recipe(model, name, coadd=None):
    return sc.Recipe(model, fit=sc.Fit(600, clip=None, tolerance=1e-12), coadd=coadd, numerics=NUMERICS, name=name)


def test_the_scalar_beside_constants_with_no_constant_is_free_constants(field):
    """The two spellings of one model (and the scalar beside free constants, a duplicate direction) reach the same
    least-squares fit: the same true residual. (The toy's sky pixels seen by one frame leave directions the data
    do not fix, so the values themselves need not agree.) The constants with no constant average to zero over
    each frame's observed chunks, exactly."""
    two = field.calibrate(_recipe(sc.Model(sky=[sc.Sky(damping=0.0)],
                                           offsets=[sc.Offsets(exclude=sc.Poly(0)), SCALAR]), 'two'))
    one = field.calibrate(_recipe(sc.Model(sky=[sc.Sky(damping=0.0)], offsets=[sc.Offsets()]), 'one'))
    dup = field.calibrate(_recipe(sc.Model(sky=[sc.Sky(damping=0.0)], offsets=[sc.Offsets(), SCALAR]), 'dup'))
    fits_ = []
    for result in (two, one, dup):
        with CalFile(result.cal_paths[0]) as cal:
            fits_.append(float(cal.solve['true_residual']))
    assert max(fits_) - min(fits_) < 1e-5 * min(fits_)
    with CalFile(two.cal_paths[0]) as a:
        assert a.num_maps == 2 and a.offsets[1].shape[1] == 1
        c, observed = a.offsets[0].astype(np.float64), a.offset_coverage[0] > 0
    means = np.array([c[k, observed[k]].mean() for k in range(c.shape[0])])
    assert np.abs(means).max() < 1e-5 * np.abs(c).max()


# =================================================================== coverage and the mosaic
def test_a_chunks_coverage_fraction_is_over_its_pixels_in_the_window():
    cm = np.zeros((4, 4), dtype=np.int32)
    cm[:, 2:] = 1
    window = np.zeros((4, 4), dtype=bool)
    window[:2] = True                                      # the job sees the top half
    counts = np.concatenate([np.zeros(4), [2, 4]])          # 2 x 2 sky pixels, then one frame's two chunks
    _, cov, frac = parse_pixel_counts_sky(counts, (2, 2), [1], [cm])
    _, _, frac_w = parse_pixel_counts_sky(counts, (2, 2), [1], [cm], window=window)
    assert np.array_equal(cov[0], [[2, 4]]) and np.allclose(frac[0], [[2 / 8, 4 / 8]])
    assert np.allclose(frac_w[0], [[2 / 4, 4 / 4]])
    left = np.zeros((4, 4), dtype=bool)
    left[:, :2] = True                                     # the job sees chunk 0 only: chunk 1's fraction is 0
    _, _, frac_l = parse_pixel_counts_sky(counts, (2, 2), [1], [cm], window=left)
    assert np.allclose(frac_l[0], [[2 / 8, 0.0]])
    with pytest.raises(ValueError, match='differ in shape'):
        parse_pixel_counts_sky(counts, (2, 2), [1], [cm], window=np.ones((3, 3), dtype=bool))


def test_a_left_out_chunk_is_nan_whatever_draws_it():
    cm = np.array([[0, 0, 1, 1], [0, 0, 1, 1]])
    needed = np.arange(cm.size)
    off = np.array([1.5, np.nan])
    seen = []

    def spline_like(chunk_map, chunk_offset):            # a renderer drawing across chunks
        seen.append(np.asarray(chunk_offset).copy())
        return np.asarray(chunk_offset)[chunk_map] * 2.0

    for func in (None, spline_like):
        vals = _render_offset(func, cm, off, needed).reshape(cm.shape)
        assert np.all(np.isnan(vals[:, 2:])) and np.all(np.isfinite(vals[:, :2]))
    assert all(np.isfinite(o).all() for o in seen)          # the renderer never saw a NaN


def test_the_mosaic_leaves_out_what_nothing_fills(field):
    model = sc.Model(sky=[sc.Sky(damping=0.0)], offsets=[sc.Offsets(), SCALAR])
    kept = field.calibrate(_recipe(model, 'kept', coadd=sc.Coadd(None, std=False)))
    # a threshold no chunk reaches: the camera draws its maps chunk by chunk (nothing fills), so every
    # pixel of every frame leaves the coadd
    gone = field.calibrate(_recipe(model, 'gone', coadd=sc.Coadd(None, std=False, min_chunk_coverage=2.0)))
    from astropy.io import fits
    w_kept = fits.getdata(kept.mosaic_paths[0], 'MEAN_MAP_WEIGHT')
    w_gone = fits.getdata(gone.mosaic_paths[0], 'MEAN_MAP_WEIGHT')
    assert (w_kept > 0).sum() > 100 and not (w_gone > 0).any()


# =================================================================== the older layout
def test_a_cal_with_a_separate_scalar_is_refused(field, tmp_path):
    import shutil
    from selfcal.pipeline.pipeline_wrapper import Calibrator
    model = sc.Model(sky=[sc.Sky(damping=0.0)], offsets=[sc.Offsets(), SCALAR])
    cal = field.calibrate(_recipe(model, 'kept', coadd=sc.Coadd(None, std=False))).cal_paths[0]
    old = str(tmp_path / 'old.h5')
    shutil.copyfile(cal, old)
    with h5py.File(old, 'a') as f:
        f.create_dataset('frame_scalar', data=np.zeros(len(f['reproj_list'])))
    with CalFile(cal) as good:
        good.check_offsets_complete()
    with CalFile(old) as bad:
        assert bad.sky(0) is not None                        # its sky still reads
        with pytest.raises(ValueError, match='older layout'):
            bad.check_offsets_complete()
    cc = Calibrator.__new__(Calibrator)
    with CalFile(cal) as good:
        cc.ref_shape, cc.reproj_list = tuple(good.ref_shape), list(good.reproj_list)
    with pytest.raises(ValueError, match='older layout'):
        cc.load_calibration(cal_path=old)
