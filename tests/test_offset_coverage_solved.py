"""The offsets' coverage over the observations the solve keeps (``offset_coverage_solved``) and the coadd that judges
chunks by it (``sc.Coadd(chunk_coverage='solved')``).

With a coverage floor (``Fit(min_observations=)``, ``Offsets(min_observations=)``) the rows of the sky pixels one
observation sees are zeroed, and an offset observed mostly there keeps only the few observations other frames share.
``offset_coverage`` still counts every observation the build made; ``offset_coverage_solved`` counts those left in
the solve. The build counts them where the floor zeroes the rows: the setup-level tests check them against the
unfloored matrix's rows (every placement, and the observation table), the end-to-end test against a frame that
overlaps the others by a few columns.
"""
import os

import numpy as np
import pytest

import selfcal as sc
from selfcal.core.system import setup_lsqr
from selfcal.io.calfile import CalFile

pytest.importorskip('numba')

from test_min_observations import _full_width  # noqa: E402
from test_observation_table import GRID, _common, _frames  # noqa: E402


def _expected(plain, floored):
    """Each offset column's entries over the rows the floored system keeps (a row with an entry of nonzero value),
    from the unfloored system's rows: the brute-force ``offset_counts_solved``."""
    A, Af = _full_width(plain), _full_width(floored)
    kept = np.diff((Af != 0).tocsr().indptr) > 0
    first = A.shape[1] - floored.offset_counts_solved.size
    Ak = A[kept]
    cols = Ak.indices[Ak.indices >= first]
    return np.bincount(cols - first, minlength=A.shape[1] - first)


FLOORS = [dict(min_observations=2),
          dict(offset_min_observations=[16], chunk_maps=[np.arange(10 * 12).reshape(10, 12).astype(np.int32)],
               det_groups_list=[np.zeros(7, int)]),
          dict(min_observations=2, offset_min_observations=[12],
               chunk_maps=[np.arange(10 * 12).reshape(10, 12).astype(np.int32)], det_groups_list=[np.zeros(7, int)])]


@pytest.mark.parametrize('floor', FLOORS)
@pytest.mark.parametrize('placement', ['serial', 'parallel', 'partitioned', 'compact'])
def test_the_solved_counts_are_the_kept_rows_entries(tmp_path, monkeypatch, floor, placement):
    """``SetupResult.offset_counts_solved``: per offset column, the entries of the rows the floor leaves; None
    without a floor. The same counts in every placement of the stored matrix and in the observation table."""
    monkeypatch.setenv('SELFCAL_SCATTER_WORKERS', '1' if placement == 'serial' else '2')
    if placement == 'partitioned':
        monkeypatch.setenv('SELFCAL_BLOCK_NNZ', '1')
        monkeypatch.setenv('SELFCAL_RMATVEC_SPLIT', '2')
    paths = _frames(tmp_path)
    kw = _common(**floor)
    spill = dict(batch_spill_dir=str(tmp_path / 'spill'))
    plain = setup_lsqr(paths, GRID, **{k: v for k, v in kw.items()
                                       if k not in ('min_observations', 'offset_min_observations')}, **spill)
    assert plain.offset_counts_solved is None
    stored = setup_lsqr(paths, GRID, **kw, **spill)
    expect = _expected(plain, stored)
    first = len(stored.pixel_counts) - stored.offset_counts_solved.size
    observed = np.asarray(stored.pixel_counts[first:])
    assert (expect <= observed).all() and (expect < observed).any()         # the floor took some of them
    if placement == 'compact':
        got = setup_lsqr(paths, GRID, matrix='compact', **kw).offset_counts_solved
    else:
        got = stored.offset_counts_solved
    np.testing.assert_array_equal(got, expect)


# ----------------------------------------------------------------------------- end to end
def tilt(det_x):
    """A gradient along the detector's x, -0.5 to 0.5."""
    from test_any_telescope import W
    return (np.asarray(det_x, dtype=np.float64) - W / 2) / W


def _field(tmp_path):
    """Twelve exposures of the toy camera (48 x 64, chunks 3 x 4) clustered at the grid's corner, and one more 72
    columns to the right: it shares four columns with the cluster, and its other columns no exposure sees."""
    from test_any_telescope import CHUNKS, H, PIX, TWO_WORKERS, W, true_sky, write_exposure
    rng = np.random.default_rng(5)
    sky = true_sky()
    places = [(int(rng.integers(0, 12)), int(rng.integers(0, 12))) for _ in range(12)] + [(4, 72)]
    for k, (oy, ox) in enumerate(places):
        img = sky[oy:oy + H, ox:ox + W] + rng.normal(0, 0.2) + rng.normal(0, 0.01, (H, W))
        write_exposure(tmp_path / 'exposures' / f'toy_exp_{k:03d}.fits', img, ox, oy, {})
    field = sc.Field(tmp_path / 'toy', sc.Camera((H, W), chunks=CHUNKS, tag='Toy'), PIX, compute=TWO_WORKERS)
    field.reproject(tmp_path / 'exposures' / 'toy_exp_*.fits', method='interp', padding=8)
    return field


def _recipe(name, coverage, floor=2):
    """The sky, a level per frame over the whole detector, and a gradient per frame and chunk (a ``times=`` term,
    subtracted by the mosaic at every observation); the floor ``Fit(min_observations=floor)``."""
    from test_any_telescope import COADD, FIT, NUMERICS
    model = sc.Model(sky=[sc.Sky(damping=0.0)],
                     offsets=[sc.Offsets('level', on='detector'), sc.Offsets('gradient', times=tilt)])
    return sc.Recipe(model, fit=FIT.replace(iterations=200, clip=None, min_observations=floor),
                     coadd=COADD.replace(chunk_coverage=coverage), numerics=NUMERICS, name=name)


def _maps(result):
    from astropy.io import fits
    (mos,) = result.mosaic_paths
    with fits.open(mos) as h:
        return np.array(h['MEAN_MAP'].data, np.float64), np.array(h['MEAN_MAP_WEIGHT'].data, np.float64)


def test_a_frame_the_floor_mostly_unties_leaves_the_coadd_where_its_offsets_are_held_by_few(tmp_path):
    """The thirteenth exposure shares four of its 64 columns with the others. With the floor its observations
    elsewhere leave the solve: ``offset_coverage_solved`` keeps, of its gradient's chunks, only those over the shared
    columns, and of its level the shared observations; ``offset_coverage`` counts every one. With
    ``Coadd(chunk_coverage='solved')`` the observations its untied chunks carry leave the coadd (the grid columns
    only it sees go empty); with the default they stay; every other pixel of the two mosaics is the same."""
    from test_any_telescope import ALL, CHUNKS, H, W
    field = _field(tmp_path)
    observed = field.calibrate(_recipe('observed', 'observed'), jobs=ALL)
    solved = field.calibrate(_recipe('solved', 'solved'), jobs=ALL)
    (cal,) = solved.cal_paths
    with CalFile(cal) as c:
        last = [os.path.basename(p) for p in c.reproj_list].index('exp_0012_det_00.h5')     # the 13th exposure
        cov, kept = c.offset_coverage, c.offset_coverage_solved
        assert len(kept) == 2
        level_obs, level_kept = cov[0][last], kept[0][last]
        grad_obs, grad_kept = cov[1][last].reshape(CHUNKS), kept[1][last].reshape(CHUNKS)
    assert 0 < level_kept[0] < 0.1 * level_obs[0]
    assert (grad_obs > 0).all()
    assert (grad_kept[:, 1:] == 0).all() and (grad_kept[:, 0] > 0).any()            # the shared columns' chunks
    others = np.concatenate([kept[1][i] for i in range(len(kept[1])) if i != last])
    assert (others > 0).any()

    mean_o, weight_o = _maps(observed)
    mean_s, weight_s = _maps(solved)
    pad = 8                                                                           # the reprojection's padding
    y0, x0 = 4 + pad, 72 + pad                                                        # its corner on the grid
    alone = np.zeros(weight_o.shape, bool)
    alone[y0 + 2:y0 + H - 2, x0 + W // CHUNKS[1] + 2:x0 + W - 2] = True               # its columns beyond chunk 0
    alone[:, x0 + W // 2] = False         # detector x = W/2: the gradient's function is 0, no dropped unknown there
    alone &= np.nan_to_num(weight_o) > 0
    assert alone.sum() > 0.5 * (H - 4) * (W - W // CHUNKS[1] - 5)
    assert (np.nan_to_num(weight_s[alone]) == 0).all()
    elsewhere = np.zeros(weight_o.shape, bool)
    elsewhere[:, :x0 - 4] = True
    np.testing.assert_array_equal(weight_s[elsewhere], weight_o[elsewhere])
    np.testing.assert_array_equal(mean_s[elsewhere], mean_o[elsewhere])


def test_the_setting_lowers_only_when_set_and_a_cal_without_the_record_is_refused(tmp_path):
    """``Coadd(chunk_coverage=)`` validates; the default does not reach the coadd's keywords (a recipe without it
    lowers as before); a cal whose solve had no floor carries no ``offset_coverage_solved``, and the solved rule
    refuses it."""
    from selfcal.config.base import ConfigError
    from selfcal.run.lower import coadd_options
    from test_any_telescope import ALL
    with pytest.raises(ConfigError, match='chunk_coverage'):
        sc.Coadd(chunk_coverage='kept')
    compute = sc.Compute()
    plain = sc.Recipe(sc.continuum(), coadd=sc.Coadd())
    assert 'chunk_coverage' not in coadd_options(plain, compute, sc.Camera((4, 4)))
    assert coadd_options(plain.replace(coadd=sc.Coadd(chunk_coverage='solved')), compute,
                         sc.Camera((4, 4)))['chunk_coverage'] == 'solved'
    field = _field(tmp_path)
    floorless = field.calibrate(_recipe('floorless', 'observed', floor=None), jobs=ALL)
    with CalFile(floorless.cal_paths[0]) as c:
        assert c.offset_coverage_solved == []
    with pytest.raises(ValueError, match='offset_coverage_solved'):
        field.mosaic(_recipe('floorless', 'solved', floor=None), jobs=ALL, overwrite=True)
