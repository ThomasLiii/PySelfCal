"""``sc.Offsets(store='groups')``: a term shared by groups of frames is written to the cal once per group, with each
frame's group, instead of once per frame. The readers give the same per-frame offsets, and the mosaic, which keeps
such a map compact, is the same to the last bit as with the term written per frame."""
import h5py
import numpy as np

import selfcal as sc
from selfcal.io.calfile import CalFile, FrameRows

from test_overlap_exclusion import pixel_chunks  # noqa: E402


def _run(tmp_path, store):
    from test_any_telescope import COADD, FIT, NUMERICS, PIX, REF, TWO_WORKERS, plane, true_sky, write_exposure
    from test_any_telescope import H as TH
    from test_any_telescope import W as TW
    rng = np.random.default_rng(3)
    sky = true_sky()
    y, x = np.mgrid[0:TH, 0:TW].astype(np.float64)
    gx, gy = plane(x, y)
    pattern = rng.normal(0, 0.05, (TH, TW))
    exposures = tmp_path / 'exposures'
    if not exposures.exists():
        for k in range(24):
            oy, ox = rng.integers(0, REF - TH), rng.integers(0, REF - TW)
            g = rng.normal(0, 0.3, 2)
            img = sky[oy:oy + TH, ox:ox + TW] + pattern + g[0] * gx + g[1] * gy + rng.normal(0, 0.01, (TH, TW))
            write_exposure(exposures / f'toy_exp_{k:03d}.fits', img, ox, oy, {})
    field = sc.Field(tmp_path / f'toy_{store}', sc.Camera((TH, TW), tag='Toy'), PIX, compute=TWO_WORKERS)
    field.reproject(exposures / 'toy_exp_*.fits', method='interp', padding=8)
    model = sc.Model(chunk_maps={'pixels': sc.ChunkMap(pixel_chunks)}, sky=[sc.Sky(damping=1e-4)],
                     offsets=[sc.Offsets('pattern', on='pixels', per='all', exclude=sc.Overlap('gradient', 'scalar'),
                                         store=store),
                              sc.Offsets('gradient', on='detector', basis=plane, n=2),
                              sc.Offsets('scalar', on='detector')])
    return field.calibrate(sc.Recipe(model, fit=FIT, coadd=COADD, numerics=NUMERICS), jobs=sc.Job('All'))


def test_a_shared_term_stored_once_reads_and_mosaics_as_per_frame(tmp_path):
    per_frame, grouped = _run(tmp_path, 'frames'), _run(tmp_path, 'groups')
    (cal_f,), (cal_g,) = per_frame.cal_paths, grouped.cal_paths
    with h5py.File(cal_g, 'r') as f:
        n_frames = f['reproj_list'].shape[0]
        assert f['offsets/map_0'].shape[0] == 1 and f['offsets/map_0'].attrs['rows'] == 'groups'
        assert f['offset_coverage/map_0'].shape[0] == 1 and f['offset_coverage_frac/map_0'].shape[0] == 1
        assert np.array_equal(f['offset_frame_group/map_0'][()], np.zeros(n_frames))
        assert f['offsets/map_1'].shape[0] == n_frames and 'offset_frame_group' in f and 'map_1' not in f['offset_frame_group']
    with CalFile(cal_f) as a, CalFile(cal_g) as b:
        for x, y in zip(a.offsets + a.offset_coverage + a.offset_coverage_frac,
                        b.offsets + b.offset_coverage + b.offset_coverage_frac):
            assert np.array_equal(x, y)
        rows = b.rows('offsets')
        assert isinstance(rows[0], FrameRows) and rows[0].shape == a.offsets[0].shape
        assert np.array_equal(rows[0][5], a.offsets[0][5])
        assert np.array_equal(np.asarray(rows[0][np.arange(n_frames) % 2 == 0]), a.offsets[0][::2])
    from astropy.io import fits
    (mos_f,), (mos_g,) = per_frame.mosaic_paths, grouped.mosaic_paths
    with fits.open(mos_f) as fa, fits.open(mos_g) as fb:
        for name in ('MEAN_MAP', 'MEAN_MAP_WEIGHT', 'STD_MAP'):
            assert np.array_equal(fa[name].data, fb[name].data, equal_nan=True), name
