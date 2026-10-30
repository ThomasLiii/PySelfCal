"""A term shared by groups of frames is written to the cal once per group, with each frame's group
(``offset_frame_group/map_<m>``), never once per frame: one row for a term every frame shares, one per group for a term
shared within groups (a frame variable). The readers give the same per-frame offsets as a cal that repeats each
group's row on its frames, and the mosaic, which keeps such a map compact, is the same to the last bit as one made
from that repeated layout."""
import os
import shutil

import h5py
import numpy as np
from astropy.io import fits

import selfcal as sc
from selfcal.io.calfile import CalFile, FrameRows

from test_overlap_exclusion import pixel_chunks  # noqa: E402

N_EXP = 24
#: An arbitrary partition of the exposures: three groups of unequal sizes (11, 8, 5), scattered over the exposure
#: order, with label values that are neither consecutive nor in order.
LABELS = np.array([17, 5, 5, 2, 17, 17, 2, 5, 5, 5, 2, 17, 5, 2, 17, 17, 5, 17, 5, 5, 5, 17, 17, 17])


def partition(frames):
    """A frame variable: each frame's label in :data:`LABELS`, by its exposure number."""
    return LABELS[[int(os.path.basename(f).split('_')[1]) for f in frames]]


def _field(tmp_path):
    from test_any_telescope import PIX, REF, TWO_WORKERS, plane, true_sky, write_exposure
    from test_any_telescope import H as TH
    from test_any_telescope import W as TW
    rng = np.random.default_rng(3)
    sky = true_sky()
    y, x = np.mgrid[0:TH, 0:TW].astype(np.float64)
    gx, gy = plane(x, y)
    pattern = {g: rng.normal(0, 0.05, (TH, TW)) for g in np.unique(LABELS)}
    for k in range(N_EXP):
        oy, ox = rng.integers(0, REF - TH), rng.integers(0, REF - TW)
        g = rng.normal(0, 0.3, 2)
        img = sky[oy:oy + TH, ox:ox + TW] + pattern[LABELS[k]] + g[0] * gx + g[1] * gy + rng.normal(0, 0.01, (TH, TW))
        write_exposure(tmp_path / 'exposures' / f'toy_exp_{k:03d}.fits', img, ox, oy, {})
    field = sc.Field(tmp_path / 'toy', sc.Camera((TH, TW), tag='Toy'), PIX, compute=TWO_WORKERS)
    field.reproject(tmp_path / 'exposures' / 'toy_exp_*.fits', method='interp', padding=8)
    return field


def _recipe(per, variables=None):
    from test_any_telescope import COADD, FIT, NUMERICS, plane
    model = sc.Model(chunk_maps={'pixels': sc.ChunkMap(pixel_chunks)}, sky=[sc.Sky(damping=1e-4)],
                     offsets=[sc.Offsets('pattern', on='pixels', per=per, exclude=sc.Overlap('gradient', 'scalar')),
                              sc.Offsets('gradient', on='detector', basis=plane, n=2),
                              sc.Offsets('scalar', on='detector')],
                     variables=variables or {})
    return sc.Recipe(model, fit=FIT, coadd=COADD, numerics=NUMERICS, name=f'rows_{per}')


def _repeated_copy(src, dst):
    """The cal ``src`` written the other way: every map's rows repeated on each frame of its group, no
    ``offset_frame_group``."""
    shutil.copy(src, dst)
    with h5py.File(dst, 'r+') as f:
        for m in list(f['offset_frame_group']):
            fg = f['offset_frame_group'][m][()]
            for group in ('offsets', 'offset_coverage', 'offset_coverage_frac'):
                rows = f[group][m][()]
                attrs = dict(f[group][m].attrs)
                del f[group][m]
                ds = f[group].create_dataset(m, data=rows[fg], compression='gzip')
                for k, v in attrs.items():
                    if k != 'rows':
                        ds.attrs[k] = v
        del f['offset_frame_group']


def test_a_term_every_frame_shares_is_one_row_and_mosaics_as_repeated(tmp_path):
    field = _field(tmp_path)
    recipe = _recipe('all')
    result = field.calibrate(recipe, jobs=sc.Job('All'))
    (cal,), (mos,) = result.cal_paths, result.mosaic_paths
    with h5py.File(cal, 'r') as f:
        n = f['reproj_list'].shape[0]
        assert f['offsets/map_0'].shape[0] == 1 and f['offsets/map_0'].attrs['rows'] == 'groups'
        assert f['offset_coverage/map_0'].shape[0] == 1 and f['offset_coverage_frac/map_0'].shape[0] == 1
        assert np.array_equal(f['offset_frame_group/map_0'][()], np.zeros(n))
        assert f['offsets/map_1'].shape[0] == n and 'map_1' not in f['offset_frame_group']   # per-frame terms as before
    rep = tmp_path / 'cal_repeated.h5'
    _repeated_copy(cal, rep)
    with CalFile(cal) as a, CalFile(rep) as b:
        for x, y in zip(a.offsets + a.offset_coverage + a.offset_coverage_frac,
                        b.offsets + b.offset_coverage + b.offset_coverage_frac):
            assert np.array_equal(x, y)
        rows = a.rows('offsets')
        assert isinstance(rows[0], FrameRows) and rows[0].shape == b.offsets[0].shape
        assert np.array_equal(np.asarray(rows[0][np.arange(n) % 2 == 0]), b.offsets[0][::2])
    with fits.open(mos) as h:
        first = {k: np.array(h[k].data) for k in ('MEAN_MAP', 'MEAN_MAP_WEIGHT', 'STD_MAP')}
    again = field.mosaic(recipe, jobs=sc.Job('All'), cal=rep, overwrite=True)
    with fits.open(again.mosaic_paths[0]) as h:
        for k, v in first.items():
            assert np.array_equal(v, h[k].data, equal_nan=True), k


def test_a_term_shared_within_any_partition_is_one_row_per_group_labelled_per_frame(tmp_path):
    """Frames grouped by any frame variable: one row per group, each frame labelled with its group's row (the groups
    numbered in the sorted order of the variable's values), the rows the same within a group and apart between them."""
    field = _field(tmp_path)
    result = field.calibrate(_recipe('part', {'part': sc.PerFrame(partition)}), jobs=sc.Job('All'))
    (cal,) = result.cal_paths
    with h5py.File(cal, 'r') as f:
        names = [os.path.basename(r.decode()) for r in f['reproj_list'][()]]
        fg = f['offset_frame_group/map_0'][()]
        assert f['offsets/map_0'].shape[0] == 3
    labels = np.unique(LABELS)
    assert np.array_equal(fg, np.searchsorted(labels, partition(names)))     # each frame -> its group's row
    with CalFile(cal) as c:
        off = c.offsets[0]
    for g in range(labels.size):
        assert np.array_equal(off[fg == g], np.repeat(off[fg == g][:1], int(np.sum(fg == g)), axis=0))
    assert all(not np.array_equal(off[fg == a][0], off[fg == b][0]) for a in range(3) for b in range(a + 1, 3))
