"""The SPHEREx per-detector-column chunk map: one chunk per column, id = the column, every row
(the reference columns included), at detector and grid resolution; a frame-shared term on it builds."""
import numpy as np

import selfcal as sc


def test_detector_column_map_is_one_chunk_per_column():
    geom = sc.SPHEREx(4, num_col=3).geometry(1)
    cm = geom.chunk_maps['detector_column']
    H, W = geom.shape
    assert cm.n_chunks == W == 2040
    assert np.array_equal(cm.det, np.tile(np.arange(W, dtype=np.int32), (H, 1)))
    assert list(cm.axes.names) == ['detector_column']
    assert sorted(geom.chunk_maps) == ['detector_column', 'readout', 'subchannel'] and geom.primary == 'subchannel'


def test_detector_column_map_follows_the_oversampling():
    geom = sc.SPHEREx(4, num_col=3).geometry(2)
    cm = geom.chunk_maps['detector_column']
    assert cm.grid.shape == tuple(2 * v for v in geom.shape)
    assert np.array_equal(cm.grid[::2, ::2], cm.det) and np.array_equal(cm.grid[1::2, 1::2], cm.det)


def test_a_shared_column_term_builds_on_the_map():
    term = sc.Offsets('columns', on='detector_column', per='all')
    model = sc.Model(sky=[sc.Sky(damping=0.0)], offsets=[sc.Offsets(on='subchannel', mean_zero=True), term])
    assert model.offsets[-1].on == 'detector_column' and model.offsets[-1].per == 'all'
