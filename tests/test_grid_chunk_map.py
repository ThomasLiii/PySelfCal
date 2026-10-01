"""Grid chunk maps: exactly ``ny x nx`` row-major chunks whatever the detector size.

Regression: the square path (``make_grid_chunk_map``, used by the ``grid`` instrument
when ``ny == nx`` and by the Euclid grid map) made extra chunks or left the leftover
rows in chunk 0 when ``n`` did not divide the detector (a 10 x 10 detector with
``chunks = [4, 4]`` got 25 chunks). Runnable as a script or under pytest.
"""
import numpy as np

from selfcal.geometry.map_helper import make_grid_chunk_map
from selfcal.instruments.grid import GridInstrument, rect_grid_chunk_map


def _check_grid(cm, ny, nx):
    H, W = cm.shape
    assert cm.min() == 0 and cm.max() == ny * nx - 1
    assert len(np.unique(cm)) == ny * nx
    rows, cols = cm // nx, cm % nx
    # row-major ids, chunk rows constant along a pixel row and non-decreasing down the detector
    assert (rows == rows[:, :1]).all() and (cols == cols[:1, :]).all()
    assert (np.diff(rows[:, 0]) >= 0).all() and (np.diff(cols[0]) >= 0).all()
    # chunk sides differ by at most one pixel
    for counts in (np.bincount(rows[:, 0], minlength=ny), np.bincount(cols[0], minlength=nx)):
        assert counts.max() - counts.min() <= 1, counts


def test_square_grid_any_size():
    for H, W, n in [(10, 10, 4), (2048, 2048, 6), (30, 40, 4), (64, 64, 4), (2040, 2040, 40), (7, 9, 3)]:
        _check_grid(make_grid_chunk_map((H, W), n), n, n)


def test_divisible_grid_is_blocks():
    """When n divides the detector every chunk is an exact (H/n) x (W/n) block."""
    for H, W, n in [(64, 64, 4), (2040, 2040, 40), (2040, 2040, 510), (12, 30, 3)]:
        r, c = np.mgrid[0:H, 0:W]
        expected = (r // (H // n)) * n + c // (W // n)
        assert np.array_equal(make_grid_chunk_map((H, W), n), expected)


def test_rect_grid_matches_square_and_any_size():
    for H, W, ny, nx in [(10, 10, 4, 4), (48, 80, 3, 5), (101, 99, 4, 7), (64, 64, 4, 4)]:
        cm = rect_grid_chunk_map((H, W), ny, nx)
        assert cm.dtype == np.int32
        _check_grid(cm, ny, nx)
        if ny == nx:
            assert np.array_equal(cm, make_grid_chunk_map((H, W), ny))


def test_grid_instrument_chunk_count():
    inst_cfg = {'name': 'grid', 'detector_shape': [10, 10], 'chunks': [4, 4]}
    cm = GridInstrument().detector_geometry(inst_cfg, 1).chunk_map
    assert cm.n_chunks == 16 and len(np.unique(cm.det)) == 16


if __name__ == '__main__':
    test_square_grid_any_size()
    test_divisible_grid_is_blocks()
    test_rect_grid_matches_square_and_any_size()
    test_grid_instrument_chunk_count()
    print('OK grid chunk maps')
