"""``Fit(window_edge=)``: the observations at the edge of the job's window (``setup_lsqr(window_edge=)``).

An observation samples the detector-plane quantities bilinearly from the four detector pixels around its position, so
one near the window's edge reads pixels on both sides. ``'partial'`` (the default) keeps it with its sample of the
window as a weight factor; ``'inside'`` keeps only the observations whose sampling lies wholly inside the window.

Synthetic frames written with :func:`selfcal.io.frames.write_frame` (no instrument): every observation sits half a
detector pixel below the next detector row (y = row + 0.5), so the window of detector rows 0..3 has one row of
observations (y = 3.5) that reads detector rows 3 (inside) and 4 (outside).
"""
import numpy as np
import pytest
import scipy.sparse as sp

import selfcal as sc
from selfcal import _state
from selfcal.config.base import ConfigError
from selfcal.core.subframe import _inside_row_mask, _prep_subframe, _valid_row_mask
from selfcal.core.system import setup_lsqr
from selfcal.io.frames import standard_frame_path, write_frame
from selfcal.run.lower import setup_options

H, W, F = 8, 6, 3
GRID = (H + F - 1, W)
WINDOW_ROWS = 4                                   # the window: detector rows 0..3


def _window():
    w = np.zeros((H, W), np.float32)
    w[:WINDOW_ROWS] = 1.0
    return w


def _pixels():
    """One chunk per detector pixel."""
    return np.arange(H * W, dtype=np.int32).reshape(H, W)


def _frames(tmp_path):
    _state.set_progress(False)
    rng = np.random.default_rng(5)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    det = np.stack([x, y + 0.5]).astype(np.float32)          # (x, y): half a pixel below the next row
    paths = []
    for k in range(F):
        vals = (1.0 + 0.1 * k + 0.01 * (y + k) + 1e-3 * rng.standard_normal((H, W))).astype(np.float32)
        paths.append(write_frame(standard_frame_path(str(tmp_path), k, 0), vals, [k, k + H, 0, W], det))
    return paths


def test_inside_rows_are_the_rows_whose_every_sampled_pixel_is_inside():
    """Kept: positions whose four bilinear corners (less those with zero weight) are inside; dropped: a position
    straddling the edge along y or x, or reaching beyond the grid."""
    win = np.zeros((6, 6), np.float32)
    win[1:4, 1:4] = 1.0                              # rows and columns 1..3
    pts = {(1.0, 1.0): True, (2.5, 2.5): True, (3.0, 3.0): True, (3.0, 2.5): True,   # (y, x)
           (3.5, 2.0): False, (2.0, 3.5): False, (0.5, 2.0): False, (3.2, 3.0): False,
           (5.5, 5.5): False, (-0.5, 2.0): False}
    coords = np.array(list(pts), dtype=np.float32).T
    want = np.array(list(pts.values()))
    got = _inside_row_mask(coords, win)
    assert np.array_equal(got, want), dict(zip(pts, got))
    assert not np.any(got & ~_valid_row_mask(coords, win))  # a subset of the rows 'partial' keeps


@pytest.mark.parametrize('edge', ['partial', 'inside'])
def test_a_frame_at_the_window_edge(tmp_path, edge):
    """'partial' keeps the straddling observations (y = 3.5) at half weight and their sampling reaches detector row 4;
    'inside' gives them no weight and no observation reads row 4. The observations wholly inside keep weight 1."""
    path = _frames(tmp_path)[0]
    _, _, weight, contribs, _ = _prep_subframe(path, chunk_maps=[_pixels()], grid_valid_weight=_window(),
                                               for_lsqr=True, window_edge=edge)
    assert np.all(weight[:WINDOW_ROWS - 1] == 1.0)
    assert np.all(weight[WINDOW_ROWS:] == 0.0)
    straddling = weight[WINDOW_ROWS - 1]
    C = sp.csr_matrix(contribs[0]).T.tocsr()                            # (observations, chunks)
    reads = np.asarray((abs(C) > 0).sum(axis=0)).ravel().reshape(H, W)   # observations reading each pixel's chunk
    used = weight.ravel() > 0
    reads_used = np.asarray((abs(C[used]) > 0).sum(axis=0)).ravel().reshape(H, W)
    if edge == 'partial':
        assert np.allclose(straddling, 0.5)
        assert np.all(reads_used[WINDOW_ROWS] > 0)
    else:
        assert np.all(straddling == 0.0)
        assert np.all(reads_used[WINDOW_ROWS:] == 0)
        assert reads[WINDOW_ROWS:].sum() == 0                          # not even an entry with no weight


def test_inside_builds_the_partial_system_without_the_straddling_observations(tmp_path):
    """The system 'inside' builds is the one 'partial' builds less the rows of the straddling observations, every
    other row and right-hand side value identical; no row reads an offset of detector row 4."""
    paths = _frames(tmp_path)
    common = dict(chunk_maps=[_pixels()], grid_valid_weight=_window(), apply_mask=False, ignore_list=[],
                  outlier_thresh=None, max_workers=1, batch_size=4)
    partial = setup_lsqr(paths, GRID, **common)
    inside = setup_lsqr(paths, GRID, **common, window_edge='inside')

    def dense(r):
        A = r.A if sp.issparse(r.A) else sp.vstack(r.A.blocks, format='csr')
        if r.active_mask is not None:
            cols = np.flatnonzero(np.asarray(r.active_mask))
            A = sp.csr_matrix((A.data, cols[A.indices], A.indptr), shape=(A.shape[0], len(r.active_mask)))
        return A.toarray(), np.asarray(r.b, dtype=np.float64)
    Ap, bp = dense(partial)
    Ai, bi = dense(inside)
    n_sky = GRID[0] * GRID[1]
    row4 = [n_sky + f * H * W + WINDOW_ROWS * W + c for f in range(F) for c in range(W)]   # each frame's row-4 chunks
    straddle = np.any(Ap[:, row4] != 0, axis=1)
    assert straddle.sum() == F * W                    # one row of observations per frame
    assert np.array_equal(Ai, Ap[~straddle])
    assert np.array_equal(bi, bp[~straddle])
    assert not np.any(Ai[:, row4])


def test_the_setting():
    """Fit(window_edge=) is checked, and reaches the build only when it is not the default (so the build of every
    existing recipe is unchanged)."""
    assert sc.Fit().window_edge == 'partial'
    with pytest.raises(ConfigError):
        sc.Fit(window_edge='hard')

    class _Compute:
        resolved_workers = 1
    recipe = sc.Recipe(sc.continuum(), fit=sc.Fit())
    assert 'window_edge' not in setup_options(recipe, _Compute(), sc.SPHEREx(2))
    inside = sc.Recipe(sc.continuum(), fit=sc.Fit(window_edge='inside'))
    assert setup_options(inside, _Compute(), sc.SPHEREx(2))['window_edge'] == 'inside'
    with pytest.raises(ValueError):
        setup_lsqr([], (2, 2), window_edge='hard')
