"""``Fit(min_observations=N)``: the sky pixels observed fewer than N times leave the solve
(``setup_lsqr(min_observations=)``): counted by the matrix build's own collation, their rows zeroed at the placement
(values and right-hand side), their columns left without data.

Synthetic frames written with :func:`selfcal.io.frames.write_frame` (no instrument): frame k sits on grid rows
k..k+H, so the first and last grid rows are observed once and the coverage grows inward; one non-finite value and one
outlier leave single observations inside the field too (the outlier only when the solve clips it, or when the
model's weight is not finite there).
"""
import numpy as np
import pytest
import scipy.sparse as sp

import selfcal as sc
from selfcal import _state
from selfcal.core.solve import apply_lsqr
from selfcal.core.system import setup_lsqr
from selfcal.io.frames import standard_frame_path, write_frame
from selfcal.models.sky_model import Coefficient

H, W, F = 8, 8, 6
GRID = (H + F - 1, W)
N_SKY = GRID[0] * GRID[1]


def _frames(tmp_path):
    """F frames, frame k on the grid rows k..k+H, a smooth value plus a per-frame offset; frame 1 has a non-finite
    value at grid (1, 2) and an outlier at grid (1, 5), both on a row frames 0 and 1 alone observe."""
    _state.set_progress(False)
    rng = np.random.default_rng(3)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    paths = []
    for k in range(F):
        vals = (1.0 + 0.05 * k + 0.01 * (y + k) + 0.002 * x + 1e-3 * rng.standard_normal((H, W))).astype(np.float32)
        if k == 1:
            vals[0, 2] = np.nan
            vals[0, 5] = 1e3
        paths.append(write_frame(standard_frame_path(str(tmp_path), k, 0), vals, [k, k + H, 0, W], np.stack([x, y])))
    return paths


def nan_above_100(value):
    """A row weight of 1, not finite where the value is above 100 (the outlier): the solve drops that row."""
    return np.where(value > 100, np.nan, 1.0)


def _setup(paths, **kw):
    common = dict(chunk_maps=[np.zeros((H, W), np.int32)], grid_valid_weight=np.ones((H, W), np.float32),
                  apply_mask=True, ignore_list=[], max_workers=1, batch_size=4)
    return setup_lsqr(paths, GRID, **{**common, **kw})


def _sky_counts(result):
    return np.asarray(result.pixel_counts[:N_SKY]).reshape(GRID)


def _to_csr(A):
    """Any of the build's storages as one csr_matrix, rows in order."""
    from selfcal.core.blockcsr import ColSplitCSR
    if sp.issparse(A):
        return A.tocsr()
    if isinstance(A, ColSplitCSR):
        rows = []
        for b, per in enumerate(A.sub):
            nb = int(A.row_bounds[b + 1] - A.row_bounds[b])
            rows.append(sp.hstack([sp.csr_matrix((d, i, ip), shape=(nb, int(A.cuts[t + 1] - A.cuts[t])))
                                   for t, (d, i, ip) in enumerate(per)], format='csr'))
        return sp.vstack(rows, format='csr')
    return sp.vstack(A.blocks, format='csr')                     # BlockCSR


def _full_width(result):
    """A with its columns at their uncompacted positions."""
    A = _to_csr(result.A)
    if result.active_mask is None:
        return A
    cols = np.flatnonzero(np.asarray(result.active_mask))
    return sp.csr_matrix((A.data, cols[A.indices], A.indptr), shape=(A.shape[0], len(result.active_mask)))


@pytest.mark.parametrize('clip', [None, 3])
def test_the_floor_removes_the_pixels_the_solve_observes_once(tmp_path, clip):
    """The pixels the unfloored solve covers once (after the mask and, with a clip, the clip) are gone; every other
    pixel keeps its coverage: the count is the solve's own sky coverage."""
    paths = _frames(tmp_path)
    plain = _sky_counts(_setup(paths, outlier_thresh=clip))
    floored = _sky_counts(_setup(paths, outlier_thresh=clip, min_observations=2))
    once = plain == 1
    assert once[0].all() and once[-1].all() and once[1, 2]          # the edge rows and the non-finite value
    assert once[1, 5] == (clip is not None)                          # the outlier: once only after the clip
    assert floored[once].sum() == 0
    np.testing.assert_array_equal(floored[~once], plain[~once])
    floored3 = _sky_counts(_setup(paths, outlier_thresh=clip, min_observations=3))
    np.testing.assert_array_equal(floored3[plain >= 3], plain[plain >= 3])
    assert floored3[(plain > 0) & (plain < 3)].sum() == 0


def test_a_row_the_weights_remove_does_not_count(tmp_path):
    """An observation whose model weight is not finite has no row in the solve, so it does not count: with no clip,
    the outlier's pixel is observed once by the solve and the floor drops it."""
    paths = _frames(tmp_path)
    weight = Coefficient('value', nan_above_100)
    plain = _sky_counts(_setup(paths, outlier_thresh=None, weight_function=weight))
    floored = _sky_counts(_setup(paths, outlier_thresh=None, weight_function=weight, min_observations=2))
    assert plain[1, 5] == 1 and _sky_counts(_setup(paths, outlier_thresh=None))[1, 5] == 2
    once = plain == 1
    assert floored[once].sum() == 0
    np.testing.assert_array_equal(floored[~once], plain[~once])


@pytest.mark.parametrize('placement', ['serial', 'parallel', 'partitioned'])
def test_the_rows_are_the_unfloored_rows_with_the_dropped_ones_zeroed(tmp_path, monkeypatch, placement):
    """The floor zeroes exactly the rows of the dropped pixels, values and right-hand side; every other entry is the
    unfloored system's, byte for byte, in the same layout (serial, parallel and column-partitioned placement)."""
    monkeypatch.setenv('SELFCAL_SCATTER_WORKERS', '1' if placement == 'serial' else '2')
    if placement == 'partitioned':
        monkeypatch.setenv('SELFCAL_BLOCK_NNZ', '1')
        monkeypatch.setenv('SELFCAL_RMATVEC_SPLIT', '2')
    paths = _frames(tmp_path)
    plain = _setup(paths, outlier_thresh=None, batch_spill_dir=str(tmp_path / 'spill'))
    floored = _setup(paths, outlier_thresh=None, min_observations=2, batch_spill_dir=str(tmp_path / 'spill'))
    A, Af = _full_width(plain), _full_width(floored)
    assert A.shape == Af.shape
    sky_col = np.array([row.indices[row.indices < N_SKY][0] for row in A])          # one sky entry per data row
    keep = np.asarray(plain.pixel_counts)[sky_col] >= 2
    assert (~keep).sum() > 0
    np.testing.assert_array_equal(A.indptr, Af.indptr)
    np.testing.assert_array_equal(A.indices, Af.indices)
    expect = A.copy()
    expect.data[np.repeat(~keep, np.diff(A.indptr))] = 0
    np.testing.assert_array_equal(Af.data, expect.data)
    b, bf = np.asarray(plain.b), np.asarray(floored.b)
    np.testing.assert_array_equal(bf[keep], b[keep])
    assert (bf[~keep] == 0).all() and (b[~keep] != 0).any()
    dropped_cols = np.unique(sky_col[~keep])
    assert (abs(Af[:, dropped_cols]).sum() == 0)                                    # left without data
    assert np.asarray(floored.active_mask)[dropped_cols].all()                       # in the layout


@pytest.mark.parametrize('storage', ['unified', 'partitioned'])
def test_the_deflated_solve_is_the_solve_without_those_rows(tmp_path, monkeypatch, storage):
    """The floored system, solved with the deflation through the solver's own path (its columns without data kept in
    the layout), gives every other unknown the solution of the system with those rows removed, and leaves the dropped
    pixels at zero: the deflation's patterns do not reach them. Both storages (the column-partitioned one is the full
    fields')."""
    extra = {}
    if storage == 'partitioned':
        for k, v in (('SELFCAL_SCATTER_WORKERS', '2'), ('SELFCAL_BLOCK_NNZ', '1'), ('SELFCAL_RMATVEC_SPLIT', '2')):
            monkeypatch.setenv(k, v)
        extra = {'batch_spill_dir': str(tmp_path / 'spill')}
    paths = _frames(tmp_path)
    plain = _setup(paths, outlier_thresh=None)
    floored = _setup(paths, outlier_thresh=None, min_observations=2, **extra)
    if storage == 'partitioned':
        from selfcal.core.blockcsr import ColSplitCSR
        assert isinstance(floored.A, ColSplitCSR)
    A = _full_width(plain)
    sky_col = np.array([row.indices[row.indices < N_SKY][0] for row in A])
    keep = np.asarray(plain.pixel_counts)[sky_col] >= 2
    dropped = np.unique(sky_col[~keep])

    def solve(Am, bm, active, n_full):
        return apply_lsqr(Am, np.asarray(bm, dtype=np.float64).copy(), ref_shape=GRID, atol=0.0, btol=0.0, conlim=0.0,
                          damp=0.0, solver='cgls', n_threads=1, iter_lim=400, active_mask=active, num_cols_full=n_full,
                          scratch_dir=str(tmp_path), sky_names=('sky',), deflate=sc.Deflate(1))

    xf = solve(floored.A, floored.b, floored.active_mask, len(floored.active_mask))
    Ar = A[keep]
    act = np.asarray((Ar != 0).sum(axis=0)).ravel() > 0
    xr = solve(Ar[:, act].tocsr(), np.asarray(plain.b)[keep], act, A.shape[1])
    assert (xf[dropped] == 0).all()
    kept_sky = np.setdiff1d(np.unique(sky_col[keep]), dropped)
    for sel in (kept_sky, slice(N_SKY, None)):
        # the partitioned operator's products in the stored float32: 2e-5, or a few float32 roundings of the largest
        # value where the values are large (the sky's, ~500)
        tol = 1e-6 if storage == 'unified' else max(2e-5, 2e-7 * float(np.abs(xr[sel]).max()))
        np.testing.assert_allclose(xf[sel], xr[sel], rtol=0, atol=tol)


def test_the_deflation_reads_no_column_outside_its_space():
    """A floor's columns keep their zero entries but leave the deflation's space (column map -1): the passes send those
    entries to a valid column, whose zero value keeps every product exact, instead of handing -1 to the compiled
    kernels (an out-of-bounds read; a segmentation fault on the full D1 field, 2026-10-22)."""
    from selfcal.core.deflation import _IDX, _prepare_piece, _Slot
    colmap = np.array([0, -1, 1, 2, 3], dtype=_IDX)      # full column 1 is outside the space; 2 sky columns in it
    lip = np.array([0, 2, 4], dtype=_IDX)
    cols = np.array([1, 3, 0, 4], dtype=np.int32)        # row 0: the outside column (value 0) and an offset; row 1: sky, offset
    data = np.array([0.0, 2.0, 1.5, 3.0], dtype=np.float32)
    _lip, pos, d64, sky, off = _prepare_piece(2, lip, cols, data, colmap, 2, buf=_Slot())
    assert (pos >= 0).all() and list(pos) == [0, 2, 0, 3]
    ips, sc, sv, rows = sky
    assert list(sc) == [0, 0] and list(sv) == [0.0, 1.5] and list(rows) == [0, 1]
    oip, oc, ov = off
    assert list(oc) == [0, 1] and list(ov) == [2.0, 3.0]


def test_a_pixel_observed_once_informs_nothing_else(tmp_path):
    """In the unconstrained least-squares fit a single-observation pixel fits its one row exactly, and the fit of
    every other row is the floored system's: dropping those pixels changes nothing else."""
    paths = _frames(tmp_path)
    plain = _setup(paths, outlier_thresh=None)
    floored = _setup(paths, outlier_thresh=None, min_observations=2)
    A, Af = _full_width(plain).toarray(), _full_width(floored).toarray()
    b, bf = np.asarray(plain.b, float), np.asarray(floored.b, float)
    sky_col = np.array([np.flatnonzero(A[i, :N_SKY])[0] for i in range(A.shape[0])])
    keep = np.asarray(plain.pixel_counts)[sky_col] >= 2
    fit = A @ np.linalg.lstsq(A, b, rcond=None)[0]
    fit_f = Af @ np.linalg.lstsq(Af, bf, rcond=None)[0]
    np.testing.assert_allclose(fit[~keep], b[~keep], atol=1e-9)          # fitted exactly by its own pixel
    np.testing.assert_allclose(fit[keep], fit_f[keep], atol=1e-9)
    assert (fit_f[~keep] == 0).all()                                     # the zeroed rows: nothing to fit


def test_the_setting_lowers_to_the_setup_keyword_and_validates():
    import selfcal as sc
    from selfcal.config.base import ConfigError
    from selfcal.run.lower import setup_options

    class _Inst:
        def default_ignore_flags(self):
            return []

    compute = sc.Compute(workers=2)
    assert setup_options(sc.Recipe(sc.continuum(), fit=sc.Fit(min_observations=2)), compute,
                         _Inst())['min_observations'] == 2
    assert 'min_observations' not in setup_options(sc.Recipe(sc.continuum()), compute, _Inst())
    for bad in (1, 0, 2.0, True, '2'):
        with pytest.raises(ConfigError):
            sc.Fit(min_observations=bad)
    with pytest.raises(ValueError):
        setup_lsqr([], GRID, min_observations=1)
    with pytest.raises(ValueError, match='matrix build'):
        setup_lsqr(['x.h5'], GRID, min_observations=2, moments_only=True)
