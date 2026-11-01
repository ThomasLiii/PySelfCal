"""The observation table's rows grouped by sky range (``setup_lsqr(matrix='compact', row_order='sky')``,
``sc.Numerics(row_order='sky')``, :func:`selfcal.core.observations.by_sky_range`).

On synthetic frames: the rows are the frame order's, permuted by range of reference pixels (a stable sort, so each
range's rows in frame order), with the right-hand side, the pixel state, the coverage floor and the constraint rows;
the transpose products agree with a float64 reference to the float32 rounding, and their sky columns are the same
bytes at any thread count (each range's rows are one thread's); the solves, deflated or not, agree with the frame
order's to the rounding; the setting runs through the API and is refused without the table.
"""

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip('numba')

import selfcal as sc  # noqa: E402
from selfcal.core.solve import _make_table_operator, apply_lsqr  # noqa: E402
from selfcal.core.system import setup_lsqr  # noqa: E402
from selfcal.models.offset_model import Basis  # noqa: E402
from selfcal.models.sky_model import Coefficient, SkyComponent, SkyModel  # noqa: E402
from selfcal.models.variables import Derived, VariableSet  # noqa: E402
from tests.test_observation_table import (DET, F, GRID, H, N_SKY, PIXELS, W, _bits, _common, _frames,  # noqa: E402
                                          arc_coef, arc_times_t, hats, tilt_plus_x)


def _pair(paths, **kw):
    """The same system as a frame-ordered table and one grouped by sky range."""
    kw = _common(**kw)
    frame = setup_lsqr(paths, GRID, matrix='compact', **kw)
    grouped = setup_lsqr(paths, GRID, matrix='compact', row_order='sky', **kw)
    assert frame.A.row_order == 'frame' and grouped.A.row_order == 'sky'
    return frame, grouped


def _rows(table):
    """The global ids of the rows the coverage floor zeroed."""
    parts = [b.fl + int(table.row_bounds[i]) for i, b in enumerate(table.blocks)]
    return np.sort(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int64)


def _same_rows(frame, grouped):
    """The table grouped by sky range holds the frame order's rows sorted by range of reference pixels (stably: each
    range's rows in frame order), the constraint rows after them unchanged; the right-hand side follows; the pixel
    state, the columns and the floored rows agree. Returns the permutation (row i is frame row perm[i])."""
    nd = frame.A.n_data_rows
    assert grouped.A.n_data_rows == nd and grouped.A.shape == frame.A.shape and grouped.A.nnz == frame.A.nnz
    lows = np.array([int(b.sky.min()) for b in grouped.A.blocks])
    highs = np.array([int(b.sky.max()) for b in grouped.A.blocks])
    assert np.all(highs[:-1] < lows[1:])                    # the blocks' ranges of pixels disjoint and ascending
    for b in grouped.A.blocks:                                # each block's rows in frame order
        assert np.all(np.diff(b.frames) > 0) and b.frame_rows[0] == 0 and b.frame_rows[-1] == b.n
    sky = np.concatenate([b.sky for b in frame.A.blocks]).astype(np.int64)
    perm = np.argsort(np.searchsorted(lows, sky, side='right'), kind='stable')
    a, c = frame.A.to_csr(), grouped.A.to_csr()
    a = sp.vstack([a[:nd][perm], a[nd:]], format='csr')
    for mat in (a, c):
        mat.sort_indices()
    np.testing.assert_array_equal(a.indptr, c.indptr)
    np.testing.assert_array_equal(a.indices, c.indices)
    np.testing.assert_array_equal(_bits(a.data), _bits(c.data))
    np.testing.assert_array_equal(_bits(frame.b[:nd][perm]), _bits(grouped.b[:nd]))
    np.testing.assert_array_equal(_bits(frame.b[nd:]), _bits(grouped.b[nd:]))
    np.testing.assert_array_equal(frame.pixel_counts, grouped.pixel_counts)
    np.testing.assert_array_equal(_bits(frame.pixel_fisher), _bits(grouped.pixel_fisher))
    if frame.active_mask is None:
        assert grouped.active_mask is None
    else:
        np.testing.assert_array_equal(frame.active_mask, grouped.active_mask)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    np.testing.assert_array_equal(np.sort(inv[_rows(frame.A)]), _rows(grouped.A))
    return perm


def _models():
    arc = (np.linspace(0, 1, H * W, dtype=np.float32).reshape(DET) * 3).astype(np.float32)
    pix = (np.arange(H * W) % 5).reshape(DET).astype(np.int32)
    chunks = _common()['chunk_maps'][0]
    line = SkyModel(components=(SkyComponent('continuum'),
                                SkyComponent('line', coefficient=Coefficient(('arc_t',), arc_coef))))
    tilt = np.linspace(-1, 1, N_SKY, dtype=np.float64).reshape(GRID)
    slope = SkyModel(components=(SkyComponent('continuum'),
                                 SkyComponent('slope', coefficient=Coefficient(('tilt', 'sky_x'), tilt_plus_x))))
    return {
        'plain': {},
        'variables': dict(chunk_maps=[chunks, pix], det_groups_list=[None, np.array([0, 0, 1, 1, 2, 2, 2])],
                          basis_list=[Basis(Coefficient('det_x', hats), n=2), None], det_aux=[arc], aux_keys=['arc'],
                          sky_model=line, sky_rhs_moments=True,
                          variables=VariableSet(frame={'t': np.arange(F, dtype=np.float64)},
                                                derived={'arc_t': Derived(('arc', 't'), arc_times_t)})),
        'sky_variable': dict(sky_model=slope, variables=VariableSet(sky={'tilt': tilt})),
        'floors': dict(min_observations=2, offset_min_observations=[12], **PIXELS),
        'constraint_rows': dict(weighted_damping=True, damp_weight=0.1),
    }


@pytest.mark.parametrize('name', ['plain', 'variables', 'sky_variable', 'floors', 'constraint_rows'])
def test_the_same_rows(tmp_path, name):
    frame, grouped = _pair(_frames(tmp_path, zero_rows=True), **_models()[name])
    if name == 'floors':
        assert _rows(grouped.A).size
    if name == 'constraint_rows':
        assert grouped.A.tail is not None
    _same_rows(frame, grouped)


def test_stored_values_by_sky_range(tmp_path):
    """The model's values kept with a table grouped by sky range are the values its passes would make."""
    kw = _common(**_models()['variables'])
    paths = _frames(tmp_path)
    computed = setup_lsqr(paths, GRID, matrix='compact', row_order='sky', **kw)
    stored = setup_lsqr(paths, GRID, matrix='compact', row_order='sky', functions='stored', **kw)
    a, c = computed.A.to_csr(), stored.A.to_csr()
    np.testing.assert_array_equal(a.indices, c.indices)
    np.testing.assert_array_equal(_bits(a.data), _bits(c.data))


def test_the_transpose_products(tmp_path, monkeypatch):
    """The pieces cut at blocks' first rows; ``A^T y`` (float32), ``A^T y`` in float64 and the fused ``(A x,
    A^T A x)`` against float64 references; the sky columns of ``A^T y`` the same bytes at 1, 2, 3 and 5 threads."""
    monkeypatch.setenv('SELFCAL_PARALLEL_RMATVEC', 'auto')
    _frame, grouped = _pair(_frames(tmp_path), batch_size=1, **_models()['constraint_rows'])
    A = grouped.A
    csr = A.to_csr().astype(np.float64)
    assert len(A.blocks) > 5                                 # one block per frame: more ranges than threads
    for k in (2, 3, 5):
        cuts = A.row_cuts(k)
        assert cuts[0] == 0 and cuts[-1] == A.n_data_rows
        assert np.isin(cuts, A.row_bounds[:len(A.blocks) + 1]).all()     # no block's rows split between pieces
    rng = np.random.default_rng(4)
    y = rng.standard_normal(A.shape[0]).astype(np.float32)
    ref = csr.T @ y.astype(np.float64)
    split = A.first_offset_column()
    assert 0 < split < A.shape[1]
    out = {}
    for t in (1, 2, 3, 5):
        op = _make_table_operator(A, t)
        out[t] = np.asarray(op.rmatvec(y))
        assert out[t].dtype == np.float32
        np.testing.assert_allclose(out[t], ref, rtol=0, atol=1e-5 * np.abs(ref).max())
        for t0 in out:
            np.testing.assert_array_equal(_bits(out[t0][:split]), _bits(out[t][:split]))
        x = rng.standard_normal(A.shape[1])
        q, g = op.normal_product(x)
        np.testing.assert_allclose(q, csr @ x, rtol=0, atol=1e-5 * np.abs(csr @ x).max())
        np.testing.assert_allclose(g, csr.T @ np.asarray(q, dtype=np.float64), rtol=1e-12,
                                   atol=1e-12 * np.abs(g).max())
        np.testing.assert_allclose(op.rmatvec_promoted(y.astype(np.float64)), ref, rtol=1e-12,
                                   atol=1e-12 * np.abs(ref).max())


@pytest.mark.parametrize('kw', [dict(solver='cgls'), dict(solver='lsqr', n_threads=2),
                                dict(solver='cgls', gradient='recurrence'),
                                dict(solver='cgls', deflate=sc.Deflate(), sky_names=['continuum', 'line'])])
def test_the_solves_agree(tmp_path, kw):
    """``apply_lsqr`` on both orders: the same solution to the rounding of the sums over rows. Six iterations: past
    about ten, this small system's weakly seen directions amplify any rounding difference (the frame order at one
    thread and at three differs by 1e-2 at twelve), which would test the system, not the order."""
    frame, grouped = _pair(_frames(tmp_path), **_models()['variables'])
    base = dict(ref_shape=GRID, iter_lim=6, atol=0, btol=0, conlim=0, use_float32=True, n_threads=3,
                return_record=True, scratch_dir=str(tmp_path))
    base.update(kw)
    x_f, rec_f = apply_lsqr(frame.A, frame.b.copy(), active_mask=frame.active_mask,
                            num_cols_full=frame.active_mask.size, **base)
    x_p, rec_p = apply_lsqr(grouped.A, grouped.b.copy(), active_mask=grouped.active_mask,
                            num_cols_full=grouped.active_mask.size, **base)
    assert rec_f.iterations == rec_p.iterations
    np.testing.assert_allclose(x_p, x_f, rtol=0, atol=1e-4 * np.abs(x_f).max())
    np.testing.assert_allclose(rec_p.true_residual, rec_f.true_residual, rtol=1e-5)


def test_through_the_api(tmp_path):
    """``sc.Numerics(row_order='sky')`` with ``sc.Compute(matrix='compact')`` on a camera: the cal agrees with the
    frame order's to the rounding; refused with the stored matrix and for an unknown order."""
    from selfcal.io.calfile import CalFile
    from tests.synthetic_exposures import DET as CAM
    from tests.synthetic_exposures import N_CHUNK_SIDE, REF_ARCSEC, write_exposures
    rng = np.random.default_rng(5)
    paths, _offsets, _scalars = write_exposures(str(tmp_path / 'exposures'), 10, rng)
    cam = sc.Camera((CAM, CAM), chunks=(N_CHUNK_SIDE, N_CHUNK_SIDE), tag='Toy')
    cals = {}
    for order in ('frame', 'sky'):
        recipe = sc.Recipe(sc.continuum(smooth=0.0), fit=sc.Fit(20, tolerance=0), coadd=None,
                           numerics=sc.Numerics(2, batch=4, row_order=order), name=f'rows_{order}')
        field = sc.Field(tmp_path / f'run_{order}', cam, REF_ARCSEC,
                         compute=sc.Compute(workers=2, stage=None, matrix='compact'))
        field.reproject(paths)
        result = field.calibrate(recipe, jobs=sc.Job('All'))
        cals[order] = result.final or result.cal_paths[0]
    with CalFile(str(cals['frame'])) as a, CalFile(str(cals['sky'])) as b:
        sa, sb = np.asarray(a.sky(0)), np.asarray(b.sky(0))
        ok = np.isfinite(sa)
        np.testing.assert_array_equal(ok, np.isfinite(sb))
        np.testing.assert_allclose(sb[ok], sa[ok], rtol=0, atol=1e-4 * np.abs(sa[ok]).max())
        np.testing.assert_array_equal(a.sky_coverage(0), b.sky_coverage(0))
    field = sc.Field(tmp_path / 'run_stored', cam, REF_ARCSEC, compute=sc.Compute(workers=2, stage=None))
    field.reproject(paths)
    recipe = sc.Recipe(sc.continuum(smooth=0.0), fit=sc.Fit(5, tolerance=0), coadd=None,
                       numerics=sc.Numerics(2, batch=4, row_order='sky'), name='rows_stored')
    with pytest.raises(sc.ConfigError, match="matrix='compact'"):
        field.calibrate(recipe, jobs=sc.Job('All'))
    with pytest.raises(sc.ConfigError, match="row_order='pixel'"):
        sc.Numerics(row_order='pixel')


def test_refusals(tmp_path):
    paths = _frames(tmp_path)
    with pytest.raises(ValueError, match="matrix='compact'"):
        setup_lsqr(paths, GRID, row_order='sky', **_common())
    with pytest.raises(ValueError, match="'frame'"):
        setup_lsqr(paths, GRID, matrix='compact', row_order='pixel', **_common())
