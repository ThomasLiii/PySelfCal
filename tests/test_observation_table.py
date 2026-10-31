"""The observation table (``setup_lsqr(matrix='compact')``, ``sc.Compute(matrix='compact')``): the matrix kept as
its observations and made again when read (:mod:`selfcal.core.observations`).

The contract is byte equality with the stored matrix: on synthetic frames written with
:func:`selfcal.io.frames.write_frame` (no instrument), the rows the table makes are the stored matrix's entry for
entry, the right-hand side and the pixel state are the same bytes, and so are the start vector, the column norms and
the solutions of ``apply_lsqr`` (CGLS and LSQR, preconditioned or not, with the deflation, with column constraints,
with the coverage floors, in float32 and float64, at one and several threads). Models with a sky coefficient, an
offset basis, grouped offsets, frame and sky variables and a constraint block are covered; what the table refuses is
checked too.
"""

import numpy as np
import pytest
import scipy.sparse as sp

pytest.importorskip('numba')

import selfcal as sc  # noqa: E402
from selfcal import _state  # noqa: E402
from selfcal.core.blockcsr import ColSplitCSR  # noqa: E402
from selfcal.core.deflation import _piece, _piece_index, _piece_spans  # noqa: E402
from selfcal.core.observations import ObservationTable  # noqa: E402
from selfcal.core.solution import compute_x0_from_Ab, compute_x0_offsets_fit  # noqa: E402
from selfcal.core.solve import _column_squares, apply_lsqr  # noqa: E402
from selfcal.core.system import setup_lsqr  # noqa: E402
from selfcal.io.frames import standard_frame_path, write_frame  # noqa: E402
from selfcal.models.offset_model import Basis  # noqa: E402
from selfcal.models.sky_model import Coefficient, SkyComponent, SkyModel  # noqa: E402
from selfcal.models.variables import Derived, VariableSet  # noqa: E402

H, W, F = 10, 12, 7
GRID = (H + F - 1, W + 3)
N_SKY = GRID[0] * GRID[1]
DET = (H, W)


def _frames(tmp_path, *, zero_rows=False, seed=3):
    """F frames, frame k on the grid rows k..k+H and columns (k % 3)..: a smooth value plus a per-frame offset and a
    gradient; the detector coordinates run across the whole detector at sub-pixel offsets (bilinear corners of
    four pixels, some at the detector's edge); frame 1 has a non-finite value and an outlier; ``zero_rows``: frame 2
    has a column of exact zeros (rows whose right-hand side is zero)."""
    _state.set_progress(False)
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    paths = []
    for k in range(F):
        vals = (1.0 + 0.05 * k + 0.01 * (y + k) + 0.002 * x + 1e-3 * rng.standard_normal((H, W))).astype(np.float32)
        if k == 1:
            vals[0, 2] = np.nan
            vals[0, 5] = 1e3
        if zero_rows and k == 2:
            vals[:, 4] = 0.0
        # sub-pixel detector positions, shifted per frame; a few land beyond the detector's last row/column
        mapping = np.stack([x + np.float32(0.3 + 0.1 * k), y + np.float32(0.25 * (k % 4))])
        c0 = k % 3
        paths.append(write_frame(standard_frame_path(str(tmp_path), k, 0), vals, [k, k + H, c0, c0 + W], mapping))
    return paths


def ramp(det_x):
    return det_x / W


def hats(det_x):
    """Two hat functions of the detector column: zero over much of the detector."""
    t = det_x / W * 2.0
    return np.stack([np.maximum(0.0, 1.0 - np.abs(t - 0.5)), np.maximum(0.0, 1.0 - np.abs(t - 1.5))], axis=1)


def hats6(det_x):
    """Six hat functions across the detector column: at most two are nonzero at any position."""
    t = det_x / W * 6.0
    return np.stack([np.maximum(0.0, 1.0 - np.abs(t - (k + 0.5))) for k in range(6)], axis=1)


def times_t(t):
    return 1.0 + 0.1 * t


def arc_coef(arc):
    return 0.5 + arc


def arc_times_t(a, t):
    return a * times_t(t)


def tilt_plus_x(t, x):
    return t + 0.01 * x


def one(frame):
    return 1.0


def _common(**kw):
    chunks = np.repeat(np.arange(4), W // 4)[None, :].repeat(H, axis=0).astype(np.int32)     # 4 column chunks
    base = dict(chunk_maps=[chunks], grid_valid_weight=np.ones(DET, np.float32), apply_mask=True, ignore_list=[],
                max_workers=2, batch_size=3, outlier_thresh=3)
    base.update(kw)
    return base


def _both(paths, partitioned=None, **kw):
    """The same system built stored and compact. ``partitioned=(monkeypatch, tmp_path)``: the stored matrix in the
    partitioned block storage of the large runs (one block per assembly batch, the table's blocks), which the
    deflation's per-chunk sums follow."""
    kw = _common(**kw)
    if partitioned is not None:
        monkeypatch, tmp_path = partitioned
        monkeypatch.setenv('SELFCAL_BLOCK_NNZ', '1')
        monkeypatch.setenv('SELFCAL_SCATTER_WORKERS', '2')
        monkeypatch.setenv('SELFCAL_RMATVEC_SPLIT', '1')
        kw['batch_spill_dir'] = str(tmp_path / 'spill')
    stored = setup_lsqr(paths, GRID, **kw)
    compact = setup_lsqr(paths, GRID, matrix='compact', **kw)
    assert isinstance(compact.A, ObservationTable)
    if partitioned is not None:
        assert isinstance(stored.A, ColSplitCSR) and stored.A.nblocks == compact.A.nblocks
    return stored, compact


def _to_csr(A):
    """Any of the build's storages as one csr_matrix, rows in order, canonical."""
    if isinstance(A, ObservationTable):
        return A.to_csr()
    if sp.issparse(A):
        out = A.tocsr()
    elif isinstance(A, ColSplitCSR):
        rows = []
        for b, per in enumerate(A.sub):
            nb = int(A.row_bounds[b + 1] - A.row_bounds[b])
            rows.append(sp.hstack([sp.csr_matrix((d, i, ip), shape=(nb, int(A.cuts[t + 1] - A.cuts[t])))
                                   for t, (d, i, ip) in enumerate(per)], format='csr'))
        out = sp.vstack(rows, format='csr')
    else:
        out = sp.vstack(A.blocks, format='csr')
    out.sort_indices()
    return out


def _same_matrix(stored, compact):
    """The table's rows are the stored matrix's: columns, values (bitwise), layout, the right-hand side, the pixel
    state and the active columns."""
    a, t = _to_csr(stored.A), _to_csr(compact.A)
    assert a.shape == t.shape, (a.shape, t.shape)
    np.testing.assert_array_equal(a.indptr, t.indptr)
    np.testing.assert_array_equal(a.indices, t.indices)
    assert a.data.dtype == t.data.dtype == np.float32
    np.testing.assert_array_equal(a.data.view(np.uint32), t.data.view(np.uint32))
    assert stored.b.dtype == compact.b.dtype
    np.testing.assert_array_equal(stored.b.view(np.uint32 if stored.b.dtype == np.float32 else np.uint64),
                                  compact.b.view(np.uint32 if stored.b.dtype == np.float32 else np.uint64))
    np.testing.assert_array_equal(stored.pixel_counts, compact.pixel_counts)
    np.testing.assert_array_equal(stored.pixel_fisher.view(np.uint64), compact.pixel_fisher.view(np.uint64))
    if stored.pixel_rhs is not None:
        np.testing.assert_array_equal(stored.pixel_rhs.view(np.uint64), compact.pixel_rhs.view(np.uint64))
    if stored.pixel_cross is not None:
        sc_, cc_ = stored.pixel_cross, compact.pixel_cross
        if isinstance(sc_, dict):
            assert sc_.keys() == cc_.keys()
            for k in sc_:
                np.testing.assert_array_equal(sc_[k].view(np.uint64), cc_[k].view(np.uint64))
        else:
            np.testing.assert_array_equal(sc_.view(np.uint64), cc_.view(np.uint64))
    if stored.active_mask is None:
        assert compact.active_mask is None
    else:
        np.testing.assert_array_equal(stored.active_mask, compact.active_mask)


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint32 if a.dtype == np.float32 else np.uint64)


def _same_solve(stored, compact, tmp_path, residual=True, **kw):
    """``apply_lsqr`` on both gives the same bytes (and the same record of the stop and, with ``residual``, the true
    residual: a column-partitioned stored matrix casts a float64 iterate to float32 before the product that measures
    it, where a csr matrix, a BlockCSR and the table form the products in float64 as scipy does, so the residuals
    of those two differ in their last digits)."""
    kw = dict(dict(ref_shape=GRID, iter_lim=30, atol=0, btol=0, conlim=0, solver='cgls', use_float32=True,
                   n_threads=3, return_record=True, scratch_dir=str(tmp_path)), **kw)
    x_s, rec_s = apply_lsqr(stored.A, stored.b.copy(), active_mask=stored.active_mask,
                            num_cols_full=None if stored.active_mask is None else stored.active_mask.size, **kw)
    x_c, rec_c = apply_lsqr(compact.A, compact.b.copy(), active_mask=compact.active_mask,
                            num_cols_full=None if compact.active_mask is None else compact.active_mask.size, **kw)
    assert x_s.dtype == x_c.dtype
    np.testing.assert_array_equal(_bits(x_s), _bits(x_c))
    assert rec_s.iterations == rec_c.iterations and rec_s.istop == rec_c.istop
    if residual:
        np.testing.assert_equal(rec_s.true_residual, rec_c.true_residual)
    else:
        np.testing.assert_allclose(rec_s.true_residual, rec_c.true_residual, rtol=1e-5)
    return x_s


# =================================================================== the matrix
def test_the_rows_are_the_stored_matrix(tmp_path):
    """One constant sky term and one chunk map: the entries, the right-hand side and the pixel state."""
    stored, compact = _both(_frames(tmp_path))
    _same_matrix(stored, compact)
    assert compact.A.nnz == _to_csr(stored.A).nnz
    assert compact.A.shape == stored.A.shape


def test_zero_right_hand_sides_keep_the_stored_structure(tmp_path):
    """Rows whose right-hand side is exactly zero drop their zero entries, as the build drops them, and no other."""
    stored, compact = _both(_frames(tmp_path, zero_rows=True), outlier_thresh=None)
    assert any(b.zb.size for b in compact.A.blocks)
    _same_matrix(stored, compact)


def test_sky_coefficient_basis_groups_and_variables(tmp_path, monkeypatch):
    """Two sky terms (the second with a coefficient of a detector map and a frame variable through a derived
    variable), a basis of two functions on one map and a plain map shared by frame groups, the cross and right-hand
    side moments on: the same system."""
    paths = _frames(tmp_path)
    arc = (np.linspace(0, 1, H * W, dtype=np.float32).reshape(DET) * 3).astype(np.float32)
    pix = (np.arange(H * W) % 5).reshape(DET).astype(np.int32)
    chunks = _common()['chunk_maps'][0]
    sky_model = SkyModel(components=(SkyComponent('continuum'),
                                     SkyComponent('line', coefficient=Coefficient(('arc_t',), arc_coef))))
    variables = VariableSet(frame={'t': np.arange(F, dtype=np.float64)},
                            derived={'arc_t': Derived(('arc', 't'), arc_times_t)})
    kw = dict(chunk_maps=[chunks, pix], det_groups_list=[None, np.array([0, 0, 1, 1, 2, 2, 2])],
              basis_list=[Basis(Coefficient('det_x', hats), n=2), None], det_aux=[arc], aux_keys=['arc'],
              sky_model=sky_model, variables=variables, sky_rhs_moments=True)
    stored, compact = _both(paths, **kw)
    _same_matrix(stored, compact)
    _same_solve(stored, compact, tmp_path)
    # the deflation sums per chunk of each storage block: the stored matrix in the partitioned storage, whose
    # blocks are the table's (a column-partitioned matrix is consumed by its operator: built for this solve)
    stored, compact = _both(paths, partitioned=(monkeypatch, tmp_path), **kw)
    _same_matrix(stored, compact)
    _same_solve(stored, compact, tmp_path, residual=False, deflate=sc.Deflate(), sky_names=['continuum', 'line'],
                iter_lim=12)


def test_stored_functions(tmp_path):
    """``functions='stored'``: the model's values kept with the table give the matrix, the floors and the solve of
    the values made at every pass; a basis that is mostly zero is packed and a dense one and the sky coefficients
    kept as they are; the setting is refused without the table."""
    paths = _frames(tmp_path)
    arc = (np.linspace(0, 1, H * W, dtype=np.float32).reshape(DET) * 3).astype(np.float32)
    pix = (np.arange(H * W) % 5).reshape(DET).astype(np.int32)
    chunks = _common()['chunk_maps'][0]
    sky_model = SkyModel(components=(SkyComponent('continuum'),
                                     SkyComponent('line', coefficient=Coefficient('arc', arc_coef))))
    kw = dict(chunk_maps=[chunks, pix], det_groups_list=[None, np.array([0, 0, 1, 1, 2, 2, 2])],
              basis_list=[Basis(Coefficient('det_x', hats6), n=6), Basis(Coefficient('det_x', ramp), n=1)],
              det_aux=[arc], aux_keys=['arc'], sky_model=sky_model)
    for floor in ({}, dict(min_observations=2)):
        computed = setup_lsqr(paths, GRID, matrix='compact', **_common(**kw, **floor))
        stored = setup_lsqr(paths, GRID, matrix='compact', functions='stored', **_common(**kw, **floor))
        assert computed.A.functions == 'computed' and stored.A.functions == 'stored'
        assert all(b.values is None for b in computed.A.blocks) and all(b.values is not None for b in stored.A.blocks)
        for b in stored.A.blocks:
            v = b.values
            assert v.sky_src.tolist() == [-1, 0] and v.sky_c.shape == (1, b.n)       # the identity term is the weight
            assert v.pk_nb.tolist() == [6] and v.pk_base.tolist() == [0] and 1 <= int(v.pk_off[1]) <= 2
            assert v.dense_fn.tolist() == [6] and v.dense.shape == (1, b.n)
            assert v.pidx.shape == v.pval.shape == (b.n, int(v.pk_off[1]))
            assert np.all((v.pidx == 255) | (v.pidx < 6)) and np.all(v.pval[v.pidx != 255] != 0)
        assert stored.A.values_nbytes > 0 and stored.A.nbytes == computed.A.nbytes + stored.A.values_nbytes
        _same_matrix(computed, stored)
        np.testing.assert_array_equal(np.concatenate([b.fl for b in computed.A.blocks]),
                                      np.concatenate([b.fl for b in stored.A.blocks]))
        _same_solve(computed, stored, tmp_path)
    with pytest.raises(ValueError, match="matrix='compact'"):
        setup_lsqr(paths, GRID, functions='stored', **_common())
    with pytest.raises(ValueError, match="'computed'"):
        setup_lsqr(paths, GRID, matrix='compact', functions='cached', **_common())
    with pytest.raises(sc.ConfigError, match="matrix='compact'"):
        sc.Compute(functions='stored')
    assert sc.Compute(matrix='compact', functions='stored').functions == 'stored'


def test_sky_variable_and_constant_sky_coefficient(tmp_path):
    """A coefficient reading a reference-grid map at the observation's pixel, and the sky coordinates."""
    paths = _frames(tmp_path)
    tilt = np.linspace(-1, 1, N_SKY, dtype=np.float64).reshape(GRID)
    sky_model = SkyModel(components=(SkyComponent('continuum'),
                                     SkyComponent('slope', coefficient=Coefficient(('tilt', 'sky_x'), tilt_plus_x))))
    stored, compact = _both(paths, sky_model=sky_model, variables=VariableSet(sky={'tilt': tilt}))
    _same_matrix(stored, compact)
    _same_solve(stored, compact, tmp_path)


PIXELS = dict(chunk_maps=[np.arange(H * W).reshape(DET).astype(np.int32)], det_groups_list=[np.zeros(F, int)])


@pytest.mark.parametrize('floor', [dict(min_observations=2), dict(offset_min_observations=[16], **PIXELS),
                                   dict(min_observations=2, offset_min_observations=[12], **PIXELS)])
def test_the_coverage_floors(tmp_path, floor):
    """The floored rows keep their entries at +0.0 and their right-hand side at 0, as the stored placement leaves
    them; the pixel state and the active columns agree (the offset floor on one constant per detector pixel shared
    by every frame: the pixels few frames see)."""
    stored, compact = _both(_frames(tmp_path), **floor)
    assert any(b.fl.size for b in compact.A.blocks)
    _same_matrix(stored, compact)
    _same_solve(stored, compact, tmp_path)


def test_constraint_rows_and_an_uncompacted_system(tmp_path):
    """Sky damping rows (a constraint block stored after the data rows); a prior row on a column without data
    leaves the compaction to the solver (no column map): both agree with the stored path."""
    paths = _frames(tmp_path)
    stored, compact = _both(paths, weighted_damping=True, damp_weight=0.1)
    assert compact.A.tail is not None and compact.A.tail.shape[0] == int((stored.pixel_counts[:N_SKY] > 0).sum())
    _same_matrix(stored, compact)
    _same_solve(stored, compact, tmp_path)

    empty = int(np.flatnonzero(stored.pixel_counts == 0)[0])     # a column without data

    def prior(info):
        return (np.array([0, 0]), np.array([empty, 0]), np.array([1.0, -1.0]), np.array([0.0]))

    stored_u, compact_u = _both(paths, priors=[prior])
    assert stored_u.active_mask is None and compact_u.active_mask is None
    _same_matrix(stored_u, compact_u)
    _same_solve(stored_u, compact_u, tmp_path)


# =================================================================== the solver's reads
@pytest.mark.parametrize('solver,float32,threads,precondition', [
    ('cgls', True, 3, True), ('cgls', True, 1, True), ('lsqr', True, 3, True), ('cgls', False, 3, True),
    ('lsqr', False, 2, False), ('cgls', True, 3, False), ('lsmr', True, 2, True)])
def test_solves_agree(tmp_path, monkeypatch, solver, float32, threads, precondition):
    monkeypatch.setenv('SELFCAL_PARALLEL_RMATVEC', 'auto')
    stored, compact = _both(_frames(tmp_path))
    _same_solve(stored, compact, tmp_path, solver=solver, use_float32=float32, n_threads=threads, precondition=precondition)


def test_sequential_transpose_product(tmp_path, monkeypatch):
    monkeypatch.setenv('SELFCAL_PARALLEL_RMATVEC', '1')
    stored, compact = _both(_frames(tmp_path))
    _same_solve(stored, compact, tmp_path)


def test_deflation_exclusions_and_start_vectors(tmp_path, monkeypatch):
    """The deflation's passes, the hard column constraints and both start vectors read the table as they read a
    stored matrix (the deflation sums per chunk of each storage block: the stored matrix in the partitioned
    storage, whose blocks are the table's)."""
    paths = _frames(tmp_path)
    groups = dict(det_groups_list=[np.array([0, 0, 0, 1, 1, 1, 1])])
    stored, compact = _both(paths, **groups)
    for fn in (compute_x0_from_Ab, compute_x0_offsets_fit):
        x_s = fn(stored.A, stored.b, GRID, active_mask=stored.active_mask)
        x_c = fn(compact.A, compact.b, GRID, active_mask=compact.active_mask)
        np.testing.assert_array_equal(_bits(x_s), _bits(x_c))
    # a hard constraint: the two groups' chunk constants carry no mean (one group per row of ``cols``)
    layout_cols = np.arange(N_SKY, N_SKY + 8).reshape(2, 4)
    P = np.full((4, 1), 0.5)
    excl = [(layout_cols, P)]
    _same_solve(stored, compact, tmp_path, exclusions=excl)
    _same_solve(stored, compact, tmp_path, exclusions=[(layout_cols, P, 'information')], precondition=False)
    # the deflated solves against the partitioned storage (consumed by each solve: built anew for each)
    for kw in (dict(), dict(exclusions=excl), dict(solver='lsqr')):
        stored, compact = _both(paths, partitioned=(monkeypatch, tmp_path), **groups)
        _same_solve(stored, compact, tmp_path, residual=False, deflate=sc.Deflate(), sky_names=['sky'],
                    iter_lim=12, **kw)


def test_column_norms_and_chunk_readers(tmp_path):
    """The column norms over 64M-entry chunks (one entry here, so every chunk straddles rows and blocks), the
    column squares, and the deflation's chunk reader with column spans."""
    stored, compact = _both(_frames(tmp_path))
    a = _to_csr(stored.A)
    np.testing.assert_array_equal(_bits(_column_squares(stored.A, a.shape[1])),
                                  _bits(_column_squares(compact.A, a.shape[1])))
    # entry chunks: the concatenation is the entry stream
    chunks = list(compact.A.entry_chunks(7))
    d = np.concatenate([p[0] for ch in chunks for p in ch])
    i = np.concatenate([p[1] for ch in chunks for p in ch])
    np.testing.assert_array_equal(_bits(d), _bits(a.data))
    np.testing.assert_array_equal(i, a.indices)
    assert all(sum(p[0].size for p in ch) == 7 for ch in chunks[:-1])
    # the deflation's pieces, with spans
    n_sky_c = int(compact.active_mask[:N_SKY].sum())
    cuts = [0, n_sky_c, n_sky_c + 5, a.shape[1]]
    for storage in (stored.A, compact.A):
        rows = []
        for desc in _piece_index(storage, rows=16):
            r0, r1, pieces = _piece(storage, desc, spans=_piece_spans(storage, cuts))
            parts = []
            for lip, cols, vals in pieces:
                parts.append(sp.csr_matrix((vals, cols, lip), shape=(r1 - r0, a.shape[1])))
            rows.append(sum(parts[1:], parts[0]))
        m = sp.vstack(rows, format='csr')
        m.sort_indices()
        np.testing.assert_array_equal(m.indptr, a.indptr)
        np.testing.assert_array_equal(m.indices, a.indices)
        np.testing.assert_array_equal(_bits(m.data), _bits(a.data))


# =================================================================== refusals and the API
def test_refusals(tmp_path):
    paths = _frames(tmp_path)
    with pytest.raises(ValueError, match='adjacency'):
        setup_lsqr(paths, GRID, matrix='compact', **_common(offset_regularization=True, reg_weights=[0.5],
                                                             adj_infos=[(np.array([0, 1]), np.array([1, 2]))]))
    with pytest.raises(ValueError, match='frame functions'):
        from selfcal.models.variables import FrameFunction
        sky_model = SkyModel(components=(SkyComponent('continuum'),
                                         SkyComponent('x', coefficient=Coefficient('ff', ramp))))
        setup_lsqr(paths, GRID, matrix='compact', sky_model=sky_model,
                   variables=VariableSet(frame_functions={'ff': FrameFunction(one)}), **_common())
    with pytest.raises(ValueError, match="'stored'"):
        setup_lsqr(paths, GRID, matrix='sparse', **_common())
    with pytest.raises(sc.ConfigError, match="'stored', 'compact'"):
        sc.Compute(matrix='dense')


@pytest.mark.parametrize('edge', ['partial', 'inside'])
def test_through_the_api(tmp_path, edge):
    """``sc.Compute(matrix='compact')`` on a camera: the cal's solution and pixel state are the stored run's, with
    the observations at the window's edge kept (``window_edge='partial'``) or left out (``'inside'``)."""
    from selfcal.io.calfile import CalFile
    from tests.synthetic_exposures import DET as CAM
    from tests.synthetic_exposures import N_CHUNK_SIDE, REF_ARCSEC, write_exposures
    rng = np.random.default_rng(5)
    paths, _offsets, _scalars = write_exposures(str(tmp_path / 'exposures'), 10, rng)
    cam = sc.Camera((CAM, CAM), chunks=(N_CHUNK_SIDE, N_CHUNK_SIDE), tag='Toy')
    recipe = sc.Recipe(sc.continuum(smooth=0.0), fit=sc.Fit(20, tolerance=0, min_observations=2, window_edge=edge),
                       coadd=None, numerics=sc.Numerics(2, batch=4), name=f'table_{edge}')
    cals = {}
    for mode in ('stored', 'compact'):
        field = sc.Field(tmp_path / f'run_{mode}', cam, REF_ARCSEC,
                         compute=sc.Compute(workers=2, stage=None, matrix=mode))
        field.reproject(paths)
        result = field.calibrate(recipe, jobs=sc.Job('All'))
        cals[mode] = result.final or result.cal_paths[0]
    with CalFile(str(cals['stored'])) as a, CalFile(str(cals['compact'])) as b:
        np.testing.assert_array_equal(_bits(a.sky(0)), _bits(b.sky(0)))
        np.testing.assert_array_equal(a.sky_coverage(0), b.sky_coverage(0))
        for oa, ob in zip(a.offsets, b.offsets):
            np.testing.assert_array_equal(_bits(np.asarray(oa)), _bits(np.asarray(ob)))


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-x', '-q']))
