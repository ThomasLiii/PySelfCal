"""Grouped constraint rows and per-map offset damping (no data).

``mean_offset_group_rows`` / ``group_adjacency_maps`` replace the k identical
per-frame copies of a det-grouped map's anchor / adjacency rows with one row
of weight ``w·√k``. That is exact in the normal equations (AᵀA, Aᵀb), which
is what these tests check: at the builder level against the per-frame form,
and end to end through ``setup_lsqr`` on a tiny synthetic problem.
``damp_offset_maps`` must reproduce the global ``damp_offset`` bit for bit when
every map gets the same weight, and leave weight-0 maps undamped.

Runnable as ``python tests/test_grouped_constraints.py`` or under pytest.
ALWAYS run from the repo root so ``import selfcal`` resolves to this worktree.
"""
import os
import tempfile

import h5py
import numpy as np
import scipy.sparse as sp
from astropy.wcs import WCS

from selfcal.core.blockcsr import ColSplitCSR
from selfcal.core.constraint_builders import grouped_adjacency_block, mean_offset_block
from selfcal.core.system import setup_lsqr

H = W = 8
F = 6
GROUPS = np.array([0, 0, 0, 1, 1, 1])                   # two det groups of 3
QUAD = (np.arange(H)[:, None] // 4) * 2 + (np.arange(W)[None, :] // 4)
HALVES = np.broadcast_to(np.arange(W)[None, :] // 4, (H, W)).copy()
QUAD_ADJ = (np.array([0, 0, 1, 2]), np.array([1, 2, 3, 3]))


def _block_matrix(blk, n_cols):
    return sp.csr_matrix((blk.data.astype(np.float64), (blk.rows_local, blk.cols)),
                         shape=(blk.num_rows, n_cols))


def _normal_equations(A, b):
    A = A.astype(np.float64)
    return (A.T @ A).toarray(), A.T @ np.asarray(b, dtype=np.float64)


# ---------------------------------------------------------------- builders

def test_mean_offset_group_rows_same_normal_equations():
    rng = np.random.default_rng(7)
    for trial in range(20):
        nf, nc, ng = int(rng.integers(4, 40)), int(rng.integers(2, 12)), int(rng.integers(1, 5))
        ftg = rng.integers(0, ng, nf)
        col_bases = [int(rng.integers(0, 50)), 0]
        n_cols = col_bases[0] + ng * nc
        # agreeing targets per group, except (on odd trials) one group that disagrees
        targets = (rng.integers(-4, 5, ng) / 4.0)[ftg]
        if trial % 2 and nf > 1:
            targets[0] += 0.5
        pf = mean_offset_block(0, targets, nf, nc, ftg, col_bases, weight=10.0)
        gr = mean_offset_block(0, targets, nf, nc, ftg, col_bases, weight=10.0, group_rows=True)
        ata_pf, atb_pf = _normal_equations(_block_matrix(pf, n_cols), pf.b)
        ata_gr, atb_gr = _normal_equations(_block_matrix(gr, n_cols), gr.b)
        np.testing.assert_allclose(ata_gr, ata_pf, rtol=1e-6, atol=1e-9)
        np.testing.assert_allclose(atb_gr, atb_pf, rtol=1e-6, atol=1e-9)
        assert gr.num_rows <= pf.num_rows


def test_mean_offset_group_rows_row_count():
    ftg = np.array([0, 0, 0, 1, 1, 2])
    col_bases = [0, 0]
    agree = np.zeros(6)
    assert mean_offset_block(0, agree, 6, 4, ftg, col_bases, group_rows=True).num_rows == 3
    # group 1's frames disagree -> its two frames keep their own rows
    disagree = np.array([0.0, 0.0, 0.0, 0.25, 0.5, 0.0])
    assert mean_offset_block(0, disagree, 6, 4, ftg, col_bases, group_rows=True).num_rows == 4


def test_grouped_adjacency_same_normal_equations():
    rng = np.random.default_rng(11)
    for _ in range(20):
        nf, nc, ng = int(rng.integers(3, 30)), int(rng.integers(3, 15)), int(rng.integers(1, 5))
        ftg = rng.integers(0, ng, nf)
        npair = int(rng.integers(1, 20))
        adj = (rng.integers(0, nc, npair), rng.integers(0, nc, npair))
        col_bases = [int(rng.integers(0, 50)), 0]
        n_cols = col_bases[0] + ng * nc
        rw = 0.1
        # the per-frame rows the worker would emit (assembly.py): rw*(O_i - O_j)
        rows, cols, data = [], [], []
        for f in range(nf):
            base = col_bases[0] + ftg[f] * nc
            for p in range(npair):
                r = f * npair + p
                rows += [r, r]
                cols += [base + adj[0][p], base + adj[1][p]]
                data += [rw, -rw]
        A_pf = sp.csr_matrix((np.float32(data).astype(np.float64), (rows, cols)),
                             shape=(nf * npair, n_cols))
        blk = grouped_adjacency_block(0, adj, rw, nc, ftg, col_bases)
        ata_pf, _ = _normal_equations(A_pf, np.zeros(nf * npair))
        ata_gr, _ = _normal_equations(_block_matrix(blk, n_cols), blk.b)
        np.testing.assert_allclose(ata_gr, ata_pf, rtol=1e-6, atol=1e-9)
        assert blk.num_rows == len(np.unique(ftg)) * npair


# ------------------------------------------------------ setup_lsqr, end to end

def _write_frame(path, sub_data):
    ys, xs = np.mgrid[0:H, 0:W]
    header = WCS(naxis=2).to_header().tostring()
    with h5py.File(path, 'w', libver='latest') as hf:
        hf.create_dataset('sub_data', data=sub_data.astype(np.float32))
        hf.create_dataset('sub_foot', data=np.ones((H, W), dtype=np.float16))
        hf.create_dataset('sub_bitmask', data=np.zeros((H, W), dtype=np.int32))
        hf.create_dataset('sub_mapping', data=np.stack([xs, ys]).astype(np.float32))
        hf.attrs['sub_header'] = header
        hf.attrs['det_header'] = header
        hf.attrs['file_path'] = os.path.basename(path)
        hf.attrs['ref_coords'] = np.array([0, H, 0, W], dtype=np.int32)


def _to_csr(A):
    if sp.issparse(A):
        return A.tocsr()
    if isinstance(A, ColSplitCSR):
        rows = []
        for b, per in enumerate(A.sub):
            nb = int(A.row_bounds[b + 1] - A.row_bounds[b])
            rows.append(sp.hstack(
                [sp.csr_matrix((d, i, ip), shape=(nb, int(A.cuts[t + 1] - A.cuts[t])))
                 for t, (d, i, ip) in enumerate(per)], format='csr'))
        return sp.vstack(rows, format='csr')
    return sp.vstack(A.blocks, format='csr')                     # BlockCSR


def _setup(**extra):
    """Two maps: a det-grouped quadrant map with adjacency + mean anchor, and a
    per-frame two-column map; per-frame scalar on; sky damping on."""
    rng = np.random.default_rng(3)
    with tempfile.TemporaryDirectory() as tmp:
        files = []
        for f in range(F):
            path = os.path.join(tmp, f"synth_{f}_det_{GROUPS[f]}.h5")
            _write_frame(path, rng.normal(size=(H, W)))
            files.append(path)
        kw = dict(chunk_maps=[QUAD, HALVES], apply_mask=False, apply_weight=False,
                  outlier_thresh=None, max_workers=1, batch_size=F,
                  offset_regularization=True, reg_weights=[0.1, 0.0],
                  adj_infos=[QUAD_ADJ, None],
                  mean_offsets_list=[np.full(F, 0.25), None],
                  det_groups_list=[GROUPS, None], use_per_frame_scalar=True,
                  weighted_damping=True, damp_weight=0.1)
        kw.update(extra)
        res = setup_lsqr(files, (H, W), **kw)
    return _to_csr(res.A), np.asarray(res.b), res.active_mask


def test_setup_lsqr_grouped_rows_same_normal_equations():
    A_pf, b_pf, act_pf = _setup()
    A_gr, b_gr, act_gr = _setup(mean_offset_group_rows=True, group_adjacency_maps=[0])
    assert (act_pf is None and act_gr is None) or np.array_equal(act_pf, act_gr)
    ata_pf, atb_pf = _normal_equations(A_pf, b_pf)
    ata_gr, atb_gr = _normal_equations(A_gr, b_gr)
    np.testing.assert_allclose(ata_gr, ata_pf, rtol=1e-6, atol=1e-9)
    np.testing.assert_allclose(atb_gr, atb_pf, rtol=1e-6, atol=1e-9)
    assert A_gr.nnz < A_pf.nnz


def test_damp_offset_maps_matches_global_damp_offset():
    A_g, b_g, _ = _setup(damp_offset=0.3)
    A_m, b_m, _ = _setup(damp_offset_maps=[0.3, 0.3])
    assert A_g.shape == A_m.shape
    for attr in ('data', 'indices', 'indptr'):
        assert np.array_equal(getattr(A_g, attr), getattr(A_m, attr))
    assert np.array_equal(b_g, b_m)


def test_damp_offset_maps_leaves_weight_zero_maps_free():
    A0, _, act = _setup()
    A1, _, _ = _setup(damp_offset_maps=[0.0, 0.3])
    extra = A1[A0.shape[0]:]                    # damping rows are appended last
    # full-layout columns of map 1 (HALVES, per frame): after sky + map 0
    n_sky, n_map0 = H * W, 2 * 4
    full_cols = np.flatnonzero(act) if act is not None else np.arange(A1.shape[1])
    touched = full_cols[np.unique(extra.indices)]
    assert extra.shape[0] == F * 2              # every map-1 column, nothing else
    assert touched.min() >= n_sky + n_map0 and touched.max() < n_sky + n_map0 + F * 2


def _raises(fn, exc=ValueError):
    try:
        fn()
    except exc:
        return True
    return False


def test_validation():
    assert _raises(lambda: _setup(damp_offset_maps=[0.3]))                  # length
    assert _raises(lambda: _setup(damp_offset_maps=[0.3, -1.0]))            # sign
    assert _raises(lambda: _setup(damp_offset=0.3, damp_offset_maps=[0, 0.3]))
    assert _raises(lambda: _setup(group_adjacency_maps=[2]))                # index


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")


if __name__ == '__main__':
    _run_all()
