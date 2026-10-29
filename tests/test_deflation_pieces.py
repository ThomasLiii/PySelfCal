"""The deflation's passes read each row chunk in pieces of one kind of column (the sky; the offsets, with the columns
groups of frames share apart from the others when those are solved iteratively), taken out of the storage's column
ranges (:func:`selfcal.core.deflation._piece`, :func:`selfcal.core.deflation._piece_spans`). A piece holds exactly the
stored entries of its columns, in storage order, so the deflated products are, byte for byte, those of the passes
reading each range whole and sorting its entries by kind."""
import h5py
import numpy as np
from astropy.io import fits
from scipy.sparse import csr_matrix

import selfcal as sc
import selfcal.core.deflation as D
from selfcal.core.blockcsr import BlockCSR, ColSplitCSR

from test_offset_rows import _field, _recipe  # noqa: E402

PARTITIONED = {'SELFCAL_SCATTER_WORKERS': '2', 'SELFCAL_BLOCK_NNZ': '1'}


def _random_csr(rng, nr, nc, per_row):
    rows = [np.sort(rng.choice(nc, size=per_row, replace=False)) for _ in range(nr)]
    ip = np.arange(nr + 1, dtype=np.int32) * per_row
    idx = np.concatenate(rows).astype(np.int32)
    return csr_matrix((rng.standard_normal(idx.size).astype(np.float32), idx, ip), shape=(nr, nc))


def test_a_piece_holds_the_stored_entries_of_its_columns():
    """A chunk of a csr matrix, of a BlockCSR and of a ColSplitCSR with two column ranges, read in the pieces between
    the cuts: each piece's rows hold the stored entries in its columns, in storage order (with and without buffers)."""
    rng = np.random.default_rng(1)
    nr, nc = 37, 50
    A = _random_csr(rng, nr, nc, 6)
    cuts = [0, 20, 33, 50]
    # the ColSplitCSR: one block, column ranges 0:25 and 25:50 (local column ids)
    sub = []
    for a, b in ((0, 25), (25, 50)):
        S = A[:, a:b].tocsr()
        sub.append((S.data, S.indices.astype(np.int32), S.indptr.astype(np.int32)))
    stores = {'csr': A, 'blocks': BlockCSR([A[:20].tocsr(), A[20:].tocsr()], np.array([0, 20, nr]), (nr, nc)),
              'split': ColSplitCSR([sub], np.array([0, nr]), np.array([0, 25, 50]), (nr, nc))}
    for name, st in stores.items():
        spans = D._piece_spans(st, cuts)
        for desc in D._piece_index(st, rows=16):
            for buf in (None, D._Slot()):
                r0, r1, pieces = D._piece(st, desc, buf=buf, spans=spans)
                assert len(pieces) == len(spans), name
                for (lip, cols, vals), (lo, hi, _t) in zip(pieces, spans):
                    want = A[r0:r1].tocsr()
                    for i in range(r1 - r0):
                        sl = slice(want.indptr[i], want.indptr[i + 1])
                        keep = (want.indices[sl] >= lo) & (want.indices[sl] < hi)
                        got = slice(int(lip[i]), int(lip[i + 1]))
                        assert np.array_equal(cols[got], want.indices[sl][keep]), name
                        assert np.array_equal(vals[got], want.data[sl][keep]), name


def _solved(cal):
    """Every solved dataset of a cal file (the sky maps, the offsets and their coverages), by path."""
    out = {}
    with h5py.File(cal, 'r') as f:
        def visit(name, obj):
            top = name.split('/')[0]
            if isinstance(obj, h5py.Dataset) and (top.startswith('sky') or top.startswith('offset')):
                out[name] = obj[()]
        f.visititems(visit)
    return out


def _solve_both(tmp_path, monkeypatch, fit):
    """The toy of test_offset_rows (a term every frame shares, then two per-frame terms) solved with ``fit`` on the
    partitioned storage twice: the passes reading pieces of one kind (as they do), and reading each column range whole
    (``_piece_spans`` without its cuts: every entry sorted by kind in the passes). The two cals' solved datasets and the
    two mosaics' maps."""
    field = _field(tmp_path)
    recipe = _recipe('all').replace(fit=fit)
    for k, v in PARTITIONED.items():
        monkeypatch.setenv(k, v)
    r1 = field.calibrate(recipe, jobs=sc.Job('All'))
    whole = D._piece_spans
    monkeypatch.setattr(D, '_piece_spans', lambda storage, cuts=None: whole(storage))
    r2 = field.calibrate(recipe.replace(name='rows_all_whole'), jobs=sc.Job('All'))
    out = []
    for r in (r1, r2):
        (cal,), (mos,) = r.cal_paths, r.mosaic_paths
        with fits.open(mos) as h:
            maps = {k: np.array(h[k].data) for k in ('MEAN_MAP', 'MEAN_MAP_WEIGHT')}
        out.append((_solved(cal), maps))
    return out


def _assert_same(a, b):
    (cal1, maps1), (cal2, maps2) = a, b
    assert set(cal1) == set(cal2)
    for k, v in cal1.items():
        assert np.array_equal(v, cal2[k], equal_nan=True), k
    for k, v in maps1.items():
        assert np.array_equal(v, maps2[k], equal_nan=True), k


def test_the_deflated_products_do_not_depend_on_the_pieces(tmp_path, monkeypatch):
    """With the deflation: the same bytes. The shared term's columns are few: the dense Schur path, whose offsets'
    normal matrix is made from each chunk's offset entries joined whole."""
    from test_any_telescope import FIT
    _assert_same(*_solve_both(tmp_path, monkeypatch, FIT.replace(deflate=sc.Deflate(3, basis='polynomial'))))


def test_the_iterative_shared_solve_does_not_depend_on_the_pieces(tmp_path, monkeypatch):
    """The shared columns' iterative solve forced (every shared column a hub): its sweeps read the shared and the other
    offset columns' pieces (no sky piece), taken by a shift; with the ranges whole, by compression. The same bytes."""
    from test_any_telescope import FIT
    monkeypatch.setattr(D, '_SCHUR_MAX', 0)
    _assert_same(*_solve_both(tmp_path, monkeypatch, FIT.replace(deflate=sc.Deflate(3, basis='polynomial'))))
