"""The deflation's passes read each row chunk in pieces of one kind of column (the sky; the offsets, with the columns
of the terms Deflate(offsets=) leaves out apart from the others), taken out of the storage's column
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


def test_an_offset_piece_with_columns_outside_the_space_keeps_its_entries():
    """The deflation's space leaves out the columns without data (a coverage floor's zeros), here the first, a middle
    and the last offset column: an offset piece (the columns past the storage's first offset column) is mapped through
    the column map, and its offset entries, columns and values, are those the whole chunk sorted by kind gives; a chunk
    with an entry at a left-out column (a zero) is sorted as a whole chunk is. The sky piece likewise."""
    rng = np.random.default_rng(2)
    nr, n_sky, n = 40, 10, 20
    colmap = np.full(n, -1, dtype=D._IDX)
    keep = np.array([c for c in range(n) if c not in (3, 10, 14, 19)])
    colmap[keep] = np.arange(keep.size)
    n_sky_S = int(np.count_nonzero(keep < n_sky))
    for with_zero in (False, True):
        rows = []
        for i in range(nr):
            c = np.sort(np.concatenate([rng.choice(n_sky, 1), rng.choice([11, 12, 13, 15, 16, 17, 18], 3, replace=False)]))
            if with_zero and i == 7:
                c = np.sort(np.append(c, 14))
            rows.append(c)
        ip = np.concatenate([[0], np.cumsum([r.size for r in rows])]).astype(D._IDX)
        cols = np.concatenate(rows).astype(D._IDX)
        vals = rng.standard_normal(cols.size).astype(np.float32)
        vals[~np.isin(cols, keep)] = 0.0                              # a left-out column holds zeros
        _l, _p, _d, sky_m, off_m, _o = D._prepare_piece(nr, ip, cols, vals, colmap, n_sky_S, buf=D._Slot())
        for lo, hi in ((0, n_sky), (n_sky, n)):
            sel = (cols >= lo) & (cols < hi)
            pip = np.concatenate([[0], np.cumsum([np.count_nonzero((r >= lo) & (r < hi)) for r in rows])]).astype(D._IDX)
            _l, _p, _d, sky_p, off_p, oid = D._prepare_piece(nr, pip, cols[sel], vals[sel], colmap, n_sky_S,
                                                              buf=D._Slot(), span=(lo, hi), sky_end=n_sky)
            if lo == 0:
                assert off_p is None and oid is None
                assert np.array_equal(sky_p[1][sky_p[2] != 0], sky_m[1][sky_m[2] != 0])
                continue
            assert (oid is None) == with_zero                          # a zero entry: the piece sorted as a chunk
            for a, b in zip(off_p, off_m):
                assert np.array_equal(a, b)
            if oid is not None:
                assert oid == (0, keep.size - n_sky_S) and off_p[1].min() >= oid[0] and off_p[1].max() < oid[1]


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


def _whole_ranges(monkeypatch):
    whole = D._piece_spans
    monkeypatch.setattr(D, '_piece_spans', lambda storage, cuts=None: whole(storage))


def _solve_both(tmp_path, monkeypatch, fit, between=_whole_ranges):
    """The toy of test_offset_rows (a term every frame shares, then two per-frame terms) solved with ``fit`` on the
    partitioned storage twice, ``between(monkeypatch)`` changing the deflation between the two (by default: the passes
    reading pieces of one kind, as they do, then reading each column range whole: ``_piece_spans`` without its cuts,
    every entry sorted by kind in the passes). The two cals' solved datasets and the two mosaics' maps."""
    field = _field(tmp_path)
    recipe = _recipe('all').replace(fit=fit)
    for k, v in PARTITIONED.items():
        monkeypatch.setenv(k, v)
    r1 = field.calibrate(recipe, jobs=sc.Job('All'))
    between(monkeypatch)
    r2 = field.calibrate(recipe.replace(name='rows_all_second'), jobs=sc.Job('All'))
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
