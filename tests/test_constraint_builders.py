"""Phase 3c unit tests: global constraint builders reproduce the legacy blocks.

The cal gates exercise mean-offset + continuum damping on real data, but NOT
line damping (production runs damp_weight_line=0) or offset damping
(damp_offset=0). These tests check all four builder paths against the exact
inline formulas that used to live in lsqr.setup_lsqr.

Runnable as ``python tests/test_constraint_builders.py`` or under pytest.
"""
import numpy as np

from selfcal.constraint_builders import (ConstraintBlock, mean_offset_block,
                                         sky_damping_block, offset_damping_block)


def test_mean_offset_block_matches_legacy():
    num_frames, nc_m, m = 7, 5, 1
    col_bases = [100, 100 + 0, 100 + num_frames * nc_m]  # arbitrary; m=1 base used
    ftg_m = np.arange(num_frames)  # per-frame groups (no det_groups)
    mean_off = np.linspace(-0.3, 0.4, num_frames)
    w = 10.0
    blk = mean_offset_block(m, mean_off, num_frames, nc_m, ftg_m, col_bases, weight=w)

    # legacy inline
    rows = np.repeat(np.arange(num_frames, dtype=np.int64), nc_m)
    offset_starts = col_bases[m] + ftg_m.astype(np.int64) * nc_m
    cols = (offset_starts[:, None] + np.arange(nc_m, dtype=np.int64)[None, :]).reshape(-1)
    data = np.full(num_frames * nc_m, w, dtype=np.float32)
    b = np.asarray(mean_off).astype(np.float64).flatten() * nc_m * w
    assert np.array_equal(blk.rows_local, rows)
    assert np.array_equal(blk.cols, cols)
    assert np.array_equal(blk.data, data)
    assert np.array_equal(blk.b, b)
    assert blk.num_rows == num_frames and blk.nnz_per_row == nc_m


def _check_sky_damping(block_index, num_sky):
    rng = np.random.default_rng(block_index + 1)
    cov = (rng.random(num_sky) * 50).astype(np.int64)
    cov[::3] = 0  # some uncovered pixels
    w = 0.1
    blk = sky_damping_block(block_index, w, cov, num_sky)
    valid = np.nonzero(cov)[0]
    data = np.sqrt(w * cov[valid]).astype(np.float32)
    cols = (block_index * num_sky + valid).astype(np.int64)
    assert np.array_equal(blk.cols, cols)
    assert np.array_equal(blk.data, data)
    assert np.array_equal(blk.rows_local, np.arange(len(valid), dtype=np.int64))
    assert np.array_equal(blk.b, np.zeros(len(valid)))
    assert blk.num_rows == len(valid) and blk.nnz_per_row == 1


def test_sky_damping_continuum_block0():
    _check_sky_damping(0, 400)


def test_sky_damping_line_block1():
    # The path the gate doesn't exercise (damp_weight_line=0 in production).
    _check_sky_damping(1, 400)


def test_sky_damping_empty_returns_none():
    assert sky_damping_block(0, 0.1, np.zeros(10, dtype=np.int64), 10) is None


def test_offset_damping_block_matches_legacy():
    num_sky_eff = 800
    rng = np.random.default_rng(42)
    cov = (rng.random(120) * 30).astype(np.int64)
    cov[::4] = 0
    w = 0.05
    blk = offset_damping_block(w, cov, num_sky_eff)
    valid = np.nonzero(cov)[0]
    data = np.sqrt(w * cov[valid]).astype(np.float32)
    cols = (valid + num_sky_eff).astype(np.int64)
    assert np.array_equal(blk.cols, cols)
    assert np.array_equal(blk.data, data)
    assert blk.num_rows == len(valid) and blk.nnz_per_row == 1
    assert offset_damping_block(w, np.zeros(5, dtype=np.int64), num_sky_eff) is None


def test_as_dict_keys():
    blk = sky_damping_block(0, 0.1, np.array([3, 0, 5], dtype=np.int64), 10)
    d = blk.as_dict()
    assert set(d) == {'rows_local', 'cols', 'data', 'b', 'num_rows', 'nnz_per_row'}


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")


if __name__ == '__main__':
    _run_all()
