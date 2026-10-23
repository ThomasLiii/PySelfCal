"""Bit-equality property test: compute_x0_from_Ab (the separate-fit start) on csr, BlockCSR and
ColSplitCSR storages (streamed in row chunks) vs the legacy algorithm (compute_x0_from_Ab of the
matrix as a coo_matrix: one bincount over its entries in row order).

Covers: f32/f64 data, int32/int64 indices, with/without active_mask, empty rows, one or several sky
blocks, and forced multi-chunk selection (via _chunk_target_entries) to prove chunked gathering reproduces the one-shot
path bit-for-bit.

Ported from workspace/memopt/scripts/test_x0_bitequal.py. Runnable as
``python tests/test_x0_bitequal.py`` or under pytest.
"""
import numpy as np
from scipy.sparse import csr_matrix, random as sprandom

from selfcal.core.solution import compute_x0_from_Ab

# Seed preserved verbatim from the source script (deterministic case sweep).
_SEED = 20260718


def build_case(rng, n_rows, ref_shape, num_sky_blocks, n_chunk_cols,
               n_frame_cols, data_dtype, idx_dtype, use_active_mask,
               density=0.05):
    num_sky = ref_shape[0] * ref_shape[1] * num_sky_blocks
    num_cols_full = num_sky + n_chunk_cols + n_frame_cols
    A_full = sprandom(n_rows, num_cols_full, density=density, format='csr',
                      random_state=rng.integers(2**31),
                      data_rvs=lambda n: rng.standard_normal(n))
    # kill some rows entirely; leave some columns empty
    kill_rows = rng.integers(0, n_rows, size=max(1, n_rows // 10))
    for r in kill_rows:
        A_full.data[A_full.indptr[r]:A_full.indptr[r + 1]] = 0.0
    A_full.eliminate_zeros()
    A_full.data = A_full.data.astype(data_dtype)
    b = rng.standard_normal(n_rows)  # float64, as in production

    if use_active_mask:
        col_nnz = np.bincount(
            A_full.indices, minlength=num_cols_full)
        active_mask = col_nnz > 0
        # legacy compact matrix: drop inactive columns (order-preserving)
        keep = np.flatnonzero(active_mask)
        remap = np.full(num_cols_full, -1)
        remap[keep] = np.arange(keep.size)
        A = csr_matrix((A_full.data, remap[A_full.indices], A_full.indptr),
                       shape=(n_rows, keep.size))
    else:
        active_mask = None
        A = A_full
    A.indices = A.indices.astype(idx_dtype)
    A.indptr = A.indptr.astype(idx_dtype)
    return A, b, active_mask


def test_x0_from_Ab_csr_bitequal():
    """compute_x0_from_Ab on a csr matrix (streamed in row chunks) == the legacy one-bincount path over the matrix
    as a coo_matrix, bit for bit: 2*2*2*3*3*3 = 216 cases.

    The nested loop is kept (rather than pytest.parametrize) so a single seeded rng advances through every case.
    """
    rng = np.random.default_rng(_SEED)
    ref_shape = (7, 9)
    n_cases = 0
    for data_dtype in (np.float32, np.float64):
        for idx_dtype in (np.int32, np.int64):
            for use_mask in (False, True):
                for num_sky_blocks in (1, 2, 4):
                    for n_frame in (1, 5, 40):
                        for chunk_target in (10**9, 50, 7):
                            n_cases += 1
                            A, b, am = build_case(rng, n_rows=200, ref_shape=ref_shape, num_sky_blocks=num_sky_blocks,
                                                  n_chunk_cols=30, n_frame_cols=n_frame, data_dtype=data_dtype,
                                                  idx_dtype=idx_dtype, use_active_mask=use_mask)
                            ref = compute_x0_from_Ab(A.tocoo(), b, ref_shape, num_sky_blocks=num_sky_blocks,
                                                     active_mask=am)
                            new = compute_x0_from_Ab(A, b, ref_shape, num_sky_blocks=num_sky_blocks, active_mask=am,
                                                     _chunk_target_entries=chunk_target)
                            assert ref.dtype == new.dtype and ref.shape == new.shape and ref.tobytes() == new.tobytes(), (
                                f"x0 not bit-equal dtype={data_dtype.__name__} idx={idx_dtype.__name__} "
                                f"mask={use_mask} J={num_sky_blocks} F={n_frame} chunk={chunk_target}: "
                                f"{int((ref != new).sum())} of {ref.size} differ")
    # Guard against silently shrinking coverage.
    assert n_cases == 216


def test_x0_from_Ab_split_storages_bitequal():
    """compute_x0_from_Ab on the large systems' storages (BlockCSR, ColSplitCSR with cuts in the sky and in the
    offsets) == its csr path, bit for bit: f32/f64 data, with/without active_mask, one or several sky blocks,
    one chunk or many (_chunk_target_entries)."""
    from selfcal.core.blockcsr import build_block_csr, partition_block_csr
    rng = np.random.default_rng(_SEED + 1)
    ref_shape = (7, 9)
    n_cases = 0
    for data_dtype in (np.float32, np.float64):
        for use_mask in (False, True):
            for num_sky_blocks in (1, 2):
                for chunk_target in (10**9, 37):
                    A, b, am = build_case(rng, n_rows=300, ref_shape=ref_shape, num_sky_blocks=num_sky_blocks,
                                          n_chunk_cols=45, n_frame_cols=6, data_dtype=data_dtype,
                                          idx_dtype=np.int32, use_active_mask=use_mask, density=0.08)
                    A.sort_indices()
                    ref = compute_x0_from_Ab(A.tocoo(), b, ref_shape, num_sky_blocks=num_sky_blocks, active_mask=am)
                    n = A.shape[1]

                    def blocks():
                        return build_block_csr(A.data.copy(), A.indices.astype(np.int32),
                                               A.indptr.astype(np.int64), A.shape, int(rng.integers(40, 900)))
                    boundary = int(am[:num_sky_blocks * 63].sum()) if am is not None else num_sky_blocks * 63
                    for storage in (blocks(),
                                    partition_block_csr(blocks(), [0, boundary // 2, boundary + 7, n]),
                                    partition_block_csr(blocks(), [0, boundary, n - 20, n - 3, n])):
                        n_cases += 1
                        new = compute_x0_from_Ab(storage, b, ref_shape, num_sky_blocks=num_sky_blocks,
                                                 active_mask=am, _chunk_target_entries=chunk_target)
                        assert ref.shape == new.shape and ref.tobytes() == new.tobytes(), (
                            f"x0 not bit-equal on {type(storage).__name__} dtype={data_dtype.__name__} "
                            f"mask={use_mask} J={num_sky_blocks} chunk={chunk_target}: "
                            f"{int((ref != new).sum())} of {ref.size} differ")
    assert n_cases == 48


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")


if __name__ == '__main__':
    _run_all()
