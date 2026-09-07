"""Bit-equality tests for the BlockCSR (Target 3) path.

1. _iter_global_entry_chunks: chunk cuts over a block list == unified-array
   slicing at the same chunk_size (contents AND boundaries).
2. compute_x0_scalar_only(BlockCSR) == legacy compute_x0_from_Ab+zeroing.
3. apply_lsqr end-to-end: unified csr vs BlockCSR of the SAME system --
   solver x bit-equal across {lsqr, lsmr} x {n_threads 1, 4} x
   {precondition on/off} x {use_float32 on/off} x {active_mask on/off}.

Ported from workspace/memopt/scripts/test_blockcsr_bitequal.py. The monolithic
main() is split into one pytest function per section; each re-seeds its own rng
(seed preserved from the source) so the sections stay independent and
deterministic. The BlockCSR path is forced by constructing it directly via
build_block_csr(target_nnz), not via the SELFCAL_BLOCK_NNZ env var, so there is
no global env state to set or clean up.

Runnable as ``python tests/test_blockcsr_bitequal.py`` or under pytest.
"""
import numpy as np
from scipy.sparse import csr_matrix, random as sprandom

from selfcal.core.blockcsr import BlockCSR, build_block_csr
from selfcal.core.solve import apply_lsqr, _iter_global_entry_chunks
from selfcal.core.solution import compute_x0_from_Ab, compute_x0_scalar_only

# Seed preserved verbatim from the source script.
_SEED = 31


def make_system(rng, m, n, density, data_dtype=np.float32, sorted_idx=True):
    A = sprandom(m, n, density=density, format='csr',
                 random_state=rng.integers(2**31),
                 data_rvs=lambda k: rng.standard_normal(k))
    A.data = A.data.astype(data_dtype)
    if sorted_idx:
        A.sort_indices()
    b = rng.standard_normal(m)
    return A, b


def to_blocks(A, target_nnz):
    # same arrays as the unified matrix, indices as int32 like Phase 6 emits
    return build_block_csr(A.data.copy(), A.indices.astype(np.int32),
                           A.indptr.astype(np.int64), A.shape, target_nnz)


def test_iter_global_entry_chunks_matches_unified_slicing():
    """Section 1: chunk iterator cuts + contents identical to unified slicing."""
    rng = np.random.default_rng(_SEED)
    n_cases = 0
    for target, chunk in [(50, 64), (17, 40), (200, 7), (33, 1000)]:
        n_cases += 1
        A, _ = make_system(rng, 300, 60, 0.08)
        bc = to_blocks(A, target)
        got = list(_iter_global_entry_chunks(bc.blocks, chunk))
        data = A.data
        cols = A.indices
        ref = [(data[s:s + chunk], cols[s:s + chunk].astype(np.int32))
               for s in range(0, data.size, chunk)]
        assert len(got) == len(ref), (
            f"chunk count differs target={target} chunk={chunk}: "
            f"{len(got)} vs {len(ref)}")
        for i, (g, r) in enumerate(zip(got, ref)):
            assert g[0].tobytes() == r[0].tobytes(), (
                f"data chunk {i} differs target={target} chunk={chunk}")
            assert (g[1].astype(np.int64).tobytes()
                    == r[1].astype(np.int64).tobytes()), (
                f"col chunk {i} differs target={target} chunk={chunk}")
    assert n_cases == 4


def test_x0_scalar_only_blockcsr_bitequal():
    """Section 2: compute_x0_scalar_only(BlockCSR) == legacy csr path."""
    rng = np.random.default_rng(_SEED)
    n_cases = 0
    for target in (10**9, 400, 37):
        for use_mask in (False, True):
            n_cases += 1
            n_sky, n_chunkcols, n_scalar = 7 * 9, 30, 25
            n_cols_full = n_sky + n_chunkcols + n_scalar
            A_full, b = make_system(rng, 500, n_cols_full, 0.05)
            scs = n_sky + n_chunkcols
            if use_mask:
                col_nnz = np.bincount(A_full.indices, minlength=n_cols_full)
                am = col_nnz > 0
                keep = np.flatnonzero(am)
                remap = np.full(n_cols_full, -1)
                remap[keep] = np.arange(keep.size)
                A = csr_matrix((A_full.data, remap[A_full.indices],
                                A_full.indptr), shape=(500, keep.size))
            else:
                am, A = None, A_full
            ref = compute_x0_from_Ab(A, b, (7, 9), num_sky_blocks=1,
                                     active_mask=am)
            ref[63:scs] = 0.0
            bc = to_blocks(A, target)
            new = compute_x0_scalar_only(bc, b, (7, 9), scs,
                                         num_sky_blocks=1, active_mask=am,
                                         _chunk_target_entries=53)
            assert ref.tobytes() == new.tobytes(), (
                f"x0 not bit-equal target={target} mask={use_mask}: "
                f"{int((ref != new).sum())} of {ref.size} differ")
    assert n_cases == 6


def test_apply_lsqr_unified_vs_blockcsr_bitequal():
    """Section 3: apply_lsqr on unified csr vs BlockCSR of the same system,
    across solver x n_threads x precondition x float32 x active_mask, each with
    three BlockCSR target_nnz cuts. 2*2*2*2*2*3 = 96 asserted cases."""
    rng = np.random.default_rng(_SEED)
    n_cases = 0
    for solver in ('lsqr', 'lsmr'):
        for n_threads in (1, 4):
            for precondition in (True, False):
                for use_f32 in (True, False):
                    for use_mask in (True, False):
                        A0, b = make_system(rng, 800, 120, 0.06,
                                            data_dtype=np.float32)
                        num_cols_full = 150 if use_mask else 120
                        am = None
                        if use_mask:
                            am = np.zeros(150, dtype=bool)
                            am[np.sort(rng.choice(150, 120, replace=False))] = True
                        x0 = rng.standard_normal(num_cols_full)
                        ref_shape = (2, 3)
                        # unified reference (fresh copies -- apply_lsqr mutates)
                        A_ref = csr_matrix((A0.data.copy(), A0.indices.copy(),
                                            A0.indptr.copy()), shape=A0.shape)
                        x_ref = apply_lsqr(
                            A_ref, b.copy(), ref_shape, x0=x0.copy(),
                            atol=1e-8, btol=1e-8, damp=0, iter_lim=15,
                            precondition=precondition, solver=solver,
                            use_float32=use_f32, n_threads=n_threads,
                            active_mask=am,
                            num_cols_full=num_cols_full if use_mask else None)
                        for target in (10**9, 900, 111):
                            n_cases += 1
                            bc = to_blocks(A0, target)
                            x_new = apply_lsqr(
                                bc, b.copy(), ref_shape, x0=x0.copy(),
                                atol=1e-8, btol=1e-8, damp=0, iter_lim=15,
                                precondition=precondition, solver=solver,
                                use_float32=use_f32, n_threads=n_threads,
                                active_mask=am,
                                num_cols_full=num_cols_full if use_mask else None)
                            assert x_ref.tobytes() == x_new.tobytes(), (
                                f"solve not bit-equal solver={solver} "
                                f"nt={n_threads} pre={precondition} "
                                f"f32={use_f32} mask={use_mask} target={target}")
    assert n_cases == 96


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")


if __name__ == '__main__':
    _run_all()
