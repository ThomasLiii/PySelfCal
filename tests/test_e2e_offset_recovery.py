"""End-to-end offset-recovery test for the real LSQR self-cal solver (no data).

This is the correctness test that lets anyone run the suite WITHOUT the /mnt
RAID data. It fabricates a tiny synthetic self-calibration problem entirely in
a temp directory, runs the REAL low-level solver path
(``setup_lsqr`` -> ``apply_lsqr``), and asserts that the solver RECOVERS the
injected per-frame offsets (and the injected sky) to solver tolerance.

Runnable as ``python tests/test_e2e_offset_recovery.py`` or under pytest.
ALWAYS run from the repo root so ``import selfcal`` resolves to this worktree.

The physics being tested
------------------------
The self-cal model assembled by ``setup_lsqr`` is, for observation ``i`` of a
reference-grid pixel ``P`` in frame ``f`` (continuum-only, unit weight):

    d_i = sky[P] + offset[f]                          (noiseless here)

with the design matrix carrying a ``+1`` in the sky column for ``P`` and a
``+1`` in the offset column for frame ``f`` (verified against the row-assembly
in ``selfcal/core/assembly.py``: the sky nnz is ``valid_weight`` and the
single-chunk offset nnz is ``valid_weight * chunk_val`` = 1).

GAUGE: adding a constant ``c`` to every offset and subtracting it from every
sky pixel leaves every ``d_i`` unchanged. With every frame observing every
pixel the observation graph is connected, so the null space is EXACTLY this
one global constant. LSQR/LSMR started from ``x0 = 0`` converge to the
minimum-norm solution (iterates live in ``range(A^T) = (null A)^perp``), which
removes the component along the gauge vector ``g = (-1 on sky, +1 on
offsets)``. The min-norm solution therefore equals the true solution iff the
true solution is orthogonal to ``g``, i.e.

    sum(sky_true over observed pixels) == 0   AND   sum(offset_true) == 0.

So the test injects a sum-zero sky pattern and sum-zero (mean-zero) per-frame
offsets, and solves with ``precondition=False`` (plain Euclidean min-norm; a
Jacobi preconditioner would minimize a column-norm-weighted norm and shift the
gauge) and ``damp=0`` (no Tikhonov shrinkage bias). Recovery is then EXACT up
to solver tolerance, with no additive-constant ambiguity to explain away.

Everything is built from dyadic rationals (multiples of 1/16), so ``sky + off``
is exactly representable in float32 and the assembled system is exactly
consistent -- recovery reaches ~1e-9, far tighter than the ~1e-6 float32 floor
a generic pattern would leave.

Schema / convention subtleties handled (see the reproj writer in
``selfcal/io/reprojection.py`` and the reader in ``selfcal/core/subframe.py``):

* ``ref_coords = [y_min, y_max, x_min, x_max]`` is the subframe's bbox in the
  ref grid; assembly places subframe pixel ``(r, c)`` at ref pixel
  ``(y_min + r, x_min + c)`` -> the SKY column.
* ``sub_mapping`` has shape ``(2, H, W)`` and maps each subframe pixel into the
  DETECTOR/chunk-map grid (a grid independent of the ref grid): plane 0 = x
  (column), plane 1 = y (row). ``_prep_subframe`` reverses it to ``(row, col)``
  before ``make_linear_interp_matrix``. We set it to the identity meshgrid so
  every subframe pixel lands on an EXACT integer detector pixel -> the interp
  matrix is a clean 0/1 selection (no fractional-weight complications).
* the loader parses ``exp_idx``/``det_idx`` out of the filename
  (``basename.split('_')[-3]`` / ``[-1]``); an unparseable name is silently
  treated as a missing file, so the filenames must split into int-valued
  fields (``synth_{exp}_det_{det}.h5``).
"""
import os
import tempfile

import h5py
import numpy as np
from astropy.wcs import WCS

from selfcal.core.system import setup_lsqr
from selfcal.core.solve import apply_lsqr
from selfcal.core.solution import parse_x_sky
from selfcal.core.layout import SystemLayout

FIXED_SEED = 20240722

# Small synthetic problem: an H x W ref grid observed by F frames.
H = W = 8
F = 6


def _minimal_header_str():
    """A minimal but valid 2-D WCS header string for the (unused) attrs."""
    return WCS(naxis=2).to_header().tostring()


def _write_synthetic_frame(path, sub_data, sub_mapping, ref_coords):
    """Write one reprojected-frame .h5 matching the schema setup_lsqr reads.

    Faithful to ``selfcal/io/reprojection.py``: same dataset names/dtypes and
    attrs. ``sub_bitmask``/``sub_foot`` are written for fidelity but are unused
    on the ``apply_mask=False`` LSQR path.
    """
    with h5py.File(path, 'w', libver='latest') as hf:
        hf.create_dataset('sub_data', data=sub_data.astype(np.float32))
        hf.create_dataset('sub_foot', data=np.ones_like(sub_data, dtype=np.float16))
        hf.create_dataset('sub_bitmask', data=np.zeros_like(sub_data, dtype=np.int32))
        hf.create_dataset('sub_mapping', data=sub_mapping.astype(np.float32))
        hf.attrs['sub_header'] = _minimal_header_str()
        hf.attrs['det_header'] = _minimal_header_str()
        hf.attrs['file_path'] = os.path.basename(path)
        hf.attrs['ref_coords'] = np.array(ref_coords, dtype=np.int32)


def _make_truth(seed=FIXED_SEED):
    """Injected ground truth: a sum-zero planar sky and sum-zero per-frame offsets.

    All values are multiples of 1/16 so ``sky + offset`` is exact in float32.
    """
    rng = np.random.default_rng(seed)

    # Sum-zero planar sky: sky[r, c] = ((2r-7) + (2c-7)) / 16. Each of the two
    # 1-D ramps sums to zero over its 8 indices, so the whole grid sums to zero
    # exactly (integer arithmetic before the /16 scale).
    r = np.arange(H)[:, None]
    c = np.arange(W)[None, :]
    sky_true = ((2 * r - (H - 1)) + (2 * c - (W - 1))).astype(np.float64) / 16.0
    assert sky_true.sum() == 0.0

    # Sum-zero per-frame offsets: draw F-1 small integers, pin the last so the
    # integer sum is exactly zero, then scale by 1/16 (dyadic -> exact float32).
    k = rng.integers(-6, 7, size=F).astype(np.int64)
    k[-1] -= int(k.sum())          # force sum(k) == 0 exactly
    assert k.sum() == 0
    offsets_true = k.astype(np.float64) / 16.0

    return sky_true, offsets_true


def _identity_sub_mapping():
    """sub_mapping (2, H, W): plane 0 = x (col), plane 1 = y (row) = identity."""
    ys, xs = np.mgrid[0:H, 0:W]
    return np.stack([xs, ys], axis=0).astype(np.float32)   # (x, y)


# Cache the (build + solve) so the multiple test_* functions share one solve.
_SOLVE_CACHE = {}


def _build_and_solve():
    """Fabricate the frames, run setup_lsqr + apply_lsqr, parse and return results."""
    if _SOLVE_CACHE:
        return _SOLVE_CACHE['result']

    sky_true, offsets_true = _make_truth()
    ref_shape = (H, W)
    sub_mapping = _identity_sub_mapping()
    # Single chunk covering the whole detector grid -> exactly ONE offset value
    # per frame (the per-frame constant). Detector grid == subframe grid so the
    # identity sub_mapping is in-bounds.
    chunk_map = np.zeros((H, W), dtype=np.int64)

    with tempfile.TemporaryDirectory() as tmp:
        file_list = []
        for f in range(F):
            # Every frame covers the full ref grid (connected graph): pixel
            # (r, c) -> ref pixel (r, c). Noiseless: d = sky + offset[f].
            sub_data = (sky_true + offsets_true[f]).astype(np.float32)
            path = os.path.join(tmp, f"synth_{f}_det_0.h5")
            _write_synthetic_frame(path, sub_data, sub_mapping,
                                   ref_coords=[0, H, 0, W])
            file_list.append(path)

        # Column layout (single source of truth) -> total_cols for the parse
        # and for apply_lsqr's active_mask expansion.
        layout = SystemLayout.build(ref_shape, [chunk_map], num_sky_blocks=1,
                                    num_frames=F)

        res = setup_lsqr(
            file_list, ref_shape,
            chunk_maps=[chunk_map],
            apply_mask=False,          # no bitmask -> unit weight everywhere
            apply_weight=False,
            outlier_thresh=None,       # noiseless: never reject a pixel
            max_workers=1,             # determinism
            batch_size=F,
            offset_regularization=False,
            compact_zero_columns=True,
        )
        assert res.A is not None, "setup_lsqr found no valid data"

        # damp=0 (no shrinkage bias), precondition=False (plain Euclidean
        # min-norm so the gauge lands at c*=0), x0=0, n_threads=1 (determinism),
        # tight tol + generous iter_lim to converge this tiny (70-unknown) system.
        x = apply_lsqr(
            res.A, res.b, ref_shape,
            x0=None,
            atol=1e-12, btol=1e-12, damp=0.0,
            iter_lim=500,
            precondition=False,
            solver='lsmr',
            use_float32=False,
            n_threads=1,
            active_mask=res.active_mask,
            num_cols_full=(layout.total_cols
                           if res.active_mask is not None else None),
        )

    sky_maps, det_offsets, frame_scalar = parse_x_sky(
        x, ref_shape,
        num_offset_groups_list=layout.num_offset_groups_list,
        num_chunks_list=layout.num_chunks_list,
        num_frames=None, num_sky_blocks=1,
    )
    sky_rec = sky_maps[0]
    # Single chunk map, one group per frame -> (F, 1); flatten to per-frame value.
    offsets_rec = det_offsets[0].reshape(-1)

    result = {
        'sky_true': sky_true,
        'offsets_true': offsets_true,
        'sky_rec': sky_rec,
        'offsets_rec': offsets_rec,
        'frame_scalar': frame_scalar,
        'max_off_err': float(np.max(np.abs(offsets_rec - offsets_true))),
        'max_sky_err': float(np.max(np.abs(sky_rec - sky_true))),
    }
    _SOLVE_CACHE['result'] = result
    return result


def test_recovers_injected_per_frame_offsets():
    """The core assertion: solved per-frame offsets == injected mean-zero offsets."""
    r = _build_and_solve()
    # dyadic-exact, consistent system -> recovery is at solver-tolerance level,
    # far below the ~1e-6 float32 floor a generic pattern would leave.
    assert r['max_off_err'] < 1e-6, (
        f"per-frame offset recovery error {r['max_off_err']:.2e} too large; "
        f"true={r['offsets_true']}, recovered={r['offsets_rec']}")


def test_recovers_sky():
    """The solved sky map matches the injected sum-zero sky pattern."""
    r = _build_and_solve()
    assert r['max_sky_err'] < 1e-6, (
        f"sky recovery error {r['max_sky_err']:.2e} too large")


def test_recovered_offsets_are_mean_zero():
    """Gauge sanity: with sum-zero truth the min-norm gauge stays at c*=0, so
    the recovered offsets are themselves mean-zero (no global constant leaked
    in from the sky block)."""
    r = _build_and_solve()
    assert abs(float(np.mean(r['offsets_rec']))) < 1e-6


def test_no_spurious_scalar_block():
    """No per-frame scalar / det-groups were requested, so the parse yields an
    empty frame-scalar block -- the offsets carry the full per-frame DC."""
    r = _build_and_solve()
    assert r['frame_scalar'].size == 0


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    r = _SOLVE_CACHE.get('result')
    if r is not None:
        print(f"\nmax|offset_rec - offset_true| = {r['max_off_err']:.3e}")
        print(f"max|sky_rec    - sky_true|    = {r['max_sky_err']:.3e}")
    print(f"\nALL {len(fns)} TESTS PASSED")


if __name__ == '__main__':
    _run_all()
