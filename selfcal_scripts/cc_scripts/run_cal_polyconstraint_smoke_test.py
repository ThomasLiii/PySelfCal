"""Plumbing smoke test for ``poly_constraints_list``.

Calls ``setup_lsqr`` directly with K=1 and a single linear-stencil polynomial
constraint group, then checks:

1. The matrix has exactly one extra row vs. an equivalent run without
   ``poly_constraints_list`` (1 chain x 1 frame = 1 constraint row).
2. That extra row has nonzeros at columns ``(col_bases[0] + 0, +1, +2)`` with
   values ``(weight*1, weight*-2, weight*1)`` and RHS 0.
3. ``compute_column_polynomial_chains`` produces the expected
   ``(num_subchannels, 3)`` chains and ``[1, -2, 1]`` stencil for
   ``degree=1, num_columns=3``, with each row's chunk ids consecutive within
   one subchannel.

Skips LSQR. Subsamples reproj files to a single frame so the run finishes in
seconds — the goal is plumbing verification, not solver convergence.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import glob as glob_module

import numpy as np

# Script lives at <repo>/selfcal_scripts/cc_scripts/; repo root is 3 dirs up.
parent_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(parent_path)

from SelfCal.lsqr import setup_lsqr
from SelfCal import WCSHelper
from SelfCal.SPHERExUtility import (
    load_lvf_params,
    make_stripped_chunk_map,
    make_stripped_chunk_valid_mask,
    compute_column_polynomial_chains,
)


# ----------------------------- Settings -----------------------------
DETECTOR = 3
NUM_SUB = 10
NUM_CH = 34
NUM_COL = 3
CH_LIST = [17]
NUM_FRAMES_LIMIT = 1  # one frame -> one poly constraint row per chain

CACHE_DIR = '/home/thomasli/spherex/selfcal/cache/'
RUN_NAME = f'SPHEREx_nep_qr2_det{DETECTOR}_6p2arcsec'
NVME_REPROJ_DIR = os.path.join(CACHE_DIR, f'reproj_nvme_{RUN_NAME}')
REF_PATH = f'/mnt/md124/thomasli/selfcal/outputs/{RUN_NAME}/ref.fits'

POLY_WEIGHT = 2.5  # arbitrary nonzero; checked against emitted row data values


# ----------------------------- Inputs -----------------------------
print("Loading reference WCS + chunk map...")
ref_wcs, ref_shape = WCSHelper.load_from_fits(REF_PATH)

lvf_params = load_lvf_params(f'lvf_params_D{DETECTOR}.npy')
det_chunk_map, _, _, _ = make_stripped_chunk_map(
    DETECTOR, num_subchannels=NUM_SUB, num_channels=NUM_CH,
    num_columns=NUM_COL, oversample_factor=1, lvf_params=lvf_params)

chunk_valid_mask_padded = make_stripped_chunk_valid_mask(
    ch=CH_LIST, num_subchannels=NUM_SUB, num_channels=NUM_CH,
    num_columns=NUM_COL, subchannel_padding=1)
det_valid_mask_padded = chunk_valid_mask_padded[det_chunk_map]

all_files = sorted(glob_module.glob(os.path.join(NVME_REPROJ_DIR, '*.h5')))
file_list = all_files[:NUM_FRAMES_LIMIT]
print(f"Using {len(file_list)} of {len(all_files)} reproj files")

num_frames = len(file_list)
num_chunks = int(det_chunk_map.max()) + 1
num_sky = ref_shape[0] * ref_shape[1]
col_base_0 = num_sky  # K=1, single map starts immediately after sky block


# ----------------------------- Helper output check -----------------------------
print("\n=== Helper: compute_column_polynomial_chains ===")
chains_helper, stencil_helper = compute_column_polynomial_chains(
    det_chunk_map, num_columns=NUM_COL, degree=1)
expected_num_subchannels = num_chunks // NUM_COL
print(f"chains.shape = {chains_helper.shape}, "
      f"expected ({expected_num_subchannels}, 3)")
assert chains_helper.shape == (expected_num_subchannels, 3), \
    f"chains shape mismatch"
np.testing.assert_array_equal(stencil_helper, np.array([1.0, -2.0, 1.0]))
# Each row should be (s*NUM_COL, s*NUM_COL+1, s*NUM_COL+2) for s=0..S-1
expected_rows = (np.arange(expected_num_subchannels)[:, None] * NUM_COL
                 + np.arange(3)[None, :])
np.testing.assert_array_equal(chains_helper, expected_rows)
print("OK helper produces correct chains + stencil")

# Edge cases
try:
    compute_column_polynomial_chains(det_chunk_map, num_columns=2, degree=1)
except ValueError as e:
    print(f"OK helper raises on num_columns < L: {e}")
else:
    raise AssertionError("expected ValueError for num_columns < L")


# ----------------------------- setup_lsqr WITHOUT poly -----------------------------
common_kwargs = dict(
    chunk_maps=[det_chunk_map],
    grid_valid_weight=det_valid_mask_padded,
    apply_mask=True, apply_weight=False,
    outlier_thresh=5.0, ignore_list=[], batch_size=1,
    offset_regularization=True,
    reg_weights=[0.0],         # disable adjacency to keep row count clean
    adj_infos=[None],
    mean_offsets_list=[None],  # no mean-offset rows
    weighted_damping=False,    # no damping rows
    max_workers=1,
)

print("\n=== Run 1: setup_lsqr without poly ===")
A_base, b_base, _ = setup_lsqr(file_list, ref_shape, **common_kwargs)
print(f"A_base.shape = {A_base.shape}, b_base.shape = {b_base.shape}")

# ----------------------------- setup_lsqr WITH poly -----------------------------
chains = np.array([[0, 1, 2]], dtype=np.int64)
stencil = np.array([1.0, -2.0, 1.0], dtype=np.float64)
poly_groups = [{'chains': chains, 'stencil': stencil, 'weight': POLY_WEIGHT}]

print("\n=== Run 2: setup_lsqr with one poly constraint group ===")
A_poly, b_poly, _ = setup_lsqr(
    file_list, ref_shape,
    poly_constraints_list=[poly_groups],
    **common_kwargs,
)
print(f"A_poly.shape = {A_poly.shape}, b_poly.shape = {b_poly.shape}")


# ----------------------------- Row-count check -----------------------------
extra_rows = A_poly.shape[0] - A_base.shape[0]
print(f"\nextra rows (poly - base) = {extra_rows}, expected 1")
assert extra_rows == 1, f"expected exactly 1 extra row, got {extra_rows}"
assert A_poly.shape[1] == A_base.shape[1], "column count must match"
print("OK matrix shape: exactly one extra row, columns unchanged")


# ----------------------------- Row content check -----------------------------
# The poly row is the last one (no mean / damping / extra constraints follow).
A_csr = A_poly.tocsr()
poly_row_idx = A_poly.shape[0] - 1
poly_row = A_csr.getrow(poly_row_idx)
indices = poly_row.indices
data = poly_row.data
order = np.argsort(indices)
indices = indices[order]
data = data[order]
print(f"\npoly row {poly_row_idx}: cols={indices.tolist()}, data={data.tolist()}")

expected_cols = np.array([col_base_0 + 0, col_base_0 + 1, col_base_0 + 2])
expected_data = POLY_WEIGHT * np.array([1.0, -2.0, 1.0])
np.testing.assert_array_equal(indices, expected_cols)
np.testing.assert_allclose(data, expected_data, rtol=0, atol=0)
assert b_poly[poly_row_idx] == 0.0, f"RHS expected 0, got {b_poly[poly_row_idx]}"
print("OK poly row has the expected (col, value) entries and RHS=0")

print("\n=== poly-constraint smoke test PASSED ===")
