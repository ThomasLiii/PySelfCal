"""K=2 smoke test for the multi-chunk-maps refactor.

Calls ``setup_lsqr`` directly (bypassing ``Calibrator``) with
``chunk_maps=[real, dummy]``. The dummy map has a single chunk and a per-frame
mean-zero constraint; the real map is the standard column-adjacency map with
the same regs as production.

Two checks:

1. Matrix shape is what ``col_bases`` predicts:
   ``num_sky + num_frames * num_chunks_real + num_frames * 1`` columns (no
   det_groups → no scalar block).
2. Diagonal LS via ``compute_x0_from_Ab`` populates both offset blocks (so the
   K=2 plumbing — per-map SHM hand-off, per-map offset rows, per-map mean
   constraint — actually wires through the worker pool).

Skips the full LSQR solve to keep the run fast. Intentionally subsamples the
reproj-file list (NUM_FRAMES_LIMIT) so the smoke test finishes in well under
a minute, since the goal is plumbing verification, not solver convergence.
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
from astropy.io import fits

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from SelfCal.lsqr import setup_lsqr
from SelfCal.solution import compute_x0_from_Ab
from SelfCal import WCSHelper
from SelfCal.SPHERExUtility import (load_lvf_params, compute_column_adjacency,
                                    make_stripped_chunk_map, make_stripped_chunk_valid_mask,
                                    fast_vertical_dist)


# ----------------------------- Settings -----------------------------
DETECTOR = 3
NUM_SUB = 10
NUM_CH = 34
NUM_COL = 3
CH_LIST = [17]  # single channel
NUM_FRAMES_LIMIT = 100  # cap reproj files to keep this a quick smoke test

CACHE_DIR = '/home/thomasli/spherex/selfcal/cache/'
RUN_NAME = f'SPHEREx_nep_qr2_det{DETECTOR}_6p2arcsec'
NVME_REPROJ_DIR = os.path.join(CACHE_DIR, f'reproj_nvme_{RUN_NAME}')
REF_PATH = f'/mnt/md124/thomasli/selfcal/outputs/{RUN_NAME}/ref.fits'

# ----------------------------- Inputs -----------------------------
print("Loading reference WCS + chunk map...")
ref_wcs, ref_shape = WCSHelper.load_from_fits(REF_PATH)

lvf_params = load_lvf_params(f'lvf_params_D{DETECTOR}.npy')
det_chunk_map_real, _, _, _ = make_stripped_chunk_map(
    DETECTOR, num_subchannels=NUM_SUB, num_channels=NUM_CH,
    num_columns=NUM_COL, oversample_factor=1, lvf_params=lvf_params)
adj_info_real = compute_column_adjacency(det_chunk_map_real, NUM_COL)

# Dummy map: single chunk covering the same shape (zeros) — absorbs per-frame DC
det_chunk_map_dummy = np.zeros_like(det_chunk_map_real, dtype=np.int32)

chunk_valid_mask_padded = make_stripped_chunk_valid_mask(
    ch=CH_LIST, num_subchannels=NUM_SUB, num_channels=NUM_CH,
    num_columns=NUM_COL, subchannel_padding=1)
det_valid_mask_padded = chunk_valid_mask_padded[det_chunk_map_real]

# Subsample reproj files
all_files = sorted(glob_module.glob(os.path.join(NVME_REPROJ_DIR, '*.h5')))
file_list = all_files[:NUM_FRAMES_LIMIT]
print(f"Using {len(file_list)} of {len(all_files)} reproj files")

num_frames = len(file_list)
num_chunks_real = int(det_chunk_map_real.max()) + 1
num_chunks_dummy = 1
num_sky = ref_shape[0] * ref_shape[1]

# ----------------------------- Predict layout -----------------------------
expected_total_cols = (num_sky
                       + num_frames * num_chunks_real
                       + num_frames * num_chunks_dummy)
print(f"Expected layout: num_sky={num_sky}, "
      f"map0 block={num_frames * num_chunks_real}, "
      f"map1 block={num_frames * num_chunks_dummy}, "
      f"total_cols={expected_total_cols}")

# ----------------------------- setup_lsqr K=2 -----------------------------
A, b, pixel_counts = setup_lsqr(
    file_list, ref_shape,
    chunk_maps=[det_chunk_map_real, det_chunk_map_dummy],
    grid_valid_weight=det_valid_mask_padded,
    apply_mask=True, apply_weight=False,
    outlier_thresh=5.0, ignore_list=[], batch_size=20,
    offset_regularization=True,
    reg_weights=[0.1, 0.1],
    adj_infos=[adj_info_real, None],   # dummy has no adjacency
    mean_offsets_list=[None, np.zeros(num_frames)],
    weighted_damping=True, damp_weight=0.1,
    max_workers=16,
)

print("\n=== Matrix-shape checks ===")
print(f"A.shape = {A.shape}")
print(f"b.shape = {b.shape}")
assert A.shape[1] == expected_total_cols, (
    f"total_cols mismatch: got {A.shape[1]}, expected {expected_total_cols}")
print("OK total_cols matches col_bases prediction")

# Per-block column population
col_real_start = num_sky
col_real_end = num_sky + num_frames * num_chunks_real
col_dummy_start = col_real_end
col_dummy_end = col_real_end + num_frames * num_chunks_dummy

real_block_nnz = ((A.col >= col_real_start) & (A.col < col_real_end)).sum()
dummy_block_nnz = ((A.col >= col_dummy_start) & (A.col < col_dummy_end)).sum()
sky_block_nnz = (A.col < num_sky).sum()
print(f"sky block nnz   = {sky_block_nnz}")
print(f"map0 block nnz  = {real_block_nnz}")
print(f"map1 block nnz  = {dummy_block_nnz}")
assert real_block_nnz > 0, "real map block has zero entries"
assert dummy_block_nnz > 0, "dummy map block has zero entries"
print("OK both offset blocks are populated")

# ----------------------------- Diagonal LS warmstart -----------------------------
x0 = compute_x0_from_Ab(A, b, ref_shape)
print(f"\nx0.shape = {x0.shape}")
assert x0.shape[0] == expected_total_cols
real_off = x0[col_real_start:col_real_end].reshape(num_frames, num_chunks_real)
dummy_off = x0[col_dummy_start:col_dummy_end].reshape(num_frames, num_chunks_dummy)
print(f"real offsets:  mean={real_off.mean():.3e}, "
      f"min={real_off.min():.3e}, max={real_off.max():.3e}")
print(f"dummy offsets: mean={dummy_off.mean():.3e}, "
      f"min={dummy_off.min():.3e}, max={dummy_off.max():.3e}")

print("\n=== K=2 smoke test PASSED ===")
