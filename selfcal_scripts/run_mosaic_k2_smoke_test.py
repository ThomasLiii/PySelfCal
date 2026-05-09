"""K=2 mosaic smoke test for Commit 4.

Builds two synthetic per-frame offset lists on a small subset of reproj
files and verifies that ``compute_coadd_map`` produces an equivalent mean
map under either:

  - K=2: ``chunk_maps=[real, dummy]``, ``offset_lists=[off_real, off_dummy]``
    where ``dummy`` is a single-chunk map that absorbs a per-frame DC term;
  - K=1: ``chunk_maps=[real]``, ``offset_lists=[off_real_combined]`` where
    the dummy's per-frame DC has been broadcast into the real map's
    per-chunk offsets.

Mathematically the per-pixel subtracted offset is identical in both runs,
so the mosaic outputs should match to floating-point tolerance. This is a
plumbing-level check (no calibration solve), proving the per-map SHM
hand-off, per-batch slicing, and ``_prep_subframe`` accumulation are
wired correctly.
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

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from SelfCal import WCSHelper
from SelfCal.coadd import compute_coadd_map
from SelfCal.SPHERExUtility import (
    load_lvf_params, make_stripped_chunk_map, make_stripped_chunk_valid_mask)


# ----------------------------- Settings -----------------------------
DETECTOR = 3
NUM_SUB = 10
NUM_CH = 34
NUM_COL = 3
CH_LIST = [17]
NUM_FRAMES_LIMIT = 20  # tiny — pure plumbing check
SEED = 42

CACHE_DIR = '/home/thomasli/spherex/selfcal/cache/'
RUN_NAME = f'SPHEREx_nep_qr2_det{DETECTOR}_6p2arcsec'
NVME_REPROJ_DIR = os.path.join(CACHE_DIR, f'reproj_nvme_{RUN_NAME}')
REF_PATH = f'/mnt/md124/thomasli/selfcal/outputs/{RUN_NAME}/ref.fits'

# ----------------------------- Inputs -----------------------------
print("Loading reference WCS + chunk maps...")
ref_wcs, ref_shape = WCSHelper.load_from_fits(REF_PATH)

lvf_params = load_lvf_params(f'lvf_params_D{DETECTOR}.npy')
det_chunk_map_real, _, _, _ = make_stripped_chunk_map(
    DETECTOR, num_subchannels=NUM_SUB, num_channels=NUM_CH,
    num_columns=NUM_COL, oversample_factor=1, lvf_params=lvf_params)
det_chunk_map_dummy = np.zeros_like(det_chunk_map_real, dtype=np.int32)

chunk_valid_mask_padded = make_stripped_chunk_valid_mask(
    ch=CH_LIST, num_subchannels=NUM_SUB, num_channels=NUM_CH,
    num_columns=NUM_COL, subchannel_padding=1)
det_valid_mask_padded = chunk_valid_mask_padded[det_chunk_map_real]

all_files = sorted(glob_module.glob(os.path.join(NVME_REPROJ_DIR, '*.h5')))
file_list = all_files[:NUM_FRAMES_LIMIT]
print(f"Using {len(file_list)} of {len(all_files)} reproj files")

num_frames = len(file_list)
num_chunks_real = int(det_chunk_map_real.max()) + 1

# Synthetic offsets: small random per-(frame, chunk) values for the real map,
# random per-frame DC for the dummy map.
rng = np.random.default_rng(SEED)
off_real = rng.standard_normal((num_frames, num_chunks_real)) * 0.05
off_dummy = (rng.standard_normal((num_frames, 1)) * 0.10).astype(np.float64)
off_real_combined = off_real + off_dummy  # broadcast (n_frames, 1) → (n_frames, n_chunks)

common = dict(
    ref_shape=ref_shape,
    file_list=file_list,
    grid_valid_weight=det_valid_mask_padded,
    apply_mask=True, apply_weight=False,
    max_workers=8, batch_size=10, oversample_factor=1,
)

# ----------------------------- K=1 (combined) -----------------------------
print("\n=== K=1 mosaic with combined offsets ===")
mean_k1, weight_k1, _ = compute_coadd_map(
    mode='mean',
    chunk_maps=[det_chunk_map_real],
    offset_lists=[off_real_combined],
    **common,
)
print(f"  mean shape={mean_k1.shape}, nz={int((weight_k1 > 0).sum())} / {weight_k1.size}")

# ----------------------------- K=2 (split) -----------------------------
print("\n=== K=2 mosaic with [real, dummy] offsets ===")
mean_k2, weight_k2, _ = compute_coadd_map(
    mode='mean',
    chunk_maps=[det_chunk_map_real, det_chunk_map_dummy],
    offset_lists=[off_real, off_dummy],
    **common,
)
print(f"  mean shape={mean_k2.shape}, nz={int((weight_k2 > 0).sum())} / {weight_k2.size}")

# ----------------------------- Compare -----------------------------
both_valid = (weight_k1 > 0) & (weight_k2 > 0)
print(f"\n=== Pixel-wise diff (K=1 combined vs K=2 split, where both valid) ===")
diff = (mean_k1 - mean_k2)[both_valid]
print(f"  n valid pixels: {both_valid.sum()}")
print(f"  max |diff|: {np.max(np.abs(diff)):.3e}")
print(f"  rms diff:   {np.sqrt(np.mean(diff ** 2)):.3e}")
print(f"  max(|mean_k1|): {np.max(np.abs(mean_k1[both_valid])):.3e}")

assert np.max(np.abs(diff)) < 1e-5, \
    "K=2 split should match K=1 combined within float-summation tolerance"
print("\n=== K=2 mosaic smoke test PASSED ===")
