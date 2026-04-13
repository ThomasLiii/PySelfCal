# %% [markdown]
# # Exploded Pipeline Benchmark — Ch16, Det1
# Each cell corresponds to one pipeline step. Run individually to time each.

# %% Cell 0: Imports & Environment Setup
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import time
import shutil
import gc
import glob as glob_module
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from threadpoolctl import threadpool_limits

parent_path = os.path.dirname(os.path.dirname(os.path.abspath("__file__")))
sys.path.insert(0, parent_path)

from SelfCal import PipelineWrapper
from SelfCal.MakeMap import set_hdd_io_limit, compute_x0_from_Ab
from SelfCal.SPHERExUtility import (
    load_calibration, load_lvf_params, compute_subchannel_adjacency,
    make_stripped_chunk_map, make_stripped_chunk_valid_mask,
    make_spherex_stripped_offset_map, fast_vertical_dist
)
from SelfCal.SPHERExAppendWav import wav_coadd
import numpy as np

HDD_IO_LIMIT = 20
set_hdd_io_limit(HDD_IO_LIMIT)

# %% Cell 0b: Copy reproj files from HDD to NVMe
CACHE_DIR = '/home/thomasli/spherex/selfcal/cache/'

selfcal_config_tmp = PipelineWrapper.PipelineConfig(
    output_dir='/mnt/md124/thomasli/selfcal/outputs/',
    run_name=f'SPHEREx_nep_qr2_det5_6p2arcsec',
    resolution_arcsec=6.2
)

nvme_reproj_dir = os.path.join(CACHE_DIR, f'reproj_nvme_{selfcal_config_tmp.run_name}')
os.makedirs(nvme_reproj_dir, exist_ok=True)

hdd_reproj_files = sorted(glob_module.glob(os.path.join(selfcal_config_tmp.reproj_dir, '*.h5')))

def copy_to_nvme(src_path):
    dst_path = os.path.join(nvme_reproj_dir, os.path.basename(src_path))
    if not os.path.exists(dst_path):
        shutil.copy2(src_path, dst_path)
    return dst_path

print(f"Copying {len(hdd_reproj_files)} reproj files to NVMe ({nvme_reproj_dir})...")
t_copy = time.time()
with ThreadPoolExecutor(max_workers=HDD_IO_LIMIT or 20) as executor:
    for _ in tqdm(executor.map(copy_to_nvme, hdd_reproj_files),
                  total=len(hdd_reproj_files), desc="HDD->NVMe", unit="file"):
        pass
print(f"Reproj file copy complete in {time.time() - t_copy:.2f} seconds.")

# NVMe can handle massively parallel reads — disable the HDD I/O throttle
set_hdd_io_limit(None)

def remap_to_nvme(file_list):
    """Replace directory prefix with nvme_reproj_dir, keeping filenames."""
    return [os.path.join(nvme_reproj_dir, os.path.basename(f)) for f in file_list]

del selfcal_config_tmp

# %% Cell 1: Settings
frame_setting = {
    'Detector': 5,
    'NumSub': 10,
    'NumCh': 34,
    'NumCol': 5,
}

detector = frame_setting['Detector']
ch = [16]

selfcal_config = PipelineWrapper.PipelineConfig(
    output_dir='/mnt/md124/thomasli/selfcal/outputs/',
    run_name=f'SPHEREx_nep_qr2_det{detector}_6p2arcsec',
    resolution_arcsec=6.2
)

calibration_kwargs = {
    'apply_mask': True,
    'apply_weight': False,
    'outlier_thresh': 5.0,
    'ignore_list': [],
    'batch_size': 20,
    'offset_regularization': True,
    'reg_weight': 0.1,
    'weighted_damping': True,
    'damp_weight': 0.1,
    'max_workers': 32,
    'postprocess_func': None,
}

lsqr_kwargs = {
    'atol': 1e-06,
    'btol': 1e-06,
    'damp': 0,
    'iter_lim': 10,
    'precondition': True,
    'solver': 'lsqr',
}

mosaic_kwargs = {
    'apply_mask': True,
    'apply_weight': False,
    'make_std_map': True,
    'apply_sigma_clipping': True,
    'sigma': 2.0,
    'ignore_list': [21],
    'cache_batch_size': 20,
    'coadd_batch_size': 30,
    'cache_intermediate': True,
    'max_workers': 32,
}

mosaic_oversample_factor = 2
FILE_SUFFIX = '_damp0p1_reg0p1_outThresh5_sigma2_test'

frame_setting_str = '_'.join([f'{key}{value}' for key, value in frame_setting.items()])
job_name = f'Ch{"-".join(map(str, ch))}'
job_tag = f'{frame_setting_str}_{job_name}{FILE_SUFFIX}'
cache_dir = f'{CACHE_DIR}cache_{job_tag}'
cal_file = f'cal_{job_tag}.h5'
mos_file = f'mosaic_{job_tag}.fits'

# %% Cell 2: Prepare Detector Inputs
t0 = time.time()

lvf_filename = f'lvf_params_D{detector}.npy'
lvf_params = load_lvf_params(lvf_filename)

det_BC, det_BW = load_calibration(band=detector, calibration_dir='/home/thomasli/spherex/SPHEREx_Spectral_Calibration')

grid_chunk_map, _, _, _ = make_stripped_chunk_map(
    detector, num_subchannels=frame_setting['NumSub'], num_channels=frame_setting['NumCh'],
    num_columns=frame_setting['NumCol'], oversample_factor=mosaic_oversample_factor, lvf_params=lvf_params
)
det_chunk_map, _, r_edges, x_edges = make_stripped_chunk_map(
    detector, num_subchannels=frame_setting['NumSub'], num_channels=frame_setting['NumCh'],
    num_columns=frame_setting['NumCol'], oversample_factor=1, lvf_params=lvf_params
)

adj_info = compute_subchannel_adjacency(det_chunk_map, frame_setting['NumCol'])

print(f"Detector inputs prepared in {time.time() - t0:.2f}s")

# %% Cell 3: Prepare Channel Inputs
t0 = time.time()

chunk_valid_mask_padded = make_stripped_chunk_valid_mask(
    ch=ch, num_subchannels=frame_setting['NumSub'], num_channels=frame_setting['NumCh'],
    num_columns=frame_setting['NumCol'], subchannel_padding=1
)
chunk_valid_mask = make_stripped_chunk_valid_mask(
    ch=ch, num_subchannels=frame_setting['NumSub'], num_channels=frame_setting['NumCh'],
    num_columns=frame_setting['NumCol'], subchannel_padding=0
)

det_valid_mask = chunk_valid_mask[det_chunk_map]
det_valid_weight = fast_vertical_dist(det_valid_mask)
if np.max(det_valid_weight) > 0:
    det_valid_weight /= np.max(det_valid_weight)

det_valid_mask_padded = chunk_valid_mask_padded[det_chunk_map]

grid_valid_mask = chunk_valid_mask[grid_chunk_map]
grid_valid_weight = fast_vertical_dist(grid_valid_mask)
if np.max(grid_valid_weight) > 0:
    grid_valid_weight /= np.max(grid_valid_weight)

print(f"Channel inputs prepared in {time.time() - t0:.2f}s")

# %% Cell 4: Calibration (setup_lsqr + apply_lsqr + save)
t0 = time.time()

cal_path = os.path.join(selfcal_config.cal_dir, cal_file)
cc = PipelineWrapper.Calibrator(selfcal_config, reproj_dir=nvme_reproj_dir)

if os.path.exists(cal_path):
    print(f"Calibration file {cal_path} already exists. Skipping calibration.")
else:
    cc.setup_lsqr(
        chunk_map=det_chunk_map,
        grid_valid_weight=det_valid_mask_padded,
        oversample_factor=1,
        adj_info=adj_info,
        **calibration_kwargs
    )
    print(f"  setup_lsqr finished in {time.time() - t0:.2f}s")
    print(f"  A shape: {cc.A.shape}, nnz: {cc.A.nnz}")
    print(f"  b shape: {cc.b.shape}")

    t1 = time.time()
    x0 = compute_x0_from_Ab(cc.A, cc.b, cc.ref_shape, len(cc.reproj_list))
    print(f"  Initial guess computed in {time.time() - t1:.2f}s, x0 shape: {x0.shape}")

    t1 = time.time()
    with threadpool_limits(limits=8, user_api='blas'):
        cc.apply_lsqr(x0=x0, **lsqr_kwargs)
    print(f"  apply_lsqr finished in {time.time() - t1:.2f}s")

    t1 = time.time()
    # Save with original HDD paths so cal file remains valid after NVMe cleanup
    nvme_list = cc.reproj_list
    cc.reproj_list = [os.path.join(selfcal_config.reproj_dir, os.path.basename(f)) for f in nvme_list]
    cal_path = cc.save_calibration(cal_file=cal_file)
    cc.reproj_list = nvme_list
    print(f"  Calibration saved in {time.time() - t1:.2f}s")

print(f"Calibration total: {time.time() - t0:.2f}s")

# %% Cell 5: Mosaicking (load_calibration + make_mosaic)
t0 = time.time()

partial_make_offset_map = partial(
    make_spherex_stripped_offset_map,
    chunk_valid_mask=chunk_valid_mask,
    lvf_params=lvf_params,
    r_edges=r_edges,
    x_edges=x_edges,
    tot_subchannels=frame_setting['NumSub'] * frame_setting['NumCh'] + 2,
    num_columns=frame_setting['NumCol'],
    fill_invalid=True
)

mm = PipelineWrapper.Mosaicker(selfcal_config, reproj_dir=nvme_reproj_dir)
mm.load_calibration(cal_path=cal_path)
mm.reproj_list = remap_to_nvme(mm.reproj_list)

maps = mm.make_mosaic(
    chunk_map=grid_chunk_map,
    grid_valid_weight=grid_valid_weight,
    oversample_factor=mosaic_oversample_factor,
    det_offset_func=partial_make_offset_map,
    cache_dir=cache_dir,
    **mosaic_kwargs
)

print(f"Mosaicking total: {time.time() - t0:.2f}s")

# %% Cell 6: Wavelength Coaddition
t0 = time.time()

wav_mean, wav_std = wav_coadd(
    det_BC, det_BW,
    mean_map=maps['mean_map']['data'],
    std_map=maps['std_map']['data'],
    reproj_list=mm.reproj_list,
    cache_list=mm.cached_list,
    ref_shape=maps['mean_map']['data'].shape,
    sigma=mosaic_kwargs['sigma'],
    batch_size=40,
    max_workers=30
)

mm.append_maps({
    'wav_mean_map': {'data': wav_mean, 'unit': 'um'},
    'wav_std_map': {'data': wav_std, 'unit': 'um'}
})

print(f"Wavelength coaddition finished in {time.time() - t0:.2f}s")

# %% Cell 7: Save Mosaic
t0 = time.time()

mm.save_mosaic(mos_file=mos_file, overwrite=True)

print(f"Mosaic saved in {time.time() - t0:.2f}s")

# %% Cell 8: Cleanup
t0 = time.time()

del cc, mm, maps
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
if os.path.exists(nvme_reproj_dir):
    shutil.rmtree(nvme_reproj_dir)
    print("NVMe reproj cache cleaned up.")
gc.collect()

print(f"Cleanup finished in {time.time() - t0:.2f}s")
