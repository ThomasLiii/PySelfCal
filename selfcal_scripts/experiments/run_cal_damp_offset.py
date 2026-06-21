import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import sys
import shutil
import time
import gc
import glob as glob_module
from functools import partial
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tqdm import tqdm
from threadpoolctl import threadpool_limits

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from selfcal.pipeline import PipelineWrapper
from selfcal.MakeMap import set_hdd_io_limit, compute_x0_from_Ab
from selfcal.core.solution import compute_x0_scalar_only
from selfcal.instruments.spherex.SPHERExUtility import make_spherex_stripped_offset_map, compute_column_polynomial_chains
# prepare_detector_inputs / prepare_channel_inputs / mask_bright_pixels are
# shared from selfcal_scripts/_run_cal_harness.py (single source of truth).
import sys as _sys
_sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from _run_cal_harness import (prepare_detector_inputs, prepare_channel_inputs,
                              mask_bright_pixels)
from selfcal.instruments.spherex.wavemap import wav_coadd


if __name__ == "__main__":
    # ----------------------------- Start of Settings -----------------------------
    frame_setting = {
        'Detector': 4,
        'NumSub': 10,
        'NumCh': 34,
        'NumCol': 10,
    }

    selfcal_config = PipelineWrapper.PipelineConfig(
        output_dir='/data3/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_NEP_2026W17_D{frame_setting["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2
    )

    calibration_kwargs = {
        'apply_mask': True,
        'apply_weight': True,  # NEW: enable inverse-variance brightness weighting (1/sqrt(|data|+floor), Poisson-optimal). Bright cirrus pixels contribute ~10x less to the LSQR fit than dim sky → offsets are determined mostly from dim background → no sky-leakage into offsets → no bowl-around-cirrus at apply time. See fix/offset-damping branch (Option A).
        'outlier_thresh': 5.0,
        'ignore_list': [],
        'batch_size': 50,
        'offset_regularization': True,
        'reg_weights': [0.1],
        'weighted_damping': True,
        'damp_weight': 0.1,
        # damp_offset reverted to 0 — hybrid test (apply_weight + damp_offset=0.1) was WORSE than applyWt alone (cirrus dark_spread widened further, dark-ring re-emerged). The two levers aren't orthogonal — both reweight toward bright/well-covered chunks. See aromatic-map-tuning session.
        'max_workers': 48,
        'postprocess_func': None, #mask_bright_pixels,
    }

    lsqr_kwargs = {
        'atol': 1e-06,
        'btol': 1e-06,
        'damp': 0,
        'iter_lim': 50,
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
        'cache_batch_size': 50,
        'coadd_batch_size': 50,
        'cache_intermediate': True,
        'max_workers': 48
    }
    
    mosaic_oversample_factor = 2

    CACHE_DIR = '/home/thomasli/selfcal-project/selfcal/cache/'
    FILE_SUFFIX = f'_damp0p1_reg0p1_applyWt_outThresh5_sigma2_polyK1'

    # Linear column constraint weight (compute_column_polynomial_chains, degree=1)
    POLY_DEGREE = 1
    POLY_WEIGHT = 0.5

    # Channels to process
    # Restricted to Aromatic for the damp_offset fix verification (fix/offset-damping branch).
    chs = ['Aromatic']
    # Max concurrent HDD reads — prevents RAID thrashing when multiple instances run.
    # Tune based on RAID config: ~4-8 for most RAID arrays. Set to None to disable.
    HDD_IO_LIMIT = 20
    # ----------------------------- End of Settings -----------------------------

    set_hdd_io_limit(HDD_IO_LIMIT)

    # Copy reproj files from HDD to NVMe for faster I/O.
    # Per-run subdirectory so multiple runs can coexist without colliding.
    nvme_reproj_dir = os.path.join(CACHE_DIR, f'reproj_nvme_{selfcal_config.run_name}')
    os.makedirs(nvme_reproj_dir, exist_ok=True)

    hdd_reproj_files = sorted(glob_module.glob(os.path.join(selfcal_config.reproj_dir, '*.h5')))

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

    frame_setting_str = '_'.join([f'{key}{value}' for key, value in frame_setting.items()])

    # 1. Prepare overarching detector inputs
    detector_inputs = prepare_detector_inputs(frame_setting, mosaic_oversample_factor)

    # 2. Iterate through channels
    for ch in chs:
        if isinstance(ch, list):
            job_name = f'Ch{"-".join(map(str, ch))}'
        else:
            job_name = ch
        t0 = time.time()
        print(f"Processing channel {job_name} for detector {frame_setting['Detector']}...")

        job_tag = f'{frame_setting_str}_{job_name}{FILE_SUFFIX}'
        cal_file = f'cal_{job_tag}.h5'
        mos_file = f"mosaic_{job_tag}.fits"
        cache_dir = f'{CACHE_DIR}cache_{job_tag}'

        # Prepare specific inputs for this channel
        channel_inputs = prepare_channel_inputs(ch, frame_setting, detector_inputs['det_chunk_map'], detector_inputs['grid_chunk_map'])
        
        # ----------------------------- Calibration -----------------------------
        cal_path = os.path.join(selfcal_config.cal_dir, cal_file)
        cc = PipelineWrapper.Calibrator(selfcal_config, reproj_dir=nvme_reproj_dir)
        if os.path.exists(cal_path):
            print(f"Calibration file {cal_path} already exists. Skipping calibration.")
        else:
            # Per-frame scalar absorbs DC; mean-anchor on map 0 chunks forces
            # within-frame structure only on the chunks. Required to avoid
            # scan-stripe residuals on narrow channel masks — see PIPELINE.md.
            num_frames_run = len(cc.reproj_list)
            poly_chains, poly_stencil = compute_column_polynomial_chains(
                detector_inputs['det_chunk_map'], frame_setting['NumCol'], degree=POLY_DEGREE,
            )
            poly_constraints_list = [[{
                'chains': poly_chains,
                'stencil': poly_stencil,
                'weight': POLY_WEIGHT,
            }]]
            cc.setup_lsqr(
                chunk_maps=[detector_inputs['det_chunk_map']],
                grid_valid_weight=channel_inputs['det_valid_mask_padded'],
                oversample_factor=1,
                adj_infos=[detector_inputs['adj_info']],
                poly_constraints_list=poly_constraints_list,
                mean_offsets_list=[np.zeros(num_frames_run)],  # Restore anchor; with mean_anchor_coverage_weighted=True, the constraint is pinned to coverage-weighted mean (Option 2.5).
                use_per_frame_scalar=True,
                **calibration_kwargs
            )

            x0 = compute_x0_scalar_only(
                cc.A, cc.b, cc.ref_shape,
                scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
                active_mask=cc.active_mask,
            )

            cc.apply_lsqr(x0=x0, use_float32=True, n_threads=48, **lsqr_kwargs)
            # Save with original HDD paths so cal file remains valid after NVMe cleanup
            nvme_list = cc.reproj_list
            cc.reproj_list = [os.path.join(selfcal_config.reproj_dir, os.path.basename(f)) for f in nvme_list]
            cal_path = cc.save_calibration(cal_file=cal_file)
            cc.reproj_list = nvme_list

        # ----------------------------- Mosaicking -----------------------------
        partial_make_offset_map = partial(make_spherex_stripped_offset_map,
                                    chunk_valid_mask=channel_inputs['chunk_valid_mask'], 
                                    lvf_params=detector_inputs['lvf_params'], 
                                    r_edges=detector_inputs['r_edges'], 
                                    x_edges=detector_inputs['x_edges'], 
                                    tot_subchannels=frame_setting['NumSub']*frame_setting['NumCh']+2, 
                                    num_columns=frame_setting['NumCol'],
                                    fill_invalid=True)
        
        mm = PipelineWrapper.Mosaicker(selfcal_config, reproj_dir=nvme_reproj_dir)
        mm.load_calibration(cal_path=cal_path)
        mm.reproj_list = remap_to_nvme(mm.reproj_list)

        maps = mm.make_mosaic(
            chunk_maps=[detector_inputs['grid_chunk_map']],
            grid_valid_weight=channel_inputs['grid_valid_weight'],
            oversample_factor=mosaic_oversample_factor,
            det_offset_funcs=[partial_make_offset_map],
            cache_dir=cache_dir,
            **mosaic_kwargs
        )

        # Append wavelength maps
        print("Coadding wavelength maps...")
        t00 = time.time()
        wav_mean, wav_std = wav_coadd(detector_inputs['det_BC'], detector_inputs['det_BW'], 
                                      mean_map=maps['mean_map']['data'], 
                                      std_map=maps['std_map']['data'], 
                                      reproj_list=mm.reproj_list, 
                                      cache_list=mm.cached_list,
                                      ref_shape=maps['mean_map']['data'].shape, 
                                      sigma=mosaic_kwargs['sigma'], 
                                      batch_size=40, max_workers=30)    
        print(f"Wavelength coaddition finished in {time.time() - t00:.2f} seconds.")

        mm.append_maps({
            'wav_mean_map': {'data': wav_mean, 'unit': 'um'},
            'wav_std_map': {'data': wav_std, 'unit': 'um'}
        })

        mm.save_mosaic(mos_file=mos_file, overwrite=True)
         
        # Clean up
        del cc, mm, maps
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        gc.collect()
        
        print(f"Finished channel {job_name} for detector {frame_setting['Detector']} in {time.time() - t0:.2f} seconds.")
        print("-" * 50 + "\n")

    # Cleanup NVMe reproj cache (skip when iterating — set KEEP_NVME_CACHE
    # to avoid re-staging 327 GB from HDD on every rerun)
    KEEP_NVME_CACHE = True
    if not KEEP_NVME_CACHE and os.path.exists(nvme_reproj_dir):
        shutil.rmtree(nvme_reproj_dir)
        print("NVMe reproj cache cleaned up.")
    elif KEEP_NVME_CACHE and os.path.exists(nvme_reproj_dir):
        print(f"NVMe reproj cache preserved at {nvme_reproj_dir} (KEEP_NVME_CACHE=True).")
