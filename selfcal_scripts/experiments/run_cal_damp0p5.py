"""Parallel test: D4 Ch1 with stronger damping + regularization.

Variant of run_cal.py for a single-channel sanity check that bumps
damp_weight 0.1 -> 0.5 and reg_weights 0.1 -> 0.5 while leaving everything
else (poly weight 0.5, NumCol=10, sigma, outlier_thresh, ...) identical to
the main run. POLY_WEIGHT stays at 0.5 to isolate the damp/reg effect.

Reuses the NVMe cache copied by the main run (run_cal.py) so this
skips the HDD->NVMe copy and the NVMe cleanup. The main run owns those.

Outputs land alongside the main run's files with FILE_SUFFIX bumped from
_damp0p1_reg0p1_outThresh5_sigma2_polyK1 to
_damp0p5_reg0p5_outThresh5_sigma2_polyK1 so nothing collides.
"""
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
from functools import partial
import numpy as np

parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

from selfcal.pipeline import PipelineWrapper
from selfcal.MakeMap import set_hdd_io_limit, OffsetModel, OffsetBlock
from selfcal.core.solution import compute_x0_scalar_only
from selfcal.instruments.spherex.SPHERExUtility import (
    load_calibration, load_lvf_params, compute_column_adjacency,
    make_stripped_chunk_map, make_stripped_chunk_valid_mask,
    make_spherex_stripped_offset_map, fast_vertical_dist,
    compute_column_polynomial_chains,
)
from selfcal.instruments.spherex.wavemap import wav_coadd

# Shared input-prep helpers (single source of truth). parent_path
# (selfcal_scripts/) is already on sys.path from the preamble above.
from _run_cal_harness import prepare_detector_inputs, prepare_channel_inputs  # noqa


if __name__ == "__main__":
    frame_setting = {
        'Detector': 4,
        'NumSub': 10,
        'NumCh': 34,
        'NumCol': 10,
    }

    selfcal_config = PipelineWrapper.PipelineConfig(
        output_dir='/mnt/md124/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_NEP_2026W17_D{frame_setting["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2,
    )

    calibration_kwargs = {
        'apply_mask': True,
        'apply_weight': False,
        'outlier_thresh': 5.0,
        'ignore_list': [],
        'batch_size': 50,
        'offset_regularization': True,
        # reg_weights (was [0.5]) moved onto the OffsetBlock built below.
        'weighted_damping': True,
        'damp_weight': 0.5,           # was 0.1
        'max_workers': 48,
        'postprocess_func': None,
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
        'max_workers': 48,
    }

    mosaic_oversample_factor = 2

    CACHE_DIR = '/home/thomasli/selfcal-project/selfcal/cache/'
    FILE_SUFFIX = '_damp0p5_reg0p5_outThresh5_sigma2_polyK1'

    POLY_DEGREE = 1
    POLY_WEIGHT = 0.5

    chs = [[1]]

    # Reuse the NVMe cache the main run already created. Do NOT copy and
    # do NOT delete on exit — the main run owns its lifecycle.
    nvme_reproj_dir = os.path.join(CACHE_DIR, f'reproj_nvme_{selfcal_config.run_name}')
    if not os.path.isdir(nvme_reproj_dir):
        raise RuntimeError(
            f"NVMe cache dir missing: {nvme_reproj_dir}. The main run is expected to have created it.")
    set_hdd_io_limit(None)

    def remap_to_nvme(file_list):
        return [os.path.join(nvme_reproj_dir, os.path.basename(f)) for f in file_list]

    frame_setting_str = '_'.join([f'{k}{v}' for k, v in frame_setting.items()])
    detector_inputs = prepare_detector_inputs(frame_setting, mosaic_oversample_factor)

    for ch in chs:
        job_name = f'Ch{"-".join(map(str, ch))}'
        t0 = time.time()
        print(f"Processing channel {job_name} for detector {frame_setting['Detector']}...")

        job_tag = f'{frame_setting_str}_{job_name}{FILE_SUFFIX}'
        cal_file = f'cal_{job_tag}.h5'
        mos_file = f'mosaic_{job_tag}.fits'
        cache_dir = f'{CACHE_DIR}cache_{job_tag}'

        channel_inputs = prepare_channel_inputs(
            ch, frame_setting,
            detector_inputs['det_chunk_map'], detector_inputs['grid_chunk_map'],
        )

        cal_path = os.path.join(selfcal_config.cal_dir, cal_file)
        cc = PipelineWrapper.Calibrator(selfcal_config, reproj_dir=nvme_reproj_dir)
        if os.path.exists(cal_path):
            print(f"Calibration file {cal_path} already exists. Skipping calibration.")
        else:
            num_frames_run = len(cc.reproj_list)
            poly_chains, poly_stencil = compute_column_polynomial_chains(
                detector_inputs['det_chunk_map'], frame_setting['NumCol'], degree=POLY_DEGREE,
            )
            poly_group = [{
                'chains': poly_chains,
                'stencil': poly_stencil,
                'weight': POLY_WEIGHT,
            }]
            offset_model = OffsetModel([
                OffsetBlock(chunk_map=detector_inputs['det_chunk_map'],
                            adj_info=detector_inputs['adj_info'],
                            reg_weight=0.5,
                            poly_constraints=poly_group,
                            mean_offset=np.zeros(num_frames_run)),
            ], use_per_frame_scalar=True)
            cc.setup_lsqr(
                offset_model=offset_model,
                grid_valid_weight=channel_inputs['det_valid_mask_padded'],
                oversample_factor=1,
                **calibration_kwargs,
            )
            x0 = compute_x0_scalar_only(
                cc.A, cc.b, cc.ref_shape,
                scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
            )
            cc.apply_lsqr(x0=x0, use_float32=True, n_threads=48, **lsqr_kwargs)
            nvme_list = cc.reproj_list
            cc.reproj_list = [os.path.join(selfcal_config.reproj_dir, os.path.basename(f)) for f in nvme_list]
            cal_path = cc.save_calibration(cal_file=cal_file)
            cc.reproj_list = nvme_list

        partial_make_offset_map = partial(
            make_spherex_stripped_offset_map,
            chunk_valid_mask=channel_inputs['chunk_valid_mask'],
            lvf_params=detector_inputs['lvf_params'],
            r_edges=detector_inputs['r_edges'],
            x_edges=detector_inputs['x_edges'],
            tot_subchannels=frame_setting['NumSub'] * frame_setting['NumCh'] + 2,
            num_columns=frame_setting['NumCol'],
            fill_invalid=True,
        )

        mm = PipelineWrapper.Mosaicker(selfcal_config, reproj_dir=nvme_reproj_dir)
        mm.load_calibration(cal_path=cal_path)
        mm.reproj_list = remap_to_nvme(mm.reproj_list)

        maps = mm.make_mosaic(
            chunk_maps=[detector_inputs['grid_chunk_map']],
            grid_valid_weight=channel_inputs['grid_valid_weight'],
            oversample_factor=mosaic_oversample_factor,
            det_offset_funcs=[partial_make_offset_map],
            cache_dir=cache_dir,
            **mosaic_kwargs,
        )

        print("Coadding wavelength maps...")
        t00 = time.time()
        wav_mean, wav_std = wav_coadd(
            detector_inputs['det_BC'], detector_inputs['det_BW'],
            mean_map=maps['mean_map']['data'],
            std_map=maps['std_map']['data'],
            reproj_list=mm.reproj_list,
            cache_list=mm.cached_list,
            ref_shape=maps['mean_map']['data'].shape,
            sigma=mosaic_kwargs['sigma'],
            batch_size=40, max_workers=30,
        )
        print(f"Wavelength coaddition finished in {time.time() - t00:.2f} seconds.")

        mm.append_maps({
            'wav_mean_map': {'data': wav_mean, 'unit': 'um'},
            'wav_std_map': {'data': wav_std, 'unit': 'um'},
        })

        mm.save_mosaic(mos_file=mos_file, overwrite=True)

        del cc, mm, maps
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)
        gc.collect()

        print(f"Finished channel {job_name} for detector {frame_setting['Detector']} in {time.time() - t0:.2f} seconds.")
