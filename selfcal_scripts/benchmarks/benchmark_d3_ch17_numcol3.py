"""Benchmark calibration + mosaicking pipeline for D3 Ch17 at NumCol=3, no poly.

Counterpart to benchmark_d3_ch17_poly.py. Config differs from that script
only in:
  - frame_setting NumCol 10 -> 3
  - poly_constraints_list dropped (no polynomial column constraint)

All other settings are identical (K=1, column adjacency reg, weighted damping,
use_per_frame_scalar=True with compute_x0_scalar_only warm start, batch sizes,
max_workers, lsqr/mosaic kwargs).

Outputs (under figures/benchmark/):
  d3_ch17_numcol3_summary.txt
  d3_ch17_numcol3_samples.json
  d3_ch17_numcol3_timeline.png

Cal + mosaic files use FILE_SUFFIX='_bench_d3_ch17_numcol3' so they don't
collide with production runs or the poly_k1 benchmark.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import gc
import glob as glob_module
import logging
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import matplotlib
matplotlib.use('Agg')
import numpy as np
from tqdm import tqdm

from selfcal.pipeline import pipeline_wrapper
from selfcal.core import coadd
from selfcal._state import set_hdd_io_limit
from selfcal.core.solution import compute_x0_scalar_only
from selfcal.instruments.spherex.spherex_utility import make_spherex_stripped_offset_map
from selfcal.instruments.spherex.wavemap import wav_coadd
from run_cal_baseline_test import prepare_detector_inputs, prepare_channel_inputs
from benchmark_d3_ch17_poly import PhaseTracker


def main():
    # selfcal library logs -> plain stdout, matching the historical print()
    # console output byte-for-byte (log parsers match on these lines).
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)

    frame_setting = {
        'Detector': 3,
        'NumSub': 10,
        'NumCh': 34,
        # NumCol=3: the pre-poly-constraint production setting;
        # benchmark_d3_ch17_poly.py is the NumCol=10 + poly counterpart.
        'NumCol': 3,
    }

    selfcal_config = pipeline_wrapper.PipelineConfig(
        output_dir='/mnt/md124/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_nep_qr2_det{frame_setting["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2,
    )

    calibration_kwargs = {
        'apply_mask': True,
        'apply_weight': False,
        'outlier_thresh': 5.0,
        'ignore_list': [],
        'batch_size': 20,
        'offset_regularization': True,
        'reg_weights': [0.1],
        'weighted_damping': True,
        'damp_weight': 0.1,
        'max_workers': 32,
        'postprocess_func': None,
    }
    lsqr_kwargs = {
        'atol': 1e-06, 'btol': 1e-06, 'damp': 0,
        'iter_lim': 50, 'precondition': True, 'solver': 'lsqr',
    }
    mosaic_kwargs = {
        'apply_mask': True, 'apply_weight': False,
        'make_std_map': True, 'apply_sigma_clipping': True, 'sigma': 2.0,
        'ignore_list': [21],
        'cache_batch_size': 20, 'coadd_batch_size': 30,
        'cache_intermediate': True, 'max_workers': 32,
    }
    mosaic_oversample_factor = 2

    CACHE_DIR = '/home/thomasli/selfcal-project/selfcal/cache/'
    BENCH_DIR = '/home/thomasli/selfcal-project/selfcal/figures/benchmark/'
    os.makedirs(BENCH_DIR, exist_ok=True)
    FILE_SUFFIX = '_bench_d3_ch17_numcol3'
    chs = [[17]]
    HDD_IO_LIMIT = 20

    set_hdd_io_limit(HDD_IO_LIMIT)

    tracker = PhaseTracker(sample_interval_s=0.5)
    tracker.start()
    print(
        f"[bench] config: D{frame_setting['Detector']} ch={chs[0]} "
        f"NumCol={frame_setting['NumCol']} no_poly use_per_frame_scalar=True",
        flush=True,
    )

    nvme_reproj_dir = os.path.join(
        CACHE_DIR, f'reproj_nvme_{selfcal_config.run_name}'
    )
    os.makedirs(nvme_reproj_dir, exist_ok=True)
    hdd_reproj_files = sorted(glob_module.glob(os.path.join(selfcal_config.reproj_dir, '*.h5')))

    with tracker.phase('transfer_hdd_to_nvme'):
        n_existing = sum(
            1 for f in hdd_reproj_files
            if os.path.exists(os.path.join(nvme_reproj_dir, os.path.basename(f)))
        )
        if n_existing == len(hdd_reproj_files):
            print(f"[bench] NVMe cache already populated ({len(hdd_reproj_files)} files); skipping copy", flush=True)
        else:
            print(f"[bench] transferring {len(hdd_reproj_files) - n_existing} files HDD->NVMe...", flush=True)
            def copy_to_nvme(src_path):
                dst_path = os.path.join(nvme_reproj_dir, os.path.basename(src_path))
                if not os.path.exists(dst_path):
                    shutil.copy2(src_path, dst_path)
                return dst_path
            with ThreadPoolExecutor(max_workers=HDD_IO_LIMIT or 20) as executor:
                for _ in tqdm(executor.map(copy_to_nvme, hdd_reproj_files),
                              total=len(hdd_reproj_files), desc="HDD->NVMe", unit="file"):
                    pass

    set_hdd_io_limit(None)

    with tracker.phase('detector_inputs'):
        det_inputs = prepare_detector_inputs(frame_setting, mosaic_oversample_factor)

    ch = chs[0]
    with tracker.phase('channel_inputs'):
        ch_inputs = prepare_channel_inputs(
            ch, frame_setting,
            det_inputs['det_chunk_map'], det_inputs['grid_chunk_map'],
        )

    frame_setting_str = '_'.join([f'{k}{v}' for k, v in frame_setting.items()])
    job_name = f"Ch{'-'.join(map(str, ch))}"
    job_tag = f'{frame_setting_str}_{job_name}{FILE_SUFFIX}'
    cal_file = f'cal_{job_tag}.h5'
    mos_file = f'mosaic_{job_tag}.fits'
    cache_dir = os.path.join(CACHE_DIR, f'cache_{job_tag}')

    cc = pipeline_wrapper.Calibrator(selfcal_config, reproj_dir=nvme_reproj_dir)
    num_frames_run = len(cc.reproj_list)
    print(f"[bench] num_frames={num_frames_run}  num_chunks={int(det_inputs['det_chunk_map'].max())+1}", flush=True)

    with tracker.phase('cal_setup_lsqr'):
        cc.setup_lsqr(
            chunk_maps=[det_inputs['det_chunk_map']],
            grid_valid_weight=ch_inputs['det_valid_mask_padded'],
            oversample_factor=1,
            adj_infos=[det_inputs['adj_info']],
            mean_offsets_list=[np.zeros(num_frames_run)],
            use_per_frame_scalar=True,
            **calibration_kwargs,
        )

    with tracker.phase('cal_warmstart'):
        x0 = compute_x0_scalar_only(
            cc.A, cc.b, cc.ref_shape,
            scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
            active_mask=cc.active_mask,
        )

    with tracker.phase('cal_apply_lsqr'):
        cc.apply_lsqr(x0=x0, use_float32=True, n_threads=32, **lsqr_kwargs)

    with tracker.phase('cal_save'):
        nvme_list = cc.reproj_list
        cc.reproj_list = [
            os.path.join(selfcal_config.reproj_dir, os.path.basename(f))
            for f in nvme_list
        ]
        cal_path = cc.save_calibration(cal_file=cal_file)
        cc.reproj_list = nvme_list

    del cc
    gc.collect()

    _orig_compute_coadd_map = coadd.compute_coadd_map

    def _instrumented_compute_coadd_map(mode, *args, **kwargs):
        with tracker.phase(f'mosaic_coadd_{mode}'):
            return _orig_compute_coadd_map(mode, *args, **kwargs)

    coadd.compute_coadd_map = _instrumented_compute_coadd_map

    try:
        mm = pipeline_wrapper.Mosaicker(selfcal_config, reproj_dir=nvme_reproj_dir)

        with tracker.phase('mosaic_load_cal'):
            mm.load_calibration(cal_path=cal_path)
            mm.reproj_list = [
                os.path.join(nvme_reproj_dir, os.path.basename(f))
                for f in mm.reproj_list
            ]

        partial_make_offset_map = partial(
            make_spherex_stripped_offset_map,
            chunk_valid_mask=ch_inputs['chunk_valid_mask'],
            lvf_params=det_inputs['lvf_params'],
            r_edges=det_inputs['r_edges'],
            x_edges=det_inputs['x_edges'],
            tot_subchannels=frame_setting['NumSub'] * frame_setting['NumCh'] + 2,
            num_columns=frame_setting['NumCol'],
            fill_invalid=True,
        )

        with tracker.phase('mosaic_make_mosaic_total'):
            maps = mm.make_mosaic(
                chunk_maps=[det_inputs['grid_chunk_map']],
                grid_valid_weight=ch_inputs['grid_valid_weight'],
                oversample_factor=mosaic_oversample_factor,
                det_offset_funcs=[partial_make_offset_map],
                cache_dir=cache_dir,
                **mosaic_kwargs,
            )
    finally:
        coadd.compute_coadd_map = _orig_compute_coadd_map

    with tracker.phase('mosaic_wav_coadd'):
        wav_mean, wav_std = wav_coadd(
            det_inputs['det_BC'], det_inputs['det_BW'],
            mean_map=maps['mean_map']['data'],
            std_map=maps['std_map']['data'],
            reproj_list=mm.reproj_list,
            cache_list=mm.cached_list,
            ref_shape=maps['mean_map']['data'].shape,
            sigma=mosaic_kwargs['sigma'],
            batch_size=40, max_workers=30,
        )

    with tracker.phase('mosaic_append_maps'):
        mm.append_maps({
            'wav_mean_map': {'data': wav_mean, 'unit': 'um'},
            'wav_std_map': {'data': wav_std, 'unit': 'um'},
        })

    with tracker.phase('mosaic_save'):
        mm.save_mosaic(mos_file=mos_file, overwrite=True)

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)

    tracker.stop()

    summary = tracker.summary_table()
    print()
    print("=" * 120, flush=True)
    print(summary, flush=True)
    print("=" * 120, flush=True)

    txt_path = os.path.join(BENCH_DIR, 'd3_ch17_numcol3_summary.txt')
    json_path = os.path.join(BENCH_DIR, 'd3_ch17_numcol3_samples.json')
    png_path = os.path.join(BENCH_DIR, 'd3_ch17_numcol3_timeline.png')

    with open(txt_path, 'w') as f:
        f.write(summary + '\n')
    tracker.save_json(json_path)
    # PhaseTracker.plot_timeline hardcodes a generic title; the output
    # filename identifies the variant.
    import matplotlib.pyplot as plt  # unused (matplotlib.use('Agg') at module top is all this script needs)
    tracker.plot_timeline(png_path)

    print(f"\n[bench] wrote {txt_path}")
    print(f"[bench] wrote {json_path}")
    print(f"[bench] wrote {png_path}")


if __name__ == '__main__':
    main()
