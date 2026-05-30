"""Time-bounded end-to-end before/after harness for the perf/algo-optimizations work.

Runs the full D3 Ch17 NumCol=3 pipeline (cal setup_lsqr + apply + mosaic
cache/mean/std/sigma_clip + wav) on a fixed --n-frames subset of the NVMe-staged
reproj files, with the production 48/50 config, phase-timed (wall + peak RSS) via
the shared PhaseTracker. Same harness on the pre-change SelfCal (git checkout
<parent> -- SelfCal/) vs the changed SelfCal gives a clean per-phase delta on
identical machine state.

Usage (single-shot):
    python bench_e2e.py --n-frames 4000 --label before
    python bench_e2e.py --n-frames 4000 --label after

Usage (orchestrated before-vs-after with crash-safe SelfCal restore):
    python bench_e2e.py --before-ref <sha-or-branch> --after-ref <sha-or-branch> \
                        --n-frames 4000 --label myrun
    # writes e2e_myrun_before_summary.txt and e2e_myrun_after_summary.txt under
    # figures/benchmark/; restores SelfCal/ to the recorded HEAD on completion
    # AND on SIGINT/SIGTERM/exception.

Reading the per-phase RSS columns
---------------------------------
PhaseTracker emits two RSS columns. ``peak_rss_gb`` is the max sampled
whole-process-tree RSS during the phase; for nested phases (mosaic_coadd_mean
/ _std / _sigma_clip inside mosaic_make_mosaic_total) it includes carry-over
from prior sub-phases that is still live and is therefore monotonically rising
even when individual phases free what they allocate. ``delta_rss_gb`` is
``peak_rss_gb - start_rss_gb`` where ``start_rss_gb`` is the RSS at phase
entry; this is the per-phase NEW-allocation signal. When auditing whether a
specific sub-phase allocated more memory than its baseline, read
``delta_rss_gb``, not ``peak_rss_gb``.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import gc
import glob as glob_module
import shutil
from functools import partial

import matplotlib
matplotlib.use('Agg')
import numpy as np

from SelfCal import PipelineWrapper, MakeMap
from SelfCal.MakeMap import set_hdd_io_limit
from SelfCal.solution import compute_x0_scalar_only
from SelfCal.SPHERExUtility import make_spherex_stripped_offset_map
from SelfCal.SPHERExAppendWav import wav_coadd
from run_cal_baseline_test import prepare_detector_inputs, prepare_channel_inputs
from benchmark_d3_ch17_poly import PhaseTracker

FRAME_SETTING = {'Detector': 3, 'NumSub': 10, 'NumCh': 34, 'NumCol': 3}
MOSAIC_OVERSAMPLE = 2
NVME_DIR = '/home/thomasli/selfcal-project/selfcal/cache/reproj_nvme_SPHEREx_nep_qr2_det3_6p2arcsec'
CACHE_ROOT = '/home/thomasli/selfcal-project/selfcal/cache'
BENCH_DIR = '/home/thomasli/selfcal-project/selfcal/figures/benchmark/'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-frames', type=int, default=4000)
    ap.add_argument('--label', required=True)
    ap.add_argument('--max-workers', type=int, default=48)
    args = ap.parse_args()
    os.makedirs(BENCH_DIR, exist_ok=True)

    cfg = PipelineWrapper.PipelineConfig(
        output_dir='/mnt/md124/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_nep_qr2_det{FRAME_SETTING["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2,
    )
    set_hdd_io_limit(None)

    tracker = PhaseTracker(sample_interval_s=0.5)
    tracker.start()

    with tracker.phase('detector_inputs'):
        det_inputs = prepare_detector_inputs(FRAME_SETTING, MOSAIC_OVERSAMPLE)
    with tracker.phase('channel_inputs'):
        ch_inputs = prepare_channel_inputs(
            [17], FRAME_SETTING, det_inputs['det_chunk_map'], det_inputs['grid_chunk_map'])

    files = sorted(glob_module.glob(os.path.join(NVME_DIR, '*.h5')))[:args.n_frames]
    frame_str = '_'.join(f'{k}{v}' for k, v in FRAME_SETTING.items())
    suffix = f'_e2e_{args.label}'
    cal_file = f'cal_{frame_str}_Ch17{suffix}.h5'
    cache_dir = os.path.join(CACHE_ROOT, f'cache_e2e_{args.label}')

    cc = PipelineWrapper.Calibrator(cfg, reproj_dir=NVME_DIR)
    cc.reproj_list = files
    num_frames = len(cc.reproj_list)
    print(f"[e2e:{args.label}] n_frames={num_frames}", flush=True)

    with tracker.phase('cal_setup_lsqr'):
        cc.setup_lsqr(
            chunk_maps=[det_inputs['det_chunk_map']],
            grid_valid_weight=ch_inputs['det_valid_mask_padded'],
            oversample_factor=1,
            adj_infos=[det_inputs['adj_info']],
            mean_offsets_list=[np.zeros(num_frames)],
            use_per_frame_scalar=True,
            apply_mask=True, apply_weight=False, outlier_thresh=5.0, ignore_list=[],
            batch_size=50, offset_regularization=True, reg_weights=[0.1],
            weighted_damping=True, damp_weight=0.1, max_workers=args.max_workers,
        )
    with tracker.phase('cal_warmstart'):
        x0 = compute_x0_scalar_only(cc.A, cc.b, cc.ref_shape,
                                    scalar_col_start=cc.col_bases[len(cc.chunk_maps)])
    with tracker.phase('cal_apply_lsqr'):
        cc.apply_lsqr(x0=x0, atol=1e-06, btol=1e-06, damp=0, iter_lim=50,
                      precondition=True, solver='lsqr', use_float32=True, n_threads=args.max_workers)
    with tracker.phase('cal_save'):
        cal_path = cc.save_calibration(cal_file=cal_file)
    del cc
    gc.collect()

    # Instrument each compute_coadd_map call by mode
    _orig = MakeMap.compute_coadd_map
    def _instr(mode, *a, **k):
        with tracker.phase(f'mosaic_coadd_{mode}'):
            return _orig(mode, *a, **k)
    MakeMap.compute_coadd_map = _instr

    try:
        mm = PipelineWrapper.Mosaicker(cfg, reproj_dir=NVME_DIR)
        with tracker.phase('mosaic_load_cal'):
            mm.load_calibration(cal_path=cal_path)
            mm.reproj_list = [os.path.join(NVME_DIR, os.path.basename(f)) for f in mm.reproj_list]
        offset_fn = partial(
            make_spherex_stripped_offset_map,
            chunk_valid_mask=ch_inputs['chunk_valid_mask'], lvf_params=det_inputs['lvf_params'],
            r_edges=det_inputs['r_edges'], x_edges=det_inputs['x_edges'],
            tot_subchannels=FRAME_SETTING['NumSub'] * FRAME_SETTING['NumCh'] + 2,
            num_columns=FRAME_SETTING['NumCol'], fill_invalid=True,
        )
        with tracker.phase('mosaic_make_mosaic_total'):
            maps = mm.make_mosaic(
                chunk_maps=[det_inputs['grid_chunk_map']],
                grid_valid_weight=ch_inputs['grid_valid_weight'],
                oversample_factor=MOSAIC_OVERSAMPLE,
                det_offset_funcs=[offset_fn], cache_dir=cache_dir,
                apply_mask=True, apply_weight=False,
                make_std_map=True, apply_sigma_clipping=True, sigma=2.0,
                ignore_list=[21], cache_batch_size=50, coadd_batch_size=50,
                cache_intermediate=True, max_workers=args.max_workers,
            )
    finally:
        MakeMap.compute_coadd_map = _orig

    with tracker.phase('mosaic_wav_coadd'):
        wav_coadd(det_inputs['det_BC'], det_inputs['det_BW'],
                  mean_map=maps['mean_map']['data'], std_map=maps['std_map']['data'],
                  reproj_list=mm.reproj_list, cache_list=mm.cached_list,
                  ref_shape=maps['mean_map']['data'].shape, sigma=2.0,
                  batch_size=50, max_workers=args.max_workers)

    tracker.stop()
    summary = tracker.summary_table()
    print("\n" + "=" * 110 + f"\n[e2e:{args.label}] n_frames={num_frames}\n" + summary + "\n" + "=" * 110, flush=True)
    txt = os.path.join(BENCH_DIR, f'e2e_{args.label}_summary.txt')
    with open(txt, 'w') as f:
        f.write(f"n_frames={num_frames}\n" + summary + "\n")
    print(f"[e2e:{args.label}] wrote {txt}")

    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)


if __name__ == '__main__':
    main()
