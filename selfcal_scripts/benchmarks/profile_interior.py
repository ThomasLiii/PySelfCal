"""Per-frame interior profiler for the cal setup_lsqr and coadd hot paths.

Step-1 tool for the perf/algo-optimizations work. Runs the per-frame interior
*in-process* (single process, no pool) so cProfile actually sees the work, on a
fixed, sorted subset of the D3 Ch17 run. Use the SAME --n-frames / --detector /
--channel before and after a change to read the delta.

Config mirrors benchmark_d3_ch17_numcol3.py (D3 Ch17, NumCol=3,
use_per_frame_scalar=True, reg_weights=[0.1], outlier_thresh=5.0).

Usage:
    python profile_interior.py --phase lsqr  --n-frames 60
    python profile_interior.py --phase cache --n-frames 60
    python profile_interior.py --phase mean  --n-frames 60   # builds cache first
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import cProfile
import glob as glob_module
import pstats
import tempfile
import time
from functools import partial

import numpy as np

from SelfCal.MakeMap import set_hdd_io_limit
from SelfCal.lsqr import _prep_lsqr
from SelfCal.subframe import _prep_subframe
from SelfCal.MapHelper import compute_crop
from SelfCal.SPHERExUtility import make_spherex_stripped_offset_map
from run_cal_baseline_test import prepare_detector_inputs, prepare_channel_inputs


RUN_DIR = '/mnt/md124/thomasli/selfcal/outputs/SPHEREx_nep_qr2_det3_6p2arcsec'
FRAME_SETTING = {'Detector': 3, 'NumSub': 10, 'NumCh': 34, 'NumCol': 3}
MOSAIC_OVERSAMPLE = 2


def get_ref_shape():
    from astropy.io import fits
    with fits.open(os.path.join(RUN_DIR, 'ref.fits')) as hdul:
        return hdul[0].data.shape if hdul[0].data is not None else (
            hdul[0].header['NAXIS2'], hdul[0].header['NAXIS1'])


def build_lsqr_common_params(file_list, ref_shape, det_inputs, ch_inputs):
    """Reproduce setup_lsqr's per-frame common_params for the K=1,
    use_per_frame_scalar=True, NumCol=3 config (no templates/groups/poly)."""
    chunk_maps = [det_inputs['det_chunk_map']]
    K = 1
    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    num_frames = len(file_list)

    det_chunk_map = det_inputs['det_chunk_map']
    num_chunks = int(det_chunk_map.max()) + 1
    num_chunks_list = [num_chunks]
    frame_to_group_list = [np.arange(num_frames)]
    num_offset_groups_list = [num_frames]
    det_template_list = [None]
    num_scalar_cols = num_frames  # use_per_frame_scalar=True

    col_bases = [num_sky, num_sky + num_chunks * num_frames]
    scalar_col_start = col_bases[K]

    adj_info = det_inputs['adj_info']
    # demote all-empty adjacency to None (matches setup_lsqr)
    if adj_info is not None and all(np.asarray(a).size == 0 for a in adj_info):
        adj_info = None

    return {
        'chunk_maps': chunk_maps,
        'grid_valid_weight': ch_inputs['det_valid_mask_padded'],
        'apply_mask': True,
        'apply_weight': False,
        'ignore_list': [],
        'oversample_factor': 1,
        'valid_threshold': 0.99,
        'outlier_thresh': 5.0,
        'num_chunks_list': num_chunks_list,
        'num_frames': num_frames,
        'ref_shape': ref_shape,
        'offset_regularization': True,
        'reg_weight_list': [0.1],
        'adj_info_list': [adj_info],
        'poly_constraint_list': [None],
        'postprocess_func': None,
        'preprocess_func': None,
        'frame_to_group_list': frame_to_group_list,
        'col_bases': col_bases,
        'scalar_col_start': scalar_col_start,
        'num_scalar_cols': num_scalar_cols,
        'det_template_list': det_template_list,
    }


def run_lsqr(file_list, ref_shape, det_inputs, ch_inputs):
    common = build_lsqr_common_params(file_list, ref_shape, det_inputs, ch_inputs)
    tasks = []
    for index, f in enumerate(file_list):
        tp = {'index': index, 'reproj_file': f}
        tp.update(common)
        tasks.append(tp)

    # warm-up (page cache + import-time lazy work) on first 5 frames, not profiled
    for tp in tasks[:5]:
        _prep_lsqr(tp)

    def work():
        n_rows = 0
        for tp in tasks:
            res = _prep_lsqr(tp)
            if res is not None:
                n_rows += res[4]
        return n_rows
    return work


def run_cache(file_list, ref_shape, det_inputs, ch_inputs, cache_dir):
    """Profile the cache-mode _prep_subframe interior (mosaic coadd cache path).
    No offsets applied (offset application cost is the spline func, profiled
    separately); this isolates load + interp + grid-weight det_to_sub + crop.

    WARNING: the per-frame nz_rows/nz_cols → bbox-crop logic below is
    duplicated from coadd._coadd_batch_worker (around lines 163-199 of
    SelfCal/coadd.py). If the production bbox-crop logic changes (e.g. a
    different valid-weight criterion, a different padding, or a fused
    crop+accumulate step), this profile will report stale numbers and
    misrepresent the cache phase's actual hot lines until kept in sync."""
    import h5py
    grid_chunk_map = det_inputs['grid_chunk_map']
    prep_config = {
        'chunk_maps': [grid_chunk_map],
        'apply_weight': False,
        'apply_mask': True,
        'ignore_list': [21],
        'grid_valid_weight': ch_inputs['grid_valid_weight'],
        'det_offset_funcs': [None],
        'oversample_factor': MOSAIC_OVERSAMPLE,
        'valid_threshold': 0.99,
        'for_lsqr': False,
        'preprocess_func': None,
        'postprocess_func': None,
    }
    os.makedirs(cache_dir, exist_ok=True)

    for f in file_list[:5]:
        _prep_subframe(file=f, chunk_offsets=None, det_aux=None, **prep_config)

    def work():
        for f in file_list:
            ref_coords, sub_data, sub_weight, _, _ = _prep_subframe(
                file=f, chunk_offsets=None, det_aux=None, **prep_config)
            nz_rows = np.any(sub_weight, axis=1)
            nz_cols = np.any(sub_weight, axis=0)
            ri = np.where(nz_rows)[0]
            ci = np.where(nz_cols)[0]
            if ri.size and ci.size:
                rmin, rmax = int(ri[0]), int(ri[-1]) + 1
                cmin, cmax = int(ci[0]), int(ci[-1]) + 1
            else:
                rmin = cmin = rmax = cmax = 0
            cache_path = os.path.join(cache_dir, f"cached_{os.path.basename(f)}")
            with h5py.File(cache_path, 'w') as hf:
                hf.create_dataset('ref_coords', data=np.array(
                    [ref_coords[0] + rmin, ref_coords[0] + rmax,
                     ref_coords[2] + cmin, ref_coords[2] + cmax]), track_times=False)
                hf.create_dataset('sub_data', data=sub_data[rmin:rmax, cmin:cmax], track_times=False)
                hf.create_dataset('sub_weight', data=sub_weight[rmin:rmax, cmin:cmax], track_times=False)
                hf.create_dataset('sub_bbox', data=np.array([rmin, rmax, cmin, cmax], dtype=np.int32), track_times=False)
    return work


def run_mean(cache_files, ref_shape):
    """Profile the mean-mode accumulation interior over cached crops.

    WARNING: this phase replays a simplified mean-accumulation loop. The
    production pattern in coadd._coadd_batch_worker (around lines 240-247 of
    SelfCal/coadd.py) accumulates into per-worker LOCAL arrays and flushes
    them to a SHARED-memory accumulator under a single per-batch lock. That
    pattern is NOT exercised here — this profile just adds straight into a
    single in-process numpy array. A future opt targeting the local→shared
    flush (lock contention, SHM bandwidth, batched-vs-streamed flush) will
    be invisible in this profile."""
    import h5py
    data_sum = np.zeros(ref_shape, dtype=np.float32)
    weight_sum = np.zeros(ref_shape, dtype=np.float32)

    def load(fp):
        with h5py.File(fp, 'r') as hf:
            return hf['ref_coords'][:], hf['sub_data'][:], hf['sub_weight'][:]

    for fp in cache_files[:5]:
        load(fp)

    def work():
        for fp in cache_files:
            ref_coords, sub_data, sub_weight = load(fp)
            sub_crop, ref_crop = compute_crop(ref_shape, ref_coords)
            data_crop = sub_data[sub_crop]
            weight_crop = sub_weight[sub_crop]
            data_sum[ref_crop] += data_crop * weight_crop
            weight_sum[ref_crop] += weight_crop
    return work


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phase', choices=['lsqr', 'cache', 'mean'], required=True)
    ap.add_argument('--n-frames', type=int, default=60)
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--label', default='')
    args = ap.parse_args()

    set_hdd_io_limit(None)
    ref_shape = get_ref_shape()
    print(f"ref_shape={ref_shape}")

    det_inputs = prepare_detector_inputs(FRAME_SETTING, MOSAIC_OVERSAMPLE)
    ch_inputs = prepare_channel_inputs(
        [17], FRAME_SETTING, det_inputs['det_chunk_map'], det_inputs['grid_chunk_map'])

    all_files = sorted(glob_module.glob(os.path.join(RUN_DIR, 'reprojected', '*.h5')))
    file_list = all_files[:args.n_frames]
    print(f"phase={args.phase}  n_frames={len(file_list)}")

    cache_dir = os.path.join(tempfile.gettempdir(), 'profile_interior_cache')

    if args.phase == 'lsqr':
        work = run_lsqr(file_list, ref_shape, det_inputs, ch_inputs)
    elif args.phase == 'cache':
        work = run_cache(file_list, ref_shape, det_inputs, ch_inputs, cache_dir)
    elif args.phase == 'mean':
        # build cache first (not profiled), then profile mean accumulation
        run_cache(file_list, ref_shape, det_inputs, ch_inputs, cache_dir)()
        cache_files = sorted(glob_module.glob(os.path.join(cache_dir, '*.h5')))
        print(f"cached {len(cache_files)} files")
        work = run_mean(cache_files, ref_shape)

    t0 = time.perf_counter()
    pr = cProfile.Profile()
    pr.enable()
    work()
    pr.disable()
    wall = time.perf_counter() - t0
    print(f"\n=== wall (profiled region): {wall:.3f} s for {len(file_list)} frames "
          f"({1000*wall/len(file_list):.1f} ms/frame) {args.label} ===\n")

    print("---- sorted by cumulative ----")
    st = pstats.Stats(pr)
    st.sort_stats('cumulative').print_stats(args.top)
    print("---- sorted by tottime ----")
    st.sort_stats('tottime').print_stats(args.top)


if __name__ == '__main__':
    main()
