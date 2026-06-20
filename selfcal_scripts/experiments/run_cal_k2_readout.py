"""K=2 calibration test on Detector 5: fiducial subchannel offsets +
detector-fixed per-readout-channel offsets.

Map 0 — fiducial subchannel chunk map (NumCol=1, 342 chunks per frame),
        free across exposures.
Map 1 — readout-channel column chunk map (32 chunks: 2 truncated 60-px
        edges + 30 full 64-px middle channels, faithful to the original
        2048-pixel detector after symmetric 4-px edge trim, with the
        readout channels at 64-pixel pitch),
        detector-fixed (a single shared offset across all frames;
        det_groups_list[1] = zeros(num_frames)).

Identifiability: mean_offsets_list = [None, np.zeros(num_frames)] anchors
map 1's offsets to mean-zero per frame so the K-1=1 shift degeneracy
between map 0 and map 1 is broken (zero-point lives in map 0).

Mosaic step is the mean-map only — std-map / sigma-clip / wav coadd /
intermediate caching are all disabled to keep the run fast for testing.
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import shutil
import time
import gc
import glob as glob_module
from concurrent.futures import ThreadPoolExecutor
from functools import partial

import numpy as np
from tqdm import tqdm

from SelfCal import PipelineWrapper
from SelfCal.MakeMap import (set_hdd_io_limit, compute_x0_from_Ab,
                             OffsetModel, OffsetBlock)
from SelfCal.SPHERExUtility import (load_lvf_params, compute_subchannel_adjacency,
                                    make_stripped_chunk_map, make_stripped_chunk_valid_mask,
                                    make_spherex_stripped_offset_map, fast_vertical_dist)


def make_readout_chunk_map(det_shape=(2040, 2040), col_start=60, col_width=64):
    """Build a per-readout-channel chunk map at detector resolution.

    The 2040-pixel detector frame already has 4-pixel-wide edges trimmed
    off the original 2048-pixel detector. In the original 2048 frame the
    32 readout channels are 64 pixels wide each (2048 / 64 = 32); after
    the symmetric 4-pixel edge trim, in the 2040 frame:

      - chunk 0: pixels [0, col_start)              — leftmost readout channel,
                                                       60 px wide (truncated by trim)
      - chunks 1..n_full: pixels [col_start + 64(i-1), col_start + 64 i)
                                                    — full 64-px middle channels
      - chunk n_full+1: pixels [col_start + 64 n_full, W)
                                                    — rightmost readout channel,
                                                       60 px wide (truncated by trim)

    With the default ``col_start=64-4=60`` and ``col_width=64``, ``W=2040``
    yields 32 chunks total (2 × 60 + 30 × 64 = 2040), faithfully
    reproducing the original 32 readout channels of the 2048-pixel detector.

    Returns
    -------
    chunk_map : (H, W) int32
    n_chunks : int
        Total readout-channel count (32 with default args).
    """
    H, W = det_shape
    chunk_map = np.full(det_shape, -1, dtype=np.int32)
    # Chunk 0: leftmost (truncated) readout channel.
    chunk_map[:, :col_start] = 0
    # Middle full readout channels start at index 1.
    n_full = (W - col_start) // col_width
    for i in range(n_full):
        x0 = col_start + i * col_width
        x1 = x0 + col_width
        chunk_map[:, x0:x1] = i + 1
    # Rightmost (truncated) readout channel.
    right_start = col_start + n_full * col_width
    n_chunks = n_full + 1
    if right_start < W:
        chunk_map[:, right_start:] = n_chunks
        n_chunks += 1
    assert (chunk_map >= 0).all(), "every pixel must be assigned a readout channel"
    return chunk_map, n_chunks


def upsample_chunk_map(det_chunk_map, factor):
    """Replicate each detector pixel into a (factor × factor) block on the
    oversampled mosaic grid, preserving chunk ids exactly."""
    if factor == 1:
        return det_chunk_map
    return np.kron(det_chunk_map, np.ones((factor, factor), dtype=det_chunk_map.dtype))


def prepare_detector_inputs(frame_setting, mosaic_setting_oversample):
    detector = frame_setting['Detector']
    num_subchannels = frame_setting['NumSub']
    num_channels = frame_setting['NumCh']
    num_columns = frame_setting['NumCol']

    lvf_filename = f'lvf_params_D{detector}.npy'
    lvf_params = load_lvf_params(lvf_filename)

    # --- Map 0 (fiducial subchannel) ---
    grid_chunk_map_sub, _, _, _ = make_stripped_chunk_map(
        detector, num_subchannels=num_subchannels, num_channels=num_channels,
        num_columns=num_columns, oversample_factor=mosaic_setting_oversample,
        lvf_params=lvf_params)
    det_chunk_map_sub, _, r_edges, x_edges = make_stripped_chunk_map(
        detector, num_subchannels=num_subchannels, num_channels=num_channels,
        num_columns=num_columns, oversample_factor=1, lvf_params=lvf_params)
    # NumCol=1 ⇒ compute_column_adjacency would produce zero pairs; vertical
    # subchannel adjacency is the natural smoothness prior in this case.
    adj_info_sub = compute_subchannel_adjacency(det_chunk_map_sub, num_columns)

    # --- Map 1 (readout-channel) ---
    det_chunk_map_ro, n_readout = make_readout_chunk_map(det_chunk_map_sub.shape)
    grid_chunk_map_ro = upsample_chunk_map(det_chunk_map_ro, mosaic_setting_oversample)

    return {
        'lvf_params': lvf_params,
        'grid_chunk_map_sub': grid_chunk_map_sub,
        'det_chunk_map_sub': det_chunk_map_sub,
        'r_edges': r_edges,
        'x_edges': x_edges,
        'adj_info_sub': adj_info_sub,
        'det_chunk_map_ro': det_chunk_map_ro,
        'grid_chunk_map_ro': grid_chunk_map_ro,
        'n_readout': n_readout,  # 64 readout channels (2 × 28-px edges + 62 × 32-px center)
    }


def prepare_channel_inputs(ch, frame_setting, det_chunk_map_sub, grid_chunk_map_sub):
    num_subchannels = frame_setting['NumSub']
    num_channels = frame_setting['NumCh']
    num_columns = frame_setting['NumCol']

    if isinstance(ch, list) or isinstance(ch, np.ndarray):
        chunk_valid_mask_padded = make_stripped_chunk_valid_mask(
            ch=ch, num_subchannels=num_subchannels, num_channels=num_channels,
            num_columns=num_columns, subchannel_padding=1)
        chunk_valid_mask = make_stripped_chunk_valid_mask(
            ch=ch, num_subchannels=num_subchannels, num_channels=num_channels,
            num_columns=num_columns, subchannel_padding=0)
    else:
        raise ValueError("Pass channels as a list of ints (e.g. [3]).")

    det_valid_mask = chunk_valid_mask[det_chunk_map_sub]
    det_valid_weight = fast_vertical_dist(det_valid_mask)
    if np.max(det_valid_weight) > 0:
        det_valid_weight /= np.max(det_valid_weight)

    det_valid_mask_padded = chunk_valid_mask_padded[det_chunk_map_sub]

    grid_valid_mask = chunk_valid_mask[grid_chunk_map_sub]
    grid_valid_weight = fast_vertical_dist(grid_valid_mask)
    if np.max(grid_valid_weight) > 0:
        grid_valid_weight /= np.max(grid_valid_weight)

    return {
        'chunk_valid_mask_padded': chunk_valid_mask_padded,
        'chunk_valid_mask': chunk_valid_mask,
        'det_valid_mask': det_valid_mask,
        'grid_valid_mask': grid_valid_mask,
        'det_valid_mask_padded': det_valid_mask_padded,
        'det_valid_weight': det_valid_weight,
        'grid_valid_weight': grid_valid_weight,
    }


if __name__ == "__main__":
    # ----------------------------- Settings -----------------------------
    frame_setting = {
        'Detector': 5,
        'NumSub': 10,
        'NumCh': 34,
        'NumCol': 1,  # map 0 is the fiducial NumCol=1 subchannel chunk map
    }

    selfcal_config = PipelineWrapper.PipelineConfig(
        output_dir='/mnt/md124/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_nep_qr2_det{frame_setting["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2,
    )

    # Map 0 gets light subchannel-adjacency reg; map 1 (readout) gets none
    # per the test spec. The per-block offset config (reg_weight, adjacency,
    # det_groups, mean-anchor) now lives on the OffsetModel built below; only
    # global solver settings stay in calibration_kwargs.
    calibration_kwargs = {
        'apply_mask': True,
        'apply_weight': False,
        'outlier_thresh': 5.0,
        'ignore_list': [],
        'batch_size': 20,
        'offset_regularization': True,
        'weighted_damping': True,
        'damp_weight': 0.1,
        'max_workers': 32,
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

    # mean-map only, no std/sigma-clip/wav, no intermediate cache
    mosaic_kwargs = {
        'apply_mask': True,
        'apply_weight': False,
        'make_std_map': False,
        'apply_sigma_clipping': False,
        'sigma': 2.0,
        'ignore_list': [21],
        'cache_batch_size': 20,
        'coadd_batch_size': 30,
        'cache_intermediate': False,
        'max_workers': 32,
    }

    mosaic_oversample_factor = 2

    CACHE_DIR = '/home/thomasli/selfcal-project/selfcal/cache/'
    FILE_SUFFIX = '_k2_readout_test'

    chs = [[3]]
    HDD_IO_LIMIT = 20
    # -------------------------- End of settings --------------------------

    set_hdd_io_limit(HDD_IO_LIMIT)

    # NVMe staging (per-run subdirectory; reused across the channel loop)
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
    set_hdd_io_limit(None)

    def remap_to_nvme(file_list):
        return [os.path.join(nvme_reproj_dir, os.path.basename(f)) for f in file_list]

    frame_setting_str = '_'.join([f'{key}{value}' for key, value in frame_setting.items()])

    detector_inputs = prepare_detector_inputs(frame_setting, mosaic_oversample_factor)
    print(f"\nMap 0 (subchannel): shape={detector_inputs['det_chunk_map_sub'].shape}, "
          f"n_chunks={int(detector_inputs['det_chunk_map_sub'].max()) + 1}")
    print(f"Map 1 (readout):    shape={detector_inputs['det_chunk_map_ro'].shape}, "
          f"n_readout={detector_inputs['n_readout']} (chunk 0 + chunk {detector_inputs['n_readout']-1}: "
          f"truncated 60-px edges; chunks 1..{detector_inputs['n_readout']-2}: 64-px middle channels)")

    for ch in chs:
        job_name = f'Ch{"-".join(map(str, ch))}'
        t0 = time.time()
        print(f"\nProcessing channel {job_name} for detector {frame_setting['Detector']}...")

        job_tag = f'{frame_setting_str}_{job_name}{FILE_SUFFIX}'
        cal_file = f'cal_{job_tag}.h5'
        mos_file = f'mosaic_{job_tag}.fits'

        channel_inputs = prepare_channel_inputs(
            ch, frame_setting,
            detector_inputs['det_chunk_map_sub'],
            detector_inputs['grid_chunk_map_sub'])

        # --------------------- Calibration ---------------------
        cal_path = os.path.join(selfcal_config.cal_dir, cal_file)
        cc = PipelineWrapper.Calibrator(selfcal_config, reproj_dir=nvme_reproj_dir)
        if os.path.exists(cal_path):
            print(f"Calibration file {cal_path} already exists. Skipping calibration.")
        else:
            num_frames = len(cc.reproj_list)
            # K=2 offset model, read as two cohesive blocks:
            #   block 0 (subchannel) — free offset per frame, light subchannel
            #     adjacency reg (reg_weight 0.1), no mean anchor.
            #   block 1 (readout)    — detector-fixed (det_groups all 0 → one
            #     shared per-readout-channel offset across the exposure list),
            #     anchored mean-zero per frame, no adjacency reg.
            # Lowers to the exact parallel-list kwargs the flat call used.
            offset_model = OffsetModel([
                OffsetBlock(chunk_map=detector_inputs['det_chunk_map_sub'],
                            adj_info=detector_inputs['adj_info_sub'],
                            reg_weight=0.1),
                OffsetBlock(chunk_map=detector_inputs['det_chunk_map_ro'],
                            det_groups=np.zeros(num_frames, dtype=int),
                            mean_offset=np.zeros(num_frames),
                            reg_weight=0.0),
            ])
            cc.setup_lsqr(
                offset_model=offset_model,
                grid_valid_weight=channel_inputs['det_valid_mask_padded'],
                oversample_factor=1,
                **calibration_kwargs,
            )

            x0 = compute_x0_from_Ab(cc.A, cc.b, cc.ref_shape,
                                    active_mask=cc.active_mask)
            cc.apply_lsqr(x0=x0, use_float32=True, n_threads=32, **lsqr_kwargs)

            # Save with HDD paths so cal file remains valid after NVMe cleanup
            nvme_list = cc.reproj_list
            cc.reproj_list = [os.path.join(selfcal_config.reproj_dir, os.path.basename(f))
                              for f in nvme_list]
            cal_path = cc.save_calibration(cal_file=cal_file)
            cc.reproj_list = nvme_list

        # --------------------- Mosaic (mean only) ---------------------
        # Map 0 gets the smooth subchannel-arc interp; map 1 uses the
        # default piecewise-constant chunk_to_det (no smoothing).
        partial_make_offset_map_sub = partial(
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
            chunk_maps=[
                detector_inputs['grid_chunk_map_sub'],
                detector_inputs['grid_chunk_map_ro'],
            ],
            grid_valid_weight=channel_inputs['grid_valid_weight'],
            oversample_factor=mosaic_oversample_factor,
            det_offset_funcs=[partial_make_offset_map_sub, None],
            cache_dir=os.path.join(CACHE_DIR, f'cache_{job_tag}'),
            **mosaic_kwargs,
        )

        mm.save_mosaic(mos_file=mos_file, overwrite=True)

        del cc, mm, maps
        gc.collect()
        print(f"Finished channel {job_name} in {time.time() - t0:.2f} seconds.")
        print("-" * 50)

    print(f"\nNVMe reproj cache preserved at {nvme_reproj_dir} for re-runs. "
          f"Manually `rm -rf` when done testing.")
