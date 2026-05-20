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
import numpy as np
from tqdm import tqdm
from threadpoolctl import threadpool_limits

from SelfCal import PipelineWrapper
from SelfCal.MakeMap import set_hdd_io_limit, compute_x0_from_Ab
from SelfCal.SPHERExUtility import (load_calibration, load_lvf_params, compute_column_adjacency,
                                    compute_column_polynomial_chains, compute_offsets_guess,
                                    make_stripped_chunk_map, make_stripped_chunk_valid_mask,
                                    fast_vertical_dist)
from SelfCal.solution import encode_x, compute_x0_scalar_only


def build_detector_stripe_map(shape, mid_width=64, edge_width=60, dtype=np.int32):
    """Detector-fixed chunk map: vertical stripes with narrower edge stripes.

    Lays out N stripes per row so that ``edge_width + (N - 2) * mid_width
    + edge_width == shape[1]``: the two outermost stripes are ``edge_width``
    px wide and the interior ``N - 2`` stripes are ``mid_width`` px wide.
    Default ``(60, 64)`` matches the layout used in earlier baseline tests
    on a 2040-wide detector → 32 stripes (60 + 30*64 + 60 = 2040).
    """
    h, w = shape
    if (w - 2 * edge_width) % mid_width != 0:
        raise ValueError(
            f"width {w} does not partition into 2 * {edge_width}-px edges + "
            f"k * {mid_width}-px middles")
    n_mid = (w - 2 * edge_width) // mid_width
    widths = [edge_width] + [mid_width] * n_mid + [edge_width]
    cols = np.repeat(np.arange(len(widths), dtype=dtype), widths)
    assert cols.shape[0] == w
    return np.broadcast_to(cols[None, :], (h, w)).copy()


def prepare_detector_inputs(frame_setting, mosaic_setting_oversample):
    detector = frame_setting['Detector']
    num_subchannels = frame_setting['NumSub']
    num_channels = frame_setting['NumCh']
    num_columns = frame_setting['NumCol']
    
    lvf_filename = f'lvf_params_D{detector}.npy'
    lvf_params = load_lvf_params(lvf_filename)

    det_BC, det_BW = load_calibration(band=detector, calibration_dir='/home/thomasli/spherex/SPHEREx_Spectral_Calibration')
    grid_chunk_map, _, _, _ = make_stripped_chunk_map(detector, num_subchannels=num_subchannels, num_channels=num_channels, num_columns=num_columns,
                                                    oversample_factor=mosaic_setting_oversample, lvf_params=lvf_params)
    det_chunk_map, _, r_edges, x_edges = make_stripped_chunk_map(detector, num_subchannels=num_subchannels, num_channels=num_channels, num_columns=num_columns,
                                            oversample_factor=1, lvf_params=lvf_params)
    
    adj_info = compute_column_adjacency(det_chunk_map, num_columns)
        
    return {
        'lvf_params': lvf_params,
        'det_BC': det_BC,
        'det_BW': det_BW,
        'grid_chunk_map': grid_chunk_map,
        'det_chunk_map': det_chunk_map,
        'r_edges': r_edges,
        'x_edges': x_edges,
        'adj_info': adj_info
    }


def prepare_channel_inputs(ch, frame_setting, det_chunk_map, grid_chunk_map):
    num_subchannels = frame_setting['NumSub']
    num_channels = frame_setting['NumCh']
    num_columns = frame_setting['NumCol']
    
    if isinstance(ch, list) or isinstance(ch, np.ndarray):
        chunk_valid_mask_padded = make_stripped_chunk_valid_mask(ch=ch, num_subchannels=num_subchannels, num_channels=num_channels, 
                                        num_columns=num_columns, subchannel_padding=1)
        chunk_valid_mask = make_stripped_chunk_valid_mask(ch=ch, num_subchannels=num_subchannels, num_channels=num_channels, 
                                        num_columns=num_columns, subchannel_padding=0)
    elif isinstance(ch, str):
        if ch == 'Aromatic':
            subch = np.arange(225, 236)
        elif ch == 'Aliphatic':
            subch = np.arange(249, 260)
        chunk_valid_mask_padded = make_stripped_chunk_valid_mask(subch=subch, num_subchannels=num_subchannels, num_channels=num_channels, 
                                        num_columns=num_columns, subchannel_padding=1)
        chunk_valid_mask = make_stripped_chunk_valid_mask(subch=subch, num_subchannels=num_subchannels, num_channels=num_channels, 
                                        num_columns=num_columns, subchannel_padding=0)

    # Pre-calculate weights safely
    det_valid_mask = chunk_valid_mask[det_chunk_map]
    det_valid_weight = fast_vertical_dist(det_valid_mask)
    if np.max(det_valid_weight) > 0:
        det_valid_weight /= np.max(det_valid_weight) 

    det_valid_mask_padded = chunk_valid_mask_padded[det_chunk_map]


    grid_valid_mask = chunk_valid_mask[grid_chunk_map]
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
        'grid_valid_weight': grid_valid_weight
    }

def mask_bright_pixels(local_vars):
    sub_data = local_vars['sub_data']
    sub_weight = local_vars['sub_weight']
    
    valid_mask = sub_weight > 0
    if np.sum(valid_mask) > 0:
        threshold = np.nanpercentile(sub_data[valid_mask], 25)
        sub_data[sub_data > threshold] = np.nan
        
    return sub_data

def build_variant_config(variant, det_chunk_map, adj_info, num_columns, num_frames):
    """Solver-config dispatch for the named test variants.

    All variants except ``oldx0_off`` enable a per-frame scalar
    (``use_per_frame_scalar=True``) and anchor each map's per-frame
    chunk-mean to 0 (``mean_offsets_list=[zeros, ...]``). This pushes per-frame
    DC into the explicit scalar column, so chunk offsets only carry
    within-frame structure — fixes the post-Apr-5 ``compute_x0_from_Ab``
    regression where low-coverage chunks were under-constrained on narrow
    channel masks.

    poly_off    : K=1, NumCol=3, no poly constraint.
    poly_k1     : K=1, NumCol=10, linear column-polynomial constraint.
    poly_k2     : K=2; same map 0 + constraint as poly_k1, plus a
                  detector-fixed 64-px-stripe map shared across frames.
    oldx0_off   : K=1, NumCol=3, no scalar — diagnostic that uses the
                  pre-Apr-5 ``compute_offsets_guess`` x0 init instead.
    scalar_off  : K=1, NumCol=3, alias of poly_off (kept for backwards
                  compatibility with prior naming in this branch).
    """
    if variant == 'oldx0_off':
        # Diagnostic: pre-Apr-5 path with no scalar.
        return {
            'chunk_maps': [det_chunk_map],
            'adj_infos': [adj_info],
            'reg_weights': [0.1],
            'poly_constraints_list': None,
            'mean_offsets_list': None,
            'det_groups_list': None,
        }
    if variant in ('poly_off', 'scalar_off'):
        return {
            'chunk_maps': [det_chunk_map],
            'adj_infos': [adj_info],
            'reg_weights': [0.1],
            'poly_constraints_list': None,
            'mean_offsets_list': [np.zeros(num_frames)],
            'det_groups_list': None,
            'use_per_frame_scalar': True,
        }
    if variant == 'poly_k1':
        chains, stencil = compute_column_polynomial_chains(
            det_chunk_map, num_columns=num_columns, degree=1)
        return {
            'chunk_maps': [det_chunk_map],
            'adj_infos': [adj_info],
            'reg_weights': [0.1],
            'poly_constraints_list': [[
                {'chains': chains, 'stencil': stencil, 'weight': 0.5},
            ]],
            'mean_offsets_list': [np.zeros(num_frames)],
            'det_groups_list': None,
            'use_per_frame_scalar': True,
        }
    if variant == 'poly_k2':
        chains, stencil = compute_column_polynomial_chains(
            det_chunk_map, num_columns=num_columns, degree=1)
        stripe_map = build_detector_stripe_map(
            det_chunk_map.shape, mid_width=64, edge_width=60,
            dtype=det_chunk_map.dtype)
        # map 0: per-frame, mean-anchored to 0. map 1: shared across frames,
        # mean-anchored to 0 (already present from prior K=2 config).
        # det_groups_list[1] = zeros triggers the scalar via the existing
        # path, but we also set use_per_frame_scalar=True for clarity.
        return {
            'chunk_maps': [det_chunk_map, stripe_map],
            'adj_infos': [adj_info, None],
            'reg_weights': [0.1, 0.0],
            'poly_constraints_list': [
                [{'chains': chains, 'stencil': stencil, 'weight': 0.5}],
                None,
            ],
            'mean_offsets_list': [np.zeros(num_frames), np.zeros(num_frames)],
            'det_groups_list': [None, np.zeros(num_frames, dtype=np.int64)],
            'use_per_frame_scalar': True,
        }
    raise ValueError(f"Unknown variant: {variant!r}")


VARIANT_NUMCOL = {'poly_off': 3, 'poly_k1': 10, 'poly_k2': 10, 'oldx0_off': 3, 'scalar_off': 3}


if __name__ == "__main__":
    # ----------------------------- Start of Settings -----------------------------
    base_frame_setting = {
        'Detector': 5,
        'NumSub': 10,
        'NumCh': 34,
        'NumCol': 3,
    }

    selfcal_config = PipelineWrapper.PipelineConfig(
        output_dir='/mnt/md124/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_nep_qr2_det{base_frame_setting["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2
    )

    # Shared kwargs across variants (per-variant ones live in build_variant_config).
    calibration_kwargs_shared = {
        'apply_mask': True,
        'apply_weight': False,
        'outlier_thresh': 5.0,
        'ignore_list': [],
        'batch_size': 20,
        'offset_regularization': True,
        'weighted_damping': True,
        'damp_weight': 0.1,
        'max_workers': 32,
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

    # Used only by prepare_detector_inputs to build the (unused) grid_chunk_map at the same
    # oversample factor as production. Kept for input-parity with the production script.
    mosaic_oversample_factor = 2

    CACHE_DIR = '/home/thomasli/selfcal-project/selfcal/cache/'

    # Variants run sequentially; each produces a distinctly-named cal_*.h5
    # (FILE_SUFFIX = f'_baseline_{variant}'). Add 'poly_off' to also rerun the
    # K=1, NumCol=3 regression baseline.
    TEST_VARIANTS = ['poly_off', 'poly_k1', 'poly_k2']

    # Cap reproj files for a quick plumbing check; set to None for full runs.
    NUM_FRAMES_LIMIT = None

    HDD_IO_LIMIT = 20
    chs = [[3]]
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

    # Iterate variants × channels. Per-variant prepare_detector_inputs is needed
    # because det_chunk_map / adj_info depend on NumCol.
    for variant in TEST_VARIANTS:
        frame_setting = dict(base_frame_setting, NumCol=VARIANT_NUMCOL[variant])
        frame_setting_str = '_'.join([f'{key}{value}' for key, value in frame_setting.items()])
        FILE_SUFFIX = f'_baseline_{variant}_fixed'

        print(f"\n{'=' * 70}\nVariant {variant} (NumCol={frame_setting['NumCol']})\n{'=' * 70}")
        detector_inputs = prepare_detector_inputs(frame_setting, mosaic_oversample_factor)

        for ch in chs:
            if isinstance(ch, list):
                job_name = f'Ch{"-".join(map(str, ch))}'
            else:
                job_name = ch
            t0 = time.time()
            print(f"Processing channel {job_name} for detector {frame_setting['Detector']}, "
                  f"variant {variant}...")

            job_tag = f'{frame_setting_str}_{job_name}{FILE_SUFFIX}'
            cal_file = f'cal_{job_tag}.h5'
            cal_path = os.path.join(selfcal_config.cal_dir, cal_file)

            channel_inputs = prepare_channel_inputs(
                ch, frame_setting, detector_inputs['det_chunk_map'], detector_inputs['grid_chunk_map'])

            cc = PipelineWrapper.Calibrator(selfcal_config, reproj_dir=nvme_reproj_dir)
            if NUM_FRAMES_LIMIT is not None:
                cc.reproj_list = cc.reproj_list[:NUM_FRAMES_LIMIT]
                print(f"NUM_FRAMES_LIMIT applied: using {len(cc.reproj_list)} frames")

            if os.path.exists(cal_path):
                print(f"Calibration file {cal_path} already exists. Skipping calibration.")
            else:
                num_frames = len(cc.reproj_list)
                variant_cfg = build_variant_config(
                    variant,
                    det_chunk_map=detector_inputs['det_chunk_map'],
                    adj_info=detector_inputs['adj_info'],
                    num_columns=frame_setting['NumCol'],
                    num_frames=num_frames,
                )
                print(f"K={len(variant_cfg['chunk_maps'])}, "
                      f"poly={'on' if variant_cfg['poly_constraints_list'] else 'off'}, "
                      f"map shapes={[m.shape for m in variant_cfg['chunk_maps']]}, "
                      f"num_chunks={[int(m.max())+1 for m in variant_cfg['chunk_maps']]}")

                cc.setup_lsqr(
                    chunk_maps=variant_cfg['chunk_maps'],
                    grid_valid_weight=channel_inputs['det_valid_mask_padded'],
                    oversample_factor=1,
                    adj_infos=variant_cfg['adj_infos'],
                    reg_weights=variant_cfg['reg_weights'],
                    poly_constraints_list=variant_cfg['poly_constraints_list'],
                    mean_offsets_list=variant_cfg['mean_offsets_list'],
                    det_groups_list=variant_cfg['det_groups_list'],
                    use_per_frame_scalar=variant_cfg.get('use_per_frame_scalar', False),
                    **calibration_kwargs_shared,
                )

                if variant == 'oldx0_off':
                    # Diagnostic: pre-Apr-5 x0 init (compute_offsets_guess reads
                    # reproj FITS files and returns per-frame per-chunk mean).
                    # Reference behavior for the original "good" production run.
                    print("Computing x0 from compute_offsets_guess (pre-Apr-5 path)...")
                    offset_guess = compute_offsets_guess(
                        reproj_list=cc.reproj_list,
                        det_chunk_map=variant_cfg['chunk_maps'][0],
                    )
                    skymap_guess = np.zeros(cc.ref_shape, dtype=np.float64)
                    x0 = encode_x(skymap_guess, offset_guess)
                    print(f"  x0: shape={x0.shape}, "
                          f"offset_guess nonzero frac={(offset_guess != 0).mean():.3f}")
                elif variant_cfg.get('use_per_frame_scalar', False):
                    # Per-frame scalar absorbs DC via compute_x0_scalar_only:
                    # scalar = diag-LS from A/b (≈ weighted mean of valid b
                    # per frame); chunks + sky start at 0. This is the fix
                    # for the post-Apr-5 narrow-channel regression.
                    print("Computing x0 from compute_x0_scalar_only...")
                    x0 = compute_x0_scalar_only(
                        cc.A, cc.b, cc.ref_shape,
                        scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
                    )
                    scalar_col_start = cc.col_bases[len(cc.chunk_maps)]
                    scalar_init = x0[scalar_col_start:]
                    print(f"  x0: shape={x0.shape}, "
                          f"scalar_init mean={scalar_init.mean():+.3e}, std={scalar_init.std():.3e}")
                else:
                    x0 = compute_x0_from_Ab(cc.A, cc.b, cc.ref_shape)

                cc.apply_lsqr(x0=x0, use_float32=True, n_threads=32, **lsqr_kwargs)
                # Save with original HDD paths so cal file remains valid after NVMe cleanup
                nvme_list = cc.reproj_list
                cc.reproj_list = [os.path.join(selfcal_config.reproj_dir, os.path.basename(f)) for f in nvme_list]
                cal_path = cc.save_calibration(cal_file=cal_file)
                cc.reproj_list = nvme_list

            del cc
            gc.collect()

            print(f"Finished {variant}/{job_name} in {time.time() - t0:.2f} seconds.")
            print("-" * 50 + "\n")

    # NVMe reproj cache intentionally NOT deleted: this test is run multiple times across
    # different variants (and possibly different frame_settings). Re-copying hundreds
    # of GB from HDD on each run is wasteful. Manually `rm -rf` when fully done testing.
    print(f"NVMe reproj cache preserved at: {nvme_reproj_dir}")
