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

from SelfCal import PipelineWrapper
from SelfCal.MakeMap import (set_hdd_io_limit, compute_x0_from_Ab,
                             OffsetModel, OffsetBlock, SkyModel)
from SelfCal.solution import compute_x0_scalar_only
from SelfCal.SPHERExUtility import load_calibration, load_lvf_params, compute_column_adjacency, \
make_stripped_chunk_map, make_stripped_chunk_valid_mask, make_spherex_stripped_offset_map, fast_vertical_dist, \
compute_column_polynomial_chains
from SelfCal.SPHERExAppendWav import wav_coadd


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
        elif ch == 'Aromatic_PAHfit':
            # 40-subchannel range centered on PAH 3.29 μm (subch 210-250,
            # symmetric ±20 from peak at ~subch 230) for the in-LSQR per-pixel
            # continuum + line amplitude joint fit. ±2σ around line, G(λ_edge)
            # ≈ 0.135 — far-edge subch contribute negligibly to line Fisher
            # info (G² ≈ 0.02) so dropping the outer 10 subch from each side
            # cuts matrix memory by ~33% with minimal information loss.
            # The original 60-subch window (200-260) OOM'd at production scale
            # (17.6k frames × 60 subch × ~300k entries → 5B nonzeros, peaked
            # ~700 GB during the apply_lsqr A^T transpose allocation).
            subch = np.arange(210, 250)
        else:
            raise ValueError(f"Unknown channel tag {ch!r}")
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

if __name__ == "__main__":
    # ----------------------------- Start of Settings -----------------------------
    frame_setting = {
        'Detector': 4,
        'NumSub': 10,
        'NumCh': 34,
        'NumCol': 5,
    }

    selfcal_config = PipelineWrapper.PipelineConfig(
        output_dir='/data3/thomasli/selfcal/outputs/',
        run_name=f'SPHEREx_NEP_2026W17_D{frame_setting["Detector"]}_6p2arcsec',
        resolution_arcsec=6.2
    )

    calibration_kwargs = {
        'apply_mask': True,
        'apply_weight': True,  # Poisson inverse-variance weighting (production fix). Off-peak PAH pixels get less weight, breaking sky→offset leakage.
        'outlier_thresh': 5.0,
        # drop source-mask bit (match mosaic); sources participate in LSQR for source-included spectral mosaic.
        # Verified on cirrus 1k: doubles covered pixels, raises corr(sky_total,baseline) 0.58->0.91, sky_line uncontaminated.
        # See memory project_pahfit_ignore_list_21.md.
        'ignore_list': [21],
        'batch_size': 50,
        'offset_regularization': True,
        # reg_weights moved onto the OffsetBlock built below (per-block config).
        'weighted_damping': True,
        'damp_weight': 0.1,
        # --- Spectral-fit (PAHfit) mode params: 2-block sky (continuum + line amplitude) ---
        # det_aux + spectral_fit are wired below into the cc.setup_lsqr call,
        # not in this dict (det_aux depends on detector_inputs which isn't
        # available at module load time).
        'damp_weight_line': 0.0,  # No Tikhonov damping on line block (production choice from cirrus 1k A1/A2/A3 sweep: A3 dampL0 extracts the most diffuse PAH signal without biasing toward zero; higher per-pixel coverage in production vs sanity should suppress the low-coverage edge blowup seen in the 1k sweep).
        # damp_offset reverted to 0 — hybrid test (apply_weight + damp_offset=0.1) was WORSE than applyWt alone (cirrus dark_spread widened further, dark-ring re-emerged). The two levers aren't orthogonal — both reweight toward bright/well-covered chunks. See aromatic-map-tuning session.
        'max_workers': 48,
        'postprocess_func': None, #mask_bright_pixels,
    }

    lsqr_kwargs = {
        'atol': 1e-06,
        'btol': 1e-06,
        'damp': 0,
        # 100 iters gives pearson r > 0.999 between consecutive cals at Fisher>=10 (per staircase convergence study); bulk PAH signal converged
        'iter_lim': 100,
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
    FILE_SUFFIX = f'_damp0p1_reg0p1_applyWt_PAHfit_dampL0_subch40_nosrcmask_NumCol5_iter100_outThresh5_sigma2_polyK1'

    # Linear column constraint weight (compute_column_polynomial_chains, degree=1)
    POLY_DEGREE = 1
    POLY_WEIGHT = 0.5

    # Channels to process — per-pixel PAH 3.29 μm continuum + line amplitude
    # joint solve, 60-subch window (200-260) for well-posed cont/line decoupling.
    chs = ['Aromatic_PAHfit']
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
            poly_group = [{
                'chains': poly_chains,
                'stencil': poly_stencil,
                'weight': POLY_WEIGHT,
            }]
            # Single offset block: per-frame scalar absorbs DC; mean-anchor on
            # the map-0 chunks forces within-frame structure only, with a linear
            # column poly-constraint + adjacency reg. Lowers to the exact
            # parallel-list kwargs the flat call used.
            offset_model = OffsetModel([
                OffsetBlock(chunk_map=detector_inputs['det_chunk_map'],
                            adj_info=detector_inputs['adj_info'],
                            reg_weight=0.1,
                            poly_constraints=poly_group,
                            mean_offset=np.zeros(num_frames_run)),
            ], use_per_frame_scalar=True)
            cc.setup_lsqr(
                offset_model=offset_model,
                grid_valid_weight=channel_inputs['det_valid_mask_padded'],
                oversample_factor=1,
                # --- Spectral-fit (PAHfit) mode ---
                # 2-block sky: x = [sky_cont | sky_line | offsets | scalar].
                # det_aux = [BC_map, BW_map] gives per-(frame, sub-pixel)
                # wavelength and per-pixel σ; the line-amp column coefficient is
                # the Gaussian profile G(λ_i) per row. continuum_plus_pah_gaussian()
                # is byte-identical to the legacy spectral_fit=True path.
                sky_model=SkyModel.continuum_plus_pah_gaussian(),
                det_aux=[detector_inputs['det_BC'], detector_inputs['det_BW']],
                **calibration_kwargs
            )

            x0 = compute_x0_scalar_only(
                cc.A, cc.b, cc.ref_shape,
                scalar_col_start=cc.col_bases[len(cc.chunk_maps)],
                num_sky_blocks=cc.num_sky_blocks,
                active_mask=cc.active_mask,
            )

            cc.apply_lsqr(x0=x0, use_float32=True, n_threads=48, **lsqr_kwargs)
            # Phase 6 recommended Fisher mask threshold — saved as cal attr
            # only; not destructively applied. Use
            # ``SelfCal.MakeMap.apply_line_fisher_mask`` at analysis time to
            # apply the mask. Empirically determined from cirrus 1k
            # A3-with-Fisher sanity: Fisher distribution is bimodal — a noise
            # tail at Fisher<10 (low-coverage / off-peak-dominated pixels that
            # blow up without damping) and a main peak around Fisher~100-500
            # (well-constrained). Threshold=10 sits in the gap, masks ~6.5% of
            # covered pixels at sanity scale, recovers corr(line, baseline)
            # =+0.224 (vs +0.056 unmasked). Production has ~17x more frames so
            # most pixels will be well above threshold.
            cc.line_fisher_threshold = 10.0
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
