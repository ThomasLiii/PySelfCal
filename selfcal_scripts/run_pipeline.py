import sys
import os
import yaml
import time
import argparse
import numpy as np
import gc
from functools import partial

# --- Ensure SelfCal is in path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(current_dir)
if parent_path not in sys.path:
    sys.path.append(parent_path)

from SelfCal import PipelineWrapper
from SelfCal.SPHERExUtility import (
    load_calibration, 
    make_spherex_offset_map, 
    compute_offsets_guess,
    load_lvf_params,
    compute_vertical_strip_adjacency,
    make_stripped_chunk_map,
    make_stripped_chunk_valid_mask
)
from SelfCal.SPHERExAppendWav import wav_coadd

def generate_settings_tag(config):
    """Generates a string tag based on critical algorithm settings."""
    lsqr = config['lsqr_settings']
    tag = (f"_reg{lsqr['reg_weight']}"
           f"_damp{lsqr['damp_weight']}"
           f"_wdamp{str(lsqr['weighted_damping'])[0]}") # T/F
    return tag

def get_masks_to_process(config):
    """
    Returns a list of (id_label, mask_array).
    Handles both standard channel lists and custom loaded masks.
    """
    masks_list = []
    
    # Mode A: Custom Masks from File
    if config['paths'].get('custom_masks_path'):
        print(f"Loading custom masks from {config['paths']['custom_masks_path']}")
        custom_masks = np.load(config['paths']['custom_masks_path'], allow_pickle=True)
        
        # Determine if it's a single mask or a list
        if isinstance(custom_masks, np.ndarray) and custom_masks.ndim == 1 and isinstance(custom_masks[0], (bool, np.bool_)):
             masks_list.append(("custom_0", custom_masks))
        else:
            for i, mask in enumerate(custom_masks):
                masks_list.append((f"custom_{i}", mask))
                
    # Mode B: Standard Channel List
    else:
        channels = config['target']['channels']
        num_sub = config['target']['num_subchannels']
        num_ch = config['target']['num_channels']
        num_col = config['target']['num_columns']
        
        for ch in channels:
            mask_padded = make_stripped_chunk_valid_mask(
                [ch], num_subchannels=num_sub, num_channels=num_ch, 
                num_columns=num_col, subchannel_padding=1
            )
            
            # For the actual mosaicking (strict mask, no padding)
            mask_strict = make_stripped_chunk_valid_mask(
                [ch], num_subchannels=num_sub, num_channels=num_ch, 
                num_columns=num_col, subchannel_padding=0
            )
            
            masks_list.append((f"Ch{ch}", {
                'padded': mask_padded,
                'strict': mask_strict
            }))
            
    return masks_list

def main(config_path):
    # 1. Load Configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup Paths
    output_dir = config['paths']['output_dir']
    run_name = config['paths']['run_name']
    full_output_dir = os.path.join(output_dir, run_name)
    cal_dir = os.path.join(full_output_dir, 'calibration')
    
    # Helper objects
    det = config['target']['detector']
    file_tag = generate_settings_tag(config)
    
    print(f"--- Initialization ---")
    print(f"Run Name: {run_name}")
    print(f"Settings Tag: {file_tag}")

    # 2. Initialize Static Data (Calibration & Chunk Maps)
    print("Loading spectral calibration...")
    det_BC, det_BW = load_calibration(band=det, calibration_dir=config['paths']['calibration_dir'])
    
    # Load LVF Params (Assuming file exists in output dir or specified location)
    lvf_path = f'lvf_params_D{det}.npy' # Adjust path as necessary
    if os.path.exists(lvf_path):
        lvf_params = load_lvf_params(lvf_path)
    else:
        print("Warning: LVF params not found locally, fitting might be required inside pipeline.")
        lvf_params = None

    print("Generating chunk maps...")
    # Generate Map
    grid_chunk_map, _, _, _ = make_stripped_chunk_map(
        det, 
        num_subchannels=config['target']['num_subchannels'],
        num_channels=config['target']['num_channels'],
        num_columns=config['target']['num_columns'],
        oversample_factor=config['mosaic_settings']['oversample_factor'],
        lvf_params=lvf_params
    )
    
    det_chunk_map, _, _, _ = make_stripped_chunk_map(
        det, 
        num_subchannels=config['target']['num_subchannels'],
        num_channels=config['target']['num_channels'],
        num_columns=config['target']['num_columns'],
        oversample_factor=1,
        lvf_params=lvf_params
    )

    # Pre-compute Adjacency
    print("Computing adjacency...")
    adj_info = compute_vertical_strip_adjacency(det_chunk_map, config['target']['num_columns'])

    # 3. Processing Loop
    work_items = get_masks_to_process(config)

    for label, masks in work_items:
        print(f"\n==========================================")
        print(f" Processing: {label}")
        print(f"==========================================")
        t_start = time.time()

        # Handle mask structure (Custom vs Standard)
        if isinstance(masks, dict):
             # Standard flow: separate padded mask for Cal, strict for Mosaic
            cal_valid_mask_full = masks['padded'] # Full 1D mask
            mos_valid_mask_full = masks['strict'] # Full 1D mask
            
            det_valid_mask_cal = cal_valid_mask_full[det_chunk_map]
            det_valid_mask_mos = mos_valid_mask_full[grid_chunk_map] # Map strict mask to grid
        else:
            # Custom flow: User provided one mask, use for both
            cal_valid_mask_full = masks
            mos_valid_mask_full = masks
            det_valid_mask_cal = masks[det_chunk_map]
            det_valid_mask_mos = masks[grid_chunk_map]

        # Define Filenames
        cal_filename = f"cal_D{det}_{label}{file_tag}.h5"
        cal_path_full = os.path.join(cal_dir, cal_filename)
        
        # ---------------------------
        # CALIBRATION PHASE
        # ---------------------------
        if config['execution']['run_calibration']:
            # Check if exists
            if os.path.exists(cal_path_full) and not config['execution']['overwrite_calibration']:
                print(f"[Calibrator] Found existing file: {cal_filename}. Skipping LSQR.")
            else:
                print(f"[Calibrator] Starting LSQR for {label}...")
                
                # Init Pipeline Wrapper
                # Note: Passing config dict directly as Wrapper expects
                cc = PipelineWrapper.Calibrator(config)
                
                lsqr_conf = config['lsqr_settings']
                
                cc.setup_lsqr(
                    apply_mask=True,
                    apply_weight=False, # As per your example
                    chunk_map=det_chunk_map,
                    det_valid_mask=det_valid_mask_cal,
                    max_workers=config['execution']['max_workers'],
                    outlier_thresh=lsqr_conf['outlier_thresh'],
                    ignore_list=[],
                    oversample_factor=1,
                    batch_size=lsqr_conf['batch_size'],
                    reg_weight=lsqr_conf['reg_weight'],
                    adj_info=adj_info,
                    weighted_damping=lsqr_conf['weighted_damping'],
                    damp_weight=lsqr_conf['damp_weight']
                )

                # Solve
                offset_guess = compute_offsets_guess(cc.reproj_list, det_chunk_map)
                # Ensure x0 size matches LSQR requirements (Sky + Offsets)
                # Note: internal implementation of PipelineWrapper handles the exact sizing logic usually
                # If PipelineWrapper expects explicit x0:
                # x0 = np.hstack([np.zeros(np.prod(cc.ref_shape)), offset_guess.flatten()]) 
                # cc.apply_lsqr(x0=x0, ...)
                
                cc.apply_lsqr(x0=None, atol=1e-06, btol=1e-06, damp=1e-3, iter_lim=100, precondition=False)
                
                cc.save_calibration(cal_file=cal_filename)
                
                del cc
                gc.collect()
        else:
            print("[Calibrator] Skipped (Config set to False).")

        # ---------------------------
        # MOSAICKING PHASE
        # ---------------------------
        if config['execution']['run_mosaic']:
            print(f"[Mosaicker] Generating mosaic for {label}...")
            
            # Ensure we have a calibration file to load
            if not os.path.exists(cal_path_full):
                print(f"Error: Calibration file {cal_path_full} not found. Cannot mosaic.")
                continue

            mm = PipelineWrapper.Mosaicker(config)
            mm.load_calibration(cal_path=cal_path_full)

            mos_conf = config['mosaic_settings']
            
            # Setup Offset Function
            # Using partial to inject specific masks
            partial_offset_func = partial(
                make_spherex_offset_map, 
                chunk_valid_mask=mos_valid_mask_full, # Pass the 1D mask array
                lvf_params=lvf_params
            )

            cache_dir_name = f"cache_D{det}_{label}{file_tag}"
            
            maps = mm.make_mosaic(
                apply_mask=True,
                apply_weight=False,
                chunk_map=grid_chunk_map, # Use grid chunk map for mosaicking
                det_valid_mask=det_valid_mask_mos,
                max_workers=config['execution']['max_workers'],
                make_std_map=True,
                apply_sigma_clipping=True,
                sigma=mos_conf['sigma_clip'],
                ignore_list=[21],
                oversample_factor=mos_conf['oversample_factor'],
                det_offset_func=partial_offset_func,
                cache_batch_size=lsqr_conf['batch_size'],
                coadd_batch_size=mos_conf['coadd_batch_size'],
                cache_dir=os.path.join(output_dir, 'cache', cache_dir_name),
                cache_intermediate=True
            )

            # Wavelength Coadd
            print("[Mosaicker] Running Wavelength Coadd...")
            wav_mean, wav_std = wav_coadd(
                det_BC, det_BW, 
                mean_map=maps['mean_map']['data'], 
                std_map=maps['std_map']['data'], 
                reproj_list=mm.reproj_list, 
                cache_list=mm.cached_list, 
                ref_shape=maps['mean_map']['data'].shape, 
                sigma=mos_conf['sigma_clip'], 
                batch_size=lsqr_conf['batch_size'], 
                max_workers=config['execution']['max_workers']
            )

            mm.append_maps({
                'wav_mean_map': {'data': wav_mean, 'unit': 'um'},
                'wav_std_map': {'data': wav_std, 'unit': 'um'}
            })

            mos_filename = f"mosaic_D{det}_{label}{file_tag}.fits"
            mm.save_mosaic(mos_file=mos_filename, overwrite=config['execution']['overwrite_mosaic'])
            
            # Cleanup
            del mm, maps
            gc.collect()
            
            # Remove Cache
            cache_path = os.path.join(output_dir, 'cache', cache_dir_name)
            if os.path.isdir(cache_path):
                try:
                    import shutil
                    shutil.rmtree(cache_path)
                except:
                    print(f"Warning: Could not remove cache {cache_path}")

        print(f"Finished {label} in {time.time() - t_start:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPHEREx SelfCal Runner")
    parser.add_argument("config", help="Path to the config.yaml file")
    args = parser.parse_args()
    
    main(args.config)