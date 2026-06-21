"""Shared input-prep helpers for the single-region run_cal driver family.

``prepare_detector_inputs`` and ``prepare_channel_inputs`` were copy-pasted
verbatim across run_cal.py and its experiment variants (run_cal_d5,
run_cal_damp0p5, run_cal_damp_offset, run_cal_pahfit). This module is the single
source of truth so those drivers shrink to config + the cal/mosaic loop.

These are pure input-prep helpers (LVF params, BC/BW calibration, detector/grid
chunk maps, column adjacency, per-channel valid masks + edge-distance weights) —
no behavior change versus the inlined copies. ``prepare_channel_inputs`` accepts
a list/ndarray of channel ids OR a str window tag (the union of the tags the
drivers used: 'Aromatic', 'Aliphatic', 'Aromatic_PAHfit'); each branch matches
its driver's original byte-for-byte (verified by output-equality).

NOTE: the K=2 readout driver and the tiled-NEP driver keep bespoke prepare_*
(different chunk-map geometry / a wider PAH window), so they do not use this.
"""
import numpy as np

from selfcal.instruments.spherex.spherex_utility import (load_calibration, load_lvf_params,
    compute_column_adjacency, make_stripped_chunk_map,
    make_stripped_chunk_valid_mask, fast_vertical_dist)

# SPHEREx per-band spectral calibration (BC/BW maps) location.
SPHEREX_CALIB_DIR = '/home/thomasli/spherex/SPHEREx_Spectral_Calibration'

# Named str channel windows -> inclusive-low / exclusive-high subchannel index
# range (np.arange(lo, hi)). Matches the drivers' inlined definitions exactly.
SUBCH_WINDOWS = {
    'Aromatic': (225, 236),
    'Aliphatic': (249, 260),
    'Aromatic_PAHfit': (210, 250),  # run_cal_pahfit's 40-subch window
}


def prepare_detector_inputs(frame_setting, mosaic_setting_oversample):
    detector = frame_setting['Detector']
    num_subchannels = frame_setting['NumSub']
    num_channels = frame_setting['NumCh']
    num_columns = frame_setting['NumCol']

    lvf_filename = f'lvf_params_D{detector}.npy'
    lvf_params = load_lvf_params(lvf_filename)

    det_BC, det_BW = load_calibration(band=detector, calibration_dir=SPHEREX_CALIB_DIR)
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
        'adj_info': adj_info,
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
        if ch not in SUBCH_WINDOWS:
            raise ValueError(f"Unknown channel tag {ch!r}")
        lo, hi = SUBCH_WINDOWS[ch]
        subch = np.arange(lo, hi)
        chunk_valid_mask_padded = make_stripped_chunk_valid_mask(subch=subch, num_subchannels=num_subchannels, num_channels=num_channels,
                                        num_columns=num_columns, subchannel_padding=1)
        chunk_valid_mask = make_stripped_chunk_valid_mask(subch=subch, num_subchannels=num_subchannels, num_channels=num_channels,
                                        num_columns=num_columns, subchannel_padding=0)
    else:
        raise ValueError(f"ch must be list/ndarray or str, got {type(ch).__name__}")

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
        'grid_valid_weight': grid_valid_weight,
    }


def mask_bright_pixels(local_vars):
    """Optional postprocess_func: NaN out pixels above the 25th percentile of the
    valid (weight>0) data in a subframe. Copied verbatim from the drivers."""
    sub_data = local_vars['sub_data']
    sub_weight = local_vars['sub_weight']

    valid_mask = sub_weight > 0
    if np.sum(valid_mask) > 0:
        threshold = np.nanpercentile(sub_data[valid_mask], 25)
        sub_data[sub_data > threshold] = np.nan

    return sub_data
