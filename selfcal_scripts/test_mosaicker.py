#!/usr/bin/env python3
"""
Test script for SPHEREx mosaicking pipeline.
"""

import sys
import os
import importlib
import gc
from functools import partial

import h5py
import numpy as np
import matplotlib as mpl

# Add parent directory to path
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_path)

# Configure matplotlib
mpl.rcParams['figure.dpi'] = 400

# Import SPHEREx modules
from SelfCal import MapHelper, PipelineWrapper
from SelfCal.SPHERExUtility import make_spherex_offset_map, make_fiducial_chunk_map, make_fiducial_chunk_mask, \
load_calibration, make_spherex_offset_map

# Reload modules for development
importlib.reload(MapHelper)


def setup_config(detector):
    """Setup configuration parameters."""
    config = {
        'output_dir': '/mnt/md124/thomasli/selfcal/outputs/',
        'run_name': f'nep_det{detector}_3p1arcsec',
        'resolution_arcsec': 3.1
    }
    return config


def load_detector_data(detector):
    """Load detector calibration and chunk map data."""
    # Load calibration
    det_BC, det_BW = load_calibration(
        band=detector, 
        calibration_dir='/home/thomasli/spherex/SPHEREx_Spectral_Calibration'
    )
    
    # Create chunk map
    oversample_factor = 4
    chunk_map, lvf_params = make_fiducial_chunk_map(
        detector, det_BC, 
        num_subchannels=10, 
        num_channels=17, 
        oversample_factor=oversample_factor
    )
    
    return det_BC, det_BW, chunk_map, lvf_params


def setup_masks(channels, chunk_map):
    """Setup validation masks for chunks and detectors."""
    chunk_valid_mask = make_fiducial_chunk_mask(
        channels, 
        num_subchannels=10, 
        num_channels=17
    )
    det_valid_mask = chunk_valid_mask[chunk_map]
    
    return chunk_valid_mask, det_valid_mask

@profile
def main():
    """Main mosaicking workflow."""
    # Configuration
    detector = 4
    channels = [10]
    
    # Setup
    config = setup_config(detector)
    det_BC, det_BW, chunk_map, lvf_params = load_detector_data(detector)
    chunk_valid_mask, det_valid_mask = setup_masks(channels, chunk_map)
    
    # Initialize mosaicker
    mm = PipelineWrapper.Mosaicker(config)
    
    # Trim reproj_list for testing (optional)
    # mm.reproj_list = mm.reproj_list[0:1200]

    # Load calibration
    cal_path = f'/mnt/md124/thomasli/selfcal/outputs/nep_det{detector}_3p1arcsec/calibration/cal_det{detector}_ch{channels[0]}.h5'
    mm.load_calibration(cal_path=cal_path)
    # mm.O = mm.O[0:1200, :]
    
    # Setup offset function
    partial_make_offset_map = partial(
        make_spherex_offset_map, 
        chunk_valid_mask=chunk_valid_mask, 
        lvf_params=lvf_params
    )
    
    # Create mosaic
    maps = mm.make_mosaic(
        apply_mask=True,
        apply_weight=True,
        chunk_map=chunk_map,
        det_valid_mask=det_valid_mask,
        max_workers=40,
        make_std_map=True,
        apply_sigma_clipping=True,
        sigma=1.0,
        ignore_list=[21],
        oversample_factor=4,
        det_offset_func=partial_make_offset_map,
        batch_size=20
    )
    
    # Save results
    output_filename = f'mosaic_det{detector}_ch{"-".join(map(str, channels))}_narrow.fits'
    mm.save_mosaic(mos_file=output_filename, overwrite=True)


if __name__ == "__main__":
    main()