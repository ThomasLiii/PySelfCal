"""Batch reprojection of detector frames onto a common reference WCS."""

import os
import h5py
import hdf5plugin
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool

from astropy.io import fits
from astropy.wcs import WCS
from astropy.wcs.utils import proj_plane_pixel_scales

from reproject import reproject_interp, reproject_exact, reproject_adaptive

from .MapHelper import bit_to_bool, bool_to_bit


def _reproject_worker(task_params):
    """Individual tasks called by batch_reproject's multiprocessing instances"""
    # Unpacked arguments for clarity
    reproj_func = task_params['reproj_func']
    file_path = task_params['file_path']
    exp_idx = task_params['exp_idx']
    det_idx = task_params['det_idx']
    sci_ext = task_params['sci_ext']
    dq_ext = task_params['dq_ext']
    ref_wcs = task_params['ref_wcs']
    sub_width = task_params['sub_width']
    output_dir = task_params['output_dir']
    replace_existing = task_params['replace_existing']
    reproject_kwargs = task_params['reproject_kwargs']

    # Save to HDF5
    output_file = os.path.join(output_dir, f'exp_{exp_idx:04d}_det_{det_idx:02d}.h5')
    if not replace_existing and os.path.exists(output_file):
        return output_file # Skip if file already exists and replace_existing is False

    reproj_func_dict = {'exact': reproject_exact, 'interp': reproject_interp, 'adaptive': reproject_adaptive}

    try:
        with fits.open(file_path) as hdul:
            det_data = hdul[sci_ext].data
            det_header = hdul[sci_ext].header
            det_bitmask = hdul[dq_ext].data
        det_width = np.shape(det_data)[-1]
        det_header_str = det_header.tostring().encode('utf-8')
        det_wcs = WCS(det_header)

        # Map detector center to world, then to reference frame pixels
        det_center = [det_data.shape[0] / 2.0, det_data.shape[1] / 2.0]
        skycoord = det_wcs.pixel_to_world(det_center[1], det_center[0])
        ref_det_center = np.array(ref_wcs.world_to_pixel(skycoord))

        # Define sub-frame boundaries in the reference frame
        ref_x_min = int(ref_det_center[0] - sub_width // 2)
        ref_x_max = ref_x_min + sub_width
        ref_y_min = int(ref_det_center[1] - sub_width // 2)
        ref_y_max = ref_y_min + sub_width

        # Create WCS for the sub-frame
        sub_wcs = ref_wcs.deepcopy()
        sub_wcs.wcs.crpix[0] -= ref_x_min # Adjust CRPIX for the sub-frame origin
        sub_wcs.wcs.crpix[1] -= ref_y_min
        sub_header_str = sub_wcs.to_header().tostring().encode('utf-8')

        # Perform reprojection
        sub_data, sub_foot = reproj_func_dict[reproj_func](
            (det_data, det_wcs),
            sub_wcs,
            shape_out=(sub_width, sub_width),
            **reproject_kwargs
        )

        # Process detector auxiliary data
        det_expanded_mask = bit_to_bool(det_bitmask, expand_bits=True)
        det_xmesh, det_ymesh = np.meshgrid(np.arange(det_width), np.arange(det_width))
        det_aux = np.stack((det_xmesh, det_ymesh, *det_expanded_mask), axis=0)
        sub_aux, _ = reproject_interp(
            (det_aux, det_wcs),
            sub_wcs,
            shape_out=(sub_width, sub_width),
            order='bilinear',
        )
        sub_mapping = sub_aux[0:2] # x, y
        sub_expanded_mask_float = sub_aux[2:]
        sub_expanded_mask_bool = sub_expanded_mask_float > 0.01
        sub_bitmask = bool_to_bit(sub_expanded_mask_bool)

        with h5py.File(output_file, 'w', libver='latest') as hf:

            # CONFIG: Zstd + Shuffle
            # Zstd creates smaller files than Gzip, relieving your I/O bottleneck.
            # **hdf5plugin.Zstd() automatically handles the filter setup.
            comp_args = {
                **hdf5plugin.Zstd(clevel=5),
                'shuffle': True,
                'track_times': False
            }

            # 1. Save Data
            hf.create_dataset('sub_data', data=sub_data, dtype=np.float32, chunks=sub_data.shape, **comp_args)
            hf.create_dataset('sub_foot', data=sub_foot, dtype=np.float16, chunks=sub_foot.shape, **comp_args)
            hf.create_dataset('sub_bitmask', data=sub_bitmask, dtype=np.int32, chunks=sub_bitmask.shape, **comp_args)
            hf.create_dataset('sub_mapping', data=sub_mapping, dtype=np.float32, chunks=sub_mapping.shape, **comp_args)

            # 2. Save Metadata as Attributes
            hf.attrs['sub_header'] = sub_header_str
            hf.attrs['det_header'] = det_header_str
            hf.attrs['file_path'] = file_path
            hf.attrs['ref_coords'] = np.array([ref_y_min, ref_y_max, ref_x_min, ref_x_max], dtype=np.int32)

        return output_file # Return path on success
    except Exception as e:
        print(f'Error processing detector {det_idx} from exposure file index {exp_idx} ({file_path}): {e}')
        # import traceback; traceback.print_exc() # Uncomment for detailed debugging
        return None # Return None on failure

def batch_reproject(exposure_list, ref_wcs, ref_shape,
                    output_dir='output/', padding_percentage=0.05, num_processes=1,
                    sci_ext_list=[], dq_ext_list=[], reproj_func='interp', exp_idx_list=None, det_idx_list=None,
                    replace_existing=False, reproject_kwargs={}):
    """Reproject individual exposures to bounding boxes in reference frame, output sored in HDF5 files.

    Parameters
    ----------
    exposure_list : list or tup
        List of strings describing path to fits files containing the exposures
    ref : object
        An astropy.wcs.WCS object that defines the WCS of the reference (mosaic) frame
    ref_shape : list or tup
        List of shape (2, ) defining the size of the mosaic
    output_dir : str, optional
        Directory for all selfcal outputs
    padding_percentage: float, optional
        Fraction of the mosaic width to pad
    num_processes : int, optional
        Number of parallel processes to use
    ignore_flags : list or tup
        List of strings describing the header keywords in data quality extension, flags corresponding listed will be ignored
    sci_ext_list : list or tup
        List of integers defining the extension in the fits files containing the science data
    dq_ext_list : list or tup
        List of integers defining the extension in the fits files containing the data quality bitmask
    reproj_func : str
        Reproject function for reprojecting the science extensions
        - 'Exact': Slowest, conserves flux
        - 'Interp': Fastest, alter PSF profile, does not conserves flux
        - 'Adaptive': Faster then 'Exact', conserves flux
    exp_idx_list : list or tup, optional
        List of integers defining the exposure index in the exposure_list, if None, will use the index of the exposure_list
    det_idx_list : list or tup, optional
        List of integers defining the detector index in the exposure_list, if None, will use the index of the in fits file
    replace_existing : bool, optional
        If True, will overwrite existing files in the output directory, default is False

    Returns
    -------
    success_file : list
        List of path to the HDF5 files containing the reprojected data
    """

    assert isinstance(exposure_list, (list, tuple)) and exposure_list, "exposure_list must be a non-empty list or tuple"
    assert isinstance(ref_wcs, WCS), "Reference WCS must be an astropy.wcs.WCS object"
    assert isinstance(ref_shape, (list, tuple, np.ndarray)) and len(ref_shape) == 2, "ref_shape must be a list or tuple of length 2"
    assert isinstance(padding_percentage, float) and 0 <= padding_percentage, "padding_percentage must be a float larger than 0"
    assert isinstance(num_processes, int) and num_processes > 0, "num_processes must be a positive integer"
    assert sci_ext_list is None or isinstance(sci_ext_list, (list, tuple, np.ndarray)), "sci_ext_list must be a list or tuple"
    assert dq_ext_list is None or isinstance(dq_ext_list, (list, tuple, np.ndarray)), "dq_ext_list must be a list or tuple"
    assert reproj_func in ['exact', 'interp', 'adaptive'], "reproj_func must be one of 'exact', 'interp', or 'adaptive'"
    assert exp_idx_list is None or isinstance(exp_idx_list, (list, tuple, np.ndarray)), "exp_idx_list must be a list or tuple"
    assert det_idx_list is None or isinstance(det_idx_list, (list, tuple, np.ndarray)), "det_idx_list must be a list or tuple"
    assert type(replace_existing) is bool, "replace_existing must be a boolean"

    os.makedirs(output_dir, exist_ok=True)
    print(f'Starting batch reprojection. Output will be saved to: {output_dir}')
    # Determine sub-frame width based on a sample detector frame
    try:
        with fits.open(exposure_list[0]) as hdul_sample:
            # Assuming first science extension is representative
            sci_ext_0 = sci_ext_list[0] if len(sci_ext_list) > 0 else 1
            if sci_ext_0 >= len(hdul_sample):
                raise ValueError(f'Sample FITS {exposure_list[0]} does not have extension {sci_ext_0}')
            det_data_0 = hdul_sample[sci_ext_0].data
            det_wcs_0 = WCS(hdul_sample[sci_ext_0].header)
    except Exception as e:
        raise ValueError(f'Could not read sample FITS file {exposure_list[0]} to determine detector properties: {e}')

    ref_reso = np.abs(proj_plane_pixel_scales(ref_wcs)[0]) # Assuming square pixels
    det_reso = np.abs(proj_plane_pixel_scales(det_wcs_0)[0])

    reso_ratio = ref_reso / det_reso
    # Calculate sub_width needed to contain the diagonal of the detector frame after reprojection, plus padding
    sub_width = int(np.ceil(np.sqrt(2) * np.max(det_data_0.shape) / reso_ratio * (1 + 2 * padding_percentage)))

    tasks = []
    for i, file_path in enumerate(exposure_list): # file_idx is the overall exposure index
        for j, (sci_ext, dq_ext) in enumerate(zip(sci_ext_list, dq_ext_list)):
            exp_idx = exp_idx_list[i] if exp_idx_list is not None else i
            det_idx = det_idx_list[j] if det_idx_list is not None else j

            # Create a dictionary of parameters for each task
            task_params = {
                'reproj_func': reproj_func,
                'file_path': file_path,
                'exp_idx': exp_idx,
                'det_idx': det_idx,
                'sci_ext': sci_ext,
                'dq_ext': dq_ext,
                'ref_wcs': ref_wcs,
                'sub_width': sub_width,
                'output_dir': output_dir,
                'replace_existing': replace_existing,
                'reproject_kwargs': reproject_kwargs
            }
            tasks.append(task_params)

    results = []
    if num_processes > 1 and len(tasks) > 0 : # Ensure there are tasks for multiprocessing
        with Pool(processes=num_processes) as pool:
            results = list(tqdm(pool.imap_unordered(_reproject_worker, tasks), total=len(tasks), desc='Reprojecting frames'))
    elif len(tasks) > 0: # Sequential execution
        for task in tqdm(tasks, desc='Reprojecting frames (sequentially)'):
            results.append(_reproject_worker(task))

    success_file = [r for r in results if r is not None]
    print(f'Batch reprojection completed. {len(success_file)} frames successfully processed out of {len(tasks)}.')
    return success_file
