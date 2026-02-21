import os
import h5py
import sys 
import glob
from tqdm import tqdm
import numpy as np

from . import WCSHelper
from . import MakeMap

from astropy.io import fits
from dataclasses import dataclass, field

from contextlib import contextmanager
import time

@contextmanager
def timer(description):
    start = time.perf_counter() # distinct from time.time(), better for execution duration
    yield
    elapsed = time.perf_counter() - start
    print(f"{description} finished in {elapsed:.2f} seconds.")

@dataclass
class PipelineConfig:
    output_dir: str
    run_name: str
    resolution_arcsec: float
    ref_path: str = None
    reproj_dir: str = None
    cal_dir: str = None
    mos_dir: str = None

    def __post_init__(self):
        # Auto-fill dependent paths if they weren't provided
        base_path = os.path.join(self.output_dir, self.run_name)
        if self.ref_path is None:
            self.ref_path = os.path.join(base_path, 'ref.fits')
        if self.reproj_dir is None:
            self.reproj_dir = os.path.join(base_path, 'reprojected')
        if self.cal_dir is None:
            self.cal_dir = os.path.join(base_path, 'calibration')
        if self.mos_dir is None:
            self.mos_dir = os.path.join(base_path, 'mosaic')

class Reprojector:
    def __init__(self, config: PipelineConfig, exposure_list=None):
        '''Initialize path to reference WCS and reprojected files'''
        self.config = config

        self.exposure_list = exposure_list
        if not os.path.exists(self.config.reproj_dir):
            os.makedirs(self.config.reproj_dir)

        self.ref_shape = None
        self.ref_wcs = None

    def define_reference(self, padding_pixels=100, use_ext=[1]):
        '''Define the smallest WCS oriented north-up, east-left frame that can contain all exposures'''
        if not os.path.exists(self.config.ref_path):
            print(f"Reference WCS not found at {self.config.ref_path}. Creating a new reference frame.")
            self.ref_wcs, self.ref_shape = WCSHelper.find_optimal_frame(
                exposure_list=self.exposure_list,
                resolution_arcsec=self.config.resolution_arcsec,
                padding_pixels=padding_pixels,
                use_ext=use_ext
            )
            WCSHelper.save_to_fits(self.ref_wcs, self.ref_shape, self.config.ref_path)
            print(f"Reference WCS saved to {self.config.ref_path}")
        else:
            self.ref_wcs, self.ref_shape = WCSHelper.load_from_fits(self.config.ref_path)
        print(f'Mosaic shape: {self.ref_shape}')
        print(f'Mosaic WCS: {self.ref_wcs}')

    def run_reproject(self, max_workers=50, reproj_func='exact', padding_percentage=0.05, 
                      sci_ext_list=None, dq_ext_list=None, exp_idx_list=None, det_idx_list=None,
                      output_dir=None, replace_existing=False, reproject_kwargs={}):
        if self.ref_wcs is None or self.ref_shape is None:
            raise ValueError("Reference WCS and shape must be defined before running reprojection. Call define_reference() first.")
        if output_dir is None:
            output_dir = self.config.reproj_dir

        with timer("Reprojection"):
            self.reproj_list = MakeMap.batch_reproject(
                # Can edit
                num_processes = max_workers, 
                reproj_func = reproj_func,  # interp: fastest, adaptive: conserves flux, exact: most accurate
                # Porbably don't want to edit
                exposure_list = self.exposure_list,
                ref_wcs = self.ref_wcs, 
                ref_shape = self.ref_shape,
                output_dir = output_dir, 
                padding_percentage = padding_percentage,
                sci_ext_list = sci_ext_list, 
                dq_ext_list = dq_ext_list,
                exp_idx_list = exp_idx_list,
                det_idx_list = det_idx_list,
                replace_existing = replace_existing,
                reproject_kwargs = reproject_kwargs
                )
            
    def check_reproj_files(self):
        for f in tqdm(self.reproj_list):
            result = MakeMap.load_reproj_file(f, fields=['sub_data',])
            if result['_is_missing_']:
                os.remove(f)
                print(f"Removed {f} due to missing data")

    def get_reproj_files(self, reproj_dir=None):
        if reproj_dir is None:
            reproj_dir = self.config.reproj_dir
        self.reproj_list = sorted(glob.glob(os.path.join(reproj_dir, '*.h5')))
        self.det_idx_list = []
        self.exp_idx_list = []
        for file in tqdm(self.reproj_list):
            file_name = os.path.basename(file)
            exp_idx, det_idx = int(file_name.split('_')[1]), int(file_name.split('_')[3].strip('.h5'))
            self.det_idx_list.append(det_idx)
            self.exp_idx_list.append(exp_idx)
        
class Calibrator(Reprojector):
    def __init__(self, config: PipelineConfig, reproj_dir=None):
        super().__init__(config)
        self.get_reproj_files(reproj_dir)
        self.ref_wcs, self.ref_shape = WCSHelper.load_from_fits(self.config.ref_path)
        self.A = None
        self.b = None
        self.x = None
        self.pixel_counts = None

    def setup_lsqr(self, chunk_map, grid_valid_weight, oversample_factor=1, apply_mask=True, apply_weight=True, max_workers=20, 
                   outlier_thresh=3.0, ignore_list=[], batch_size=10, offset_regularization=False, reg_weight=0.0, adj_info=None, mean_offsets=None,
                   postprocess_func=None, preprocess_func=None, weighted_damping=False, damp_weight=0.1):
        self.chunk_map = chunk_map
        with timer("Setup LSQR"):
            self.A, self.b, self.pixel_counts = MakeMap.setup_lsqr(self.reproj_list, self.ref_shape,
                apply_mask=apply_mask, apply_weight=apply_weight, chunk_map=chunk_map, grid_valid_weight=grid_valid_weight,
                max_workers=max_workers, outlier_thresh=outlier_thresh, ignore_list=ignore_list, oversample_factor=oversample_factor,
                batch_size=batch_size, offset_regularization=offset_regularization, reg_weight=reg_weight, adj_info=adj_info, mean_offsets=mean_offsets, postprocess_func=postprocess_func, preprocess_func=preprocess_func,
                weighted_damping=weighted_damping, damp_weight=damp_weight)
    
    
    def apply_lsqr(self, x0=None, atol=1e-06, btol=1e-06, damp=1e-2, iter_lim=300, precondition=True, resume=False):
        if resume:
            if self.x is None:
                print("No previous solution found. Starting from scratch.")
            else:
                x0 = self.x
                print("Resuming LSQR from previous solution.")
        if self.A is None or self.b is None:
            raise ValueError("LSQR matrix A and vector b must be set up before applying LSQR.")
        with timer("LSQR"):
            self.x = MakeMap.apply_lsqr(self.A, self.b, ref_shape=self.ref_shape, num_frames=len(self.reproj_list),
                                                        x0=x0, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim, precondition=precondition)
    
    def load_calibration(self, cal_path=None):
        if cal_path is None:
            cal_path = os.path.join(self.config.cal_dir, 'cal.h5')
        with h5py.File(cal_path, 'r') as f:
            skymap = f['skymap']
            offset = f['offset']
            self.x = MakeMap.encode_x(skymap, offset)

    def save_calibration(self, cal_dir=None, cal_file='cal.h5'):
        if cal_dir is None:
            cal_dir = self.config.cal_dir
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)
        skymap, offset = MakeMap.parse_x(self.x, ref_shape=self.ref_shape, num_frames=len(self.reproj_list))
        num_sky = self.ref_shape[0] * self.ref_shape[1]

        skymap_coverage, offset_coverage, offset_coverage_frac = MakeMap.parse_pixel_counts(
            pixel_counts=self.pixel_counts, ref_shape=self.ref_shape, num_frames=len(self.reproj_list), chunk_map=self.chunk_map)

        cal_path = os.path.join(cal_dir, cal_file)
        with h5py.File(cal_path, 'w') as f:
            f.create_dataset('offset', data=offset, compression='gzip')
            f.create_dataset('skymap', data=skymap, compression='gzip')
            f.create_dataset('reproj_list', data=np.array(self.reproj_list, dtype='S'))
            f.create_dataset('skymap_coverage', data=skymap_coverage, compression='gzip')
            f.create_dataset('offset_coverage', data=offset_coverage, compression='gzip')
            f.create_dataset('offset_coverage_frac', data=offset_coverage_frac, compression='gzip')
        print(f"Calibration saved to {cal_path}")
        return cal_path

    def get_skymap(self):
        skymap, offset = MakeMap.parse_x(self.x, ref_shape=self.ref_shape, num_frames=len(self.reproj_list))
        return skymap

    def get_offset(self):
        skymap, offset = MakeMap.parse_x(self.x, ref_shape=self.ref_shape, num_frames=len(self.reproj_list))
        return offset

class Mosaicker(Reprojector):
    def __init__(self, config: PipelineConfig, reproj_dir=None):
        super().__init__(config)
        self.get_reproj_files(reproj_dir)
        self.ref_wcs, self.ref_shape = WCSHelper.load_from_fits(self.config.ref_path)
        self.cal_path = None
        self.cached_list = []
        self.offset = None
        self.skymap = None
        self.maps = {'mean_map': {'data': None, 'weight': None, 'aux': None, 'unit': 'MJy/sr'},
                     'std_map': {'data': None, 'weight': None, 'aux': None, 'unit': 'MJy/sr'},
                     'sc_mean_map': {'data': None, 'weight': None, 'aux': None, 'unit': 'MJy/sr'}}

    def load_calibration(self, cal_path):
        with h5py.File(cal_path, 'r') as f:
            self.offset = f['offset'][:]
            self.skymap = f['skymap'][:]
            self.reproj_list = [s.decode('utf-8') for s in f['reproj_list'][:]]
            self.offset_coverage_frac = f['offset_coverage_frac'][:]
            self.skymap_coverage = f['skymap_coverage'][:]
            self.offset_coverage = f['offset_coverage'][:]
        print(f"Calibration loaded from {cal_path}")
        self.cal_path = cal_path

    def make_mosaic(self, chunk_map, grid_valid_weight, oversample_factor=1, apply_mask=True, apply_weight=True, max_workers=20, 
        make_std_map=False, apply_sigma_clipping=False, sigma=2.0, normalize_offset=False, apply_offset=True, ignore_list=[], 
        det_offset_func=None, cache_batch_size=10, coadd_batch_size=10, cache_dir='cache/', 
        cache_intermediate=False, det_aux=None, preprocess_func=None, postprocess_func=None, valid_chunk_thresh=0.01):
        
        self.chunk_map = chunk_map

        offset_param = None
        if apply_offset:
            if self.offset is not None:
                offset = self.offset.copy()
                offset_valid_mask = (self.offset_coverage_frac >= valid_chunk_thresh)
                if normalize_offset:
                    offset[offset_valid_mask] = offset[offset_valid_mask] - np.mean(offset[offset_valid_mask])
                offset[~offset_valid_mask] = 0.0
                offset_param = offset
            else:
                print("Warning: Calibration offsets not available. No offsets will be applied.")

        # Bundle arguments common to all compute_coadd_map calls
        common_kwargs = {
            'ref_shape': self.ref_shape,
            'file_list': self.reproj_list,
            'offset_list': offset_param,
            'apply_weight': apply_weight,
            'apply_mask': apply_mask,
            'chunk_map': chunk_map,
            'max_workers': max_workers,
            'grid_valid_weight': grid_valid_weight,
            'ignore_list': ignore_list,
            'oversample_factor': oversample_factor,
            'det_offset_func': det_offset_func,
            'cache_dir': cache_dir,
            'use_cached': False,
            'det_aux': det_aux,
            'preprocess_func': preprocess_func,
            'postprocess_func': postprocess_func
        }

        if cache_intermediate:
            print("Caching intermediate computations...")
            with timer("Cache computation"):
                cached_list = MakeMap.compute_coadd_map(
                    mode='cache',
                    batch_size=cache_batch_size,
                    **common_kwargs
                )
            self.cached_list = cached_list
            common_kwargs['file_list'] = cached_list
            common_kwargs['use_cached'] = True

        print("Computing mean map...")
        with timer("Mean map computation"):
            self.maps['mean_map']['data'], self.maps['mean_map']['weight'], self.maps['mean_map']['aux'] = MakeMap.compute_coadd_map(
                mode='mean', 
                batch_size=coadd_batch_size,
                **common_kwargs
            )
        
        if make_std_map:
            print("Computing std map...")
            with timer("Std map computation"):
                self.maps['std_map']['data'], self.maps['std_map']['weight'], self.maps['std_map']['aux'] = MakeMap.compute_coadd_map(
                    mode='std', 
                    mean_map=self.maps['mean_map']['data'], 
                    batch_size=coadd_batch_size,
                    **common_kwargs
                )

        if make_std_map and apply_sigma_clipping:
            print("Computing sigma-clipped mean map...")
            
            with timer("Sigma-clipped mean map computation"):
                self.maps['sc_mean_map']['data'], self.maps['sc_mean_map']['weight'], self.maps['sc_mean_map']['aux'] = MakeMap.compute_coadd_map(
                    mode='sigma_clip',
                    mean_map=self.maps['mean_map']['data'],
                    std_map=self.maps['std_map']['data'],
                    sigma=sigma,
                    batch_size=coadd_batch_size,
                    **common_kwargs
                    )

        return self.maps
    
    def append_maps(self, new_maps):
        for map_name in new_maps:
            self.maps[map_name] = {'data': None, 'weight': None, 'aux': None, 'unit': None}
            for key in new_maps[map_name]:
                self.maps[map_name][key] = new_maps[map_name][key]

    def save_mosaic(self, mos_dir=None, mos_file='mosaic.fits', overwrite=False):
        '''
        Extension naming convention:
        Coadd Maps: 
            - 'MEAN_MAP': Simple mean coadd
            - 'MEAN_MAP_WEIGHT': Weight map for mean coadd
            - 'STD_MAP': Standard deviation of pixel values per pixel
            - 'STD_MAP_WEIGHT': Weight map for std coadd
            - 'SC_MEAN_MAP': Sigma-clipped mean coadd
            - 'SC_MEAN_MAP_WEIGHT': Weight map for sigma-clipped mean coadd
        Auxiliary Maps:
            - 'WAV_MEAN': Mean wavelength map
            - 'WAV_STD': Standard deviation of wavelength map
        '''
        if mos_dir is None:
            mos_dir = self.config.mos_dir
        if not os.path.exists(mos_dir):
            os.makedirs(mos_dir)

        mos_path = os.path.join(mos_dir, mos_file)

        hdu_list = []
        for m in self.maps:
            if self.maps[m]['data'] is not None:
                hdu = fits.ImageHDU(data=self.maps[m]['data'], header=self.ref_wcs.to_header())
                hdu.header['NAXIS1'] = self.ref_shape[1]
                hdu.header['NAXIS2'] = self.ref_shape[0]
                hdu.header['NAXIS'] = 2
                hdu.header['BUNIT'] = self.maps[m]['unit']
                hdu.header['EXTNAME'] = m.upper()
                hdu_list.append(hdu)
            if self.maps[m]['weight'] is not None:
                hdu = fits.ImageHDU(data=self.maps[m]['weight'], header=self.ref_wcs.to_header())
                hdu.header['NAXIS1'] = self.ref_shape[1]
                hdu.header['NAXIS2'] = self.ref_shape[0]
                hdu.header['NAXIS'] = 2
                hdu.header['BUNIT'] = 'Weight'
                hdu.header['EXTNAME'] = f"{m.upper()}_WEIGHT"
                hdu_list.append(hdu)
            if self.maps[m]['aux'] is not None:
                hdu = fits.ImageHDU(data=self.maps[m]['aux'], header=self.ref_wcs.to_header())
                hdu.header['NAXIS1'] = self.ref_shape[1]
                hdu.header['NAXIS2'] = self.ref_shape[0]
                hdu.header['NAXIS'] = 2
                hdu.header['BUNIT'] = 'Auxiliary'
                hdu.header['EXTNAME'] = f"{m.upper()}_AUX"
                hdu_list.append(hdu)


        primary_hdu = fits.PrimaryHDU()

        hdul = fits.HDUList([primary_hdu] + hdu_list)
        hdul.writeto(mos_path, overwrite=overwrite)
        print(f"Mosaic saved to {mos_path}")
        return mos_path
