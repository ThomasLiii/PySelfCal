import os
import h5py
import sys 
import glob
from tqdm import tqdm
import numpy as np

from . import WCSHelper
from . import MakeMap

from astropy.io import fits

from contextlib import contextmanager
import time

@contextmanager
def timer(description):
    start = time.perf_counter() # distinct from time.time(), better for execution duration
    yield
    elapsed = time.perf_counter() - start
    print(f"{description} finished in {elapsed:.2f} seconds.")

class Reprojector:
    def __init__(self, config, exposure_list=None):
        '''Initialize path to reference WCS and reprojected files'''
        self.config = config
        # check if the config has the required keys
        required_keys = ['output_dir', 'run_name', 'resolution_arcsec']
        for key in required_keys:
            if key not in self.config:
                raise ValueError(f"Configuration must include '{key}'")

        self.exposure_list = exposure_list
        if 'ref_path' not in self.config:
            self.config['ref_path'] = os.path.join(self.config['output_dir'], self.config['run_name'], 'ref.fits')
        if 'reproj_dir' not in self.config:
            self.config['reproj_dir'] = os.path.join(self.config['output_dir'], self.config['run_name'], 'reprojected')
        if not os.path.exists(self.config['reproj_dir']):
            os.makedirs(self.config['reproj_dir'])

        self.ref_shape = None
        self.ref_wcs = None

    def define_reference(self, padding_pixels=100, use_ext=[1]):
        '''Define the smallest WCS oriented north-up, east-left frame that can contain all exposures'''
        if not os.path.exists(self.config['ref_path']):
            print(f"Reference WCS not found at {self.config['ref_path']}. Creating a new reference frame.")
            self.ref_wcs, self.ref_shape = WCSHelper.find_optimal_frame(
                exposure_list=self.exposure_list,
                resolution_arcsec=self.config['resolution_arcsec'],
                padding_pixels=padding_pixels,
                use_ext=use_ext
            )
            WCSHelper.save_to_fits(self.ref_wcs, self.ref_shape, os.path.join(self.config['output_dir'], self.config['run_name'], 'ref.fits'))
            print(f"Reference WCS saved to {self.config['ref_path']}")
        else:
            self.ref_wcs, self.ref_shape = WCSHelper.load_from_fits(self.config['ref_path'])
        print(f'Mosaic shape: {self.ref_shape}')
        print(f'Mosaic WCS: {self.ref_wcs}')

    def run_reproject(self, max_workers=50, reproj_func='exact', padding_percentage=0.05, 
                      sci_ext_list=None, dq_ext_list=None, exp_idx_list=None, det_idx_list=None,
                      output_dir=None, replace_existing=False, reproject_kwargs={}):
        if self.ref_wcs is None or self.ref_shape is None:
            raise ValueError("Reference WCS and shape must be defined before running reprojection. Call define_reference() first.")
        if output_dir is None:
            output_dir = self.config['reproj_dir']

        start_time = time.time()
        self.file_list = MakeMap.batch_reproject(
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
        end_time = time.time()
        print(f"Reprojection completed in {end_time - start_time:.2f} seconds.")
        
    def check_reproj_files(self):
        for f in tqdm(self.file_list):
            result = MakeMap.load_reproj_file(f, fields=['sub_data',])
            if result['_is_missing_']:
                os.remove(f)
                print(f"Removed {f} due to missing data")

    def get_reproj_files(self, reproj_dir=None):
        if reproj_dir is None:
            reproj_dir = self.config['reproj_dir']
        self.file_list = sorted(glob.glob(os.path.join(reproj_dir, '*.h5')))
        self.det_idx_list = []
        self.exp_idx_list = []
        for file in tqdm(self.file_list):
            file_name = os.path.basename(file)
            self.det_idx_list.append(int(file_name[file_name.find('det_')+4:file_name.find('det_')+6]))
            self.exp_idx_list.append(int(file_name[file_name.find('exp_')+4:file_name.find('exp_')+8]))
        
class Calibrator(Reprojector):
    def __init__(self, config, reproj_dir=None):
        super().__init__(config)
        self.config['cal_dir'] = os.path.join(self.config['output_dir'], self.config['run_name'], 'calibration')
        self.get_reproj_files(reproj_dir)
        self.ref_wcs, self.ref_shape = WCSHelper.load_from_fits(self.config['ref_path'])
        self.A = None
        self.B = None

    def setup_lsqr(self, apply_mask=True, apply_weight=True, chunk_map=None, det_valid_mask=None, max_workers=20, 
                   outlier_thresh=3.0, ignore_list=[], oversample_factor=1, batch_size=10):
        start_time = time.time()
        self.A, self.b = MakeMap.setup_lsqr(self.file_list, self.ref_shape, self.exp_idx_list, self.det_idx_list,
               apply_mask=apply_mask, apply_weight=apply_weight, chunk_map=chunk_map, det_valid_mask=det_valid_mask,
               max_workers=max_workers, outlier_thresh=outlier_thresh, ignore_list=ignore_list, oversample_factor=oversample_factor,
               batch_size=batch_size)
        end_time = time.time()
        print(f"LSQR setup completed in {end_time - start_time:.2f} seconds.")

    def apply_lsqr(self, x0=None, atol=1e-06, btol=1e-06, damp=1e-2, iter_lim=300):
        start_time = time.time()
        if self.A is None or self.b is None:
            raise ValueError("LSQR matrix A and vector b must be set up before applying LSQR.")
        self.O, self.S = MakeMap.apply_lsqr(self.A, self.b, self.ref_shape, self.exp_idx_list, self.det_idx_list, 
                                                    x0=x0, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim)
        end_time = time.time()
        print(f"LSQR solved in {end_time - start_time:.2f} seconds.")
    
    def save_calibration(self, cal_dir=None, cal_file='cal.h5'):
        if cal_dir is None:
            cal_dir = self.config['cal_dir']
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)

        cal_path = os.path.join(cal_dir, cal_file)
        with h5py.File(cal_path, 'w') as f:
            f.create_dataset('O', data=self.O, compression='gzip')
            f.create_dataset('S', data=self.S, compression='gzip')
            f.create_dataset('file_list', data=np.array(self.file_list, dtype='S'))
        print(f"Calibration saved to {cal_path}")
        return cal_path

class Mosaicker(Reprojector):
    def __init__(self, config, reproj_dir=None):
        super().__init__(config)
        self.get_reproj_files(reproj_dir)
        self.ref_wcs, self.ref_shape = WCSHelper.load_from_fits(self.config['ref_path'])
        self.config['mos_dir'] = os.path.join(self.config['output_dir'], self.config['run_name'], 'mosaic')
        self.cal_path = None
        self.O = None
        self.S = None
        self.maps = {'mean_map': None, 'mean_weight': None,
                     'std_map': None, 'std_weight': None,
                     'sc_mean_map': None, 'sc_mean_weight': None}

    def load_calibration(self, cal_path):
        with h5py.File(cal_path, 'r') as f:
            self.O = f['O'][:]
            self.S = f['S'][:]
        print(f"Calibration loaded from {cal_path}")
        self.cal_path = cal_path

    def make_mosaic(self, apply_mask=True, apply_weight=True, chunk_map=None, det_valid_mask=None, max_workers=20, 
        make_std_map=False, apply_sigma_clipping=False, sigma=2.0, normalize_offset=True, apply_offset=True, ignore_list=[], 
        oversample_factor=1, det_offset_func=None, cache_batch_size=10, coadd_batch_size=10, cache_dir='cache/', cache_intermediate=False):

        if self.O is None:
            print("Warning: Calibration not loaded. No calibration will be applied to the mosaic.")
        
        offset_param = None
        if self.O is not None and apply_offset:
            O = self.O
            if normalize_offset:
                O = O - np.mean(O[O != 0])
            offset_param = O

        # Bundle arguments common to all compute_coadd_map calls
        common_kwargs = {
            'ref_shape': self.ref_shape,
            'file_list': self.file_list,
            'offset_list': offset_param,
            'det_idx_list': self.det_idx_list,
            'exp_idx_list': self.exp_idx_list,
            'apply_weight': apply_weight,
            'apply_mask': apply_mask,
            'chunk_map': chunk_map,
            'max_workers': max_workers,
            'det_valid_mask': det_valid_mask,
            'ignore_list': ignore_list,
            'oversample_factor': oversample_factor,
            'det_offset_func': det_offset_func,
            'cache_dir': cache_dir,
            'use_cached': False,
        }

        if cache_intermediate:
            print("Caching intermediate computations...")
            with timer("Cache computation"):
                cached_files = MakeMap.compute_coadd_map(
                    mode='cache',
                    batch_size=cache_batch_size,
                    **common_kwargs
                )
            common_kwargs['file_list'] = cached_files
            common_kwargs['use_cached'] = True

        print("Computing mean map...")
        with timer("Mean map computation"):
            self.maps['mean_map'], self.maps['mean_weight'] = MakeMap.compute_coadd_map(
                mode='mean', 
                batch_size=coadd_batch_size,
                **common_kwargs
            )
        
        if make_std_map:
            print("Computing std map...")
            with timer("Std map computation"):
                self.maps['std_map'], self.maps['std_weight'] = MakeMap.compute_coadd_map(
                    mode='std', 
                    mean_map=self.maps['mean_map'], 
                    batch_size=coadd_batch_size,
                    **common_kwargs
                )

        if make_std_map and apply_sigma_clipping:
            print("Computing sigma-clipped mean map...")
            
            with timer("Sigma-clipped mean map computation"):
                self.maps['sc_mean_map'], self.maps['sc_mean_weight'] = MakeMap.compute_coadd_map(
                    mode='sigma_clip',
                    mean_map=self.maps['mean_map'],
                    std_map=self.maps['std_map'],
                    sigma=sigma,
                    batch_size=coadd_batch_size,
                    **common_kwargs
                    )

        return self.maps

        
    def save_mosaic(self, mos_dir=None, mos_file='mosaic.fits', overwrite=False):
        if mos_dir is None:
            mos_dir = self.config['mos_dir']
        if not os.path.exists(mos_dir):
            os.makedirs(mos_dir)

        mos_path = os.path.join(mos_dir, mos_file)

        hdu_list = []
        for m in self.maps:
            if self.maps[m] is not None:
                hdu = fits.ImageHDU(data=self.maps[m], header=self.ref_wcs.to_header())
                hdu.header['NAXIS1'] = self.ref_shape[1]
                hdu.header['NAXIS2'] = self.ref_shape[0]
                hdu.header['NAXIS'] = 2
                hdu.header['BUNIT'] = 'MJy/sr'
                hdu.header['EXTNAME'] = m.upper()
                hdu_list.append(hdu)


        primary_hdu = fits.PrimaryHDU()

        hdul = fits.HDUList([primary_hdu] + hdu_list)
        hdul.writeto(mos_path, overwrite=overwrite)
        print(f"Mosaic saved to {mos_path}")
        return mos_path
