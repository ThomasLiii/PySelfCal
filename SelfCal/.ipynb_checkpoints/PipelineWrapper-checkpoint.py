import os
import h5py
import sys 
from tqdm import tqdm
import numpy as np

sys.path.insert(0, '/home/thomasli/spherex/selfcal')
import EuclidUtility
import WCSHelper
import MakeMap
import MapHelper



from astropy.io import fits

from scipy.sparse.linalg import lsqr

from MapHelper import bit_to_bool, make_weight, find_outliers, map_pixels, det_to_sub
from WCSHelper import load_from_fits, save_to_fits, find_optimal_frame

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
            self.ref_wcs, self.ref_shape = find_optimal_frame(
                exposure_list=self.exposure_list,
                resolution_arcsec=self.config['resolution_arcsec'],
                padding_pixels=padding_pixels,
                use_ext=use_ext
            )
            save_to_fits(self.ref_wcs, self.ref_shape, os.path.join(self.config['output_dir'], self.config['run_name'], 'ref.fits'))
            print(f"Reference WCS saved to {self.config['ref_path']}")
        else:
            self.ref_wcs, self.ref_shape = WCSHelper.load_from_fits(self.config['ref_path'])
        print(f'Mosaic shape: {self.ref_shape}')
        print(f'Mosaic WCS: {self.ref_wcs}')

    def run_reproject(self, max_workers=50, method='exact', padding_percentage=0.05, oversample_factor=2, 
                      sci_ext_list=None, dq_ext_list=None, exp_idx_list=None, det_idx_list=None,
                      output_dir=None, replace_existing=False):
        if self.ref_wcs is None or self.ref_shape is None:
            raise ValueError("Reference WCS and shape must be defined before running reprojection. Call define_reference() first.")
        if output_dir is None:
            output_dir = self.config['reproj_dir']
        self.reproj_list = MakeMap.batch_reproject(
            # Can edit
            num_processes = max_workers, 
            method = method,  # interp: fastest, adaptive: conserves flux, exact: most accurate

            # Porbably don't want to edit
            exposure_list = self.exposure_list,
            ref_wcs = self.ref_wcs, 
            ref_shape = self.ref_shape,
            output_dir = output_dir, 
            padding_percentage = padding_percentage,
            oversample_factor = oversample_factor,
            sci_ext_list = sci_ext_list, 
            dq_ext_list = dq_ext_list,
            exp_idx_list = exp_idx_list,
            det_idx_list = det_idx_list,
            replace_existing = replace_existing
            )
        
    def check_reproj_files(self):
        for f in tqdm(self.reproj_list):
            result = MakeMap.load_reproj_file(f, fields=['sub_data',])
            if result['_is_missing_']:
                os.remove(f)
                print(f"Removed {f} due to missing data")

    def get_reproj_files(self, reproj_dir=None):
        if reproj_dir is None:
            reproj_dir = self.config['reproj_dir']
        self.reproj_list = sorted(glob.glob(os.path.join(reproj_dir, '*.h5')))
        self.det_idx_list = []
        self.exp_idx_list = []
        for file in tqdm(self.reproj_list):
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

    def setup_lsqr(self, apply_mask=True, apply_weight=True, chunk_map=None, chunk_valid_mask=None, max_workers=20, outlier_thresh=3.0):
        self.A, self.b = MakeMap.setup_lsqr(self.reproj_list, self.ref_shape, self.exp_idx_list, self.det_idx_list,
               apply_mask=apply_mask, apply_weight=apply_weight, chunk_map=chunk_map, chunk_valid_mask=chunk_valid_mask,
               max_workers=max_workers, outlier_thresh=outlier_thresh)
        
    def apply_lsqr(self, x0=None, atol=1e-06, btol=1e-06, damp=1e-2, iter_lim=300):
        if self.A is None or self.b is None:
            raise ValueError("LSQR matrix A and vector b must be set up before applying LSQR.")
        self.O, self.S, self.D = MakeMap.apply_lsqr(self.A, self.b, self.ref_shape, self.exp_idx_list, self.det_idx_list, 
                                                    x0=x0, atol=atol, btol=btol, damp=damp, iter_lim=iter_lim)
    
    def save_calibration(self, cal_dir=None, cal_file='cal.h5'):
        if cal_dir is None:
            cal_dir = self.config['cal_dir']
        if not os.path.exists(cal_dir):
            os.makedirs(cal_dir)

        cal_path = os.path.join(cal_dir, cal_file)
        with h5py.File(cal_path, 'w') as f:
            f.create_dataset('O', data=self.O, compression='gzip')
            f.create_dataset('S', data=self.S, compression='gzip')
            f.create_dataset('D', data=self.D, compression='gzip')
            f.create_dataset('reproj_list', data=np.array(self.reproj_list, dtype='S'))
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
        self.D = None

    def load_calibration(self, cal_path):
        with h5py.File(cal_path, 'r') as f:
            self.O = f['O'][:]
            self.S = f['S'][:]
            self.D = f['D'][:]
        print(f"Calibration loaded from {cal_path}")
        self.cal_path = cal_path

    def make_mosaic(self, apply_mask=True, apply_weight=True, chunk_map=None, chunk_valid_mask=None, max_workers=20, 
    make_std_map=False, apply_sigma_clipping=False, sigma=2.0):
        
        if self.O is None or self.D is None:
            print("Waning: Calibration not loaded. No calibration will be applied to the mosaic.")

        mean, weight = MakeMap.compute_mean_map(
            ref_shape=self.ref_shape,
            reproj_file_list=self.reproj_list,
            exp_offset=self.O-np.mean(self.O, axis=0) if self.O is not None else None,
            det_offset=[self.D-np.mean(self.D[chunk_valid_mask==1], axis=0)] if self.D is not None else None,
            det_idx_list=self.det_idx_list,
            exp_idx_list=self.exp_idx_list,
            apply_weight=apply_weight,
            apply_mask=apply_mask,
            chunk_map=chunk_map,
            max_workers=max_workers,
            chunk_valid_mask = chunk_valid_mask,
        )
        if make_std_map:
            std, _ = MakeMap.compute_std_map(
                mean_map=mean,
                ref_shape=self.ref_shape,
                reproj_file_list=self.reproj_list,
                exp_offset=self.O-np.mean(self.O, axis=0) if self.O is not None else None,
                det_offset=[self.D-np.mean(self.D[chunk_valid_mask==1], axis=0)] if self.D is not None else None,
                det_idx_list=self.det_idx_list,
                exp_idx_list=self.exp_idx_list,
                apply_weight=apply_weight,
                apply_mask=apply_mask,
                chunk_map=chunk_map,
                max_workers=max_workers,
                chunk_valid_mask = chunk_valid_mask
            )
        if make_std_map and apply_sigma_clipping:
            sc_mean, weight = MakeMap.compute_sc_mean(
                mean_map=mean,
                std_map=std,
                sigma=sigma,
                ref_shape=self.ref_shape,
                reproj_file_list=self.reproj_list,
                exp_offset=self.O-np.mean(self.O, axis=0) if self.O is not None else None,
                det_offset=[self.D-np.mean(self.D[chunk_valid_mask==1], axis=0)] if self.D is not None else None,
                exp_idx_list=self.exp_idx_list,
                det_idx_list=self.det_idx_list,
                apply_weight=apply_weight,
                apply_mask=True,
                chunk_map=chunk_map,
                chunk_valid_mask=chunk_valid_mask,
                max_workers=max_workers
            )
        self.mean_map = mean
        self.std_map = std if make_std_map else None
        self.sc_mean = sc_mean if make_std_map and apply_sigma_clipping else None

        return self.mean_map, self.std_map, self.sc_mean

        
    def save_mosaic(self, mos_dir=None, mos_file='mosaic.fits', overwrite=False):
        if mos_dir is None:
            mos_dir = self.config['mos_dir']
        if not os.path.exists(mos_dir):
            os.makedirs(mos_dir)

        mos_path = os.path.join(mos_dir, mos_file)

        mosaic_hdu = fits.PrimaryHDU(data=self.sc_mean, header=self.ref_wcs.to_header())
        mosaic_hdu.header['NAXIS1'] = self.ref_shape[1]
        mosaic_hdu.header['NAXIS2'] = self.ref_shape[0]
        mosaic_hdu.header['NAXIS'] = 2
        mosaic_hdu.header['BUNIT'] = 'MJy/sr'
        mosaic_hdu.header['EXTNAME'] = 'MOSAIC'
        mosaic_hdu.header['CAL_FILE'] = self.cal_path

        std_hdu = fits.ImageHDU(data=self.std_map, header=self.ref_wcs.to_header())
        std_hdu.header['NAXIS1'] = self.ref_shape[1]
        std_hdu.header['NAXIS2'] = self.ref_shape[0]
        std_hdu.header['NAXIS'] = 2
        std_hdu.header['BUNIT'] = 'MJy/sr'
        std_hdu.header['EXTNAME'] = 'STD'

        hdul = fits.HDUList([mosaic_hdu, std_hdu])
        hdul.writeto(mos_path, overwrite=overwrite)
        print(f"Mosaic saved to {mos_path}")
        return mos_path
