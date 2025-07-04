from py import PipelineWrapper
from astropy.io import fits
import numpy as np
import glob
from py.SPHERExUtility import interpolate_array, make_chunk_map, make_chunk_mask, visualize_chunk_map

detector = 2

config = {}
config['output_dir'] = '/home/thomasli/spherex/selfcal/outputs'
config['run_name'] = f'nep_det{detector}_6p2arcsec'
config['resolution_arcsec'] = 6.2

chunk_map = make_chunk_map(detector, interp_factor=5)
chs = [[6], [7], [8], [9], [10], [11], [12], [13], [14], [15], [16], [17]]

for ch in chs:
    ch_name = '-'.join([str(i) for i in ch])
    print(f"Processing channels {ch_name}")
    
    chunk_valid_mask = make_chunk_mask(ch, interp_factor=5)

    cc = PipelineWrapper.Calibrator(config)

    cc.setup_lsqr(
        apply_mask=True, 
        apply_weight=True, 
        chunk_map=chunk_map, 
        chunk_valid_mask=chunk_valid_mask, 
        max_workers=20, 
        outlier_thresh=1.0
        )

    cc.apply_lsqr(x0=None, atol=1e-06, btol=1e-06, damp=1e-2, iter_lim=300)

    cal_path = cc.save_calibration(cal_file=f'cal_det{detector}_ch{ch_name}.h5')

    mm = PipelineWrapper.Mosaicker(config)
    mm.load_calibration(cal_path=cal_path)

    # mean, std, sc = mm.make_mosaic(
    #     apply_mask=True, 
    #     apply_weight=True, 
    #     chunk_map=chunk_map, 
    #     chunk_valid_mask=chunk_valid_mask, 
    #     max_workers=40,
    #     make_std_map=True, 
    #     apply_sigma_clipping=True, 
    #     sigma=2.0
    # )

    # mm.save_mosaic(mos_file=f'nep_6p2arcsec_det{detector}_ch{ch_name}.fits', overwrite=True)