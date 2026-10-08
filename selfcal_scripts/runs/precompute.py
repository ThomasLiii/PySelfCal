"""Fit and save the LVF arc parameters of Detectors 1 to 6 from the 2025-09 calibration maps. The
package ships them; run this only to regenerate them."""
from selfcal.instruments import spherex

PRECOMPUTE = dict(detectors=[1, 2, 3, 4, 5, 6], calib_dir='/data3/SPHEREx/SpecCal_202509/ParameterFiles', num_sub=10,
                  num_ch=34)

if __name__ == '__main__':
    spherex.precompute_lvf(**PRECOMPUTE)
