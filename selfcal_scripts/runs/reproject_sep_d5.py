"""Reproject the SEP Detector 5 exposures (the /data3 copy) onto the Detector 4 reference grid."""
import selfcal as sc
from selfcal_scripts.recipes.site import ORCA

FIELD = sc.Field('/data3/thomasli/selfcal/outputs/SPHEREx_SEP_2025_D5_6p2arcsec', sc.SPHEREx(5, num_col=5), 6.2,
                 compute=ORCA)
REPROJECT = dict(exposures=['/data3/SPHEREx/SPHEREx_sep_data/*/*/*/*D5*.fits'],
                 reference='/data3/thomasli/selfcal/outputs/SPHEREx_SEP_2025_D4_6p2arcsec/ref.fits',
                 compute=ORCA.replace(workers=50))

if __name__ == '__main__':
    FIELD.reproject(**REPROJECT)
