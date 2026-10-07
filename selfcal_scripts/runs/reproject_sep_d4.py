"""Reproject the SEP Detector 4 exposures (the /data3 copy) onto a reference grid fitted to them
(was configs/reproject_sep_d4.toml)."""
import selfcal as sc
from selfcal_scripts.recipes.site import ORCA

FIELD = sc.Field('/data3/thomasli/selfcal/outputs/SPHEREx_SEP_2025_D4_6p2arcsec', sc.SPHEREx(4, num_col=5), 6.2,
                 compute=ORCA)
REPROJECT = dict(exposures=['/data3/SPHEREx/SPHEREx_sep_data/*/*/*/*D4*.fits'], compute=ORCA.replace(workers=50))

if __name__ == '__main__':
    FIELD.reproject(**REPROJECT)
