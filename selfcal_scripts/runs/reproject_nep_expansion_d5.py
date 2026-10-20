"""Reproject the NEP Detector 5 expansion: the deep data set through 2026W29 (QR1 new gain + QR2, the new weeks
included) and one ring of the all-sky survey around it, onto an expanded canvas of the 2026W17 Detector 5 run's
projection (``reference``: its WCS; the canvas is fitted to these exposures, 100 px wider). The W17 run's frames are
moved onto the new canvas first (workspace/survey-expansion/scripts/expand_field.py: ref_coords shifted by the
whole-pixel CRPIX offset), so this reprojects only the frames the W17 run does not have. Drops exposures with a poor
astrometric solution (FINAST != 0)."""
import selfcal as sc
from selfcal_scripts.recipes.site import ORCA, OUTPUTS

FIELD = sc.Field(f'{OUTPUTS}/SPHEREx_NEP_expansion_2026W29_D5_6p2arcsec', sc.SPHEREx(5), 6.2,
                 compute=ORCA.replace(workers=24))
REPROJECT = dict(exposures=['/mnt/md124/SPHEREx/SPHEREx_nep_data/qr1_newgain/*/*/*/*D5*.fits',
                            '/mnt/md124/SPHEREx/SPHEREx_nep_data/qr2/*/*/*/*D5*.fits',
                            '/mnt/md124/SPHEREx/SPHEREx_nep_expansion_data/*/*/*/*D5*.fits'],
                 reference=f'{OUTPUTS}/SPHEREx_NEP_2026W17_D5_6p2arcsec/ref.fits',
                 verify=True)

if __name__ == '__main__':
    FIELD.reproject(**REPROJECT)
