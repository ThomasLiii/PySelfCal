"""Reproject the NEP Detector 4 exposures (QR1 new gain + QR2) onto the Detector 5 reference grid
(was configs/reproject_d4.toml). Drops exposures with a poor astrometric solution (FINAST != 0)."""
from selfcal_scripts.recipes.spherex import nep

FIELD = nep(4)
REPROJECT = dict(exposures=['/mnt/md124/SPHEREx/SPHEREx_nep_data/qr1_newgain/*/*/*/*D4*.fits',
                            '/mnt/md124/SPHEREx/SPHEREx_nep_data/qr2/*/*/*/*D4*.fits'],
                 reference='/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D5_6p2arcsec/ref.fits',
                 compute=FIELD.compute.replace(workers=50))

if __name__ == '__main__':
    FIELD.reproject(**REPROJECT)
