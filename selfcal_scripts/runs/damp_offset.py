"""NEP Detector 4 (the /data3 copy), the Aromatic window: the NumCol 10 fiducial with shot-noise
weights in the solve; the staged frames are kept (was configs/damp_offset.toml)."""
import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import ORCA
from selfcal_scripts.recipes.spherex import POLY_K1

FIELD = sc.Field('/data3/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D4_6p2arcsec', sc.SPHEREx(4, num_col=10), 6.2,
                 compute=ORCA.replace(keep_staged=True))
RECIPE = POLY_K1.replace(fit=POLY_K1.fit.replace(shot_noise_weights=True),
                         name='damp0p1_reg0p1_applyWt_outThresh5_sigma2_polyK1')
RUN = dict(jobs=spherex.window('Aromatic'))

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
