"""NEP Detector 4, the Aromatic and Aliphatic windows: the NumCol 10 fiducial with shot-noise
weights in the solve and a 1x coadd."""
from selfcal.instruments import spherex
from selfcal_scripts.recipes.spherex import POLY_K1, nep

FIELD = nep(4, num_col=10)
RECIPE = POLY_K1.replace(fit=POLY_K1.fit.replace(shot_noise_weights=True),
                         coadd=POLY_K1.coadd.replace(oversample=1),
                         name='damp0p1_reg0p1_outThresh1_sigma2_polyK1')
RUN = dict(jobs=(spherex.window('Aromatic'), spherex.window('Aliphatic')))

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
