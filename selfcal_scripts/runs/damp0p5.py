"""NEP Detector 4, channel 1: the NumCol 10 fiducial with five times the sky damping and smoothness,
reading the frames another run staged (was configs/damp0p5.toml)."""
import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import ORCA
from selfcal_scripts.recipes.spherex import POLY_K1, nep

FIELD = nep(4, num_col=10, compute=ORCA.replace(stage='reuse'))
RECIPE = POLY_K1.replace(model=sc.continuum(smooth=0.5, poly_prior=sc.Poly(1, weight=0.5), damping=0.5),
                         name='damp0p5_reg0p5_outThresh5_sigma2_polyK1')
RUN = dict(jobs=spherex.channel(1))

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
