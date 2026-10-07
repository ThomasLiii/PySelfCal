"""NEP Detector 5, channels 23 to 34: the NumCol 10 fiducial (was configs/d5.toml)."""
from selfcal.instruments import spherex
from selfcal_scripts.recipes.spherex import POLY_K1, nep

FIELD = nep(5, num_col=10)
RECIPE = POLY_K1
RUN = dict(jobs=spherex.channels(23, 34))

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
