"""NEP Detector 4: a 1,000-frame probe of the three-line fit and 4 N-pass passes, sky first, untiled."""
import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import ORCA
from selfcal_scripts.recipes.spherex import SUBCHANNEL, multiline3, nep

FIELD = nep(4, compute=ORCA.replace(workers=24))
RECIPE = multiline3('multiline3_probe1k_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2')
RUN = dict(jobs=spherex.window('Multiline3', subchannels=range(200, 321)),
           passes=sc.Passes(4, order='sky_first', sky_clip=sc.Clip(5.0, per=SUBCHANNEL),
                            offset=sc.Refit(clip=sc.Clip(2.5, per=SUBCHANNEL))),
           frames='/home/thomasli/selfcal-project/selfcal/cache/reproj_nep1k_multiline')

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
