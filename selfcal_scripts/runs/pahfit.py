"""NEP Detector 4 (the /data3 copy), the PAH 3.29 um window at NumCol 5: a constant sky plus the
catalogue PAH line (undamped), the column-polynomial offsets, shot-noise weights, the source-mask
bit ignored (was configs/pahfit.toml)."""
import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import ORCA

FIELD = sc.Field('/data3/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D4_6p2arcsec', sc.SPHEREx(4, num_col=5), 6.2,
                 compute=ORCA.replace(keep_staged=True))
RECIPE = sc.Recipe(sc.spectral([sc.Sky('pah_3p29', times=sc.catalog('pah_3p29'), damping=0.0)],
                               poly_prior=sc.Poly(1, weight=0.5)),
                   fit=sc.Fit(100, clip=5.0, ignore_flags=[21], shot_noise_weights=True),
                   coadd=sc.Coadd(clip=2.0, oversample=2, ignore_flags=[21]),
                   name='damp0p1_reg0p1_applyWt_PAHfit_dampL0_subch40_nosrcmask_NumCol5_iter100_outThresh5_sigma2_polyK1')
RUN = dict(jobs=spherex.window('Aromatic_PAHfit', subchannels=range(210, 250)))

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
