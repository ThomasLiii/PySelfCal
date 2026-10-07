"""NEP QR2 Detector 5, channel 3, one column: a free offset per frame and subchannel plus a
detector-fixed offset per readout channel, shared by every frame (was configs/k2_readout.toml)."""
import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import ORCA
from selfcal_scripts.recipes.spherex import nep_qr2

FIELD = nep_qr2(5, num_col=1, compute=ORCA.replace(workers=32, coadd_workers=32, keep_staged=True, cache_frames=False))
RECIPE = sc.Recipe(sc.two_block(second='readout', smooth=0.1, second_smooth=0.0),
                   fit=sc.Fit(50, clip=5.0),
                   coadd=sc.Coadd(clip=None, std=False, oversample=2, ignore_flags=[21], instrument_maps=False),
                   numerics=sc.Numerics(32, batch=20, mosaic_batch=20, coadd_batch=30),
                   name='k2_readout_test')
RUN = dict(jobs=spherex.channel(3))

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
