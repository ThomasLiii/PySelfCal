"""NEP Detector 4: as nep_d4_npass_big6, with 5 passes, offset first (the frames re-levelled against the
stitched first sky before the first exact sky)."""
import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import ORCA
from selfcal_scripts.recipes.spherex import NEP_BIG6_TILES, SUBCHANNEL, multiline3, nep

FIELD = nep(4, compute=ORCA.replace(stage_dir='reproj_nvme_SPHEREx_NEP_2026W17_D4_6p2arcsec', memory_guard=True))
RECIPE = multiline3('multiline3_NEPbig6of_STITCHED_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2')
RUN = dict(jobs=spherex.window('Multiline3', subchannels=range(200, 321)),
           tiles=sc.Tiles(boxes=NEP_BIG6_TILES, tile_name='multiline3_NEPbig6_{tile}_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2',
                          stitched_name='{name}'),
           passes=sc.Passes(5, sky_clip=sc.Clip(5.0, per=SUBCHANNEL),
                            offset=sc.Refit(clip=sc.Clip(2.5, per=SUBCHANNEL))))

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
