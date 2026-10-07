"""NEP Detector 4: the three-line PAH fit on the 16 overlap tiles, then the N-pass solve (8 passes, sky
first) (was configs/nep_d4_npass.toml)."""
import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import ORCA
from selfcal_scripts.recipes.spherex import NEP_OVERLAP_TILES, SUBCHANNEL, multiline3, nep

FIELD = nep(4, compute=ORCA.replace(stage_dir='reproj_nvme_SPHEREx_NEP_2026W17_D4_6p2arcsec', memory_guard=True))
RECIPE = multiline3('multiline3_NEPovlp_STITCHED_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2')
RUN = dict(jobs=spherex.window('Multiline3', subchannels=range(200, 321)),
           tiles=sc.Tiles(boxes=NEP_OVERLAP_TILES, tile_name='multiline3_NEPovlp_{tile}_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2',
                          stitched_name='{name}'),
           passes=sc.Passes(8, order='sky_first', sky_clip=sc.Clip(5.0, per=SUBCHANNEL),
                            offset=sc.Refit(clip=sc.Clip(2.5, per=SUBCHANNEL))))

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
