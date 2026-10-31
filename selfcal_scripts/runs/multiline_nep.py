"""NEP Detector 4: the three-line PAH fit over the Multiline3 window, solved on the 16 adaptive-overlap
tiles and stitched. The field's frames are staged to fast disk for the run."""
import selfcal as sc
from selfcal_scripts.spherex import jobs
from selfcal_scripts.recipes.site import ORCA
from selfcal_scripts.recipes.spherex import NEP_OVERLAP_TILES, multiline3, nep

FIELD = nep(4, compute=ORCA.replace(workers=24, stage_dir='reproj_nvme_SPHEREx_NEP_2026W17_D4_6p2arcsec', memory_guard=True))
RECIPE = multiline3('multiline3_NEPovlp_STITCHED_iter300_cheb2_NumCol3_outThresh5_sigma2')
RUN = dict(jobs=jobs.window('Multiline3', range(200, 321)),
           tiles=sc.Tiles(boxes=NEP_OVERLAP_TILES, tile_name='multiline3_NEPovlp_{tile}_iter300_cheb2_NumCol3_outThresh5_sigma2',
                          stitched_name='{name}'))

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
