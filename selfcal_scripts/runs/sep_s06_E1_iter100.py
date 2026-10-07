"""SEP Detector 4, tile S06 only, the first pass alone: experiment E1 on the multi-line fit
(was configs/sep_s06_E1_iter100.toml)."""
import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import ORCA
from selfcal_scripts.recipes.spherex import MULTILINE3_PASSES, SEP6_TILES, multiline3, sep

#: The SEP products of these runs live on /data3; the frames are read from the md124 copy.
SEP = sep(4, compute=ORCA.replace(stage_dir='reproj_nvme_SEP_D4_npass', memory_guard=True),
          outputs='/data3/thomasli/selfcal/outputs')
FRAMES = '/mnt/md124/thomasli/selfcal/outputs/SPHEREx_SEP_2025_D4_6p2arcsec/reprojected'

FIELD = SEP
RECIPE = multiline3('multiline3_SEP6_E1iter100_STITCHED_unused', iterations=100)
RUN = dict(jobs=spherex.window('Multiline3', subchannels=range(200, 321)),
           tiles=sc.Tiles(boxes=SEP6_TILES, only=['S06'], tile_name='multiline3_SEP6_E1iter100_{tile}_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2',
                          stitched_name='{name}'),
           passes=MULTILINE3_PASSES.replace(n=1),
           frames=FRAMES)

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
