"""SEP Detector 4 at NumCol 1: the three-line PAH fit on six tiles, then 5 N-pass passes, offset first
(was configs/sep_d4_npass_multiline_c1_of.toml)."""
import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import ORCA
from selfcal_scripts.recipes.spherex import MULTILINE3_PASSES, SEP6_TILES, multiline3, sep

#: The SEP products of these runs live on /data3; the frames are read from the md124 copy.
SEP = sep(4, num_col=1, compute=ORCA.replace(stage_dir='reproj_nvme_SEP_D4_npass', memory_guard=True),
          outputs='/data3/thomasli/selfcal/outputs')
FRAMES = '/mnt/md124/thomasli/selfcal/outputs/SPHEREx_SEP_2025_D4_6p2arcsec/reprojected'

FIELD = SEP
RECIPE = multiline3('multiline3_SEP6c1of_STITCHED_iter300_polybasisD2noortho_NumCol1_outThresh5_sigma2')
RUN = dict(jobs=spherex.window('Multiline3', subchannels=range(200, 321)),
           tiles=sc.Tiles(boxes=SEP6_TILES, tile_name='multiline3_SEP6c1_{tile}_iter300_polybasisD2noortho_NumCol1_outThresh5_sigma2', stitched_name='{name}'),
           passes=MULTILINE3_PASSES,
           frames=FRAMES)

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
