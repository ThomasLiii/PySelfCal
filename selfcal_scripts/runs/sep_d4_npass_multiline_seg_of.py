"""SEP Detector 4: the three-line PAH fit with a piecewise polynomial (two segments) in the solve and the
offset refit (ridge 0.03), six tiles, 5 passes offset first."""
import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import ORCA
from selfcal_scripts.recipes.spherex import MULTILINE3_PASSES, SEP6_TILES, multiline3, sep

#: The SEP products of these runs live on /data3; the frames are read from the md124 copy.
SEP = sep(4, compute=ORCA.replace(stage_dir='reproj_nvme_SEP_D4_npass', memory_guard=True),
          outputs='/data3/thomasli/selfcal/outputs')
FRAMES = '/mnt/md124/thomasli/selfcal/outputs/SPHEREx_SEP_2025_D4_6p2arcsec/reprojected'

SEGMENTS = ((200, 259), (260, 320))
FIELD = SEP
RECIPE = multiline3('multiline3_SEP6segof_STITCHED_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2', segments=SEGMENTS)
RUN = dict(jobs=spherex.window('Multiline3', subchannels=range(200, 321)),
           tiles=sc.Tiles(boxes=SEP6_TILES, tile_name='multiline3_SEP6seg_{tile}_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2', stitched_name='{name}'),
           passes=MULTILINE3_PASSES.replace(offset=MULTILINE3_PASSES.offset.replace(segments=SEGMENTS, ridge=0.03)),
           frames=FRAMES)

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
