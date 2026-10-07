"""SEP Detector 4: as sep_d4_npass, with 5 passes, offset first (was configs/sep_d4_npass_offsetfirst.toml)."""
import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import ORCA
from selfcal_scripts.recipes.spherex import MULTILINE3_PASSES, SEP_HALVES, multiline3, sep

#: The SEP products of these runs live on /data3; the frames are read from the md124 copy.
SEP = sep(4, compute=ORCA.replace(stage_dir='reproj_nvme_SEP_D4_npass', memory_guard=True),
          outputs='/data3/thomasli/selfcal/outputs')
FRAMES = '/mnt/md124/thomasli/selfcal/outputs/SPHEREx_SEP_2025_D4_6p2arcsec/reprojected'

FIELD = SEP
RECIPE = multiline3('PAHfit_realistic_dampL5e-3_HALVES_npass_wideG_of', iterations=100, lines=[spherex.line('aromatic', damping=5e-3).replace(name='pah_3p29')],
                    window=range(200, 260))
RUN = dict(jobs=spherex.window('Aromatic_PAHfit_realistic_NEbSWEEP', subchannels=range(200, 260)),
           tiles=sc.Tiles(boxes=SEP_HALVES, tile_name='PAHfit_realistic_dampL5e-3_HALF_{tile}_p1off',
                          stitched_name='{name}'),
           passes=MULTILINE3_PASSES,
           frames=FRAMES)

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
