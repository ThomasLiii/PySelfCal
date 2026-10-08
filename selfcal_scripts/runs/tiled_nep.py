"""NEP Detector 4 (the /data3 copy) at NumCol 5: the tiled PAH 3.29 um fit, 2x2 tiles overlapping by 50 pixels,
400 iterations, stitched."""
import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import ORCA
from selfcal_scripts.recipes.spherex import nep, pahfit_tiled

FIELD = nep(4, num_col=5, compute=ORCA.replace(workers=48, stage_dir='reproj_nvme_pahfit_region_10k', memory_guard=True),
              outputs='/data3/thomasli/selfcal/outputs')
RECIPE = pahfit_tiled('damp0p1_reg0p1_applyWt_PAHfit_dampL5e-3_subch60_nosrcmask_NumCol5_full_NEP_iter400_stitched', iterations=400)
RUN = dict(jobs=spherex.window('Aromatic_PAHfit', subchannels=range(200, 260)),
           tiles=sc.Tiles((2, 2), overlap=50, names=('NW', 'NE', 'SW', 'SE'),
                          tile_name='damp0p1_reg0p1_applyWt_PAHfit_dampL5e-3_subch60_nosrcmask_NumCol5_full_{tile}_iter400_subchPoly3_w100_outThresh5_sigma2_polyK1',
                          stitched_name='{name}'))

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
