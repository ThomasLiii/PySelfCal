"""SEP Detector 4 (the /data3 copy) at NumCol 5: the tiled PAH 3.29 um fit, 4x4 tiles overlapping by 50 pixels,
200 iterations, stitched (was configs/tiled_sep_4x4.toml)."""
import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import ORCA
from selfcal_scripts.recipes.spherex import pahfit_tiled, sep

FIELD = sep(4, num_col=5, compute=ORCA.replace(workers=24, stage_dir='reproj_nvme_pahfit_sep_4x4', memory_guard=True),
              outputs='/data3/thomasli/selfcal/outputs')
RECIPE = pahfit_tiled('damp0p1_reg0p1_applyWt_PAHfit_dampL5e-3_subch60_nosrcmask_NumCol5_sep4x4_SEP_iter200_stitched', iterations=200)
RECIPE = RECIPE.replace(numerics=sc.Numerics(24))
RUN = dict(jobs=spherex.window('Aromatic_PAHfit', subchannels=range(200, 260)),
           tiles=sc.Tiles((4, 4), overlap=50, names=tuple(f'r{r}c{c}' for r in range(4) for c in range(4)),
                          tile_name='damp0p1_reg0p1_applyWt_PAHfit_dampL5e-3_subch60_nosrcmask_NumCol5_sep4x4_{tile}_iter200_subchPoly3_w100_outThresh5_sigma2_polyK1',
                          stitched_name='{name}'))

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
