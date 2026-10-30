"""SEP Detector 4 on the expanded field (the 2026W29 data plus one ring of the all-sky survey): the three-line
PAH fit of the NEP expansion (runs/multiline_nep_expansion.py, its recipe itself: the same model, fit and numerics),
one pass, on eight tiles balanced by frame count.

The model: a constant sky plus the aromatic, aliphatic and plateau lines (the shipped templates), all undamped;
two offset terms at NumCol 1, a degree-2 Chebyshev polynomial of the position across the LVF arcs over the
window per frame (``terms.spectral_polynomial``; its constant holds the frame's level) and a free gradient along
detector x per frame and subchannel (``terms.frame_slope``); no damping or
regularisation of any kind; shot-noise weights, the source-mask bit ignored, a 5-sigma clip; the default solver
to convergence, at most 400 iterations; no mosaic. The tiles are solved one at a time, then stitched; ``ONLY``
limits the run to the tiles listed (no stitch: a later run of every tile reuses their cals and stitches them).

The tiles: the x split at the median frame centre, each column's y splits at the quartiles of its frames' centre
rows (workspace/survey-expansion/scripts/sep_expansion_tiles.py), 3,346-3,356 frames each -- eight rather than six
so that a tile holds fewer frames than the NEP expansion's (~4,240, whose solves peaked at 241 GB). A tile's frames
and its matrix build's spill go to the NVMe (``site.NVME``): ~0.11 GB of spill per frame, ~0.37 TB per tile, and
the staged frames stay there (~0.5 TB for the field by the last tile), so check the free space before a run.
"""
import selfcal as sc
from selfcal_scripts.spherex import jobs
from selfcal_scripts.recipes.site import ORCA, OUTPUTS
from selfcal_scripts.recipes.spherex import MULTILINE3_WINDOW
from selfcal_scripts.runs.multiline_nep_expansion import RECIPE as NEP_EXPANSION_RECIPE

#: The expanded SEP canvas (17946 x 17942 pixels, 26,810 frames) in eight tiles of ~3,350 frames each.
SEP_EXPANSION8_TILES = {
    'S01': (0, 7896, 0, 9222), 'S02': (7896, 9736, 0, 9222), 'S03': (9736, 10705, 0, 9222),
    'S04': (10705, 17946, 0, 9222), 'S05': (0, 7856, 9222, 17942), 'S06': (7856, 9319, 9222, 17942),
    'S07': (9319, 9964, 9222, 17942), 'S08': (9964, 17946, 9222, 17942)}

#: The tiles this run solves (None: every tile, then the stitch).
ONLY = None

FIELD = sc.Field(f'{OUTPUTS}/SPHEREx_SEP_expansion_2026W29_D4_6p2arcsec', sc.SPHEREx(4), 6.2,
                 compute=ORCA.replace(memory_guard=True))
RECIPE = NEP_EXPANSION_RECIPE.replace(
    name='multiline3_xslope_SEPexp8_STITCHED_conv400_cheb2_NumCol1_outThresh5_nodamp_noreg')
RUN = dict(jobs=jobs.window('Multiline3', MULTILINE3_WINDOW),
           tiles=sc.Tiles(boxes=SEP_EXPANSION8_TILES, only=ONLY,
                          tile_name='multiline3_xslope_SEPexp8_{tile}_conv400_cheb2_NumCol1_outThresh5_nodamp_noreg',
                          stitched_name='{name}'))

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
