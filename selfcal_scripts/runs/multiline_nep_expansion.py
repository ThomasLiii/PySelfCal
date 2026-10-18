"""NEP Detector 4 on the expanded field (the 2026W29 data plus one ring of the all-sky survey): the three-line
PAH fit over the Multiline3 window with a free x-slope offset term, on six tiles balanced by frame count.

The model: a constant sky plus the aromatic, aliphatic and plateau lines (the shipped templates), all undamped;
two offset terms at NumCol 1, the hard degree-2 polynomial along the subchannels of the window (one per frame)
and a free gradient along detector x per frame and subchannel (``spherex.frame_slope``); no damping or
regularisation of any kind; shot-noise weights, the source-mask bit ignored, a 5-sigma clip; the default solver
to convergence, at most 400 iterations; no mosaic. The tiles are solved one at a time; ``ONLY`` limits the run
to the tiles listed (no stitch: a later run of every tile reuses their cals and stitches them).

The frames of a tile and the matrix build's spill go to the NVMe (``site.NVME``): the spill peaks at ~0.11 GB per
frame (460 GB measured for tile B01's 4,236 frames), so check the free space before a run.
"""
import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import ORCA, OUTPUTS
from selfcal_scripts.recipes.spherex import MULTILINE3_WINDOW

#: The expanded NEP canvas (16631 x 16633 pixels, 25,421 frames) in six tiles of ~4,236 frames each.
NEP_EXPANSION6_TILES = {
    'B01': (0, 7797, 0, 8034), 'B02': (0, 7728, 8034, 16631), 'B03': (7797, 9757, 0, 8034),
    'B04': (7728, 9193, 8034, 16631), 'B05': (9757, 16633, 0, 8034), 'B06': (9193, 16633, 8034, 16631)}

#: The tiles this run solves (None: every tile, then the stitch).
ONLY = None

FIELD = sc.Field(f'{OUTPUTS}/SPHEREx_NEP_expansion_2026W29_D4_6p2arcsec', sc.SPHEREx(4, num_col=1), 6.2,
                 compute=ORCA.replace(keep_staged=True, memory_guard=True))
LINES = [spherex.line(name, damping=0.0) for name in ('aromatic', 'aliphatic', 'plateau')]
RECIPE = sc.Recipe(sc.Model(sky=[sc.Sky(damping=0.0), *LINES],
                            offsets=[sc.Offsets(polynomial=sc.Poly(2, window=MULTILINE3_WINDOW)),
                                     spherex.frame_slope('xslope')]),
                   fit=sc.Fit(400, clip=5.0, ignore_flags=[21], shot_noise_weights=True),
                   coadd=None, numerics=sc.Numerics(32),
                   name='multiline3_xslope_NEPexp6_STITCHED_conv400_polybasisD2_NumCol1_outThresh5_nodamp_noreg')
RUN = dict(jobs=spherex.window('Multiline3', subchannels=MULTILINE3_WINDOW),
           tiles=sc.Tiles(boxes=NEP_EXPANSION6_TILES, only=ONLY,
                          tile_name='multiline3_xslope_NEPexp6_{tile}_conv400_polybasisD2_NumCol1_outThresh5_nodamp_noreg',
                          stitched_name='{name}'))

if __name__ == '__main__':
    print(FIELD.calibrate(RECIPE, **RUN))
