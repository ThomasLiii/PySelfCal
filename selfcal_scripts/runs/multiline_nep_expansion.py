"""NEP Detector 4 on the expanded field (the 2026W29 data plus one ring of the all-sky survey: 25,421 frames on the
16,631 x 16,633 grid): the three-line PAH fit over the Multiline3 window, the whole field in one solve, with the
data-quality flags split between the offsets and the sky.

The recipe is :func:`selfcal_scripts.recipes.spherex.multiline3_flag_split` (a constant sky and the aromatic,
aliphatic and plateau lines, the spectral polynomial and the x-slope offsets, no damping, the listed bad detector
positions given no weight), solved in the two passes of
:data:`~selfcal_scripts.recipes.spherex.FLAG_SPLIT_PASSES`: pass 1 fits the sky terms and the offsets jointly on the
observations that carry no flag; pass 2 fits the sky terms alone on the observations with no flag but the source bit,
pass 1's offsets subtracted. The pass-2 cal's sky terms are the maps (no mosaic).

    ./selfcal_scripts/run.sh selfcal_scripts/runs/multiline_nep_expansion.py [--dry-run]
    ./selfcal_scripts/run.sh selfcal_scripts/runs/multiline_nep_expansion.py --first-pass-only [--snapshots N]

``--first-pass-only``: pass 1 alone, as a plain calibration that writes its solution every ``N`` iterations as a cal
file (``sc.Snapshots``), so that a stopped solve can be continued from its last snapshot. The command without it then
reuses that cal and runs pass 2.

Measured on this field (2026-11-15 to 17; 24 workers, 40 threads, the compact matrix): pass 1 had 10.78 billion
observations and 376.75 million unknowns; its setup took 2,978 s and peaked at 600 GiB in the deflation's setup (the
peak grows by ~5.5 GiB per thread), and its 600 iterations ran to the cap (~29 h in all). Pass 2 took 13,852 s. The
frames are staged to the NVMe (``site.NVME``, ~800 GB), and pass 2's matrix build spills there too (projected
~1.3 TB from the 911 GB of the 17,647-frame W17 field), so check the free space before a run.
"""
import argparse

import selfcal as sc
from selfcal_scripts.spherex import jobs
from selfcal_scripts.recipes.site import ORCA, OUTPUTS
from selfcal_scripts.recipes.spherex import (FLAG_SPLIT_PASSES, MULTILINE3_WINDOW, flag_split_first_pass,
                                             multiline3_flag_split)

FIELD = sc.Field(f'{OUTPUTS}/SPHEREx_NEP_expansion_2026W29_D4_6p2arcsec', sc.SPHEREx(4), 6.2,
                 compute=ORCA.replace(workers=24, matrix='compact', memory_guard=True))
RECIPE = multiline3_flag_split()
RUN = dict(jobs=jobs.window('Multiline3', MULTILINE3_WINDOW), passes=FLAG_SPLIT_PASSES)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--first-pass-only', action='store_true')
    ap.add_argument('--snapshots', type=int, default=0, help='with --first-pass-only: every this many iterations')
    a = ap.parse_args()
    if a.first_pass_only:
        snapshots = sc.Snapshots(every=a.snapshots) if a.snapshots else None
        print(FIELD.calibrate(flag_split_first_pass(RECIPE), jobs=RUN['jobs'], snapshots=snapshots))
    else:
        if a.snapshots:
            ap.error('--snapshots goes with --first-pass-only (a pass schedule writes none)')
        print(FIELD.calibrate(RECIPE, **RUN))
