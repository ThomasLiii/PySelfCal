"""SEP Detector 4 on the expanded field (the 2026W29 data plus one ring of the all-sky survey: 26,810 frames on the
17,946 x 17,942 grid): the three-line PAH fit with a slope sky term over the Multiline3 window, the whole field in one
solve, in one pass on every observation with no flag but the source bit.

The recipe is :func:`selfcal_scripts.recipes.spherex.multiline3_slope` (a constant sky, a slope of the wavelength and
the aromatic, aliphatic and plateau lines; the spectral polynomial and the x-slope offsets; no damping; the listed bad
detector positions given no weight), 350 iterations. The cal's sky terms are the maps (no mosaic).

    ./selfcal_scripts/run.sh selfcal_scripts/runs/multiline_sep_expansion_slope.py [--dry-run] [--snapshots N]

``--snapshots N``: write the solution every ``N`` iterations as a cal file (``sc.Snapshots``), so that a stopped solve
can be continued from its last snapshot.

Measured on this field (2026-11-21 to 23; 24 workers, 32 threads, the compact matrix, the memory guard; export of
branch row-memory 81dafb4): 29.95 billion observations and 1.088 billion unknowns; the setup took 10,594 s, the row
regroup peaked at 603 GiB and the deflation's setup at 617.6 GiB, and the iterations held 525 GiB; iterations 1 to 350
ran from 2026-11-21 21:13:47 to 2026-11-23 22:36:47. That fit ran with 600 iterations and snapshots every 50, and was
stopped after its iteration-350 snapshot; that snapshot is the field's cal of this recipe (recorded with
``selfcal adopt``). A launch from main 4217065, without the row-memory branch's fixes, was killed for memory in the row
regroup (main process at 701 GiB). The frames are staged to the NVMe (``site.NVME``, 857 GiB), so check the free space
before a run.
"""
import argparse

import selfcal as sc
from selfcal_scripts.spherex import jobs
from selfcal_scripts.recipes.site import ORCA, OUTPUTS
from selfcal_scripts.recipes.spherex import MULTILINE3_WINDOW, multiline3_slope

FIELD = sc.Field(f'{OUTPUTS}/SPHEREx_SEP_expansion_2026W29_D4_6p2arcsec', sc.SPHEREx(4), 6.2,
                 compute=ORCA.replace(workers=24, matrix='compact', memory_guard=True))
RECIPE = multiline3_slope()
RUN = dict(jobs=jobs.window('Multiline3', MULTILINE3_WINDOW))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshots', type=int, default=0, help='write the solution every this many iterations')
    a = ap.parse_args()
    snapshots = sc.Snapshots(every=a.snapshots) if a.snapshots else None
    print(FIELD.calibrate(RECIPE, **RUN, snapshots=snapshots))
