"""The NumCol3 channel maps of the NEP 2026W17 field (the recipe since 2026-09-29): every channel of
every detector, 204 maps.

    selfcal run selfcal_scripts/runs/numcol3_maps.py           # Detectors 1 to 6
    selfcal run selfcal_scripts/runs/numcol3_maps.py 5 6       # some detectors
    selfcal plan selfcal_scripts/runs/numcol3_maps.py          # what each detector would make

A detector is one action: its frames are staged to fast disk once, its maps are made one after the
other (each channel's cal, then its mosaic) and the stage is removed. A map that exists and was made
by this recipe is reused, so a stopped campaign continues where it stopped. The maps of the TOML
production (2026-09-29) have no records and are refused until adopted (``selfcal adopt`` this
script checks each and records it).
"""
import sys

from selfcal.instruments import spherex
from selfcal_scripts.recipes.spherex import NUMCOL3, nep

FIELDS = [nep(detector) for detector in range(1, 7)]
RECIPE = NUMCOL3
RUN = dict(jobs=spherex.channels(1, 34))

if __name__ == '__main__':
    chosen = [int(a) for a in sys.argv[1:]] or range(1, 7)
    for field in FIELDS:
        if field.instrument.detector in chosen:
            print(field.calibrate(RECIPE, **RUN))
