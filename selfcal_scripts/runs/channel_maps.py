"""The fiducial channel maps of the NEP 2026W17 field (the recipe since 2026-10-14: ``CHANNEL_MAPS``, the
transfer function's recipe with the wavelength maps), solved on 17 channels and one column: every channel of
every detector but Ch9, 96 maps. Ch9 of Detectors 1 to 6 are the transfer-function campaign's own maps of this
recipe, delivered on 2026-10-14 with their wavelength maps added (``workspace/DATA_LOCATIONS.md``); their
products have no record of this script, which would refuse them.

    selfcal run selfcal_scripts/runs/channel_maps.py           # Detectors 1 to 6
    selfcal run selfcal_scripts/runs/channel_maps.py 5 6       # some detectors
    selfcal plan selfcal_scripts/runs/channel_maps.py          # what each detector would make

A detector is one action: its frames are staged to fast disk once, its maps are made one after the
other (each channel's cal, gauged, then its mosaic) and the stage is removed. A map that exists and was
made by this recipe is reused, so a stopped campaign continues where it stopped. Each cal records how its
solve ended (``CalFile(cal).solve``: the chi-squared rule, or ``"none"`` at the iteration cap).
"""
import sys

from selfcal_scripts.recipes.spherex import CHANNEL_MAPS, nep
from selfcal_scripts.spherex import jobs

FIELDS = [nep(detector) for detector in range(1, 7)]
RECIPE = CHANNEL_MAPS
#: Ch9 is the campaign's (see above).
RUN = dict(jobs=jobs.channels([c for c in range(1, 18) if c != 9]))

if __name__ == '__main__':
    chosen = [int(a) for a in sys.argv[1:]] or range(1, 7)
    for field in FIELDS:
        if field.instrument.detector in chosen:
            print(field.calibrate(RECIPE, **RUN))
