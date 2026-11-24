"""Channel maps of the final channel recipe on the NEP expanded fields (the 2026W29 data plus one ring of the all-sky
survey, about 25,500 frames per detector on a 16,631 x 16,633 grid): ``pixels_final(c, num_ch=n)`` for channel ``c`` of
a detector's ``n`` channels (17 or 34), each the job ``jobs.channel(c, num_ch=n)``: its cal and its coadded mosaic. The
recipe and the jobs are those of final_channel_maps.py, which makes the same maps on the NEP 2026W17 fields.

    selfcal run selfcal_scripts/runs/final_channel_maps_expansion.py 4 34 17                   # Detector 4, Ch17 of 34
    selfcal run selfcal_scripts/runs/final_channel_maps_expansion.py 4 34 17 --frames DIR      # frames read from DIR
    selfcal run selfcal_scripts/runs/final_channel_maps_expansion.py 4 34 17 --plan           # what it would make

A map that exists and was made by this recipe is reused. With --frames the frames are read from that directory (the
field's own reprojected/ on the RAID, or a copy of it on fast disk); without it, the field's compute settings stage them.
FIELD, RECIPE and RUN at the top level are one channel's run (Detector 4, Ch17 of 34), what ``selfcal plan`` of this
script shows.
"""
import argparse

import selfcal as sc
from selfcal_scripts.recipes.site import ORCA, OUTPUTS
from selfcal_scripts.recipes.spherex import pixels_final
from selfcal_scripts.spherex import jobs


def nep_expansion(detector, compute=ORCA, outputs=OUTPUTS):
    """The NEP expanded field of ``detector`` (2026W29 data plus one ring), 6.2" pixels."""
    return sc.Field(f'{outputs}/SPHEREx_NEP_expansion_2026W29_D{detector}_6p2arcsec', sc.SPHEREx(detector), 6.2,
                    compute=compute)


FIELD = nep_expansion(4)
RECIPE = pixels_final(17, num_ch=34)
RUN = dict(jobs=jobs.channel(17, num_ch=34))


def channel_maps(detector, num_ch, channels=(), frames=None, plan=False):
    """Make, or with ``plan`` print the plan of, the maps of ``channels`` (all ``num_ch`` when empty) of ``detector``."""
    field = nep_expansion(detector)
    for c in channels or range(1, num_ch + 1):
        recipe, job = pixels_final(c, num_ch=num_ch), jobs.channel(c, num_ch=num_ch)
        print(field.plan(recipe, jobs=job, frames=frames) if plan else field.calibrate(recipe, jobs=job, frames=frames))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('detector', type=int)
    ap.add_argument('num_ch', type=int, choices=(17, 34))
    ap.add_argument('channels', type=int, nargs='*')
    ap.add_argument('--frames', default=None)
    ap.add_argument('--plan', action='store_true')
    a = ap.parse_args()
    channel_maps(a.detector, a.num_ch, a.channels, a.frames, a.plan)
