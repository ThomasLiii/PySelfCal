"""Real channel maps of the final channel recipe on the NEP 2026W17 fields: ``pixels_final(c, num_ch=n)`` for each
channel ``c`` of a detector's ``n`` channels (17 or 34), each the job ``jobs.channel(c)``: its cal and its coadded
mosaic. The recipe is made for each channel, since its spectral term, its per-pixel constants and its fit window
follow the channel's subchannels.

    selfcal run selfcal_scripts/runs/final_channel_maps.py 1 17                   # Detector 1, Ch1-17 of 17
    selfcal run selfcal_scripts/runs/final_channel_maps.py 1 34 5 6 --frames DIR  # Ch5 and Ch6 of 34, frames from DIR
    selfcal run selfcal_scripts/runs/final_channel_maps.py 1 34 5 --plan          # what Ch5 of 34 would make

A map that exists and was made by this recipe is reused, so a stopped run continues where it stopped. With --frames
the frames are read from that directory (a copy of the field's reprojected frames already on fast disk); without it,
the field's compute settings stage them for each channel. FIELD, RECIPE and RUN at the top level are one
channel's run (Detector 1, Ch9 of 17), what ``selfcal plan`` of this script shows.
"""
import argparse

from selfcal_scripts.recipes.spherex import nep, pixels_final
from selfcal_scripts.spherex import jobs

FIELD = nep(1)
RECIPE = pixels_final(9, num_ch=17)
RUN = dict(jobs=jobs.channel(9))


def channel_maps(detector, num_ch, channels=(), frames=None, plan=False):
    """Make, or with ``plan`` print the plan of, the maps of ``channels`` (all ``num_ch`` when empty) of ``detector``."""
    field = nep(detector)
    for c in channels or range(1, num_ch + 1):
        recipe, job = pixels_final(c, num_ch=num_ch), jobs.channel(c)
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
