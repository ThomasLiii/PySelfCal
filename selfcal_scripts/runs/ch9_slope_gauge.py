"""Channel 9 of the NEP 2026W17 field with the 2026-10 channel recipe (``SLOPE_GAUGE``: the per-frame gradient
model, shot-noise weights floored at the frame's median, deflated CGLS to the chi-squared stop, the post-solve
plane gauge on the map, the weighted coadd with its mean, std and 2-sigma clipped mean), NumCol 1 with 17
channels -- the library form of the workspace runs that chose the recipe (workspace/lsqr-convergence).

    selfcal run selfcal_scripts/runs/ch9_slope_gauge.py            # Detectors 1 to 6, real frames
    selfcal run selfcal_scripts/runs/ch9_slope_gauge.py 6          # one detector
    selfcal plan selfcal_scripts/runs/ch9_slope_gauge.py

A simulated scene of the transfer-function campaign (its ``SB`` frames, reprojected on the detector's NEP grid)
is solved and coadded the same way from a field directory holding that grid's ``ref.fits``::

    python selfcal_scripts/runs/ch9_slope_gauge.py 6 --scene /path/to/D6_sim000001_<hash>/SB --field /path/to/field

(the mosaic is then the input of the campaign's estimator, ``measure_scene.py``).
"""
import argparse
import os

import selfcal as sc
from selfcal.instruments import spherex
from selfcal_scripts.recipes.site import NVME
from selfcal_scripts.recipes.spherex import SLOPE_GAUGE, nep

#: Ch9 is solved on 17 channels and one column (the chunk axes of the slope model).
FIELDS = [nep(detector, num_col=1, num_ch=17, compute=sc.Compute(NVME, workers=30, coadd_workers=32))
          for detector in range(1, 7)]
RECIPE = SLOPE_GAUGE
RUN = dict(jobs=spherex.channel(9))


def scene_field(detector, field_dir, scratch=None):
    """A field on a scene's grid (``field_dir/ref.fits``) for ``detector``, the scene's frames given at the action."""
    compute = sc.Compute(scratch or os.path.join(NVME, 'scene_scratch'), workers=30, coadd_workers=32, stage=None)
    return sc.Field(field_dir, sc.SPHEREx(detector, num_col=1, num_ch=17), 6.2, compute=compute)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('detectors', nargs='*', type=int, help='detectors (default: 1 to 6)')
    ap.add_argument('--scene', help='a scene component directory (its frames) instead of the real frames')
    ap.add_argument('--field', help='the field directory of a scene run (holds ref.fits)')
    ap.add_argument('--scratch', help='fast scratch for a scene run')
    a = ap.parse_args()
    chosen = a.detectors or list(range(1, 7))
    if a.scene:
        if len(chosen) != 1 or not a.field:
            raise SystemExit('a scene run takes one detector and --field')
        print(scene_field(chosen[0], a.field, a.scratch).calibrate(RECIPE, frames=sc.frames_in(a.scene), **RUN))
    else:
        for field in FIELDS:
            if field.instrument.detector in chosen:
                print(field.calibrate(RECIPE, **RUN))
