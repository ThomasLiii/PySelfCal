"""Detector positions that read impossible values while carrying no flag the fit respects, and the observation weight
that leaves them out of a solve.

Some detector positions read values far outside the sky's range (-100 to -1,600 MJy/sr, against a sky of +0.05 to
+0.15 and a noise of 0.01 to 0.03) frame after frame, flagged with the source-mask bit only or not at all. A recipe that
reads source-flagged pixels (``Fit(ignore_flags=[21])``) then lets each such pixel outweigh the thousands of other
observations of its frame and subchannel, and the frame's offsets tilt along the band.

The list of a detector, ``data/bad_pixels_d<detector>.json``, holds those positions as ``(x, y)`` pairs on the
2040 x 2040 detector, with the rule that selected them and the counts behind it. The Detector 4 list was measured
on 2026-11-13 by a census of the 17,647 reprojected frames of the NEP 2026W17 field: the positions more than 50 sigma
from their frame's median in more than 1 % of the frames that read them and in at least 5 frames, or below -1 MJy/sr in
at least 5 frames (4,464 positions). A floor on the stored value (-1 MJy/sr in the recipes) catches the impossible
values of positions that show them in fewer frames than the list requires.

    from selfcal_scripts.spherex import bad_pixels

    weight = sc.Variance(bad_pixels.variance_without, of=('variance', 'det_x', 'det_y', 'value'),
                         pixels=bad_pixels.load(4), floor=-1.0)

``sc.Variance`` drops an observation whose variance is not finite, so :func:`variance_without` returns NaN for the
observations whose detector position rounds to a listed pixel or whose value is below the floor, and the stored
variance for every other one.
"""
import functools
import json
import os

import numpy as np

__all__ = ['DETECTOR_SHAPE', 'list_path', 'load', 'variance_without']

#: The detector's size in pixels, (y, x).
DETECTOR_SHAPE = (2040, 2040)

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')


def list_path(detector):
    """The path of ``detector``'s list of positions."""
    return os.path.join(_DATA, f'bad_pixels_d{int(detector)}.json')


def load(detector):
    """``detector``'s listed positions as a tuple of ``(x, y)`` pairs (hashable, so it can be a recipe parameter)."""
    with open(list_path(detector)) as f:
        return tuple((int(x), int(y)) for x, y in json.load(f)['pixels'])


@functools.lru_cache(maxsize=4)
def _mask(pixels):
    m = np.zeros(DETECTOR_SHAPE, dtype=bool)
    p = np.asarray(pixels, dtype=np.int64).reshape(-1, 2)
    m[p[:, 1], p[:, 0]] = True
    return m


def variance_without(variance, det_x, det_y, value=None, pixels=(), floor=None):
    """``variance``, with NaN where the observation's detector position (rounded) is one of ``pixels`` or its stored
    ``value`` is below ``floor`` (None: no floor)."""
    out = np.array(variance, dtype=np.float64, copy=True)
    xi = np.rint(np.asarray(det_x, dtype=np.float64))
    yi = np.rint(np.asarray(det_y, dtype=np.float64))
    ok = (np.isfinite(xi) & np.isfinite(yi) & (xi >= 0) & (xi < DETECTOR_SHAPE[1])
          & (yi >= 0) & (yi < DETECTOR_SHAPE[0]))
    bad = np.zeros(out.shape, dtype=bool)
    pixels = pixels if isinstance(pixels, tuple) else tuple(map(tuple, pixels))
    bad[ok] = _mask(pixels)[yi[ok].astype(np.int64), xi[ok].astype(np.int64)]
    if floor is not None:
        bad |= np.asarray(value, dtype=np.float64) < float(floor)
    out[bad] = np.nan
    return out
