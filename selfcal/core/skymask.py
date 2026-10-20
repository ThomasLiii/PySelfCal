"""Sky pixels the frames flag too often, dropped from every frame's rows (``Fit(sky_mask=...)``).

A data-quality bit that marks a feature fixed on the sky (a source mask, say) lands slightly differently in each
frame: pointing, rotation and the point-spread function move its edge. A sky pixel near the feature is then flagged
by some frames and not by others, and the frames that do not flag it bring the feature's flux into the solve. With
:class:`~selfcal.run.recipe.SkyMask` the setup first counts, per pixel of the reference grid, the frames whose valid
pixels cover it and how many of those set one of the bits, and drops the pixel from every frame when the flagged
fraction is above a threshold: a fixed mask on the sky, decided from the frames' own flags before the solve.

It reads only what every frame file has (values, bit mask, grid box; :mod:`selfcal.io.frames`), so it is the same for
any instrument; an instrument whose frames carry no such bit simply does not use it.
"""
from __future__ import annotations

import numpy as np

from ..geometry.map_helper import bit_to_bool
from ..io.reproj import load_reproj_file
from .shmbuf import worker_pool_context


def _flag_counts(args):
    """Per grid pixel: the frames whose valid pixels cover it, and those of them that set one of the bits."""
    files, ref_shape, bits, not_masking = args
    H, W = ref_shape
    dtype = np.uint16 if len(files) < 65535 else np.uint32
    hits = np.zeros(ref_shape, dtype)
    flagged = np.zeros(ref_shape, dtype)
    for f in files:
        r = load_reproj_file(f, fields=['sub_data', 'ref_coords', 'sub_bitmask'])
        y0, y1, x0, x1 = (int(v) for v in r['ref_coords'])
        gy0, gy1, gx0, gx1 = max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)       # the box on the grid
        if gy1 <= gy0 or gx1 <= gx0:
            continue
        box = (slice(gy0 - y0, gy1 - y0), slice(gx0 - x0, gx1 - x0))
        bm = np.asarray(r['sub_bitmask'][box], dtype=np.int64)
        # valid as the solve sees it, apart from the bits themselves: finite, no other masking bit set
        valid = np.isfinite(r['sub_data'][box]) & bit_to_bool(bm, not_masking, invert=True)
        hits[gy0:gy1, gx0:gx1] += valid
        flagged[gy0:gy1, gx0:gx1] += valid & ((bm & bits) != 0)
    return hits, flagged


def flagged_fraction_mask(files, ref_shape, flags, above, ignore_list=(), max_workers=1):
    """The reference-grid pixels to drop from every frame: those whose covering frames set one of ``flags`` in more
    than a fraction ``above`` of their valid pixels there (``above=0``: flagged by any frame).

    A frame covers a grid pixel where its value is finite and no masking bit other than ``flags`` is set (the bits of
    ``ignore_list`` never mask). Returns ``(drop, hits, flagged)``: the boolean mask and the two counts (uint32).
    """
    ref_shape = (int(ref_shape[0]), int(ref_shape[1]))
    flags = sorted({int(b) for b in flags})
    if not flags:
        raise ValueError("flagged_fraction_mask: no flag bits given")
    if not 0.0 <= float(above) < 1.0:
        raise ValueError(f"flagged_fraction_mask: above={above} must be in [0, 1)")
    bits = 0
    for b in flags:
        bits |= 1 << b
    not_masking = sorted({int(b) for b in ignore_list} | set(flags))
    files = list(files)
    n = max(1, min(int(max_workers), 16, len(files)))          # each worker holds two grid-sized count arrays
    tasks = [(files[i::2 * n], ref_shape, bits, not_masking) for i in range(min(2 * n, len(files)))]
    hits = np.zeros(ref_shape, np.uint32)
    flagged = np.zeros(ref_shape, np.uint32)
    if n == 1:
        results = map(_flag_counts, tasks)
        for h, f in results:
            hits += h
            flagged += f
    else:
        with worker_pool_context().Pool(n) as pool:
            for h, f in pool.imap_unordered(_flag_counts, tasks):
                hits += h
                flagged += f
    drop = (hits > 0) & (flagged > float(above) * hits)
    return drop, hits, flagged


def grid_window(mask, ref_coords, shape):
    """``mask`` (reference grid) on a frame's box ``ref_coords = [y0, y1, x0, x1]`` of shape ``shape``; False where
    the box runs past the grid."""
    H, W = mask.shape
    y0, y1, x0, x1 = (int(v) for v in ref_coords)
    out = np.zeros(shape, dtype=bool)
    gy0, gy1, gx0, gx1 = max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)
    if gy1 > gy0 and gx1 > gx0:
        out[gy0 - y0:gy1 - y0, gx0 - x0:gx1 - x0] = mask[gy0:gy1, gx0:gx1]
    return out
