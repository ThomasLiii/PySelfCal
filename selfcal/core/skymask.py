"""Sky pixels the frames flag too often, dropped from every frame's rows (``Fit(sky_mask=...)``).

A data-quality bit that marks a feature fixed on the sky (a source mask, say) lands slightly differently in each
frame: pointing, rotation and the point-spread function move its edge. A sky pixel near the feature is then flagged
by some frames and not by others, and the frames that do not flag it bring the feature's flux into the solve. With
:class:`~selfcal.run.recipe.SkyMask` the setup first counts, per pixel of the reference grid, the frames whose valid
pixels cover it and how many of those set one of the bits, and drops the pixel from every frame when the flagged
fraction is above a threshold: a fixed mask on the sky, decided from the frames' own flags before the solve. Any
number of rules, each a set of bits with its own threshold, in one read of the frames.

It reads only what every frame file has (values, bit mask, grid box; :mod:`selfcal.io.frames`), so it is the same for
any instrument; an instrument whose frames carry no such bit simply does not use it.
"""
from __future__ import annotations

import numpy as np

from ..io.reproj import load_reproj_file
from .shmbuf import worker_pool_context


def _detector_valid(det_valid, mapping):
    """Where the detector-grid weight ``det_valid`` (bool) interpolates to a positive value at the detector
    coordinates ``mapping = (x, y)`` of a frame's pixels (bilinear, as the setup samples the grid valid weight;
    False off the detector)."""
    x, y = mapping[0], mapping[1]
    out = np.zeros(x.shape, dtype=bool)
    ok = np.isfinite(x) & np.isfinite(y)
    xs, ys = x[ok], y[ok]
    x0, y0 = np.floor(xs).astype(np.int64), np.floor(ys).astype(np.int64)
    fx, fy = xs - x0, ys - y0
    H, W = det_valid.shape
    hit = np.zeros(xs.shape, dtype=bool)
    for dy, wy in ((0, 1 - fy), (1, fy)):
        for dx, wx in ((0, 1 - fx), (1, fx)):
            yy, xx = y0 + dy, x0 + dx
            inside = (yy >= 0) & (yy < H) & (xx >= 0) & (xx < W) & (wy * wx > 0)
            hit[inside] |= det_valid[yy[inside], xx[inside]]
    out[ok] = hit
    return out


def _flag_counts(args):
    """Per grid pixel: the frames whose valid pixels cover it, and, per rule, those of them that set one of the
    rule's bits."""
    files, ref_shape, rule_bits, masking, det_valid = args
    H, W = ref_shape
    dtype = np.uint16 if len(files) < 65535 else np.uint32
    hits = np.zeros(ref_shape, dtype)
    flagged = np.zeros((len(rule_bits),) + tuple(ref_shape), dtype)
    for f in files:
        r = load_reproj_file(f, fields=['sub_data', 'ref_coords', 'sub_bitmask']
                             + (['sub_mapping'] if det_valid is not None else []))
        y0, y1, x0, x1 = (int(v) for v in r['ref_coords'])
        gy0, gy1, gx0, gx1 = max(y0, 0), min(y1, H), max(x0, 0), min(x1, W)       # the box on the grid
        if gy1 <= gy0 or gx1 <= gx0:
            continue
        box = (slice(gy0 - y0, gy1 - y0), slice(gx0 - x0, gx1 - x0))
        bm = r['sub_bitmask'][box]
        bm = bm.view(np.dtype(f'u{bm.dtype.itemsize}'))           # the bits, as unsigned (the top bit included)
        top = np.iinfo(bm.dtype).max
        # valid as the solve sees it, apart from the rules' bits: finite, no other masking bit set
        valid = np.isfinite(r['sub_data'][box]) & ((bm & bm.dtype.type(masking & top)) == 0)
        if det_valid is not None:                                  # the detector region the solve uses
            valid &= _detector_valid(det_valid, r['sub_mapping'][(slice(None),) + box])
        hits[gy0:gy1, gx0:gx1] += valid
        for k, bits in enumerate(rule_bits):
            flagged[k, gy0:gy1, gx0:gx1] += valid & ((bm & bm.dtype.type(bits & top)) != 0)
    return hits, flagged


def flagged_fraction_mask(files, ref_shape, rules, ignore_list=(), max_workers=1, grid_valid_weight=None):
    """The reference-grid pixels to drop from every frame, for ``rules = [(flags, above), ...]``: a pixel is dropped
    when, for any rule, its covering frames set one of the rule's ``flags`` in more than a fraction ``above`` of them
    (``above=0``: flagged by any frame).

    A frame covers a grid pixel where its value is finite, no masking bit outside the rules' bits is set (the bits
    of ``ignore_list`` never mask) and, with ``grid_valid_weight`` (the setup's detector-grid valid weight), that
    weight is positive at the pixel's detector coordinates: the frames' pixels the solve would use. One read of every frame for all the rules. Returns ``(drop, hits, flagged)``: the
    boolean mask, the covering counts and the flagged counts per rule (``(len(rules), H, W)``, uint32).
    """
    ref_shape = (int(ref_shape[0]), int(ref_shape[1]))
    rule_bits, aboves, all_bits = [], [], set()
    for flags, above in rules:
        flags = {int(b) for b in flags}
        if not flags or any(b < 0 or b > 63 for b in flags):
            raise ValueError(f"flagged_fraction_mask: flags {sorted(flags)} must be bit numbers 0..63")
        if not 0.0 <= float(above) < 1.0:
            raise ValueError(f"flagged_fraction_mask: above={above} must be in [0, 1)")
        bits = 0
        for b in flags:
            bits |= 1 << b
        rule_bits.append(bits)
        aboves.append(float(above))
        all_bits |= flags
    if not rule_bits:
        raise ValueError("flagged_fraction_mask: no rules given")
    masking = (1 << 64) - 1                                    # every bit masks but the ignored ones and the rules'
    for b in {int(b) for b in ignore_list} | all_bits:
        masking &= ~(1 << b)
    files = list(files)
    n = max(1, min(int(max_workers), len(files)))              # each worker holds 1 + len(rules) grid-sized counts
    det_valid = None if grid_valid_weight is None else np.asarray(grid_valid_weight) > 0
    tasks = [(files[i::2 * n], ref_shape, rule_bits, masking, det_valid) for i in range(min(2 * n, len(files)))]
    hits = np.zeros(ref_shape, np.uint32)
    flagged = np.zeros((len(rule_bits),) + ref_shape, np.uint32)
    if n == 1:
        for h, f in map(_flag_counts, tasks):
            hits += h
            flagged += f
    else:
        with worker_pool_context().Pool(n) as pool:
            for h, f in pool.imap_unordered(_flag_counts, tasks):
                hits += h
                flagged += f
    drop = np.zeros(ref_shape, dtype=bool)
    for k, above in enumerate(aboves):
        drop |= (hits > 0) & (flagged[k] > above * hits)
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
