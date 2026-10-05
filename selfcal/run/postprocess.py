"""Optional per-subframe postprocess hooks (named, selectable from config).

These run inside the calibration subframe loop; ``setup_lsqr(postprocess_func=)``
gets the resolved callable, which receives a
:class:`selfcal.core.subframe.FrameContext` and returns the new ``sub_data``. Off by default — a run gets a postprocess only when
its config names one, and no production config does. Kept here so a config can
opt in by name without editing Python.
"""
import numpy as np


def mask_bright_pixels(ctx):
    """NaN out pixels above the 25th percentile of the valid (weight>0) data in a
    subframe. The 25th-percentile threshold and in-place NaN masking are
    intentional and must not change: any run that opts in relies on this exact
    behavior for comparable outputs."""
    sub_data = ctx.sub_data
    sub_weight = ctx.sub_weight

    valid_mask = sub_weight > 0
    if np.sum(valid_mask) > 0:
        threshold = np.nanpercentile(sub_data[valid_mask], 25)
        sub_data[sub_data > threshold] = np.nan

    return sub_data
