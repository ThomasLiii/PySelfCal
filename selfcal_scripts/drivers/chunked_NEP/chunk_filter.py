"""Back-compat shim: the frame-selection helpers moved to the package at
``selfcal.io.frame_select`` (reusable beyond the NEP quadrant build). Import
from there going forward; this re-export keeps the existing quadrant drivers
working unchanged.
"""
from selfcal.io.frame_select import (  # noqa: F401
    compute_overlapping_frames,
    compute_overlapping_frames_from_cache,
    load_ref_coords_table,
    filter_by_center,
)
