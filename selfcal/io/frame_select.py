"""Spatial frame -> tile assignment for tiled (region-partitioned) calibration.

Given reprojected sub-frame HDF5 files and a tile bbox in reference-mosaic pixel
coordinates (half-open ``[y0, y1) x [x0, x1)``), select the frames belonging to
that tile -- either every frame whose footprint AABB overlaps the tile, or
(more selective, bounded per-tile memory) every frame whose footprint *center*
falls in the tile.

``ref_coords`` schema (root attr of each reprojected file, written by
``selfcal.io.reprojection``): dtype int32, shape (4,), order
``[y_min, y_max, x_min, x_max]`` (half-open). Half-open AABB overlap test:
intervals ``[a0, a1)`` and ``[b0, b1)`` overlap iff ``a0 < b1 and b0 < a1``,
applied independently on y and x.

Instrument-agnostic: depends only on the ``ref_coords`` attr, so it applies to
any region-partitioned calibration build (e.g. SPHEREx sky-field tiles or
Euclid multi-region mosaics).
"""
from __future__ import annotations

import warnings

import h5py
import numpy as np


def compute_overlapping_frames(reproj_files, chunk_bbox):
    """``(kept_files, kept_idx)`` for frames whose ref_coords AABB overlaps
    ``chunk_bbox=(y0, y1, x0, x1)`` (half-open). Reads each file's ``ref_coords``
    attr; unreadable/foreign files are skipped with a warning."""
    cy0, cy1, cx0, cx1 = chunk_bbox
    if not (cy0 < cy1 and cx0 < cx1):
        raise ValueError(
            f"chunk_bbox must be a non-empty half-open box; got "
            f"y[{cy0}, {cy1}) x[{cx0}, {cx1})")

    n = len(reproj_files)
    keep = np.zeros(n, dtype=bool)
    for i, path in enumerate(reproj_files):
        try:
            with h5py.File(path, "r") as f:
                if "ref_coords" not in f.attrs:
                    continue
                rc = np.asarray(f.attrs["ref_coords"], dtype=np.int64)
        except (OSError, KeyError) as exc:
            warnings.warn(
                f"compute_overlapping_frames: could not read ref_coords from "
                f"{path}: {exc!r}; skipping.", RuntimeWarning, stacklevel=2)
            continue
        if rc.shape != (4,):
            warnings.warn(
                f"compute_overlapping_frames: unexpected ref_coords shape "
                f"{rc.shape} in {path}; expected (4,); skipping.",
                RuntimeWarning, stacklevel=2)
            continue
        fy0, fy1, fx0, fx1 = int(rc[0]), int(rc[1]), int(rc[2]), int(rc[3])
        if (fy0 < cy1) and (cy0 < fy1) and (fx0 < cx1) and (cx0 < fx1):
            keep[i] = True
    kept_idx = np.nonzero(keep)[0].astype(np.int64)
    kept_files = [reproj_files[i] for i in kept_idx]
    return kept_files, kept_idx


def load_ref_coords_table(reproj_files):
    """Read ``ref_coords`` from every file once -> ``(N, 4)`` int64. Files lacking
    the attr / unreadable get a sentinel ``[0,0,0,0]`` (empty box -> no overlap).
    Lets the overlap/center tests run vectorized across many tiles in one pass."""
    n = len(reproj_files)
    out = np.zeros((n, 4), dtype=np.int64)
    for i, path in enumerate(reproj_files):
        try:
            with h5py.File(path, "r") as f:
                if "ref_coords" not in f.attrs:
                    continue
                rc = np.asarray(f.attrs["ref_coords"], dtype=np.int64)
                if rc.shape == (4,):
                    out[i] = rc
        except (OSError, KeyError):
            continue
    return out


def compute_overlapping_frames_from_cache(reproj_files, ref_coords_array, chunk_bbox):
    """Vectorized AABB overlap using a pre-cached ``(N, 4)`` ref_coords table
    (from :func:`load_ref_coords_table`). Returns ``(kept_files, kept_idx)``."""
    cy0, cy1, cx0, cx1 = chunk_bbox
    if not (cy0 < cy1 and cx0 < cx1):
        raise ValueError(
            f"chunk_bbox must be a non-empty half-open box; got "
            f"y[{cy0}, {cy1}) x[{cx0}, {cx1})")
    rc = np.asarray(ref_coords_array, dtype=np.int64)
    if rc.ndim != 2 or rc.shape[1] != 4:
        raise ValueError(f"ref_coords_array must have shape (N, 4); got {rc.shape}")
    if rc.shape[0] != len(reproj_files):
        raise ValueError(
            f"ref_coords_array first axis ({rc.shape[0]}) must match "
            f"len(reproj_files) ({len(reproj_files)})")
    keep = ((rc[:, 0] < cy1) & (cy0 < rc[:, 1])
            & (rc[:, 2] < cx1) & (cx0 < rc[:, 3]))
    kept_idx = np.nonzero(keep)[0].astype(np.int64)
    kept_files = [reproj_files[i] for i in kept_idx]
    return kept_files, kept_idx


def filter_by_center(reproj_files, rc_table, chunk_bbox, halo=0):
    """``(kept_files, kept_idx)`` for frames whose footprint CENTER falls in
    ``chunk_bbox`` expanded by ``halo`` px per side. With ``halo=0`` and a
    non-overlapping tiling each frame goes to at most one tile, bounding per-tile
    solve memory. ``rc_table`` is the ``(N, 4)`` table from
    :func:`load_ref_coords_table`."""
    cy0, cy1, cx0, cx1 = chunk_bbox
    cy = (rc_table[:, 0] + rc_table[:, 1]) / 2.0
    cx = (rc_table[:, 2] + rc_table[:, 3]) / 2.0
    mask = ((cy >= cy0 - halo) & (cy < cy1 + halo)
            & (cx >= cx0 - halo) & (cx < cx1 + halo))
    kept_idx = np.where(mask)[0].astype(np.int64)
    kept_files = [reproj_files[int(i)] for i in kept_idx]
    return kept_files, kept_idx
