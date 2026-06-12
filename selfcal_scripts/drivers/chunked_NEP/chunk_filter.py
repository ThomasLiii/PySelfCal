"""
Frame filter for 4-chunk PAH-fit mosaic build.

Given a list of reprojected sub-frame HDF5 files and a chunk bbox in
reference-mosaic pixel coordinates (half-open [y0, y1) x [x0, x1)),
return the subset of files whose ref_coords AABB overlaps the chunk.

ref_coords schema (from survey of exp_0040_det_00.h5 + SelfCal/reproject.py):
  - Lives in file.attrs['ref_coords'] (HDF5 root attrs, NOT a dataset).
  - dtype int32, shape (4,).
  - Order: [y_min, y_max, x_min, x_max] (row-min, row-max, col-min, col-max).
  - Half-open: y_max = y_min + sub_height (exclusive), x_max exclusive.
    Confirmed by SelfCal/reproject.py:77-78 and consumers MapHelper.py:200-204
    that slice as footprint[y_min:y_max, x_min:x_max].

Half-open AABB overlap test: two intervals [a0, a1) and [b0, b1) overlap iff
    a0 < b1 and b0 < a1
Applied independently on y and x axes.
"""

from __future__ import annotations

import h5py
import numpy as np


def compute_overlapping_frames(
    reproj_files: list[str],
    chunk_bbox: tuple[int, int, int, int],
) -> tuple[list[str], np.ndarray]:
    """Return (kept_files, kept_idx) where each frame's ref_coords AABB
    overlaps chunk_bbox.

    Parameters
    ----------
    reproj_files
        Absolute paths to reprojected sub-frame HDF5 files (one per
        exposure-detector combo), each carrying root attr 'ref_coords'
        = [y_min, y_max, x_min, x_max] (half-open) in reference-mosaic
        pixel coordinates.
    chunk_bbox
        (y0, y1, x0, x1) half-open bbox of the chunk in the same
        reference-mosaic pixel frame.

    Returns
    -------
    kept_files : list[str]
        Subset of reproj_files (preserving input order) whose ref_coords
        AABB overlaps chunk_bbox on both axes.
    kept_idx : np.ndarray
        int64 array of indices into the original reproj_files for the
        kept entries (kept_files == [reproj_files[i] for i in kept_idx]).
    """
    cy0, cy1, cx0, cx1 = chunk_bbox
    if not (cy0 < cy1 and cx0 < cx1):
        raise ValueError(
            f"chunk_bbox must be a non-empty half-open box; got "
            f"y[{cy0}, {cy1}) x[{cx0}, {cx1})"
        )

    n = len(reproj_files)
    keep = np.zeros(n, dtype=bool)

    for i, path in enumerate(reproj_files):
        try:
            with h5py.File(path, "r") as f:
                if "ref_coords" not in f.attrs:
                    # Not a reprojected sub-frame file we recognize; skip.
                    continue
                rc = np.asarray(f.attrs["ref_coords"], dtype=np.int64)
        except (OSError, KeyError) as exc:
            # Unreadable / corrupt file: skip with a warning so the build
            # doesn't die on a single bad frame.
            import warnings

            warnings.warn(
                f"compute_overlapping_frames: could not read ref_coords "
                f"from {path}: {exc!r}; skipping.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        if rc.shape != (4,):
            import warnings

            warnings.warn(
                f"compute_overlapping_frames: unexpected ref_coords shape "
                f"{rc.shape} in {path}; expected (4,); skipping.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue

        fy0, fy1, fx0, fx1 = int(rc[0]), int(rc[1]), int(rc[2]), int(rc[3])

        # Standard half-open AABB overlap test on each axis.
        # [a0, a1) overlaps [b0, b1) iff a0 < b1 AND b0 < a1.
        if (fy0 < cy1) and (cy0 < fy1) and (fx0 < cx1) and (cx0 < fx1):
            keep[i] = True

    kept_idx = np.nonzero(keep)[0].astype(np.int64)
    kept_files = [reproj_files[i] for i in kept_idx]
    return kept_files, kept_idx


def compute_overlapping_frames_from_cache(
    reproj_files: list[str],
    ref_coords_array: np.ndarray,
    chunk_bbox: tuple[int, int, int, int],
) -> tuple[list[str], np.ndarray]:
    """Vectorized variant when ref_coords for all files have already been
    cached into an (N, 4) int array. Useful if compute_overlapping_frames
    is called 4 times (once per chunk) over the same file list -- read
    the attrs once, then call this for each chunk_bbox.

    ref_coords_array[i] = [y_min, y_max, x_min, x_max] (half-open).
    """
    cy0, cy1, cx0, cx1 = chunk_bbox
    if not (cy0 < cy1 and cx0 < cx1):
        raise ValueError(
            f"chunk_bbox must be a non-empty half-open box; got "
            f"y[{cy0}, {cy1}) x[{cx0}, {cx1})"
        )
    rc = np.asarray(ref_coords_array, dtype=np.int64)
    if rc.ndim != 2 or rc.shape[1] != 4:
        raise ValueError(
            f"ref_coords_array must have shape (N, 4); got {rc.shape}"
        )
    if rc.shape[0] != len(reproj_files):
        raise ValueError(
            f"ref_coords_array first axis ({rc.shape[0]}) must match "
            f"len(reproj_files) ({len(reproj_files)})"
        )

    fy0 = rc[:, 0]
    fy1 = rc[:, 1]
    fx0 = rc[:, 2]
    fx1 = rc[:, 3]
    keep = (fy0 < cy1) & (cy0 < fy1) & (fx0 < cx1) & (cx0 < fx1)
    kept_idx = np.nonzero(keep)[0].astype(np.int64)
    kept_files = [reproj_files[i] for i in kept_idx]
    return kept_files, kept_idx


def load_ref_coords_table(reproj_files: list[str]) -> np.ndarray:
    """Read ref_coords from every file once and return as (N, 4) int64.
    Files that lack the attr or fail to open get a sentinel row of -1s
    so the overlap test (which requires fy0 < cy1 AND fx0 < cx1 with
    cy1, cx1 > 0) excludes them naturally only if both bounds are
    nonpositive -- so we instead use a sentinel that guarantees no
    overlap: [0, 0, 0, 0] (empty box on both axes).
    """
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


def filter_by_center(
    reproj_files: list[str],
    rc_table: np.ndarray,
    chunk_bbox: tuple[int, int, int, int],
    halo: int = 0,
) -> tuple[list[str], np.ndarray]:
    """Filter frames whose CENTER falls in (chunk_bbox expanded by halo on all sides).

    Center is computed as the midpoint of ref_coords. Each frame goes to AT MOST
    one chunk when halo=0 (assuming the 4 chunk_bboxes tile the ref grid with no
    overlap), so the per-chunk frame count is bounded by N_total / 4.

    Unlike compute_overlapping_frames_from_cache (which is AABB-based and pulls in
    every frame whose footprint *intersects* chunk_bbox), this filter is much
    more selective and is used to keep per-chunk solve memory under control on
    the full-dataset spatial chunking.

    Parameters
    ----------
    reproj_files, rc_table : as in compute_overlapping_frames_from_cache.
    chunk_bbox : (y0, y1, x0, x1) half-open.
    halo : int, default 0
        Expand chunk_bbox by `halo` px on every side before testing center
        membership. Use halo>0 to share frames near seams across adjacent chunks.

    Returns
    -------
    kept_files : list[str]
    kept_idx : np.ndarray of int64
    """
    cy0, cy1, cx0, cx1 = chunk_bbox
    cy = (rc_table[:, 0] + rc_table[:, 1]) / 2.0
    cx = (rc_table[:, 2] + rc_table[:, 3]) / 2.0
    mask = ((cy >= cy0 - halo) & (cy < cy1 + halo)
            & (cx >= cx0 - halo) & (cx < cx1 + halo))
    kept_idx = np.where(mask)[0].astype(np.int64)
    kept_files = [reproj_files[int(i)] for i in kept_idx]
    return kept_files, kept_idx
