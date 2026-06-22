"""Tiled (region-partitioned) calibration.

Generalizes the chunked-NEP pattern (4 copy-pasted quadrant drivers +
stitch_cals.py + launch_chain.sh) into one reusable mechanism: partition a large
reference mosaic into overlapping tiles, assign each frame to a tile by
footprint center (bounded per-tile memory) or AABB overlap, calibrate each tile
independently, then merge the per-tile sky maps with Fisher-weighted
inverse-variance averaging.

Tiling is instrument-agnostic — usable for any large-region build (e.g. Euclid
multi-region), not just SPHEREx NEP.

- :class:`TileSpec` / :func:`make_tile_grid` — tile geometry (an n_y x n_x grid
  with per-side overlap; reproduces the NEP 2x2 quadrant bboxes).
- :func:`assign_frames` — frame -> tile assignment via
  :mod:`selfcal.io.frame_select`.
- :func:`stitch` — Fisher-weighted merge of per-tile cal files (faithful port of
  the chunked-NEP stitcher; byte-equal output on the same inputs).
- :class:`TiledCalibration` — ties the three together; ``run(run_tile=...)``
  calibrates each tile via a caller-supplied per-tile recipe.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import h5py
import numpy as np

from ..io.frame_select import (load_ref_coords_table, filter_by_center,
                               compute_overlapping_frames_from_cache)

STITCHER_VERSION = "fisher-stream-v2"


@dataclass(frozen=True)
class TileSpec:
    """A tile: a name and a half-open ``(y0, y1, x0, x1)`` bbox in ref-grid px."""

    name: str
    bbox: tuple


def make_tile_grid(ref_shape, n_y, n_x, overlap_px=0, names=None):
    """An ``n_y`` x ``n_x`` grid of tiles over ``ref_shape``, each grown by
    ``overlap_px // 2`` px on every interior side so adjacent tiles overlap by
    ``overlap_px`` (footprints provide the sky overlap the Fisher stitch blends).

    With ``ref_shape=(12676, 12672), n_y=n_x=2, overlap_px=50`` this reproduces
    the NEP quadrant bboxes (NW/NE/SW/SE) exactly. Default names are ``r{j}c{i}``;
    pass ``names`` (row-major) to override.
    """
    H, W = ref_shape
    y_b = [round(i * H / n_y) for i in range(n_y + 1)]
    x_b = [round(i * W / n_x) for i in range(n_x + 1)]
    half = overlap_px // 2
    tiles = []
    for jy in range(n_y):
        y0 = max(0, y_b[jy] - (half if jy > 0 else 0))
        y1 = min(H, y_b[jy + 1] + (half if jy < n_y - 1 else 0))
        for jx in range(n_x):
            x0 = max(0, x_b[jx] - (half if jx > 0 else 0))
            x1 = min(W, x_b[jx + 1] + (half if jx < n_x - 1 else 0))
            tiles.append(TileSpec(name=f"r{jy}c{jx}", bbox=(y0, y1, x0, x1)))
    if names is not None:
        if len(names) != len(tiles):
            raise ValueError(f"names has {len(names)} entries, need {len(tiles)}")
        tiles = [TileSpec(name=nm, bbox=t.bbox) for nm, t in zip(names, tiles)]
    return tiles


def assign_frames(reproj_files, tiles, frame_filter='center', halo=0):
    """Assign frames to tiles. Returns ``{tile_name: (files, idx)}``.

    ``frame_filter='center'`` (default) keeps frames whose footprint center is in
    the tile (+halo) — bounded per-tile memory, each frame in <=1 tile when tiles
    don't overlap and halo=0. ``'overlap'`` keeps every frame whose footprint
    AABB intersects the tile. One ref_coords table read is shared across tiles.
    """
    rc_table = load_ref_coords_table(reproj_files)
    out = {}
    for t in tiles:
        if frame_filter == 'center':
            out[t.name] = filter_by_center(reproj_files, rc_table, t.bbox, halo=halo)
        elif frame_filter == 'overlap':
            out[t.name] = compute_overlapping_frames_from_cache(reproj_files, rc_table, t.bbox)
        else:
            raise ValueError(f"frame_filter must be 'center' or 'overlap', got {frame_filter!r}")
    return out


def _bbox_nonzero(cov):
    rows = np.where(cov.any(axis=1))[0]
    cols = np.where(cov.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def stitch(input_paths, output_path, ref_shape=None, line=True, verbose=True):
    """Fisher-weighted inverse-variance merge of per-tile cal files into one
    cal-shaped h5.

    Each output sky pixel is ``sum_t F_t * sky_t / sum_t F_t`` over tiles with
    Fisher ``F_t > 0``; ``skymap_fisher`` is ``sum_t F_t`` and ``skymap_coverage``
    the summed pixel counts. Per-frame quantities (offsets, frame_scalar,
    reproj_list) are dropped (not composable across disjoint frame subsets) and
    ``num_maps`` is written 0; re-mosaic against the stitched cal if needed.

    Reads the continuum (and, when ``line``, the first line block) via the
    top-level ``skymap`` / ``skymap_line`` names, which resolve for both the v2
    schema and the v3 hard-link aliases. Output datasets/dtypes match the
    chunked-NEP stitcher, so the result is byte-equal on the same inputs.
    """
    if len(input_paths) < 1:
        raise ValueError(f"need at least 1 input cal file, got {len(input_paths)}")
    for p in input_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    if ref_shape is None:
        with h5py.File(input_paths[0], "r") as f:
            ref_shape = tuple(f["skymap"].shape)
    H, W = ref_shape

    num_cont = np.zeros((H, W), dtype=np.float64)
    den_cont = np.zeros((H, W), dtype=np.float64)
    cov_cont = np.zeros((H, W), dtype=np.int64)
    n_cont = np.zeros((H, W), dtype=np.uint8)
    if line:
        num_line = np.zeros((H, W), dtype=np.float64)
        den_line = np.zeros((H, W), dtype=np.float64)
        cov_line = np.zeros((H, W), dtype=np.int64)
        n_line = np.zeros((H, W), dtype=np.uint8)

    for p in input_paths:
        t0 = time.time()
        with h5py.File(p, "r") as f:
            cov_c_full = f["skymap_coverage"][:]
            bb = _bbox_nonzero(cov_c_full)
            if bb is None:
                if verbose:
                    print(f"[stitch] {os.path.basename(p)}: empty coverage; skipping", flush=True)
                continue
            r0, r1, c0, c1 = bb
            sky_c = f["skymap"][r0:r1, c0:c1]
            fish_c = f["skymap_fisher"][r0:r1, c0:c1]
            cov_c = cov_c_full[r0:r1, c0:c1]
            if line:
                sky_l = f["skymap_line"][r0:r1, c0:c1]
                fish_l = f["skymap_line_fisher"][r0:r1, c0:c1]
                cov_l = f["skymap_line_coverage"][r0:r1, c0:c1]
        del cov_c_full

        sl = (slice(r0, r1), slice(c0, c1))
        m_c = fish_c > 0
        if m_c.any():
            fc64 = fish_c.astype(np.float64, copy=False)
            sc64 = sky_c.astype(np.float64, copy=False)
            num_cont[sl] += fc64 * sc64 * m_c
            den_cont[sl] += fc64 * m_c
            cov_cont[sl] += cov_c
            n_cont[sl] += m_c.astype(np.uint8)
            del fc64, sc64
        if line:
            m_l = fish_l > 0
            if m_l.any():
                fl64 = fish_l.astype(np.float64, copy=False)
                sl64 = sky_l.astype(np.float64, copy=False)
                num_line[sl] += fl64 * sl64 * m_l
                den_line[sl] += fl64 * m_l
                cov_line[sl] += cov_l
                n_line[sl] += m_l.astype(np.uint8)
                del fl64, sl64
            del sky_l, fish_l, cov_l
        del sky_c, fish_c, cov_c
        if verbose:
            print(f"[stitch] accumulated {os.path.basename(p)} in {time.time()-t0:.1f}s "
                  f"(bbox y[{r0}:{r1}] x[{c0}:{c1}])", flush=True)

    m_c = den_cont > 0.0
    sky_cont = np.zeros((H, W), dtype=np.float32)
    np.divide(num_cont, den_cont, out=sky_cont, where=m_c, casting="unsafe")
    skymap_fisher = den_cont.astype(np.float32)
    if line:
        m_l = den_line > 0.0
        sky_line = np.zeros((H, W), dtype=np.float32)
        np.divide(num_line, den_line, out=sky_line, where=m_l, casting="unsafe")
        skymap_line_fisher = den_line.astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    tmp = output_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("skymap", data=sky_cont)
        f.create_dataset("skymap_fisher", data=skymap_fisher)
        f.create_dataset("skymap_coverage", data=cov_cont)
        f.create_dataset("n_contrib_cont", data=n_cont)
        if line:
            f.create_dataset("skymap_line", data=sky_line)
            f.create_dataset("skymap_line_fisher", data=skymap_line_fisher)
            f.create_dataset("skymap_line_coverage", data=cov_line)
            f.create_dataset("n_contrib_line", data=n_line)
        f.attrs["num_maps"] = np.int64(0)
        f.attrs["num_sky_blocks"] = np.int64(2 if line else 1)
        f.attrs["stitched_from"] = np.array([str(p) for p in input_paths], dtype="S")
        f.attrs["stitched_method"] = "fisher_weighted_inverse_variance"
        f.attrs["stitcher_version"] = STITCHER_VERSION
    os.replace(tmp, output_path)
    if verbose:
        print(f"[stitch] wrote {output_path} "
              f"(cont covered {int(m_c.sum()):,}px)", flush=True)
    return output_path


class TiledCalibration:
    """Orchestrates a tiled-region calibration: frame assignment, per-tile
    calibration via a caller-supplied recipe, and the Fisher stitch.

    ``run_tile(tile, files) -> cal_path`` is the per-tile recipe (e.g. the
    standard setup_lsqr + apply_lsqr + save_calibration for that tile's frames
    and run name); the orchestrator only handles assignment, sequencing, and
    merging. Tiles run sequentially by default (per-tile peak RSS is large).
    """

    def __init__(self, reproj_files, tiles, frame_filter='center', halo=0):
        self.reproj_files = list(reproj_files)
        self.tiles = list(tiles)
        self.frame_filter = frame_filter
        self.halo = halo
        self._assignment = None

    def assign_frames(self):
        if self._assignment is None:
            self._assignment = assign_frames(
                self.reproj_files, self.tiles,
                frame_filter=self.frame_filter, halo=self.halo)
        return self._assignment

    def run(self, run_tile, sequential=True):
        """Calibrate each tile. ``run_tile(tile, files) -> cal_path``. Returns
        ``{tile_name: cal_path}``. Aborts on the first tile that raises."""
        if not sequential:
            raise NotImplementedError(
                "parallel tile execution not supported (per-tile RSS is large)")
        assignment = self.assign_frames()
        cal_paths = {}
        for t in self.tiles:
            files, _idx = assignment[t.name]
            cal_paths[t.name] = run_tile(t, files)
        return cal_paths

    def stitch(self, cal_paths, output_path, **kwargs):
        """Fisher-stitch the per-tile cal files (a list, or the dict from
        :meth:`run`) into ``output_path``."""
        if isinstance(cal_paths, dict):
            cal_paths = [cal_paths[t.name] for t in self.tiles if t.name in cal_paths]
        return stitch(list(cal_paths), output_path, **kwargs)
