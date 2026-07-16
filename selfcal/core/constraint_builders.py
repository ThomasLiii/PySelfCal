"""Composable builders for the global LSQR constraint blocks.

These are the constraint rows appended in the parent process *after* the data
rows (per-frame adjacency / polynomial constraints are emitted inside the
worker and stay there for now). Each builder returns a :class:`ConstraintBlock`
(or ``None`` when it would be empty); ``setup_lsqr`` appends them in a fixed
order that the CSR scatter depends on:

    1. per-map mean-offset anchors (one per chunk map)
    2. sky damping, in sky-component order (continuum, then each line block)
    3. offset damping

Bit-identity contract: these reproduce the historical inline blocks in
``lsqr.setup_lsqr`` exactly (same cols/data/b, same dtypes, same nnz_per_row).
``sky_damping_block`` is the per-component generalization of the old
continuum + line damping special case: block ``j`` lives at columns
``j*num_sky + valid_pixels`` with ``data = sqrt(weight * coverage)``.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class ConstraintBlock:
    """One global constraint block. Mirrors the dict the CSR scatter consumes."""

    rows_local: np.ndarray
    cols: np.ndarray
    data: np.ndarray
    b: np.ndarray
    num_rows: int
    nnz_per_row: object  # int (uniform) or ndarray (per-row)

    def as_dict(self):
        return {
            'rows_local': self.rows_local,
            'cols': self.cols,
            'data': self.data,
            'b': self.b,
            'num_rows': self.num_rows,
            'nnz_per_row': self.nnz_per_row,
        }


def mean_offset_block(m, mean_off, num_frames, num_chunks_m, ftg_m, col_bases,
                      weight=10.0):
    """Per-frame mean-offset anchor for chunk map ``m``.

    Constrains each frame's per-chunk offset mean toward ``mean_off`` (length
    num_frames) with the given Lagrange ``weight``. Caller must skip template-
    mode maps (no per-chunk offsets) and None ``mean_off``.
    """
    mean_offsets_arr = np.asarray(mean_off)
    nc_m = num_chunks_m
    rows_local = np.repeat(np.arange(num_frames, dtype=np.int64), nc_m)
    offset_starts = col_bases[m] + ftg_m.astype(np.int64) * nc_m
    cols = (offset_starts[:, None] + np.arange(nc_m, dtype=np.int64)[None, :]).reshape(-1)
    data = np.full(num_frames * nc_m, weight, dtype=np.float32)
    b = mean_offsets_arr.astype(np.float64).flatten() * nc_m * weight
    return ConstraintBlock(rows_local, cols, data, b, num_rows=num_frames, nnz_per_row=nc_m)


def sky_damping_block(block_index, weight, coverage, num_sky):
    """Coverage-weighted Tikhonov damping for sky block ``block_index``.

    block 0 = continuum, block 1+ = line components. Columns are
    ``block_index*num_sky + valid_pixels``; one nnz per damped pixel with
    ``data = sqrt(weight * coverage[pixel])``. Returns None if no covered pixel.
    """
    valid = np.nonzero(coverage)[0]
    if len(valid) == 0:
        return None
    data = np.sqrt(weight * coverage[valid]).astype(np.float32)
    n = len(valid)
    cols = (block_index * num_sky + valid).astype(np.int64, copy=False)
    return ConstraintBlock(np.arange(n, dtype=np.int64), cols, data,
                           np.zeros(n, dtype=np.float64), num_rows=n, nnz_per_row=1)


def line_separability_block(block_index, lam, num_sky):
    """Per-pixel Tikhonov rows on sky block ``block_index`` with EXPLICIT
    per-pixel amplitudes ``lam`` (one row per pixel with ``lam > 0``,
    ``data = lam[pixel]``).

    Used by the separability "water-filling" line damping: with per-pixel
    cont/line separability ``I_P`` (the Schur complement of the per-pixel 2x2
    normal-matrix block), ``lam[P] = sqrt(max(0, tau2 - I_P))`` tops the
    effective information up to ``I_P + lam^2 >= tau2``. Pixels with plenty of
    wavelength diversity get lam = 0 (NO bias — unlike a uniform line damping);
    diversity-poor pixels get lifted off the LSQR 1/sigma noise-amplification
    floor, which removes the semi-convergence cliff. Returns None if no pixel
    needs lifting.
    """
    valid = np.nonzero(lam > 0)[0]
    if len(valid) == 0:
        return None
    data = lam[valid].astype(np.float32)
    n = len(valid)
    cols = (block_index * num_sky + valid).astype(np.int64, copy=False)
    return ConstraintBlock(np.arange(n, dtype=np.int64), cols, data,
                           np.zeros(n, dtype=np.float64), num_rows=n, nnz_per_row=1)


def line_spatial_coherence_block(block_index, lam_edge_h, lam_edge_v, num_sky):
    """Diversity-adaptive SPATIAL-COHERENCE prior on sky block ``block_index``.

    Tikhonov difference rows ``lam * (x[P] - x[Q]) = 0`` for ref-grid neighbor
    pairs: horizontal pairs weighted by ``lam_edge_h`` (shape (H, W-1), pair
    (y,x)-(y,x+1)) and vertical by ``lam_edge_v`` (shape (H-1, W), pair
    (y,x)-(y+1,x)). Rows are emitted only where lam > 0.

    Unlike amplitude damping (pull toward ZERO), this pulls a
    wavelength-diversity-poor pixel toward its NEIGHBORS' value, so the
    ill-constrained cont/line split at low-I_P pixels borrows constraint from
    better-sampled neighbors with no amplitude bias. Because the sampling
    diversity oscillates spatially (survey-geometry pattern), neighbor pairs
    straddle the pattern's phases and the prior cancels the pattern-scale
    differential rather than imprinting it. Returns None if no edge.
    """
    yh, xh = np.nonzero(lam_edge_h > 0)
    yv, xv = np.nonzero(lam_edge_v > 0)
    W = lam_edge_v.shape[1]
    ph = yh * W + xh
    pv = yv * W + xv
    lam = np.concatenate([lam_edge_h[yh, xh], lam_edge_v[yv, xv]]).astype(np.float32)
    n = lam.size
    if n == 0:
        return None
    pcol = np.concatenate([ph, pv]).astype(np.int64)
    qcol = np.concatenate([ph + 1, pv + W]).astype(np.int64)
    rows_local = np.repeat(np.arange(n, dtype=np.int64), 2)
    cols = np.empty(2 * n, dtype=np.int64)
    cols[0::2] = block_index * num_sky + pcol
    cols[1::2] = block_index * num_sky + qcol
    data = np.empty(2 * n, dtype=np.float32)
    data[0::2] = lam
    data[1::2] = -lam
    return ConstraintBlock(rows_local, cols, data, np.zeros(n, dtype=np.float64),
                           num_rows=n, nnz_per_row=2)


def offset_damping_block(weight, offset_block_coverage, num_sky_eff):
    """Coverage-weighted damping on the offset columns (``damp_offset``).

    Columns are ``num_sky_eff + valid_offset_cols``. Returns None if empty.
    """
    valid = np.nonzero(offset_block_coverage)[0]
    if len(valid) == 0:
        return None
    data = np.sqrt(weight * offset_block_coverage[valid]).astype(np.float32)
    n = len(valid)
    cols = (valid + num_sky_eff).astype(np.int64, copy=False)
    return ConstraintBlock(np.arange(n, dtype=np.int64), cols, data,
                           np.zeros(n, dtype=np.float64), num_rows=n, nnz_per_row=1)
