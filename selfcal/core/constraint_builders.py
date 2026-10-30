"""Composable builders for the global LSQR constraint blocks.

These are the constraint rows appended in the parent process *after* the data
rows (the per-frame adjacency rows are emitted inside the
worker and are not built by these helpers, except the grouped adjacency of
``group_adjacency_maps``). Each builder returns a :class:`ConstraintBlock`
(or ``None`` when it would be empty); ``setup_lsqr`` appends them in a fixed
order that the CSR scatter depends on:

    1. grouped adjacency, one block per map in ``group_adjacency_maps``
       (absent by default)
    2. sky damping, in sky-component order (continuum, then each line block)
    3. offset damping, per map with ``damp_offset_maps``
    4. user priors (``setup_lsqr(priors=...)``), in the order given — any
       linear rows a caller builds from :class:`SystemInfo`
       (:func:`as_constraint_block` normalises them)

Bit-identity contract: the exact cols/data/b values, dtypes, and nnz_per_row
emitted here are frozen by the byte-equality regression goldens (reference
``cal_*.h5`` files of the byte-equality gates (``selfcal_scripts/gates/``), diffed with
``selfcal_scripts/gates/diff_cal_h5.py``); re-baseline those before changing
any of them. ``sky_damping_block`` damps every sky component uniformly: block
``j`` lives at columns ``j*num_sky + valid_pixels`` with
``data = sqrt(weight * coverage)`` (block 0 = continuum, blocks 1+ = line
components).
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix


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
        """Return the block as the dict the CSR scatter consumes, keyed by field name.

        The arrays are shared with the block, not copied;
        :func:`~selfcal.core.system.setup_lsqr` collects one such dict per block.
        """
        return {
            'rows_local': self.rows_local,
            'cols': self.cols,
            'data': self.data,
            'b': self.b,
            'num_rows': self.num_rows,
            'nnz_per_row': self.nnz_per_row,
        }


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


def offset_damping_block(weight, offset_block_coverage, num_sky_eff):
    """Coverage-weighted damping on the offset columns of one map (``damp_offset_maps``).

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


def grouped_adjacency_block(m, adj_info, reg_weight, num_chunks_m, ftg_m, col_bases):
    """Adjacency regularization ``rw·(O_i − O_j) = 0`` for a det-grouped map,
    emitted once per group instead of once per frame.

    The worker emits these rows per frame, so a map shared by the k frames of
    a group gets k identical copies; their normal-equation contribution equals
    one copy with weight ``rw·√k``. Exact in AᵀA / Aᵀb, while nnz drops from
    frames×pairs to groups×pairs (7.5e9 → 8.3e6 for the N=510 Euclid EDFN
    solve). The caller must zero the worker-side ``reg_weight`` for this map.
    """
    chunk_i = np.asarray(adj_info[0], dtype=np.int64)
    chunk_j = np.asarray(adj_info[1], dtype=np.int64)
    npair = len(chunk_i)
    groups, counts = np.unique(np.asarray(ftg_m).astype(np.int64), return_counts=True)
    rows_l, cols_l, data_l = [], [], []
    r0 = 0
    for g, k in zip(groups, counts):
        base = col_bases[m] + g * num_chunks_m
        w = np.float32(reg_weight * np.sqrt(k))
        rows_l.append(np.repeat(np.arange(npair, dtype=np.int64) + r0, 2))
        cols_l.append(np.stack([base + chunk_i, base + chunk_j], axis=1).reshape(-1))
        data_l.append(np.tile(np.array([w, -w], dtype=np.float32), npair))
        r0 += npair
    return ConstraintBlock(np.concatenate(rows_l), np.concatenate(cols_l),
                           np.concatenate(data_l), np.zeros(r0, dtype=np.float64),
                           num_rows=r0, nnz_per_row=2)


@dataclass(frozen=True)
class SystemInfo:
    """What a user prior sees when it builds its rows: the column layout of the
    unknowns (:class:`~selfcal.core.layout.SystemLayout`: the sky blocks, each
    offset map's ``col_bases`` / groups / chunks) and the
    number of observations that touch each column (``pixel_counts``, the full
    column space) — known only after the data rows are built, which is when the
    priors run. ``sky_names`` names the sky blocks in order."""
    layout: object
    pixel_counts: np.ndarray
    sky_names: tuple = ()


def as_constraint_block(result, n_cols, name='prior'):
    """Normalise what a user prior returns into a :class:`ConstraintBlock`.

    ``result``: a ``ConstraintBlock``, a ``(rows_local, cols, data, b)`` tuple
    (rows numbered from 0 within the prior; ``b`` one value per row; entries
    in any order, duplicates summed), or ``None`` (no rows). Columns are global
    column ids (``SystemInfo.layout``)."""
    if result is None:
        return None
    if isinstance(result, ConstraintBlock):
        return result if result.num_rows else None
    rows, cols, data, b = result
    rows = np.asarray(rows, dtype=np.int64).ravel()
    cols = np.asarray(cols, dtype=np.int64).ravel()
    data = np.asarray(data, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if not (rows.size == cols.size == data.size):
        raise ValueError(f"{name}: rows, cols and data must have the same length "
                         f"({rows.size}, {cols.size}, {data.size})")
    if rows.size == 0:
        return None
    if rows.min() < 0 or cols.min() < 0 or cols.max() >= n_cols:
        raise ValueError(f"{name}: row or column index out of range (columns must be in [0, {n_cols}))")
    n_rows = int(rows.max()) + 1
    if b.size != n_rows:
        raise ValueError(f"{name}: b has {b.size} values for {n_rows} rows")
    if not (np.isfinite(data).all() and np.isfinite(b).all()):
        raise ValueError(f"{name}: non-finite prior coefficients or targets")
    A = coo_matrix((data, (rows, cols)), shape=(n_rows, n_cols)).tocsr()
    A.sum_duplicates()
    counts = np.diff(A.indptr)
    rows_local = np.repeat(np.arange(n_rows, dtype=np.int64), counts)
    return ConstraintBlock(rows_local, A.indices.astype(np.int64), A.data.astype(np.float32), b,
                           num_rows=n_rows, nnz_per_row=counts.astype(np.int64))
