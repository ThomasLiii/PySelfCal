"""Composable builders for the global LSQR constraint blocks.

These are the constraint rows appended in the parent process *after* the data
rows (per-frame adjacency / polynomial constraints are emitted inside the
worker and are not built by these helpers, except the grouped adjacency of
``group_adjacency_maps``). Each builder returns a :class:`ConstraintBlock`
(or ``None`` when it would be empty); ``setup_lsqr`` appends them in a fixed
order that the CSR scatter depends on:

    1. per-map mean-offset anchors (one per chunk map)
    2. grouped adjacency, one block per map in ``group_adjacency_maps``
       (absent by default)
    3. sky damping, in sky-component order (continuum, then each line block)
    4. offset damping, per map with ``damp_offset_maps``
    5. user priors (``setup_lsqr(priors=...)``), in the order given — any
       linear rows a caller builds from :class:`SystemInfo`
       (:func:`as_constraint_block` normalises them)

Bit-identity contract: the exact cols/data/b values, dtypes, and nnz_per_row
emitted here are frozen by the byte-equality regression goldens (reference
``cal_*.h5`` files of the byte-equality gates (``selfcal_scripts/gates/``), diffed with
``selfcal_scripts/drivers/diff_cal_h5.py``); re-baseline those before changing
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


def mean_offset_block(m, mean_off, num_frames, num_chunks_m, ftg_m, col_bases,
                      weight=10.0, group_rows=False, n_basis=1):
    """Per-frame mean-offset anchor for chunk map ``m``.

    Constrains each frame's per-chunk offset mean toward ``mean_off`` (length
    num_frames) with the given Lagrange ``weight``. Caller must skip template-
    mode maps (no per-chunk offsets) and None ``mean_off``.

    ``group_rows``: when several frames share one offset group (det_groups),
    their per-frame rows are identical (same columns, same weight, and the same
    target when the targets agree). k identical rows contribute k·w²·11ᵀ to AᵀA
    and k·w·β to Aᵀb, exactly what one row with weight w·√k and rhs β·√k
    contributes. Emitting one row per group is therefore exact in the normal
    equations while cutting nnz from frames×chunks to groups×chunks (3.8e9 →
    4e6 for the N=510 Euclid EDFN solve). A group whose frames disagree on the
    target keeps its per-frame rows. Off by default: the per-frame form is
    frozen by the byte-equality goldens.

    ``n_basis`` > 1 (a map whose offsets are coefficients of ``n_basis``
    functions per chunk, columns ``c*n_basis + k``): one anchor per function
    ``k`` — the mean over chunks of each function's coefficient.
    """
    if n_basis > 1:
        return _mean_offset_block_basis(m, mean_off, num_frames, num_chunks_m, ftg_m, col_bases,
                                        weight, group_rows, int(n_basis))
    mean_offsets_arr = np.asarray(mean_off)
    nc_m = num_chunks_m
    if group_rows:
        ftg = np.asarray(ftg_m).astype(np.int64)
        targets = mean_offsets_arr.astype(np.float64).flatten()
        chunk_idx = np.arange(nc_m, dtype=np.int64)
        rows_l, cols_l, data_l, b_l = [], [], [], []
        r = 0
        for g in np.unique(ftg):
            members = np.nonzero(ftg == g)[0]
            tg = targets[members]
            if np.all(tg == tg[0]):
                w = weight * np.sqrt(len(members))
                rows_l.append(np.full(nc_m, r, dtype=np.int64))
                cols_l.append(col_bases[m] + g * nc_m + chunk_idx)
                data_l.append(np.full(nc_m, w, dtype=np.float32))
                b_l.append(tg[0] * nc_m * w)
                r += 1
            else:
                for fi in members:
                    rows_l.append(np.full(nc_m, r, dtype=np.int64))
                    cols_l.append(col_bases[m] + g * nc_m + chunk_idx)
                    data_l.append(np.full(nc_m, weight, dtype=np.float32))
                    b_l.append(targets[fi] * nc_m * weight)
                    r += 1
        return ConstraintBlock(np.concatenate(rows_l), np.concatenate(cols_l),
                               np.concatenate(data_l), np.array(b_l, dtype=np.float64),
                               num_rows=r, nnz_per_row=nc_m)
    rows_local = np.repeat(np.arange(num_frames, dtype=np.int64), nc_m)
    offset_starts = col_bases[m] + ftg_m.astype(np.int64) * nc_m
    cols = (offset_starts[:, None] + np.arange(nc_m, dtype=np.int64)[None, :]).reshape(-1)
    data = np.full(num_frames * nc_m, weight, dtype=np.float32)
    b = mean_offsets_arr.astype(np.float64).flatten() * nc_m * weight
    return ConstraintBlock(rows_local, cols, data, b, num_rows=num_frames, nnz_per_row=nc_m)


def _mean_offset_block_basis(m, mean_off, num_frames, num_cols_m, ftg_m, col_bases, weight,
                             group_rows, nb):
    """``mean_offset_block`` for a map with ``nb`` basis functions per chunk
    (``num_cols_m = n_chunks * nb`` columns per group): rows per (frame or
    group, function)."""
    targets = np.asarray(mean_off, dtype=np.float64).ravel()
    nc = num_cols_m // nb
    ftg = np.asarray(ftg_m).astype(np.int64)
    units = []                                   # (group, target, row weight)
    if group_rows:
        for g in np.unique(ftg):
            members = np.nonzero(ftg == g)[0]
            tg = targets[members]
            if np.all(tg == tg[0]):
                units.append((g, tg[0], weight * np.sqrt(len(members))))
            else:
                units.extend((g, targets[f], weight) for f in members)
    else:
        units = [(ftg[f], targets[f], weight) for f in range(num_frames)]
    rows_l, cols_l, data_l, b_l = [], [], [], []
    chunk = np.arange(nc, dtype=np.int64)
    r = 0
    for g, tgt, w in units:
        start = col_bases[m] + g * num_cols_m
        for k in range(nb):
            rows_l.append(np.full(nc, r, dtype=np.int64))
            cols_l.append(start + chunk * nb + k)
            data_l.append(np.full(nc, w, dtype=np.float32))
            b_l.append(tgt * nc * w)
            r += 1
    return ConstraintBlock(np.concatenate(rows_l), np.concatenate(cols_l), np.concatenate(data_l),
                           np.array(b_l, dtype=np.float64), num_rows=r, nnz_per_row=nc)


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
    offset map's ``col_bases`` / groups / chunks, the per-frame scalars) and the
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
