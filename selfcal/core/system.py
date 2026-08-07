"""LSQR system assembly: parent-side orchestration.

``setup_lsqr`` builds the sparse design matrix + RHS for K offset blocks and
N sky components: it resolves the sky model + column layout, stages shared
memory, dispatches the per-batch row assembly (selfcal.core.assembly) across
a process pool, appends the global constraint blocks
(selfcal.core.constraint_builders), assembles the CSR, and drops
zero-coverage columns (early column compaction). The solver stage that
consumes the result lives in selfcal.core.solve. Also home to the post-solve
coverage/Fisher parsers.
"""
from __future__ import annotations

import logging
import os
import shutil
import tempfile

from typing import TYPE_CHECKING, NamedTuple

import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing.shared_memory import SharedMemory
from scipy.sparse import coo_matrix, csr_matrix

from .. import _state
from .layout import SystemLayout
from .spill import spill_pixel_state, PixelSpill
from ..models.sky_model import SkyModel
from .constraint_builders import (mean_offset_block, sky_damping_block,
                                  offset_damping_block)
from .assembly import _prep_lsqr_batch_worker

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = [
    "setup_lsqr",
    "SetupResult",
    "parse_pixel_counts_sky",
    "parse_pixel_fisher_sky",
    "parse_line_separability",
    "apply_line_fisher_mask",
]


class SetupResult(NamedTuple):
    """What :func:`setup_lsqr` hands back.

    Named fields (rather than a positional tuple) so that optional results
    are simply ``None`` instead of changing the tuple length — callers unpack
    by name and never branch on ``len()``.

    ``pixel_counts`` / ``pixel_fisher`` / ``pixel_cross`` are None exactly
    when ``pixel_spill`` is set: the arrays are parked on scratch disk and the
    caller restores them (once) when it needs them. ``active_mask`` is None
    unless the early zero-coverage-column compaction ran (see
    ``compact_zero_columns`` in :func:`setup_lsqr`).
    """
    A: object
    b: object
    pixel_counts: object
    pixel_fisher: object
    pixel_cross: object
    active_mask: object = None
    pixel_spill: object = None


def setup_lsqr(file_list: list[str], ref_shape: tuple[int, int],
               chunk_maps: list[np.ndarray] | None = None,
               grid_valid_weight: np.ndarray | None = None,
               apply_mask: bool = True, apply_weight: bool = False,
               valid_threshold: float = 0.99,
               outlier_thresh: float | None = 3, outlier_subchannel_edges=None,
               max_workers: int = 20,
               ignore_list: list[int] | None = None, oversample_factor: int = 1,
               batch_size: int = 10, offset_regularization: bool = False,
               reg_weights: list[float] | None = None, adj_infos: list | None = None,
               poly_constraints_list: list | None = None,
               mean_offsets_list: list | None = None, det_groups_list: list | None = None,
               det_templates: list | None = None,
               poly_basis_list: list | None = None,
               use_per_frame_scalar: bool = False,
               postprocess_func: Callable | None = None,
               preprocess_func: Callable | None = None,
               weighted_damping: bool = False, damp_weight: float = 0.1,
               damp_offset: float = 0.0,
               det_aux: list[np.ndarray] | None = None,
               spectral_fit: bool = False, line_center: float | None = None,
               line_sigma: float | None = None,
               damp_weight_line: float | None = None,
               sky_model: SkyModel | None = None,
               compact_zero_columns: bool = True,
               batch_spill_dir: str | None = None) -> SetupResult:
    """Prepares the LSQR matrix A and vector b for all subframes in parallel.

    ``batch_spill_dir``: when set, workers stream each batch's bulk COO
    arrays (rows/cols/data, 12 B per entry — the dominant setup-phase
    resident) to files under this directory instead of SharedMemory, and
    the main process reads them back as read-only memmaps. The bytes and
    every accumulation order are identical (byte-equal outputs); the
    difference is that the ~12 B/nnz batch payload becomes page cache —
    reclaimable under memory pressure — instead of anonymous RAM + tmpfs.
    Files are deleted as each batch is scattered. ``b`` stays on the
    SharedMemory path (its dtype varies per batch and it is ~25x smaller).
    Default None keeps the pure-SharedMemory behavior.

    The model is ``d_i = s(p_i) + Σ_m o^(m)[g_m(k), c_m(i)] + ε``: K independent
    additive offset blocks, each with its own chunk map, frame-to-group mapping,
    template, regularization, and mean-offset constraint. With K=1 the
    multi-map machinery reduces exactly to a plain single-chunk-map solve
    with no extra terms.

    Build phases (the numbered comment sections in the function body):
    1 collate worker batch results; 2a–2e row bookkeeping, constraint blocks,
    b, indptr; 3 allocate the CSR buffers; 4a/4b scatter data + constraint
    rows into them; 5 finalize the CSR / BlockCSR.

    Parameters
    ----------
    file_list : list
        List of paths to the reprojected HDF5 files.
    ref_shape : tuple, list
        Shape of the reference frame (height, width).
    chunk_maps : list of np.ndarray or None
        K chunk maps (each must be 0-indexed and contiguous). All maps must
        share the same shape. ``None`` or an empty list disables the offset
        block entirely.
    grid_valid_weight : np.ndarray, optional
        Weight indicating valid pixels for each grid pixel.
    reg_weights : list of float or None
        Per-map adjacency regularization weights (length K). Defaults to all 0.
    adj_infos : list of tuple or None
        Per-map precomputed adjacency information (length K). Each entry is a
        ``(chunk_i, chunk_j)`` tuple or ``None``.
    poly_constraints_list : list or None
        Per-map polynomial-order constraint groups (length K). Each entry is
        ``None`` (no poly constraints on this map) or a list of dicts, each
        ``{'chains': (num_chains, L) int ndarray, 'stencil': (L,) float
        ndarray, 'weight': float}``. The constraint is
        ``λ · Σ_ℓ stencil[ℓ] · o[chains[r, ℓ]] = 0`` per chain r per frame.
        Generalizes the constant-prior ``adj_infos`` (stencil=[1,-1]) to
        arbitrary finite-difference operators. Skipped in template mode.
    mean_offsets_list : list or None
        Per-map mean-offset constraint targets (length K). Each entry is a
        length-num_frames array or ``None`` to disable for that map.
    det_groups_list : list or None
        Per-map frame→group labels (length K). Each entry is ``None`` (one
        group per frame) or an array of length num_frames. A single per-frame
        scalar bias column is added if any map uses det_groups (or if
        ``use_per_frame_scalar`` is True).
    use_per_frame_scalar : bool, optional
        When True, add a per-frame scalar column (one per frame) even when
        no map uses ``det_groups`` — useful for absorbing per-frame DC into
        an explicit column so the per-frame chunk offsets only carry
        within-frame structure. Pair with ``mean_offsets_list=[zeros]`` and
        a zero-chunk x0 init to push DC into the scalar.
    det_templates : list or None
        Per-map fixed spatial templates (length K). When set for map m, that
        map solves only for a per-frame amplitude α[k] (block size = num_frames).
    compact_zero_columns : bool, optional
        Enable the early drop of zero-coverage columns from the assembled
        CSR (default True); ``apply_lsqr`` then skips its own full-nnz
        column elimination. Automatically skipped when any map uses template
        mode, or when a constraint row touches an otherwise-uncovered
        column. Set False to keep the uncompacted column layout and let
        ``apply_lsqr`` compact instead (debug aid for isolating a suspected
        regression to the compaction step).
    apply_mask : bool, optional
        Apply each subframe's data-quality mask when reading it (default True).
    apply_weight : bool, optional
        Weight each pixel's data row by its ``valid_weight`` instead of unit
        weight (default False).
    valid_threshold : float, optional
        Minimum valid fraction for a reprojected pixel to be kept (default 0.99).
    outlier_thresh : float or None, optional
        Sigma threshold for per-subframe outlier rejection; ``None`` disables it.
    max_workers : int, optional
        Number of worker processes in the row-assembly pool (default 20).
    ignore_list : list of int or None, optional
        Data-quality flag values to treat as invalid; ``None``/empty ignores none.
    oversample_factor : int, optional
        Sub-pixel oversampling factor applied when reading each subframe.
    batch_size : int, optional
        Number of subframes assembled per worker batch (default 10).
    offset_regularization : bool, optional
        Enable the per-map adjacency/offset regularization rows.
    poly_basis_list : list or None, optional
        Per-map hard polynomial basis (length K). When set for map m, that map
        solves for polynomial coefficients (shape-only; DC absorbed by the
        per-frame scalar) rather than per-chunk offsets.
    postprocess_func : callable or None, optional
        Optional per-subframe postprocessing hook run inside the worker.
    preprocess_func : callable or None, optional
        Optional per-subframe preprocessing hook run inside the worker.
    weighted_damping : bool, optional
        Enable coverage-weighted Tikhonov damping of the sky blocks.
    damp_weight : float, optional
        Coverage-weighted damping weight for the continuum sky block (default 0.1).
    damp_offset : float, optional
        Coverage-weighted damping weight for the offset columns; 0 disables it.
    det_aux : list of np.ndarray or None, optional
        Detector-grid auxiliary maps ``[BC_map]`` (optionally ``[BC_map, BW_map]``)
        required by a spectral ``sky_model`` to evaluate the line coefficient per
        sub-pixel.
    spectral_fit : bool, optional
        Deprecated shim: when True, lowers to a continuum+PAH-Gaussian
        ``SkyModel``. Prefer passing ``sky_model=`` explicitly.
    line_center : float or None, optional
        Line-center wavelength for the ``spectral_fit`` shim.
    line_sigma : float or None, optional
        Line Gaussian sigma for the ``spectral_fit`` shim.
    damp_weight_line : float or None, optional
        Damping weight for spectral (line) sky blocks; defaults to
        ``3 * damp_weight`` when a spectral model is active.
    sky_model : SkyModel or None, optional
        Forward-looking sky-model object driving per-pixel sky row emission;
        defaults to continuum-only (or continuum+PAH when ``spectral_fit``).
    batch_spill_dir : str or None, optional
        Directory for streaming each batch's bulk COO arrays to page-cache-backed
        files instead of SharedMemory (see the note above); ``None`` keeps the
        pure-SharedMemory behavior.

    Returns
    -------
    result : SetupResult
        Named tuple ``(A, b, pixel_counts, pixel_fisher, pixel_cross,
        active_mask, pixel_spill)``. Returns ``(None, None)`` instead when no
        valid data is found in any subframe.
    """
    # Mutable-default normalization: an empty ignore_list means "ignore nothing".
    if ignore_list is None:
        ignore_list = []

    # Entry validation of caller-supplied arguments. These must survive
    # ``python -O`` (asserts do not), so they are explicit raises: pure type
    # checks -> TypeError, value/shape/length/positivity checks -> ValueError.
    if not (isinstance(file_list, (list, np.ndarray)) and file_list):
        raise ValueError("file_list must be a non-empty list")
    if not (isinstance(ref_shape, (list, np.ndarray, tuple)) and len(ref_shape) == 2):
        raise ValueError("ref_shape must be a list of length 2")
    if not (grid_valid_weight is None or isinstance(grid_valid_weight, np.ndarray)):
        raise TypeError("grid_valid_weight must be a numpy array")
    if not isinstance(apply_mask, bool):
        raise TypeError("apply_mask must be a boolean")
    if not isinstance(apply_weight, bool):
        raise TypeError("apply_weight must be a boolean")
    if not (isinstance(outlier_thresh, (int, float, type(None))) and (outlier_thresh is None or outlier_thresh > 0)):
        raise ValueError("outlier_thresh must be a positive number or None")
    if not (isinstance(max_workers, int) and max_workers > 0):
        raise ValueError("max_workers must be a positive integer")
    if not isinstance(ignore_list, (list, np.ndarray)):
        raise TypeError("ignore_list must be a list or array of data quality flags to ignore")
    if not (isinstance(batch_size, int) and batch_size > 0):
        raise ValueError("batch_size must be a positive integer")

    # Normalize chunk_maps and per-map arguments to length-K lists.
    if chunk_maps is None:
        chunk_maps = []
    if not isinstance(chunk_maps, list):
        raise TypeError("chunk_maps must be a list")
    for cm in chunk_maps:
        if not isinstance(cm, np.ndarray):
            raise TypeError("every chunk_maps entry must be a numpy array")
    K = len(chunk_maps)

    def _default(x, fill):
        return [fill] * K if x is None else x

    reg_weights = _default(reg_weights, 0.0)
    adj_infos = _default(adj_infos, None)
    poly_constraints_list = _default(poly_constraints_list, None)
    mean_offsets_list = _default(mean_offsets_list, None)
    det_groups_list = _default(det_groups_list, None)
    det_templates = _default(det_templates, None)
    poly_basis_list = _default(poly_basis_list, None)

    # Normalize and validate poly-constraint groups: each entry is None or a
    # non-empty list of dicts; each dict has matching chains.shape[1] == len(stencil).
    # Cast chains to int64 and stencil to float64 once here so workers don't
    # re-cast per call.
    normalized_poly = []
    for m, groups in enumerate(poly_constraints_list):
        if groups is None:
            normalized_poly.append(None)
            continue
        norm_groups = []
        for g_idx, grp in enumerate(groups):
            chains = np.asarray(grp['chains'], dtype=np.int64)
            stencil = np.asarray(grp['stencil'], dtype=np.float64)
            weight = float(grp['weight'])
            if chains.ndim != 2:
                raise ValueError(
                    f"poly_constraints_list[{m}][{g_idx}]['chains'] must be 2-D")
            if stencil.ndim != 1:
                raise ValueError(
                    f"poly_constraints_list[{m}][{g_idx}]['stencil'] must be 1-D")
            if chains.shape[1] != stencil.shape[0]:
                raise ValueError(
                    f"poly_constraints_list[{m}][{g_idx}]: chains.shape[1]="
                    f"{chains.shape[1]} != len(stencil)={stencil.shape[0]}")
            norm_groups.append({'chains': chains, 'stencil': stencil, 'weight': weight})
        normalized_poly.append(norm_groups if norm_groups else None)
    poly_constraints_list = normalized_poly

    # Adjacency tuples with all-empty arrays (e.g. NumCol=1 from
    # compute_column_adjacency) produce zero adjacency constraints anyway —
    # demote to None so the SHM packing below doesn't try to create a
    # zero-byte segment, which raises ValueError.
    adj_infos = [
        None if (adj is not None and all(np.asarray(a).size == 0 for a in adj)) else adj
        for adj in adj_infos
    ]
    for name, arr in (('reg_weights', reg_weights), ('adj_infos', adj_infos),
                      ('poly_constraints_list', poly_constraints_list),
                      ('mean_offsets_list', mean_offsets_list),
                      ('det_groups_list', det_groups_list),
                      ('det_templates', det_templates),
                      ('poly_basis_list', poly_basis_list)):
        if len(arr) != K:
            raise ValueError(f"{name} must have length {K} (got {len(arr)})")

    ref_h, ref_w = ref_shape
    num_sky = ref_h * ref_w
    num_frames = len(file_list)

    # --- Spectral-fit mode: 2-block sky (continuum + line amplitude per pixel) ---
    # When spectral_fit is True, the sky block grows from num_sky to 2*num_sky:
    # x[:num_sky] is the continuum sky map, x[num_sky:2*num_sky] is the line
    # amplitude map (PAH 3.29 μm by default). The data row for one observation
    # of ref pixel P at LVF wavelength λ_i gains a second sky nnz:
    #
    #   data_i = w_i * (sky_cont[P] + G(λ_i) * sky_line[P]) + offsets + scalar
    #
    # where G(λ) is the Gaussian line profile (peak = 1 at line_center,
    # sigma = line_sigma). λ_i is sampled per (frame, sub-pixel) via the
    # det_aux plumbing: BC_map must be passed as det_aux[0]. Optionally
    # det_aux[1] = BW_map gives per-pixel σ (mixed with PAH intrinsic).
    # --- Sky model resolution ---
    # sky_model= is the forward-looking API; the spectral_fit flag (+
    # line_center / line_sigma) is a deprecated shim that lowers to the
    # equivalent SkyModel, so callers using the flags get an identical system
    # to passing that model explicitly. The model's components drive the
    # per-pixel sky row emission in the worker (continuum -> J=1 identity
    # fast path; +line -> interleave with the profile coefficient).
    if sky_model is None:
        if spectral_fit:
            sky_model = SkyModel.continuum_plus_pah_gaussian(line_center, line_sigma)
        else:
            sky_model = SkyModel.continuum_only()
    num_sky_blocks = sky_model.n_blocks
    if num_sky_blocks > 1:
        # A spectral SkyModel (>=1 non-continuum block) needs the wavelength aux
        # map(s) and gets decoupled line-block damping by default (the line
        # columns have smaller average coefficients than continuum, so ~3x more
        # Tikhonov shrinkage at the same data S/N).
        if damp_weight_line is None:
            damp_weight_line = 3.0 * damp_weight
        if det_aux is None or len(det_aux) < 1:
            raise ValueError(
                "A spectral SkyModel (>1 sky block) requires det_aux=[BC_map] "
                "(or [BC_map, BW_map] for per-pixel σ). Pass BC_map from "
                "selfcal.instruments.spherex.spherex_utility.load_calibration(band=detector).")
        logger.info(f"Spectral mode ON: {num_sky_blocks} sky blocks {sky_model.names}, "
                    f"{num_sky_blocks * num_sky} sky cols, damp_weight_line={damp_weight_line}.")
    # Positional det_aux -> named aux dict (SPHEREx convention: [BC, BW]).
    aux_keys = ['BC', 'BW'][:len(det_aux)] if det_aux is not None else []

    # --- Column layout (single source of truth: selfcal.core.layout.SystemLayout) ---
    # SystemLayout computes the per-map group mapping, template normalization,
    # col_bases, the per-frame scalar block, and the total column count. The
    # Calibrator builds the same layout from the same inputs (see
    # pipeline_wrapper.Calibrator.setup_lsqr) so the parent-side and parse-side
    # column arithmetic can never drift.
    any_det_groups = any(g is not None for g in det_groups_list)
    layout = SystemLayout.build(
        ref_shape, chunk_maps, num_sky_blocks=num_sky_blocks, num_frames=num_frames,
        det_groups_list=det_groups_list, det_templates=det_templates,
        use_per_frame_scalar=use_per_frame_scalar, poly_basis_list=poly_basis_list)
    frame_to_group_list = layout.frame_to_group_list
    num_offset_groups_list = layout.num_offset_groups_list
    num_chunks_list = layout.num_chunks_list
    det_template_arr_list = layout.det_template_arr_list
    num_scalar_cols = layout.num_scalar_cols
    col_bases = layout.col_bases
    scalar_col_start = layout.scalar_col_start
    total_cols = layout.total_cols

    if any_det_groups or use_per_frame_scalar:
        logger.info(f"Locking detector offsets: {num_frames} frames -> "
                    f"groups {num_offset_groups_list} + {num_frames} frame scalars")
    if any(t is not None for t in det_template_arr_list):
        tmpl_indices = [m for m, t in enumerate(det_template_arr_list) if t is not None]
        logger.info(f"Template mode for maps {tmpl_indices}: {num_frames} alpha unknowns each")

    common_params = {
        'chunk_maps': chunk_maps,
        'grid_valid_weight': grid_valid_weight,
        'apply_mask': apply_mask,
        'apply_weight': apply_weight,
        'ignore_list': ignore_list,
        'oversample_factor': oversample_factor,
        'valid_threshold': valid_threshold,
        'outlier_thresh': outlier_thresh,
        'outlier_subchannel_edges': outlier_subchannel_edges,
        'num_chunks_list': num_chunks_list,
        'num_frames': num_frames,
        'ref_shape': ref_shape,
        'offset_regularization': offset_regularization,
        'reg_weight_list': reg_weights,
        'adj_info_list': adj_infos,
        'poly_constraint_list': poly_constraints_list,
        'poly_basis_list': poly_basis_list,
        'postprocess_func': postprocess_func,
        'preprocess_func': preprocess_func,
        'frame_to_group_list': frame_to_group_list,
        'col_bases': col_bases,
        'scalar_col_start': scalar_col_start,
        'num_scalar_cols': num_scalar_cols,
        'det_template_list': det_template_arr_list,
        'num_sky_blocks': num_sky_blocks,
        'sky_components': sky_model.components,
        'aux_keys': aux_keys,
        'line_center': line_center,
        'line_sigma': line_sigma,
    }

    # Move large arrays to shared memory so forked processes can access them
    # without pickling. Each process reconstructs numpy views in the worker.
    shm_objects = []

    if K > 0:
        chunk_maps_meta = []
        for cm in chunk_maps:
            shm_cm = SharedMemory(create=True, size=cm.nbytes)
            np.ndarray(cm.shape, dtype=cm.dtype, buffer=shm_cm.buf)[:] = cm
            shm_objects.append(shm_cm)
            chunk_maps_meta.append((shm_cm.name, cm.shape, cm.dtype))
        common_params['chunk_maps_meta'] = chunk_maps_meta
        common_params['chunk_maps'] = None  # populated by worker from SHM

    if det_aux is not None:
        # Pack each detector-grid array into SHM. Workers reconstruct via
        # det_aux_metas. Mirrors the chunk_maps_meta pattern above. Used by
        # spectral-fit mode to expose BC_map (and optionally BW_map) at row
        # assembly time so the line-amplitude column coefficient can be
        # evaluated per sub-pixel.
        det_aux_metas = []
        for arr in det_aux:
            arr_f32 = np.ascontiguousarray(arr, dtype=np.float32)
            shm_da = SharedMemory(create=True, size=arr_f32.nbytes)
            np.ndarray(arr_f32.shape, dtype=arr_f32.dtype, buffer=shm_da.buf)[:] = arr_f32
            shm_objects.append(shm_da)
            det_aux_metas.append((shm_da.name, arr_f32.shape, arr_f32.dtype))
        common_params['det_aux_metas'] = det_aux_metas
        common_params['det_aux'] = None  # populated by worker from SHM

    if grid_valid_weight is not None:
        shm_gvw = SharedMemory(create=True, size=grid_valid_weight.nbytes)
        np.ndarray(grid_valid_weight.shape, dtype=grid_valid_weight.dtype, buffer=shm_gvw.buf)[:] = grid_valid_weight
        shm_objects.append(shm_gvw)
        common_params['gvw_shm_name'] = shm_gvw.name
        common_params['gvw_shape'] = grid_valid_weight.shape
        common_params['gvw_dtype'] = grid_valid_weight.dtype
        common_params['grid_valid_weight'] = None

    if any(a is not None for a in adj_infos):
        adj_metas = []
        for adj in adj_infos:
            if adj is None:
                adj_metas.append(None)
                continue
            per_map = []
            for arr in adj:
                shm = SharedMemory(create=True, size=arr.nbytes)
                np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)[:] = arr
                shm_objects.append(shm)
                per_map.append((shm.name, arr.shape, arr.dtype))
            adj_metas.append(per_map)
        common_params['adj_metas'] = adj_metas
        common_params['adj_info_list'] = None  # populated by worker from SHM

    all_individual_tasks = []
    for index, reproj_file in enumerate(file_list):
        task_params = {'index': index, 'reproj_file': reproj_file}
        task_params.update(common_params)
        all_individual_tasks.append(task_params)

    _spill_run_dir = None
    if batch_spill_dir is not None:
        os.makedirs(batch_spill_dir, exist_ok=True)
        _spill_run_dir = tempfile.mkdtemp(prefix='batch_spill_',
                                          dir=batch_spill_dir)
        logger.info(f"Batch COO spill -> {_spill_run_dir} (page-cache backed).")

    batched_tasks = []
    for i in range(0, len(all_individual_tasks), batch_size):
        batch = {'sub_tasks': all_individual_tasks[i : i + batch_size],
                 'batch_id': i // batch_size,
                 'spill_dir': _spill_run_dir}
        batched_tasks.append(batch)

    logger.info(f"Processing {len(all_individual_tasks)} items in {len(batched_tasks)} batches...")

    # Per-batch streaming accumulators for pixel_counts and pixel_fisher.
    # Allocated up front and accumulated batch-by-batch as worker results
    # arrive, so no full-nnz temporary is ever materialized (a post-loop
    # bincount would need a full-nnz float64 squared-data temp — 8 B per
    # matrix entry, i.e. tens of GB once nnz reaches several 1e9).
    pixel_counts = np.zeros(total_cols, dtype=np.int64)
    pixel_fisher = np.zeros(total_cols, dtype=np.float64)
    # Per-pixel sky-block cross moments Σ a_i·a_j per pair (i, j), i < j
    # (a_0 = w for the continuum, a_j = w·coeff_j for spectral blocks).
    # Together with pixel_fisher's per-block diagonals these give the per-pixel
    # J x J normal-matrix block, whose LAST-block Schur complement is the true
    # line SEPARABILITY I_P — which line-Fisher (a magnitude, not a diversity,
    # metric) cannot measure. Accumulated for multi-block sky models; computed
    # batch-streaming from the A triplets, so it never touches the workers or
    # the A/b bytes. Returned as a bare (num_sky,) array for the 2-block case
    # (pair (0,1)), a {(i,j): array} dict for J >= 3.
    if num_sky_blocks >= 2:
        _cross_pairs = [(i, j) for i in range(num_sky_blocks)
                        for j in range(i + 1, num_sky_blocks)]
        pixel_cross = {p: np.zeros(num_sky, dtype=np.float64) for p in _cross_pairs}
    else:
        pixel_cross = None

    def _read_shm(info):
        """Read array from shared memory and clean up the segment."""
        name, shape, dtype = info
        shm = SharedMemory(name=name)
        arr = np.ndarray(shape, dtype=dtype, buffer=shm.buf).copy()
        shm.close()
        shm.unlink()
        return arr

    # ----------------------------------------------------------------
    # Phase 1: collate worker results.
    #
    # We retain each batch's (rows, cols, data, b) so Phase 4 can scatter
    # them directly into the final CSR buffers in batch-id order. Per-batch
    # we also accumulate row_nnz_per_batch[batch_id] = bincount(local_rows)
    # so the CSR indptr can be built without re-touching every batch's row
    # array. pixel_counts / pixel_fisher / pixel_cross are accumulated
    # batch-streaming as each result arrives (allocation and rationale above).
    # ----------------------------------------------------------------
    batch_results = [None] * len(batched_tasks)
    row_nnz_per_batch = [None] * len(batched_tasks)
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(_prep_lsqr_batch_worker, batch): i
                       for i, batch in enumerate(batched_tasks)}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Building A, b matrix",
                               disable=not _state.progress_enabled):
                batch_id = futures[future]
                result = future.result()
                if result is None:
                    continue
                shm_infos = result['shm']
                if 'files' in result:
                    # Spill path: bulk arrays live in files; read-only
                    # memmaps expose identical bytes through page cache.
                    fr, fc, fd = result['files']
                    batch_results[batch_id] = {
                        'rows': np.memmap(fr, dtype=np.int32, mode='r'),
                        'cols': np.memmap(fc, dtype=np.int32, mode='r'),
                        'data': np.memmap(fd, dtype=np.float32, mode='r'),
                        'b':    _read_shm(shm_infos[0]),
                        'num_rows': result['num_rows'],
                        'files': result['files'],
                    }
                else:
                    batch_results[batch_id] = {
                        'rows': _read_shm(shm_infos[0]),     # local int32 row ids
                        'cols': _read_shm(shm_infos[1]),
                        'data': _read_shm(shm_infos[2]),
                        'b':    _read_shm(shm_infos[3]),
                        'num_rows': result['num_rows'],
                    }
                _b_cols = batch_results[batch_id]['cols']
                _b_data = batch_results[batch_id]['data']
                _b_rows = batch_results[batch_id]['rows']
                # Per-batch streaming accumulation. Avoids holding a
                # full-nnz float64 squared-data temp later.
                pixel_counts += np.bincount(_b_cols, minlength=total_cols)
                pixel_fisher += np.bincount(
                    _b_cols,
                    weights=_b_data.astype(np.float64) ** 2,
                    minlength=total_cols,
                )
                if pixel_cross is not None:
                    # Pair each data row's block-i entry with its block-j entry
                    # through the shared local row id -> Σ a_i·a_j per pixel.
                    # Every data row has exactly one entry per sky block (the
                    # J-interleave in assembly), so a per-row scatter of block
                    # i's values indexes cleanly from block j's entries.
                    _masks = [(_b_cols >= jj * num_sky) & (_b_cols < (jj + 1) * num_sky)
                              for jj in range(num_sky_blocks)]
                    _rowvals = []
                    for jj in range(num_sky_blocks):
                        _v = np.zeros(result['num_rows'], dtype=np.float64)
                        _v[_b_rows[_masks[jj]]] = _b_data[_masks[jj]]
                        _rowvals.append(_v)
                    for (ii, jj) in _cross_pairs:
                        _m_j = _masks[jj]
                        pixel_cross[(ii, jj)] += np.bincount(
                            _b_cols[_m_j] - jj * num_sky,
                            weights=_b_data[_m_j].astype(np.float64)
                                    * _rowvals[ii][_b_rows[_m_j]],
                            minlength=num_sky,
                        )
                # Per-batch row nnz over LOCAL row ids (0..num_rows-1). We
                # add the global row offset in Phase 3 (cumulative across
                # batches). Keeping this batch-local is what lets Phase 4
                # stream-scatter without revisiting all batches twice.
                row_nnz_per_batch[batch_id] = np.bincount(
                    _b_rows, minlength=result['num_rows']
                ).astype(np.int32, copy=False)
    finally:
        for shm in shm_objects:
            shm.close()
            shm.unlink()

    # ----------------------------------------------------------------
    # Phase 2: compute per-batch row offsets, total_rows_data, and the
    # data-row half of row_nnz.
    # ----------------------------------------------------------------
    batch_row_starts = [0] * len(batched_tasks)
    total_rows_data = 0
    any_kept = False
    for batch_id in range(len(batched_tasks)):
        batch_row_starts[batch_id] = total_rows_data
        r = batch_results[batch_id]
        if r is None:
            continue
        any_kept = True
        total_rows_data += r['num_rows']

    if not any_kept:
        logger.warning("No valid data found in any subframe.")
        return None, None

    # Ensure Python int to avoid numpy int32 overflow on big runs.
    total_rows_data = int(total_rows_data)

    # ----------------------------------------------------------------
    # Phase 2b: build constraint blocks (rows-local-to-block, cols, data,
    # b). We assemble them up front so we can pre-count their per-row nnz,
    # finalize total_rows, and allocate the CSR buffers once.
    # ----------------------------------------------------------------
    num_sky_eff = num_sky_blocks * num_sky
    sky_pixel_counts = pixel_counts[:num_sky]                       # continuum coverage
    if num_sky_blocks == 2:
        line_pixel_counts = pixel_counts[num_sky:2*num_sky]         # line amplitude coverage
    else:
        line_pixel_counts = None
    offset_pixel_counts = pixel_counts[num_sky_eff:]

    # Global constraint blocks (see selfcal.constraint_builders). Emission order
    # is load-bearing for the CSR scatter: mean-offset anchors (per map) ->
    # sky damping (continuum, then line blocks) -> offset damping.
    constraint_blocks = []

    # --- Per-frame mean-offset constraints (one block per chunk map) ---
    #TODO: Pass weight from higher level instead of hardcoding here
    constraint_weight = 10.0
    for m in range(K):
        mean_off = mean_offsets_list[m]
        if mean_off is None:
            continue
        if det_template_arr_list[m] is not None:
            logger.warning(f"Skipping mean-offset constraint for map {m}: template mode does not have per-chunk offsets")
            continue
        if poly_basis_list[m] is not None:
            logger.warning(f"Skipping mean-offset constraint for map {m}: hard poly-basis is shape-only (DC in the scalar)")
            continue
        logger.info(f"Applying target mean offset constraints for map {m} ({num_frames} frames)...")
        constraint_blocks.append(mean_offset_block(
            m, mean_off, num_frames, num_chunks_list[m], frame_to_group_list[m],
            col_bases, weight=constraint_weight).as_dict())

    # --- Coverage-weighted sky damping (continuum, then each line block) ---
    if weighted_damping and damp_weight > 0:
        logger.info("Applying Coverage-Weighted Damping (continuum)...")
        blk = sky_damping_block(0, damp_weight, sky_pixel_counts, num_sky)
        if blk is not None:
            constraint_blocks.append(blk.as_dict())

        # --- SPECTRAL-BLOCK DAMPING (blocks 1..J-1) ---
        # Each spectral component is damped by its own ``damp_weight`` when the
        # component sets one, else by the shared ``damp_weight_line`` (for
        # J == 2 this reduces exactly to the single shared ``damp_weight_line``).
        for j in range(1, num_sky_blocks):
            comp = sky_model.components[j]
            w_j = getattr(comp, 'damp_weight', None)
            if w_j is None:
                w_j = damp_weight_line
            if w_j is None or w_j <= 0:
                continue
            logger.info(f"Applying Coverage-Weighted Damping ({comp.name}, damp={w_j})...")
            blk = sky_damping_block(
                j, w_j, pixel_counts[j * num_sky:(j + 1) * num_sky], num_sky)
            if blk is not None:
                constraint_blocks.append(blk.as_dict())

    # --- Coverage-weighted offset damping ---
    if damp_offset > 0:
        logger.info(f"Applying Coverage-Weighted Offset Damping (damp_offset={damp_offset})...")
        n_offset_cols = scalar_col_start - num_sky_eff
        blk = offset_damping_block(damp_offset, offset_pixel_counts[:n_offset_cols], num_sky_eff)
        if blk is not None:
            constraint_blocks.append(blk.as_dict())


    # ----------------------------------------------------------------
    # Phase 2c: finalize total_rows + build row_nnz over the entire row
    # space (data rows first, then each constraint block).
    # ----------------------------------------------------------------
    constraint_row_total = sum(blk['num_rows'] for blk in constraint_blocks)
    total_rows = total_rows_data + constraint_row_total

    row_nnz = np.empty(total_rows, dtype=np.int32)
    # Data-row half: concat per-batch row_nnz in batch-id order. None entries
    # contribute zero rows so they simply don't appear.
    cursor = 0
    for batch_id in range(len(batched_tasks)):
        rn = row_nnz_per_batch[batch_id]
        if rn is None:
            continue
        row_nnz[cursor:cursor + rn.size] = rn
        cursor += rn.size
    assert cursor == total_rows_data, (cursor, total_rows_data)
    # Constraint-row half: each block is uniform nnz_per_row.
    for blk in constraint_blocks:
        row_nnz[cursor:cursor + blk['num_rows']] = blk['nnz_per_row']
        cursor += blk['num_rows']
    assert cursor == total_rows
    # Free per-batch row_nnz; no longer needed.
    row_nnz_per_batch = None

    # ----------------------------------------------------------------
    # Phase 2d: build full_b (vector of right-hand-sides). Concatenated in
    # the same order as the row scatter: per-batch b in batch-id order,
    # then each constraint block's b.
    # ----------------------------------------------------------------
    b_pieces = []
    for batch_id in range(len(batched_tasks)):
        r = batch_results[batch_id]
        if r is None:
            continue
        b_pieces.append(r['b'])
    for blk in constraint_blocks:
        b_pieces.append(blk['b'])
    # Emit float32 b when EVERY value is exactly float32-representable
    # (data-row b is a product of f32s; f64-ness normally enters only via
    # constraint-block zeros / mean-offset targets). Consumers restore the
    # old bit-exact arithmetic by upcasting where f64 mattered:
    # compute_x0_* upcast b before their products, and apply_lsqr upcasts
    # back to f64 for use_float32=False solves. Any non-representable
    # value (e.g. arbitrary mean-offset targets) falls back to f64.
    _can_f32 = all(
        (p.dtype == np.float32
         or p.size == 0
         or np.array_equal(p, p.astype(np.float32).astype(np.float64)))
        for p in b_pieces)
    if b_pieces and _can_f32:
        full_b = np.concatenate(
            [p.astype(np.float32, copy=False) for p in b_pieces])
    elif b_pieces:
        full_b = np.concatenate(b_pieces)
    else:
        full_b = np.zeros(0, dtype=np.float64)
    # Drop per-batch b refs so the streaming scatter loop holds the
    # smallest possible footprint.
    for batch_id in range(len(batched_tasks)):
        r = batch_results[batch_id]
        if r is not None:
            r['b'] = None
    del b_pieces
    for blk in constraint_blocks:
        blk['b'] = None

    # ----------------------------------------------------------------
    # Phase 2e: build CSR indptr (int64) + total_nnz.
    # ----------------------------------------------------------------
    indptr = np.zeros(total_rows + 1, dtype=np.int64)
    np.cumsum(row_nnz, out=indptr[1:])
    total_nnz = int(indptr[-1])
    # row_nnz no longer needed; we will use indptr+write_cursor for scatter.
    del row_nnz

    # ----------------------------------------------------------------
    # Early column compaction. If no template-mode map is in use, every
    # "active" column (pixel_counts > 0) gets compacted now so apply_lsqr
    # can skip its full-nnz col_map gather. Template-mode runs keep the
    # uncompacted layout (apply_lsqr handles compaction itself).
    # ----------------------------------------------------------------
    compaction_active = not any(t is not None for t in det_template_arr_list)
    # Debug knob: caller can force compaction off to compare against the
    # uncompacted-CSR layout — useful for isolating a suspected regression
    # to the compaction step itself. Default True is the production path.
    if not compact_zero_columns:
        compaction_active = False
    if compaction_active:
        # Some non-template runs still need the uncompacted path — e.g. when
        # the caller passes mean_offsets_list (per-frame chunk constraint rows
        # write into specific column indices that must not be dropped).
        # mean_offsets / damping rows already register coverage via the
        # data rows, so columns they touch will have pixel_counts > 0.
        # Constraint rows pointing at otherwise-uncovered cols would still
        # need those slots; guard with a coverage check.
        active_mask = pixel_counts > 0
        # Verify every constraint col is "active" — otherwise gating must
        # bail out and let apply_lsqr handle compaction.
        all_constraint_cols_active = True
        for blk in constraint_blocks:
            cols_b = blk['cols']
            if cols_b.size and not active_mask[cols_b].all():
                all_constraint_cols_active = False
                break
        if not all_constraint_cols_active:
            compaction_active = False

    if compaction_active:
        n_active = int(active_mask.sum())
        assert n_active < 2**31, (
            f"n_active={n_active} overflows int32 column ids")
        # int32 halves both the map itself (total_cols entries) and every
        # per-batch col_map[cols_b] gather output in Phase 4. Values are
        # exact (cumsum <= n_active < 2^31).
        col_map = np.cumsum(active_mask, dtype=np.int32)
        col_map -= 1
        logger.info(f"Compacting zero-coverage columns inline ({n_active}/{total_cols} active).")
    else:
        active_mask = None
        col_map = None
        n_active = total_cols

    # ----------------------------------------------------------------
    # Park the pixel state for the CSR build. pixel_counts/fisher/cross were
    # last read by the Phase-2 constraint builders and the column compaction
    # just above; nothing between here and the return touches them, yet they
    # are large — 16 B per column (counts + fisher) plus 8 B x num_sky per
    # cross pair, e.g. ~17 GB for a 4-sky-block model on a ~1.5e8-pixel
    # grid — and they would otherwise sit on top of the process-lifetime
    # memory peak (the Phase-5 CSR/BlockCSR build below). A round trip
    # through scratch disk removes them from that peak; the arrays come
    # back bit-identical (np.save/np.load of int64/float64), and
    # Calibrator.apply_lsqr spills them again for the solve.
    # ----------------------------------------------------------------
    _pix_spill_dir, _ = spill_pixel_state(pixel_counts, pixel_fisher,
                                          pixel_cross,
                                          label='for the CSR build')
    if _pix_spill_dir is not None:
        pixel_counts = pixel_fisher = pixel_cross = None

    # ----------------------------------------------------------------
    # Phase 3: allocate CSR buffers.
    # ----------------------------------------------------------------
    csr_data = np.empty(total_nnz, dtype=np.float32)
    csr_indices = np.empty(total_nnz, dtype=np.int32)
    write_cursor = np.zeros(total_rows, dtype=np.int32)

    # ----------------------------------------------------------------
    # Phase 4a: streaming scatter of data rows (one batch at a time;
    # release each batch's arrays immediately after scatter so the peak
    # footprint stays small).
    #
    # Within-batch row collisions are common (e.g. spectral_fit mode places
    # 2 sky nnz + 1 offset + 1 scalar in the SAME row). A naive
    #   slots = indptr[rows_b] + write_cursor[rows_b]
    # would have all duplicates read the same write_cursor value and
    # overwrite the same slot. We instead stable-sort by row and use a
    # within-row cumcount so every entry lands in its own slot. Stable
    # sort preserves the original col-within-row order across the batch,
    # which matches what coo_matrix(...).tocsr() does for stable sort_by_row.
    # ----------------------------------------------------------------
    # The scatter runs over SUB-SLICES of each batch (in original entry
    # order) rather than the whole batch at once: the sort/cumcount
    # machinery allocates ~45 B per entry of transients, which at full
    # batch width (~4e8 entries) is ~18-23 GB sitting exactly on the
    # setup peak. Sub-slicing divides that by the slice count.
    # Slot assignment is IDENTICAL: for entries of the same row split
    # across slices, earlier slices write first and advance write_cursor,
    # so later slices continue at the updated cursor — the concatenation
    # of per-slice stable-sort orders over a row equals the whole-batch
    # stable-sort order (stable sort preserves original order within
    # equal keys; slice boundaries preserve original order across
    # slices). Hence byte-identical csr_data/csr_indices.
    scatter_chunk = 64_000_000
    for batch_id in range(len(batched_tasks)):
        batch = batch_results[batch_id]
        if batch is None:
            continue
        row_offset = batch_row_starts[batch_id]
        n_batch = batch['rows'].shape[0]
        for s0 in range(0, n_batch, scatter_chunk):
            s1 = min(s0 + scatter_chunk, n_batch)
            rows_b = batch['rows'][s0:s1].astype(np.int64, copy=False) + row_offset
            if compaction_active:
                # int32 compact col indices (col_map is int32; gather emits
                # int32 directly, astype is a no-op).
                cols_b = col_map[batch['cols'][s0:s1]].astype(np.int32, copy=False)
            else:
                cols_b = batch['cols'][s0:s1].astype(np.int32, copy=False)
            data_b = batch['data'][s0:s1]

            n_b = rows_b.shape[0]
            if n_b > 0:
                # Stable sort by row so duplicates are contiguous; preserves
                # within-row col order from the worker output.
                order = np.argsort(rows_b, kind="stable")
                rows_s = rows_b[order]
                cols_s = cols_b[order]
                data_s = data_b[order]

                # Within-row cumcount in sorted order.
                is_new_group = np.empty(n_b, dtype=bool)
                is_new_group[0] = True
                is_new_group[1:] = rows_s[1:] != rows_s[:-1]
                group_starts = np.flatnonzero(is_new_group)
                group_idx = np.cumsum(is_new_group, dtype=np.int64) - 1
                within_row = np.arange(n_b, dtype=np.int64) - group_starts[group_idx]

                slots = indptr[rows_s] + write_cursor[rows_s] + within_row
                csr_data[slots] = data_s
                csr_indices[slots] = cols_s

                # Update write_cursor: add the per-row count contributed by
                # this slice (carries the running count to later slices).
                unique_rows = rows_s[group_starts]
                counts = np.diff(np.append(group_starts, n_b)).astype(np.int32, copy=False)
                write_cursor[unique_rows] += counts
            del rows_b, cols_b, data_b
        # Free the batch refs (and its spill files, if any).
        _files = batch.get('files')
        batch_results[batch_id] = None
        del batch
        if _files:
            for _p in _files:
                try:
                    os.remove(_p)
                except OSError:
                    pass
    batch_results = None
    if _spill_run_dir is not None:
        shutil.rmtree(_spill_run_dir, ignore_errors=True)

    # ----------------------------------------------------------------
    # Phase 4b: scatter constraint blocks at their reserved row ranges.
    #
    # Same row-collision concern as Phase 4a: e.g. the per-frame
    # mean-offset block has nc_m entries per row. Use the same
    # stable-sort + within-row cumcount pattern. write_cursor for
    # constraint rows starts at 0 (no prior batches touched them) but
    # we still write through write_cursor for uniformity.
    # ----------------------------------------------------------------
    cb_cursor = total_rows_data
    for blk in constraint_blocks:
        nrows = blk['num_rows']
        nnz_per_row = blk['nnz_per_row']
        rows_b = blk['rows_local'] + cb_cursor
        cols_b = blk['cols']
        data_b = blk['data']
        if compaction_active:
            cols_b = col_map[cols_b].astype(np.int32, copy=False)
        else:
            cols_b = cols_b.astype(np.int32, copy=False)

        n_b = rows_b.shape[0]
        if n_b > 0:
            # nnz_per_row can be a scalar (uniform) or a numpy array (varying per-row).
            is_uniform_one = (np.ndim(nnz_per_row) == 0 and int(nnz_per_row) == 1)
            if is_uniform_one:
                # Fast path: every constraint row contributes exactly one
                # entry, so no duplicates exist and no sort is required.
                slots = indptr[rows_b] + write_cursor[rows_b]
                csr_data[slots] = data_b
                csr_indices[slots] = cols_b
                write_cursor[rows_b] += 1
            else:
                order = np.argsort(rows_b, kind="stable")
                rows_s = rows_b[order]
                cols_s = cols_b[order]
                data_s = data_b[order]

                is_new_group = np.empty(n_b, dtype=bool)
                is_new_group[0] = True
                is_new_group[1:] = rows_s[1:] != rows_s[:-1]
                group_starts = np.flatnonzero(is_new_group)
                group_idx = np.cumsum(is_new_group, dtype=np.int64) - 1
                within_row = np.arange(n_b, dtype=np.int64) - group_starts[group_idx]

                slots = indptr[rows_s] + write_cursor[rows_s] + within_row
                csr_data[slots] = data_s
                csr_indices[slots] = cols_s

                unique_rows = rows_s[group_starts]
                counts = np.diff(np.append(group_starts, n_b)).astype(np.int32, copy=False)
                write_cursor[unique_rows] += counts
        cb_cursor += nrows
    del write_cursor

    # ----------------------------------------------------------------
    # Phase 5: build CSR. Indices are row-grouped but NOT col-sorted within
    # row (sufficient for LSQR/LSMR matvec/rmatvec — convergence depends
    # only on A as a linear map). We mark has_sorted_indices=False so
    # scipy callers that depend on it (e.g. .T → CSC view) can either
    # tolerate or trigger sort themselves.
    #
    # Once total nnz reaches 2**31, the unified csr_matrix constructor would
    # upcast-COPY the int32 csr_indices to int64 (indptr's dtype wins) —
    # +nnz*8 alloc at handoff and +nnz*4 held permanently, plus 50% more
    # index bytes streamed per SpMV. In that regime we emit a BlockCSR
    # instead: int32 index VIEWS sliced per row-block, each block's nnz kept
    # below 2**31. Per-row index sorting is row-local, so sorting each block
    # is bit-identical to sorting the unified matrix. The global int64
    # indptr is released (blocks carry shifted int32 copies).
    # SELFCAL_BLOCK_NNZ overrides the activation threshold AND per-block
    # target (tests force small blocks with it); production default splits
    # only when int64 would otherwise be forced.
    # ----------------------------------------------------------------
    _block_thr = int(os.environ.get('SELFCAL_BLOCK_NNZ', 2**31))
    total_nnz = int(indptr[-1])
    if total_nnz >= _block_thr:
        from .blockcsr import build_block_csr
        target = max(1, min(2**30, _block_thr))
        full_A = build_block_csr(csr_data, csr_indices, indptr,
                                 (total_rows, n_active), target)
        del csr_data, csr_indices, indptr
        logger.info(f"Phase 5: BlockCSR with {len(full_A.blocks)} int32 row-blocks "
                    f"(nnz={total_nnz}).")
        for _blk in full_A.blocks:
            _blk.has_sorted_indices = False
            _blk.sort_indices()
            _blk.sum_duplicates()
    else:
        full_A = csr_matrix(
            (csr_data, csr_indices, indptr),
            shape=(total_rows, n_active),
            copy=False,
        )
        full_A.has_sorted_indices = False
        # Sort indices within each row in place. This is per-row qsort with no
        # allocation peak and gives downstream code (CSC views, indices-binary
        # search, etc.) the canonical layout. We also call sum_duplicates() as a
        # defensive no-op (the bucket-sort build guarantees no duplicate (row,
        # col) entries) so that downstream consumers can rely on canonical CSR.
        full_A.sort_indices()
        full_A.sum_duplicates()

    # NOTE: the pixel state is deliberately NOT restored here. Restoring it
    # before returning would re-inflate it right alongside the finished CSR —
    # i.e. exactly at the peak the spill exists to avoid — and
    # Calibrator.apply_lsqr would immediately write it back out again.
    # Instead the spill directory travels to the caller, which keeps the
    # arrays parked until save_calibration actually needs them.
    #
    # J == 2 keeps the bare (num_sky,) pair-(0,1) array return for downstream
    # compatibility; J >= 3 returns the {(i, j): array} dict. When spilled,
    # that reshaping happens on restore instead (see PixelSpill.restore).
    if pixel_cross is not None and num_sky_blocks == 2:
        pixel_cross = pixel_cross[(0, 1)]
    return SetupResult(A=full_A, b=full_b,
                       pixel_counts=pixel_counts, pixel_fisher=pixel_fisher,
                       pixel_cross=pixel_cross,
                       active_mask=active_mask if compaction_active else None,
                       pixel_spill=(PixelSpill(_pix_spill_dir, num_sky_blocks)
                                    if _pix_spill_dir is not None else None))


def parse_pixel_counts_sky(pixel_counts: np.ndarray, ref_shape: tuple[int, int],
                           num_offset_groups_list: list[int],
                           chunk_maps: list[np.ndarray],
                           num_sky_blocks: int = 1,
                           num_chunks_list: list[int] | None = None
                           ) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Generic coverage slicing for any number of sky blocks.

    Returns ``(sky_coverages, offset_coverages, offset_valid_fracs)`` where
    ``sky_coverages`` is a length-``num_sky_blocks`` list of ``ref_shape`` arrays
    (block 0 = continuum, 1.. = line blocks). :func:`parse_pixel_counts` is the
    back-compat fixed-tuple wrapper for <=2 blocks.

    ``num_chunks_list`` (optional) is the *layout* column count per map — the
    number of offset columns actually allocated in ``pixel_counts``. For an
    ordinary per-chunk map this equals ``cm.max()+1`` (default when ``None``), so
    behavior is byte-identical. For template / hard-poly-basis maps the offset
    columns are amplitudes / polynomial coefficients, NOT one-per-chunk, so the
    layout count must be used to slice + advance the cursor correctly; the caller
    (``save_calibration``) overrides the coverage for those maps, so a
    shape-correct placeholder frac is returned rather than a per-chunk division.
    """
    num_sky = ref_shape[0] * ref_shape[1]
    sky_coverages = [pixel_counts[j * num_sky:(j + 1) * num_sky].reshape(ref_shape)
                     for j in range(num_sky_blocks)]
    cursor = num_sky_blocks * num_sky

    offset_coverages = []
    offset_valid_fracs = []
    for m, (ng, cm) in enumerate(zip(num_offset_groups_list, chunk_maps)):
        real_chunks = int(np.max(cm)) + 1
        nchk = real_chunks if num_chunks_list is None else int(num_chunks_list[m])
        block = ng * nchk
        offset_coverage = pixel_counts[cursor:cursor + block].reshape(ng, nchk)
        if nchk == real_chunks:
            chunk_sizes = np.bincount(cm[cm >= 0].ravel(), minlength=real_chunks)
            offset_valid_frac = (offset_coverage / np.maximum(chunk_sizes, 1))
        else:
            # Layout columns are not one-per-chunk (template alpha / poly-basis
            # coeffs); the caller overrides this map's coverage, so a shape-correct
            # placeholder avoids an ill-defined per-chunk division.
            offset_valid_frac = offset_coverage.astype(np.float64)
        offset_coverages.append(offset_coverage)
        offset_valid_fracs.append(offset_valid_frac)
        cursor += block
    return sky_coverages, offset_coverages, offset_valid_fracs


def parse_pixel_counts(pixel_counts, ref_shape, num_offset_groups_list, chunk_maps,
                       num_sky_blocks=1):
    """Back-compat fixed-tuple coverage slicing for <=2 sky blocks.

    Returns ``(skymap_coverage, offset_coverages, offset_valid_fracs)`` for 1
    block and ``(skymap_coverage, line_coverage, offset_coverages,
    offset_valid_fracs)`` for 2. Use :func:`parse_pixel_counts_sky` for N>2.
    """
    sky_coverages, offset_coverages, offset_valid_fracs = parse_pixel_counts_sky(
        pixel_counts, ref_shape, num_offset_groups_list, chunk_maps,
        num_sky_blocks=num_sky_blocks)
    if num_sky_blocks == 2:
        return sky_coverages[0], sky_coverages[1], offset_coverages, offset_valid_fracs
    return sky_coverages[0], offset_coverages, offset_valid_fracs


def parse_pixel_fisher_sky(pixel_fisher: np.ndarray, ref_shape: tuple[int, int],
                           num_sky_blocks: int = 1) -> list[np.ndarray]:
    """Generic Fisher slicing: returns a length-``num_sky_blocks`` list of
    ``ref_shape`` arrays (block 0 = continuum). See :func:`parse_pixel_counts_sky`.
    """
    num_sky = ref_shape[0] * ref_shape[1]
    return [pixel_fisher[j * num_sky:(j + 1) * num_sky].reshape(ref_shape)
            for j in range(num_sky_blocks)]


def parse_pixel_fisher(pixel_fisher, ref_shape, num_sky_blocks=1):
    """Back-compat fixed-tuple Fisher slicing for <=2 sky blocks.

    Returns ``(skymap_fisher, line_fisher)`` where ``line_fisher`` is None for
    1 block. Use :func:`parse_pixel_fisher_sky` for N>2.
    """
    sky_fishers = parse_pixel_fisher_sky(pixel_fisher, ref_shape, num_sky_blocks=num_sky_blocks)
    if num_sky_blocks == 2:
        return sky_fishers[0], sky_fishers[1]
    return sky_fishers[0], None


def _separability_from_moments(pixel_fisher, pixel_cross, num_sky, num_sky_blocks,
                               block=None):
    """Per-pixel separability ``I_P`` of one sky block from the streamed moments.

    ``I_P`` is the Schur complement of block ``block`` (default: the last)
    against all other sky blocks in the per-pixel J x J normal-matrix block.
    J == 2 gives the closed form ``Σw²G² − (Σw²G)²/Σw²``; J == 3 (last block)
    uses the 2x2 nuisance inverse (cont + slope) with a tiny ridge for pixels
    where the nuisance pair is collinear (e.g. single-BC pixels: t constant →
    cont/slope degenerate — the ridge slightly *underestimates* I there, i.e.
    errs toward more damping). Any other (J, block) goes through the general
    vectorized (J−1) x (J−1) nuisance solve with the same diagonal-ridge
    convention, evaluated in pixel slabs to bound memory.
    ``pixel_cross`` may be the bare pair-(0,1) array (J == 2) or the pair dict.
    """
    L = num_sky_blocks - 1
    if block is None:
        block = L
    F = [pixel_fisher[j * num_sky:(j + 1) * num_sky] for j in range(num_sky_blocks)]
    if not isinstance(pixel_cross, dict):
        pixel_cross = {(0, 1): np.asarray(pixel_cross)}
    if num_sky_blocks == 2 and block == 1:
        c = pixel_cross[(0, 1)]
        I_P = F[1] - np.where(F[0] > 0, c ** 2 / np.maximum(F[0], 1e-300), 0.0)
        return np.maximum(I_P, 0.0)
    if num_sky_blocks == 3 and block == 2:
        c01 = pixel_cross[(0, 1)]
        s0, s1 = pixel_cross[(0, 2)], pixel_cross[(1, 2)]
        # Diagonal ridge (not a det ridge): keeps the rank-deficient case
        # (e.g. single-BC pixels, where cont/slope are exactly collinear)
        # numerically stable — det >= r*(F0+F1) so the 0/0 cancellation never
        # happens, and the O(r) bias shrinks I (more damping = safe direction).
        r = 1e-8 * (F[0] + F[1] + 1.0)
        F0r, F1r = F[0] + r, F[1] + r
        det = np.maximum(F0r * F1r - c01 ** 2, 1e-300)
        # s^T M'^{-1} s with M' = [[F0+r, c01], [c01, F1+r]]
        quad = (F1r * s0 ** 2 - 2.0 * c01 * s0 * s1 + F0r * s1 ** 2) / det
        return np.maximum(F[2] - quad, 0.0)

    # --- General path: arbitrary J and target block. Per pixel, solve the
    # (J-1)x(J-1) nuisance normal block M (Gram of all non-target coefficient
    # vectors, PSD) with the same relative diagonal ridge as the J=3 closed
    # form, then I_P = F_k − s^T (M + rI)^{-1} s. Batched over pixel slabs.
    def _c(i, j):
        return F[i] if i == j else pixel_cross[(min(i, j), max(i, j))]

    nuis = [j for j in range(num_sky_blocks) if j != block]
    out = np.empty(num_sky, dtype=np.float64)
    slab = max(1, 8_000_000 // max(L * L, 1))          # ~500 MB of M per slab
    for p0 in range(0, num_sky, slab):
        sl = slice(p0, min(p0 + slab, num_sky))
        n = sl.stop - sl.start
        M = np.empty((n, L, L), dtype=np.float64)
        for a in range(L):
            for b in range(a, L):
                v = _c(nuis[a], nuis[b])[sl]
                M[:, a, b] = v
                if b != a:
                    M[:, b, a] = v
        s = np.stack([_c(nuis[a], block)[sl] for a in range(L)], axis=1)  # (n, L)
        r = 1e-8 * (M.trace(axis1=1, axis2=2) + 1.0)
        M[:, np.arange(L), np.arange(L)] += r[:, None]
        x = np.linalg.solve(M, s[..., None])[..., 0]
        out[sl] = F[block][sl] - np.einsum('nl,nl->n', s, x)
    return np.maximum(out, 0.0)


def parse_line_separability(pixel_cross: np.ndarray | dict, pixel_fisher: np.ndarray,
                            ref_shape: tuple[int, int], num_sky_blocks: int = 2,
                            block: int | None = None) -> np.ndarray:
    """Per-pixel line separability map ``I_P`` (see
    :func:`_separability_from_moments`) reshaped to ``ref_shape``.

    ``pixel_cross`` is the cross-moment return from ``setup_lsqr`` (bare array
    for 2-block models, pair dict for >=3); ``pixel_fisher`` the full
    accumulator; ``block`` selects which sky block's Schur complement to compute
    (default: the last). Unlike the line Fisher (Σw²G², a magnitude metric),
    I_P measures wavelength DIVERSITY: a pixel observed many times at a single
    BC has large Fisher but I_P = 0 — exactly the degenerate pixels that blow up
    under LSQR semi-convergence. Mask on I_P, not Fisher.
    """
    num_sky = ref_shape[0] * ref_shape[1]
    return _separability_from_moments(
        pixel_fisher, pixel_cross, num_sky, num_sky_blocks,
        block=block).reshape(ref_shape)


def apply_line_fisher_mask(sky_line: np.ndarray, line_fisher: np.ndarray,
                           threshold: float) -> tuple[np.ndarray, np.ndarray]:
    """Apply the line-Fisher mask at read time: zero sky_line where Fisher < threshold.

    Cals saved by Calibrator.save_calibration contain RAW sky_line (no
    destructive mask) plus the per-pixel line Fisher info. The
    line_fisher_threshold attribute (if present) is the recommended
    threshold, but the analyst is free to pick their own.

    Parameters
    ----------
    sky_line : np.ndarray
        Raw skymap_line as saved (ref grid).
    line_fisher : np.ndarray
        Raw skymap_line_fisher as saved (same shape).
    threshold : float
        Fisher threshold; pixels with line_fisher < threshold are zeroed.

    Returns
    -------
    sky_line_masked : np.ndarray
        Copy of sky_line with low-Fisher pixels set to 0.
    mask : np.ndarray of bool
        Boolean mask (True where masked-out).
    """
    mask = line_fisher < float(threshold)
    out = sky_line.copy()
    out[mask] = 0.0
    return out, mask


