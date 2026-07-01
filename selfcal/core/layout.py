"""SystemLayout — single source of truth for the LSQR unknowns-vector column layout.

The solution vector ``x`` produced by :func:`selfcal.lsqr.setup_lsqr` is a flat
concatenation of blocks::

    x = [ sky block            (num_sky_blocks * num_sky cols)
        | offset block 0 | offset block 1 | ... | offset block K-1
        | per-frame scalars    (num_scalar_cols cols) ]

where ``num_sky = ref_h * ref_w``. Each offset block ``m`` spans
``num_offset_groups[m] * num_chunks[m]`` columns, except in *template* mode
where it collapses to ``num_frames`` per-frame amplitude columns.

Two call sites need the exact same column arithmetic: ``setup_lsqr`` (to tell
the worker processes their per-map column bases) and
``pipeline_wrapper.Calibrator`` (to parse the solved ``x`` back into maps in
``save_calibration`` / ``parse_x``). Historically each computed it inline and
they had to be kept in lockstep by hand. ``SystemLayout.build`` computes it once
so they cannot drift.

Bit-identity contract: ``build`` reproduces the historical inline computation in
``lsqr.setup_lsqr`` verbatim — including template-mode collapse to one chunk and
``num_frames`` alpha columns, and the ``np.unique(..., return_inverse=True)``
group mapping. Do not "fix" edge cases here without deliberately re-baselining
the regression goldens.
"""
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SystemLayout:
    """Column layout of the LSQR unknowns vector (see module docstring)."""

    ref_shape: tuple
    num_sky_blocks: int
    num_frames: int
    num_offset_groups_list: list
    num_chunks_list: list
    det_template_arr_list: list
    frame_to_group_list: list
    num_scalar_cols: int
    col_bases: list  # length K+1; col_bases[K] == scalar_col_start
    poly_basis_list: list = None  # per-map hard-poly spec or None

    # Derived (filled in __post_init__).
    num_sky: int = field(init=False)
    num_sky_eff: int = field(init=False)
    scalar_col_start: int = field(init=False)
    total_cols: int = field(init=False)

    def __post_init__(self):
        self.num_sky = int(self.ref_shape[0]) * int(self.ref_shape[1])
        self.num_sky_eff = self.num_sky_blocks * self.num_sky
        self.scalar_col_start = self.col_bases[-1]
        self.total_cols = self.scalar_col_start + self.num_scalar_cols

    @property
    def num_maps(self):
        return len(self.num_chunks_list)

    @classmethod
    def build(cls, ref_shape, chunk_maps, *, num_sky_blocks, num_frames,
              det_groups_list=None, det_templates=None, use_per_frame_scalar=False,
              poly_basis_list=None):
        """Compute the column layout from the setup inputs.

        ``det_groups_list`` / ``det_templates`` / ``poly_basis_list`` may be
        ``None`` (default for every map) or length-K lists. The result is
        independent of how the caller spelled the defaults, so the parent
        process and the Calibrator get identical layouts from equivalent inputs.

        A map with ``poly_basis_list[m]`` set is a **hard polynomial-basis**
        offset: its columns are the coefficients ``a[frame, col, d=1..D]``, so
        the block reuses the chunk machinery with ``num_chunks := num_col * D``
        and one group per frame (no det_groups / template).
        """
        K = len(chunk_maps)
        if det_groups_list is None:
            det_groups_list = [None] * K
        if det_templates is None:
            det_templates = [None] * K
        if poly_basis_list is None:
            poly_basis_list = [None] * K

        any_det_groups = any(g is not None for g in det_groups_list)

        frame_to_group_list = []
        num_offset_groups_list = []
        num_chunks_list = []
        det_template_arr_list = []
        for m in range(K):
            cm = chunk_maps[m]
            pb = poly_basis_list[m]
            if pb is not None:
                # Hard poly-basis: columns are a[frame, col, d] coeffs. Reuse the
                # chunk machinery: num_chunks := num_col*degree, one group/frame.
                assert det_groups_list[m] is None and det_templates[m] is None, \
                    f"poly_basis[{m}] is incompatible with det_groups/det_templates"
                from ..models.offset_basis import n_coef as _n_coef
                ftg = np.arange(num_frames)
                num_offset_groups_m = num_frames
                num_chunks_m = int(pb['num_col']) * _n_coef(pb)
                tmpl = None
                frame_to_group_list.append(ftg)
                num_offset_groups_list.append(num_offset_groups_m)
                num_chunks_list.append(num_chunks_m)
                det_template_arr_list.append(tmpl)
                continue
            num_chunks_m = int(cm.max()) + 1
            if det_groups_list[m] is not None:
                det_groups_arr = np.asarray(det_groups_list[m])
                unique_groups, ftg = np.unique(det_groups_arr, return_inverse=True)
                num_offset_groups_m = len(unique_groups)
            else:
                ftg = np.arange(num_frames)
                num_offset_groups_m = num_frames

            if det_templates[m] is not None:
                assert det_groups_list[m] is not None, \
                    f"det_templates[{m}] requires det_groups_list[{m}]"
                tmpl = np.asarray(det_templates[m], dtype=np.float32)
                # Template mode collapses (groups, chunks) into one per-frame alpha.
                num_offset_groups_m = num_frames
                num_chunks_m = 1
            else:
                tmpl = None

            frame_to_group_list.append(ftg)
            num_offset_groups_list.append(num_offset_groups_m)
            num_chunks_list.append(num_chunks_m)
            det_template_arr_list.append(tmpl)

        num_scalar_cols = num_frames if (any_det_groups or use_per_frame_scalar) else 0

        num_sky = int(ref_shape[0]) * int(ref_shape[1])
        col_bases = [num_sky_blocks * num_sky]
        for m in range(K):
            if det_template_arr_list[m] is not None:
                block = num_frames  # one alpha column per frame
            else:
                block = num_chunks_list[m] * num_offset_groups_list[m]
            col_bases.append(col_bases[-1] + block)

        return cls(
            ref_shape=tuple(ref_shape),
            num_sky_blocks=num_sky_blocks,
            num_frames=num_frames,
            num_offset_groups_list=num_offset_groups_list,
            num_chunks_list=num_chunks_list,
            det_template_arr_list=det_template_arr_list,
            frame_to_group_list=frame_to_group_list,
            num_scalar_cols=num_scalar_cols,
            col_bases=col_bases,
            poly_basis_list=list(poly_basis_list),
        )

    # --- column-slice helpers (used by parsing; safe convenience accessors) ---
    def sky_block_slice(self, j=0):
        """Columns of sky sub-block ``j`` (j=0 continuum, j=1 line)."""
        return slice(j * self.num_sky, (j + 1) * self.num_sky)

    def offset_slice(self, m):
        """Columns of offset block (chunk map) ``m``."""
        return slice(self.col_bases[m], self.col_bases[m + 1])

    def scalar_slice(self):
        """Columns of the per-frame scalar block (empty slice if none)."""
        return slice(self.scalar_col_start, self.total_cols)
