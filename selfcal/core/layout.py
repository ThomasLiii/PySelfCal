"""SystemLayout — single source of truth for the LSQR unknowns-vector column layout.

The solution vector ``x`` produced by :func:`selfcal.core.system.setup_lsqr` is
a flat concatenation of blocks::

    x = [ sky block            (num_sky_blocks * num_sky cols)
        | offset block 0 | offset block 1 | ... | offset block K-1 ]

where ``num_sky = ref_h * ref_w``. Each offset block ``m`` spans
``num_offset_groups[m] * num_chunks[m]`` columns.

Two call sites need the exact same column arithmetic: ``setup_lsqr`` (to tell
the worker processes their per-map column bases) and
``pipeline_wrapper.Calibrator`` (to parse the solved ``x`` back into maps in
``save_calibration`` / ``parse_x``). Historically each computed it inline and
they had to be kept in lockstep by hand. ``SystemLayout.build`` computes it once
so they cannot drift.

Bit-identity contract: the layout computed here — including the
``np.unique(..., return_inverse=True)`` group mapping — fixes the column order
of every solved cal file. Do not "fix" edge cases here without deliberately
re-baselining the byte-equality regression goldens (reference ``cal_*.h5``
outputs of the byte-equality gates (``selfcal_scripts/gates/``) compared with
``selfcal_scripts/gates/diff_cal_h5.py``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = ["SystemLayout"]


@dataclass
class SystemLayout:
    """Column layout of the LSQR unknowns vector (see module docstring)."""

    ref_shape: tuple
    num_sky_blocks: int
    num_frames: int
    num_offset_groups_list: list
    num_chunks_list: list
    frame_to_group_list: list
    col_bases: list  # length K+1; col_bases[K] == total_cols

    # Derived (filled in __post_init__).
    num_sky: int = field(init=False)
    num_sky_eff: int = field(init=False)
    total_cols: int = field(init=False)

    def __post_init__(self):
        self.num_sky = int(self.ref_shape[0]) * int(self.ref_shape[1])
        self.num_sky_eff = self.num_sky_blocks * self.num_sky
        self.total_cols = self.col_bases[-1]

    @property
    def num_maps(self) -> int:
        """Number of offset blocks (chunk maps) in the layout."""
        return len(self.num_chunks_list)

    @classmethod
    def build(cls, ref_shape: tuple[int, int], chunk_maps: list[np.ndarray], *,
              num_sky_blocks: int, num_frames: int,
              det_groups_list: list[np.ndarray | None] | None = None,
              basis_list: list | None = None) -> SystemLayout:
        """Compute the column layout from the setup inputs.

        ``det_groups_list`` / ``basis_list`` may be ``None`` (default for every map) or length-K
        lists. The result is independent of how the caller spelled the defaults, so the parent
        process and the Calibrator get identical layouts from equivalent inputs.

        Parameters
        ----------
        ref_shape : tuple[int, int]
            ``(ref_h, ref_w)`` of the reference sky grid; ``num_sky = ref_h * ref_w``.
        chunk_maps : list[np.ndarray]
            Length-K list of per-map chunk-id images; the chunk count for map
            ``m`` is ``int(chunk_maps[m].max()) + 1``.
        num_sky_blocks : int
            Number of stacked sky sub-blocks (1 continuum-only, 2 with a line block).
        num_frames : int
            Number of exposure frames.
        det_groups_list : list[np.ndarray | None] | None, optional
            Per-map arrays mapping each frame to a detector-group id, or ``None``
            for one group per frame.
        basis_list : list | None, optional
            Per-map :class:`~selfcal.models.offset_model.Basis` (``n`` functions
            of data variables) or ``None``. A basis of ``n`` functions gives map
            ``m`` ``n`` columns per chunk (``num_chunks := n_chunks * n``); ``n == 1``
            leaves the layout unchanged.

        Returns
        -------
        SystemLayout
            The fully computed column layout for the unknowns vector.
        """
        K = len(chunk_maps)
        if det_groups_list is None:
            det_groups_list = [None] * K
        if basis_list is None:
            basis_list = [None] * K

        frame_to_group_list = []
        num_offset_groups_list = []
        num_chunks_list = []
        for m in range(K):
            cm = chunk_maps[m]
            num_chunks_m = int(cm.max()) + 1
            n_basis_m = 1 if basis_list[m] is None else int(basis_list[m].n)
            if n_basis_m > 1:
                num_chunks_m *= n_basis_m
            if det_groups_list[m] is not None:
                det_groups_arr = np.asarray(det_groups_list[m])
                unique_groups, ftg = np.unique(det_groups_arr, return_inverse=True)
                num_offset_groups_m = len(unique_groups)
            else:
                ftg = np.arange(num_frames)
                num_offset_groups_m = num_frames
            frame_to_group_list.append(ftg)
            num_offset_groups_list.append(num_offset_groups_m)
            num_chunks_list.append(num_chunks_m)

        num_sky = int(ref_shape[0]) * int(ref_shape[1])
        col_bases = [num_sky_blocks * num_sky]
        for m in range(K):
            col_bases.append(col_bases[-1] + num_chunks_list[m] * num_offset_groups_list[m])

        return cls(
            ref_shape=tuple(ref_shape),
            num_sky_blocks=num_sky_blocks,
            num_frames=num_frames,
            num_offset_groups_list=num_offset_groups_list,
            num_chunks_list=num_chunks_list,
            frame_to_group_list=frame_to_group_list,
            col_bases=col_bases,
        )

    # --- column-slice helpers (used by parsing; safe convenience accessors) ---
    def sky_block_slice(self, j: int = 0) -> slice:
        """Columns of sky sub-block ``j`` (j=0 continuum, j=1 line).

        Parameters
        ----------
        j : int, optional
            Sky sub-block index (0 continuum, 1 line). Defaults to 0.

        Returns
        -------
        slice
            Column slice spanning sky sub-block ``j``.
        """
        return slice(j * self.num_sky, (j + 1) * self.num_sky)

    def offset_slice(self, m: int) -> slice:
        """Columns of offset block (chunk map) ``m``.

        Parameters
        ----------
        m : int
            Offset-block (chunk-map) index in ``[0, num_maps)``.

        Returns
        -------
        slice
            Column slice spanning offset block ``m``.
        """
        return slice(self.col_bases[m], self.col_bases[m + 1])
