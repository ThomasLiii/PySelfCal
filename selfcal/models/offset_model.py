"""OffsetModel / OffsetBlock — bundle the per-map offset configuration.

``setup_lsqr`` historically takes seven parallel length-K lists, all indexed by
the same map index ``m``::

    chunk_maps, det_groups_list, det_templates, reg_weights,
    adj_infos, poly_constraints_list, mean_offsets_list

Keeping those in lockstep by hand is exactly where the K=2 readout and
chunked-NEP dual-poly configs get error-prone (e.g. matching ``[None, zeros]``
in one list against ``[None, target]`` in another). ``OffsetModel`` bundles each
map's configuration into one :class:`OffsetBlock` so a multi-map setup reads as
cohesive blocks.

This is a thin bundling/lowering layer: :meth:`OffsetModel.to_setup_kwargs`
expands back to the exact parallel-list kwargs ``setup_lsqr`` already consumes,
so driving ``setup_lsqr`` via an ``OffsetModel`` is numerically identical to
calling it with the equivalent flat kwargs (verified byte-equal by the gate).
The flat kwargs remain supported as a deprecated transitional API; ``OffsetModel``
is the forward-looking spelling and is slated to become the sole API once all
drivers migrate.

Per-block (lives on ``OffsetBlock``): chunk map, frame grouping, template,
adjacency + its weight, polynomial-chain constraints, mean-offset anchor.
Global solver settings (per-pixel sky/line damping ``damp_weight`` /
``damp_weight_line``, ``damp_offset``, ``spectral_fit``/sky model, masking,
weighting, workers, ...) are NOT per-block and stay as ``setup_lsqr`` kwargs.
``use_per_frame_scalar`` is a model-level flag (the scalar block is shared
across maps), so it lives on ``OffsetModel``.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class OffsetBlock:
    """Configuration for one chunk map (one offset block of the solve).

    Parameters
    ----------
    chunk_map : np.ndarray
        Detector-grid → chunk-id integer map for this block.
    det_groups : np.ndarray or None
        Per-frame group labels (length num_frames). ``None`` (default) solves a
        free offset per frame. ``np.zeros(num_frames)`` locks all frames to one
        shared offset vector (the K=2 detector-fixed readout-stripe pattern).
    template : np.ndarray or None
        Fixed spatial pattern; when set, the block solves only a per-frame
        amplitude. Requires ``det_groups`` to be set (matches setup_lsqr).
    reg_weight : float
        Adjacency-smoothness weight for this block (the flat ``reg_weights[m]``).
    adj_info : object or None
        Adjacency pairs ``(chunk_i, chunk_j)`` for this block (``adj_infos[m]``).
    poly_constraints : list or None
        List of polynomial-chain constraint groups, each a dict
        ``{'chains', 'stencil', 'weight'}`` (``poly_constraints_list[m]``).
        Multiple groups are allowed (e.g. column + subchannel chains) and are
        applied in list order.
    mean_offset : object or None
        Per-frame mean-anchor target for this block (``mean_offsets_list[m]``),
        typically ``np.zeros(num_frames)``.
    """

    chunk_map: np.ndarray
    det_groups: object = None
    template: object = None
    reg_weight: float = 0.0
    adj_info: object = None
    poly_constraints: object = None
    mean_offset: object = None


@dataclass(frozen=True)
class OffsetModel:
    """Ordered collection of :class:`OffsetBlock` + model-level solver flags."""

    blocks: tuple
    use_per_frame_scalar: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'blocks', tuple(self.blocks))
        if len(self.blocks) < 1:
            raise ValueError("OffsetModel needs at least one OffsetBlock")
        for i, b in enumerate(self.blocks):
            if not isinstance(b, OffsetBlock):
                raise TypeError(f"blocks[{i}] is {type(b).__name__}, expected OffsetBlock")

    @property
    def num_maps(self):
        return len(self.blocks)

    @property
    def chunk_maps(self):
        return [b.chunk_map for b in self.blocks]

    def to_setup_kwargs(self):
        """Expand to the parallel-list kwargs ``setup_lsqr`` consumes.

        Always emits explicit length-K lists. Passing ``[None]*K`` /
        ``[0.0]*K`` is equivalent to passing the bare ``None`` defaults
        (``setup_lsqr`` fills ``None`` to ``[None]*K`` / ``[0.0]*K``), so this
        is numerically identical to the flat-kwarg call.
        """
        return {
            'chunk_maps': [b.chunk_map for b in self.blocks],
            'det_groups_list': [b.det_groups for b in self.blocks],
            'det_templates': [b.template for b in self.blocks],
            'reg_weights': [b.reg_weight for b in self.blocks],
            'adj_infos': [b.adj_info for b in self.blocks],
            'poly_constraints_list': [b.poly_constraints for b in self.blocks],
            'mean_offsets_list': [b.mean_offset for b in self.blocks],
            'use_per_frame_scalar': self.use_per_frame_scalar,
        }
