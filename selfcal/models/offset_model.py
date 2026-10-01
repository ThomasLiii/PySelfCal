"""OffsetModel / OffsetBlock — bundle the per-map offset configuration.

``setup_lsqr`` historically takes seven parallel length-K lists, all indexed by
the same map index ``m``::

    chunk_maps, det_groups_list, det_templates, reg_weights,
    adj_infos, poly_constraints_list, mean_offsets_list

Keeping those in lockstep by hand is error-prone in any multi-map
configuration — e.g. a two-map setup must pair ``det_groups_list=[None, zeros]``
in one list with ``mean_offsets_list=[None, target]`` in another (see the
``k2_readout`` runner mode for a real two-map instance). ``OffsetModel`` bundles
each map's configuration into one :class:`OffsetBlock` so a multi-map setup
reads as cohesive blocks.

This is a thin bundling/lowering layer: :meth:`OffsetModel.to_setup_kwargs`
expands back to the exact parallel-list kwargs ``setup_lsqr`` already consumes,
so driving ``setup_lsqr`` via an ``OffsetModel`` is numerically identical to
calling it with the equivalent flat kwargs (verified byte-equal: rerunning a
reference config through both spellings produces identical ``cal_*.h5`` output;
byte-equality gates in ``selfcal_scripts/gates/`` +
``selfcal_scripts/drivers/diff_cal_h5.py``). The flat parallel-list kwargs
remain supported but are deprecated; new code should construct an
``OffsetModel``.

Per-block (lives on ``OffsetBlock``): chunk map, frame grouping, template,
adjacency + its weight, polynomial-chain constraints, mean-offset anchor.
Global solver settings (per-pixel sky/line damping ``damp_weight`` /
``damp_weight_line``, ``damp_offset``, ``spectral_fit``/sky model, masking,
weighting, workers, ...) are NOT per-block and stay as ``setup_lsqr`` kwargs.
``use_per_frame_scalar`` is a model-level flag (the scalar block is shared
across maps), so it lives on ``OffsetModel``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ['Basis', 'OffsetBlock', 'OffsetModel']


@dataclass(frozen=True)
class Basis:
    """``n`` known functions of data variables, evaluated at every observation.

    ``coefficient`` (a :class:`~selfcal.models.sky_model.Coefficient`) names the
    variables and the function. The function returns, for the ``n_obs``
    observations of a frame, an ``(n_obs, n)`` array, a sequence of ``n``
    per-observation arrays (or scalars), or — for ``n == 1`` — one array.
    :meth:`evaluate` always returns ``(n_obs, n)`` float32.
    """
    coefficient: object
    n: int = 1

    def __post_init__(self):
        if int(self.n) < 1:
            raise ValueError(f"a basis needs n >= 1 functions (got {self.n})")
        object.__setattr__(self, 'n', int(self.n))

    @property
    def variables(self) -> tuple:
        return self.coefficient.variables

    @property
    def main_variables(self) -> tuple:
        return self.coefficient.main_variables

    def evaluate(self, obs) -> np.ndarray:
        c = self.coefficient
        names = c.main_variables
        x0 = obs[names[0]]
        n_obs = int(np.shape(x0)[0])
        f = c.function
        v = f.evaluate(x0, obs) if hasattr(f, 'evaluate') else f(*[obs[k] for k in names])
        return as_basis_values(v, n_obs, self.n)


def as_basis_values(v, n_obs, n):
    """Normalise a basis function's result to ``(n_obs, n)`` float32."""
    if isinstance(v, (list, tuple)):
        if len(v) != n:
            raise ValueError(f"a basis of {n} functions returned {len(v)} columns")
        cols = [np.broadcast_to(np.asarray(c, dtype=np.float64), (n_obs,)) for c in v]
        return np.stack(cols, axis=1).astype(np.float32)
    a = np.asarray(v)
    if a.ndim == 0 and n == 1:
        return np.full((n_obs, 1), a, dtype=np.float32)
    if a.ndim == 1 and n == 1 and a.shape[0] == n_obs:
        return a.astype(np.float32).reshape(n_obs, 1)
    if a.ndim == 2 and a.shape == (n_obs, n):
        return a.astype(np.float32)
    raise ValueError(f"a basis of {n} function(s) over {n_obs} observations returned shape {a.shape}; "
                     f"expected ({n_obs}, {n}), {n} arrays" + (" or one array" if n == 1 else ""))


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
        shared offset vector (a detector-fixed pattern — e.g. a readout
        stripe — as in the ``k2_readout`` runner mode).
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
    poly_basis : dict or None
        Hard polynomial-basis offset (replaces the soft ``poly_constraints``
        subchannel penalty). When set, this block does NOT solve a free offset
        per chunk; instead the per-frame offset IS a degree-D Chebyshev
        polynomial in an abstract per-chunk coordinate, independent per group
        (for SPHEREx: coordinate = subchannel, group = column), solved for its
        coefficients ``a[frame, group, d]`` (d=1..D; the per-frame scalar owns
        the DC). Dict keys (see :mod:`selfcal.models.offset_basis`): ``degree``
        (int D), ``num_groups`` (number of independent polynomials),
        ``coord_lo``/``coord_hi`` (coordinate window), ``chunk_coord``/
        ``chunk_group`` (per-chunk coordinate / group-index arrays). When set,
        ``adj_info``/``reg_weight``/``poly_constraints`` for this block are
        ignored (the polynomial is exact, no weight knob).
    basis : Basis or None
        ``n`` known functions of data variables multiplying the block
        (:class:`Basis`). The block's unknowns become one coefficient per
        (frame group, chunk, function) — column ``g*(n_chunks*n) + c*n + k`` —
        and observation ``i`` of chunk ``c`` contributes
        ``chunk_weight(i, c) * phi_k(v_i)``. ``n == 1`` is a *coefficient*: the
        chunk offset times a known function (a detector pattern times the
        frame's temperature, a per-frame gain times a previous sky, ...); the
        column layout is then the plain block's. ``adj_info`` and
        ``poly_constraints`` are given in the expanded column space (the
        lowering expands chunk pairs over the functions). ``None``: the
        classic block (``phi = 1``).
    """

    chunk_map: np.ndarray
    det_groups: object = None
    template: object = None
    reg_weight: float = 0.0
    adj_info: object = None
    poly_constraints: object = None
    mean_offset: object = None
    poly_basis: object = None
    basis: object = None

    def __post_init__(self):
        b = self.basis
        if b is None:
            return
        if self.poly_basis is not None and b.n != 1:
            raise ValueError("a hard poly-basis block takes a coefficient (a basis of one function) only")
        if self.template is not None and b.n != 1:
            raise ValueError("a template block takes a coefficient (a basis of one function) only")


@dataclass(frozen=True)
class OffsetModel:
    """Ordered collection of :class:`OffsetBlock` + model-level solver flags."""

    blocks: tuple
    use_per_frame_scalar: bool = False

    def __post_init__(self):
        object.__setattr__(self, 'blocks', tuple(self.blocks))
        for i, b in enumerate(self.blocks):
            if not isinstance(b, OffsetBlock):
                raise TypeError(f"blocks[{i}] is {type(b).__name__}, expected OffsetBlock")

    @classmethod
    def sky_only(cls) -> "OffsetModel":
        """No offset blocks at all: a sky-only solve (the offsets already
        subtracted from the data by a hook, e.g. the N-pass SKY pass)."""
        return cls(())

    @property
    def num_maps(self) -> int:
        """Number of offset blocks (maps) in the model; 0 for a sky-only model."""
        return len(self.blocks)

    @property
    def chunk_maps(self) -> list[np.ndarray]:
        """The per-block chunk maps, in block order."""
        return [b.chunk_map for b in self.blocks]

    def to_setup_kwargs(self) -> dict:
        """Expand to the parallel-list kwargs ``setup_lsqr`` consumes.

        Always emits explicit length-K lists. Passing ``[None]*K`` /
        ``[0.0]*K`` is equivalent to passing the bare ``None`` defaults
        (``setup_lsqr`` fills ``None`` to ``[None]*K`` / ``[0.0]*K``), so this
        is numerically identical to the flat-kwarg call.

        Returns
        -------
        dict
            The parallel-list ``setup_lsqr`` kwargs: ``chunk_maps``,
            ``det_groups_list``, ``det_templates``, ``reg_weights``,
            ``adj_infos``, ``poly_constraints_list``, ``mean_offsets_list``,
            ``poly_basis_list``, ``basis_list`` (each a length-K list indexed
            by map ``m``), plus the model-level ``use_per_frame_scalar`` flag.
        """
        return {
            'chunk_maps': [b.chunk_map for b in self.blocks],
            'det_groups_list': [b.det_groups for b in self.blocks],
            'det_templates': [b.template for b in self.blocks],
            'reg_weights': [b.reg_weight for b in self.blocks],
            'adj_infos': [b.adj_info for b in self.blocks],
            'poly_constraints_list': [b.poly_constraints for b in self.blocks],
            'mean_offsets_list': [b.mean_offset for b in self.blocks],
            'poly_basis_list': [b.poly_basis for b in self.blocks],
            'basis_list': [b.basis for b in self.blocks],
            'use_per_frame_scalar': self.use_per_frame_scalar,
        }
