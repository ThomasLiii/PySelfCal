"""OffsetModel / OffsetBlock: the per-map offset configuration of a solve.

``setup_lsqr`` takes parallel length-K lists, all indexed by the same map index ``m``::

    chunk_maps, det_groups_list, reg_weights, adj_infos, basis_list

``OffsetModel`` bundles each map's configuration into one :class:`OffsetBlock`, and
:meth:`OffsetModel.to_setup_kwargs` expands it back to exactly those lists, so a multi-map
model reads as cohesive blocks and the solve sees the same keywords (byte-equal to the flat
spelling).

Per block (:class:`OffsetBlock`): the chunk map, the frame grouping, the adjacency pairs and their
weight, the basis of known functions. Global solver settings (the sky model and its per-pixel
damping ``damp_weight`` / ``damp_weight_line``, masking, weighting, workers, ...) are NOT per-block
and stay ``setup_lsqr`` keywords; the offset damping is per map (``damp_offset_maps``). Hard
constraints on a block's columns (``exclusions``) go to the solver, not here.
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
        """Every data variable the basis may read: its coefficient's main variables and
        any its function reads by name."""
        return self.coefficient.variables

    @property
    def main_variables(self) -> tuple:
        """The data variables the function is applied to, in argument order (always read)."""
        return self.coefficient.main_variables

    def evaluate(self, obs, dtype=np.float32, rows=False, out=None) -> np.ndarray:
        """The ``n`` functions at the observations of ``obs``, as an ``(n_obs, n)`` array of ``dtype`` (float32: the
        solve's), or with ``rows`` an ``(n, n_obs)`` array, one function per contiguous row, written into ``out``
        when given.

        ``obs`` maps variable names to one value per observation (one frame's
        :class:`~selfcal.models.variables.ObservationVariables`); ``n_obs`` is the
        length of the first main variable. A function with an ``evaluate`` method
        (a profile of :mod:`selfcal.models.profiles`) is called as
        ``evaluate(x, obs)``, with ``x`` the first main variable's values; any other
        function with the main variables' values as positional arguments.
        :func:`as_basis_values` normalises the result and raises ``ValueError`` when
        its shape does not fit.
        """
        c = self.coefficient
        names = c.main_variables
        x0 = obs[names[0]]
        n_obs = int(np.shape(x0)[0])
        f = c.function
        v = f.evaluate(x0, obs) if hasattr(f, 'evaluate') else f(*[obs[k] for k in names])
        return as_basis_values(v, n_obs, self.n, dtype, rows=rows, out=out)


def as_basis_values(v, n_obs, n, dtype=np.float32, rows=False, out=None):
    """Normalise a basis function's result to ``(n_obs, n)`` of ``dtype`` (float32: the solve's); with ``rows``, to
    ``(n, n_obs)``, each function one contiguous row, cast once from the function's values straight into ``out``
    when given (the same values either way: a float64 result is cast to ``dtype`` element by element)."""
    if isinstance(v, (list, tuple)):
        if len(v) != n:
            raise ValueError(f"a basis of {n} functions returned {len(v)} columns")
        if rows:                    # a float32 or float64 result is cast once; any other goes through float64
            cols = [np.broadcast_to(a if a.dtype in (np.float32, np.float64) else a.astype(np.float64), (n_obs,))
                    for a in (np.asarray(c) for c in v)]
            target = np.empty((n, n_obs), dtype=dtype) if out is None else out
            np.stack(cols, axis=0, out=target)
            return target
        cols = [np.broadcast_to(np.asarray(c, dtype=np.float64), (n_obs,)) for c in v]
        return np.stack(cols, axis=1).astype(dtype)
    a = np.asarray(v)
    if rows:
        if a.ndim == 0 and n == 1:
            res = np.full((1, n_obs), a, dtype=dtype)
        elif a.ndim == 1 and n == 1 and a.shape[0] == n_obs:
            res = a.astype(dtype).reshape(1, n_obs)
        elif a.ndim == 2 and a.shape == (n_obs, n):
            res = np.ascontiguousarray(a.T, dtype=dtype)
        else:
            res = None
        if res is not None:
            if out is None:
                return res
            out[...] = res
            return out
    elif a.ndim == 0 and n == 1:
        return np.full((n_obs, 1), a, dtype=dtype)
    elif a.ndim == 1 and n == 1 and a.shape[0] == n_obs:
        return a.astype(dtype).reshape(n_obs, 1)
    elif a.ndim == 2 and a.shape == (n_obs, n):
        return a.astype(dtype)
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
        stripe — as in the ``sc.two_block`` model).
    reg_weight : float
        The weight of the adjacency rows (0: none).
    adj_info : tuple or None
        The adjacent chunk pairs ``(chunk_i, chunk_j)`` the rows tie together.
    basis : Basis or None
        ``n`` known functions of data variables multiplying the offset at each
        observation: the unknowns become one coefficient per chunk and function.
    """

    chunk_map: np.ndarray
    det_groups: object = None
    reg_weight: float = 0.0
    adj_info: object = None
    basis: object = None


@dataclass(frozen=True)
class OffsetModel:
    """Ordered collection of :class:`OffsetBlock`."""

    blocks: tuple

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
            ``det_groups_list``, ``reg_weights``, ``adj_infos``, ``basis_list``
            (each a length-K list indexed by map ``m``).
        """
        return {
            'chunk_maps': [b.chunk_map for b in self.blocks],
            'det_groups_list': [b.det_groups for b in self.blocks],
            'reg_weights': [b.reg_weight for b in self.blocks],
            'adj_infos': [b.adj_info for b in self.blocks],
            'basis_list': [b.basis for b in self.blocks],
        }
