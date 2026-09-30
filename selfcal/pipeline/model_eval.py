"""Evaluating a solved model at a frame's observations, outside the solve.

The row assembly evaluates every function of the model (sky coefficients,
offset bases, weights) at each frame's observations. The mosaic and the N-pass
need the same values for the offset terms whose offsets are coefficients of
known functions of data variables (an offset term with a ``coefficient`` or a
``basis``): such an offset is not constant per chunk, so it cannot be rendered
from per-chunk values; it is evaluated per observation here with the same
variable machinery (:mod:`selfcal.models.variables`) and the same bilinear
chunk weights as the solve.
"""
from __future__ import annotations

import os

import numpy as np

from ..geometry.map_helper import compute_chunk_contrib, make_linear_interp_matrix
from ..io.calfile import CalFile
from ..io.reproj import load_reproj_file
from ..models.variables import FrameObservations, ObservationVariables

__all__ = ['frame_observations', 'BasisOffsetSubtractor', 'ComposedHook']


def frame_observations(ctx, index, variables, pixels=None, oversample_factor=1, det_shape=None,
                       frame_row=None):
    """The data variables of a frame's subframe pixels outside the solve.

    ``ctx``: the frame hook's :class:`~selfcal.core.subframe.FrameContext`;
    ``index``: the frame's position in the solve (the built-in ``frame``);
    ``frame_row``: its position in ``variables``' frame values (default
    ``index``); ``variables``: a
    :class:`~selfcal.models.variables.VariableSet` (None: built-ins only).
    ``pixels`` defaults to every subframe pixel with a detector position.
    Returns ``(obs, interp)``: the lazy variables and the bilinear matrix
    (rows = the pixels) onto the detector grid of shape ``det_shape``."""
    sm = np.asarray(ctx.sub_mapping, dtype=np.float64)
    if pixels is None:
        pixels = np.nonzero(np.isfinite(sm[0]) & np.isfinite(sm[1]))
    rows, cols = pixels
    interp = None
    det = {}
    if det_shape is not None:
        coords_yx = np.stack([sm[1][rows, cols], sm[0][rows, cols]]) * oversample_factor
        interp = make_linear_interp_matrix(coords_yx, input_shape=tuple(det_shape))
        if variables is not None:
            for k, dmap in variables.detector.items():
                det[k] = np.asarray(interp @ np.asarray(dmap, dtype=np.float64).ravel())
    layers = {}
    if variables is not None and variables.layers:
        got = load_reproj_file(ctx.file, fields=[f'layers/{n}' for n in variables.layers])
        layers = {n: got[f'layers/{n}'] for n in variables.layers}
    fv = {}
    if variables is not None:
        row = index if frame_row is None else frame_row
        fv = {k: np.asarray(v)[row] for k, v in variables.frame.items()}
    fctx = None
    if variables is not None and variables.frame_functions:
        fctx = FrameObservations(file=ctx.file, index=index, pixels=pixels, ref_coords=ctx.ref_coords,
                                 sub_data=ctx.sub_data, sub_weight=ctx.sub_weight, sub_mapping=ctx.sub_mapping)
    obs = ObservationVariables(
        pixels, index=index, ref_coords=ctx.ref_coords, sub_mapping=ctx.sub_mapping, detector=det,
        frame_values=fv, sky=None if variables is None else variables.sky, layers=layers,
        derived=None if variables is None else variables.derived,
        frame_functions=None if variables is None else variables.frame_functions, context=fctx)
    return obs, interp


class BasisOffsetSubtractor:
    """A frame hook subtracting the offset terms that carry known functions of
    data variables, at every observation of each frame.

    ``cal_path``: the calibration whose ``offsets/map_<m>`` hold, per frame,
    the term's unknowns (``n_chunks * n`` coefficients, chunk-major).
    ``terms``: ``[(m, basis, det_chunk_map)]``; ``variables``: the data
    variables, their frame values aligned with ``frame_names`` (default: the
    cal's frame list). The offset at observation ``i`` of chunk weights
    ``w_c(i)`` is ``Σ_c w_c(i) Σ_k a[c, k] φ_k(v_i)``.
    """

    def __init__(self, cal_path, terms, variables=None, oversample_factor=1, frame_names=None):
        self.terms = [(int(m), b, np.asarray(cm)) for m, b, cm in terms]
        self.variables = variables
        self.oversample_factor = oversample_factor
        with CalFile(cal_path) as cal:
            offs = cal.offsets
            names = [os.path.basename(p) for p in cal.reproj_list]
            self.rows = {m: np.asarray(offs[m]) for m, _, _ in self.terms}
        self.index = {n: i for i, n in enumerate(names)}
        fn = names if frame_names is None else [os.path.basename(p) for p in frame_names]
        self.var_index = {n: i for i, n in enumerate(fn)}

    def __call__(self, ctx):
        sub_data = ctx.sub_data
        name = os.path.basename(ctx.file)
        i = self.index.get(name)
        if i is None:
            return sub_data
        shapes = {cm.shape for _, _, cm in self.terms}
        for shape in shapes:
            obs, interp = frame_observations(ctx, i, self.variables, oversample_factor=self.oversample_factor,
                                             det_shape=shape, frame_row=self.var_index.get(name, i))
            if obs.n == 0:
                continue
            total = np.zeros(obs.n, dtype=np.float64)
            for m, basis, cm in self.terms:
                if cm.shape != shape:
                    continue
                phi = np.asarray(basis.evaluate(obs), dtype=np.float64)          # (n_obs, n)
                nb = phi.shape[1]
                n_chunks = int(cm.max()) + 1
                a = self.rows[m][i].reshape(n_chunks, nb)
                contrib = compute_chunk_contrib(cm, interp)                       # (n_chunks, n_obs)
                off = np.asarray(contrib.T @ a)                                   # (n_obs, n)
                total += np.nan_to_num((off * phi).sum(axis=1))
            rows, cols = obs.pixels
            sub_data[rows, cols] -= total.astype(sub_data.dtype)
        return sub_data


class ComposedHook:
    """Several frame hooks applied in order (each returns the new ``sub_data``)."""

    def __init__(self, hooks):
        self.hooks = [h for h in hooks if h is not None]

    def __call__(self, ctx):
        for h in self.hooks:
            ctx.sub_data = h(ctx)
        return ctx.sub_data
