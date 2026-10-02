"""Per-frame hooks of the Euclid recipe, selectable from a run config::

    [hooks]
    pre_cal     = { name = "star_position_mask", positions = ".../star_positions.npz",
                    radius_px = 160, radius2_px = 80, n_tier1 = 60 }
    post_cal    = { name = "residual_mask", mask_dir = ".../resid_masks", mask_pixels = true,
                    frame_weight = true }
    post_mosaic = { name = "residual_mask", mask_dir = ".../resid_masks" }

Each entry names a factory below; the remaining keys are its parameters. The
factory returns a callable ``hook(ctx: FrameContext) -> sub_data``: a :class:`StarMask` or a
:class:`ResidualMask`, objects the worker processes can receive. In Python, build them directly::

    sc.Fit(raw_frame_hook=euclid.StarMask(".../star_positions.npz", radius_px=160))
"""
from __future__ import annotations

import os

import numpy as np

__all__ = ['StarMask', 'ResidualMask', 'star_position_mask', 'residual_mask', 'HOOKS']


class StarMask:
    """FIT-ONLY preprocess hook: zero ``sub_weight`` in a disk around every bright
    star REF position the frame's footprint covers. ``positions``: an npz with a
    ``positions`` array of (y, x) reference-grid pixels sorted by brightness
    (one-off detection sweep), or the array itself. The first ``n_tier1``
    stars get ``radius_px``, the rest ``radius2_px`` (default: the same).

    A plain object (not a closure), so it can be sent to the worker processes."""

    def __init__(self, positions, radius_px, radius2_px=None, n_tier1=None):
        self.positions = positions if isinstance(positions, str) else np.asarray(positions)
        pos = np.load(positions)['positions'] if isinstance(positions, str) else np.asarray(positions)
        radius = np.full(len(pos), int(radius2_px if radius2_px is not None else radius_px), dtype=np.int64)
        radius[:int(n_tier1 if n_tier1 is not None else len(pos))] = int(radius_px)
        self._pos, self._radius = pos, radius
        self.radius_px, self.radius2_px, self.n_tier1 = radius_px, radius2_px, n_tier1

    def __repr__(self):
        return (f"StarMask({self.positions if isinstance(self.positions, str) else '<array>'!r}, "
                f"radius_px={self.radius_px}, radius2_px={self.radius2_px}, n_tier1={self.n_tier1})")

    def __call__(self, ctx):
        y0, _, x0, _ = ctx.ref_coords
        w = ctx.sub_weight
        h, wd = w.shape
        for (py, px), R in zip(self._pos, self._radius):
            cy, cx = py - y0, px - x0
            if -R < cy < h + R and -R < cx < wd + R:
                ys = np.arange(max(int(cy - R), 0), min(int(cy + R) + 1, h))
                xs = np.arange(max(int(cx - R), 0), min(int(cx + R) + 1, wd))
                if len(ys) and len(xs):
                    dy = ys[:, None] - cy
                    dx = xs[None, :] - cx
                    w[np.ix_(ys, xs)] *= (dy * dy + dx * dx > R * R)
        return ctx.sub_data


class ResidualMask:
    """Postprocess hook (cal and/or mosaic): NaN out this frame's residual-flagged
    pixels (``<frame>.npz`` with packed ``bits`` + ``shape``, from a pass-1
    residual analysis) so they drop from the fit / coadd, and/or multiply the
    frame's whole ``sub_weight`` by its inverse-variance weight from
    ``frame_weights.npz`` (``names`` / ``weights``). Frames without files pass
    through unchanged.

    A plain object (not a closure), so it can be sent to the worker processes; each process
    reads the frame-weight table once."""

    def __init__(self, mask_dir, mask_pixels=True, frame_weight=False):
        self.mask_dir, self.mask_pixels, self.frame_weight = mask_dir, mask_pixels, frame_weight
        self._table = None

    def __repr__(self):
        return (f"ResidualMask({self.mask_dir!r}, mask_pixels={self.mask_pixels}, "
                f"frame_weight={self.frame_weight})")

    def __getstate__(self):
        state = dict(self.__dict__)
        state['_table'] = None
        return state

    def __call__(self, ctx):
        sub_data = ctx.sub_data
        name = os.path.basename(ctx.file).replace('.h5', '')
        if self.frame_weight:
            if self._table is None:
                fwp = os.path.join(self.mask_dir, "frame_weights.npz")
                if os.path.exists(fwp):
                    with np.load(fwp) as z:
                        self._table = dict(zip([str(n) for n in z['names']], z['weights']))
                else:
                    self._table = {}
            w = float(self._table.get(name, 1.0))
            if w < 1.0:
                ctx.sub_weight *= w
        if self.mask_pixels:
            p = os.path.join(self.mask_dir, f"{name}.npz")
            if os.path.exists(p):
                with np.load(p) as z:
                    bad = np.unpackbits(z['bits'])[:sub_data.size].reshape(tuple(z['shape'])).astype(bool)
                if bad.any():
                    sub_data = sub_data.copy()
                    sub_data[bad] = np.nan
        return sub_data


def star_position_mask(positions, radius_px, radius2_px=None, n_tier1=None):
    """The :class:`StarMask` hook (the name a run config's ``[hooks]`` table uses)."""
    return StarMask(positions, radius_px, radius2_px=radius2_px, n_tier1=n_tier1)


def residual_mask(mask_dir, mask_pixels=True, frame_weight=False):
    """The :class:`ResidualMask` hook (the name a run config's ``[hooks]`` table uses)."""
    return ResidualMask(mask_dir, mask_pixels=mask_pixels, frame_weight=frame_weight)


HOOKS = {'star_position_mask': star_position_mask, 'residual_mask': residual_mask}
