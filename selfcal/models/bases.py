"""Basis functions of the detector coordinates for offset terms (``sc.Offsets(times=sc.Function(...))``)."""
import numpy as np

__all__ = ['tilt']


def tilt(det_x, centre, half):
    """A linear basis along a detector axis, ``(det_x - centre) / half``: -1 at the edge ``centre - half``,
    +1 at ``centre + half``; for a detector of ``n`` pixels, ``centre = (n - 1) / 2`` and ``half = n / 2``.
    An offset term ``times`` it, one coefficient per frame, is a free gradient per frame -- the
    model the post-solve gauge (:class:`~selfcal.run.recipe.Gauge`) is defined for."""
    return (np.asarray(det_x, dtype=np.float64) - centre) / half
