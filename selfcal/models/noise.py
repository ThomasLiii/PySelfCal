"""Noise models for the observation weight: functions of data variables that return each
observation's noise variance, for ``sc.Model(weight=sc.Variance(...))``."""
import numpy as np

from .model import Variance

__all__ = ['shot_noise', 'shot_noise_variance']


def shot_noise_variance(value, flags, floor='median'):
    """The variance of shot noise, proportional to the signal and floored: ``max(|value|, floor)``
    per observation (in the data's units; the gain that makes it a variance is a common factor the
    weights do not need).

    ``floor``: ``"median"`` (default), the median of the frame's finite values whose ``flags``
    are 0, so no observation weighs more than the frame's typical one and brighter ones are
    down-weighted as ``1/value``; or a number, the same floor for every frame. A frame with no
    such values gets NaN (its observations carry no information and are dropped)."""
    v = np.asarray(value, dtype=np.float64)
    if isinstance(floor, str):
        if floor != 'median':
            raise ValueError(f"shot_noise_variance(floor={floor!r}): 'median' or a number")
        sel = np.isfinite(v) & (np.asarray(flags) == 0)
        if not sel.any():
            return np.full(v.shape, np.nan)
        m = float(np.median(v[sel]))
    else:
        m = float(floor)
    return np.maximum(np.abs(v), m)


def shot_noise(floor='median') -> Variance:
    """``sc.Variance(shot_noise_variance, of=("value", "flags"), floor=floor)``: the observation
    weight of a shot-noise-limited detector, ``1/sqrt(max(|value|, floor))`` on the solve's rows
    and ``1/max(|value|, floor)`` in the coadd, the floor the frame's median by default (see
    :func:`shot_noise_variance`)."""
    return Variance(shot_noise_variance, of=('value', 'flags'), floor=floor)
