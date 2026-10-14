"""The slope offset basis of the transfer-function campaign (copy of its deflation/xbasis.py)."""
import numpy as np

X_CENTRE, X_HALF = 1019.5, 1020.0


def xtilde(det_x):
    """(x - centre) / half-width: -1 at the left edge, +1 at the right."""
    return (np.asarray(det_x, dtype=np.float64) - X_CENTRE) / X_HALF


def ytilde(det_y):
    """(y - centre) / half-width along the other detector axis: -1 at the bottom edge, +1 at the top."""
    return (np.asarray(det_y, dtype=np.float64) - X_CENTRE) / X_HALF


def xquad(det_x):
    """xtilde^2 - 1/3: the quadratic across the detector, orthogonal to the constant over a full row."""
    return xtilde(det_x) ** 2 - 1.0 / 3.0


def xgrad(det_x, gpat):
    """A rank-1 across-detector gradient: the fixed subchannel pattern (a detector map) times xtilde(det_x)."""
    return np.asarray(gpat, dtype=np.float64) * xtilde(det_x)
