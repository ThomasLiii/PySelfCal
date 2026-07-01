"""Hard polynomial-basis offset — the mean-zero Chebyshev shape basis.

Replaces the soft ``subch_poly`` penalty (a weight ``λ`` on finite-difference
rows that pull a free per-chunk offset toward a degree-D polynomial) with the
offset *being* a degree-D polynomial in subchannel by construction: the solve
carries the polynomial coefficients directly, so there is **no weight knob** and
the null space that let the offset grow a spurious PAH bump (see
``project_spectral_fit_lvf_branch``) is gone.

Design (chosen 2026-07-01):
  * offset[frame, subch, col] = scalar[frame] + Σ_{d=1..D} a[frame, col, d] · B_d(subch)
  * B_d = Chebyshev T_d on x = 2·(subch−lo)/(hi−lo) − 1 ∈ [−1,1], **mean-subtracted
    over the window grid** so each B_d is orthogonal to the constant → the per-frame
    scalar owns the DC and the polynomial is shape-only (d starts at 1, no constant).
  * independent polynomial per column (a[frame, col, d] free in col).
  * Chebyshev (not raw x^d) for conditioning at D ≥ 3.

This module is the single source of truth for B_d, used both by the row assembly
(coefficient of each observation) and by reconstruction (coefficients → per-chunk
offset map for save/apply).
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as _C


def cheb_shape_basis(subch, degree, lo, hi):
    """Mean-zero Chebyshev shape basis, degrees d=1..``degree``.

    Parameters
    ----------
    subch : array-like
        Subchannel indices to evaluate at (any shape; flattened result rows).
    degree : int
        Polynomial degree D (>= 1). Returns D basis columns (d = 1..D; the
        constant d=0 is intentionally omitted — the per-frame scalar owns the DC).
    lo, hi : int
        Inclusive subchannel window the polynomial is defined over (maps to
        x ∈ [-1, 1]); the mean is taken over the integer grid ``lo..hi``.

    Returns
    -------
    B : (N, D) float64
        ``B[i, d-1] = T_d(x(subch_i)) - mean_grid(T_d)``. Orthogonal to the
        constant by construction (each column sums ~0 over the window grid).
    """
    if degree < 1:
        raise ValueError(f"degree must be >= 1 (d=0 is the scalar), got {degree}")
    if hi <= lo:
        raise ValueError(f"need hi > lo, got lo={lo} hi={hi}")
    subch = np.asarray(subch, dtype=np.float64).ravel()
    span = float(hi - lo)
    x = np.clip(2.0 * (subch - lo) / span - 1.0, -1.0, 1.0)
    grid = np.arange(int(lo), int(hi) + 1, dtype=np.float64)
    xg = 2.0 * (grid - lo) / span - 1.0
    B = np.empty((subch.size, degree), dtype=np.float64)
    for d in range(1, degree + 1):
        coef = np.zeros(d + 1); coef[d] = 1.0            # Chebyshev T_d
        B[:, d - 1] = _C.chebval(x, coef) - _C.chebval(xg, coef).mean()
    return B


def reconstruct_offset_map(coeffs, num_subch, num_col, degree, lo, hi, scalar=None):
    """Evaluate coefficients → per-chunk offset for one frame (for save/apply).

    ``coeffs`` is ``(num_col, degree)`` (a[col, d=1..D]); returns a length
    ``num_subch*num_col`` vector indexed by chunk = subch*num_col + col, matching
    the chunk-id convention. Subchannels outside ``[lo, hi]`` are clamped by
    ``cheb_shape_basis`` (x clipped), but in practice only in-window chunks carry
    data. Adds the per-frame ``scalar`` (DC) to every chunk if given.
    """
    subs = np.arange(num_subch)
    B = cheb_shape_basis(subs, degree, lo, hi)           # (num_subch, degree)
    per_subch_col = B @ np.asarray(coeffs).T              # (num_subch, num_col)
    if scalar is not None:
        per_subch_col = per_subch_col + float(scalar)
    # chunk = subch*num_col + col
    return per_subch_col.reshape(num_subch * num_col)


if __name__ == "__main__":  # quick self-test
    lo, hi, D = 200, 259, 2
    B = cheb_shape_basis(np.arange(lo, hi + 1), D, lo, hi)
    print(f"basis shape {B.shape}, column means (should be ~0): {B.mean(0)}")
    print(f"B(lo)={B[0]}, B(mid)={B[len(B)//2]}, B(hi)={B[-1]}")
    # a pure-quadratic offset must be represented exactly (mean-removed)
    subs = np.arange(lo, hi + 1, dtype=float)
    x = 2 * (subs - lo) / (hi - lo) - 1
    truth = 3.0 * (x**2)                                  # quadratic shape
    truth = truth - truth.mean()                         # its mean-zero part
    fit, *_ = np.linalg.lstsq(B, truth, rcond=None)
    resid = np.max(np.abs(B @ fit - truth))
    print(f"exact-quadratic reconstruction max resid = {resid:.2e} (should be ~0)")
