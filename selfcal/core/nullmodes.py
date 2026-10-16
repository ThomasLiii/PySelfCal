"""Null modes: the sky patterns at a coverage boundary that the offsets absorb, found from the system and set by rule.

A selfcal system fixes the sky only where the frames' offsets cannot imitate it. Where frames CROSS a region in
every direction, a smooth sky pattern of any size leaves its curvature in every frame and the data see it. Where
frames only END, at the field's outer edge or along the cut of a frame selection, a smooth pattern confined to that
region is nearly linear along each of the short frame ends that reach it, so a model with a constant and a gradient
per frame (or per chunk of a frame) absorbs it almost exactly: the data constrain it weakly or not at all, an
iterative solve leaves it wherever its path put it, and two solves of the same data can differ there by far more
than the noise while their fits agree to rounding.

This module finds those directions the way :mod:`selfcal.core.deflation` finds its patterns' offset response, but
with smooth LOCAL patterns, and sets the sky's amplitude along the undetermined ones by a stated rule.

**Patterns.** Uniform cubic B-splines with a knot spacing of ``cell`` pixels over the covered bounding box of one
sky term (``n + 3`` functions per axis for ``n`` cells; at every sky pixel four by four of them are non-zero,
:func:`bspline_weights`). Each pattern ``F`` is paired with its exact offset response ``g(F) = argmin |B_sky F +
B_off g|`` on the column-scaled system ``B`` the solver sees (the offsets' normal matrix, solved on each of its
connected blocks by the pseudo-inverse, as the deflation does); ``z = (F, g(F))``.

**Spectrum.** With ``E = (B z)^T (B z)`` and ``E_sky = (B_sky F)^T (B_sky F)`` over the patterns, the generalized
eigenvectors ``E v = nu^2 E_sky v`` are the directions, normalised so each one's sky image has unit norm, and
``nu`` is a direction's nullness: how much of it the data see (1: a pattern the offsets cannot touch; 0: an
exact gauge). The data's one-sigma constraint on a direction's sky amplitude is ``s / nu`` (``s`` the noise per
row), given in MJy/sr rms over the covered sky.

**Rule.** For the directions with ``nu`` below ``ratio`` (and at or above the exact gauges' :data:`GAUGE_RATIO`: the
sky's zero point and the plane keep the solve's own convention and the plane gauge's), the solved sky is projected
in map units onto their sky parts, that component is removed from the sky and its paired offsets added to the
offsets, so the sky holds none of what the data did not determine; the fit changes by exactly ``sqrt(c^T E c)`` in
residual norm (recorded). ``ratio=None``: the analysis alone, nothing changed.

Everything is recorded: the grid, every direction's nullness and one-sigma amplitude, the solved sky's component
along each, the component removed (the cal's ``null_modes`` group, the solve record's ``null_modes`` JSON);
:func:`render` draws the removed component on the grid.

One sky term at a time (``terms``: which; default the first); the systems ``setup_lsqr`` builds (one sky entry
per row). :class:`NullModes` is used by :func:`~selfcal.core.solve.apply_lsqr` (``null_modes=``; the run engine's
``sc.Fit(null_modes=sc.NullModes(...))``).
"""
from __future__ import annotations

import json
import logging
import math
import time

import numpy as np
from scipy.sparse import coo_matrix
from threadpoolctl import threadpool_limits

from .deflation import DEPENDENT, GAUGE_RATIO, _block_pinv_solve, _pieces

logger = logging.getLogger(__name__)

__all__ = ['NullModes', 'bspline_weights', 'render', 'GAUGE_RATIO']

#: Cells per axis of the spline grid are at least this many (a box narrower than a cell still gets a grid).
_MIN_CELLS = 1


def bspline_weights(t):
    """Uniform cubic B-splines at positions ``t`` (array, in cell units, ``0 <= t < n``): the first of the four
    non-zero basis functions (``i``: functions ``i`` to ``i + 3`` of the ``n + 3``) and their four weights,
    ``(len(t), 4)``, which sum to 1."""
    t = np.asarray(t, dtype=np.float64)
    i = np.floor(t).astype(np.int64)
    u = t - i
    u2 = u * u
    u3 = u2 * u
    w = np.stack([(1.0 - u) ** 3, 3.0 * u3 - 6.0 * u2 + 4.0, -3.0 * u3 + 3.0 * u2 + 3.0 * u + 1.0, u3], axis=1) / 6.0
    return i, w


def _grid(ix, iy, cell):
    """The spline grid of a term's covered pixels: ``(x0, y0, nx, ny)`` with ``nx, ny`` cells of ``cell`` px."""
    x0, x1, y0, y1 = int(ix.min()), int(ix.max()), int(iy.min()), int(iy.max())
    nx = max(_MIN_CELLS, int(math.ceil((x1 - x0 + 1) / float(cell))))
    ny = max(_MIN_CELLS, int(math.ceil((y1 - y0 + 1) / float(cell))))
    return x0, y0, nx, ny


def render(shape, grid, coefficients, cell):
    """The pattern combination ``coefficients`` (``(nx + 3) * (ny + 3)``, x-major) of the grid ``(x0, y0, nx,
    ny)`` with ``cell`` px cells, drawn on a map of ``shape``; zero outside the grid's box."""
    x0, y0, nx, ny = (int(v) for v in grid)
    h, w = int(shape[0]), int(shape[1])
    c = np.asarray(coefficients, dtype=np.float64).reshape(nx + 3, ny + 3)
    out = np.zeros((h, w), dtype=np.float64)
    xs = np.arange(x0, min(w, x0 + nx * int(math.ceil(cell))))
    ys = np.arange(y0, min(h, y0 + ny * int(math.ceil(cell))))
    if xs.size == 0 or ys.size == 0:
        return out
    jx, wx = bspline_weights(np.minimum((xs - x0) / float(cell), nx - 1e-9))
    jy, wy = bspline_weights(np.minimum((ys - y0) / float(cell), ny - 1e-9))
    for a in range(4):
        cx = c[jx + a]                                        # (len(xs), ny + 3)
        for b in range(4):
            out[np.ix_(ys, xs)] += (wy[:, b][:, None] * wx[:, a][None, :]) * cx[:, jy + b].T
    return out


class NullModes:
    """The null-mode analysis of one solve, and its rule (see the module docstring).

    Built on the column-scaled system the solver runs on, like :class:`~selfcal.core.deflation.Deflation`:
    ``storage`` (the matrix: a csr matrix, a :class:`~selfcal.core.blockcsr.BlockCSR` or a
    :class:`~selfcal.core.blockcsr.ColSplitCSR`, read only), ``scale`` (``x * scale`` is physical; None: no
    scaling), ``active_mask`` (the full layout's columns in the system), ``ref_shape`` and ``sky_names`` (the
    sky terms: the first ``len(sky_names) * h * w`` columns of the full layout), ``settings`` (a
    :class:`~selfcal.run.recipe.NullModes`: ``cell``, ``ratio``, ``terms``), ``threads`` (of the BLAS products).

    Then :meth:`component` (the solved sky's amplitude along every direction), :meth:`apply` (the rule on a
    solver solution, in place) and :meth:`summary` / :meth:`arrays` (what the record and the cal keep)."""

    def __init__(self, storage, *, scale, active_mask, ref_shape, sky_names, settings, threads=1):
        t0 = time.perf_counter()
        self.settings = settings
        self.cell = float(settings.cell)
        m, n = int(storage.shape[0]), int(storage.shape[1])
        h, w = int(ref_shape[0]), int(ref_shape[1])
        npix = h * w
        names = tuple(sky_names) if sky_names else ('sky',)
        nsb = len(names)
        terms = tuple(names[:1]) if settings.terms is None else tuple(settings.terms)
        unknown = [t for t in terms if t not in names]
        if unknown:
            raise ValueError(f"NullModes(terms={settings.terms!r}): no sky term {unknown} (the model's: {list(names)})")
        if len(terms) != 1:
            raise ValueError(f"NullModes: one sky term at a time (terms={list(terms)})")
        self.term = terms[0]
        j_term = names.index(self.term)
        # The columns with entries (all of a compacted system's), and the sky columns among them.
        covered = np.zeros(n, dtype=bool)
        for _r0, _r1, pieces in _pieces(storage):
            for _lip, cols, _data in pieces:
                covered[cols] = True
        S = None if covered.all() else np.flatnonzero(covered)
        del covered
        nS = n if S is None else int(S.size)
        self.S, self.nS = S, nS
        if active_mask is not None:
            full_sky = np.flatnonzero(np.asarray(active_mask, dtype=bool)[:nsb * npix])
        else:
            full_sky = np.arange(nsb * npix, dtype=np.int64)
        n_sky = int(full_sky.size)
        sky_S = np.arange(n_sky) if S is None else S[S < n_sky]
        self.n_sky_S = n_sky_S = int(sky_S.size)
        self.n_off = n_off = nS - n_sky_S
        fs = full_sky[sky_S]
        term_of = fs // npix
        pix = fs - term_of * npix
        iy, ix = np.divmod(pix, w)
        del fs, pix
        self._colmap = None
        if S is not None:
            cm = np.full(n, -1, dtype=np.int64)
            cm[S] = np.arange(nS)
            self._colmap = cm
        scale_S = None if scale is None else np.asarray(scale if S is None else np.asarray(scale)[S], dtype=np.float64)
        self._scale_sky = None if scale_S is None else scale_S[:n_sky_S]
        self._scale_off = None if scale_S is None else scale_S[n_sky_S:]
        col_norm_sky = np.ones(n_sky_S) if scale_S is None else 1.0 / self._scale_sky
        # The term's pixels among the sky columns, its spline grid, and each pixel's 4 x 4 weights.
        sel = np.flatnonzero(term_of == j_term)
        if sel.size == 0:
            raise ValueError(f"NullModes: the sky term {self.term!r} has no covered pixel")
        self._sel = sel
        self.grid = _grid(ix[sel], iy[sel], self.cell)
        x0, y0, nx, ny = self.grid
        self.nxp, self.nyp = nx + 3, ny + 3
        k = self.nxp * self.nyp
        self.k = k
        jx, wx = bspline_weights(np.minimum((ix[sel] - x0) / self.cell, nx - 1e-9))
        jy, wy = bspline_weights(np.minimum((iy[sel] - y0) / self.cell, ny - 1e-9))
        self._jx, self._jy = jx.astype(np.int32), jy.astype(np.int32)
        self._wx, self._wy = wx, wy                                        # float64: render() agrees to rounding
        self._cn = col_norm_sky[sel]                                        # solver pattern = physical x this
        del term_of, iy, ix
        self.n_pix = int(sel.size)
        # The Euclidean Gram of the patterns over the covered pixels (physical units): 256 pixel sums.
        self.Gphys = self._pattern_gram(np.ones(self.n_pix))
        logger.info(f"NullModes: {k} spline patterns ({nx} x {ny} cells of {self.cell:g} px) of the sky term "
                    f"{self.term!r} on {self.n_pix} covered pixels; {nS} columns ({n_sky_S} sky, {n_off} other)")
        rows = int(min(1 << 21, max(1 << 15, (1 << 30) // (8 * k))))
        self._rows = rows
        self._storage = storage
        with threadpool_limits(limits=int(max(1, threads)), user_api='blas'):
            # Pass 1: U = B_sky F per row chunk; E_sky += U^T U; R += (B_off^T U)^T; G = B_off^T B_off.
            t = time.perf_counter()
            E_sky = np.zeros((k, k))
            R = np.zeros((k, n_off))
            gparts = []
            for r0, r1, pieces in _pieces(storage, rows=rows):
                nr = r1 - r0
                U, sub = self._chunk_images(pieces, nr, with_offsets=True)
                E_sky += U.T @ U
                if sub is not None:
                    R += (sub.T @ U).T
                    g = (sub.T @ sub).tocoo()
                    gparts.append((g.row.astype(np.int64), g.col.astype(np.int64), g.data))
                    del g
                del U, sub
            if gparts:
                G = coo_matrix((np.concatenate([p[2] for p in gparts]),
                                (np.concatenate([p[0] for p in gparts]), np.concatenate([p[1] for p in gparts]))),
                               shape=(n_off, n_off)).tocsr()
            else:
                G = coo_matrix((n_off, n_off)).tocsr()
            del gparts
            G.sum_duplicates()
            t_pass1 = time.perf_counter() - t
            # The offset response of every pattern: g(F) = -G^+ B_off^T B_sky F.
            t = time.perf_counter()
            X, info = _block_pinv_solve(G, R.T)                              # (n_off, k)
            res = G @ (-X) + R.T
            self.pairing_residual = float(np.max(np.linalg.norm(res, axis=0) /
                                                 np.maximum(np.linalg.norm(R, axis=1), 1e-300)))
            self._response = -X                                              # solver units, (n_off, k)
            del G, R, X, res
            t_pair = time.perf_counter() - t
            # Pass 2: the paired images B z = B_sky F + B_off g; E += (B z)^T (B z).
            t = time.perf_counter()
            E = np.zeros((k, k))
            for r0, r1, pieces in _pieces(storage, rows=rows):
                nr = r1 - r0
                U, sub = self._chunk_images(pieces, nr, with_offsets=True)
                if sub is not None:
                    U += sub @ self._response
                E += U.T @ U
                del U, sub
            t_pass2 = time.perf_counter() - t
        self.E, self.E_sky = E, E_sky
        # The directions: E v = nu^2 E_sky v, with the patterns of no coverage (and dependent ones) left out.
        d = np.diag(E_sky)
        kept = d > DEPENDENT * max(float(d.max()), 1e-300)
        self.pattern_kept = kept
        Ek, Es = E[np.ix_(kept, kept)], E_sky[np.ix_(kept, kept)]
        w1, U1 = np.linalg.eigh(0.5 * (Es + Es.T))
        ok1 = w1 > DEPENDENT * max(float(w1.max()), 1e-300)
        T = U1[:, ok1] / np.sqrt(w1[ok1])
        E2 = T.T @ Ek @ T
        lam, U2 = np.linalg.eigh(0.5 * (E2 + E2.T))
        Vk = T @ U2                                                          # (kept, k_eff): E_sky-orthonormal
        V = np.zeros((k, Vk.shape[1]))
        V[kept] = Vk
        self.V = V
        self.nullness = np.sqrt(np.maximum(lam, 0.0))
        self.k_eff = int(self.nullness.size)
        # The directions' sky parts' Euclidean Gram (physical) and rms per unit amplitude over the covered pixels.
        self.Gd = V.T @ self.Gphys @ V
        self.sky_rms_unit = np.sqrt(np.maximum(np.diag(self.Gd), 0.0) / self.n_pix)
        self.setup_s = time.perf_counter() - t0
        self.applied = np.zeros(self.k_eff, dtype=bool)
        self.amplitude = None
        self.removed = None
        self.removed_rms = self.removed_peak = self.residual_change = math.nan
        self.noise_variance = math.nan
        logger.info(
            f"NullModes: offset response solved on {info['blocks']} blocks (largest {info['largest']}, "
            f"{info['zero_eigenvalues']} zero eigenvalues), normal-equation residual {self.pairing_residual:.1e}; "
            f"{self.k_eff} directions from {int(kept.sum())} of {k} patterns; nullness from "
            f"{self.nullness.min():.2e} to {self.nullness.max():.2e} (median {np.median(self.nullness):.2e}), "
            f"{int((self.nullness < GAUGE_RATIO).sum())} exact gauge(s); passes {t_pass1:.1f} s + {t_pass2:.1f} s, "
            f"blocks {t_pair:.1f} s; setup {self.setup_s:.1f} s")

    # ------------------------------------------------------------------ the patterns at the pixels
    def _pidx(self, a, c):
        """The pattern index of weight ``(a, c)`` at every covered pixel of the term."""
        return (self._jx.astype(np.int64) + a) * self.nyp + (self._jy.astype(np.int64) + c)

    def _pattern_gram(self, weights):
        """``sum_pixels weights * F F^T`` over the term's pixels (k x k), by 256 bincounts."""
        k = self.k
        G = np.zeros(k * k)
        wgt = np.asarray(weights, dtype=np.float64)
        pidx = [[self._pidx(a, c) for c in range(4)] for a in range(4)]
        wv = [[wgt * self._wx[:, a] * self._wy[:, c] for c in range(4)] for a in range(4)]
        for a in range(4):
            for c in range(4):
                base = pidx[a][c] * k
                for a2 in range(4):
                    for c2 in range(4):
                        G += np.bincount(base + pidx[a2][c2], weights=wv[a][c] * self._wx[:, a2] * self._wy[:, c2],
                                         minlength=k * k)
        return G.reshape(k, k)

    def pattern_values(self, coefficients, physical=True):
        """``sum_p coefficients[p] F_p`` at every covered pixel of the term (physical units, or the solver's
        with ``physical=False``)."""
        c = np.asarray(coefficients, dtype=np.float64)
        out = np.zeros(self.n_pix)
        for a in range(4):
            for c2 in range(4):
                out += c[self._pidx(a, c2)] * (self._wx[:, a] * self._wy[:, c2])
        return out if physical else out * self._cn

    def pattern_dots(self, values):
        """``F_p . values`` for every pattern (k,), ``values`` over the term's covered pixels."""
        v = np.asarray(values, dtype=np.float64)
        out = np.zeros(self.k)
        for a in range(4):
            for c in range(4):
                out += np.bincount(self._pidx(a, c), weights=v * self._wx[:, a] * self._wy[:, c], minlength=self.k)
        return out

    def _chunk_images(self, pieces, nr, with_offsets):
        """``U = B_sky F`` for a row chunk (nr x k, float64) and, with ``with_offsets``, the chunk's offset
        columns as a csr matrix (nr x n_off) or None."""
        k, n_sky_S = self.k, self.n_sky_S
        U = np.zeros(nr * k)
        offs = []
        # The term's pixels: their position among the sky columns -> index into the term's arrays.
        for lip, cols, data in pieces:
            pos = cols if self._colmap is None else self._colmap[cols]
            rr = np.repeat(np.arange(nr, dtype=np.int64), np.diff(lip))
            d64 = data.astype(np.float64)
            o = pos >= n_sky_S
            if with_offsets and o.any():
                offs.append((rr[o], pos[o] - n_sky_S, d64[o]))
            s = ~o
            if not s.any():
                continue
            loc = self._loc[pos[s]]                                           # -1: a sky column of another term
            ok = loc >= 0
            if not ok.any():
                continue
            loc, rrs, val = loc[ok], rr[s][ok], d64[s][ok] * self._cn[loc[ok]]
            for a in range(4):
                for c in range(4):
                    p = (self._jx[loc].astype(np.int64) + a) * self.nyp + (self._jy[loc].astype(np.int64) + c)
                    U += np.bincount(rrs * k + p, weights=val * self._wx[loc, a] * self._wy[loc, c], minlength=nr * k)
        sub = None
        if offs:
            rr = np.concatenate([x[0] for x in offs])
            cc = np.concatenate([x[1] for x in offs])
            dd = np.concatenate([x[2] for x in offs])
            sub = coo_matrix((dd, (rr, cc)), shape=(nr, self.n_off)).tocsr()
        return U.reshape(nr, k), sub

    @property
    def _loc(self):
        """Sky column (among the covered sky columns) -> index among the term's pixels, -1 otherwise."""
        loc = getattr(self, '_loc_cache', None)
        if loc is None:
            loc = np.full(self.n_sky_S, -1, dtype=np.int64)
            loc[self._sel] = np.arange(self._sel.size)
            self._loc_cache = loc
        return loc

    # ------------------------------------------------------------------ the solved sky and the rule
    def sigma(self, noise_variance):
        """Each direction's one-sigma sky amplitude from the data alone, MJy/sr rms over the covered pixels:
        ``sqrt(noise_variance) / nu`` times the direction's sky rms per unit amplitude (inf for an exact gauge)."""
        s = math.sqrt(max(float(noise_variance), 0.0))
        with np.errstate(divide='ignore'):
            return np.where(self.nullness > 0, s / np.where(self.nullness > 0, self.nullness, 1.0), np.inf) * self.sky_rms_unit

    def _sky_physical(self, x):
        xs = x if self.S is None else x[self.S]
        sky = np.asarray(xs[:self.n_sky_S], dtype=np.float64)[self._sel]
        return sky if self._scale_sky is None else sky * self._scale_sky[self._sel]

    def component(self, x):
        """The solved sky's amplitude along every direction (physical: the least-squares fit of the sky over the
        covered pixels to all directions' sky parts at once), (k_eff,)."""
        r = self.V.T @ self.pattern_dots(self._sky_physical(x))
        return self._solve_gram(self.Gd, r)

    @staticmethod
    def _solve_gram(G, r):
        w, U = np.linalg.eigh(0.5 * (G + G.T))
        ok = w > DEPENDENT * max(float(w.max()), 1e-300)
        return U[:, ok] @ ((U[:, ok].T @ r) / w[ok])

    def apply(self, x, noise_variance, residual_norm=math.nan):
        """The rule on the solver's solution ``x`` (its layout, in place): the directions with nullness in
        ``[GAUGE_RATIO, ratio)`` lose their sky component (a joint projection of the sky over the covered pixels
        onto their sky parts), the paired offsets take it. Records the directions' sigma (from ``noise_variance``),
        the sky's component along every direction before the rule, what was removed and the exact change of the
        residual norm (against ``residual_norm`` when given). ``ratio=None``: nothing changed. Returns ``x``."""
        self.noise_variance = float(noise_variance)
        self.amplitude = self.component(x)
        ratio = self.settings.ratio
        if ratio is None:
            return x
        sel = (self.nullness < float(ratio)) & (self.nullness >= GAUGE_RATIO)
        self.applied = sel
        if not sel.any():
            self.removed = np.zeros(self.k)
            self.removed_rms = self.removed_peak = self.residual_change = 0.0
            return x
        Vs = self.V[:, sel]
        r = Vs.T @ self.pattern_dots(self._sky_physical(x))
        alpha = self._solve_gram(Vs.T @ self.Gphys @ Vs, r)
        c = Vs @ alpha                                                        # pattern coefficients, physical
        self.removed = c
        d_sky = self.pattern_values(c)                                        # physical, the term's pixels
        self.removed_rms = float(np.sqrt(np.mean(d_sky ** 2)))
        self.removed_peak = float(np.max(np.abs(d_sky)))
        self.residual_change = float(np.sqrt(max(c @ self.E @ c, 0.0)))
        self.residual_norm = float(residual_norm)
        # x -= c . z: the sky part in solver units, the offsets by the paired response.
        xs = x if self.S is None else x[self.S]
        sky = xs[:self.n_sky_S]
        d_solver = d_sky * self._cn
        sky[self._sel] = (np.asarray(sky[self._sel], dtype=np.float64) - d_solver).astype(sky.dtype)
        off = xs[self.n_sky_S:]
        off[...] = (np.asarray(off, dtype=np.float64) - self._response @ c).astype(off.dtype)
        if self.S is not None:
            x[self.S] = xs
        logger.info(f"NullModes: {int(sel.sum())} of {self.k_eff} directions below nullness {ratio:g} set to zero sky "
                    f"amplitude: removed {self.removed_rms:.3e} rms / {self.removed_peak:.3e} peak over the covered "
                    f"pixels; the residual norm changes by {self.residual_change:.3e}"
                    + (f" ({self.residual_change / residual_norm:.1e} of {residual_norm:.6e})" if residual_norm == residual_norm and residual_norm > 0 else ''))
        return x

    # ------------------------------------------------------------------ what is kept
    def summary(self) -> dict:
        """What the solve's record keeps (reproducible run to run; the setup time apart)."""
        st = self.settings
        nu = self.nullness
        q = np.quantile(nu, [0.0, 0.1, 0.5, 0.9, 1.0]) if nu.size else np.full(5, math.nan)
        sig = self.sigma(self.noise_variance) if self.noise_variance == self.noise_variance else None
        out = {'cell': float(st.cell), 'basis': 'bspline', 'term': self.term, 'ratio': None if st.ratio is None else float(st.ratio),
               'grid': [int(v) for v in self.grid], 'patterns': int(self.k), 'patterns_kept': int(self.pattern_kept.sum()),
               'directions': int(self.k_eff), 'covered_pixels': int(self.n_pix), 'columns': int(self.nS),
               'pairing_residual': float(self.pairing_residual),
               'nullness_quantiles': [float(v) for v in q], 'exact_gauges': int((nu < GAUGE_RATIO).sum()),
               'below_ratio': None if st.ratio is None else int(((nu < float(st.ratio)) & (nu >= GAUGE_RATIO)).sum()),
               'applied': int(self.applied.sum())}
        if sig is not None:
            fin = np.isfinite(sig)
            out['sigma_sky_rms_max'] = float(sig[fin].max()) if fin.any() else None
            if self.applied.any():
                out['sigma_sky_rms_applied_min'] = float(sig[self.applied].min())
        if self.removed is not None:
            out.update(removed_rms=self.removed_rms, removed_peak=self.removed_peak, residual_change=self.residual_change)
        if self.amplitude is not None and self.k_eff:
            a = np.abs(self.amplitude) * self.sky_rms_unit
            out['component_rms_max'] = float(a.max())
        return out

    def summary_json(self) -> str:
        return json.dumps(self.summary(), sort_keys=True)

    def arrays(self) -> dict:
        """The datasets of the cal's ``null_modes`` group: ``nullness`` and ``sigma_sky_rms`` per direction,
        ``directions`` (k x k_eff, pattern coefficients, E_sky-orthonormal), ``amplitude`` (the solved sky's
        component along each, before the rule, physical), ``applied`` (which the rule took), ``removed`` (the
        component removed, pattern coefficients, physical), ``sky_rms_unit``; and the group's attributes
        (:meth:`summary`)."""
        out = {'nullness': self.nullness.astype(np.float64), 'directions': self.V.astype(np.float64),
               'sky_rms_unit': self.sky_rms_unit.astype(np.float64), 'applied': self.applied.copy(),
               'pattern_kept': self.pattern_kept.copy()}
        if self.noise_variance == self.noise_variance:
            out['sigma_sky_rms'] = self.sigma(self.noise_variance).astype(np.float64)
        if self.amplitude is not None:
            out['amplitude'] = self.amplitude.astype(np.float64)
        out['removed'] = (np.zeros(self.k) if self.removed is None else self.removed).astype(np.float64)
        return out

    def render_removed(self, shape):
        """The removed component on a map of ``shape`` (zero when nothing was removed)."""
        c = np.zeros(self.k) if self.removed is None else self.removed
        return render(shape, self.grid, c, self.cell)


def write_group(f, nm, name='null_modes'):
    """Write the analysis ``nm`` into the open cal file ``f`` as the group ``name`` (replaced when present):
    the arrays of :meth:`NullModes.arrays` as datasets, the summary as attributes (JSON-friendly values)."""
    if name in f:
        del f[name]
    grp = f.create_group(name)
    for k, v in nm.arrays().items():
        grp.create_dataset(k, data=v)
    for k, v in nm.summary().items():
        grp.attrs[k] = json.dumps(v) if isinstance(v, (list, dict)) or v is None else v
