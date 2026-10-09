"""CGLS: conjugate gradients on the least-squares normal equations, with the solver hooks of this package.

CGLS minimises ``|A x - b|^2 + damp^2 |x|^2`` with one product by ``A`` and one by ``A^T`` per
iteration, the same products and, in exact arithmetic, the same iterates as LSQR
(:mod:`~selfcal.core.lsqr_inplace`). It differs in what it carries between iterations, which is what
matters once a solve has converged in floating point:

- the step along the search direction is ``|s|^2 / |A p|^2``, with ``s`` the gradient: when the
  gradient has fallen to rounding, the step vanishes and the iterate stays where it is (LSQR
  renormalises its vectors every iteration, so after convergence it keeps taking full-size steps made
  of rounding noise);
- the residual is carried as a vector in the products' precision and recomputed from ``b - A x``
  every :data:`REFRESH` iterations, so what the solver believes about the misfit cannot drift from the
  truth; the iterate, the direction and the gradient are kept in ``vectors``' precision (float64 by
  default), the reductions always in float64.

So a CGLS solve can be given a generous iteration count: past convergence the extra iterations cost
time, not accuracy. Its stopping tests are LSQR's, on the same quantities (``|r| / |b|`` against
``btol``; ``|A^T r| / (|A| |r|)`` against ``atol``, with ``|A|`` estimated as the largest ``|A p| / |p|``
seen), so ``Fit(tolerance=)`` means the same; it has no estimate of ``cond(A)``, so ``conlim`` and the
``"condition"`` test never stop it. The history, the callback (snapshots), the watch (monitors and stop
rules) and ``constrain`` (deflation) are the hooks of :func:`~selfcal.core.lsqr_inplace.lsqr_inplace`.

Memory: the residual and one product in the products' precision (``m`` each, like LSQR's ``u`` and its
product), and three ``n``-vectors in ``vectors``' precision (the iterate, the direction, the gradient).
The right-hand side is read again for each refresh (``refresh``: a function returning it, an array or a
parked copy; None: no refresh).
"""
from __future__ import annotations

import numpy as np
from math import sqrt
from scipy.sparse.linalg import aslinearoperator

__all__ = ['cgls', 'REFRESH']

#: The residual is recomputed from ``b - A x`` every this many iterations.
REFRESH = 50
_BLOCK = 50_000_000


from concurrent.futures import ThreadPoolExecutor

from .lsqr_inplace import _vec_threads

_POOL = None
_BUFS = {}


def _pool():
    global _POOL
    if _POOL is None:
        n = _vec_threads()
        _POOL = ThreadPoolExecutor(max_workers=n) if n > 1 else False
    return _POOL or None


def _buffer(key, dtype):
    """One scratch block per (thread key, dtype), kept between calls."""
    b = _BUFS.get((key, dtype))
    if b is None:
        b = _BUFS[(key, dtype)] = np.empty(_BLOCK, dtype=dtype)
    return b


def _blocks(n):
    return [(s, min(n, s + _BLOCK)) for s in range(0, n, _BLOCK)]


import threading

_LOCAL = threading.local()


def _run(fn, blocks):
    """``fn(thread key, start, stop)`` over the blocks, in threads when the vector is long; the results in
    block order (so a reduction's sum is the same for any thread count). The key names the running thread's
    own scratch."""
    pool = _pool() if len(blocks) > 1 else None

    def call(t):
        key = getattr(_LOCAL, 'key', None)
        if key is None:
            key = _LOCAL.key = threading.get_ident()
        return fn(key, *t)
    if pool is None:
        return [call(t) for t in blocks]
    return list(pool.map(call, blocks))


def _dot(a, b):
    """``a . b`` accumulated in float64: each block copied into a float64 scratch block (the same products and
    sums as a blocked ``astype`` + ``dot``, without faulting in a fresh copy), the blocks in threads, their
    partial sums added in block order (the result is independent of the thread count)."""
    def part(k, s, e):
        t = _buffer(k, np.float64)[:e - s]
        np.copyto(t, a[s:e], casting='unsafe')
        return float(np.dot(t, t if a is b else b[s:e].astype(np.float64, copy=False)))
    acc = 0.0
    for v in _run(part, _blocks(a.shape[0])):
        acc += v
    return acc


def _sub_scaled(r, alpha, q):
    """``r -= alpha * q`` in place, in blocks through scratch blocks of ``r``'s dtype, the blocks in threads
    (elementwise: the same bytes in any order)."""
    a = r.dtype.type(alpha)

    def part(k, s, e):
        t = _buffer(k, r.dtype)[:e - s]
        np.multiply(q[s:e], a, out=t, casting='unsafe')
        np.subtract(r[s:e], t, out=r[s:e])
    _run(part, _blocks(r.shape[0]))


def cgls(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8, iter_lim=None, show=False, calc_var=False,
         x0=None, x0_owned=False, history=None, callback=None, callback_every=1, watch=None, constrain=None,
         refresh=None, vectors=np.float64):
    """CGLS with the arguments and the return tuple of :func:`~selfcal.core.lsqr_inplace.lsqr_inplace`
    (``conlim`` and ``calc_var`` accepted, unused: no condition estimate, no variance).

    ``b`` is taken over (its buffer becomes the residual when its dtype is the products'); ``x0``
    (the full solver layout) is the start, copied unless ``x0_owned``. ``refresh``: a function returning
    the right-hand side the solve started from (read in blocks: an array or a
    :class:`~selfcal.core.spill.ParkedVector`), for the residual's recomputation every :data:`REFRESH`
    iterations; None: never recomputed. ``vectors``: the dtype of the iterate, the direction and the
    gradient. The returned ``x`` is in that dtype; ``anorm`` is the estimate above, ``acond`` 0."""
    A = aslinearoperator(A)
    m, n = A.shape
    pdt = np.dtype(A.dtype)                            # the products' precision
    vdt = np.dtype(vectors)
    if iter_lim is None:
        iter_lim = 2 * n
    damp = float(damp)
    d2 = damp * damp
    var = np.zeros(0)

    b = np.asarray(b)
    bnorm = sqrt(_dot(b, b))
    if x0 is None:
        x = np.zeros(n, dtype=vdt)
        r = b if b.dtype == pdt else b.astype(pdt)
    else:
        x = np.asarray(x0, dtype=vdt) if x0_owned and np.asarray(x0).dtype == vdt else np.array(x0, dtype=vdt)
        r = b if b.dtype == pdt else b.astype(pdt)
        np.subtract(r, A.matvec(x.astype(pdt, copy=False)), out=r, casting='unsafe')
    del b

    def gradient():
        g = np.asarray(A.rmatvec(r))
        g = g.astype(vdt, copy=False) if g.dtype == vdt else g.astype(vdt)
        if d2 > 0:
            g -= d2 * x
        if constrain is not None:
            constrain(g)
        return g

    s = gradient()
    gamma = _dot(s, s)
    arnorm = sqrt(gamma)
    p = s.copy()
    rnorm = sqrt(_dot(r, r))
    anorm = 0.0
    itn, istop = 0, 0

    def rec(itn, rnorm, arnorm, test1, test2):
        if history is not None:
            history.add(itn, rnorm, sqrt(rnorm * rnorm + d2 * _dot(x, x)) if d2 > 0 else rnorm, arnorm, anorm,
                        0.0, sqrt(_dot(x, x)), test1, test2)

    if show:
        print(' ')
        print('CGLS            Least-squares solution of  Ax = b')
        print(f'The matrix A has {m} rows and {n} columns; products in {pdt.name}, vectors in {vdt.name}')
        print(f'damp = {damp:20.14e}   atol = {atol:8.2e}   btol = {btol:8.2e}   iter_lim = {iter_lim:8g}')
    rec(0, rnorm, arnorm, rnorm / bnorm if bnorm > 0 else 0.0, float('nan'))
    if watch is not None:
        watch(0, x, 0, None)
    if gamma == 0 or rnorm == 0:
        return x, 0, 0, rnorm, rnorm, anorm, 0.0, arnorm, sqrt(_dot(x, x)), var

    while itn < iter_lim:
        itn += 1
        q = np.asarray(A.matvec(p.astype(pdt, copy=False)))
        qq = _dot(q, q) + (d2 * _dot(p, p) if d2 > 0 else 0.0)
        if qq <= 0:
            istop = 5
            break
        pp = _dot(p, p)
        if pp > 0:
            anorm = max(anorm, sqrt((qq - d2 * pp) / pp)) if qq > d2 * pp else anorm
        alpha = gamma / qq
        x += vdt.type(alpha) * p
        _sub_scaled(r, alpha, q)
        del q
        if refresh is not None and itn % REFRESH == 0:
            b0 = refresh()
            if b0 is not None:
                ax = np.asarray(A.matvec(x.astype(pdt, copy=False)))
                for s0 in range(0, m, _BLOCK):
                    e0 = min(m, s0 + _BLOCK)
                    np.subtract(np.asarray(b0[s0:e0], dtype=pdt), ax[s0:e0], out=r[s0:e0], casting='unsafe')
                del ax, b0
        s = gradient()
        gamma_new = _dot(s, s)
        rnorm = sqrt(_dot(r, r))
        arnorm = sqrt(max(gamma_new, 0.0))
        test1 = rnorm / bnorm if bnorm > 0 else 0.0
        test2 = arnorm / (anorm * rnorm) if anorm > 0 and rnorm > 0 else 0.0
        if itn >= iter_lim:
            istop = 7
        if not np.isfinite(gamma_new):
            istop = 5
        if test2 <= atol:
            istop = 2
        if test1 <= btol:
            istop = 1
        rec(itn, rnorm, arnorm, test1, test2)
        if watch is not None:
            istop = watch(itn, x, istop, (test1 <= btol, test2 <= atol, False, False, not np.isfinite(gamma_new),
                                          False, itn >= iter_lim))
        if show and (itn <= 10 or itn >= iter_lim - 10 or istop != 0 or itn % 10 == 0):
            print(f'{itn:6g} {x[0]:12.5e} {rnorm:10.3e} {arnorm:10.3e}  {test1:8.1e} {test2:8.1e} {anorm:8.1e}')
        if callback is not None and istop == 0 and itn % callback_every == 0:
            callback(itn, x)
        if istop != 0:
            break
        beta = gamma_new / gamma
        gamma = gamma_new
        p *= vdt.type(beta)
        p += s
        del s
    if show:
        print(f'CGLS finished: istop = {istop}, itn = {itn}, r1norm = {rnorm:.1e}, arnorm = {arnorm:.1e}')
    return (x, istop, itn, rnorm, sqrt(rnorm * rnorm + d2 * _dot(x, x)) if d2 > 0 else rnorm, anorm, 0.0,
            arnorm, sqrt(_dot(x, x)), var)
