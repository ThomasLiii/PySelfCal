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
parked copy; None: no refresh, except in a damped solve, which then keeps a copy of ``b``: ``m`` more).
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


def _add_scaled(x, alpha, p):
    """``x += alpha * p`` in place (the product rounded, then the sum), in blocks in threads."""
    a = x.dtype.type(alpha)

    def part(k, s, e):
        t = _buffer(k, x.dtype)[:e - s]
        np.multiply(p[s:e], a, out=t, casting='unsafe')
        np.add(x[s:e], t, out=x[s:e])
    _run(part, _blocks(x.shape[0]))


def _scale_add(p, beta, s_):
    """``p = beta * p + s_`` in place (``p *= beta; p += s_``: the same two roundings), in blocks in threads."""
    b = p.dtype.type(beta)

    def part(k, s, e):
        np.multiply(p[s:e], b, out=p[s:e])
        np.add(p[s:e], s_[s:e], out=p[s:e])
    _run(part, _blocks(p.shape[0]))


def _as_dtype(v, dtype, held, key):
    """``v`` in ``dtype``: ``v`` itself when it is, else ``v.astype(dtype)``'s values written block by block
    in the threads into a buffer ``held[key]`` that the caller keeps between iterations (no fresh vector
    faulted in per iteration)."""
    dtype = np.dtype(dtype)
    if v.dtype == dtype:
        return v
    buf = held.get(key)
    if buf is None or buf.shape != v.shape or buf.dtype != dtype:
        buf = held[key] = np.empty(v.shape, dtype=dtype)

    def part(k, s, e):
        np.copyto(buf[s:e], v[s:e], casting='unsafe')
    _run(part, _blocks(v.shape[0]))
    return buf


def cgls(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8, iter_lim=None, show=False, calc_var=False,
         x0=None, x0_owned=False, history=None, callback=None, callback_every=1, watch=None, constrain=None,
         refresh=None, vectors=np.float64, gradient='recomputed'):
    """CGLS with the arguments and the return tuple of :func:`~selfcal.core.lsqr_inplace.lsqr_inplace`
    (``conlim`` and ``calc_var`` accepted, unused: no condition estimate, no variance).

    ``gradient``: how the gradient ``A^T r`` of each iteration is had. ``'recomputed'``: from the residual, a
    transpose product per iteration (two passes over the matrix with the matvec). ``'recurrence'``: carried
    along, ``s - alpha * (A^T A p + damp^2 p)``, the transpose product being that of the matvec's own result,
    which an operator with ``normal_product(p) -> (A p, A^T A p)`` gives from one pass over the matrix (the
    observation table) and any other from a second product of ``A p``, in float64 either way: a gradient
    carried by recurrence keeps the rounding of every term it was built from, where a recomputed one keeps
    only its own, so the terms are made in the vectors' precision; the gradient is recomputed from the
    residual at every refresh (:data:`REFRESH` iterations), as the residual is, and whenever the recurrence's
    gradient norm grows from one iteration to the next (past convergence the recurrence's rounding takes over
    and it drifts, where a gradient from the residual stays at the rounding floor). The two forms differ by the
    rounding of the recurrence, not in what they converge to.

    ``b`` is taken over (its buffer becomes the residual when its dtype is the products'); ``x0``
    (the full solver layout) is the start, copied unless ``x0_owned``. ``refresh``: a function returning
    the right-hand side the solve started from (read in blocks: an array or a
    :class:`~selfcal.core.spill.ParkedVector`), for the residual's recomputation every :data:`REFRESH`
    iterations; None: never recomputed, unless the solve is damped (``damp > 0``), which then keeps a copy
    of ``b`` to recompute it from: the recursive residual drifts by rounding once the solve has converged,
    and the damping term turns that drift into a divergence on some BLAS kernels (OpenBLAS's Haswell and
    Zen kernels: x1.14 per iteration). ``vectors``: the dtype of the iterate, the direction and the
    gradient. The returned ``x`` is in that dtype; ``anorm`` is the estimate above, ``acond`` 0."""
    if gradient not in ('recomputed', 'recurrence'):
        raise ValueError(f"gradient={gradient!r}: 'recomputed' or 'recurrence'")
    normal = getattr(A, 'normal_product', None) if gradient == 'recurrence' else None
    rmatvec64 = getattr(A, 'rmatvec_promoted', None) if gradient == 'recurrence' else None
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
    if refresh is None and d2 > 0:                     # a damped solve needs its true residual (see refresh)
        kept = b.copy()
        refresh = lambda: kept  # noqa: E731
    rdt = vdt if gradient == 'recurrence' else pdt        # the recurrence keeps the residual in the vectors' precision
    if x0 is None:
        x = np.zeros(n, dtype=vdt)
        r = b if b.dtype == rdt else b.astype(rdt)
    else:
        x = np.asarray(x0, dtype=vdt) if x0_owned and np.asarray(x0).dtype == vdt else np.array(x0, dtype=vdt)
        r = b if b.dtype == rdt else b.astype(rdt)
        np.subtract(r, A.matvec(x.astype(pdt, copy=False)), out=r, casting='unsafe')
    del b

    held = {}                                          # this solve's cast buffers (the direction, the gradient)

    def gradient_from_residual():
        g = np.asarray(rmatvec64(r) if rmatvec64 is not None and r.dtype != pdt else A.rmatvec(r))
        g = _as_dtype(g, vdt, held, 'gradient')
        if d2 > 0:
            g -= d2 * x
        if constrain is not None:
            constrain(g)
        return g

    s = gradient_from_residual()
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
        pd = _as_dtype(p, pdt, held, 'direction')
        t = None
        if gradient == 'recurrence':
            if normal is not None:
                q, t = normal(pd)
                q, t = np.asarray(q), np.asarray(t)
            else:                                  # no fused product: the transpose product of q, in float64
                q = np.asarray(A.matvec(pd))
                t = np.asarray(A.rmatvec(q.astype(np.float64, copy=False)))
        else:
            q = np.asarray(A.matvec(pd))
        qq = _dot(q, q) + (d2 * _dot(p, p) if d2 > 0 else 0.0)
        if qq <= 0:
            istop = 5
            break
        pp = _dot(p, p)
        if pp > 0:
            anorm = max(anorm, sqrt((qq - d2 * pp) / pp)) if qq > d2 * pp else anorm
        alpha = gamma / qq
        _add_scaled(x, alpha, p)
        _sub_scaled(r, alpha, q)
        del q
        refreshed = False
        if refresh is not None and itn % REFRESH == 0:
            b0 = refresh()
            if b0 is not None:
                ax = np.asarray(A.matvec(x.astype(pdt, copy=False)))
                for s0 in range(0, m, _BLOCK):
                    e0 = min(m, s0 + _BLOCK)
                    np.subtract(np.asarray(b0[s0:e0], dtype=pdt), ax[s0:e0], out=r[s0:e0], casting='unsafe')
                del ax, b0
                refreshed = True
        if t is not None and not refreshed:
            # the gradient by recurrence: s <- s - alpha (A^T A p + damp^2 p), kept in the constrained subspace
            tv = _as_dtype(t, vdt, held, 'normal')
            if d2 > 0:
                _add_scaled(tv, d2, p)
            _sub_scaled(s, alpha, tv)
            if constrain is not None:
                constrain(s)
        else:
            s = gradient_from_residual()
        gamma_new = _dot(s, s)
        if t is not None and not refreshed and gamma_new > gamma:
            # the recurrence's gradient grew: past convergence its rounding takes over and the recurrence drifts,
            # where a gradient from the residual stays at the rounding floor; recompute it (one more pass)
            s = gradient_from_residual()
            gamma_new = _dot(s, s)
        del t
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
        _scale_add(p, beta, s)
        if gradient == 'recomputed':
            del s
    if show:
        print(f'CGLS finished: istop = {istop}, itn = {itn}, r1norm = {rnorm:.1e}, arnorm = {arnorm:.1e}')
    return (x, istop, itn, rnorm, sqrt(rnorm * rnorm + d2 * _dot(x, x)) if d2 > 0 else rnorm, anorm, 0.0,
            arnorm, sqrt(_dot(x, x)), var)
