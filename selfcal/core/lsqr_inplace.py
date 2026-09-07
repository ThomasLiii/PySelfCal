"""LSQR with in-place vector updates — bit-identical to ``scipy.sparse.linalg.lsqr``.

scipy's ``lsqr`` (Paige & Saunders) writes every vector update as a fresh
expression, ``u = A.matvec(v) - alfa * u``, so each iteration transiently
holds five m-length vectors (``b``, ``u``, the matvec output, ``alfa*u`` and
the difference) and seven n-length ones, on top of the matrix. For a
production tile (m ≈ 2e9 rows, float32) that is +20-40 GB of periodic spikes
above the solve plateau — visible as the sawtooth in every RSS trace — and
``b`` stays alive for the whole solve although it is read exactly once.

This copy keeps scipy's scalar recurrences VERBATIM (same expressions, same
scalar types, same ``np.linalg.norm`` calls on vectors of the same length
and dtype) and only rewrites the vector updates as in-place elementwise
operations with the same rounding sequence:

    u = A.matvec(v) - alfa*u   ->   u *= alfa;  np.subtract(out, u, out=u)
    v = A.rmatvec(u) - beta*v  ->   v *= beta;  np.subtract(out, v, out=v)
    dk = (1/rho)*w             ->   np.multiply(w, 1/rho, out=tmp)
    x = x + t1*w               ->   np.multiply(w, t1, out=tmp);  x += tmp
    w = v + t2*w               ->   w *= t2;  w += v

Elementwise IEEE operations have no accumulation order, scalar*array is
commutative per element, and every reduction (the norms) still runs on an
array with the same values, length and dtype — so ``x`` and every returned
scalar are bit-identical to scipy's (unit-tested against scipy 1.16.2 for
float32 and float64 systems, with and without ``x0``). Steady footprint:
``u`` + ``x, v, w, tmp``; transient: one matvec / rmatvec output.

NumPy promotion is mirrored, not assumed: before each in-place update the
target buffer is upcast (exactly) whenever the out-of-place expression would
have produced a wider dtype — e.g. scipy's ``damp > 0`` scalar chain yields a
float64 ``rho`` (``math.sqrt`` then ``np.sign``), after which ``x`` and ``w``
become float64 in scipy too. In the all-float32, ``damp = 0`` regime
production uses, nothing promotes and no extra vector is ever allocated.
``b`` is released after its single use; pass it without keeping another
reference (e.g. ``lsqr_inplace(op, holder.pop(), ...)``) to actually free it.
``var`` is only allocated when ``calc_var`` (scipy allocates an n-length
float64 array regardless); otherwise an empty array is returned in its slot.
"""
import numpy as np
from math import sqrt
from scipy.sparse.linalg import aslinearoperator
from scipy.sparse.linalg._isolve.lsqr import _sym_ortho

eps = np.finfo(np.float64).eps

_MSG = ('The exact solution is  x = 0                              ',
        'Ax - b is small enough, given atol, btol                  ',
        'The least-squares solution is good enough, given atol     ',
        'The estimate of cond(Abar) has exceeded conlim            ',
        'Ax - b is small enough for this machine                   ',
        'The least-squares solution is good enough for this machine',
        'Cond(Abar) seems to be too large for this machine         ',
        'The iteration limit has been reached                      ')


def _promote(arr, other):
    """``arr`` widened (exactly) to the dtype ``arr <op> other`` would have."""
    rt = np.result_type(arr, other)
    return arr if rt == arr.dtype else arr.astype(rt)


def _scratch(tmp, arr, scalar):
    """Scratch buffer of the dtype ``scalar * arr`` would have."""
    rt = np.result_type(arr, scalar)
    return tmp if tmp.dtype == rt else np.empty(arr.shape, rt)


def lsqr_inplace(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
                 iter_lim=None, show=False, calc_var=False, x0=None,
                 x0_owned=False):
    """Drop-in for ``scipy.sparse.linalg.lsqr`` (same arguments, same return
    tuple) with in-place vector updates. See the module docstring.

    ``x0`` is copied unless ``x0_owned=True``, in which case its buffer is
    used as the solution vector directly (the caller must hold no other
    reference it cares about) — one n-length vector less."""
    A = aslinearoperator(A)
    b = np.atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()

    m, n = A.shape
    if iter_lim is None:
        iter_lim = 2 * n
    var = np.zeros(n) if calc_var else np.zeros(0)

    if x0 is not None:
        x0 = np.asarray(x0) if x0_owned else np.array(x0)

    if show:
        print(' ')
        print('LSQR            Least-squares solution of  Ax = b')
        str1 = f'The matrix A has {m} rows and {n} columns'
        str2 = f'damp = {damp:20.14e}   calc_var = {calc_var:8g}'
        str3 = f'atol = {atol:8.2e}                 conlim = {conlim:8.2e}'
        str4 = f'btol = {btol:8.2e}               iter_lim = {iter_lim:8g}'
        print(str1)
        print(str2)
        print(str3)
        print(str4)

    itn = 0
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1/conlim
    anorm = 0
    acond = 0
    dampsq = damp**2
    ddnorm = 0
    res2 = 0
    xnorm = 0
    xxnorm = 0
    z = 0
    cs2 = -1
    sn2 = 0

    # Set up the first vectors u and v for the bidiagonalization.
    # These satisfy  beta*u = b - A@x,  alfa*v = A'@u.
    bnorm = np.linalg.norm(b)
    if x0 is None:
        x = np.zeros(n)
        beta = bnorm.copy()
        u = b                       # owned from here on (never read again as b)
    else:
        x = x0
        out = A.matvec(x)
        u = _promote(b, out)                # u = b - A x  (b's buffer if no promotion)
        np.subtract(u, out, out=u)
        del out
        beta = np.linalg.norm(u)
    del b

    if beta > 0:
        u = _promote(u, 1/beta)
        u *= (1/beta)
        v = A.rmatvec(u)
        alfa = np.linalg.norm(v)
    else:
        v = x.copy()
        alfa = 0

    if alfa > 0:
        v = _promote(v, 1 / alfa)
        v *= (1 / alfa)
    w = v.copy()
    tmp = np.empty_like(w)          # scratch for dk / t1*w

    rhobar = alfa
    phibar = beta
    rnorm = beta
    r1norm = rnorm
    r2norm = rnorm

    arnorm = alfa * beta
    if arnorm == 0:
        if show:
            print(_MSG[0])
        return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var

    head1 = '   Itn      x[0]       r1norm     r2norm '
    head2 = ' Compatible    LS      Norm A   Cond A'

    if show:
        print(' ')
        print(head1, head2)
        test1 = 1
        test2 = alfa / beta
        str1 = f'{itn:6g} {x[0]:12.5e}'
        str2 = f' {r1norm:10.3e} {r2norm:10.3e}'
        str3 = f'  {test1:8.1e} {test2:8.1e}'
        print(str1, str2, str3)

    while itn < iter_lim:
        itn = itn + 1
        # beta*u = A v - alfa*u   (in place: u <- alfa*u; u <- out - u)
        out = A.matvec(v)
        u = _promote(u, alfa)
        u *= alfa
        u = _promote(u, out)
        np.subtract(out, u, out=u)
        del out
        beta = np.linalg.norm(u)

        if beta > 0:
            u = _promote(u, 1/beta)
            u *= (1/beta)
            anorm = sqrt(anorm**2 + alfa**2 + beta**2 + dampsq)
            # alfa*v = A' u - beta*v   (in place: v <- beta*v; v <- out - v)
            out = A.rmatvec(u)
            v = _promote(v, beta)
            v *= beta
            v = _promote(v, out)
            np.subtract(out, v, out=v)
            del out
            alfa = np.linalg.norm(v)
            if alfa > 0:
                v = _promote(v, 1 / alfa)
                v *= (1 / alfa)

        if damp > 0:
            rhobar1 = sqrt(rhobar**2 + dampsq)
            cs1 = rhobar / rhobar1
            sn1 = damp / rhobar1
            psi = sn1 * phibar
            phibar = cs1 * phibar
        else:
            rhobar1 = rhobar
            psi = 0.

        cs, sn, rho = _sym_ortho(rhobar1, beta)

        theta = sn * alfa
        rhobar = -cs * alfa
        phi = cs * phibar
        phibar = sn * phibar
        tau = sn * phi

        # Update x and w.
        t1 = phi / rho
        t2 = -theta / rho
        tmp = _scratch(tmp, w, (1 / rho))
        np.multiply(w, (1 / rho), out=tmp)          # dk = (1/rho) * w
        ddnorm = ddnorm + np.linalg.norm(tmp)**2
        if calc_var:
            var = var + tmp**2
        tmp = _scratch(tmp, w, t1)
        np.multiply(w, t1, out=tmp)                 # t1 * w
        x = _promote(x, tmp)
        x += tmp                                    # x = x + t1*w
        w = _promote(w, t2)
        w *= t2
        w = _promote(w, v)
        w += v                                      # w = v + t2*w

        delta = sn2 * rho
        gambar = -cs2 * rho
        rhs = phi - delta * z
        zbar = rhs / gambar
        xnorm = sqrt(xxnorm + zbar**2)
        gamma = sqrt(gambar**2 + theta**2)
        cs2 = gambar / gamma
        sn2 = theta / gamma
        z = rhs / gamma
        xxnorm = xxnorm + z**2

        acond = anorm * sqrt(ddnorm)
        res1 = phibar**2
        res2 = res2 + psi**2
        rnorm = sqrt(res1 + res2)
        arnorm = alfa * abs(tau)

        if damp > 0:
            r1sq = rnorm**2 - dampsq * xxnorm
            r1norm = sqrt(abs(r1sq))
            if r1sq < 0:
                r1norm = -r1norm
        else:
            r1norm = rnorm
        r2norm = rnorm

        test1 = rnorm / bnorm
        test2 = arnorm / (anorm * rnorm + eps)
        test3 = 1 / (acond + eps)
        t1 = test1 / (1 + anorm * xnorm / bnorm)
        rtol = btol + atol * anorm * xnorm / bnorm

        if itn >= iter_lim:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        if show:
            prnt = False
            if n <= 40:
                prnt = True
            if itn <= 10:
                prnt = True
            if itn >= iter_lim-10:
                prnt = True
            if test3 <= 2*ctol:
                prnt = True
            if test2 <= 10*atol:
                prnt = True
            if test1 <= 10*rtol:
                prnt = True
            if istop != 0:
                prnt = True

            if prnt:
                str1 = f'{itn:6g} {x[0]:12.5e}'
                str2 = f' {r1norm:10.3e} {r2norm:10.3e}'
                str3 = f'  {test1:8.1e} {test2:8.1e}'
                str4 = f' {anorm:8.1e} {acond:8.1e}'
                print(str1, str2, str3, str4)

        if istop != 0:
            break

    if show:
        print(' ')
        print('LSQR finished')
        print(_MSG[istop])
        print(' ')
        str1 = f'istop ={istop:8g}   r1norm ={r1norm:8.1e}'
        str2 = f'anorm ={anorm:8.1e}   arnorm ={arnorm:8.1e}'
        str3 = f'itn   ={itn:8g}   r2norm ={r2norm:8.1e}'
        str4 = f'acond ={acond:8.1e}   xnorm  ={xnorm:8.1e}'
        print(str1 + '   ' + str2)
        print(str3 + '   ' + str4)
        print(' ')

    return x, istop, itn, r1norm, r2norm, anorm, acond, arnorm, xnorm, var
