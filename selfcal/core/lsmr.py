"""LSMR, scipy's, with its state recorded at every iteration.

A copy of ``scipy.sparse.linalg.lsmr`` (scipy 1.16.2; Fong & Saunders 2011) whose only additions are
the ``history`` argument: a :class:`~selfcal.core.solve_record.SolveHistory` that receives the
scalars the solver computes anyway (``normr``, ``normar``, ``normA``, ``condA``, ``normx`` and the
two stopping ratios) before the first iteration and after each one, and ``callback(itn, x)``,
called with the iterate every ``callback_every`` iterations while the solve goes on (the snapshots
of :mod:`selfcal.core.snapshots`), and ``watch(itn, x, istop, tests)``, called after the history
row of every iteration, which returns the ``istop`` the solve goes on with (the monitors and stop
rules of :mod:`selfcal.core.monitor`). scipy's ``lsmr`` has no hook for any of them. The statements
of the solve are scipy's, in scipy's order, so ``x`` and every returned scalar are bit-identical to
scipy's (``tests/test_solve_record.py`` checks float32 and float64 systems, with and without ``x0``
and damping); with ``history``, ``callback`` and ``watch`` None it is scipy's function. The Givens
rotation it uses, scipy's private ``_sym_ortho``, is copied here too, so the
module imports only scipy's public API.

Copyright (C) 2010 David Fong and Michael Saunders (the algorithm and scipy's implementation).
"""
from math import sqrt

from numpy import atleast_1d, inf, result_type, sign, zeros
from numpy.linalg import norm
from scipy.sparse.linalg import aslinearoperator

__all__ = ['lsmr']


def _sym_ortho(a, b):
    """A stable Givens rotation: ``(c, s, r)`` with ``c a + s b = r``, ``-s a + c b = 0``.

    scipy's private ``scipy.sparse.linalg._isolve.lsqr._sym_ortho`` (scipy 1.16.2), copied
    statement for statement so the solve stays bit-identical to scipy's LSMR without importing a
    private module (S.-C. Choi's ``SymOrtho``, "Iterative Methods for Singular Linear Equations and
    Least-Squares Problems", dissertation, Stanford, 2006)."""
    if b == 0:
        return sign(a), 0, abs(a)
    elif a == 0:
        return 0, sign(b), abs(b)
    elif abs(b) > abs(a):
        tau = a / b
        s = sign(b) / sqrt(1 + tau * tau)
        c = s * tau
        r = b / s
    else:
        tau = b / a
        c = sign(a) / sqrt(1 + tau * tau)
        s = c * tau
        r = a / c
    return c, s, r


def lsmr(A, b, damp=0.0, atol=1e-6, btol=1e-6, conlim=1e8,
         maxiter=None, show=False, x0=None, history=None, callback=None, callback_every=1, watch=None):
    """``scipy.sparse.linalg.lsmr`` (same arguments, same return tuple), with the solver's state
    recorded in ``history`` (a :class:`~selfcal.core.solve_record.SolveHistory`, or None) at
    iteration 0 and after each iteration. See the module docstring.

    ``callback`` (or None) is called as ``callback(itn, x)`` after iteration ``itn`` whenever ``itn``
    is a multiple of ``callback_every`` and the solve goes on (never after the iteration it stops
    at), after ``history`` got that iteration's row. ``x`` is the solver's own buffer: read it, never
    change it or keep it past the call.

    ``watch`` (or None) is called as ``watch(0, x, 0, None)`` after the starting row of ``history``
    (its result unused) and as ``istop = watch(itn, x, istop, tests)`` after the row of each
    iteration, before ``callback``, with the solver's seven stopping conditions in ``istop`` order
    (see :func:`selfcal.core.lsqr_inplace.lsqr_inplace`); the solve stops when the returned code is
    not 0."""
    A = aslinearoperator(A)
    b = atleast_1d(b)
    if b.ndim > 1:
        b = b.squeeze()

    msg = ('The exact solution is x = 0, or x = x0, if x0 was given  ',
           'Ax - b is small enough, given atol, btol                  ',
           'The least-squares solution is good enough, given atol     ',
           'The estimate of cond(Abar) has exceeded conlim            ',
           'Ax - b is small enough for this machine                   ',
           'The least-squares solution is good enough for this machine',
           'Cond(Abar) seems to be too large for this machine         ',
           'The iteration limit has been reached                      ',
           'A stop rule of the caller ended the solve                 ')

    hdg1 = '   itn      x(1)       norm r    norm Ar'
    hdg2 = ' compatible   LS      norm A   cond A'
    pfreq = 20   # print frequency (for repeating the heading)
    pcount = 0   # print counter

    m, n = A.shape

    # stores the num of singular values
    minDim = min([m, n])

    if maxiter is None:
        maxiter = minDim

    if x0 is None:
        dtype = result_type(A, b, float)
    else:
        dtype = result_type(A, b, x0, float)

    if show:
        print(' ')
        print('LSMR            Least-squares solution of  Ax = b\n')
        print(f'The matrix A has {m} rows and {n} columns')
        print(f'damp = {damp:20.14e}\n')
        print(f'atol = {atol:8.2e}                 conlim = {conlim:8.2e}\n')
        print(f'btol = {btol:8.2e}             maxiter = {maxiter:8g}\n')

    u = b
    normb = norm(b)
    if x0 is None:
        x = zeros(n, dtype)
        beta = normb.copy()
    else:
        x = atleast_1d(x0.copy())
        u = u - A.matvec(x)
        beta = norm(u)

    if beta > 0:
        u = (1 / beta) * u
        v = A.rmatvec(u)
        alpha = norm(v)
    else:
        v = zeros(n, dtype)
        alpha = 0

    if alpha > 0:
        v = (1 / alpha) * v

    # Initialize variables for 1st iteration.

    itn = 0
    zetabar = alpha * beta
    alphabar = alpha
    rho = 1
    rhobar = 1
    cbar = 1
    sbar = 0

    h = v.copy()
    hbar = zeros(n, dtype)

    # Initialize variables for estimation of ||r||.

    betadd = beta
    betad = 0
    rhodold = 1
    tautildeold = 0
    thetatilde = 0
    zeta = 0
    d = 0

    # Initialize variables for estimation of ||A|| and cond(A)

    normA2 = alpha * alpha
    maxrbar = 0
    minrbar = 1e+100
    normA = sqrt(normA2)
    condA = 1
    normx = 0

    # Items for use in stopping rules, normb set earlier
    istop = 0
    ctol = 0
    if conlim > 0:
        ctol = 1 / conlim
    normr = beta

    # Reverse the order here from the original matlab code because
    # there was an error on return when arnorm==0
    normar = alpha * beta
    if history is not None:
        history.add(itn, normr, normr, normar, normA, condA, normx,
                    float(normr) / float(normb) if normb > 0 else 0.0, float('nan'))
    if watch is not None:
        watch(itn, x, istop, None)
    if normar == 0:
        if show:
            print(msg[0])
        return x, istop, itn, normr, normar, normA, condA, normx

    if normb == 0:
        x[()] = 0
        return x, istop, itn, normr, normar, normA, condA, normx

    if show:
        print(' ')
        print(hdg1, hdg2)
        test1 = 1
        test2 = alpha / beta
        str1 = f'{itn:6g} {x[0]:12.5e}'
        str2 = f' {normr:10.3e} {normar:10.3e}'
        str3 = f'  {test1:8.1e} {test2:8.1e}'
        print(''.join([str1, str2, str3]))

    # Main iteration loop.
    while itn < maxiter:
        itn = itn + 1

        # Perform the next step of the bidiagonalization to obtain the
        # next  beta, u, alpha, v.  These satisfy the relations
        #         beta*u  =  A@v   -  alpha*u,
        #        alpha*v  =  A'@u  -  beta*v.

        u *= -alpha
        u += A.matvec(v)
        beta = norm(u)

        if beta > 0:
            u *= (1 / beta)
            v *= -beta
            v += A.rmatvec(u)
            alpha = norm(v)
            if alpha > 0:
                v *= (1 / alpha)

        # At this point, beta = beta_{k+1}, alpha = alpha_{k+1}.

        # Construct rotation Qhat_{k,2k+1}.

        chat, shat, alphahat = _sym_ortho(alphabar, damp)

        # Use a plane rotation (Q_i) to turn B_i to R_i

        rhoold = rho
        c, s, rho = _sym_ortho(alphahat, beta)
        thetanew = s*alpha
        alphabar = c*alpha

        # Use a plane rotation (Qbar_i) to turn R_i^T to R_i^bar

        rhobarold = rhobar
        zetaold = zeta
        thetabar = sbar * rho
        rhotemp = cbar * rho
        cbar, sbar, rhobar = _sym_ortho(cbar * rho, thetanew)
        zeta = cbar * zetabar
        zetabar = - sbar * zetabar

        # Update h, h_hat, x.

        hbar *= - (thetabar * rho / (rhoold * rhobarold))
        hbar += h
        x += (zeta / (rho * rhobar)) * hbar
        h *= - (thetanew / rho)
        h += v

        # Estimate of ||r||.

        # Apply rotation Qhat_{k,2k+1}.
        betaacute = chat * betadd
        betacheck = -shat * betadd

        # Apply rotation Q_{k,k+1}.
        betahat = c * betaacute
        betadd = -s * betaacute

        # Apply rotation Qtilde_{k-1}.
        # betad = betad_{k-1} here.

        thetatildeold = thetatilde
        ctildeold, stildeold, rhotildeold = _sym_ortho(rhodold, thetabar)
        thetatilde = stildeold * rhobar
        rhodold = ctildeold * rhobar
        betad = - stildeold * betad + ctildeold * betahat

        # betad   = betad_k here.
        # rhodold = rhod_k  here.

        tautildeold = (zetaold - thetatildeold * tautildeold) / rhotildeold
        taud = (zeta - thetatilde * tautildeold) / rhodold
        d = d + betacheck * betacheck
        normr = sqrt(d + (betad - taud)**2 + betadd * betadd)

        # Estimate ||A||.
        normA2 = normA2 + beta * beta
        normA = sqrt(normA2)
        normA2 = normA2 + alpha * alpha

        # Estimate cond(A).
        maxrbar = max(maxrbar, rhobarold)
        if itn > 1:
            minrbar = min(minrbar, rhobarold)
        condA = max(maxrbar, rhotemp) / min(minrbar, rhotemp)

        # Test for convergence.

        # Compute norms for convergence testing.
        normar = abs(zetabar)
        normx = norm(x)

        # Now use these norms to estimate certain other quantities,
        # some of which will be small near a solution.

        test1 = normr / normb
        if (normA * normr) != 0:
            test2 = normar / (normA * normr)
        else:
            test2 = inf
        test3 = 1 / condA
        t1 = test1 / (1 + normA * normx / normb)
        rtol = btol + atol * normA * normx / normb

        # The following tests guard against extremely small values of
        # atol, btol or ctol.  (The user may have set any or all of
        # the parameters atol, btol, conlim  to 0.)
        # The effect is equivalent to the normAl tests using
        # atol = eps,  btol = eps,  conlim = 1/eps.

        if itn >= maxiter:
            istop = 7
        if 1 + test3 <= 1:
            istop = 6
        if 1 + test2 <= 1:
            istop = 5
        if 1 + t1 <= 1:
            istop = 4

        # Allow for tolerances set by the user.

        if test3 <= ctol:
            istop = 3
        if test2 <= atol:
            istop = 2
        if test1 <= rtol:
            istop = 1

        if history is not None:
            history.add(itn, normr, normr, normar, normA, condA, normx, test1, test2)
        if watch is not None:
            istop = watch(itn, x, istop, (test1 <= rtol, test2 <= atol, test3 <= ctol, 1 + t1 <= 1,
                                          1 + test2 <= 1, 1 + test3 <= 1, itn >= maxiter))

        # See if it is time to print something.

        if show:
            if (n <= 40) or (itn <= 10) or (itn >= maxiter - 10) or \
               (itn % 10 == 0) or (test3 <= 1.1 * ctol) or \
               (test2 <= 1.1 * atol) or (test1 <= 1.1 * rtol) or \
               (istop != 0):

                if pcount >= pfreq:
                    pcount = 0
                    print(' ')
                    print(hdg1, hdg2)
                pcount = pcount + 1
                str1 = f'{itn:6g} {x[0]:12.5e}'
                str2 = f' {normr:10.3e} {normar:10.3e}'
                str3 = f'  {test1:8.1e} {test2:8.1e}'
                str4 = f' {normA:8.1e} {condA:8.1e}'
                print(''.join([str1, str2, str3, str4]))

        if callback is not None and istop == 0 and itn % callback_every == 0:
            callback(itn, x)

        if istop > 0:
            break

    # Print the stopping condition.

    if show:
        print(' ')
        print('LSMR finished')
        print(msg[istop])
        print(f'istop ={istop:8g}    normr ={normr:8.1e}')
        print(f'    normA ={normA:8.1e}    normAr ={normar:8.1e}')
        print(f'itn   ={itn:8g}    condA ={condA:8.1e}')
        print(f'    normx ={normx:8.1e}')
        print(str1, str2)
        print(str3, str4)

    return x, istop, itn, normr, normar, normA, condA, normx
