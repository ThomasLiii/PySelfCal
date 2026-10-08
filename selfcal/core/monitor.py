"""Convergence monitors of a solve, and the opt-in rules that may stop it early.

The weakest directions of a self-calibration system are usually its largest scales: a smooth sky
pattern that the offsets can nearly absorb (a near-null mode). Such a mode barely changes
``|b - A x|`` or ``|A^T r|`` while it moves, so a residual or a gradient test can declare a solve
converged while its field-wide gradient and curvature are still far from their least-squares
values. This module records what a solve does, at a cadence the caller chooses, and stops a solve
early only on rules the caller asks for, recording which one fired and what it measured.

:class:`Watch` is called by the solver (``lsqr_inplace`` / the vendored ``lsmr``, argument
``watch``) after the history row of every iteration, ``itn = 0`` (the start) included. It runs the
**checks** at the iterations they are due (each costs what it computes; nothing in between):

* the **true residual** ``|b - A x|``: one product ``A x`` with the solver's operator, accumulated in
  float64 over pieces of the right-hand side the solve keeps for its final residual (a
  :class:`~selfcal.core.spill.ParkedVector`: in memory, or memory-mapped from scratch disk for a
  large system), with the ratio of the solver's estimate ``r1norm`` to it (a drift of the
  recurrences, such as norms accumulated in float32 cause, shows as a ratio away from 1);
* the **true gradient** ``|A^T (b - A x) - damp^2 x|`` (``|A^T r|`` without damping), one more
  product ``A^T r`` with the residual of the first, with the ratio of
  the estimate ``arnorm``; both in the solver's column-scaled unknowns, as the estimates are;
* the **large scale** (:class:`SmoothFit`): for each sky term, the least-squares fit of the 2-D
  monomials of degree at most ``degree`` (1, x, y, x^2, xy, y^2 for degree 2) in the
  coordinates of the reference grid scaled to [-1, 1], on every ``step``-th pixel of every
  ``step``-th row of the term's covered pixels (its active columns), read from the iterate through
  :class:`~selfcal.core.snapshots.Iterate` a row at a time (no copy of ``x``). Recorded: the
  coefficients (physical units; ``x`` is per half-field), the rms of the fitted surface without
  its mean (the term's large-scale amplitude), and its **relative change** since the previous
  check, ``rms(s_k - s_prev) / rms(s_k)`` over the sampled pixels, the means removed. Why
  polynomials and not a DCT: the covered region of a field is a footprint, not the grid's
  rectangle, and a DCT's modes are tied to the grid's edges and not orthogonal on a footprint;
  a polynomial least-squares fit on the covered pixels is meaningful on any footprint, its few
  coefficients are the field-wide gradient and curvature of each term, and the
  relative change of the surface does not depend on the basis chosen for it. The mean is left
  out of the change: a sky term's mean trades with the offsets' gauge, and in a bright field it
  would dominate the coefficients and hide the movement of the shape.

The **stop rules** (:class:`StopRules`, from a :class:`~selfcal.run.recipe.Stop`) decide each
iteration whether the solve ends there (``istop = 8``, :data:`RULE_ISTOP`):

* ``residual``: the relative decrease of ``|r|`` over the last ``window`` iterations is below
  ``below`` -- of the solver's estimate ``r1norm`` (every iteration, free), or with ``every = m``
  of the true residual (a check every ``m`` iterations; ``window`` a multiple of ``m``);
* ``gradient``: the largest ``|A^T r|`` over the last ``window`` iterations is at most ``below``
  times its value at the start of this solve -- the estimate ``arnorm`` every iteration, or the
  true gradient at checks every ``m`` iterations (CG-type gradients are not monotone, hence the
  window);
* ``large_scale``: the relative change of every sky term's smooth fit is below ``below`` at the
  last two checks (a check every ``every`` iterations, iteration 0 included: the first possible
  stop is at ``2 * every``);
* ``lsqr_tests``: the solver's own tolerance tests, each on its own: ``compatible`` (``istop`` 1,
  ``|r| / |b| <= btol + atol |A| |x| / |b|``), ``least_squares`` (2, ``|A^T r| / (|A| |r|) <=
  atol``), ``condition`` (3, the estimate of cond(A) above ``conlim``). With a Stop they stop the
  solve only as one of its rules (``Fit(tolerance=0)`` turns them off, and refuses them here);

combined over the rules given by ``combine`` (``"all"``: every rule holds at the same iteration,
the default; ``"any"``: one of them), and never before ``min_iterations``. A rule checked every
``m`` iterations keeps its verdict until its next check. The iteration limit stays the hard cap,
and the machine-precision stops (``istop`` 4-6: ``|r|``, the gradient or cond(A) at the limit of
the arithmetic, which also catch an exact solution) end the solve whatever the rules say. The stop
decision depends on the Stop's own settings only: a :class:`~selfcal.run.schedule.Monitor` adds
checks (and can be changed freely without changing the solution), it never feeds a rule.

Cadences: the checks run at the union of the monitor's (every ``Monitor.every`` iterations) and
each check-based rule's (its ``every``), iteration 0 included, and compute at each iteration only
what is due there. One large-scale fit per solve: a large-scale rule's ``degree`` and ``step``, or
else the monitor's (a monitor that asks for another degree or step than the rule is refused).

Cost and memory: no product per iteration; a true residual costs one ``A x`` and reads ``b`` once,
a true gradient one ``A^T r`` more (each with the transient vector of a solver iteration, freed
before the solve goes on: no copy of ``A``, no second ``x``); a large-scale check reads
``n_pix / step^2`` sampled pixels per sky term a row at a time plus a few tables per sampled row.
Everything runs in the solver's thread, on the operator's own thread pools: no process is started.

Records: :meth:`Watch.arrays` (the checks, ``check_*``, ``true_residual``, ``large_scale_*``, and
the rules' values per iteration, ``stop_*``) extends the solve's history file; :meth:`Watch.finish`
writes the stop record (``stop_rule``, ``stop_iteration``, ``stop_values``, ``stop_policy``) on the
:class:`~selfcal.core.solve_record.SolveRecord` (the cal's ``solve`` group, the action's record)
and says in the log, in words, which rule ended the solve and what it measured.
"""
from __future__ import annotations

import json
import logging
import math
from math import sqrt

import numpy as np
from scipy.sparse.linalg import aslinearoperator

from .snapshots import Iterate

logger = logging.getLogger(__name__)

__all__ = ['RULE_ISTOP', 'LSQR_TESTS', 'DEFAULT_DEGREE', 'DEFAULT_STEP', 'monomials', 'SmoothFit', 'StopRules',
           'Watch', 'resolve_large_scale']

#: The ``istop`` of a solve a stop rule ended.
RULE_ISTOP = 8

#: The solver's tolerance tests a :class:`~selfcal.run.recipe.Stop` can keep (``istop`` 1, 2, 3).
LSQR_TESTS = ('compatible', 'least_squares', 'condition')

#: The large-scale fit's default degree and sampling step.
DEFAULT_DEGREE = 2
DEFAULT_STEP = 4

# The pieces of b a true residual is accumulated over (float64 temporaries of ~400 MB).
_BLOCK = 50_000_000

_RULE_TITLES = {'residual': 'the residual rule', 'gradient': 'the gradient rule',
                'large_scale': 'the large-scale rule', 'lsqr_tests': "the solver's tests"}


def monomials(degree) -> list[tuple[int, int]]:
    """The exponents ``(a, b)`` of the monomials ``x^a y^b`` of total degree at most ``degree``, by
    degree, then by decreasing power of ``x``: ``1, x, y, x^2, xy, y^2`` for degree 2."""
    return [(a, d - a) for d in range(int(degree) + 1) for a in range(d, -1, -1)]


def _monomial_name(a, b):
    if a == b == 0:
        return '1'
    return ('' if a == 0 else 'x' if a == 1 else f'x^{a}') + ('' if b == 0 else 'y' if b == 1 else f'y^{b}')


def _scaled(n):
    """Pixel index -> [-1, 1] over ``n`` pixels (0 for a single pixel)."""
    half = (n - 1) / 2
    return (lambda i: (np.asarray(i, dtype=np.float64) - half) / half) if half > 0 else \
        (lambda i: np.zeros(np.shape(i), dtype=np.float64))


def _plain(v):
    """A value for JSON: floats that are not finite become None, numpy values Python ones."""
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, np.ndarray):
        return _plain(v.tolist())
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (float, np.floating)):
        v = float(v)
        return v if math.isfinite(v) else None
    return v


def resolve_large_scale(stop=None, monitor=None):
    """``(degree, step)`` of a solve's large-scale fit, or ``(None, None)`` without one: the
    large-scale rule's (``stop.large_scale``), else the monitor's (``monitor.large_scale``: ``True``
    for :data:`DEFAULT_DEGREE`, a degree, or ``False`` / None for none; ``monitor.step``, None for
    :data:`DEFAULT_STEP`). A monitor that names another degree or step than the rule's is refused
    (ValueError): the solve makes one fit."""
    rule = getattr(stop, 'large_scale', None) if stop is not None else None
    wanted = getattr(monitor, 'large_scale', None) if monitor is not None else None
    mstep = getattr(monitor, 'step', None) if monitor is not None else None
    if rule is not None:
        degree, step = int(rule.degree), int(rule.step)
        if wanted is not None and wanted is not True and wanted is not False and int(wanted) != degree:
            raise ValueError(f"Monitor(large_scale={wanted}) asks for a fit of degree {wanted}; the Fit's large-scale "
                             f"stop rule fits degree {degree}: give Monitor(large_scale=True) (the rule's fit)")
        if mstep is not None and int(mstep) != step:
            raise ValueError(f"Monitor(step={mstep}): the Fit's large-scale stop rule samples every {step}th pixel; "
                             f"leave the monitor's step unset (the rule's)")
        return degree, step
    if wanted is None or wanted is False:
        return None, None
    degree = DEFAULT_DEGREE if wanted is True else int(wanted)
    return degree, DEFAULT_STEP if mstep is None else int(mstep)


# --------------------------------------------------------------------------- the large-scale fit
class SmoothFit:
    """The large-scale content of the sky terms of an iterate: for each sky term, the least-squares
    fit of the monomials of degree at most ``degree`` (:func:`monomials`, coordinates of the
    reference grid scaled to [-1, 1]) to its covered pixels on every ``step``-th row and column.

    ``ref_shape``: the reference grid; ``names``: the sky terms, in block order (term ``j`` holds the
    full columns ``j * n_pix ... (j + 1) * n_pix - 1``, row-major). :meth:`bind` (once, before the
    solve) finds each sampled row's compact columns and the normal matrix of each term's samples;
    :meth:`fit` reads an :class:`~selfcal.core.snapshots.Iterate` a row at a time."""

    def __init__(self, ref_shape, names, degree=DEFAULT_DEGREE, step=DEFAULT_STEP):
        if int(degree) < 1:
            raise ValueError(f"large-scale fit of degree {degree}: at least 1 (the mean alone has no shape)")
        if int(step) < 1:
            raise ValueError(f"large-scale fit on every {step}th pixel: at least 1")
        self.ref_shape = (int(ref_shape[0]), int(ref_shape[1]))
        self.names = tuple(names)
        self.degree, self.step = int(degree), int(step)
        self.powers = monomials(self.degree)
        #: The monomials' names, in the coefficients' order.
        self.basis = tuple(_monomial_name(a, b) for a, b in self.powers)
        self._terms = None
        self._active = None

    def _rows_basis(self, xs, y):
        """The monomials at the pixels ``xs`` (scaled) of a row at scaled ``y``: (len(xs), P)."""
        out = np.empty((len(xs), len(self.powers)), dtype=np.float64)
        for k, (a, b) in enumerate(self.powers):
            out[:, k] = xs ** a * (y ** b)
        return out

    def bind(self, active=None):
        """Prepare the fit for a solve whose active columns are ``active`` (an
        :class:`~selfcal.core.snapshots.ActiveColumns`; None: every column is in the solver's
        vector, every pixel counts as covered). Sets ``samples``, the pixels each term's fit uses."""
        h, w = self.ref_shape
        n_pix = h * w
        s = self.step
        sx, sy = _scaled(w), _scaled(h)
        cols = np.arange(0, w, s)
        xs_all = sx(cols)
        rows = np.arange(0, h, s)
        terms = []
        for j in range(len(self.names)):
            base = j * n_pix
            c0s = np.zeros(len(rows), dtype=np.int64)
            normal = np.zeros((len(self.powers), len(self.powers)))
            gsum = np.zeros(len(self.powers))
            n = 0
            if active is not None:
                c, prev = active.compact_start(base), base
            for i, y in enumerate(rows):
                start = base + int(y) * w
                if active is not None:
                    c += int(np.count_nonzero(active.mask[prev:start]))
                    prev = start
                    c0s[i] = c
                    keep = active.mask[start:start + w][::s]
                    xs = xs_all[keep]
                else:
                    c0s[i] = start
                    xs = xs_all
                if len(xs) == 0:
                    continue
                g = self._rows_basis(xs, float(sy(y)))
                normal += g.T @ g
                gsum += g.sum(axis=0)
                n += len(xs)
            if n > 0:
                mean = gsum / n
                cov = normal / n - np.outer(mean, mean)
                inv = np.linalg.pinv(normal)
            else:
                cov = inv = None
            terms.append({'base': base, 'c0': c0s, 'n': n, 'cov': cov, 'inv': inv})
        self._terms, self._active, self._rows, self._xs = terms, active, rows, xs_all
        self._sy = sy
        #: The pixels each term's fit uses (its covered pixels on the sampled grid).
        self.samples = tuple(t['n'] for t in terms)

    def release(self):
        """Drop the reference to the solve's active columns (:meth:`fit` needs :meth:`bind` again)."""
        self._active = None
        self._terms = None if self._terms is None else [{k: v for k, v in t.items() if k != 'c0'}
                                                        for t in self._terms]

    def fit(self, iterate) -> np.ndarray:
        """The coefficients of each sky term's fit to ``iterate`` (an
        :class:`~selfcal.core.snapshots.Iterate`), ``(terms, monomials)`` float64 (NaN for a term
        with no covered sample)."""
        if self._terms is None or 'c0' not in self._terms[0]:
            raise ValueError("SmoothFit: bind() it to the solve first")
        h, w = self.ref_shape
        s = self.step
        out = np.full((len(self.names), len(self.powers)), np.nan)
        for j, t in enumerate(self._terms):
            if t['n'] == 0:
                continue
            rhs = np.zeros(len(self.powers))
            for i, y in enumerate(self._rows):
                start = t['base'] + int(y) * w
                row = iterate.read(start, start + w, compact_start=t['c0'][i] if self._active is not None else None)
                vals = row[::s]
                if self._active is not None:
                    keep = self._active.mask[start:start + w][::s]
                    vals, xs = vals[keep], self._xs[keep]
                else:
                    xs = self._xs
                if len(xs) == 0:
                    continue
                rhs += self._rows_basis(xs, float(self._sy(y))).T @ vals.astype(np.float64)
            out[j] = t['inv'] @ rhs
        return out

    def rms(self, coefficients) -> np.ndarray:
        """The rms over the sampled pixels of each term's fitted surface, its mean removed."""
        out = np.full(len(self.names), np.nan)
        for j, t in enumerate(self._terms):
            if t['cov'] is not None and np.all(np.isfinite(coefficients[j])):
                c = coefficients[j]
                out[j] = sqrt(max(float(c @ t['cov'] @ c), 0.0))
        return out

    def change(self, new, old) -> np.ndarray:
        """Each term's relative change from the fit ``old`` to ``new``: ``rms(s_new - s_old) /
        rms(s_new)`` over the sampled pixels, the means removed (0 when neither has a shape, inf when
        only ``old`` has one; NaN for a term with no covered sample)."""
        out = np.full(len(self.names), np.nan)
        size = self.rms(new)
        for j, t in enumerate(self._terms):
            if t['cov'] is None or not (np.all(np.isfinite(new[j])) and np.all(np.isfinite(old[j]))):
                continue
            d = new[j] - old[j]
            moved = sqrt(max(float(d @ t['cov'] @ d), 0.0))
            out[j] = moved / size[j] if size[j] > 0 else (0.0 if moved == 0 else math.inf)
        return out


# --------------------------------------------------------------------------- the stop rules
def _tests_of(value):
    """The solver tests a Stop keeps: a tuple of :data:`LSQR_TESTS` names."""
    if value is True:
        return LSQR_TESTS
    if not value:
        return ()
    names = (value,) if isinstance(value, str) else tuple(value)
    bad = [n for n in names if n not in LSQR_TESTS]
    if bad:
        raise ValueError(f"lsqr_tests: unknown tests {bad}; the solver's are {LSQR_TESTS}")
    return tuple(n for n in LSQR_TESTS if n in names)


class StopRules:
    """The stop rules of one solve (see the module docstring), from a
    :class:`~selfcal.run.recipe.Stop` (any object with its attributes): their state, their values at
    every iteration, and the decision."""

    def __init__(self, stop):
        self.policy = stop
        self.residual = getattr(stop, 'residual', None)
        self.gradient = getattr(stop, 'gradient', None)
        self.large_scale = getattr(stop, 'large_scale', None)
        self.lsqr = _tests_of(getattr(stop, 'lsqr_tests', False))
        self.min_iterations = int(getattr(stop, 'min_iterations', 0) or 0)
        self.combine = getattr(stop, 'combine', 'all')
        if self.combine not in ('all', 'any'):
            raise ValueError(f"combine={self.combine!r}: 'all' or 'any'")
        #: The rules given, in this order: residual, gradient, large_scale, lsqr_tests.
        self.given = tuple(n for n, v in (('residual', self.residual), ('gradient', self.gradient),
                                          ('large_scale', self.large_scale), ('lsqr_tests', self.lsqr)) if v)
        if not self.given:
            raise ValueError("a Stop needs at least one rule (residual, gradient, large_scale or lsqr_tests)")
        self.state = dict.fromkeys(self.given)            # each rule's latest evaluation
        self.series = {n: [] for n in self.given}         # each rule's value per iteration (NaN: not evaluated)
        self.holds = []                                   # the combination per iteration
        self._ls_change = {}                              # itn -> the large-scale rule's largest change
        self._ls_terms = {}                               # itn -> each term's change
        #: ``(iteration, [the rules that held])`` when a rule ended the solve, else None.
        self.fired = None

    # ---- what the rules need from the checks ---------------------------------------------------
    def check_every(self):
        """``{what: [cadence, ...]}``: the checks the rules need (``residual`` / ``gradient``: the true
        values, ``large_scale``: the fit)."""
        out = {'residual': [], 'gradient': [], 'large_scale': []}
        if self.residual is not None and self.residual.every:
            out['residual'].append(int(self.residual.every))
        if self.gradient is not None and self.gradient.every:
            out['gradient'].append(int(self.gradient.every))
        if self.large_scale is not None:
            out['large_scale'].append(int(self.large_scale.every))
        return out

    # ---- one iteration ---------------------------------------------------------------------------
    def skip(self):
        """Iteration 0 (the start): nothing is evaluated."""
        for n in self.given:
            self.series[n].append(np.nan)
        self.holds.append(False)

    def update(self, itn, tests, watch) -> bool:
        """Evaluate the rules after iteration ``itn`` (``tests``: the solver's seven conditions, in
        ``istop`` order); True when the combination holds and ``itn >= min_iterations``."""
        evaluated = {}
        if self.residual is not None:
            evaluated['residual'] = self._residual(itn, watch)
        if self.gradient is not None:
            evaluated['gradient'] = self._gradient(itn, watch)
        if self.large_scale is not None:
            evaluated['large_scale'] = self._large_scale(itn, watch)
        if self.lsqr:
            flags = {name: bool(tests[LSQR_TESTS.index(name)]) for name in self.lsqr}
            evaluated['lsqr_tests'] = {'holds': any(flags.values()), 'value': float(any(flags.values())),
                                       'tests': flags, 'at': itn}
        for n in self.given:
            st = evaluated.get(n)
            if st is not None:
                self.state[n] = st
            self.series[n].append(st['value'] if st is not None and st['value'] is not None else np.nan)
        verdicts = [self.state[n] is not None and bool(self.state[n]['holds']) for n in self.given]
        ok = (all(verdicts) if self.combine == 'all' else any(verdicts)) and itn >= self.min_iterations
        self.holds.append(bool(ok))
        if ok:
            self.fired = (int(itn), [n for n, v in zip(self.given, verdicts) if v])
        return ok

    def _residual(self, itn, watch):
        rule = self.residual
        w, every = int(rule.window), rule.every
        if itn < w:
            return None
        if every is None:
            old, new = watch.estimate(itn - w, 'r1norm'), watch.estimate(itn, 'r1norm')
            measured = 'the estimate r1norm'
        else:
            if itn % int(every):
                return None
            old, new = watch.check_value(itn - w, 'true_residual'), watch.check_value(itn, 'true_residual')
            measured = f'the true |b - A x|, every {int(every)} iterations'
        decrease = (old - new) / old if old > 0 else 0.0
        return {'holds': bool(decrease < rule.below), 'value': float(decrease), 'below': float(rule.below),
                'window': w, 'measured': measured, 'at': itn, 'from': float(old), 'to': float(new)}

    def _gradient(self, itn, watch):
        rule = self.gradient
        w, every = int(rule.window), rule.every
        if itn < w:
            return None
        if every is None:
            largest = max(watch.estimate(k, 'arnorm') for k in range(itn - w + 1, itn + 1))
            start = watch.estimate(0, 'arnorm')
            measured = 'the estimate arnorm'
        else:
            e = int(every)
            if itn % e:
                return None
            largest = max(watch.check_value(k, 'true_gradient') for k in range(itn - w + e, itn + 1, e))
            start = watch.check_value(0, 'true_gradient')
            measured = f'the true |A^T r|, every {e} iterations'
        ratio = largest / start if start > 0 else 0.0
        return {'holds': bool(ratio <= rule.below), 'value': float(ratio), 'below': float(rule.below), 'window': w,
                'measured': measured, 'at': itn, 'largest': float(largest), 'start': float(start)}

    def _large_scale(self, itn, watch):
        rule = self.large_scale
        e = int(rule.every)
        if itn % e or itn < e:
            return None
        changes = watch.smooth.change(watch.check_value(itn, 'coefficients'),
                                      watch.check_value(itn - e, 'coefficients'))
        largest = float(np.nanmax(changes)) if np.any(np.isfinite(changes)) else math.nan
        self._ls_change[itn] = largest
        self._ls_terms[itn] = changes
        if itn < 2 * e:
            return None
        previous = self._ls_change[itn - e]
        value = max(largest, previous) if math.isfinite(largest) and math.isfinite(previous) else math.nan
        return {'holds': bool(math.isfinite(value) and value < rule.below), 'value': value,
                'below': float(rule.below), 'every': e, 'at': [itn - e, itn], 'degree': int(rule.degree),
                'step': int(rule.step), 'changes': [previous, largest],
                'terms': dict(zip(watch.smooth.names, (float(c) for c in changes)))}

    # ---- the record --------------------------------------------------------------------------------
    def values(self, itn) -> dict:
        """The rules' state at iteration ``itn`` (their latest evaluations), for the record."""
        return _plain({'iteration': int(itn), 'combine': self.combine, 'min_iterations': self.min_iterations,
                       'rules': {n: self.state[n] for n in self.given}})

    def arrays(self) -> dict:
        """The rules' values per iteration (``stop_<rule>``, NaN where not evaluated) and the
        combination (``stop_holds``), one row per iteration."""
        out = {f'stop_{n}': np.asarray(self.series[n], dtype=np.float64) for n in self.given}
        out['stop_holds'] = np.asarray(self.holds, dtype=bool)
        return out

    def policy_json(self) -> str:
        p = self.policy
        d = p.to_dict() if hasattr(p, 'to_dict') else {'repr': repr(p)}
        return json.dumps(_plain(d), sort_keys=True)

    def describe(self, n, st) -> str:
        """Rule ``n``'s state ``st`` in words."""
        if st is None:
            if n in ('residual', 'gradient'):
                return f"{_RULE_TITLES[n]} not evaluated yet (it needs {int(getattr(self, n).window)} iterations)"
            if n == 'large_scale':
                return f"{_RULE_TITLES[n]} not evaluated yet (it needs checks at 0, {int(self.large_scale.every)} " \
                       f"and {2 * int(self.large_scale.every)})"
            return f"{_RULE_TITLES[n]} not evaluated"
        verdict = 'holds' if st['holds'] else 'does not hold'
        if n == 'residual':
            return (f"{_RULE_TITLES[n]} {verdict}: {st['measured']} fell {100 * st['value']:.4g} % over the last "
                    f"{st['window']} iterations to iteration {st['at']} ({st['from']:.6e} -> {st['to']:.6e}; the rule: "
                    f"less than {100 * st['below']:.4g} %)")
        if n == 'gradient':
            return (f"{_RULE_TITLES[n]} {verdict}: the largest |A^T r| ({st['measured']}) over the last {st['window']} "
                    f"iterations to iteration {st['at']}, {st['largest']:.6e}, is {st['value']:.4g} of its value at the "
                    f"start, {st['start']:.6e} (the rule: at most {st['below']:g})")
        if n == 'large_scale':
            terms = ', '.join(f"{k} {v:.4g}" for k, v in st['terms'].items())
            return (f"{_RULE_TITLES[n]} {verdict}: the smooth fit (degree {st['degree']}, every {st['step']}th pixel) of "
                    f"every sky term changed by at most {st['value']:.4g} of its size at the last two checks, "
                    f"iterations {st['at'][0]} and {st['at'][1]} (changes {st['changes'][0]:.4g}, {st['changes'][1]:.4g}; "
                    f"at {st['at'][1]}: {terms}; the rule: less than {st['below']:g})")
        held = [k for k, v in st['tests'].items() if v]
        return (f"{_RULE_TITLES[n]} {verdict} at iteration {st['at']}"
                + (f": {', '.join(f'{k} (istop {LSQR_TESTS.index(k) + 1})' for k in held)}" if held else
                   f" (kept: {', '.join(st['tests'])})"))


# --------------------------------------------------------------------------- the watch
class Watch:
    """The monitors and stop rules of one solve: what the solver calls after the history row of
    every iteration (see the module docstring).

    ``stop``: a :class:`~selfcal.run.recipe.Stop` (or None); ``monitor``: a
    :class:`~selfcal.run.schedule.Monitor` (or None); ``ref_shape`` and ``sky_names``: the sky
    terms' layout (None: one term, ``sky``); ``history``: the solve's
    :class:`~selfcal.core.solve_record.SolveHistory` (the estimates); ``damp``: the solver's
    damping. :meth:`bind` gives it the operator, the right-hand side and the column scaling before
    the solve; the solver then calls it as ``watch(itn, x, istop, tests)``."""

    def __init__(self, *, stop=None, monitor=None, ref_shape=(1, 1), sky_names=None, history=None, damp=0.0):
        self.rules = StopRules(stop) if stop is not None else None
        self.monitor = monitor
        if monitor is not None and int(monitor.every) < 1:
            raise ValueError(f"Monitor(every={monitor.every}): at least 1 iteration")
        degree, step = resolve_large_scale(stop, monitor)
        self.smooth = SmoothFit(ref_shape, tuple(sky_names) if sky_names else ('sky',), degree, step) \
            if degree is not None else None
        self.history = history
        self.damp = float(damp)
        cad = self.rules.check_every() if self.rules is not None else {'residual': [], 'gradient': [],
                                                                       'large_scale': []}
        if monitor is not None:
            m = int(monitor.every)
            if monitor.residual:
                cad['residual'].append(m)
            if monitor.gradient:
                cad['gradient'].append(m)
            if self.smooth is not None and monitor.large_scale is not False and monitor.large_scale is not None:
                cad['large_scale'].append(m)
        self._cadence = cad
        self.checks = []              # one dict per check, in iteration order
        self._by_itn = {}
        self._op = self._b = None
        self.stopped = None           # the iteration a rule ended the solve at

    # ---- setting up --------------------------------------------------------------------------------
    def bind(self, operator, b, *, scale=None, active=None, num_cols=None):
        """Give the watch the solve's operator (the column-scaled ``A`` the solver runs on),
        ``b`` (a function returning the right-hand side the solver started from: the parked copy),
        the column scaling ``scale`` (``x * scale`` is physical; None: unscaled), the active columns
        ``active`` (an :class:`~selfcal.core.snapshots.ActiveColumns`, None: every column) and the
        full layout's ``num_cols``; prepares the large-scale fit."""
        self._op = aslinearoperator(operator)
        self._b = b
        self._scale, self._active, self._num_cols = scale, active, num_cols
        if self.smooth is not None:
            self.smooth.bind(active)

    def _due(self, itn):
        return {k: any(itn % e == 0 for e in v) for k, v in self._cadence.items()}

    # ---- the solver's call ---------------------------------------------------------------------------
    def __call__(self, itn, x, istop, tests):
        """After iteration ``itn`` (0: the start, ``tests`` None): run the checks due, evaluate the
        rules; returns the ``istop`` the solve goes on with (0: go on)."""
        due = self._due(itn)
        if any(due.values()):
            self._check(itn, x, due)
        if self.rules is None:
            return istop
        if tests is None:
            self.rules.skip()
            return istop
        if self.rules.update(itn, tests, self):
            self.stopped = int(itn)
            return RULE_ISTOP
        for code in (4, 5, 6, 7):          # machine precision, then the iteration limit (the solver's order)
            if tests[code - 1]:
                return code
        return 0

    def _check(self, itn, x, due):
        if self._op is None:
            raise ValueError("Watch: bind() it to the solve first")
        entry = {'itn': int(itn), 'true_residual': math.nan, 'true_gradient': math.nan, 'coefficients': None,
                 'rms': None, 'change': None}
        if due['residual'] or due['gradient']:
            entry['true_residual'], entry['true_gradient'] = self._residual(x, gradient=due['gradient'])
        if due['large_scale'] and self.smooth is not None:
            it = Iterate(itn, x, scale=self._scale, active=self._active, num_cols=self._num_cols)
            c = self.smooth.fit(it)
            entry['coefficients'] = c
            entry['rms'] = self.smooth.rms(c)
            prev = next((e for e in reversed(self.checks) if e['coefficients'] is not None), None)
            entry['change'] = (self.smooth.change(c, prev['coefficients']) if prev is not None
                               else np.full(len(self.smooth.names), np.nan))
        self.checks.append(entry)
        self._by_itn[int(itn)] = entry
        logger.info(self._describe_check(entry))

    def _residual(self, x, gradient):
        """``(|b - A x|, |A^T (b - A x) - damp^2 x| or NaN)``, accumulated in float64; one product
        ``A x`` (and ``A^T r`` with ``gradient``), the residual written over the first product's
        output for the second."""
        ax = self._op.matvec(x)
        b = self._b()
        rr = 0.0
        for s in range(0, ax.shape[0], _BLOCK):
            e = min(ax.shape[0], s + _BLOCK)
            piece = np.array(b[s:e], dtype=np.float64)        # a copy: b is read, never written
            piece -= ax[s:e]
            rr += float(np.dot(piece, piece))
            if gradient:
                ax[s:e] = piece
            del piece
        del b
        if not gradient:
            return sqrt(rr), math.nan
        g = self._op.rmatvec(ax)
        del ax
        gg = 0.0
        d2 = self.damp * self.damp
        for s in range(0, g.shape[0], _BLOCK):
            e = min(g.shape[0], s + _BLOCK)
            piece = np.array(g[s:e], dtype=np.float64)
            if d2 > 0:
                piece -= d2 * np.asarray(x[s:e], dtype=np.float64)
            gg += float(np.dot(piece, piece))
            del piece
        del g
        return sqrt(rr), sqrt(gg)

    # ---- what the rules read ---------------------------------------------------------------------------
    def estimate(self, itn, name) -> float:
        """The solver's estimate ``name`` (a history column) after iteration ``itn``."""
        return float(self.history.value(itn, name))

    def check_value(self, itn, name):
        """The value ``name`` of the check at iteration ``itn`` (``true_residual``, ``true_gradient``,
        ``coefficients``)."""
        return self._by_itn[int(itn)][name]

    # ---- the end ---------------------------------------------------------------------------------------
    def finish(self, record):
        """Enter the stop record on ``record`` (a :class:`~selfcal.core.solve_record.SolveRecord`; a
        solve with stop rules: ``stop_rule``, ``stop_iteration``, ``stop_values``, ``stop_policy``),
        keep this watch on it (its history file gets :meth:`arrays`), and say in the log what
        ended the solve. Drops the watch's references to the solve (the operator, and with it
        ``A``; the right-hand side; the column scaling; the active columns), so that keeping the
        record keeps none of them alive."""
        self._op = self._b = self._scale = self._active = None
        if self.smooth is not None:
            self.smooth.release()
        record.watch = self
        if self.rules is None:
            return
        r = self.rules
        if r.fired is not None:
            itn, held = r.fired
            record.stop_rule = ', '.join(held)
        else:
            itn = int(record.iterations)
            record.stop_rule = 'none'
        record.stop_iteration = int(itn)
        record.stop_values = json.dumps(r.values(itn), sort_keys=True)
        record.stop_policy = r.policy_json()
        logger.info(self.describe_stop(record))

    def describe_stop(self, record) -> str:
        """What ended the solve, in words, with each rule's state there."""
        r = self.rules
        joined = ' and ' if r.combine == 'all' else ' or '
        policy = (f"stop rules ({joined.strip()} of: {', '.join(_RULE_TITLES[n][4:] for n in r.given)}"
                  + (f"; not before iteration {r.min_iterations}" if r.min_iterations else '') + ')')
        if r.fired is not None:
            itn, held = r.fired
            head = (f"Stopped by {' and '.join(_RULE_TITLES[n] for n in held)} at iteration {itn} "
                    f"({policy}; istop {RULE_ISTOP}).")
        else:
            itn = int(record.iterations)
            head = (f"No stop rule ended the solve ({policy}); it stopped after {itn} iterations with istop "
                    f"{record.istop}: {record.stop}. The rules then:")
        return head + ''.join(f"\n  {r.describe(n, r.state[n])}." for n in r.given)

    def arrays(self, before=0) -> dict:
        """The history file's arrays of this watch: of the checks (``check_itn``, ``check_iteration``
        cumulative from ``before`` (-1 when unknown), ``true_residual`` and ``residual_ratio`` =
        r1norm / truth, ``true_gradient`` and ``gradient_ratio`` = arnorm / truth, the
        ``large_scale_*`` fit), each only when computed at some check (NaN at the others), and the
        rules' values per iteration (:meth:`StopRules.arrays`)."""
        out = {}
        if self.checks:
            itn = np.asarray([c['itn'] for c in self.checks], dtype=np.int64)
            out['check_itn'] = itn
            out['check_iteration'] = itn + int(before) if before is not None and before >= 0 else np.full_like(itn, -1)
            for key, est, ratio in (('true_residual', 'r1norm', 'residual_ratio'),
                                    ('true_gradient', 'arnorm', 'gradient_ratio')):
                vals = np.asarray([c[key] for c in self.checks], dtype=np.float64)
                if np.all(np.isnan(vals)):
                    continue
                out[key] = vals
                if self.history is not None:
                    e = np.asarray([self.estimate(k, est) for k in itn], dtype=np.float64)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        out[ratio] = e / vals
            if any(c['coefficients'] is not None for c in self.checks):
                sm = self.smooth
                shape = (len(self.checks), len(sm.names))
                coef = np.full(shape + (len(sm.basis),), np.nan)
                rms, change = np.full(shape, np.nan), np.full(shape, np.nan)
                for k, c in enumerate(self.checks):
                    if c['coefficients'] is not None:
                        coef[k], rms[k], change[k] = c['coefficients'], c['rms'], c['change']
                out.update(large_scale_terms=np.asarray(sm.names), large_scale_basis=np.asarray(sm.basis),
                           large_scale_degree=np.asarray(sm.degree), large_scale_step=np.asarray(sm.step),
                           large_scale_samples=np.asarray(sm.samples, dtype=np.int64),
                           large_scale_coefficients=coef, large_scale_rms=rms, large_scale_change=change)
        if self.rules is not None:
            out.update(self.rules.arrays())
        return out

    def summary(self) -> dict | None:
        """The monitor's settings and its last check, for the action's record (None without a
        monitor)."""
        m = self.monitor
        if m is None:
            return None
        out = {'every': int(m.every), 'residual': bool(m.residual), 'gradient': bool(m.gradient),
               'large_scale': None if self.smooth is None else self.smooth.degree,
               'step': None if self.smooth is None else self.smooth.step, 'checks': len(self.checks)}
        if self.checks:
            last = self.checks[-1]
            out['last'] = {'itn': last['itn'], 'true_residual': last['true_residual'],
                           'true_gradient': last['true_gradient'],
                           'large_scale_change': (None if last['change'] is None else
                                                  dict(zip(self.smooth.names, last['change'])))}
        return _plain(out)

    def _describe_check(self, e) -> str:
        parts = []
        itn = e['itn']
        if not math.isnan(e['true_residual']):
            est = self.estimate(itn, 'r1norm') if self.history is not None else math.nan
            parts.append(f"|b - A x| = {e['true_residual']:.6e} (r1norm / true {est / e['true_residual']:.6f})"
                         if e['true_residual'] > 0 else f"|b - A x| = {e['true_residual']:.6e}")
        if not math.isnan(e['true_gradient']):
            est = self.estimate(itn, 'arnorm') if self.history is not None else math.nan
            parts.append(f"|A^T r| = {e['true_gradient']:.6e} (arnorm / true {est / e['true_gradient']:.6f})"
                         if e['true_gradient'] > 0 else f"|A^T r| = {e['true_gradient']:.6e}")
        if e['coefficients'] is not None:
            sm = self.smooth
            terms = '; '.join(f"{name} rms {rms:.4e}, change {ch:.4g}"
                              for name, rms, ch in zip(sm.names, e['rms'], e['change']))
            parts.append(f"large scale (degree {sm.degree}): {terms}")
        return f"monitor at iteration {itn}: " + ', '.join(parts)
