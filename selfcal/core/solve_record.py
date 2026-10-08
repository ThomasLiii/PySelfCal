"""The record of one iterative solve: how it stopped, its final state, and its state at every iteration.

:func:`~selfcal.core.solve.apply_lsqr` fills a :class:`SolveRecord` for every solve, LSQR or
LSMR: the method, the iterations run, ``istop`` and its meaning, the solver's final estimates
(``r1norm``, ``r2norm``, ``arnorm``, ``anorm``, ``acond``, ``xnorm``), the tolerances and
``conlim`` it ran with, its wall time, and the true residual ``|b - A x|``, computed once at the end
with one product (the solver's ``r1norm`` is a recurrence estimate; a drift between the two is how
the float32-norm failure of 2026-10 showed up). Its :class:`SolveHistory` holds the solver's
estimates at each iteration, collected from the scalars the solver computes anyway (no extra
products, a few floats per iteration).

``Calibrator.save_calibration`` writes the record as the attributes of the cal file's ``solve``
group (:meth:`SolveRecord.write`, read back by :func:`read` and
:attr:`selfcal.io.calfile.CalFile.solve`), all but the wall time (:data:`VOLATILE`): every value
there is a function of the solve, so a cal file stays byte-identical from run to run. The run
engine also enters the whole record in the action's record (``solves``) and saves the history as
``<field>/records/<cal stem>_history.npz`` (:meth:`SolveRecord.save_history`), and records in it
the identity of the system solved and, for a solve continued from another's cal, its source
(:meth:`SolveRecord.set_system`, :meth:`SolveRecord.continues`; :mod:`selfcal.core.warm_start`).

A solve with monitors or stop rules (:mod:`selfcal.core.monitor`) keeps its
:class:`~selfcal.core.monitor.Watch` on the record: the history file gets the checks' arrays and
the rules' values per iteration, and a solve with stop rules records how they ended it
(``stop_rule``, ``stop_iteration``, ``stop_values``, ``stop_policy``), in the cal and the action's
record. Monitors alone add nothing to the cal: a monitored solve writes the same cal file.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np

__all__ = ['STOP_REASONS', 'HISTORY_COLUMNS', 'VOLATILE', 'OPTIONAL', 'JSON_VALUES', 'SolveHistory', 'SolveRecord',
           'read', 'history_path']

#: The version of the record's layout (the ``version`` attribute of a cal's ``solve`` group).
VERSION = 1

#: The values of a record that change from one run of the same solve to the next (kept out of the cal
#: file, which is byte-reproducible; the action's record holds them).
VOLATILE = ('wall_s',)

#: The values of a record present only when set (a saved history, the system solved, a start, the
#: stop rules).
OPTIONAL = ('history_file', 'system', 'system_identity', 'start_from', 'start_identity', 'stop_rule',
            'stop_iteration', 'stop_values', 'stop_policy')

#: The values held as JSON text in the cal's attributes, decoded in the action's record.
JSON_VALUES = ('stop_values', 'stop_policy')

#: What each ``istop`` of LSQR and LSMR means (the two solvers share the codes).
STOP_REASONS = {
    0: 'the starting vector solves the system (zero residual or zero gradient)',
    1: '|b - A x| is small enough, given atol and btol (a compatible system)',
    2: 'the least-squares solution is good enough, given atol',
    3: 'the estimate of cond(A) exceeded conlim',
    4: '|b - A x| is small enough for this machine',
    5: 'the least-squares solution is good enough for this machine',
    6: 'cond(A) seems to be too large for this machine',
    7: 'the iteration limit was reached',
    8: 'a stop rule ended the solve (Fit(stop=...); stop_rule says which)',
}

#: The columns of a :class:`SolveHistory`, in order.
HISTORY_COLUMNS = ('itn', 'r1norm', 'r2norm', 'arnorm', 'anorm', 'acond', 'xnorm', 'test1', 'test2', 'elapsed_s')


class SolveHistory:
    """The solver's estimates at each iteration, row 0 the starting state.

    The solver calls :meth:`add` once before its first iteration (``itn = 0``: the residual of the
    starting vector) and once at the end of each iteration, with the scalars it has just computed:
    ``r1norm`` (``|b - A x|``), ``r2norm`` (with the damping term), ``arnorm`` (``|A^T r|``),
    ``anorm`` and ``acond`` (the running estimates of ``|A|`` and ``cond(A)``), ``xnorm``, and the
    stopping tests' ratios ``test1 = |r| / |b|`` and ``test2 = |A^T r| / (|A| |r|)`` (NaN at
    row 0). ``elapsed_s`` is the time since the history was made, just before the solver started.
    """

    def __init__(self):
        self._t0 = time.perf_counter()
        self._rows = []

    def add(self, itn, r1norm, r2norm, arnorm, anorm, acond, xnorm, test1, test2):
        """Append the state after iteration ``itn``."""
        self._rows.append((int(itn), float(r1norm), float(r2norm), float(arnorm), float(anorm), float(acond),
                           float(xnorm), float(test1), float(test2), time.perf_counter() - self._t0))

    def __len__(self):
        return len(self._rows)

    def arrays(self) -> dict[str, np.ndarray]:
        """The history as one array per column (:data:`HISTORY_COLUMNS`): ``itn`` int64, the rest
        float64."""
        cols = list(zip(*self._rows)) if self._rows else [()] * len(HISTORY_COLUMNS)
        return {name: np.asarray(col, dtype=np.int64 if name == 'itn' else np.float64)
                for name, col in zip(HISTORY_COLUMNS, cols)}

    def last(self) -> dict | None:
        """The last row as a dict, or None for an empty history."""
        return dict(zip(HISTORY_COLUMNS, self._rows[-1])) if self._rows else None

    def value(self, itn, name) -> float:
        """The column ``name`` of the row of iteration ``itn`` (row ``itn``: the solver adds one row per
        iteration from 0)."""
        row = self._rows[int(itn)]
        if row[0] != int(itn):
            raise ValueError(f"the history's row {itn} is iteration {row[0]}")
        return row[HISTORY_COLUMNS.index(name)]


@dataclass
class SolveRecord:
    """How one solve ran and ended (see the module docstring).

    ``iterations`` is the number this solve ran; ``iterations_total`` the cumulative count of the
    solution it leaves (the same until a solve continues another's). ``istop`` is the solver's stop
    code and ``stop`` its meaning (:data:`STOP_REASONS`). The norms are the solver's final
    estimates, in its own (column-scaled) unknowns; ``true_residual`` is ``|b - A x|`` and ``bnorm``
    is ``|b|``, both accumulated in float64 from one product at the end. LSMR reports one residual
    norm, its ``normr``, as both ``r1norm`` and ``r2norm``. ``rows`` and ``columns`` are the system's
    shape as the solver saw it (the active columns), ``wall_s`` the solver's wall time (the
    final product excluded), ``history`` the :class:`SolveHistory` (not an attribute; saved by
    :meth:`save_history`) and ``history_file`` where it was saved.

    ``system`` and ``system_identity``: the fingerprint of the system solved and the JSON it is the
    fingerprint of (:class:`~selfcal.core.warm_start.System`: the frames in order, the model's
    unknowns, the grid, the job), so a later solve can check a start from this one
    (:meth:`set_system`). A continuation (:meth:`continues`) records its source, ``start_from`` (its
    path) and ``start_identity`` (its product fingerprint, ``fingerprint:<sha256>``, or the hash of
    its bytes, ``sha256:<sha256>``), and ``iterations_total`` becomes the source's total plus this
    solve's iterations (-1 when the source has no record of its iterations).

    A solve with stop rules (``Fit(stop=...)``, :mod:`selfcal.core.monitor`): ``stop_rule``, the
    rules that ended it (``"gradient"``, ``"residual, large_scale"``; ``"none"`` when it ended
    otherwise), ``stop_iteration``, the iteration (of this solve) the rules were judged at last
    (where they ended it, or its last), ``stop_values``, the JSON of every rule's state there (its
    measured value, its threshold, where it was measured), and ``stop_policy``, the JSON of the
    rules. ``watch``: the solve's :class:`~selfcal.core.monitor.Watch` (not an attribute; its arrays
    go to the history file), None for a solve without monitors or rules."""
    method: str
    iterations: int
    iterations_total: int
    iteration_limit: int
    istop: int
    stop: str
    r1norm: float
    r2norm: float
    arnorm: float
    anorm: float
    acond: float
    xnorm: float
    true_residual: float
    bnorm: float
    atol: float
    btol: float
    conlim: float
    damp: float
    rows: int
    columns: int
    wall_s: float
    history: SolveHistory | None = field(default=None, repr=False)
    history_file: str | None = None
    system: str | None = None
    system_identity: str | None = None
    start_from: str | None = None
    start_identity: str | None = None
    stop_rule: str | None = None
    stop_iteration: int | None = None
    stop_values: str | None = None
    stop_policy: str | None = None
    watch: object = field(default=None, repr=False)

    @classmethod
    def from_result(cls, method, result, *, history, true_residual, bnorm, atol, btol, conlim, damp,
                    iteration_limit, shape, wall_s) -> SolveRecord:
        """The record of a solve from the solver's return tuple (scipy's layout: LSQR's ten values,
        or LSMR's eight)."""
        if method == 'lsmr':
            _, istop, itn, normr, normar, norma, conda, normx = result[:8]
            r1, r2, ar, an, ac, xn = normr, normr, normar, norma, conda, normx
        else:
            _, istop, itn, r1, r2, an, ac, ar, xn = result[:9]
        return cls(method=str(method), iterations=int(itn), iterations_total=int(itn),
                   iteration_limit=int(iteration_limit), istop=int(istop), stop=STOP_REASONS.get(int(istop), '?'),
                   r1norm=float(r1), r2norm=float(r2), arnorm=float(ar), anorm=float(an), acond=float(ac),
                   xnorm=float(xn), true_residual=float(true_residual), bnorm=float(bnorm), atol=float(atol),
                   btol=float(btol), conlim=float(conlim), damp=float(damp), rows=int(shape[0]),
                   columns=int(shape[1]), wall_s=float(wall_s), history=history)

    def attrs(self) -> dict:
        """The record as plain values (what the cal's ``solve`` group and the action's record hold),
        with ``version``; ``history_file`` only when the history was saved, the system and the start
        only when set."""
        out = {'version': VERSION}
        for name in self.__dataclass_fields__:
            if name in ('history', 'watch'):
                continue
            value = getattr(self, name)
            if name in OPTIONAL and value is None:
                continue
            out[name] = value
        return out

    def entry(self) -> dict:
        """The record as the action's record holds it: :meth:`attrs` with the JSON values decoded
        (:data:`JSON_VALUES`) and, for a monitored solve, ``monitor`` (its settings and last check,
        :meth:`~selfcal.core.monitor.Watch.summary`)."""
        import json
        out = self.attrs()
        for k in JSON_VALUES:
            if k in out:
                out[k] = json.loads(out[k])
        summary = self.watch.summary() if self.watch is not None else None
        if summary is not None:
            out['monitor'] = summary
        return out

    def set_system(self, system):
        """Record the identity of the system solved (a :class:`~selfcal.core.warm_start.System`)."""
        self.system_identity = system.identity_json()
        self.system = system.fingerprint()

    def continues(self, start):
        """Record that the solve started from ``start`` (a :class:`~selfcal.core.warm_start.WarmStart`):
        its path and identity, and the cumulative iteration count (-1 when the source's is unknown)."""
        self.start_from = start.path
        self.start_identity = start.identity
        self.iterations_total = (-1 if start.iterations_total is None
                                 else int(start.iterations_total) + int(self.iterations))

    def write(self, group):
        """Write the record as the attributes of the HDF5 group ``group`` (a cal file's ``solve``), all
        but the :data:`VOLATILE` values."""
        for k, v in self.attrs().items():
            if k not in VOLATILE:
                group.attrs[k] = v

    def save_history(self, path) -> str:
        """Save the history as an NPZ file at ``path`` (written whole or not at all): one array per
        column of :data:`HISTORY_COLUMNS`, plus ``method``, plus, for a solve with monitors or stop
        rules, the arrays of its :class:`~selfcal.core.monitor.Watch` (the checks numbered from the
        start's total too, ``check_iteration``). Sets :attr:`history_file`; returns the path."""
        from ..io.atomic import atomic_path
        path = os.path.abspath(os.fspath(path))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        arrays = (self.history or SolveHistory()).arrays()
        if self.watch is not None:
            before = (int(self.iterations_total) - int(self.iterations)) if int(self.iterations_total) >= 0 else None
            arrays.update(self.watch.arrays(before=before))
        with atomic_path(path) as tmp:
            with open(tmp, 'wb') as f:
                np.savez(f, method=np.array(self.method), **arrays)
        self.history_file = path
        return path


def read(f) -> dict | None:
    """The solve record of an open cal file ``f`` (an ``h5py.File``) as a dict, or None when it has
    none (a cal made before records existed, a stitched cal, an N-pass pass product)."""
    if 'solve' not in f:
        return None
    out = {}
    for k, v in f['solve'].attrs.items():
        if isinstance(v, bytes):
            v = v.decode()
        elif isinstance(v, np.generic):
            v = v.item()
        out[k] = v
    return out


def history_path(records_dir, cal_path) -> str:
    """Where the history of the solve that made ``cal_path`` goes: ``<records_dir>/<cal stem>_history.npz``."""
    stem = os.path.splitext(os.path.basename(os.fspath(cal_path)))[0]
    return os.path.join(os.fspath(records_dir), f'{stem}_history.npz')
