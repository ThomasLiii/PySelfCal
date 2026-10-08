"""Snapshots of a solve: its solution every ``k`` iterations, each written as a complete cal file.

``field.calibrate(recipe, snapshots=sc.Snapshots(every=k))`` hands the solver (LSQR or LSMR) a
callback that :func:`~selfcal.core.solve.apply_lsqr` calls after every ``k``-th iteration while the
solve goes on (not after the iteration it stops at: the cal is written then). The callback sees the
solver's own iterate, compact (the active columns only) and column-scaled; :class:`Iterate` reads it
in the physical units and full column layout of the solution, block by block, exactly as the end of
the solve converts it (``x * M``, the active columns placed by ``active_mask``, every other column
0). So a snapshot at iteration ``i`` holds the solution a solve of exactly ``i`` iterations would end
with (``Fit(iterations=i, tolerance=0)``), bit for bit.

:class:`SnapshotWriter` writes each snapshot as ``<cal dir>/snapshots/<cal stem>_it<NNNN>.h5``
(``NNNN`` the cumulative iteration, at least four digits) in the final cal's schema, so every reader
of a cal reads it: the mosaic (``field.mosaic(recipe, cal=<snapshot>)``), a warm start
(``field.calibrate(recipe, start=<snapshot>)``), :class:`~selfcal.io.calfile.CalFile` and the
analysis tools. The parts of a cal that do not depend on the solution (the sky terms' coverage,
Fisher information and separability, the offsets' coverage, the chunk maps, the frame list) are
written once, before the solve, to a template next to the snapshots (``.<cal stem>_static-<pid>.h5``,
removed at the end of the solve); each snapshot is a copy of the template plus the solution (the sky
maps, written a band of chunk rows at a time; each offset term's offsets; the per-frame scalar) and
the attributes of its ``solve`` group: ``snapshot = True``, ``iteration`` (cumulative across warm
starts: the start's ``iterations_total`` plus this solve's iteration), ``iterations``,
``iterations_total``, the solver's estimates at that iteration (``r1norm`` ... ``xnorm``, ``test1``,
``test2``), what the solver runs with, and the identity of the system and of the start, as the final
cal has them (:mod:`selfcal.core.warm_start`). Each file is written atomically
(:func:`~selfcal.io.atomic.atomic_path`), from the solver's thread: no process is started.

Memory: a snapshot never holds a second copy of the solution. It reads one band of chunk rows of a
sky map at a time (196 rows of a 12544 x 12538 float32 map, ~10 MB), one offset term (expanded per
frame, as the cal stores it) and the per-frame scalar, besides the solver's own vectors.

Retention: ``keep=m`` deletes the oldest snapshot this solve wrote once it has written ``m + 1``;
snapshots of earlier runs are never deleted. A snapshot that cannot be written (a full disk) is
logged as an error and skipped: the solve goes on. Snapshots are not products: they have no sidecar
and never make a cal current or stale.
"""
from __future__ import annotations

import glob
import logging
import os
import shutil
import time

import numpy as np

__all__ = ['ActiveColumns', 'Iterate', 'SnapshotWriter', 'snapshot_path', 'SNAPSHOT_DIR']

logger = logging.getLogger(__name__)

#: The directory of a cal's snapshots, inside the cal's own directory.
SNAPSHOT_DIR = 'snapshots'

# The columns per entry of the table of active columns before each block (ActiveColumns).
_PREFIX_BLOCK = 1 << 22


def snapshot_path(cal_path, iteration) -> str:
    """The snapshot of the solve that writes ``cal_path`` at (cumulative) iteration ``iteration``:
    ``<cal dir>/snapshots/<cal stem>_it<NNNN>.h5``."""
    directory, name = os.path.split(os.path.abspath(os.fspath(cal_path)))
    stem = os.path.splitext(name)[0]
    return os.path.join(directory, SNAPSHOT_DIR, f'{stem}_it{int(iteration):04d}.h5')


class ActiveColumns:
    """The active columns of a compacted solve (``active_mask``: a column of the full layout is in
    the solver's compact vector when its mask is true, in order), with the number of active columns
    before each block of :data:`_PREFIX_BLOCK` columns: the compact index of a full column is then
    found without a full-size table."""

    def __init__(self, mask):
        self.mask = np.asarray(mask, dtype=bool)
        counts = [int(np.count_nonzero(self.mask[s:s + _PREFIX_BLOCK]))
                  for s in range(0, self.mask.size, _PREFIX_BLOCK)]
        self._before = np.concatenate(([0], np.cumsum(counts, dtype=np.int64)))

    def compact_start(self, start) -> int:
        """The compact index of the first active column at or after full column ``start``."""
        block = int(start) // _PREFIX_BLOCK
        return int(self._before[block]) + int(np.count_nonzero(self.mask[block * _PREFIX_BLOCK:int(start)]))


class Iterate:
    """The solver's iterate after iteration ``itn``, read in the solution's physical units and full
    column layout, block by block (:meth:`read`).

    ``x``: the solver's vector (compact when ``active`` is set; the solver's own buffer, read
    while the solver waits); ``scale``: the column scaling the solve runs with (``x * scale`` is
    physical; None: unscaled); ``active``: :class:`ActiveColumns` of a compacted solve (None: every
    column is in ``x``); ``num_cols``: the full layout's columns. ``estimates``: the solver's
    estimates at the iteration (a row of :class:`~selfcal.core.solve_record.SolveHistory`);
    ``settings``: what the solver runs with (``method``, ``atol``, ``btol``, ``conlim``, ``damp``,
    ``iteration_limit``, ``rows``, ``columns``)."""

    def __init__(self, itn, x, *, scale=None, active=None, num_cols=None, estimates=None, settings=None):
        self.itn = int(itn)
        self._x = x
        self._scale = scale
        self._active = active
        self.num_cols = int(x.shape[0] if num_cols is None else num_cols)
        self.estimates = dict(estimates or {})
        self.settings = dict(settings or {})
        #: The solution's dtype: that of ``x * scale`` (the solve's end makes the same product).
        self.dtype = x.dtype if scale is None else np.result_type(x.dtype, scale.dtype)

    def read(self, start, stop) -> np.ndarray:
        """The solution's columns ``start:stop`` of the full layout, a new array of :attr:`dtype`:
        each active column ``x[c] * scale[c]`` (the solver's ``c``-th column), every other column
        0, exactly as :func:`~selfcal.core.solve.apply_lsqr` converts its final vector."""
        start, stop = int(start), int(stop)
        if self._active is None:
            seg = self._x[start:stop]
            return seg * self._scale[start:stop] if self._scale is not None else seg.copy()
        mask = self._active.mask[start:stop]
        c0 = self._active.compact_start(start)
        c1 = c0 + int(np.count_nonzero(mask))
        out = np.zeros(stop - start, dtype=self.dtype)
        seg = self._x[c0:c1]
        out[mask] = seg * self._scale[c0:c1] if self._scale is not None else seg
        return out


class SnapshotWriter:
    """Writes the snapshots of one solve (see the module docstring).

    ``cal_path``: the cal the solve writes (the snapshots go to ``snapshots/`` beside it, named
    after it); ``every``: the iterations between two snapshots; ``keep``: how many to keep (None:
    all). ``frames``: the frame list the snapshots record (``reproj_list``; None: the calibrator's;
    the run engine gives the frames' permanent paths, as the cal records them). ``system``: the
    :class:`~selfcal.core.warm_start.System` solved; ``start``: the
    :class:`~selfcal.core.warm_start.WarmStart` the solve continues (None: a solve from the default
    guess), whose ``iterations_total`` the iterations count from.

    :meth:`bind` writes the template from a set-up
    :class:`~selfcal.pipeline.pipeline_wrapper.Calibrator` before its solve; calling the writer with
    an :class:`Iterate` writes a snapshot; :meth:`close` removes the template; :meth:`summary` says
    what was written."""

    def __init__(self, cal_path, every, keep=None, *, frames=None, system=None, start=None):
        if int(every) < 1:
            raise ValueError(f"snapshots every {every} iterations: at least 1")
        if keep is not None and int(keep) < 1:
            raise ValueError(f"keep={keep} snapshots: at least 1 (None: all)")
        self.cal_path = os.path.abspath(os.fspath(cal_path))
        self.directory = os.path.join(os.path.dirname(self.cal_path), SNAPSHOT_DIR)
        self.stem = os.path.splitext(os.path.basename(self.cal_path))[0]
        self.every = int(every)
        self.keep = None if keep is None else int(keep)
        self.frames = None if frames is None else list(frames)
        self.system = system
        self.start = start
        #: The iterations before this solve's first: 0, the start's total, or None (unknown).
        self.before = 0 if start is None else start.iterations_total
        self.written = []           # the snapshots this solve wrote and keeps, oldest first
        self.removed = []           # those it deleted (retention)
        self.failed = []            # (iteration, error) of those it could not write
        self._cc = None
        self._template = None

    def __repr__(self):
        return f"SnapshotWriter({self.cal_path!r}, every={self.every}, keep={self.keep})"

    # ---- the template ------------------------------------------------------------------------
    def bind(self, cc):
        """Write the template, the parts of the cal that do not depend on the solution, from ``cc``
        (a set-up :class:`~selfcal.pipeline.pipeline_wrapper.Calibrator`, its pixel state still in
        memory or parked by its setup), and keep ``cc`` to write the solution's parts."""
        import h5py

        from ..io.atomic import _alive
        os.makedirs(self.directory, exist_ok=True)
        for stale in glob.glob(os.path.join(glob.escape(self.directory), f'.{glob.escape(self.stem)}_static-*.h5')):
            pid = os.path.basename(stale)[len(self.stem) + len('._static-'):-len('.h5')]
            if pid.isdigit() and not _alive(int(pid)):
                os.remove(stale)              # the template of a solve that died
        earlier = sorted(glob.glob(os.path.join(glob.escape(self.directory), f'{glob.escape(self.stem)}_it*.h5')))
        if earlier:
            logger.info(f"snapshots: {len(earlier)} snapshots of {self.stem} from earlier runs in {self.directory} "
                        f"are left as they are (one of the same iteration is replaced)")
        if self.before is None:
            logger.warning(f"snapshots: the start records no iteration count; the snapshots of {self.stem} are "
                           f"named by this solve's iterations, and record iteration = -1")
        self._template = os.path.join(self.directory, f'.{self.stem}_static-{os.getpid()}.h5')
        t0 = time.perf_counter()
        with h5py.File(self._template, 'w') as f:
            cc.write_cal_static(f, reproj_list=self.frames)
        self._cc = cc
        logger.info(f"snapshots: every {self.every} iterations "
                    f"({'all kept' if self.keep is None else f'the last {self.keep} kept'}) in {self.directory}; "
                    f"the cal's parts that do not depend on the solution written once "
                    f"({os.path.getsize(self._template) / 2**20:.1f} MB, {time.perf_counter() - t0:.1f} s)")

    def close(self):
        """Remove the template (the snapshots stay)."""
        if self._template is not None and os.path.exists(self._template):
            os.remove(self._template)
        self._template = None
        self._cc = None

    # ---- a snapshot --------------------------------------------------------------------------
    def iteration(self, itn) -> int:
        """The cumulative iteration of this solve's iteration ``itn`` (-1: unknown)."""
        return -1 if self.before is None else int(self.before) + int(itn)

    def path(self, itn) -> str:
        """The file of the snapshot after this solve's iteration ``itn``, named by its cumulative
        iteration (by ``itn`` when the start records none)."""
        it = self.iteration(itn)
        return snapshot_path(self.cal_path, itn if it < 0 else it)

    def solve_attrs(self, iterate) -> dict:
        """The attributes of a snapshot's ``solve`` group (see the module docstring)."""
        from .solve_record import VERSION
        it = self.iteration(iterate.itn)
        s, e = iterate.settings, iterate.estimates
        out = {'version': VERSION, 'snapshot': True, 'iteration': it, 'method': str(s.get('method', '')),
               'iterations': iterate.itn, 'iterations_total': it}
        if 'iteration_limit' in s:
            out['iteration_limit'] = int(s['iteration_limit'])
        for k in ('r1norm', 'r2norm', 'arnorm', 'anorm', 'acond', 'xnorm', 'test1', 'test2'):
            if k in e:
                out[k] = float(e[k])
        for k in ('atol', 'btol', 'conlim', 'damp'):
            if k in s:
                out[k] = float(s[k])
        for k in ('rows', 'columns'):
            if k in s:
                out[k] = int(s[k])
        if self.system is not None:
            out['system'] = self.system.fingerprint()
            out['system_identity'] = self.system.identity_json()
        if self.start is not None:
            out['start_from'] = self.start.path
            if self.start.identity is not None:
                out['start_identity'] = self.start.identity
        return out

    def __call__(self, iterate):
        """Write the snapshot of ``iterate`` (an :class:`Iterate`), then delete the oldest beyond
        ``keep``. A file-system error is logged and the snapshot skipped."""
        import h5py

        from ..io.atomic import atomic_path
        if self._cc is None:
            raise ValueError("SnapshotWriter: bind() it to the calibrator before the solve")
        path = self.path(iterate.itn)
        t0 = time.perf_counter()
        try:
            with atomic_path(path) as tmp:
                shutil.copyfile(self._template, tmp)
                with h5py.File(tmp, 'r+') as f:
                    self._cc.write_cal_solution(f, iterate)
                    group = f.create_group('solve')
                    for k, v in self.solve_attrs(iterate).items():
                        group.attrs[k] = v
        except OSError as e:
            self.failed.append((iterate.itn, f'{type(e).__name__}: {e}'))
            logger.error(f"snapshot at iteration {iterate.itn} not written ({path}): {e}; the solve goes on")
            return
        if path in self.written:                 # (never: the iterations increase)
            self.written.remove(path)
        self.written.append(path)
        while self.keep is not None and len(self.written) > self.keep:
            old = self.written.pop(0)
            try:
                os.remove(old)
                self.removed.append(old)
            except FileNotFoundError:
                pass
        r1 = iterate.estimates.get('r1norm')
        logger.info(f"snapshot: iteration {iterate.itn}" + (f" ({self.iteration(iterate.itn)} in all)"
                                                            if self.before else '')
                    + f" -> {path} ({os.path.getsize(path) / 2**20:.1f} MB, {time.perf_counter() - t0:.1f} s"
                    + (f"; r1norm {r1:.6e}" if r1 is not None else '') + ')')

    def summary(self) -> dict:
        """What the solve's snapshots are: ``every``, ``keep``, ``directory``, the snapshots kept
        (``written``), how many retention deleted (``removed``), and the failures (``failed``)."""
        return {'every': self.every, 'keep': self.keep, 'directory': self.directory, 'written': list(self.written),
                'removed': len(self.removed), 'failed': [list(f) for f in self.failed]}
