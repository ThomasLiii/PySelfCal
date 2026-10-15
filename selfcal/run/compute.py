"""The machine a run uses: scratch disk, worker processes, staging. Never changes a byte.

:class:`Compute` given to a :class:`~selfcal.run.field.Field` is the default of its actions;
an action may override it. A site profile is a module-level constant::

    ORCA = sc.Compute(scratch="/home/me/selfcal/cache", workers=48, coadd_workers=96)
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import KW_ONLY, dataclass, field
from typing import Literal

from ..config.base import Config, ConfigError

__all__ = ['Compute', 'Tuning', 'pin_threads', 'environment', 'TUNING_VARIABLES']

#: The thread-count variables every action pins to 1 before any worker starts.
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS',
                    'NUMEXPR_NUM_THREADS')


def pin_threads():
    """Pin the BLAS / OpenMP thread counts to 1, for this process and the workers it starts.

    The solver's thread pool is the only parallelism inside a process; an unpinned BLAS changes
    the last bits of the N-pass refit and runs workers x cores threads. The variables are set
    for the workers, and ``threadpoolctl`` limits the libraries this process has already
    loaded (numpy is usually imported before selfcal). Returns the previous variable values.
    """
    old = {k: os.environ.get(k) for k in THREAD_VARIABLES}
    for k in THREAD_VARIABLES:
        os.environ[k] = '1'
    try:
        from threadpoolctl import threadpool_limits
        threadpool_limits(1)
    except Exception:                      # threadpoolctl missing or a library it cannot reach
        pass
    return old


#: The byte-neutral SELFCAL_* environment knobs, by Tuning setting.
TUNING_VARIABLES = {
    'start_method': 'SELFCAL_MP_START_METHOD',
    'scatter_workers': 'SELFCAL_SCATTER_WORKERS',
    'scatter_timeout_s': 'SELFCAL_SCATTER_TIMEOUT_S',
    'block_nnz': 'SELFCAL_BLOCK_NNZ',
    'split_ranges': 'SELFCAL_RMATVEC_SPLIT',
    'split_max_gb': 'SELFCAL_RMATVEC_SPLIT_MAX_GB',
    'split_extra_gb': 'SELFCAL_SPLIT_EXTRA_GB',
    'vector_threads': 'SELFCAL_VEC_THREADS',
    'flush_stripes': 'SELFCAL_COADD_FLUSH_STRIPES',
    'spill_min_gb': 'SELFCAL_SPILL_MIN_GB',
    'spill_dir': 'SELFCAL_SPILL_DIR',
}


@dataclass(frozen=True)
class Tuning(Config):
    """Expert knobs of the solver's machinery; every one leaves the products byte-identical.

    Each is a ``SELFCAL_*`` environment variable of the library (:data:`TUNING_VARIABLES`); a
    setting left at None defers to the variable (or the library's default). An action sets the
    given ones for its duration, before any worker pool starts, and records every resolved value.
    ``start_method``: the worker pools' start method (``"forkserver"``; ``"fork"`` only to debug);
    ``scatter_workers`` / ``scatter_timeout_s``: the parallel scatter of the assembly (0 or 1:
    serial); ``block_nnz``: the matrix size above which it is stored in blocks;
    ``split_ranges`` / ``split_max_gb`` / ``split_extra_gb``: the column-partitioned storage of the
    sequential transpose product; ``vector_threads``: the solver's elementwise vector updates;
    ``flush_stripes``: the coadd's row stripes; ``spill_min_gb`` / ``spill_dir``: when and where the
    solver parks arrays on disk.
    """
    _: KW_ONLY
    start_method: Literal['forkserver', 'fork', 'spawn'] | None = None
    scatter_workers: int | None = None
    scatter_timeout_s: float | None = None
    block_nnz: int | None = None
    split_ranges: int | None = None
    split_max_gb: float | None = None
    split_extra_gb: float | None = None
    vector_threads: int | None = None
    flush_stripes: int | None = None
    spill_min_gb: float | None = None
    spill_dir: str | None = None

    def environment(self) -> dict:
        """``{variable: value}`` of the settings that are given."""
        return {var: str(getattr(self, k)) for k, var in TUNING_VARIABLES.items() if getattr(self, k) is not None}

    @staticmethod
    def resolved() -> dict:
        """Every knob's value in effect (the variable, or ``"default"``)."""
        return {k: os.environ.get(var, 'default') for k, var in TUNING_VARIABLES.items()}



@dataclass(frozen=True)
class Compute(Config):
    """Resources for a run; every setting leaves the products byte-identical.

    ``scratch``: fast local disk for the staged frames, the solver's spill files and the coadd's
    frame cache (None: frames are read in place, ``io_limit`` at a time). ``workers``: processes
    that assemble the system (default ``min(48, CPUs)``); ``coadd_workers``: the coadd's (default
    ``workers``). ``stage``: ``"copy"`` (a verified copy into a staging directory the run owns),
    ``"reuse"`` (a staging another run made) or None (read in place); ``stage_dir``: where (default
    ``<scratch>/reproj_nvme_<field name>``); ``keep_staged``: keep the copy afterwards.
    ``io_limit``: concurrent reads from the slow disk while staging. ``cache_frames``: the coadd
    caches each corrected frame for its later passes. ``memory_guard``: end the process cleanly
    before the machine runs out of memory (None: on for tiled runs). ``tuning``: the expert knobs of
    the solver's machinery (:class:`Tuning`).
    """
    scratch: str | None = None
    _: KW_ONLY
    workers: int | None = None
    coadd_workers: int | None = None
    stage: Literal['copy', 'reuse'] | None = 'copy'
    stage_dir: str | None = None
    keep_staged: bool = False
    io_limit: int = 20
    cache_frames: bool = True
    memory_guard: bool | None = None
    tuning: Tuning = field(default_factory=Tuning)

    def _validate(self):
        for k in ('workers', 'coadd_workers', 'io_limit'):
            v = getattr(self, k)
            if v is not None and v < 1:
                raise ConfigError(f"Compute({k}={v}): at least 1")

    @property
    def stages(self) -> bool:
        """Whether frames are staged (a ``stage`` and somewhere to stage them); else read in place."""
        return self.stage is not None and (self.scratch is not None or self.stage_dir is not None)

    @property
    def resolved_workers(self) -> int:
        return self.workers if self.workers is not None else min(48, os.cpu_count() or 1)

    @property
    def resolved_coadd_workers(self) -> int:
        return self.coadd_workers if self.coadd_workers is not None else self.resolved_workers


@contextlib.contextmanager
def environment(compute, numerics=None):
    """Set the action's environment for its duration: the tuning knobs and the bit-relevant
    transpose-product threads of ``numerics``, restored afterwards (also when the action raises).
    The BLAS / OpenMP thread pins (:func:`pin_threads`) are not restored: they hold for the rest
    of the process."""
    env = dict(compute.tuning.environment())
    if numerics is not None:
        env['SELFCAL_PARALLEL_RMATVEC'] = ('auto' if numerics.rmatvec_threads is None
                                           else str(numerics.rmatvec_threads))
        env['SELFCAL_RMATVEC_BUFFER_GB'] = '16'
    old = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        yield env
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
