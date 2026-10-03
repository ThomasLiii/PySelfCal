"""The machine a run uses: scratch disk, worker processes, staging. Never changes a byte.

:class:`Compute` given to a :class:`~selfcal.run.field.Field` is the default of its actions;
an action may override it. A site profile is a module-level constant::

    ORCA = sc.Compute(scratch="/home/me/selfcal/cache", workers=48, coadd_workers=96)
"""
from __future__ import annotations

import os
from dataclasses import KW_ONLY, dataclass
from typing import Literal

from ..config.base import Config, ConfigError

__all__ = ['Compute', 'pin_threads']

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
    before the machine runs out of memory (None: on for tiled runs).
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
