"""Module-level mutable state shared across selfcal submodules."""
from __future__ import annotations

from multiprocessing import Lock as _MPLock, BoundedSemaphore as _MPSemaphore

__all__ = ["set_hdd_io_limit", "set_progress"]

# Semaphore to limit concurrent HDD reads. With many workers doing random reads
# on a RAID array, seek thrashing kills throughput. Uses multiprocessing.BoundedSemaphore
# so it works across both threads (ThreadPoolExecutor) and forked processes (Pool).
_hdd_io_semaphore = None
_coadd_turn = None

def _init_coadd_worker(cond, counters):
    """Pool initializer for the coadd workers: the striped turnstile — a
    Condition plus one per-row-stripe batch counter — that orders the
    per-batch flushes into the shared totals."""
    global _coadd_turn
    _coadd_turn = (cond, counters)

def set_hdd_io_limit(n: int | None) -> None:
    """Set the max number of concurrent file reads from slow storage.
    Call before any parallel processing starts. Works across both threads and processes.
    """
    global _hdd_io_semaphore
    _hdd_io_semaphore = _MPSemaphore(n) if n and n > 0 else None


# Whether library functions render tqdm progress bars (stderr). Applications
# (the runner, interactive sessions) usually want them; embedding callers and
# batch logs usually do not. Read at each call site via
# ``disable=not _state.progress_enabled`` so a mid-run toggle takes effect on
# the next bar. Default True preserves the historical behavior.
progress_enabled = True

def set_progress(enabled: bool) -> None:
    """Enable/disable tqdm progress bars rendered by selfcal library calls."""
    global progress_enabled
    progress_enabled = bool(enabled)
