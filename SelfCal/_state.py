"""Module-level mutable state shared across SelfCal submodules."""

from multiprocessing import Lock as _MPLock, BoundedSemaphore as _MPSemaphore

# Semaphore to limit concurrent HDD reads. With many workers doing random reads
# on a RAID array, seek thrashing kills throughput. Uses multiprocessing.BoundedSemaphore
# so it works across both threads (ThreadPoolExecutor) and forked processes (Pool).
_hdd_io_semaphore = None
_coadd_flush_lock = None

def _init_coadd_worker(lock):
    """Pool initializer: store the multiprocessing Lock as a module global."""
    global _coadd_flush_lock
    _coadd_flush_lock = lock

def set_hdd_io_limit(n):
    """Set the max number of concurrent file reads from slow storage.
    Call before any parallel processing starts. Works across both threads and processes.
    """
    global _hdd_io_semaphore
    _hdd_io_semaphore = _MPSemaphore(n) if n and n > 0 else None
