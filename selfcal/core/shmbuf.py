"""Shared-memory arrays handed to worker processes EXPLICITLY (no fork inheritance).

Why this exists: forking a worker pool from a multi-threaded parent is unsafe —
a child inherits every lock in the state it had at fork time, owned by threads
that do not exist in the child. The runner's RSS-guardrail thread prints to
stderr every 15 s; a child forked while that print holds the stderr buffer lock
hangs at its own EXIT (multiprocessing's child bootstrap flushes stderr), the
parent's pool shutdown joins it forever, and its finished siblings pile up as
zombies. Measured: 60/60 children hang under a printing thread with fork,
0/60 with forkserver. So pools use a forkserver context (children are forked
from a single-threaded server) and get their shared arrays through this class.

``SharedBuffer`` backs an ndarray with a ``memfd`` (anonymous tmpfs file):
shmem-class RAM like the old MAP_SHARED|MAP_ANON buffers — counted by the
hard-RSS guardrail, hole-punchable with MADV_REMOVE, and NOT subject to the
``/dev/shm`` mount size (``multiprocessing.shared_memory`` is; a big tile's
CSR can exceed it). Pickling a SharedBuffer sends the file descriptor
(``multiprocessing.reduction.DupFd`` — works with forkserver/spawn); the
receiving process maps the same pages. Pages live while any mapping does.
"""
import ctypes
import mmap
import os

import numpy as np

_libc = None


def _memfd_create(name: str, flags: int = 1) -> int:   # 1 = MFD_CLOEXEC
    if hasattr(os, 'memfd_create'):
        return os.memfd_create(name, flags)
    global _libc
    if _libc is None:
        _libc = ctypes.CDLL(None, use_errno=True)
        _libc.memfd_create.argtypes = [ctypes.c_char_p, ctypes.c_uint]
        _libc.memfd_create.restype = ctypes.c_int
    fd = _libc.memfd_create(name.encode(), flags)
    if fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, f"memfd_create failed: {os.strerror(err)}")
    return fd


class SharedBuffer:
    """n-element ndarray (``.array``) in a memfd-backed MAP_SHARED mapping.

    Picklable: the receiver gets a duplicate of the fd and maps the same
    physical pages. Close the receiver's copy with :meth:`close` when done
    (drops its mapping; the pages persist while the creator's array lives).
    """

    def __init__(self, n, dtype, name='selfcal'):
        self.n = int(n)
        self.dtype = np.dtype(dtype)
        nbytes = max(1, self.n * self.dtype.itemsize)
        self.fd = _memfd_create(name)
        os.ftruncate(self.fd, nbytes)
        self._map(nbytes)

    def _map(self, nbytes):
        self.mm = mmap.mmap(self.fd, nbytes, flags=mmap.MAP_SHARED)
        self.array = np.frombuffer(self.mm, dtype=self.dtype, count=self.n)

    def __reduce__(self):
        from multiprocessing.reduction import DupFd
        return (SharedBuffer._rebuild, (DupFd(self.fd), self.n, self.dtype.str))

    @staticmethod
    def _rebuild(dup, n, dtype_str):
        self = object.__new__(SharedBuffer)
        self.n = int(n)
        self.dtype = np.dtype(dtype_str)
        self.fd = dup.detach()
        self._map(max(1, self.n * self.dtype.itemsize))
        return self

    def close(self):
        """Release THIS process's mapping and fd (the array must not be in use)."""
        self.array = None
        if getattr(self, 'mm', None) is not None:
            try:
                self.mm.close()
            except BufferError:
                pass                     # a view is still alive; GC closes it later
            self.mm = None
        if getattr(self, 'fd', -1) >= 0:
            os.close(self.fd)
            self.fd = -1

    def __del__(self):
        try:
            if getattr(self, 'fd', -1) >= 0:
                os.close(self.fd)        # mapping (and its pages) outlive the fd
                self.fd = -1
        except OSError:
            pass


def worker_pool_context():
    """multiprocessing context for the pipeline's process pools.

    ``SELFCAL_MP_START_METHOD`` (default ``forkserver``): ``forkserver`` is
    fork-safe by construction (see module docstring); ``fork`` restores the
    old behaviour for debugging. The forkserver preload list is emptied so
    driver scripts with module-level argv parsing are never re-imported.
    """
    import multiprocessing
    method = os.environ.get('SELFCAL_MP_START_METHOD', 'forkserver')
    ctx = multiprocessing.get_context(method)
    if method == 'forkserver':
        ctx.set_forkserver_preload([])
    return ctx
