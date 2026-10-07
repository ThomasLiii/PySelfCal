"""Write a file so that it exists only once it is complete.

A product written in place and interrupted (a killed job, a full disk) leaves a truncated file
that the next run finds and reuses. :func:`atomic_path` hands the writer a temporary name in the
same directory and renames it to the final name only when the writer finished without an
error; the rename is atomic on POSIX file systems, so the final name holds either nothing, the
previous complete file, or the new complete file::

    with atomic_path(cal_path) as tmp:
        with h5py.File(tmp, 'w') as f:
            ...

The temporary name keeps the final extension (``<name>.part-<pid>-<n><ext>``), so writers that
infer the format from it (``np.save``, ``fits.writeto``) behave the same. Nothing inside an
HDF5, FITS or NumPy file records its own name, so the bytes are those of a direct write.
"""
from __future__ import annotations

import contextlib
import glob
import itertools
import os
import time

__all__ = ['atomic_path', 'is_partial', 'sweep_partials']

_counter = itertools.count()
_MARK = '.part-'


def is_partial(path) -> bool:
    """Whether ``path`` is the temporary name of an unfinished :func:`atomic_path` write."""
    return _MARK in os.path.basename(os.fspath(path))


def _alive(pid) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:                 # another user's process
        return True
    return True


def sweep_partials(path, min_age_s=3600.0) -> list[str]:
    """Delete the temporary files that interrupted writes of ``path`` left behind: those whose
    process is gone (killed, out of memory) and that nothing has written to for ``min_age_s``
    seconds (a write from another machine sharing the disk is still being written to). Returns
    the deleted paths."""
    directory, name = os.path.split(os.fspath(path))
    stem, ext = os.path.splitext(name)
    removed = []
    for p in glob.glob(os.path.join(directory or '.', glob.escape(stem) + _MARK + '*' + glob.escape(ext))):
        tail = os.path.basename(p)[len(stem) + len(_MARK):len(os.path.basename(p)) - len(ext)]
        pid = tail.split('-', 1)[0]
        if not pid.isdigit() or _alive(int(pid)):
            continue
        try:
            if time.time() - os.stat(p).st_mtime < min_age_s:
                continue
            os.remove(p)
            removed.append(p)
        except FileNotFoundError:
            pass
    return removed


@contextlib.contextmanager
def atomic_path(path):
    """Yield a temporary path next to ``path``; rename it to ``path`` when the block succeeds,
    delete it when the block raises."""
    path = os.fspath(path)
    directory, name = os.path.split(path)
    stem, ext = os.path.splitext(name)
    tmp = os.path.join(directory, f'{stem}{_MARK}{os.getpid()}-{next(_counter)}{ext}')
    try:
        yield tmp
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
