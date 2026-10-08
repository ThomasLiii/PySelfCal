"""Spill/restore for large write-once arrays that idle through a hot phase.

The setup products ``pixel_counts`` / ``pixel_fisher`` / ``pixel_cross`` are
written while ``setup_lsqr`` collates worker results, consumed by its
constraint-row builders and zero-column compaction (the numbered ``Phase N``
comment sections inside ``setup_lsqr``, selfcal/core/system.py), and then
not read again until ``save_calibration`` — yet they are among the largest
things resident during BOTH the setup peak (the CSR allocate/scatter/build
steps at the end of ``setup_lsqr``) and the entire LSQR solve. Each is one
value per solve column — dominated by the J·N_pix sky columns for J sky
blocks over an N_pix-pixel reference grid — so roughly 8·J·N_pix bytes
apiece (e.g. ~17 GB each at N_pix ~ 5e8 pixels, J = 4).

``np.save``/``np.load`` round-trip int64/float64 arrays losslessly, so
parking them on scratch disk and reloading is byte-identical to never having
spilled — the arrays and every downstream float accumulation are unchanged.
``pixel_cross`` is a dict whose ITERATION ORDER matters (save-time consumers
walk ``items()`` and their float accumulation order must not change), so the
key order is recorded and reproduced exactly.

Spilling only triggers above ``$SELFCAL_SPILL_MIN_GB`` (default 4 GB) so
small runs pay no I/O at all; for tens-of-GB arrays the save+reload round
trip costs tens of seconds of scratch-disk I/O, negligible against a
setup+solve that runs for hours. ``$SELFCAL_SPILL_DIR`` overrides the
location.

:class:`ParkedVector` keeps the right-hand side of an LSQR solve, which the solver overwrites, on
disk in the run's scratch directory for the residual checks after it (always on disk, never in
memory).
"""
import contextlib
import logging
import os
import re
import shutil
import socket
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np

logger = logging.getLogger(__name__)


def _nbytes(counts, fisher, cross):
    n = sum(a.nbytes for a in (counts, fisher) if a is not None)
    if isinstance(cross, dict):
        n += sum(a.nbytes for a in cross.values())
    elif cross is not None:
        n += cross.nbytes
    return n


def spill_pixel_state(counts, fisher, cross, label='', min_gb=None):
    """Write the three arrays to a fresh scratch dir and report it.

    Returns ``(spill_dir, n_bytes)``; ``spill_dir`` is None when there was
    nothing to spill or the total sits below the threshold, in which case the
    caller keeps its arrays.
    """
    if counts is None and fisher is None and cross is None:
        return None, 0
    n_bytes = _nbytes(counts, fisher, cross)
    if min_gb is None:
        min_gb = float(os.environ.get('SELFCAL_SPILL_MIN_GB', 4.0))
    if n_bytes < min_gb * 2**30:
        return None, n_bytes
    base = os.environ.get('SELFCAL_SPILL_DIR') or tempfile.gettempdir()
    spill_dir = tempfile.mkdtemp(prefix='selfcal_pixel_spill_', dir=base)
    # flush=True dropped in the print->logger migration: logging handlers
    # flush per emitted record, so logger.info has no (and needs no) flush kwarg.
    logger.info(f"Spilling pixel state ({n_bytes/2**30:.1f} GB) to {spill_dir}"
                f"{' ' + label if label else ''}...")
    jobs = []
    if counts is not None:
        jobs.append(('pixel_counts.npy', counts))
    if fisher is not None:
        jobs.append(('pixel_fisher.npy', fisher))
    if cross is not None:
        if isinstance(cross, dict):
            keys = list(cross.keys())
            jobs.append(('pixel_cross_keys.npy',
                         np.asarray(keys, dtype=np.int64)))
            jobs.extend((f'pixel_cross_{n}.npy', cross[k])
                        for n, k in enumerate(keys))
        else:
            jobs.append(('pixel_cross.npy', cross))
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(lambda j: np.save(os.path.join(spill_dir, j[0]), j[1],
                                      allow_pickle=False), jobs))
    return spill_dir, n_bytes


class PixelSpill:
    """Handle to pixel state parked on disk, restored on first demand.

    Lets ``setup_lsqr`` hand the arrays off WITHOUT materialising them: they
    stay on scratch until ``save_calibration`` actually reads them, so they
    never sit alongside the finished CSR (the point of peak resident memory
    in a large run) and are not written out a second time by ``apply_lsqr``.

    ``num_sky_blocks`` is carried so the J==2 convention — ``pixel_cross``
    collapses from the ``{(i, j): array}`` dict to the bare pair-(0,1) array —
    is applied on restore, so callers receive the same shape whether or not
    the arrays were spilled.
    """

    __slots__ = ('spill_dir', 'num_sky_blocks')

    def __init__(self, spill_dir, num_sky_blocks):
        self.spill_dir = spill_dir
        self.num_sky_blocks = int(num_sky_blocks)

    def restore(self):
        """→ (pixel_counts, pixel_fisher, pixel_cross); removes the scratch dir."""
        counts, fisher, cross = restore_pixel_state(self.spill_dir)
        if isinstance(cross, dict) and self.num_sky_blocks == 2:
            cross = cross[(0, 1)]
        self.spill_dir = None
        return counts, fisher, cross

    def peek(self):
        """→ (pixel_counts, pixel_fisher, pixel_cross) memory-mapped read-only; the scratch dir
        stays (the snapshots of a solve read the coverage from it before the solve)."""
        counts, fisher, cross = restore_pixel_state(self.spill_dir, cleanup=False, mmap_mode='r')
        if isinstance(cross, dict) and self.num_sky_blocks == 2:
            cross = cross[(0, 1)]
        return counts, fisher, cross

    def discard(self):
        """Drop the scratch dir without reading it (caller never needed it)."""
        if self.spill_dir is not None:
            shutil.rmtree(self.spill_dir, ignore_errors=True)
            self.spill_dir = None

    def __repr__(self):
        return f"<PixelSpill {self.spill_dir!r} J={self.num_sky_blocks}>"


def restore_pixel_state(spill_dir, cleanup=True, mmap_mode=None):
    """Reload what :func:`spill_pixel_state` wrote → (counts, fisher, cross).

    Missing files come back as None; the ``pixel_cross`` dict is rebuilt in
    its original insertion order. ``mmap_mode`` (``np.load``'s, e.g. ``'r'``)
    maps the arrays instead of reading them (keep ``cleanup=False`` then).
    """
    def _load(name):
        p = os.path.join(spill_dir, name)
        return np.load(p, mmap_mode=mmap_mode) if os.path.exists(p) else None

    with ThreadPoolExecutor(max_workers=4) as ex:
        f_counts = ex.submit(_load, 'pixel_counts.npy')
        f_fisher = ex.submit(_load, 'pixel_fisher.npy')
        keys = _load('pixel_cross_keys.npy')
        if keys is not None:
            arrs = list(ex.map(_load, [f'pixel_cross_{n}.npy'
                                       for n in range(len(keys))]))
            cross = {tuple(int(v) for v in k): a for k, a in zip(keys, arrs)}
        else:
            cross = _load('pixel_cross.npy')
        counts, fisher = f_counts.result(), f_fisher.result()
    if cleanup:
        shutil.rmtree(spill_dir, ignore_errors=True)
    return counts, fisher, cross


class ParkedVector:
    """A copy of a vector parked on disk for reads after the solve has overwritten the original.

    LSQR overwrites its right-hand side ``b`` (``lsqr_inplace`` uses its buffer as the first
    Lanczos vector), yet the record of the solve needs ``|b - A x|`` at the end (and the monitors
    at their checks). The copy is never kept in memory, whatever its size: it is written to
    ``directory`` (the run's scratch directory, which the run engine gives) as one ``.npy`` file,
    and read back a piece at a time through a short-lived memory map (:meth:`__getitem__`), so the
    solve's resident memory does not grow by the vector and no read maps more than the piece it
    returns. ``np.save`` round-trips the values exactly.

    Parking is best effort: when ``directory`` is None, or the file cannot be written (no space, an
    unwritable directory, ...), the reason is logged (a warning for a failure) and :attr:`available`
    is False; the solve goes on, and the residuals that need the vector are recorded as NaN.
    :meth:`discard` removes the file. The file is named after this machine and process,
    ``selfcal_parked_<host>_<pid>_<random>.npy``; parking sweeps the files a dead process of this
    machine left in ``directory`` (a killed run).
    """

    __slots__ = ('_path', '_dtype', '_shape', '_offset', 'reason')

    PREFIX = 'selfcal_parked_'

    def __init__(self, vec, directory, label=''):
        self._path = None
        self._dtype = self._shape = self._offset = None
        #: Why the vector is not available (None when it is).
        self.reason = None
        what = label or 'a vector'
        if directory is None:
            self.reason = 'no scratch directory was given to park it in'
            logger.info(f"Not parking {what}: {self.reason}")
            return
        vec = np.asarray(vec)
        path = None
        try:
            directory = os.fspath(directory)
            os.makedirs(directory, exist_ok=True)
            sweep_parked(directory)
            fd, path = tempfile.mkstemp(prefix=f'{self.PREFIX}{_HOST}_{os.getpid()}_', suffix='.npy',
                                        dir=directory)
            logger.info(f"Parking {what} ({vec.nbytes / 2**30:.2f} GB) in {path}")
            with os.fdopen(fd, 'wb') as fh:
                np.save(fh, vec, allow_pickle=False)
            mapped = np.load(path, mmap_mode='r')          # the header only: dtype, shape, offset
            self._dtype, self._shape, self._offset = mapped.dtype, mapped.shape, int(mapped.offset)
            del mapped
            self._path, path = path, None
        except Exception as e:                    # no space, unwritable, ...: the solve goes on
            self.reason = f'{type(e).__name__}: {e}'
            logger.warning(f"Could not park {what} in {directory} ({self.reason}); the solve goes on, and its "
                           f"true residual |b - A x| is recorded as NaN")
        finally:
            if path is not None:                  # a partial file (an error, an interrupt)
                with contextlib.suppress(OSError):
                    os.remove(path)

    @property
    def available(self) -> bool:
        """Whether the copy is on disk, readable."""
        return self._path is not None

    @property
    def path(self) -> str | None:
        """The file of the copy (None when it is not available)."""
        return self._path

    @property
    def shape(self) -> tuple:
        """The vector's shape (None when it is not available)."""
        return self._shape

    @property
    def dtype(self):
        """The vector's dtype (None when it is not available)."""
        return self._dtype

    def __len__(self):
        return int(self._shape[0])

    def __getitem__(self, key) -> np.ndarray:
        """The elements ``key`` (a slice of step 1) as a new array, read through a memory map of
        just those elements, unmapped before returning."""
        if self._path is None:
            raise ValueError(f"the parked vector is not available ({self.reason or 'discarded'})")
        if not isinstance(key, slice):
            raise TypeError("a parked vector is read by slices")
        start, stop, step = key.indices(len(self))
        if step != 1:
            raise ValueError("a parked vector is read by slices of step 1")
        if stop <= start:
            return np.empty(0, dtype=self._dtype)
        piece = np.memmap(self._path, dtype=self._dtype, mode='r', shape=(stop - start,),
                          offset=self._offset + start * self._dtype.itemsize)
        out = np.array(piece)                     # a copy; dropping the map unmaps it
        del piece
        return out

    def discard(self):
        """Remove the copy."""
        if self._path is not None:
            with contextlib.suppress(OSError):
                os.remove(self._path)
            self._path = None


_HOST = socket.gethostname().replace('_', '-') or 'host'
_PARKED = re.compile(r'^' + re.escape(ParkedVector.PREFIX) + r'(?P<host>[^_]+)_(?P<pid>\d+)_.*\.npy$')


def sweep_parked(directory) -> list[str]:
    """Delete the parked vectors (:class:`ParkedVector`) that dead processes of this machine left in
    ``directory`` (a killed run); returns their paths."""
    from ..io.atomic import _alive
    removed = []
    try:
        names = os.listdir(directory)
    except OSError:
        return removed
    for name in names:
        m = _PARKED.match(name)
        if m is None or m['host'] != _HOST or _alive(int(m['pid'])):
            continue
        path = os.path.join(directory, name)
        with contextlib.suppress(FileNotFoundError):
            os.remove(path)
            removed.append(path)
    if removed:
        logger.info(f"Removed {len(removed)} parked vector(s) of dead processes from {directory}")
    return removed
