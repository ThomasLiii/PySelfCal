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
"""
import os
import shutil
import tempfile
from concurrent.futures import ThreadPoolExecutor

import numpy as np


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
    print(f"Spilling pixel state ({n_bytes/2**30:.1f} GB) to {spill_dir}"
          f"{' ' + label if label else ''}...", flush=True)
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

    def discard(self):
        """Drop the scratch dir without reading it (caller never needed it)."""
        if self.spill_dir is not None:
            shutil.rmtree(self.spill_dir, ignore_errors=True)
            self.spill_dir = None

    def __repr__(self):
        return f"<PixelSpill {self.spill_dir!r} J={self.num_sky_blocks}>"


def restore_pixel_state(spill_dir, cleanup=True):
    """Reload what :func:`spill_pixel_state` wrote → (counts, fisher, cross).

    Missing files come back as None; the ``pixel_cross`` dict is rebuilt in
    its original insertion order.
    """
    def _load(name):
        p = os.path.join(spill_dir, name)
        return np.load(p) if os.path.exists(p) else None

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
