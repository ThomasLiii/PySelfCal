"""Spill/restore for large write-once arrays that idle through a hot phase.

The setup products ``pixel_counts`` / ``pixel_fisher`` / ``pixel_cross`` are
written during Phase 1, consumed by the Phase-2 constraint builders and the
Top-2 compaction, and then not read again until ``save_calibration`` — yet
they are among the largest things resident during BOTH the setup peak (the
Phase-3/4/6 CSR build) and the entire LSQR solve. At full-NEP multiline J=4
scale that is ~17 GB of pure ballast in each.

``np.save``/``np.load`` round-trip int64/float64 arrays losslessly, so
parking them on scratch disk and reloading is byte-identical to never having
spilled — the arrays and every downstream float accumulation are unchanged.
``pixel_cross`` is a dict whose ITERATION ORDER matters (save-time consumers
walk ``items()`` and their float accumulation order must not change), so the
key order is recorded and reproduced exactly.

Spilling only triggers above ``$SELFCAL_SPILL_MIN_GB`` (default 4 GB) so
small runs pay no I/O at all; at production scale a round trip is ~30 s
against a multi-hour tile. ``$SELFCAL_SPILL_DIR`` overrides the location.
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
