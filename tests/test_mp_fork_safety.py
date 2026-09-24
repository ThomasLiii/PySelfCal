"""Regression tests for the process-pool fork hazard (2026-09-09 production
hang): a pool forked from a multi-threaded parent inherits locks held by
other threads; a child that inherits the stderr buffer lock hangs at its own
exit. Pools must use the forkserver context (selfcal.core.shmbuf), and shared
arrays must travel as explicit SharedBuffer handles, not by inheritance.

The fork variant is NOT exercised by default (it hangs by design);
SELFCAL_TEST_FORK_HAZARD=1 runs it to demonstrate the mechanism.
"""
import concurrent.futures as cf
import os
import sys
import threading
import time

import numpy as np
import pytest

from selfcal.core.shmbuf import SharedBuffer, worker_pool_context


def _write_slice(args):
    buf, k = args
    buf.array[k * 10:(k + 1) * 10] = k + 1
    buf.close()
    return os.getpid()


def _noop(k):
    return k


class _Chatter:
    """A daemon thread printing to stderr continuously (worst case for fork)."""
    def __enter__(self):
        self.stop = threading.Event()
        self.saved = sys.stderr
        sys.stderr = open(os.devnull, 'w')
        def run():
            while not self.stop.is_set():
                print("x" * 2000, file=sys.stderr, flush=True)
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()
        return self
    def __exit__(self, *a):
        self.stop.set()
        self.t.join(timeout=5)
        sys.stderr.close()
        sys.stderr = self.saved


def test_sharedbuffer_roundtrip_forkserver():
    buf = SharedBuffer(80, np.int32, 'test')
    with cf.ProcessPoolExecutor(max_workers=4, mp_context=worker_pool_context()) as ex:
        list(ex.map(_write_slice, [(buf, k) for k in range(8)]))
    for k in range(8):
        assert buf.array[k * 10:(k + 1) * 10].tolist() == [k + 1] * 10


def test_pool_survives_printing_thread():
    """forkserver pools must never hang under a thread that holds the stderr
    lock at pool-creation time (fork hangs 60/60 here)."""
    with _Chatter():
        for _ in range(int(os.environ.get('SELFCAL_TEST_POOL_ITERS', '8'))):
            with cf.ProcessPoolExecutor(max_workers=4, mp_context=worker_pool_context()) as ex:
                futs = [ex.submit(_noop, k) for k in range(8)]
                for f in futs:
                    f.result(timeout=60)


@pytest.mark.skipif(os.environ.get('SELFCAL_TEST_FORK_HAZARD') != '1',
                    reason="demonstrates the hang (set SELFCAL_TEST_FORK_HAZARD=1)")
def test_fork_hazard_demonstration():
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    with _Chatter():
        hung = 0
        for _ in range(10):
            p = ctx.Process(target=_noop, args=(0,))
            p.start(); p.join(timeout=4)
            if p.is_alive():
                hung += 1; p.kill(); p.join()
    assert hung > 0, "expected fork children to hang at exit under a printing thread"
