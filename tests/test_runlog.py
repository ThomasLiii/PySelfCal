"""The run log (selfcal_scripts/runner/runlog.py).

A child Python process starts the log, then writes through every path a run
uses — print, logging, stderr, a forked worker, a forkserver worker, a raw fd
write — and dies on an uncaught exception. The log must hold all of it
(traceback included, header first), the console must still receive it, and the
process must not hang at exit while the forkserver still holds the pipe.

Runnable as ``python tests/test_runlog.py`` or under pytest.
"""
import os
import subprocess
import sys
import tempfile
import textwrap
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CHILD = textwrap.dedent('''
    import logging, multiprocessing as mp, os, sys
    sys.path.insert(0, {repo!r})
    from selfcal_scripts.runner.runlog import start_run_log

    def work(tag):
        print(f"worker-print {{tag}} pid {{os.getpid()}}")
        sys.stderr.write(f"worker-stderr {{tag}}\\n")
        return tag

    if __name__ == "__main__":
        logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
        log = start_run_log({path!r}, config_path={cfg!r}, repo={repo!r})
        assert log is not None
        print("main-print")
        logging.getLogger("x").info("main-logging")
        print("main-stderr", file=sys.stderr)
        os.write(1, b"raw-fd-write\\n")
        with mp.get_context("fork").Pool(2) as p:
            assert sorted(p.map(work, ["fork-a", "fork-b"])) == ["fork-a", "fork-b"]
        ctx = mp.get_context("forkserver")
        with ctx.Pool(2) as p:
            assert sorted(p.map(work, ["fs-a", "fs-b"])) == ["fs-a", "fs-b"]
        print("last-line-before-error")
        raise RuntimeError("deliberate failure for the log test")
''')


def test_run_log_captures_everything_and_exits_promptly():
    tmp = tempfile.mkdtemp(prefix='selfcal_runlog_')
    log_path = os.path.join(tmp, 'logs', 'cal_test.log')
    cfg_path = os.path.join(tmp, 'demo.toml')
    with open(cfg_path, 'w') as f:
        f.write('task = "cal"\nrun_name = "demo"\n')
    script = os.path.join(tmp, 'child.py')
    with open(script, 'w') as f:
        f.write(CHILD.format(repo=REPO, path=log_path, cfg=cfg_path))
    t0 = time.time()
    proc = subprocess.run([sys.executable, script], capture_output=True, text=True, timeout=120)
    elapsed = time.time() - t0
    assert proc.returncode != 0
    log = open(log_path).read()
    expected = ['main-print', 'main-logging', 'main-stderr', 'raw-fd-write',
                'worker-print fork-a', 'worker-print fork-b', 'worker-stderr fork-a',
                'worker-print fs-a', 'worker-print fs-b', 'worker-stderr fs-b',
                'last-line-before-error', 'Traceback (most recent call last)',
                'RuntimeError: deliberate failure for the log test']
    for token in expected:
        assert token in log, f"log is missing {token!r}"
    # header first, with the config text
    assert log.startswith('[runlog] started ')
    assert '[runlog] ---- config file ----' in log and 'run_name = "demo"' in log
    assert log.index('[runlog] ---- end config ----') < log.index('main-print')
    # the console still sees everything (stdout and stderr both reach the original stdout now)
    console = proc.stdout + proc.stderr
    for token in expected:
        assert token in console, f"console is missing {token!r}"
    assert elapsed < 30, f"process took {elapsed:.1f} s to exit (tee drain should not wait for EOF)"


def test_header_shows_module_launch():
    tmp = tempfile.mkdtemp(prefix='selfcal_runlog_mod_')
    pkg = os.path.join(tmp, 'tfpkg')
    os.makedirs(pkg)
    open(os.path.join(pkg, '__init__.py'), 'w').close()
    log_path = os.path.join(tmp, 'm.log')
    with open(os.path.join(pkg, 'entry.py'), 'w') as f:
        f.write(f"import sys\nsys.path.insert(0, {REPO!r})\n"
                "from selfcal_scripts.runner.runlog import start_run_log\n"
                f"start_run_log({log_path!r})\nprint('module-body')\n")
    env = dict(os.environ, PYTHONPATH=tmp)
    proc = subprocess.run([sys.executable, '-m', 'tfpkg.entry', '--flag', 'x y'], capture_output=True, text=True,
                          timeout=60, env=env, cwd=tmp)
    assert proc.returncode == 0, proc.stderr
    log = open(log_path).read()
    assert f"[runlog] command: {sys.executable} -m tfpkg.entry --flag 'x y'" in log, log[:300]
    assert 'module-body' in log


def test_default_log_path_layout():
    from selfcal_scripts.runner.config import RunConfig
    import datetime
    from selfcal_scripts.runner.runlog import default_log_path
    cfg = RunConfig(task='cal', output_dir='/out', run_name='SPHEREx_det{detector}', instrument_cfg={'detector': 3})
    p = default_log_path(cfg, now=datetime.datetime(2026, 9, 16, 10, 15, 0))
    assert p == f"/out/SPHEREx_det3/logs/cal_20260916-101500_{os.getpid()}.log"
    cfg2 = RunConfig(task='precompute', cache_dir='/c/')
    assert default_log_path(cfg2).startswith('/c/logs/precompute_')
    assert default_log_path(RunConfig(task='precompute', cache_dir=None)) is None


if __name__ == '__main__':
    sys.path.insert(0, REPO)
    test_run_log_captures_everything_and_exits_promptly()
    print("OK run log captures main/worker/fd output + traceback; console unchanged; prompt exit")
    test_header_shows_module_launch()
    print("OK header renders python -m launches")
    test_default_log_path_layout()
    print("OK default log path layout")
