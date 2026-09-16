"""Record a run's console output in a log file.

``start_run_log`` duplicates everything written to this process's stdout and
stderr into a file while it still reaches the terminal. It works at the file
descriptor level: fds 1 and 2 are pointed at a pipe read by a ``tee`` process,
so prints from worker processes (forked pools, the forkserver) are captured as
well, and no thread is added to a process that later forks worker pools. The
``tee`` runs in its own session, so Ctrl-C in the terminal does not kill it
before the traceback is written.
"""
import atexit
import datetime
import fcntl
import os
import shlex
import socket
import subprocess
import sys
import termios
import time

_active = None


def default_log_path(cfg, now=None):
    """``<output_dir>/<run_name>/logs/<task>_<YYYYmmdd-HHMMSS>_<pid>.log``.

    Falls back to ``<cache_dir>/logs/`` for tasks without a run folder; None if
    neither is configured.
    """
    run_name = cfg.resolved_run_name()
    if cfg.output_dir and run_name:
        base = os.path.join(cfg.output_dir, run_name, 'logs')
    elif getattr(cfg, 'cache_dir', None):
        base = os.path.join(cfg.cache_dir, 'logs')
    else:
        return None
    stamp = (now or datetime.datetime.now()).strftime('%Y%m%d-%H%M%S')
    return os.path.join(base, f'{cfg.task}_{stamp}_{os.getpid()}.log')


def _git_description(repo):
    try:
        sha = subprocess.run(['git', '-C', repo, 'rev-parse', '--short', 'HEAD'], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(['git', '-C', repo, 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, timeout=5).stdout.strip()
        branch = subprocess.run(['git', '-C', repo, 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True,
                                text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return 'unknown (git unavailable)'
    if not sha:
        return 'unknown (not a git checkout)'
    return f"{sha} on {branch}" + (f" with {len(dirty.splitlines())} modified tracked file(s)" if dirty else "")


def _command_line():
    """The command as launched: ``python -m pkg.mod ...`` for module runs."""
    spec = getattr(sys.modules.get('__main__'), '__spec__', None)
    head = [sys.executable, '-m', spec.name] if spec is not None and spec.name else [sys.executable, sys.argv[0]]
    return shlex.join(head + sys.argv[1:])


def _header(config_path, repo):
    lines = [
        f"[runlog] started {datetime.datetime.now().isoformat(timespec='seconds')} on {socket.gethostname()} "
        f"(pid {os.getpid()})",
        f"[runlog] command: {_command_line()}",
        f"[runlog] cwd: {os.getcwd()}",
    ]
    if repo:
        lines.append(f"[runlog] code: {_git_description(repo)} ({repo})")
    if config_path:
        lines.append(f"[runlog] config: {os.path.abspath(config_path)}")
        try:
            with open(config_path) as f:
                text = f.read()
            lines += ["[runlog] ---- config file ----", text.rstrip('\n'), "[runlog] ---- end config ----"]
        except OSError as e:
            lines.append(f"[runlog] (could not read config: {e})")
    return '\n'.join(lines) + '\n'


class RunLog:
    def __init__(self, path, tee, saved_fds, drain_fd):
        self.path = path
        self._tee = tee
        self._saved = saved_fds
        self._drain = drain_fd
        self._stopped = False

    def stop(self, drain_timeout=5.0):
        """Point fds 1/2 back at the original targets and let ``tee`` catch up.

        Waits only until the pipe is empty, not for EOF: a lingering worker
        (e.g. the forkserver, which exits after this process) may still hold
        the write end, and ``tee`` exits by itself once it lets go.
        """
        if self._stopped:
            return
        self._stopped = True
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except (OSError, ValueError):
                pass
        os.dup2(self._saved[0], 1)
        os.dup2(self._saved[1], 2)
        deadline = time.monotonic() + drain_timeout
        buf = bytearray(4)
        while time.monotonic() < deadline:
            try:
                fcntl.ioctl(self._drain, termios.FIONREAD, buf)
            except OSError:
                break
            if int.from_bytes(buf, sys.byteorder) == 0:
                break
            time.sleep(0.01)
        time.sleep(0.02)  # tee has read the last chunk; give it a moment to write it out
        for fd in (self._drain, *self._saved):
            try:
                os.close(fd)
            except OSError:
                pass


def start_run_log(path, config_path=None, repo=None):
    """Start teeing stdout+stderr (this process and its children) into ``path``.

    Returns the ``RunLog`` (also stopped automatically at interpreter exit), or
    None if the log cannot be set up — the run then continues without one.
    """
    global _active
    if _active is not None:
        return _active
    try:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, 'a') as f:
            f.write(_header(config_path, repo))
    except OSError as e:
        print(f"[run] WARNING: cannot write log file {path}: {e}; continuing without a log", flush=True)
        return None
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except (OSError, ValueError):
            pass
    saved = (os.dup(1), os.dup(2))
    r, w = os.pipe()
    try:
        tee = subprocess.Popen(['tee', '-a', path], stdin=r, stdout=saved[0], stderr=subprocess.DEVNULL,
                               start_new_session=True)
    except OSError as e:
        for fd in (r, w, *saved):
            os.close(fd)
        print(f"[run] WARNING: cannot start tee for the log ({e}); continuing without a log", flush=True)
        return None
    drain = os.dup(r)
    os.close(r)
    os.dup2(w, 1)
    os.dup2(w, 2)
    os.close(w)
    # Line-buffer the Python streams so the log (and a redirected console)
    # stays current; the bytes written are unchanged.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, 'reconfigure', None)
        if reconfigure is not None:
            try:
                reconfigure(line_buffering=True)
            except (OSError, ValueError):
                pass
    _active = RunLog(path, tee, saved, drain)
    atexit.register(_active.stop)
    return _active
