"""Run records: what an action was asked, what it ran, and what it made.

Every action of a :class:`~selfcal.run.field.Field` writes
``<field>/records/<action>_<YYYYmmdd-HHMMSS>_<pid>.json`` (its log, when there is one, has the
same stem under ``logs/``): the resolved settings (every default expanded), the engine runs they
lowered to (:meth:`~selfcal.run.runspec.RunSpec.describe`), the code version, the packages, the
environment knobs in effect, the products and the outcome. The record is written when the action starts and rewritten when it ends.

:func:`rerun` runs a recorded action again from its record (``selfcal rerun RECORD``);
:func:`write_request` writes the same form for an action that runs elsewhere
(:meth:`~selfcal.run.field.Field.submit`).
"""
from __future__ import annotations

import datetime
import itertools
import json
import os
import platform
import socket
import sys
import time
import warnings

from ..config.base import Config, ConfigError, decode, encode
from ..io.atomic import atomic_path

__all__ = ['Record', 'code_version', 'rerun', 'load_action', 'write_request', 'script_path']

SCHEMA = 1

# During a rerun: the script the record names (the process itself runs `selfcal rerun`), and the
# digest of the frames the record ran on.
_RERUN_SCRIPT = None
_RERUN_FRAMES = None


def script_path():
    """The run script of this process: the one a rerun's record names, else the main module's file
    (None in an interactive session or a notebook)."""
    if _RERUN_SCRIPT:
        return _RERUN_SCRIPT
    main = sys.modules.get('__main__')
    path = getattr(main, '__file__', None)
    return os.path.abspath(path) if path else None


def _claim(directory, stem) -> str:
    """A new file ``<directory>/<stem>.json`` (``<stem>_2.json``, ... when it exists), created now so
    that two actions of one process in the same second never share it."""
    os.makedirs(directory, exist_ok=True)
    for n in itertools.count(1):
        path = os.path.join(directory, f'{stem}.json' if n == 1 else f'{stem}_{n}.json')
        try:
            os.close(os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
            return path
        except FileExistsError:
            continue


def code_version():
    """``{commit, branch, modified}`` of the git checkout selfcal runs from (None outside one)."""
    import subprocess
    repo = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        sha = subprocess.run(['git', '-C', repo, 'rev-parse', 'HEAD'], capture_output=True, text=True,
                             timeout=5).stdout.strip()
        if not sha:
            return None
        branch = subprocess.run(['git', '-C', repo, 'rev-parse', '--abbrev-ref', 'HEAD'], capture_output=True,
                                text=True, timeout=5).stdout.strip()
        dirty = subprocess.run(['git', '-C', repo, 'status', '--porcelain', '--untracked-files=no'],
                               capture_output=True, text=True, timeout=5).stdout.split('\n')
    except (OSError, subprocess.SubprocessError):
        return None
    return {'commit': sha, 'branch': branch, 'modified': [line[3:] for line in dirty if line.strip()],
            'repository': repo}


def _packages():
    out = {'python': platform.python_version()}
    for name in ('selfcal', 'numpy', 'scipy', 'astropy', 'h5py'):
        module = sys.modules.get(name)
        out[name] = getattr(module, '__version__', None) if module is not None else None
    return out


def _functions(obj, out):
    """The user functions a settings object refers to: ``{"module:qualname": {"file", "sha256"}}``,
    the module's file and the hash of its source (functions of selfcal itself are covered by the
    code version)."""
    import dataclasses
    import hashlib
    import types

    from ..config.functions import by_value, function_ref
    if isinstance(obj, Config):
        for f in dataclasses.fields(obj):
            _functions(getattr(obj, f.name), out)
    elif isinstance(obj, (tuple, list)):
        for x in obj:
            _functions(x, out)
    elif isinstance(obj, dict):
        for x in obj.values():
            _functions(x, out)
    elif isinstance(obj, by_value):
        out[f'by_value:{obj.__qualname__}'] = {'file': None, 'sha256': obj.sha256, 'source': obj.source}
    elif isinstance(obj, types.FunctionType) or (callable(obj) and not isinstance(obj, type)
                                                 and hasattr(type(obj), '__module__')):
        target = obj if isinstance(obj, types.FunctionType) else type(obj)
        try:
            ref = function_ref(target)
        except Exception:
            return out
        if ref.startswith('selfcal.') or ref in out:
            return out
        module = sys.modules.get(getattr(target, '__module__', ''), None)
        path = getattr(module, '__file__', None)
        digest = None
        if path and os.path.exists(path):
            with open(path, 'rb') as f:
                digest = hashlib.sha256(f.read()).hexdigest()
        out[ref] = {'file': path, 'sha256': digest}
    return out


class Record:
    """The JSON record of one action (see the module docstring)."""

    def __init__(self, field, action, settings: dict, lowered=(), log_path=None, frames=None):
        self.started = time.time()
        stamp = datetime.datetime.fromtimestamp(self.started).strftime('%Y%m%d-%H%M%S')
        self.path = _claim(os.path.join(field.path, 'records'), f'{action}_{stamp}_{os.getpid()}')
        self.stem = os.path.basename(self.path)[:-len('.json')]
        self.data = {
            'selfcal_record': SCHEMA, 'action': action,
            'started': datetime.datetime.fromtimestamp(self.started).isoformat(timespec='seconds'),
            'host': socket.gethostname(), 'pid': os.getpid(), 'cwd': os.getcwd(),
            'script': script_path(), 'frames': frames,
            'argv': list(sys.argv), 'code': code_version(), 'packages': _packages(),
            'environment': {k: v for k, v in os.environ.items()
                            if k.startswith('SELFCAL_') or k.endswith('_NUM_THREADS')},
            'field': field.to_dict(),
            'settings': {k: (v.to_dict() if isinstance(v, Config) else encode(v)) for k, v in settings.items()},
            'functions': _functions(list(settings.values()) + [field], {}),
            'lowered': [spec.describe() for spec in lowered],
            'log': log_path, 'status': 'running', 'products': {}, 'wall_s': None, 'error': None,
        }
        self.write()

    def write(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with atomic_path(self.path) as tmp:
            with open(tmp, 'w') as f:
                json.dump(self.data, f, indent=1, default=str)

    def finish(self, products=None, error=None):
        self.data['wall_s'] = round(time.time() - self.started, 3)
        self.data['finished'] = datetime.datetime.now().isoformat(timespec='seconds')
        if products is not None:
            self.data['products'] = products
        if error is not None:
            self.data['status'] = 'failed'
            self.data['error'] = f'{type(error).__name__}: {error}'
        else:
            self.data['status'] = 'done'
        self.write()

    @staticmethod
    def read(path) -> dict:
        """A record's contents."""
        with open(path) as f:
            return json.load(f)


def write_request(field, action, settings) -> str:
    """Write the JSON request of an action that runs elsewhere (:meth:`Field.submit`): the record's
    form without an outcome. Returns its path."""
    stamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    path = _claim(os.path.join(field.path, 'records'), f'submit_{action}_{stamp}_{os.getpid()}')
    data = {'selfcal_record': SCHEMA, 'action': action, 'status': 'submitted',
            'started': datetime.datetime.now().isoformat(timespec='seconds'), 'host': socket.gethostname(),
            'cwd': os.getcwd(), 'script': script_path(),
            'code': code_version(), 'field': field.to_dict(),
            'settings': {k: (v.to_dict() if isinstance(v, Config) else encode(v)) for k, v in settings.items()}}
    with atomic_path(path) as tmp:
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=1, default=str)
    return path


def load_action(path):
    """``(field, action, settings)`` of a record or a request, rebuilt from its JSON. Functions
    of the run script are found through the script's directory, which is put on ``sys.path``."""
    data = Record.read(path)
    if data.get('selfcal_record') != SCHEMA:
        raise ConfigError(f"{path}: not a selfcal record (schema {data.get('selfcal_record')!r})")
    script = data.get('script')
    if script and os.path.dirname(script) not in sys.path:
        sys.path.insert(0, os.path.dirname(script))
    if data.get('cwd') and os.path.isdir(data['cwd']):
        os.chdir(data['cwd'])                  # relative paths in the settings are relative to it
    field = decode(data['field'])
    settings = {k: decode(v) for k, v in data['settings'].items()}
    return field, data['action'], settings, data


def rerun(path, *, overwrite=False):
    """Run the action of a record (or a submitted request) again, with the settings it recorded.
    The products it would make that already exist and are current are reused unless
    ``overwrite``, which makes them again (for a reprojection: the frames). Warns when the code
    differs from the record's, and when the frames it finds are not the ones the record ran on.
    The caller's working directory and ``sys.path`` are restored afterwards."""
    global _RERUN_SCRIPT, _RERUN_FRAMES
    cwd, sys_path = os.getcwd(), list(sys.path)
    try:
        field, action, settings, data = load_action(path)
        then, now = data.get('code') or {}, code_version() or {}
        if then.get('commit') != now.get('commit') or then.get('modified') or now.get('modified'):
            warnings.warn(f"{path}: recorded with code {then.get('commit', '?')[:10]}"
                          f"{' (modified)' if then.get('modified') else ''}, running {now.get('commit', '?')[:10]}"
                          f"{' (modified)' if now.get('modified') else ''}: the products may differ", stacklevel=2)
        _RERUN_SCRIPT, _RERUN_FRAMES = data.get('script'), data.get('frames')
        if action == 'reproject':
            settings['replace'] = bool(overwrite)
            return field.reproject(settings.pop('exposures'), **settings)
        if action in ('calibrate', 'mosaic'):
            settings['overwrite'] = bool(overwrite)
            run = field.calibrate if action == 'calibrate' else field.mosaic
            return run(settings.pop('recipe'), **settings)
        raise ConfigError(f"{path}: cannot rerun the action {action!r}")
    finally:
        _RERUN_SCRIPT = _RERUN_FRAMES = None
        os.chdir(cwd)
        sys.path[:] = sys_path


def check_rerun_frames(frames_digest):
    """In a rerun, warn when the frames the action found are not the ones its record ran on."""
    if _RERUN_FRAMES and frames_digest != _RERUN_FRAMES:
        warnings.warn(f"the record ran on {_RERUN_FRAMES.get('n')} frames ({str(_RERUN_FRAMES.get('sha256'))[:12]}), "
                      f"this rerun finds {frames_digest.get('n')} ({str(frames_digest.get('sha256'))[:12]}): "
                      f"the frame directory changed", stacklevel=3)
