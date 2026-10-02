"""Run records: what an action was asked, what it ran, and what it made.

Every action of a :class:`~selfcal.run.field.Field` writes
``<field>/records/<action>_<YYYYmmdd-HHMMSS>_<pid>.json`` (its log, when there is one, has the
same stem under ``logs/``): the resolved settings (every default expanded), the run configs they
lowered to, the code version, the packages, the products and the outcome. The record is written
when the action starts and rewritten when it ends.
"""
from __future__ import annotations

import datetime
import json
import os
import platform
import socket
import sys
import time

from ..config.base import Config, encode

__all__ = ['Record', 'code_version']

SCHEMA = 1


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


def _encode_cfg(cfg) -> dict:
    """A run config in JSON form (objects encoded as settings, functions by reference)."""
    from dataclasses import fields
    out = {}
    for f in fields(cfg):
        v = getattr(cfg, f.name)
        if f.name == 'instrument' and not isinstance(v, (str, type(None))):
            v = {'object': repr(v.inst) if hasattr(v, 'inst') else repr(getattr(v, 'camera', v))}
        out[f.name] = encode(v)
    return out


class Record:
    """The JSON record of one action (see the module docstring)."""

    def __init__(self, field, action, settings: dict, lowered=(), log_path=None):
        self.started = time.time()
        stamp = datetime.datetime.fromtimestamp(self.started).strftime('%Y%m%d-%H%M%S')
        self.stem = f'{action}_{stamp}_{os.getpid()}'
        self.path = os.path.join(field.path, 'records', f'{self.stem}.json')
        main = sys.modules.get('__main__')
        self.data = {
            'selfcal_record': SCHEMA, 'action': action,
            'started': datetime.datetime.fromtimestamp(self.started).isoformat(timespec='seconds'),
            'host': socket.gethostname(), 'pid': os.getpid(), 'cwd': os.getcwd(),
            'script': os.path.abspath(main.__file__) if getattr(main, '__file__', None) else None,
            'argv': list(sys.argv), 'code': code_version(), 'packages': _packages(),
            'environment': {k: v for k, v in os.environ.items()
                            if k.startswith('SELFCAL_') or k.endswith('_NUM_THREADS')},
            'field': field.to_dict(),
            'settings': {k: (v.to_dict() if isinstance(v, Config) else encode(v)) for k, v in settings.items()},
            'lowered': [_encode_cfg(low.cfg) for low in lowered],
            'log': log_path, 'status': 'running', 'products': {}, 'wall_s': None, 'error': None,
        }
        self.write()

    def write(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = f'{self.path}.tmp'
        with open(tmp, 'w') as f:
            json.dump(self.data, f, indent=1, default=str)
        os.replace(tmp, self.path)

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
