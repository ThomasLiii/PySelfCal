"""Functions passed as settings, and the worker processes that call them.

The pipeline's worker processes are started with ``forkserver``: they do not inherit the
parent's memory, so every function they call must be importable by name, and they import the
run script again (only its declarations run; its work sits under ``if __name__ ==
"__main__":``). A function given to a setting is therefore checked when the setting is built,
and recorded by its import path ``"module:qualname"``:

* a function at the top level of a module: recorded as is;
* a function at the top level of the running script: recorded under the script's file stem
  (``tour.py`` -> ``"tour:above_80"``), which also names it in the cal file, so the same run
  launched another way records the same name. The parent never runs the script twice (the
  stem is an alias of the running script), nor does a worker (:func:`import_module`);
* a lambda, a function defined inside another function, a function defined under the
  ``__main__`` guard, or one defined in a notebook cell: rejected, with the fix; or wrapped in
  :class:`by_value`, which sends the function itself (needs ``cloudpickle``).
"""
from __future__ import annotations

import ast
import functools
import hashlib
import importlib
import inspect
import os
import pickle
import sys
import types

from .base import ConfigError

__all__ = ['function_ref', 'load_callable', 'import_module', 'describe_callable', 'check_picklable', 'by_value']

_MAIN_NAMES = ('__main__', '__mp_main__')


def _running_script(module_name='__main__'):
    """The module object of the running script and its file stem, or (module, None) when the
    module has no file (an interactive session, a notebook). ``module_name`` is the module the
    function lives in: ``__main__``, or ``__mp_main__`` in a worker process, where
    multiprocessing imports the script under that name and ``__main__`` is its own bootstrap."""
    main = sys.modules.get(module_name)
    path = getattr(main, '__file__', None)
    if not path:
        return main, None
    return main, os.path.splitext(os.path.basename(path))[0]


def import_module(name):
    """``importlib.import_module(name)``, except that a name equal to the running script's file
    stem returns the running script itself (in a worker process, the copy multiprocessing
    imported as ``__mp_main__``) instead of importing and running the file a second time."""
    module = sys.modules.get(name)
    if module is not None:
        return module
    for main_name in _MAIN_NAMES:
        main = sys.modules.get(main_name)
        path = getattr(main, '__file__', None)
        if path and os.path.splitext(os.path.basename(path))[0] == name:
            sys.modules[name] = main
            return main
    return importlib.import_module(name)


def load_callable(ref: str):
    """The object ``"module:qualname"`` names (see :func:`import_module` for script functions)."""
    module, sep, qual = str(ref).partition(':')
    if not sep or not module or not qual:
        raise ValueError(f"function reference {ref!r} must look like 'package.module:name'")
    obj = import_module(module)
    for part in qual.split('.'):
        obj = getattr(obj, part)
    return obj


# --------------------------------------------------------------------------- the script's top level
_TOP_LEVEL: dict[tuple, set] = {}


def _is_main_guard(test) -> bool:
    if not (isinstance(test, ast.Compare) and len(test.ops) == 1 and isinstance(test.ops[0], ast.Eq)):
        return False
    sides = [test.left, test.comparators[0]]
    names = [s for s in sides if isinstance(s, ast.Name) and s.id == '__name__']
    consts = [s for s in sides if isinstance(s, ast.Constant) and s.value == '__main__']
    return bool(names and consts)


def _bound_names(body, out):
    """Names a module body binds when it is imported (the ``__main__`` guard's body excluded)."""
    for node in body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        out.add(n.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                out.add((alias.asname or alias.name).split('.')[0])
        elif isinstance(node, ast.If):
            if not _is_main_guard(node.test):
                _bound_names(node.body, out)
            _bound_names(node.orelse, out)
        elif isinstance(node, (ast.Try, ast.With, ast.For, ast.While)):
            for field in ('body', 'orelse', 'finalbody'):
                _bound_names(getattr(node, field, []) or [], out)
            for handler in getattr(node, 'handlers', []) or []:
                _bound_names(handler.body, out)
    return out


def _top_level_names(path) -> set:
    try:
        key = (path, os.path.getmtime(path))
    except OSError:
        return set()
    if key not in _TOP_LEVEL:
        with open(path) as f:
            tree = ast.parse(f.read(), filename=path)
        _TOP_LEVEL[key] = _bound_names(tree.body, set())
    return _TOP_LEVEL[key]


# --------------------------------------------------------------------------- references
def function_ref(fn, what='function') -> str:
    """The import path ``"module:qualname"`` worker processes use to find ``fn``.

    Raises :class:`~selfcal.config.base.ConfigError` naming ``what`` and the fix when ``fn``
    cannot be imported by a worker: a lambda, a nested function or closure, a function defined
    under the script's ``__main__`` guard or in a notebook cell. A function at the top level of
    the running script is recorded under the script's file stem (see the module docstring).
    """
    if isinstance(fn, functools.partial):
        raise ConfigError(f"{what}: a functools.partial is not a reference; pass the function and its "
                          f"keyword arguments to sc.Function(fn, of=..., **params)")
    module = getattr(fn, '__module__', None)
    qual = getattr(fn, '__qualname__', None) or getattr(fn, '__name__', None)
    if module is None or qual is None:
        raise ConfigError(f"{what}: {fn!r} has no importable name; wrap a Python function defined with "
                          f"`def` at the top level of a module")
    if getattr(fn, '__name__', '') == '<lambda>':
        raise ConfigError(f"{what}: a lambda cannot be sent to the worker processes; define it with "
                          f"`def` at the top level of a module or of the run script")
    if '<locals>' in qual:
        raise ConfigError(f"{what}: {qual.replace('.<locals>', '')} is defined inside another function, "
                          f"so the worker processes cannot import it; move it to the top level of a "
                          f"module or of the run script")
    if module in _MAIN_NAMES:
        main, stem = _running_script(module)
        if stem is None:
            raise ConfigError(
                f"{what}: {qual} is defined in a notebook cell or an interactive session, which the "
                f"worker processes cannot import. Write it to a module and import it (in a notebook: "
                f"`%%writefile myfuncs.py` in a cell of its own, then `from myfuncs import {qual}`)")
        if not stem.isidentifier():
            raise ConfigError(f"{what}: the run script's file name {stem!r} is not a Python module name "
                              f"(rename it, e.g. {stem.replace('-', '_')!r}.py) so that the worker "
                              f"processes can import {qual} from it")
        top = qual.split('.')[0]
        if top not in _top_level_names(main.__file__):
            raise ConfigError(
                f"{what}: {qual} is defined under `if __name__ == \"__main__\":` in "
                f"{os.path.basename(main.__file__)}; the worker processes import the script without "
                f"running that block, so they would not find it. Move the definition above the guard")
        other = sys.modules.get(stem)
        if other is not None and other is not main:
            raise ConfigError(f"{what}: the run script's name {stem!r} is also the name of the module "
                              f"{getattr(other, '__file__', other)!r}; rename the script")
        sys.modules[stem] = main              # the stem names the running script, never a re-run
        return f'{stem}:{qual}'
    try:
        obj = load_callable(f'{module}:{qual}')
    except Exception as e:
        raise ConfigError(f"{what}: {module}:{qual} cannot be imported ({type(e).__name__}: {e})") from None
    if obj is not fn and obj != fn:
        raise ConfigError(f"{what}: {module}:{qual} names another object than the one given (a function "
                          f"replaced by a decorator?); pass the function the module defines")
    return f'{module}:{qual}'


def describe_callable(obj) -> str:
    """A best-effort ``"module:qualname"`` of a function, or of an object's class (never raises)."""
    if isinstance(obj, by_value):
        return f'by_value:{obj.__qualname__}'
    target = obj if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType, type)) else type(obj)
    module = getattr(target, '__module__', None) or '?'
    qual = getattr(target, '__qualname__', None) or getattr(target, '__name__', None) or repr(target)
    if module in _MAIN_NAMES:
        _, stem = _running_script(module)
        module = stem or module
    return f'{module}:{qual}'


def check_picklable(obj, what):
    """Raise :class:`~selfcal.config.base.ConfigError` when ``obj`` cannot be sent to a worker process."""
    if isinstance(obj, (types.FunctionType, types.BuiltinFunctionType)):
        function_ref(obj, what)
        return
    function_ref(type(obj), what)
    try:
        pickle.loads(pickle.dumps(obj))
    except Exception as e:
        raise ConfigError(f"{what}: {type(obj).__name__} cannot be sent to the worker processes "
                          f"({type(e).__name__}: {e}); keep only importable functions and plain "
                          f"values in it") from None


def _code_digest(fn, source, fallback):
    """What a function computes, the same in every process: the hash of its source, its default and
    keyword-default values and the values its closure holds. (Its cloudpickle blob also holds the
    file it was compiled from, which changes with every notebook kernel and every way of launching
    the script.) ``fallback`` when the source cannot be read."""
    if source is None:
        return fallback
    import json

    from .base import encode
    cells = []
    for cell in getattr(fn, '__closure__', None) or ():
        try:
            cells.append(cell.cell_contents)
        except ValueError:                 # an empty cell
            cells.append(None)
    parts = {'source': source, 'defaults': encode(list(getattr(fn, '__defaults__', None) or ())),
             'kwdefaults': encode(dict(getattr(fn, '__kwdefaults__', None) or {})), 'closure': encode(cells)}
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=repr).encode()).hexdigest()


class by_value:
    """A function sent to the worker processes by value instead of by name.

    For a function the workers cannot import, typically one defined in a notebook cell::

        def ratio(wavelength, scale=2.0):
            return wavelength / scale

        model = sc.Model(sky=[sc.Sky(), sc.Sky("line", times=sc.by_value(ratio))])

    The function is serialised with the optional package ``cloudpickle`` (``pip install
    cloudpickle``), together with what it refers to; its source is kept in the run's record. A
    run with such a function cannot be submitted or rerun from its record: write the function to a
    module when the run should be repeatable.
    """

    def __init__(self, fn):
        try:
            import cloudpickle
        except ImportError:
            raise ConfigError("sc.by_value(...) needs the optional package cloudpickle (pip install cloudpickle); "
                              "or write the function to a module and import it") from None
        if isinstance(fn, by_value):
            fn = fn.function
        if not callable(fn):
            raise ConfigError(f"sc.by_value(...): expected a function, got {fn!r}")
        self._function = fn
        self._blob = cloudpickle.dumps(fn)
        self.__name__ = getattr(fn, '__name__', 'function')
        self.__qualname__ = getattr(fn, '__qualname__', self.__name__)
        try:
            self.source = inspect.getsource(fn)
        except (OSError, TypeError):
            self.source = None
        self.sha256 = hashlib.sha256(self._blob).hexdigest()
        self.digest = _code_digest(fn, self.source, self.sha256)

    @property
    def function(self):
        if self._function is None:
            self._function = pickle.loads(self._blob)
        return self._function

    @property
    def __signature__(self):
        return inspect.signature(self.function)

    def __call__(self, *args, **kwargs):
        return self.function(*args, **kwargs)

    def __getstate__(self):
        return {'_blob': self._blob, '__name__': self.__name__, '__qualname__': self.__qualname__,
                'source': self.source, 'sha256': self.sha256, 'digest': self.digest, '_function': None}

    def __eq__(self, other):
        return isinstance(other, by_value) and other.digest == self.digest

    def __hash__(self):
        return hash(self.digest)

    def __repr__(self):
        return f"sc.by_value({self.__qualname__})"
