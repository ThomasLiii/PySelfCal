"""The settings base class: frozen dataclasses checked when they are built.

Every settings object of the Python API (a model term, a fit, a coadd, the machine, an
instrument) is a frozen dataclass deriving from :class:`Config`. Building one checks it:

* every keyword is a setting of the class (a misspelt one is named, with the closest
  setting: ``Fit(iteratons=100)`` -> "did you mean 'iterations'?");
* every value has the declared type, after the obvious conversions (a list to a tuple, an
  int to a float, a path to a string, a numpy scalar to a Python one), and a choice is one
  of the declared choices;
* the class's own cross-field rules (:meth:`Config._validate`).

The objects are immutable: :meth:`Config.replace` returns a changed copy, checked again.
``repr`` is the Python that rebuilds the object (only settings that differ from their
defaults are shown), and :meth:`Config.to_dict` / :meth:`Config.from_dict` give the JSON
form that run records store and products are fingerprinted by. A setting added to a class
after products were made with it is declared with :func:`added`, so that those products keep
their fingerprints.

A subclass is a plain frozen dataclass; settings after ``_: KW_ONLY`` are keyword-only::

    @dataclass(frozen=True)
    class Fit(Config):
        iterations: int = 50
        _: KW_ONLY
        clip: float | None = 5.0

        def _validate(self):
            if self.iterations < 1:
                raise ConfigError("Fit(iterations=...): at least 1")
"""
from __future__ import annotations

import collections.abc
import dataclasses
import difflib
import importlib
import os
import sys
import types
import typing

import numpy as np

__all__ = ['Config', 'ConfigError', 'FrozenDict', 'added']


class ConfigError(ValueError):
    """A setting that cannot be used. The message names the object, the setting and the fix."""


class FrozenDict(dict):
    """A dict that cannot be changed: the value of a mapping-valued setting."""

    def _immutable(self, *args, **kwargs):
        raise TypeError("settings cannot be changed in place; build a new object with .replace(...)")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = __ior__ = _immutable

    def __hash__(self):
        return hash(tuple(sorted(self.items(), key=lambda kv: repr(kv[0]))))

    def __reduce__(self):
        return (FrozenDict, (dict(self),))

    def __repr__(self):
        return repr(dict(self))


# --------------------------------------------------------------------------- class introspection
_HINTS: dict[type, dict] = {}
_NAMES: dict[type, tuple] = {}


def _hints(cls) -> dict:
    """The resolved type hints of the settings of ``cls`` (a setting whose annotation cannot be
    resolved is left out: it is not checked)."""
    if cls not in _HINTS:
        try:
            hints = typing.get_type_hints(cls)
        except Exception:
            hints = {}
            for klass in reversed(cls.__mro__):
                module = sys.modules.get(klass.__module__)
                namespace = vars(module) if module is not None else {}
                for name, ann in klass.__dict__.get('__annotations__', {}).items():
                    try:
                        hints[name] = eval(ann, namespace) if isinstance(ann, str) else ann
                    except Exception:
                        hints.pop(name, None)
        _HINTS[cls] = hints
    return _HINTS[cls]


def setting_names(cls) -> tuple:
    """The names of the settings a :class:`Config` class takes, in declaration order."""
    if cls not in _NAMES:
        _NAMES[cls] = tuple(f.name for f in dataclasses.fields(cls) if f.init)
    return _NAMES[cls]


def added(default, *, since):
    """A setting added to a class after products were made with it (``since``: the date, e.g.
    ``"2026-10-08"``): ``stop: Stop | None = added(None, since="2026-10")`` (``sc.Fit``).

    While the setting equals its ``default`` it is left out of :meth:`Config.to_dict`, and so of the
    encoding every product is fingerprinted by: the products made before it existed, and those made
    since with the default, keep their fingerprints. Set to another value, it is encoded (those
    products are made by other inputs). A record encoded before the setting existed decodes with
    the setting at its default."""
    return dataclasses.field(default=default, metadata={'since': str(since)})


def _default(f):
    if f.default is not dataclasses.MISSING:
        return f.default
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    return dataclasses.MISSING


# --------------------------------------------------------------------------- the base class
class Config:
    """Base class of the settings objects (see the module docstring)."""

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # The dataclass decorator keeps a __repr__ the class already has: give every subclass
        # this one (the Python that rebuilds the object) unless it defines its own.
        if '__repr__' not in cls.__dict__:
            cls.__repr__ = Config.__repr__

    def __new__(cls, *args, **kwargs):
        if dataclasses.is_dataclass(cls):
            names = setting_names(cls)
            for key in kwargs:
                if key not in names:
                    close = difflib.get_close_matches(key, names, n=1)
                    raise TypeError(
                        f"{cls.__name__}() got an unexpected keyword argument {key!r}"
                        + (f". Did you mean {close[0]!r}?" if close else
                           f"; its settings are {', '.join(names) or 'none'}"))
        return super().__new__(cls)

    def __post_init__(self):
        cls = type(self)
        hints = _hints(cls)
        for f in dataclasses.fields(self):
            if not f.init or f.name not in hints:
                continue
            value = getattr(self, f.name)
            new = coerce(value, hints[f.name], cls.__name__, f.name)
            if new is not value:
                object.__setattr__(self, f.name, new)
        self._validate()

    def _validate(self):
        """Cross-field rules of the class; raise :class:`ConfigError` (override)."""

    # ---- copies ----------------------------------------------------------------------------
    def replace(self, **changes):
        """A copy with ``changes`` applied, checked like a new object."""
        return dataclasses.replace(self, **changes)

    # ---- printing --------------------------------------------------------------------------
    def __repr__(self):
        cls = type(self)
        parts, keyword = [], False
        for f in dataclasses.fields(self):
            if not f.init or not f.repr:
                continue
            value = getattr(self, f.name)
            default = _default(f)
            if default is not dataclasses.MISSING and _same(value, default):
                keyword = True                 # a skipped setting: the next ones need their names
                continue
            if f.kw_only or keyword:
                parts.append(f"{f.name}={python_repr(value)}")
            else:
                parts.append(python_repr(value))
        return f"{cls.__name__}({', '.join(parts)})"

    # ---- the JSON form ---------------------------------------------------------------------------
    def to_dict(self) -> dict:
        """Every setting (defaults included, but those :func:`added` later while at their default) in
        a JSON-ready form, with the class under ``"type"``."""
        out = {'type': type_name(type(self))}
        for f in dataclasses.fields(self):
            if not f.init:
                continue
            value = getattr(self, f.name)
            if 'since' in f.metadata and _same(value, _default(f)):
                continue
            out[f.name] = encode(value)
        return out

    @staticmethod
    def from_dict(d):
        """The object :meth:`to_dict` describes (its class is imported from ``d["type"]``)."""
        return decode(d)


def _same(a, b):
    try:
        if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
            return a is b
        return bool(a == b) and type(a) is type(b)
    except Exception:
        return False


# --------------------------------------------------------------------------- coercion
def _describe(hint) -> str:
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        return ' or '.join(_describe(a) for a in args if a is not type(None)) + (
            ' or None' if type(None) in args else '')
    if origin is typing.Literal:
        return 'one of ' + ', '.join(repr(a) for a in args)
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            return f'a sequence of {_describe(args[0])}'
        return f'a sequence of {len(args)} values' if args else 'a sequence'
    if origin in (dict, collections.abc.Mapping):
        return 'a mapping'
    if origin is collections.abc.Callable or hint in (typing.Callable, collections.abc.Callable):
        return 'a function'
    if hint is float:
        return 'a number'
    if hint is int:
        return 'an integer'
    if hint is bool:
        return 'True or False'
    if hint is str:
        return 'a string'
    return getattr(hint, '__name__', str(hint))


def _show(v) -> str:
    if isinstance(v, np.ndarray):
        return f'an array of shape {v.shape}'
    text = repr(v)
    return text if len(text) <= 60 else text[:57] + '...'


def _fail(hint, value, owner, name):
    raise ConfigError(f"{owner}({name}=...): expected {_describe(hint)}, got {_show(value)}")


def coerce(value, hint, owner='settings', name='value'):
    """``value`` converted to the type ``hint`` declares, or :class:`ConfigError` naming
    ``owner(name=...)``. Conversions: list or range to tuple, int to float, a path to a string,
    a numpy scalar to a Python one, a mapping to a :class:`FrozenDict`, and a single value to a
    one-element tuple where a tuple of such values is declared."""
    if hint is typing.Any or hint is object:
        return value
    origin, args = typing.get_origin(hint), typing.get_args(hint)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        for a in args:
            if a is type(None):
                continue
            try:
                return coerce(value, a, owner, name)
            except ConfigError:
                continue
        _fail(hint, value, owner, name)
    if origin is typing.Literal:
        if any(value == a and type(value) is type(a) for a in args):
            return value
        close = (difflib.get_close_matches(value, [a for a in args if isinstance(a, str)], n=1)
                 if isinstance(value, str) else [])
        raise ConfigError(f"{owner}({name}={value!r}): expected one of {', '.join(map(repr, args))}"
                          + (f". Did you mean {close[0]!r}?" if close else ''))
    if origin is tuple:
        if len(args) == 2 and args[1] is Ellipsis:
            item = args[0]
            if not _is_sequence(value):
                if _accepts_single(value, item):
                    return (coerce(value, item, owner, name),)
                _fail(hint, value, owner, name)
            return tuple(coerce(x, item, owner, f'{name}[{i}]') for i, x in enumerate(value))
        if not _is_sequence(value):
            _fail(hint, value, owner, name)
        items = list(value)
        if args and len(items) != len(args):
            raise ConfigError(f"{owner}({name}=...): expected {len(args)} values, got {len(items)}")
        if not args:
            return tuple(items)
        return tuple(coerce(x, a, owner, f'{name}[{i}]') for i, (x, a) in enumerate(zip(items, args)))
    if origin in (dict, collections.abc.Mapping):
        if not isinstance(value, collections.abc.Mapping):
            _fail(hint, value, owner, name)
        kt, vt = args if len(args) == 2 else (typing.Any, typing.Any)
        return FrozenDict({coerce(k, kt, owner, name): coerce(v, vt, owner, f'{name}[{k!r}]')
                           for k, v in value.items()})
    if origin is collections.abc.Callable or hint in (typing.Callable, collections.abc.Callable):
        if not callable(value):
            _fail(hint, value, owner, name)
        return value
    if hint is bool:
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        _fail(hint, value, owner, name)
    if hint is int:
        if isinstance(value, (int, np.integer)) and not isinstance(value, (bool, np.bool_)):
            return int(value)
        _fail(hint, value, owner, name)
    if hint is float:
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
            return float(value)
        _fail(hint, value, owner, name)
    if hint is str:
        if isinstance(value, str):
            return value
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        _fail(hint, value, owner, name)
    if isinstance(hint, type):
        if isinstance(value, hint):
            return value
        _fail(hint, value, owner, name)
    return value


def _is_sequence(v):
    return isinstance(v, (list, tuple, range)) or (isinstance(v, np.ndarray) and v.ndim == 1)


def _accepts_single(value, item_hint):
    """A lone value where a tuple of ``item_hint`` is declared: a string or a settings object."""
    if isinstance(value, str):
        return True
    return isinstance(value, Config)


# --------------------------------------------------------------------------- repr and JSON
def python_repr(v) -> str:
    """The Python source that rebuilds a setting's value (as far as one exists)."""
    if isinstance(v, Config):
        return repr(v)
    if isinstance(v, tuple):
        inner = ', '.join(python_repr(x) for x in v)
        return f'({inner},)' if len(v) == 1 else f'({inner})'
    if isinstance(v, list):
        return '[' + ', '.join(python_repr(x) for x in v) + ']'
    if isinstance(v, dict):
        return '{' + ', '.join(f'{python_repr(k)}: {python_repr(x)}' for k, x in v.items()) + '}'
    if isinstance(v, np.ndarray):
        return f'<array {v.shape} {v.dtype}>'
    if isinstance(v, (types.FunctionType, types.BuiltinFunctionType, np.ufunc)):
        return getattr(v, '__qualname__', getattr(v, '__name__', repr(v)))
    return repr(v)


def type_name(cls) -> str:
    """``"module:QualName"`` of a class (how records name it)."""
    return f'{cls.__module__}:{cls.__qualname__}'


def import_name(name: str):
    """The object ``"module:qualname"`` names."""
    module, _, qual = name.partition(':')
    obj = importlib.import_module(module)
    for part in qual.split('.'):
        obj = getattr(obj, part)
    return obj


def encode(v):
    """A setting's value in JSON form: settings objects as dicts with their ``type``, tuples as
    lists, ranges and functions tagged, arrays by shape, type and hash (not their values)."""
    if isinstance(v, Config):
        return v.to_dict()
    if isinstance(v, (tuple, list)):
        return [encode(x) for x in v]
    if isinstance(v, dict):
        return {str(k): encode(x) for k, x in v.items()}
    if isinstance(v, range):
        return {'range': [v.start, v.stop, v.step]}
    if isinstance(v, np.ndarray):
        import hashlib
        return {'array': {'shape': list(v.shape), 'dtype': str(v.dtype),
                          'sha1': hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest()}}
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, os.PathLike):
        return os.fspath(v)
    if callable(v):
        from .functions import by_value, describe_callable
        if isinstance(v, by_value):
            return {'by_value': v.__qualname__, 'digest': v.digest, 'sha256': v.sha256, 'source': v.source}
        if isinstance(v, (types.FunctionType, types.BuiltinFunctionType, type, np.ufunc)):
            return {'function': describe_callable(v)}
        # an object (a hook): its class and its state, so a record can rebuild it (the pickle) and a
        # fingerprint can name it the same in every process (the class and the state; not the repr,
        # which can hold a memory address, nor the pickle, which names the module the script was
        # loaded as)
        import base64
        import pickle
        try:
            blob = base64.b64encode(pickle.dumps(v)).decode()
        except Exception:
            return {'function': describe_callable(v)}
        return {'object': describe_callable(v), 'state': _object_state(v), 'pickle': blob, 'repr': repr(v)[:200]}
    return v


def _object_state(obj):
    """An object's state in JSON form, the same in every process: its ``__getstate__()`` (or its
    attributes) with the values encoded as settings and other objects by class and state."""
    try:
        state = obj.__getstate__() if hasattr(obj, '__getstate__') else vars(obj)
    except Exception:
        return None
    if state is None:
        return None
    if not isinstance(state, dict):
        state = {'state': state}

    def plain(v):
        if isinstance(v, (dict, list, tuple)):
            return ({str(k): plain(x) for k, x in v.items()} if isinstance(v, dict) else [plain(x) for x in v])
        v = encode(v)
        if v is None or isinstance(v, (str, int, float, bool, dict, list)):
            return v
        from .functions import describe_callable
        return {'class': describe_callable(v), 'state': _object_state(v)}
    return plain(state)


def decode(v):
    """The inverse of :func:`encode` (arrays cannot be rebuilt: they are recorded by hash only)."""
    if isinstance(v, list):
        return [decode(x) for x in v]
    if isinstance(v, dict):
        if 'type' in v and isinstance(v['type'], str) and ':' in v['type']:
            cls = import_name(v['type'])
            fields = {k: decode(x) for k, x in v.items() if k != 'type'}
            rebuild = getattr(cls, '_from_fields', None)     # a class whose constructor is not its fields
            return rebuild(**fields) if rebuild is not None else cls(**fields)
        if set(v) == {'range'}:
            return range(*v['range'])
        if set(v) == {'function'}:
            from .functions import load_callable
            return load_callable(v['function'])
        if 'object' in v and 'pickle' in v:
            import base64
            import pickle
            return pickle.loads(base64.b64decode(v['pickle']))
        if 'by_value' in v:
            raise ConfigError(f"the function {v['by_value']} was sent by value (sc.by_value); a record cannot "
                              f"rebuild it: write it to a module")
        if set(v) == {'array'}:
            raise ConfigError(f"an array setting is recorded by its hash only ({v['array']}); "
                              f"it cannot be rebuilt from the record")
        return {k: decode(x) for k, x in v.items()}
    return v
