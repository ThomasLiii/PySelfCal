"""The run engine's old import path: the engine is now :mod:`selfcal.run`.

Every module of the engine is aliased here (``selfcal_scripts.runner.engine`` IS
``selfcal.run.engine``), so scripts and out-of-tree modes written against this path keep
working and share the engine's state: a mode registered through either name lands in the one
registry. New code imports :mod:`selfcal.run`.
"""
import importlib as _importlib
import sys as _sys

from selfcal.run import RunConfig, get_instrument, load_config, run  # noqa: F401

_MODULES = ('config', 'engine', 'pipelines', 'npass', 'staging', 'runlog', 'postprocess',
            'modes', 'modes.base', 'modes.continuum', 'modes.spectral', 'modes.two_block_fixed',
            'modes.model')
for _name in _MODULES:
    _module = _importlib.import_module(f'selfcal.run.{_name}')
    _sys.modules[f'{__name__}.{_name}'] = _module
    if '.' not in _name:
        globals()[_name] = _module
del _name, _module
