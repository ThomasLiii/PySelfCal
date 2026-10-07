"""Configuration: the settings base class, function references, and path resolution.

* :mod:`selfcal.config.base`: :class:`Config`, the base of every settings object of the Python
  API (checked when built, immutable, printable as the Python that rebuilds it);
* :mod:`selfcal.config.functions`: functions passed as settings, checked so that the worker
  processes can import them;
* :mod:`selfcal.config.paths`: where calibration resources are found (an explicit path, a
  ``SELFCAL_*`` environment variable, or a default).
"""
from .base import Config, ConfigError, FrozenDict
from .functions import by_value, function_ref, import_module, load_callable
from .paths import (
                    ENV_LVF_PARAMS_DIR,
                    ENV_SPHEREX_CALIB_DIR,
                    ENV_SPHEREX_CHANNEL_FILE,
                    SelfCalConfigError,
                    resolve_path,
)

__all__ = ['Config', 'ConfigError', 'FrozenDict', 'by_value', 'function_ref', 'import_module', 'load_callable',
           'resolve_path', 'SelfCalConfigError', 'ENV_LVF_PARAMS_DIR', 'ENV_SPHEREX_CALIB_DIR',
           'ENV_SPHEREX_CHANNEL_FILE']
