"""Path / resource resolution for selfcal.

External users set environment variables (or pass explicit paths). Defaults may
be absolute paths that exist only on the original processing host; they are
used as a fallback only when present, so runs on that host need no
configuration, while an installed package on any other machine fails with an
actionable error instead of silently using a wrong path.

Resolution order: explicit argument > ``$env_var`` > ``default``.
"""
import os
from pathlib import Path


class SelfCalConfigError(RuntimeError):
    """Raised when a required path/resource cannot be resolved."""


# Environment variables (documented for external users).
ENV_SPHEREX_CALIB_DIR = 'SELFCAL_SPHEREX_CALIB_DIR'
ENV_SPHEREX_CHANNEL_FILE = 'SELFCAL_SPHEREX_CHANNEL_FILE'
ENV_LVF_PARAMS_DIR = 'SELFCAL_LVF_PARAMS_DIR'


def resolve_path(explicit=None, *, env_var=None, default=None, what='path',
                 must_exist=True):
    """Resolve a path from an explicit value, an env var, or a default.

    Parameters
    ----------
    explicit : str or os.PathLike or None
        Caller-supplied path; wins if given.
    env_var : str or None
        Environment variable consulted next.
    default : str or os.PathLike or None
        Fallback (e.g. a packaged resource, or a host-specific absolute path
        that may not exist on other machines).
    what : str
        Human label used in error messages.
    must_exist : bool
        If True, raise when the resolved path does not exist on disk.
    """
    candidate = source = None
    if explicit is not None:
        candidate, source = str(explicit), 'explicit argument'
    elif env_var and os.environ.get(env_var):
        candidate, source = os.environ[env_var], f'${env_var}'
    elif default is not None:
        candidate, source = str(default), 'default'

    if candidate is None:
        raise SelfCalConfigError(
            f"Could not resolve {what}: pass it explicitly"
            + (f" or set ${env_var}." if env_var else "."))
    if must_exist and not Path(candidate).exists():
        raise SelfCalConfigError(
            f"{what} ({source}) does not exist: {candidate}. "
            f"Pass an explicit path"
            + (f" or set ${env_var}." if env_var else "."))
    return candidate
