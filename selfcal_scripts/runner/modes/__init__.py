"""Mode registry. Importing this package registers the built-in modes; a new mode
is a new module here with an ``@register_mode`` class (no engine edits)."""
from .base import CalMode, register_mode, get_mode, available_modes  # noqa: F401
from . import continuum, pahfit, pahfit_lvf, k2_readout, tiled, multiline  # noqa: F401  (self-register on import)
