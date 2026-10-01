"""Instruments: the one place a telescope enters the pipeline.

Importing this package registers the built-ins (``spherex``, ``euclid``, ``grid``); other
packages register theirs through the ``selfcal.instruments`` entry-point group.
Select by name with ``[instrument].name`` in a run config, or in code::

    from selfcal.instruments import get_instrument
    inst = get_instrument("spherex")
"""
from .base import (Instrument, register_instrument, get_instrument, available_instruments,   # noqa: F401
                   Job, ChunkMap, DetectorGeometry, JobGeometry, ExposureLayout)
from . import grid                                    # noqa: F401  (registers 'grid')
from .spherex import adapter as _spherex_adapter      # noqa: F401  (registers 'spherex')
from .euclid import adapter as _euclid_adapter        # noqa: F401  (registers 'euclid')

__all__ = ['Instrument', 'register_instrument', 'get_instrument', 'available_instruments',
           'Job', 'ChunkMap', 'DetectorGeometry', 'JobGeometry', 'ExposureLayout']
