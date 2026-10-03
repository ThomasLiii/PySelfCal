"""This package holds the built-in ``euclid`` instrument for Euclid NISP and its helpers.

Importing :mod:`selfcal.instruments` registers the instrument; a run config selects it with
``[instrument].name = "euclid"``.

- :mod:`~selfcal.instruments.euclid.adapter`: the instrument class (16 detectors per exposure,
  grid, stripe and tilt chunk maps, edge taper, mosaic renderers).
- :mod:`~selfcal.instruments.euclid.conventions`: NISP constants (FITS extension numbers,
  detector shape, DQ bits to ignore) and the square grid chunk map.
- :mod:`~selfcal.instruments.euclid.hooks`: the recipe's per-frame hooks, ``star_position_mask``
  and ``residual_mask``.
- :mod:`~selfcal.instruments.euclid.exposures`: lists of exposure files from a VOTable catalogue,
  a CSV file or a directory.
- :mod:`~selfcal.instruments.euclid.settings`: Euclid as settings of the Python API,
  :class:`~selfcal.instruments.euclid.settings.Euclid` (available from this package, with the hook
  classes: ``euclid.Euclid``, ``euclid.StarMask``, ``euclid.ResidualMask``).
"""
_EXPORTS = {'Euclid': 'settings', 'StarMask': 'hooks', 'ResidualMask': 'hooks'}


def __getattr__(name):
    # Loaded on first use, so worker processes importing the instrument stay light.
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(f'{__name__}.{_EXPORTS[name]}'), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
