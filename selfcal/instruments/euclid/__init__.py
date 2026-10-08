"""Euclid NISP: the instrument ``sc.Euclid`` and its helpers.

- :mod:`~selfcal.instruments.euclid.settings`: :class:`~selfcal.instruments.euclid.settings.Euclid`,
  which implements the instrument contract (16 detectors per exposure, grid, stripe and tilt chunk
  maps, edge taper, mosaic renderers); available from this package, with the hook classes:
  ``euclid.Euclid``, ``euclid.StarMask``, ``euclid.ResidualMask``.
- :mod:`~selfcal.instruments.euclid.adapter`: the helpers of the chunk maps, the edge taper and the
  mosaic renderers.
- :mod:`~selfcal.instruments.euclid.conventions`: NISP constants (FITS extension numbers, detector
  shape, DQ bits to ignore) and the square grid chunk map.
- :mod:`~selfcal.instruments.euclid.hooks`: the recipe's per-frame hooks,
  :class:`~selfcal.instruments.euclid.hooks.StarMask` and
  :class:`~selfcal.instruments.euclid.hooks.ResidualMask`.
- :mod:`~selfcal.instruments.euclid.exposures`: lists of exposure files from a VOTable catalogue, a
  CSV file or a directory.
"""
_EXPORTS = {'Euclid': 'settings', 'StarMask': 'hooks', 'ResidualMask': 'hooks'}


def __getattr__(name):
    # Loaded on first use, so worker processes importing the instrument stay light.
    if name in _EXPORTS:
        import importlib
        return getattr(importlib.import_module(f'{__name__}.{_EXPORTS[name]}'), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
