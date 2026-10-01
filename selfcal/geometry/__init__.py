"""Array and WCS helpers shared by the pipeline stages and the instruments.

- :mod:`~selfcal.geometry.map_helper`: detector- and grid-map utilities: bit masks, weights,
  outlier flags, chunk maps, bilinear resampling, mean-preserving splines.
- :mod:`~selfcal.geometry.wcs_helper`: the WCS of the reference grid, built to cover the exposures
  or derived from an existing projection, and saved as or loaded from a FITS file (``ref.fits``).
"""
