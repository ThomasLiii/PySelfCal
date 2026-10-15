"""SPHEREx L2b exposure files: the reader the reprojection uses, and the noise variance they carry.

An L2b file holds one detector frame: ``IMAGE`` (MJy/sr, with the celestial WCS), ``FLAGS``
(the bit mask), ``VARIANCE`` (MJy²/sr², the level-2 pipeline's noise variance of every pixel),
``ZODI``, ``PSF`` and ``WCS-WAVE``. Every image extension is stored in the orientation of
``IMAGE`` (the MWIR detectors' x flip, ``DETCOORD = 'sky'``, is applied to all of them), so a
plane lines up with the data as stored.
"""
from __future__ import annotations

import numpy as np
from astropy.io import fits

from ...io.frames import ExposureData, fill_missing

__all__ = ['read_l2b', 'l2b_variance', 'VARIANCE_EXT', 'VARIANCE_LAYER']

#: The extension of an L2b file that holds the noise variance, and the frame layer it becomes.
VARIANCE_EXT = 'VARIANCE'
VARIANCE_LAYER = 'variance'


def _variance_plane(hdu) -> np.ndarray:
    # The extension as a layer: float64, every value that is not finite or not positive (bad
    # pixels, which the flags exclude) replaced by the mean of its usable neighbours, so that the
    # bilinear reprojection of the plane is finite wherever the data are.
    v = np.asarray(hdu.data, dtype=np.float64)
    return fill_missing(v, ~(np.isfinite(v) & (v > 0)))


def l2b_variance(path, ext=VARIANCE_EXT) -> np.ndarray | None:
    """The noise-variance plane of the L2b file ``path`` (its extension ``ext``) as the reader
    stores it: float64, every value that is not finite or not positive (bad pixels, which the flags
    exclude) replaced by the mean of its usable neighbours (:func:`~selfcal.io.frames.fill_missing`),
    so that a bilinear reprojection of the plane is finite wherever the data are. None when the file
    has no such extension. For a pass that adds the layer to frames made before the reader carried
    it: ``add_layers(frame, {'variance': l2b_variance(source)})``
    (:func:`~selfcal.io.reprojection.add_layers`)."""
    with fits.open(path) as hdul:
        if ext not in hdul:
            return None
        return _variance_plane(hdul[ext])


def read_l2b(path, sci_ext, dq_ext=None, header_only=False) -> ExposureData:
    """The L2b exposure reader (``SPHEREx.layout().reader``): the values and the WCS header of
    extension ``sci_ext`` (``IMAGE``) and the flag bits of ``dq_ext`` (``FLAGS``; None: no mask),
    as the default reader reads them, plus the file's ``VARIANCE`` as the layer ``variance``
    (:func:`l2b_variance`; a file without that extension gives no layer), stored with every
    reprojected frame as ``layers/variance``: the source of a ``sc.Layer()`` data variable, for
    inverse-variance weights (``sc.Variance``)."""
    with fits.open(path) as hdul:
        hdu = hdul[sci_ext]
        header = hdu.header.copy()
        if header_only:
            return ExposureData(header=header, shape=tuple(hdu.shape[-2:]) if hdu.shape else None)
        data = hdu.data
        mask = hdul[dq_ext].data if dq_ext is not None else None
        layers = {VARIANCE_LAYER: _variance_plane(hdul[VARIANCE_EXT])} if VARIANCE_EXT in hdul else {}
    return ExposureData(header=header, data=data, mask=mask, layers=layers)
