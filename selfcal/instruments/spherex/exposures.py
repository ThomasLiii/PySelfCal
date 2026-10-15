"""SPHEREx L2b exposure files: the reader the reprojection uses, and the noise variance they carry.

An L2b file holds one detector frame: ``IMAGE`` (MJy/sr, with the celestial WCS), ``FLAGS``
(the bit mask), ``VARIANCE`` (MJy²/sr², the level-2 pipeline's noise variance of every pixel),
``ZODI``, ``PSF`` and ``WCS-WAVE``. Every image extension is stored in the orientation of
``IMAGE`` (the MWIR detectors' x flip, ``DETCOORD = 'sky'``, is applied to all of them), so a
plane lines up with the data as stored.

The ``FLAGS`` bits (the extension's header): TRANSIENT 0, OVERFLOW 1, SUR_ERROR 2, PHANTOM 4,
REFERENCE 5, NONFUNC 6, DICHROIC 7, MISSING_DATA 9, HOT 10, COLD 11, FULLSAMPLE 12, PHANMISS 14,
NONLINEAR 15, PERSIST 17, OUTLIER 19, SOURCE 21. The dead pixels whose values are garbage (tens to
thousands of MJy/sr, of either sign) carry SUR_ERROR ("error in onboard processing") or NONFUNC
("permanently unusable"), so the reader dilates those two bits by one pixel (:data:`DILATED_BITS`):
the exact reprojection spreads such a value into every output pixel the dead pixel's area touches,
beyond the bilinear footprint of the reprojected mask, and the dilation makes the mask cover that
spread (measured 2026-10-15 on NEP D3 frames: every one of 612 leaked output pixels on 12 frames
caught, for 3.9 % of the unflagged pixels; see ``workspace/covariance/report.md``).
"""
from __future__ import annotations

import numpy as np
from astropy.io import fits

from ...io.frames import ExposureData, dilate_bits, fill_missing

__all__ = ['read_l2b', 'l2b_variance', 'l2b_flags', 'VARIANCE_EXT', 'VARIANCE_LAYER', 'FLAGS_EXT', 'DILATED_BITS']

#: The extension of an L2b file that holds the noise variance, and the frame layer it becomes.
VARIANCE_EXT = 'VARIANCE'
VARIANCE_LAYER = 'variance'
#: The extension that holds the bit mask, and the bits the reader dilates by one pixel (SUR_ERROR, NONFUNC).
FLAGS_EXT = 'FLAGS'
DILATED_BITS = (2, 6)


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


def l2b_flags(path, ext=FLAGS_EXT, dilate=DILATED_BITS) -> np.ndarray:
    """The bit mask of the L2b file ``path`` (extension ``ext``) as the reader returns it: the bits
    ``dilate`` set on every pixel within one pixel of a pixel that has them
    (:func:`~selfcal.io.frames.dilate_bits`; the module docstring says why). For a pass that
    refreshes the mask of frames made before the reader dilated it:
    ``refresh_mask(frame, l2b_flags(source))`` (:func:`~selfcal.io.reprojection.refresh_mask`)."""
    with fits.open(path) as hdul:
        m = np.asarray(hdul[ext].data)
    return dilate_bits(m, dilate) if dilate else m


def read_l2b(path, sci_ext, dq_ext=None, header_only=False, *, dilate=DILATED_BITS) -> ExposureData:
    """The L2b exposure reader (``SPHEREx.layout().reader``): the values and the WCS header of
    extension ``sci_ext`` (``IMAGE``), the flag bits of ``dq_ext`` (``FLAGS``; None: no mask) with
    the bits ``dilate`` (default :data:`DILATED_BITS`: SUR_ERROR, NONFUNC) dilated by one pixel (the module
    docstring says why), and the file's ``VARIANCE`` as the layer ``variance`` (:func:`l2b_variance`;
    a file without that extension gives no layer), stored with every reprojected frame as
    ``layers/variance``: the source of a ``sc.Layer()`` data variable, for inverse-variance weights
    (``sc.Variance``)."""
    with fits.open(path) as hdul:
        hdu = hdul[sci_ext]
        header = hdu.header.copy()
        if header_only:
            return ExposureData(header=header, shape=tuple(hdu.shape[-2:]) if hdu.shape else None)
        data = hdu.data
        mask = None
        if dq_ext is not None:
            mask = np.asarray(hdul[dq_ext].data)
            if dilate:
                mask = dilate_bits(mask, dilate)
        layers = {VARIANCE_LAYER: _variance_plane(hdul[VARIANCE_EXT])} if VARIANCE_EXT in hdul else {}
    return ExposureData(header=header, data=data, mask=mask, layers=layers)
