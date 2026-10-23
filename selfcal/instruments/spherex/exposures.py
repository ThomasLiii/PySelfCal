"""SPHEREx L2b exposure files: the reader the reprojection uses, and the noise variance they carry.

An L2b file holds one detector frame: ``IMAGE`` (MJy/sr, with the celestial WCS), ``FLAGS``
(the bit mask), ``VARIANCE`` (MJy²/sr², the level-2 pipeline's noise variance of every pixel),
``ZODI``, ``PSF`` and ``WCS-WAVE``. Every image extension is stored in the orientation of
``IMAGE`` (the MWIR detectors' x flip, ``DETCOORD = 'sky'``, is applied to all of them), so a
plane lines up with the data as stored.

The ``FLAGS`` bits (the extension's header): TRANSIENT 0, OVERFLOW 1, SUR_ERROR 2, PHANTOM 4,
REFERENCE 5, NONFUNC 6, DICHROIC 7, MISSING_DATA 9, HOT 10, COLD 11, FULLSAMPLE 12, PHANMISS 14,
NONLINEAR 15, PERSIST 17, OUTLIER 19, SOURCE 21; the reader adds DILATED_DEAD 30. The dead pixels
whose values are garbage (tens to thousands of MJy/sr, of either sign) carry SUR_ERROR ("error in
onboard processing") or NONFUNC ("permanently unusable"): the exact reprojection spreads such a value
into every output pixel the dead pixel's area touches, beyond the bilinear footprint of the
reprojected mask. So the reader sets the bit :data:`DILATED_DEAD_BIT` (30) on every pixel within one
pixel of a pixel with either of those bits (:data:`DILATED_BITS`), the dead pixels themselves
included, and keeps the two bits as the L2b file has them; a mask that drops every flagged pixel then
covers that spread (measured 2026-10-15 on NEP D3 frames: every one of 612 leaked output pixels on
12 frames caught, for 3.9 % of the unflagged pixels; see ``workspace/covariance/report.md``). From
2026-10-15 to 2026-10-22 the reader dilated the two bits themselves instead (``into=None``).
"""
from __future__ import annotations

import numpy as np
from astropy.io import fits

from ...io.frames import ExposureData, dilate_bits, fill_missing

__all__ = ['read_l2b', 'l2b_variance', 'l2b_flags', 'VARIANCE_EXT', 'VARIANCE_LAYER', 'FLAGS_EXT', 'DILATED_BITS',
           'DILATED_DEAD_BIT']

#: The extension of an L2b file that holds the noise variance, and the frame layer it becomes.
VARIANCE_EXT = 'VARIANCE'
VARIANCE_LAYER = 'variance'
#: The extension that holds the bit mask; the dead-pixel bits whose one-pixel dilation the reader flags
#: (SUR_ERROR, NONFUNC); the bit it flags it in (DILATED_DEAD; the L2b flags use bits 0 to 21).
FLAGS_EXT = 'FLAGS'
DILATED_BITS = (2, 6)
DILATED_DEAD_BIT = 30


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


def l2b_flags(path, ext=FLAGS_EXT, dilate=DILATED_BITS, into=DILATED_DEAD_BIT) -> np.ndarray:
    """The bit mask of the L2b file ``path`` (extension ``ext``) as the reader returns it: the bits
    as the file has them, and the bit ``into`` set on every pixel within one pixel of a pixel with
    any of ``dilate`` (:func:`~selfcal.io.frames.dilate_bits`; the module docstring says why;
    ``into=None``: the bits ``dilate`` dilated themselves, the reader of 2026-10-15 to 2026-10-22).
    For a pass that gives frames made before the reader set the bit the same mask:
    ``refresh_mask(frame, l2b_flags(source), bits=(DILATED_DEAD_BIT,))``
    (:func:`~selfcal.io.reprojection.refresh_mask`; the frame's other bits are kept)."""
    with fits.open(path) as hdul:
        m = np.asarray(hdul[ext].data)
    return dilate_bits(m, dilate, into=into) if dilate else m


def read_l2b(path, sci_ext, dq_ext=None, header_only=False, *, dilate=DILATED_BITS,
             into=DILATED_DEAD_BIT) -> ExposureData:
    """The L2b exposure reader (``SPHEREx.layout().reader``): the values and the WCS header of
    extension ``sci_ext`` (``IMAGE``), the flag bits of ``dq_ext`` (``FLAGS``; None: no mask) as the
    file has them plus the bit ``into`` (default :data:`DILATED_DEAD_BIT`) on every pixel within one
    pixel of a pixel with any of ``dilate`` (default :data:`DILATED_BITS`: SUR_ERROR, NONFUNC; the
    module docstring says why), and the file's ``VARIANCE`` as the layer ``variance``
    (:func:`l2b_variance`; a file without that extension gives no layer), stored with every
    reprojected frame as ``layers/variance``: the source of a ``sc.Layer()`` data variable, for
    inverse-variance weights (``sc.Variance``)."""
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
                mask = dilate_bits(mask, dilate, into=into)
        layers = {VARIANCE_LAYER: _variance_plane(hdul[VARIANCE_EXT])} if VARIANCE_EXT in hdul else {}
    return ExposureData(header=header, data=data, mask=mask, layers=layers)
