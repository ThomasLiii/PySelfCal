"""Synthetic exposures for the runner tests: a smooth sky sampled by a 64-px
square detector at random pointings, with injected per-frame chunk offsets and
scalars, written as FITS files (science image + celestial WCS in extension 1,
an integer DQ mask in extension 2) — the layout the built-in ``grid``
instrument reads by default.
"""
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

DET = 64                     # detector side (px)
N_CHUNK_SIDE = 4             # chunks per side -> 16 chunks
PIX_ARCSEC = 20.0            # detector pixel scale
REF_ARCSEC = 20.0            # reference-grid pixel scale used by the runner (resolution_arcsec)


# ---------------------------------------------------------------------------- synthetic exposures
def make_truth(rng, ref_side=160):
    """A smooth sky on a reference grid + per-exposure chunk offsets and scalars."""
    y, x = np.mgrid[0:ref_side, 0:ref_side].astype(np.float64)
    sky = 5.0 + 0.02 * x + 0.01 * y + 0.5 * np.sin(x / 11.0) * np.cos(y / 13.0)
    return sky


def write_exposures(out_dir, n_exp, rng, ra0=180.0, dec0=30.0, ref_side=160, noise=0.02,
                    det_shape=(DET, DET), chunks=(N_CHUNK_SIDE, N_CHUNK_SIDE), with_dq=True,
                    extra_term=None, offset_sigma=0.3, scalar_sigma=1.0):
    """Write ``n_exp`` FITS exposures (ext 1 sci + WCS + FINAST=0, ext 2 int32 DQ unless
    ``with_dq`` is False) of a ``det_shape`` detector partitioned into ``chunks`` (ny, nx)
    offset chunks, sampling a common sky at random pointings; returns
    (paths, injected offsets, scalars). ``extra_term = (S, c_det)`` adds a second
    sky term: the reference-grid map ``S`` times ``c_det``, a coefficient per
    DETECTOR pixel (so each sky pixel sees a different coefficient in every frame)."""
    from selfcal.instruments.grid import rect_grid_chunk_map
    os.makedirs(out_dir, exist_ok=True)
    sky = make_truth(rng, ref_side)
    H, W = det_shape
    ny, nx = chunks
    n_chunks = ny * nx
    chunk_map = rect_grid_chunk_map((H, W), ny, nx)
    offsets = rng.normal(0, offset_sigma, size=(n_exp, n_chunks))
    offsets -= offsets.mean(axis=1, keepdims=True)          # mean-zero per frame (matches the anchor)
    scalars = rng.normal(0, scalar_sigma, size=n_exp)
    scale = PIX_ARCSEC / 3600.0
    paths = []
    for k in range(n_exp):
        # pointing: a detector-sized window inside the sky, random integer shift
        oy = rng.integers(0, ref_side - H)
        ox = rng.integers(0, ref_side - W)
        det_true = sky[oy:oy + H, ox:ox + W]
        img = det_true + offsets[k][chunk_map] + scalars[k] + rng.normal(0, noise, size=(H, W))
        if extra_term is not None:
            s_map, c_det = extra_term
            img = img + s_map[oy:oy + H, ox:ox + W] * c_det
        w = WCS(naxis=2)
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w.wcs.crpix = [W / 2 + 0.5 - ox, H / 2 + 0.5 - oy]         # so that sky pixel (0,0) maps consistently
        w.wcs.crval = [ra0, dec0]
        w.wcs.cdelt = [-scale, scale]
        hdr = w.to_header()
        hdr['FINAST'] = 0
        hdr['BUNIT'] = 'toy'
        hdus = [fits.PrimaryHDU(), fits.ImageHDU(img.astype(np.float32), header=hdr, name='SCI')]
        if with_dq:
            dq = np.zeros((H, W), dtype=np.int32)
            dq[rng.random((H, W)) < 0.01] = 1 << 3                # a few flagged pixels
            hdus.append(fits.ImageHDU(dq, name='DQ'))
        p = os.path.join(out_dir, f'toy_exp_{k:03d}_D0.fits')
        fits.HDUList(hdus).writeto(p, overwrite=True)
        paths.append(p)
    return paths, offsets, scalars


# ---------------------------------------------------------------------------- known functions of data variables
def xtilde(det_x):
    """(column - centre) / half-width of the toy detector: -1 at its left edge, +1 at its right (a
    known function an offset term's coefficient can read, as the user's across-detector slope term)."""
    return (np.asarray(det_x, dtype=np.float64) - (DET - 1) / 2.0) / (DET / 2.0)


def x_ramp(det_x):
    """column / (DET - 1): a coefficient rising across the toy detector (a second sky term)."""
    return np.asarray(det_x, dtype=np.float64) / (DET - 1)
