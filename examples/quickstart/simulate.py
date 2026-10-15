"""Simulate a small dithered survey for the selfcal quickstart.

Run it from the directory that should hold the outputs (the repository root in
the quickstart):

    python examples/quickstart/simulate.py

It writes N_EXP exposures, quickstart_output/exposures/sim_000.fits, ..., and the
injected truth, quickstart_output/truth.npz. Every exposure is

    image = sky + offset[chunk] + scalar + noise

where `sky` is one fixed sky (a smooth background and a few compact sources)
seen at a random pointing, `offset` is an additive offset per chunk of the
detector (4 x 4 chunks of 16 x 16 pixels) that changes from exposure to
exposure, and `scalar` is one additive level per exposure. Extension 1 holds the
image and its celestial WCS, extension 2 an integer data-quality (DQ) mask: the
layout `sc.Camera((64, 64), chunks=(4, 4), dq_ext=2)` reads (sci_ext = 1, dq_ext = 2).
"""
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from selfcal.instruments.grid import rect_grid_chunk_map

OUT_DIR = "quickstart_output"   # relative to the current directory
N_EXP = 24                      # number of exposures
DET_SHAPE = (64, 64)            # detector (rows, columns)
CHUNKS = (4, 4)                 # offset chunks (rows, columns), as the camera's chunks
PIXEL_ARCSEC = 10.0             # detector pixel scale
RA0, DEC0 = 150.0, 2.0          # field centre (degrees)
DITHER_ARCMIN = 4.0             # pointings are uniform within +-4 arcmin of the centre
OFFSET_RMS = 0.3                # rms of the chunk offsets
SCALAR_RMS = 1.0                # rms of the per-exposure level
NOISE_RMS = 0.05                # white noise per pixel
BAD_FRACTION = 0.01             # fraction of pixels flagged in the DQ mask
SEED = 2026

# Compact sources: east and north of the field centre (arcmin), peak, Gaussian sigma (arcmin).
SOURCES = [(-3.0, 2.0, 3.0, 0.5), (2.5, -1.5, 2.0, 0.4), (4.0, 4.5, 1.5, 0.8),
           (-5.0, -4.0, 2.5, 0.6), (0.5, 5.5, 1.0, 0.35)]


def true_sky(ra, dec):
    """The simulated sky at (ra, dec), in degrees: a background with a gradient and
    a large-scale ripple, plus the SOURCES."""
    x = (ra - RA0) * np.cos(np.radians(DEC0)) * 60.0       # arcmin east of the centre
    y = (dec - DEC0) * 60.0                                # arcmin north of the centre
    sky = 5.0 + 0.05 * x + 0.03 * y + 0.4 * np.sin(x / 1.5) * np.cos(y / 2.0)
    for x0, y0, peak, sigma in SOURCES:
        sky = sky + peak * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2.0 * sigma ** 2))
    return sky


def exposure_wcs(ra, dec):
    """A north-up, east-left TAN projection of the detector centred on (ra, dec)."""
    wcs = WCS(naxis=2)
    wcs.wcs.ctype = ["RA---TAN", "DEC--TAN"]
    wcs.wcs.crval = [ra, dec]
    wcs.wcs.crpix = [(DET_SHAPE[1] + 1) / 2.0, (DET_SHAPE[0] + 1) / 2.0]
    wcs.wcs.cdelt = [-PIXEL_ARCSEC / 3600.0, PIXEL_ARCSEC / 3600.0]
    return wcs


def main():
    rng = np.random.default_rng(SEED)
    exp_dir = os.path.join(OUT_DIR, "exposures")
    os.makedirs(exp_dir, exist_ok=True)

    # The chunk of every detector pixel, numbered as sc.Camera numbers them
    # (chunk = row * n_columns + column).
    chunk_map = rect_grid_chunk_map(DET_SHAPE, *CHUNKS)
    n_chunks = CHUNKS[0] * CHUNKS[1]
    offsets = rng.normal(0.0, OFFSET_RMS, size=(N_EXP, n_chunks))
    offsets -= offsets.mean(axis=1, keepdims=True)   # an exposure's mean level is its scalar
    scalars = rng.normal(0.0, SCALAR_RMS, size=N_EXP)

    rows, cols = np.mgrid[0:DET_SHAPE[0], 0:DET_SHAPE[1]]
    names = []
    for k in range(N_EXP):
        east, north = rng.uniform(-DITHER_ARCMIN, DITHER_ARCMIN, size=2) / 60.0
        wcs = exposure_wcs(RA0 + east / np.cos(np.radians(DEC0)), DEC0 + north)
        ra, dec = wcs.pixel_to_world_values(cols, rows)          # sky position of every pixel
        image = (true_sky(ra, dec) + offsets[k][chunk_map] + scalars[k]
                 + rng.normal(0.0, NOISE_RMS, size=DET_SHAPE))
        # DQ: bit 3 set on a few random pixels. Any set bit not in ignore_list masks the pixel.
        dq = np.where(rng.random(DET_SHAPE) < BAD_FRACTION, 1 << 3, 0).astype(np.int32)
        name = f"sim_{k:03d}.fits"
        hdus = [fits.PrimaryHDU(),
                fits.ImageHDU(image.astype(np.float32), header=wcs.to_header(), name="SCI"),
                fits.ImageHDU(dq, name="DQ")]
        fits.HDUList(hdus).writeto(os.path.join(exp_dir, name), overwrite=True)
        names.append(name)

    truth_path = os.path.join(OUT_DIR, "truth.npz")
    np.savez(truth_path, files=np.array(names), offsets=offsets, scalars=scalars,
             chunk_map=chunk_map)
    print(f"wrote {N_EXP} exposures of {DET_SHAPE[0]} x {DET_SHAPE[1]} pixels to {exp_dir}/")
    print(f"wrote the injected offsets {offsets.shape} and scalars {scalars.shape} to {truth_path}")


if __name__ == "__main__":
    main()
