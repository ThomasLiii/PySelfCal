"""A minimal, telescope-free instrument for the runner tests.

It implements exactly the surface the run engine and the ``continuum`` mode use
today (see workspace/unify/audit/A_runner.md §4.1): a square detector, a grid
chunk map (one axis of ``num_col`` columns per row-band), no wavelength maps, no
spectral capability. Synthetic exposures are written as FITS files with the
same HDU layout the engine's reprojection currently assumes (science image with
a celestial WCS + ``FINAST`` in extension 1, an integer DQ mask in extension 2).
"""
import os

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from selfcal.geometry.map_helper import make_grid_chunk_map, compute_chunk_adjacency

DET = 64                     # detector side (px)
N_CHUNK_SIDE = 4             # chunks per side -> 16 chunks
PIX_ARCSEC = 20.0            # detector pixel scale
REF_ARCSEC = 20.0            # reference-grid pixel scale used by the runner (resolution_arcsec)


class ToyJob:
    def __init__(self, name):
        self.name, self.kind, self.value = name, 'all', None


class ToyInstrument:
    """Duck-typed Instrument: the members the engine + continuum mode call."""
    name = 'toy'
    capabilities = frozenset()
    aux_keys = ()

    def jobs(self, inst_cfg):
        return [ToyJob(inst_cfg.get('job_name', 'All'))]

    def frame_tag(self, inst_cfg):
        return f"Toy{inst_cfg.get('detector', 0)}_N{N_CHUNK_SIDE}"

    def detector_inputs(self, inst_cfg, oversample):
        det_map = make_grid_chunk_map((DET, DET), N_CHUNK_SIDE).astype(np.int32)
        grid_map = np.kron(det_map, np.ones((oversample, oversample), dtype=np.int32)) if oversample > 1 else det_map
        return {'det_chunk_map': det_map, 'grid_chunk_map': grid_map}

    def channel_inputs(self, inst_cfg, det_inputs, job):
        det_w = np.ones((DET, DET), dtype=np.float32)
        g = det_inputs['grid_chunk_map'].shape
        return {'det_valid_mask_padded': det_w, 'det_valid_mask': det_w,
                'grid_valid_weight': np.ones(g, dtype=np.float32), 'grid_valid_mask': np.ones(g, dtype=np.float32)}

    # continuum mode helpers (SPHEREx spellings, generic content)
    def column_adjacency(self, det_chunk_map, num_columns):
        return compute_chunk_adjacency(det_chunk_map, reg_axis='both')

    def column_poly_chains(self, det_chunk_map, num_columns, degree=1):
        raise NotImplementedError("the toy instrument declares no polynomial chains")

    def aux(self, det_inputs):
        return None

    def offset_render(self, inst_cfg, det_inputs, channel_inputs):
        return None                      # block-constant chunk_to_det

    def wavelength_maps(self, det_inputs):
        return None

    def wavelength_append(self, det_inputs, mm, maps, sigma):
        return None

    def precompute(self, inst_cfg):
        pass


# ---------------------------------------------------------------------------- synthetic exposures
def make_truth(rng, ref_side=160):
    """A smooth sky on a reference grid + per-exposure chunk offsets and scalars."""
    y, x = np.mgrid[0:ref_side, 0:ref_side].astype(np.float64)
    sky = 5.0 + 0.02 * x + 0.01 * y + 0.5 * np.sin(x / 11.0) * np.cos(y / 13.0)
    return sky


def write_exposures(out_dir, n_exp, rng, ra0=180.0, dec0=30.0, ref_side=160, noise=0.02):
    """Write ``n_exp`` FITS exposures (ext 1 sci + WCS + FINAST=0, ext 2 int32 DQ) that
    sample a common sky with random pointings; returns (paths, injected offsets, scalars)."""
    os.makedirs(out_dir, exist_ok=True)
    sky = make_truth(rng, ref_side)
    n_chunks = N_CHUNK_SIDE ** 2
    chunk_map = make_grid_chunk_map((DET, DET), N_CHUNK_SIDE)
    offsets = rng.normal(0, 0.3, size=(n_exp, n_chunks))
    offsets -= offsets.mean(axis=1, keepdims=True)          # mean-zero per frame (matches the anchor)
    scalars = rng.normal(0, 1.0, size=n_exp)
    scale = PIX_ARCSEC / 3600.0
    paths = []
    for k in range(n_exp):
        # pointing: a detector-sized window inside the sky, random integer shift
        oy, ox = rng.integers(0, ref_side - DET, size=2)
        det_true = sky[oy:oy + DET, ox:ox + DET]
        img = det_true + offsets[k][chunk_map] + scalars[k] + rng.normal(0, noise, size=(DET, DET))
        w = WCS(naxis=2)
        w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
        w.wcs.crpix = [DET / 2 + 0.5 - ox, DET / 2 + 0.5 - oy]     # so that sky pixel (0,0) maps consistently
        w.wcs.crval = [ra0, dec0]
        w.wcs.cdelt = [-scale, scale]
        hdr = w.to_header()
        hdr['FINAST'] = 0
        hdr['BUNIT'] = 'toy'
        dq = np.zeros((DET, DET), dtype=np.int32)
        dq[rng.random((DET, DET)) < 0.01] = 1 << 3                # a few flagged pixels
        p = os.path.join(out_dir, f'toy_exp_{k:03d}_D0.fits')
        fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(img.astype(np.float32), header=hdr, name='SCI'),
                      fits.ImageHDU(dq, name='DQ')]).writeto(p, overwrite=True)
        paths.append(p)
    return paths, offsets, scalars
