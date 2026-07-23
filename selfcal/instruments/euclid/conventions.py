"""Euclid NISP instrument conventions for the selfcal pipeline.

Makes Euclid a first-class instrument: the FITS extension layout, DQ ignore
bits, detector geometry, and the square grid chunk map that the Euclid mosaic
needs live here (used by notebooks/euclid_mosaic.ipynb).

Euclid NISP is broadband imaging: no spectral aux maps, so the default
continuum-only :class:`~selfcal.models.sky_model.SkyModel` applies and no SPHEREx
LVF / spectral machinery is touched.
"""
import numpy as np

from ...geometry.map_helper import make_grid_chunk_map

# NISP focal plane: 16 detectors, each FITS file laid out as
# (primary, sci, ?, dq) repeating per detector -> sci at 3k+1, dq at 3k+3.
N_DETECTORS = 16
#: Per-detector science array shape (px).
DET_SHAPE = (2040, 2040)
#: DQ bits to ignore when building the valid-pixel mask.
DQ_IGNORE = [11, 15]


def sci_ext_list(n_detectors=N_DETECTORS):
    """Science-array FITS extension indices, one per detector (3k+1)."""
    return np.arange(0, n_detectors) * 3 + 1


def dq_ext_list(n_detectors=N_DETECTORS):
    """Data-quality FITS extension indices, one per detector (3k+3)."""
    return np.arange(0, n_detectors) * 3 + 3


def det_idx_list(n_detectors=N_DETECTORS):
    """Detector indices 0..n-1 (used as det_groups for the locked-offset solve)."""
    return np.arange(0, n_detectors)


def chunk_map(n_chunks_per_side, det_shape=DET_SHAPE):
    """Square grid chunk map for a NISP detector (delegates to the generic
    geometry builder)."""
    return make_grid_chunk_map(det_shape, n_chunks_per_side)
