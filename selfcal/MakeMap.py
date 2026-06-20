"""
MakeMap -- backward-compatible re-export shim.

All functionality has been split into focused submodules:
  _state.py    - module-level state (semaphores, locks)
  io.py        - HDF5 I/O (load_reproj_file)
  reproject.py - batch reprojection
  subframe.py  - shared subframe preparation
  coadd.py     - co-addition (mean, std, sigma-clip)
  lsqr.py      - LSQR/LSMR matrix build & solve
  solution.py  - solution vector parse/encode utilities

Import from here for backward compatibility, or from the
submodules directly for lighter-weight imports.
"""

# --- State & config ---
from ._state import set_hdd_io_limit  # noqa: F401

# --- I/O ---
from .io import load_reproj_file  # noqa: F401

# --- Reprojection ---
from .reproject import batch_reproject  # noqa: F401

# --- Co-addition ---
from .coadd import compute_coadd_map  # noqa: F401

# --- LSQR ---
from .lsqr import (setup_lsqr, apply_lsqr, parse_pixel_counts, parse_pixel_fisher,  # noqa: F401
                   parse_pixel_counts_sky, parse_pixel_fisher_sky, apply_line_fisher_mask)  # noqa: F401

# --- Solution utilities ---
from .solution import parse_x, parse_x_sky, encode_x, compute_x0_from_Ab, compute_x0_scalar_only  # noqa: F401

# --- Layout + offset model (refactor/selfcal-package) ---
from .layout import SystemLayout  # noqa: F401
from .offset_model import OffsetModel, OffsetBlock  # noqa: F401

# --- Sky model + line profiles (refactor/selfcal-package, Phase 3) ---
from .sky_model import SkyModel, SkyComponent, ContinuumComponent, LineComponent  # noqa: F401
from .profiles import GaussianProfile, TemplateProfile, QuadratureSigma, LineProfile  # noqa: F401
