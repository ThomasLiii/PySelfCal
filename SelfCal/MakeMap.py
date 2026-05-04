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
from .lsqr import setup_lsqr, apply_lsqr, parse_pixel_counts  # noqa: F401

# --- Solution utilities ---
from .solution import parse_x, encode_x, compute_x0_from_Ab  # noqa: F401
