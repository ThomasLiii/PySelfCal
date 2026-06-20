"""The Instrument convention — documentation-grade, duck-typed (not enforced).

The selfcal *core* (reprojection, solver, mosaicker) takes plain arrays and
callables; it never imports ``selfcal.instruments``. An "instrument" is simply a
module/namespace that supplies those inputs for a given telescope. This file
documents the conventions the in-tree instruments follow so a third instrument
knows the minimum to provide; it intentionally defines a ``typing.Protocol``
rather than a base class to subclass — conform by duck typing.

What an instrument typically provides (all optional except a name):

- ``default_ext_lists() -> (sci, dq)``    FITS science / data-quality extension
  indices per detector (e.g. Euclid ``conventions.sci_ext_list`` / ``dq_ext_list``).
- ``default_ignore_bits() -> list[int]``  DQ bits to ignore (e.g. Euclid [11,15]).
- ``build_chunk_map(...) -> np.ndarray``   detector→chunk-id map (SPHEREx LVF arcs
  via ``spherex.SPHERExUtility.make_stripped_chunk_map``; a square grid via
  ``conventions.chunk_map`` / ``geometry.make_grid_chunk_map``).
- ``adjacency(chunk_map, ...)``            regularization adjacency pairs.
- ``offset_render_func(...) -> callable``  per-map ``(chunk_map, offset)->grid``
  renderer for the mosaic step (None ⇒ block-constant via ``chunk_to_det``).
- ``load_aux_maps(detector) -> dict``      wavelength aux maps for spectral fits
  (SPHEREx: ``{'BC':..., 'BW':...}`` from ``spherex.SPHERExUtility.load_calibration``;
  broadband instruments: ``{}``).
- ``line_catalog()``                       named spectral components / windows
  (SPHEREx only; e.g. a PAH-3.29 factory). Broadband instruments omit this and
  use the default continuum-only SkyModel.

A minimal new instrument needs only: an exposure-list loader, its ext/DQ
conventions, and a chunk-map recipe (reuse ``geometry.make_grid_chunk_map`` if a
regular grid suffices). The synthetic instrument in the test suite is the
living reference for that minimum.
"""
from typing import Protocol, runtime_checkable


@runtime_checkable
class Instrument(Protocol):
    name: str
