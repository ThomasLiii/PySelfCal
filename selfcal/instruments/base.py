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
  via ``spherex.spherex_utility.make_stripped_chunk_map``; a square grid via
  ``conventions.chunk_map`` / ``geometry.make_grid_chunk_map``).
- ``adjacency(chunk_map, ...)``            regularization adjacency pairs.
- ``offset_render_func(...) -> callable``  per-map ``(chunk_map, offset)->grid``
  renderer for the mosaic step (None ⇒ block-constant via ``chunk_to_det``).
- ``load_aux_maps(detector) -> dict``      wavelength aux maps for spectral fits
  (SPHEREx: ``{'BC':..., 'BW':...}`` from ``spherex.spherex_utility.load_calibration``;
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
    """Runner-facing instrument contract (duck-typed). The generic run engine in
    ``selfcal_scripts.runner`` drives a calibration entirely through this surface
    plus the ``CalMode`` interface — it never imports an instrument or names a
    telescope. ``capabilities`` is a set of feature tags (e.g. ``"wavelength"``,
    ``"subchannel"``) a mode can declare it ``requires``; a broadband (non-LVF)
    instrument simply omits them. See ``spherex/adapter.py`` for the reference impl.
    The older duck-typed reproject conventions above still hold for that stage."""

    name: str
    capabilities: frozenset

    def jobs(self, inst_cfg) -> list:
        """Expand the [instrument] selection into a list of jobs (channel loop)."""
        ...

    def detector_inputs(self, inst_cfg, oversample) -> dict:
        """Detector-level geometry built once per run (chunk maps, aux, edges).
        Must NOT include adjacency — that is offset-structure-specific (the mode)."""
        ...

    def channel_inputs(self, inst_cfg, det_inputs, job) -> dict:
        """Per-job geometry (valid masks + weights for the job's region)."""
        ...

    def aux(self, det_inputs) -> list | None:
        """Per-pixel aux maps spectral modes need (SPHEREx: [BC, BW]); else None."""
        ...

    def offset_render(self, inst_cfg, det_inputs, channel_inputs):
        """Per-(chunk_map, offset)->grid renderer for the mosaic (or None)."""
        ...

    def wavelength_append(self, det_inputs, mm, maps, sigma) -> None:
        """Optional: append per-pixel wavelength maps after mosaicking (LVF only)."""
        ...
