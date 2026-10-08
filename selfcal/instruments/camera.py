"""A camera as settings: ``sc.Camera(shape, chunks=...)``, any imager without code.

A detector of ``shape`` pixels cut into a grid of rectangular chunks, read from FITS files
(or by your own ``reader``); optionally with per-pixel detector maps and per-frame header
values as data variables. Most cameras need nothing else; a camera with another chunk geometry
subclasses :class:`~selfcal.instruments.contract.Instrument`.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import Any, Callable

import numpy as np

from ..config.base import ConfigError, FrozenDict
from ..config.functions import function_ref
from ..models.model import Header
from .base import ExposureLayout
from .contract import ChunkMap, Geometry, Instrument, Job

__all__ = ['Camera']


@dataclass(frozen=True)
class Camera(Instrument):
    """An imager: a detector of ``shape`` = (rows, cols) pixels in ``chunks`` = (rows, cols)
    rectangular chunks (one number: square).

    Exposure files: the science image in FITS extension ``sci_ext``, the data-quality mask in
    ``dq_ext`` (None: every pixel valid), the reference grid from the WCS of ``reference_ext``
    (default ``(sci_ext,)``); or any format, read by ``reader(path, sci_ext, dq_ext,
    header_only=False)`` returning :class:`~selfcal.io.frames.ExposureData`. ``detector_maps``:
    per-pixel data variables (``{name: array}``, the detector's shape); ``headers``: per-frame data
    variables read from each frame's header (``{name: "KEYWORD"}`` or ``sc.Header(...)``).
    ``tag``: the products' tag (default ``Grid<rows>x<cols>``; products are named
    ``cal_<tag>_Chunks<ny>x<nx>_<job>...``); ``unit``: the mosaic's ``BUNIT``; ``job``: the job's
    name.
    """
    shape: tuple[int, int]
    _: KW_ONLY
    chunks: tuple[int, int] | int = (4, 4)
    sci_ext: int = 1
    dq_ext: int | None = None
    reference_ext: tuple[int, ...] | None = None
    reader: Callable | None = None
    detector_maps: dict[str, Any] = field(default_factory=FrozenDict)
    headers: dict[str, Any] = field(default_factory=FrozenDict)
    tag: str | None = None
    unit: str = ''
    job: str = 'All'

    # A constant, not a setting (without an annotation it is no dataclass field): the geometry reads
    # the settings only (the detector maps are arrays, fingerprinted by their content), so the engine
    # may keep it between actions.
    geometry_is_pure = True

    def _validate(self):
        if isinstance(self.chunks, int):
            object.__setattr__(self, 'chunks', (self.chunks, self.chunks))
        if min(self.shape) < 1 or min(self.chunks) < 1:
            raise ConfigError(f"Camera(shape={self.shape}, chunks={self.chunks}): positive sizes")
        if self.reader is not None:
            function_ref(self.reader, 'Camera(reader=...)')
        for name, m in self.detector_maps.items():
            if np.shape(m) != tuple(self.shape):
                raise ConfigError(f"Camera(detector_maps={{{name!r}: ...}}): shape {np.shape(m)}, the detector is "
                                  f"{tuple(self.shape)}")
        headers = {}
        for name, h in self.headers.items():
            if isinstance(h, str):
                h = Header(h)
            if not isinstance(h, Header):
                raise ConfigError(f"Camera(headers={{{name!r}: ...}}): a header keyword or sc.Header(...)")
            headers[name] = h
        object.__setattr__(self, 'headers', FrozenDict(headers))

    @property
    def product_tag(self) -> str:
        ny, nx = self.chunks
        return f"{self.tag if self.tag is not None else f'Grid{self.shape[0]}x{self.shape[1]}'}_Chunks{ny}x{nx}"

    def default_jobs(self):
        return (Job(self.job),)

    def check_jobs(self, jobs):
        if len(jobs) != 1 or jobs[0].name != self.job:
            raise ConfigError(f"Camera: one job, {self.job!r}; got {[getattr(j, 'name', j) for j in jobs]}")

    # ---- the contract -------------------------------------------------------------------
    def geometry(self, oversample=1):
        """One ``ny x nx`` rectangular chunk map named ``grid`` (``int32``, chunk id ``row * nx +
        col``; :meth:`~selfcal.instruments.base.ChunkMap.rectangles`), replicated onto the
        reference grid in ``oversample x oversample`` blocks. Its axes are ``row`` (scanned
        vertically) and ``col`` (horizontally); the standard offset block regularises along both,
        and ``row`` is the default group axis of a polynomial-basis offset term. The detector maps
        are the aux maps (``float32``); there is no wavelength map."""
        grid = ChunkMap.rectangles('grid', self.shape, self.chunks)
        return Geometry(self.shape, oversample, maps=[grid],
                        aux={k: np.asarray(v, dtype=np.float32) for k, v in self.detector_maps.items()})

    def layout(self) -> ExposureLayout:
        """A file holding one detector: the science image in ``sci_ext``, the data-quality mask in
        ``dq_ext`` (None or negative: no mask, every pixel valid), the reference frame from the WCS
        of ``reference_ext`` (default ``(sci_ext,)``); read by ``reader`` (None: FITS extensions).
        The header cache is ``headers_<tag>`` (``headers_grid`` without a tag)."""
        return ExposureLayout(
            sci_ext=[self.sci_ext],
            dq_ext=None if self.dq_ext is None or self.dq_ext < 0 else [self.dq_ext],
            detector_ids=[0],
            ref_use_ext=tuple(self.reference_ext) if self.reference_ext is not None else (self.sci_ext,),
            cache_tag=f"headers_{self.tag if self.tag is not None else 'grid'}",
            reader=self.reader)

    def frame_variable_names(self):
        return ('exposure', 'detector') + tuple(self.headers)

    def frame_variables(self, frames):
        """``exposure`` and ``detector`` (from the frame file names), and each of ``headers`` read
        from the frames' stored headers (its default where a frame lacks it)."""
        out = super().frame_variables(frames)
        if self.headers:
            from ..io.frames import frame_header_values
            values = frame_header_values(frames, [h.key for h in self.headers.values()])
            for name, h in self.headers.items():
                arr = values[h.key]
                if h.default is not None:
                    missing = (np.array([x is None for x in arr]) if arr.dtype == object else ~np.isfinite(arr))
                    arr = arr.copy()
                    arr[missing] = h.default
                out[name] = arr
        return out
