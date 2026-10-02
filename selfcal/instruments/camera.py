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
from .contract import Instrument, Job
from .grid import GridInstrument

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
        return f"{self.tag or f'Grid{self.shape[0]}x{self.shape[1]}'}_Chunks{ny}x{nx}"

    def default_jobs(self):
        return (Job(self.job),)

    def check_jobs(self, jobs):
        if len(jobs) != 1 or jobs[0].name != self.job:
            raise ConfigError(f"Camera: one job, {self.job!r}; got {[getattr(j, 'name', j) for j in jobs]}")

    def table(self) -> dict:
        """The ``[instrument]`` table of the ``grid`` instrument for this camera."""
        table = {'name': 'grid', 'detector_shape': list(self.shape), 'chunks': list(self.chunks),
                 'sci_ext': self.sci_ext}
        if self.dq_ext is not None:
            table['dq_ext'] = self.dq_ext
        if self.reference_ext is not None:
            table['ref_use_ext'] = list(self.reference_ext)
        if self.tag is not None:
            table['tag'] = self.tag
        if self.job != 'All':
            table['job_name'] = self.job
        if self.unit:
            table['unit'] = self.unit
        return table

    def engine(self, jobs):
        """``("grid", table)``, or an engine object when the camera has a reader, detector maps or
        header variables (which the ``grid`` table cannot hold)."""
        if self.reader is None and not self.detector_maps and not self.headers:
            return 'grid', self.table()
        return _CameraEngine(self), self.table()


class _CameraEngine(GridInstrument):
    """The ``grid`` instrument with a camera's reader, detector maps and header variables."""

    def __init__(self, camera):
        self.camera = camera
        self.name = 'grid'

    def __reduce__(self):
        return (_CameraEngine, (self.camera,))

    def exposure_layout(self, inst_cfg):
        layout = super().exposure_layout(inst_cfg)
        if self.camera.reader is None:
            return layout
        from dataclasses import replace
        return replace(layout, reader=self.camera.reader)

    def detector_geometry(self, inst_cfg, oversample):
        geom = super().detector_geometry(inst_cfg, oversample)
        if not self.camera.detector_maps:
            return geom
        from dataclasses import replace
        return replace(geom, aux={k: np.asarray(v, dtype=np.float32) for k, v in self.camera.detector_maps.items()})

    def frame_variable_names(self, inst_cfg):
        return ('exposure', 'detector') + tuple(self.camera.headers)

    def frame_variables(self, frames, inst_cfg=None):
        out = super().frame_variables(frames, inst_cfg)
        if self.camera.headers:
            from ..io.frames import frame_header_values
            values = frame_header_values(frames, [h.key for h in self.camera.headers.values()])
            for name, h in self.camera.headers.items():
                arr = values[h.key]
                if h.default is not None:
                    missing = (np.array([x is None for x in arr]) if arr.dtype == object else ~np.isfinite(arr))
                    arr = arr.copy()
                    arr[missing] = h.default
                out[name] = arr
        return out
