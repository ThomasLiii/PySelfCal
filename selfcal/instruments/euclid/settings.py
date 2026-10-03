"""Euclid NISP as settings: ``sc.Euclid(band="Y")``."""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Literal

from ...config.base import ConfigError
from ..contract import Instrument, Job
from . import conventions as ec

__all__ = ['Euclid']


@dataclass(frozen=True)
class Euclid(Instrument):
    """Euclid NISP: ``detectors`` detectors of ``det_shape`` pixels per exposure, in the photometric
    ``band``. Chunk maps: a square grid of ``chunks`` x ``chunks`` (primary), row and column stripes
    of ``strips`` (default ``chunks``) and tilted stripes of ``tilt_strips``; detector edges are
    zeroed over ``edge_zero_px`` and tapered over ``edge_ramp_px``. ``reference_ext``: the
    extensions whose WCS define the reference grid. ``tag``: the products' tag (with the band:
    ``cal_EDFN_Y<suffix>.h5``). One job, named after the band."""
    _: KW_ONLY
    band: Literal['Y', 'J', 'H'] = 'Y'
    chunks: int = 40
    strips: int | None = None
    tilt_strips: int = 60
    edge_zero_px: int = 0
    edge_ramp_px: int = 0
    detectors: int = ec.N_DETECTORS
    det_shape: tuple[int, int] = tuple(ec.DET_SHAPE)
    reference_ext: tuple[int, ...] = (1, 10, 37, 46)
    tag: str = 'EDFN'

    def _validate(self):
        for k in ('chunks', 'tilt_strips', 'detectors'):
            if getattr(self, k) < 1:
                raise ConfigError(f"Euclid({k}={getattr(self, k)}): at least 1")
        if self.strips is not None and self.strips < 1:
            raise ConfigError(f"Euclid(strips={self.strips}): at least 1")

    @property
    def product_tag(self) -> str:
        return self.tag

    def default_jobs(self):
        return (Job(self.band),)

    def default_ignore_flags(self):
        return tuple(ec.DQ_IGNORE)

    def check_jobs(self, jobs):
        if len(jobs) != 1 or jobs[0].name != self.band:
            raise ConfigError(f"Euclid: one job, the band ({self.band!r}); got {[getattr(j, 'name', j) for j in jobs]}")

    def engine(self, jobs):
        """``("euclid", table)``: the registered instrument and its ``[instrument]`` table."""
        table = {'name': 'euclid', 'band': self.band, 'chunks': self.chunks, 'tilt_strips': self.tilt_strips,
                 'edge_zero_px': self.edge_zero_px, 'edge_ramp_px': self.edge_ramp_px,
                 'detectors': self.detectors, 'det_shape': list(self.det_shape),
                 'ref_use_ext': list(self.reference_ext), 'tag': self.tag}
        if self.strips is not None:
            table['strips'] = self.strips
        return 'euclid', table
