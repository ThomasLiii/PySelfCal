"""Instruments: the one place a telescope enters the pipeline.

An instrument is a settings object of the Python API (:mod:`selfcal.instruments.contract`):
``sc.SPHEREx`` (:mod:`~selfcal.instruments.spherex`), ``sc.Euclid``
(:mod:`~selfcal.instruments.euclid`), ``sc.Camera`` (:mod:`~selfcal.instruments.camera`), or a
subclass of ``sc.Instrument`` of your own. The run engine calls its methods; the geometry it
returns is made of the types of :mod:`~selfcal.instruments.base`.
"""
from .base import ChunkMap, DetectorGeometry, ExposureLayout, JobGeometry  # noqa: F401

__all__ = ['ChunkMap', 'DetectorGeometry', 'JobGeometry', 'ExposureLayout']
