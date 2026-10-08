"""selfcal -- sparse-LSQR self-calibration + mosaicking for any imaging telescope.

For every observation — one value of one frame on one reference-grid pixel —
the solver fits::

    data = Σ_j S_j[pixel] · c_j(v)  +  Σ_m O_m[group(frame), chunk_m, k] · φ_mk(v)  +  s(frame)

``S_j`` are sky maps, ``O_m`` offsets on chunk maps of the detector (shared by
groups of frames), ``s`` a per-frame scalar; ``c_j`` and ``φ_mk`` are ANY
functions of *data variables* ``v``: detector maps (SPHEREx: its wavelength
map), per-frame values (time, filter, a half-wave-plate angle, a temperature),
reference-grid maps, planes stored with each frame, the built-in coordinates,
or functions of those and of the frame itself.

Quick start (the Python API: everything configured in Python, checked when built)::

    import selfcal as sc

    camera = sc.Camera((64, 64), chunks=(4, 4), dq_ext=2, tag="Sim")
    field = sc.Field("quickstart_output/quickstart", camera, pixel_scale=10.0)

    if __name__ == "__main__":       # worker processes import this file again
        field.reproject("quickstart_output/exposures/sim_*.fits")
        result = field.calibrate(sc.continuum(smooth=0.1))
        print(result)

The pieces: an instrument (``sc.Camera``, ``sc.SPHEREx``, ``sc.Euclid``, or an
``sc.Instrument`` subclass), a ``sc.Field`` (one data set and its folder), a model
(``sc.continuum()``, ``sc.spectral(...)``, or ``sc.Model`` with ``sc.Sky`` and
``sc.Offsets`` terms), a ``sc.Recipe`` (the model with ``sc.Fit``, ``sc.Coadd`` and
``sc.Numerics``), and ``sc.Compute`` (the machine). Big fields add ``sc.Tiles``
and ``sc.Passes``. The library underneath (``Calibrator``, ``Mosaicker``,
``SkyModel``, ``OffsetModel``, ...) stays available for direct use; the run
engine behind the actions is :mod:`selfcal.run`.

Names load on first use, so ``import selfcal`` (and every worker process) stays light.
"""
from __future__ import annotations

__version__ = "0.1.0"

import importlib as _importlib
import logging as _logging
from typing import TYPE_CHECKING

# Library convention: selfcal never configures logging for the application.
# Every module logs to logging.getLogger(__name__) under this "selfcal" root;
# the NullHandler silences it until the application attaches its own handler
# (e.g. logging.basicConfig(level=logging.INFO, format="%(message)s") for the
# plain console output the pipeline scripts use). The Python API's actions show
# the messages on stdout when the application has configured none.
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

# name -> the module that defines it (loaded on first access: PEP 562)
_EXPORTS = {
    # the Python API
    **dict.fromkeys(('Config', 'ConfigError', 'by_value'), 'selfcal.config'),
    **dict.fromkeys(('Function', 'template', 'gaussian', 'linear', 'catalog', 'Poly', 'Sky', 'Offsets', 'Header',
                     'PerFrame', 'DetectorMap', 'SkyMap', 'SolvedSky', 'Layer', 'Derived', 'FrameFunction', 'Prior',
                     'Model', 'continuum', 'spectral', 'two_block'), 'selfcal.models.model'),
    **dict.fromkeys(('ChunkGroups', 'Clip', 'Fit', 'Coadd', 'Numerics', 'Recipe'), 'selfcal.run.recipe'),
    **dict.fromkeys(('Tiles', 'Refit', 'Passes'), 'selfcal.run.schedule'),
    **dict.fromkeys(('Compute', 'Tuning'), 'selfcal.run.compute'),
    **dict.fromkeys(('Field', 'frames_in', 'Submitted'), 'selfcal.run.field'),
    'rerun': 'selfcal.run.records',
    'compare': 'selfcal.run.compare',
    **dict.fromkeys(('Result', 'MosaicFile'), 'selfcal.run.result'),
    'Plan': 'selfcal.run.plan',
    **dict.fromkeys(('Instrument', 'Job', 'Geometry', 'ChunkMap', 'ChunkAxes', 'JobGeometry', 'ExposureLayout'),
                    'selfcal.instruments.contract'),
    'Camera': 'selfcal.instruments.camera',
    'SPHEREx': 'selfcal.instruments.spherex.settings',
    'Euclid': 'selfcal.instruments.euclid.settings',
    # the library
    **dict.fromkeys(('set_hdd_io_limit', 'set_progress'), 'selfcal._state'),
    **dict.fromkeys(('PipelineConfig', 'Reprojector', 'Calibrator', 'Mosaicker'), 'selfcal.pipeline.pipeline_wrapper'),
    **dict.fromkeys(('SkyModel', 'SkyComponent', 'ContinuumComponent', 'SpectralComponent', 'LineComponent',
                     'Coefficient', 'ImportedFunction'), 'selfcal.models.sky_model'),
    **dict.fromkeys(('SpectralProfile', 'GaussianProfile', 'LinearProfile', 'TemplateProfile', 'QuadratureSigma',
                     'LineProfile'), 'selfcal.models.profiles'),
    **dict.fromkeys(('OffsetModel', 'OffsetBlock', 'Basis'), 'selfcal.models.offset_model'),
    'VariableSet': 'selfcal.models.variables',
    'SystemLayout': 'selfcal.core.layout',
    **dict.fromkeys(('TiledCalibration', 'TileSpec', 'make_tile_grid'), 'selfcal.pipeline.tiled'),
    **dict.fromkeys(('resolve_path', 'SelfCalConfigError'), 'selfcal.config'),
}

__all__ = sorted(_EXPORTS) + ['priors', 'run']


def __getattr__(name):
    if name in ('priors', 'run'):
        return _importlib.import_module(f'selfcal.{name}')
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'selfcal' has no attribute {name!r}")
    value = getattr(_importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_EXPORTS))


if TYPE_CHECKING:                     # what the names are, for type checkers and the documentation
    from ._state import set_hdd_io_limit, set_progress
    from .config import Config, ConfigError, SelfCalConfigError, by_value, resolve_path
    from .core.layout import SystemLayout
    from .instruments.camera import Camera
    from .instruments.contract import (
        ChunkAxes,
        ChunkMap,
        ExposureLayout,
        Geometry,
        Instrument,
        Job,
        JobGeometry,
    )
    from .instruments.euclid.settings import Euclid
    from .instruments.spherex.settings import SPHEREx
    from .models.model import (
        Derived,
        DetectorMap,
        FrameFunction,
        Function,
        Header,
        Layer,
        Model,
        Offsets,
        PerFrame,
        Poly,
        Prior,
        Sky,
        SkyMap,
        SolvedSky,
        catalog,
        continuum,
        gaussian,
        linear,
        spectral,
        template,
        two_block,
    )
    from .models.offset_model import Basis, OffsetBlock, OffsetModel
    from .models.profiles import (
        GaussianProfile,
        LinearProfile,
        LineProfile,
        QuadratureSigma,
        SpectralProfile,
        TemplateProfile,
    )
    from .models.sky_model import (
        Coefficient,
        ContinuumComponent,
        ImportedFunction,
        LineComponent,
        SkyComponent,
        SkyModel,
        SpectralComponent,
    )
    from .models.variables import VariableSet
    from .pipeline.pipeline_wrapper import Calibrator, Mosaicker, PipelineConfig, Reprojector
    from .pipeline.tiled import TiledCalibration, TileSpec, make_tile_grid
    from .run.compare import compare
    from .run.compute import Compute, Tuning
    from .run.field import Field, Submitted, frames_in
    from .run.plan import Plan
    from .run.recipe import ChunkGroups, Clip, Coadd, Fit, Numerics, Recipe
    from .run.records import rerun
    from .run.result import MosaicFile, Result
    from .run.schedule import Passes, Refit, Tiles
