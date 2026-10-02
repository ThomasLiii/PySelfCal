"""selfcal -- sparse-LSQR self-calibration + mosaicking for any imaging telescope.

For every observation — one value of one frame on one reference-grid pixel —
the solver fits::

    data = Σ_j S_j[pixel] · c_j(v)  +  Σ_m O_m[group(frame), chunk_m, k] · φ_mk(v)  +  s(frame)

``S_j`` are sky maps, ``O_m`` offsets on chunk maps of the detector (shared by
groups of frames), ``s`` a per-frame scalar; ``c_j`` and ``φ_mk`` are ANY
functions of *data variables* ``v``: detector maps (SPHEREx: its wavelength
map), per-frame values (time, filter, a half-wave-plate angle, a temperature),
reference-grid maps, planes stored with each frame, the built-in coordinates,
or functions of those and of the frame itself (``selfcal.models.variables``).
Priors are built in (damping, smoothness, polynomial shape, anchors) or any
linear rows a function returns (``selfcal.models.priors``). A telescope enters
through an ``Instrument`` (``selfcal.instruments``: SPHEREx, Euclid, a
configurable grid imager, or your own subclass with a reader for its raw
format), or by writing frames directly (``selfcal.io.frames.write_frame``).

Quick start::

    from selfcal import PipelineConfig, Calibrator, OffsetModel, OffsetBlock

    cfg = PipelineConfig(output_dir=..., run_name=..., resolution_arcsec=6.2)
    cc = Calibrator(cfg)
    cc.setup_lsqr(offset_model=OffsetModel([OffsetBlock(chunk_map=chunk_map)]),
                  grid_valid_weight=mask, ...)
    cc.apply_lsqr(...)
    cc.save_calibration(cal_file='cal.h5')

A sky term modulated by the season, read from each frame's time::

    from selfcal import SkyModel, SkyComponent, Coefficient, VariableSet
    annual = SkyComponent('annual', Coefficient('time', my_sine))     # any callable
    cc.setup_lsqr(..., sky_model=SkyModel((SkyComponent('continuum'), annual)),
                  variables=VariableSet(frame={'time': mjd_per_frame}))

The run engine (``selfcal.run``) drives all of this from a TOML
config with a ``[model]`` table; see ``docs/bring_your_own_telescope.md``.
"""
__version__ = "0.1.0"

import logging as _logging

# Library convention: selfcal never configures logging for the application.
# Every module logs to logging.getLogger(__name__) under this "selfcal" root;
# the NullHandler silences it until the application attaches its own handler
# (e.g. logging.basicConfig(level=logging.INFO, format="%(message)s") for the
# plain console output the pipeline scripts use).
_logging.getLogger(__name__).addHandler(_logging.NullHandler())

from ._state import set_hdd_io_limit, set_progress
from .pipeline.pipeline_wrapper import (PipelineConfig, Reprojector, Calibrator,
                                       Mosaicker)
from .models.sky_model import (SkyModel, SkyComponent, ContinuumComponent,
                               SpectralComponent, LineComponent)
from .models.profiles import (SpectralProfile, GaussianProfile, LinearProfile,
                              TemplateProfile, QuadratureSigma, LineProfile)
from .models.offset_model import OffsetModel, OffsetBlock, Basis
from .models.sky_model import Coefficient, ImportedFunction
from .models.variables import VariableSet, Derived, FrameFunction
from .core.layout import SystemLayout
from .pipeline.tiled import TiledCalibration, TileSpec, make_tile_grid
from .config import resolve_path, SelfCalConfigError

__all__ = [
    "set_hdd_io_limit", "set_progress",
    "PipelineConfig", "Reprojector", "Calibrator", "Mosaicker",
    "SkyModel", "SkyComponent", "ContinuumComponent", "SpectralComponent",
    "LineComponent",
    "SpectralProfile", "GaussianProfile", "LinearProfile", "TemplateProfile",
    "QuadratureSigma", "LineProfile",
    "OffsetModel", "OffsetBlock", "Basis",
    "Coefficient", "ImportedFunction",
    "VariableSet", "Derived", "FrameFunction",
    "SystemLayout",
    "TiledCalibration", "TileSpec", "make_tile_grid",
    "resolve_path", "SelfCalConfigError",
]
