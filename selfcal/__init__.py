"""selfcal -- sparse-LSQR self-calibration + mosaicking for astronomical imaging.

Supports SPHEREx (LVF spectral) and Euclid (broadband) data. The curated names
below are the high-level API; lower-level functions live in the submodules
(``selfcal.core``, ``selfcal.geometry``, ``selfcal.io``, ``selfcal.models``);
``selfcal.core.lsqr`` remains as a back-compat re-export of the split
assembly/system/solve modules.

Quick start::

    from selfcal import PipelineConfig, Calibrator, Mosaicker, SkyModel

    cfg = PipelineConfig(output_dir=..., run_name=..., resolution_arcsec=6.2)
    cc = Calibrator(cfg)
    cc.setup_lsqr(chunk_maps=[chunk_map], grid_valid_weight=mask, ...)
    cc.apply_lsqr(...)
    cc.save_calibration(cal_file='cal.h5')

Spectral / arbitrary-template fitting (any number of components)::

    from selfcal import SkyModel, ContinuumComponent, SpectralComponent
    from selfcal import GaussianProfile, TemplateProfile
    sky = SkyModel((ContinuumComponent(),
                    SpectralComponent(name='pah', profile=GaussianProfile(3.29, sigma))))
    cc.setup_lsqr(..., sky_model=sky, det_aux=[bc_map, bw_map])
"""
__version__ = "0.1.0"

from ._state import set_hdd_io_limit
from .pipeline.pipeline_wrapper import (PipelineConfig, Reprojector, Calibrator,
                                       Mosaicker)
from .models.sky_model import (SkyModel, SkyComponent, ContinuumComponent,
                               SpectralComponent, LineComponent)
from .models.profiles import (SpectralProfile, GaussianProfile, TemplateProfile,
                              QuadratureSigma, LineProfile)
from .models.offset_model import OffsetModel, OffsetBlock
from .core.layout import SystemLayout
from .pipeline.tiled import TiledCalibration, TileSpec, make_tile_grid
from .config import resolve_path, SelfCalConfigError

__all__ = [
    "set_hdd_io_limit",
    "PipelineConfig", "Reprojector", "Calibrator", "Mosaicker",
    "SkyModel", "SkyComponent", "ContinuumComponent", "SpectralComponent",
    "LineComponent",
    "SpectralProfile", "GaussianProfile", "TemplateProfile", "QuadratureSigma",
    "LineProfile",
    "OffsetModel", "OffsetBlock",
    "SystemLayout",
    "TiledCalibration", "TileSpec", "make_tile_grid",
    "resolve_path", "SelfCalConfigError",
]
