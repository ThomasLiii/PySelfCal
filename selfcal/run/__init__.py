"""selfcal.run -- the run engine: one run (a field, a recipe, jobs) from configuration to products.

The engine is instrument- and mode-agnostic. A run is resolved once into a
:class:`~selfcal.run.engine.RunContext` (the instrument, the mode, the detector
geometry and every product name) and executed by tasks (:mod:`selfcal.run.pipelines`:
``cal``, optionally tiled; ``mosaic``; ``npass``; ``reproject``; ``precompute``) built on two
primitives, one joint solve and one coadd. A run is described by a TOML config read by
:func:`~selfcal.run.config.load_config` (``python -m selfcal_scripts.run --config <file>``).
Adding a calibration variant is a new mode (:mod:`selfcal.run.modes`); adding a telescope is
an instrument (:mod:`selfcal.instruments`); neither touches the engine.

The engine was ``selfcal_scripts.runner`` before; that import path is an alias of this package.
"""
from .config import RunConfig, get_instrument, load_config  # noqa: F401
from .pipelines import run  # noqa: F401
