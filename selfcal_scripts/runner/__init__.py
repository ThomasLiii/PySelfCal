"""selfcal_scripts.runner — the generic, instrument-agnostic run engine.

A run is a TOML config + ``python -m selfcal_scripts.run --config <file>``. The
config picks an instrument (geometry), a mode (calibration recipe), and a task
(cal / tiled / reproject / precompute); the engine sequences the rest. Adding a
calibration variant is a new mode module; adding a telescope is a new instrument
adapter — neither touches the engine. See ``configs/`` for examples and the
repo's PIPELINE.md for the schema.
"""
from .config import RunConfig, load_config, get_instrument  # noqa: F401
from .pipelines import run  # noqa: F401
