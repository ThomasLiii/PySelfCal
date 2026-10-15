"""selfcal.run -- the Python API's actions and the run engine behind them.

An action of a :class:`~selfcal.run.field.Field` (``calibrate``, ``mosaic``, ``reproject``) plans
first (:mod:`selfcal.run.plan`), lowers its settings to engine runs
(:class:`~selfcal.run.runspec.RunSpec`, :mod:`selfcal.run.lower`) and runs them: each resolved once
into a :class:`~selfcal.run.engine.RunContext` (the instrument, the detector geometry, the model and
every product name) and executed by a task (:mod:`selfcal.run.pipelines`: ``cal``, optionally tiled;
``mosaic``; ``npass``; ``reproject``) built on two primitives, one joint solve and one coadd. The
engine is instrument-agnostic: a telescope is an instrument object
(:class:`~selfcal.instruments.contract.Instrument`) and a calibration variant a model
(:class:`~selfcal.models.model.Model`); neither touches the engine.
"""
