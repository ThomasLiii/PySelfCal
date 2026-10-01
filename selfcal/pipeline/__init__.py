"""The pipeline stages as classes, and the tiled and N-pass solves built on them.

:class:`~selfcal.pipeline.pipeline_wrapper.Reprojector`,
:class:`~selfcal.pipeline.pipeline_wrapper.Calibrator` and
:class:`~selfcal.pipeline.pipeline_wrapper.Mosaicker` run the three stages (frame files, then a
``cal_*.h5`` file, then a mosaic) with :mod:`selfcal.io` and :mod:`selfcal.core`; the run engine
(:mod:`selfcal_scripts.runner`) drives them from a TOML config.

- :mod:`~selfcal.pipeline.pipeline_wrapper`: the three stage classes and
  :class:`~selfcal.pipeline.pipeline_wrapper.PipelineConfig`, the paths of a run.
- :mod:`~selfcal.pipeline.tiled`: a large field calibrated as overlapping tiles whose sky maps are
  merged with Fisher weights.
- :mod:`~selfcal.pipeline.npass`: primitives of the N-pass alternating solve: exact sky passes
  from per-pixel moments summed over tiles, and per-frame offset refits.
- :mod:`~selfcal.pipeline.model_eval`: frame hooks that apply a solved model outside the solve
  (offset terms with known functions, the observation weight).
"""
