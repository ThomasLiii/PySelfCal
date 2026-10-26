"""The pipeline stages as classes, and the tiled solve built on them (the N-pass solve is
:mod:`selfcal.npass`).

:class:`~selfcal.pipeline.pipeline_wrapper.Reprojector`,
:class:`~selfcal.pipeline.pipeline_wrapper.Calibrator` and
:class:`~selfcal.pipeline.pipeline_wrapper.Mosaicker` run the three stages (frame files, then a
``cal_*.h5`` file, then a mosaic) with :mod:`selfcal.io` and :mod:`selfcal.core`; the run engine
(:mod:`selfcal.run`) drives them for the Python API's actions.

- :mod:`~selfcal.pipeline.pipeline_wrapper`: the three stage classes and
  :class:`~selfcal.pipeline.pipeline_wrapper.PipelineConfig`, the paths of a run.
- :mod:`~selfcal.pipeline.tiled`: a large field calibrated as overlapping tiles whose sky maps are
  merged with Fisher weights.
- :mod:`~selfcal.pipeline.model_eval`: frame hooks that apply a solved model outside the solve
  (offset terms with known functions, the observation weight).
"""
