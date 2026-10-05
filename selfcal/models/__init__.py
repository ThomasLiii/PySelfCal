"""The self-calibration model: data variables, sky terms, offset terms and priors.

:class:`~selfcal.models.spec.ModelSpec` describes a whole model as data (the ``[model]`` table
of a run config, or a mode's preset) and lowers it into the arguments of
:meth:`~selfcal.pipeline.pipeline_wrapper.Calibrator.setup_lsqr`: a
:class:`~selfcal.models.sky_model.SkyModel`, an :class:`~selfcal.models.offset_model.OffsetModel`,
a :class:`~selfcal.models.variables.VariableSet`, the observation weight and the priors. The
solver itself is :mod:`selfcal.core`.

- :mod:`~selfcal.models.spec`: :class:`~selfcal.models.spec.ModelSpec`, the model as data, built
  from a config table, checked against an instrument and lowered for the solver.
- :mod:`~selfcal.models.variables`: data variables, the named per-observation quantities every
  function of the model reads, and their sources.
- :mod:`~selfcal.models.sky_model`: the sky terms, each a per-pixel map times a known
  coefficient of data variables.
- :mod:`~selfcal.models.profiles`: ready-made coefficient functions (Gaussian, tabulated, linear).
- :mod:`~selfcal.models.offset_model`: the offset blocks of a solve, one per chunk map, lowered to
  the keyword arguments of :func:`~selfcal.core.system.setup_lsqr`.
- :mod:`~selfcal.models.offset_structure`: adjacency pairs, polynomial chains and polynomial bases
  built along the axes of a chunk grid.
- :mod:`~selfcal.models.offset_basis`: the mean-zero Chebyshev basis of a hard polynomial offset.
- :mod:`~selfcal.models.priors`: priors as functions that return linear rows on the unknowns,
  and ready-made ones.
"""
