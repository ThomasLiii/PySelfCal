"""The numerical core: the sparse least-squares solve and the coadd engine.

The modules work on plain arrays and frame files and never import an instrument.
:class:`~selfcal.pipeline.pipeline_wrapper.Calibrator` drives the solve and
:class:`~selfcal.pipeline.pipeline_wrapper.Mosaicker` the coadd; the model being solved is
described in :mod:`selfcal.models`.

Building and solving the system:

- :mod:`~selfcal.core.system`: :func:`~selfcal.core.system.setup_lsqr`, which builds the sparse
  system over a process pool, and the coverage and Fisher parsers.
- :mod:`~selfcal.core.assembly`: the rows of one subframe, built in the worker processes.
- :mod:`~selfcal.core.constraint_builders`: the global constraint rows (mean-offset anchors,
  grouped adjacency, damping, user priors).
- :mod:`~selfcal.core.layout`: :class:`~selfcal.core.layout.SystemLayout`, the column layout of
  the unknowns vector.
- :mod:`~selfcal.core.solve`: :func:`~selfcal.core.solve.apply_lsqr` preconditions the system and
  runs LSQR or LSMR with a thread-parallel sparse product.
- :mod:`~selfcal.core.lsqr_inplace`: scipy's LSQR with in-place vector updates (bit-identical,
  less memory).
- :mod:`~selfcal.core.lsmr`: scipy's LSMR, with its state recorded at every iteration
  (bit-identical).
- :mod:`~selfcal.core.solve_record`: the record of a solve: how it stopped, its final estimates,
  the true residual and its history per iteration.
- :mod:`~selfcal.core.solution`: the solution vector split into maps, initial guesses, and the
  closed-form per-pixel sky solve.
- :mod:`~selfcal.core.lsqr`: the former single module, now a re-export of ``assembly``,
  ``system`` and ``solve``.

Coadding:

- :mod:`~selfcal.core.coadd`: a mosaic's mean, standard-deviation and sigma-clipped maps.
- :mod:`~selfcal.core.subframe`: loads a frame file and prepares its values, weights and chunk
  contributions, for both the solve and the coadd.

Memory and parallelism:

- :mod:`~selfcal.core.blockcsr`: int32-indexed block storage for a matrix with ``2**31`` or more
  nonzeros.
- :mod:`~selfcal.core.shmbuf`: shared-memory arrays handed explicitly to worker processes.
- :mod:`~selfcal.core.spill`: parks large per-pixel setup arrays on scratch disk during the solve.
"""
