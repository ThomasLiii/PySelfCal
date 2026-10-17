"""The N-pass alternating solve (``calibrate(passes=...)``, the run engine's task ``npass``): after a
first joint solve, the model's sky terms and its offsets solved in turn, each exactly given the other.

Model, per frame *k* and observation *i* (reference pixel *p(i)*)::

    d_i = Σ_j S_j(p(i)) c_j(v_i) + O_k(i) + ε_i

with J sky terms ``S_j``, each a per-pixel map times its coefficient ``c_j`` of the observation's data
variables ``v`` (``c ≡ 1`` for a constant term; see :mod:`selfcal.models.sky_model`), and ``O_k`` the
frame's offset: whatever the model's offset terms are, plus a per-frame scalar. Nothing here knows
their form: the passes take the model's rows from the engine's row assembly and subtract with the
solve's own chunk weights and functions.

- :mod:`~selfcal.npass.sky`, the SKY pass: given the offsets, each pixel's sky terms are an
  independent problem; the per-pixel sums of the data minus the offsets are added over the tiles and
  every pixel is solved once.
- :mod:`~selfcal.npass.offsets`, the OFFSET pass: given the sky, each frame's offsets are an
  independent problem; each frame's rows of the data minus the sky are reduced to its normal
  equations and every frame is solved directly.
- :mod:`~selfcal.npass.passes`: one pass over a run's tiles, in the run engine's context.

Neither pass builds the joint matrix. :mod:`selfcal.run.npass` schedules the passes.
"""
