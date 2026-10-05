"""Ready-made priors for ``sc.Model(priors=[...])``.

Each returns a :class:`~selfcal.models.model.Prior` naming one of the prior functions of
:mod:`selfcal.models.priors`; ``weight`` scales its rows. Write your own as a function of the
terms it constrains (the contract of :mod:`selfcal.models.priors`) wrapped in ``sc.Prior``.
"""
from __future__ import annotations

from .models.model import Prior

__all__ = ['frame_smoothness', 'sky_smoothness', 'toward']


def frame_smoothness(term, variable='frame', *, power=0.0, covered_only=True, weight=1.0, name=None) -> Prior:
    """Pull together the offsets of groups that neighbour in the frame variable ``variable``
    (``"frame"``, a time, ...): one row ``(x[g2] - x[g1]) / |Δv|^power`` per neighbouring pair
    of groups, per chunk and basis function (``power=0.5``: a random walk in ``variable``)."""
    return Prior('frame_smoothness', (term,), weight=weight, name=name, variable=variable, power=power,
                 covered_only=covered_only)


def sky_smoothness(term, *, covered_only=True, weight=1.0, name=None) -> Prior:
    """Pull together neighbouring pixels of the sky term ``term`` (``covered_only=False`` also
    smooths into the gaps)."""
    return Prior('sky_smoothness', (term,), weight=weight, name=name, covered_only=covered_only)


def toward(term, variable, *, covered_only=True, weight=1.0, name=None) -> Prior:
    """Pull a term's unknowns toward known values: a sky term toward the sky variable
    ``variable`` (a reference-grid map); the per-frame scalar, or every chunk of an offset term,
    toward the frame variable ``variable``."""
    return Prior('toward_variable', (term,), weight=weight, name=name, variable=variable,
                 covered_only=covered_only)
