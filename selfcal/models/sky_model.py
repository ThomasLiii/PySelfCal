"""SkyModel — the per-pixel sky model solved by the LSQR system.

Generalizes the hardcoded ``num_sky_blocks`` integer (1 = continuum only,
2 = continuum + one PAH Gaussian line) into an ordered list of named
:class:`SkyComponent` objects. Each component contributes one sky block of
``num_sky = ref_h * ref_w`` columns; the data row for one observation of
reference pixel ``P`` gains one nnz per component::

    data_i = w_i * Σ_c coeff_c(λ_i) * sky_c[P]  +  offsets + scalar

where ``coeff_c`` is the component's per-observation coefficient:
``None`` (identity, ``coeff = 1`` → store ``w_i`` directly, the bit-exact
continuum path) for :class:`ContinuumComponent`, or the line profile ``G(λ_i)``
for :class:`LineComponent`.

Bit-identity: ``SkyModel.continuum_only()`` reproduces the legacy
``num_sky_blocks==1`` emission and ``SkyModel.continuum_plus_gaussian_line()``
reproduces ``num_sky_blocks==2`` (component order [continuum, line], same
interleave, same float ops). The row assembly (``selfcal.core.assembly``)
consumes this model when emitting the per-observation sky coefficients; this
module holds no assembly logic itself.

Components are small frozen dataclasses (scalars + a profile holding at most a
small template array) so they pickle cleanly into the multiprocessing task
dicts; large arrays travel via SHM, never inside a component.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

__all__ = [
    'SkyComponent',
    'ContinuumComponent',
    'SpectralComponent',
    'LineComponent',
    'SkyModel',
]


@dataclass(frozen=True)
class SkyComponent:
    """Base class / duck-typed protocol for a sky block.

    Attributes / methods a component must provide:
      - ``name``: unique str (h5 group key, layout key).
      - ``aux_requirements``: tuple of aux-map keys it samples (e.g. ('BC','BW')).
      - ``damp_weight``: per-component Tikhonov weight or None (use solver default).
      - ``coefficients(aux) -> np.ndarray | None``: per-observation coefficient,
        vectorized over a subframe's valid pixels. ``None`` means the identity
        coefficient (the assembly stores the pixel weight directly — no multiply).
    """

    name: str
    aux_requirements: tuple = ()
    damp_weight: float = None

    def coefficients(self, aux: dict[str, np.ndarray]) -> np.ndarray | None:  # pragma: no cover - interface
        """Per-observation coefficient for this component's sky block.

        Parameters
        ----------
        aux : dict[str, np.ndarray]
            Auxiliary maps sampled at the subframe's valid pixels, keyed by
            aux-map name (the keys listed in ``aux_requirements``).

        Returns
        -------
        np.ndarray or None
            The per-pixel coefficient ``coeff_c(λ)`` for each valid pixel, or
            ``None`` to signal the identity coefficient (the row assembly then
            stores the pixel weight directly, no multiply).
        """
        raise NotImplementedError


@dataclass(frozen=True)
class ContinuumComponent(SkyComponent):
    name: str = 'continuum'
    aux_requirements: tuple = ()
    damp_weight: float = None

    def coefficients(self, aux: dict[str, np.ndarray]) -> None:
        """Return ``None`` (the identity coefficient) for the continuum block.

        Parameters
        ----------
        aux : dict[str, np.ndarray]
            Auxiliary maps at the valid pixels (unused; continuum needs none).

        Returns
        -------
        None
            Always ``None`` — signals the assembly to store ``valid_weight``
            directly, preserving the legacy single-block fast path.
        """
        # Identity coefficient: the row assembly stores valid_weight directly
        # (no multiply-by-1.0), preserving the legacy single-block fast path.
        return None


@dataclass(frozen=True)
class SpectralComponent(SkyComponent):
    """A spectral sky block fitting the per-pixel amplitude of an arbitrary
    spectral template (analytical or numerical).

    The coefficient at each observation is ``profile(λ)`` for any
    :class:`~selfcal.models.profiles.SpectralProfile` (Gaussian, numerical template,
    Lorentzian/Voigt, ...). Not line-specific — a "line" is just the common case
    of a peaked profile.
    """

    name: str = 'spectral'
    profile: object = None
    wavelength_key: str = 'BC'
    damp_weight: float = None
    aux_requirements: tuple = field(default=(), init=False)

    def __post_init__(self):
        reqs = (self.wavelength_key,)
        prof_reqs = getattr(self.profile, 'aux_requirements', ())
        for k in prof_reqs:
            if k not in reqs:
                reqs = reqs + (k,)
        object.__setattr__(self, 'aux_requirements', reqs)

    def coefficients(self, aux: dict[str, np.ndarray]) -> np.ndarray:
        """Evaluate the profile at this component's wavelength key.

        Parameters
        ----------
        aux : dict[str, np.ndarray]
            Auxiliary maps at the valid pixels; must contain ``wavelength_key``
            and any keys the profile's ``aux_requirements`` declares.

        Returns
        -------
        np.ndarray
            The float32 per-pixel coefficient ``profile(λ)``.
        """
        return self.profile.evaluate(aux[self.wavelength_key], aux)


# Back-compat alias (the abstraction is general, not line-specific).
LineComponent = SpectralComponent


@dataclass(frozen=True)
class SkyModel:
    """Ordered tuple of :class:`SkyComponent` (default: continuum only)."""

    components: tuple = (ContinuumComponent(),)

    def __post_init__(self):
        object.__setattr__(self, 'components', tuple(self.components))
        if len(self.components) < 1:
            raise ValueError("SkyModel needs at least one component")
        names = [c.name for c in self.components]
        if len(names) != len(set(names)):
            raise ValueError(f"duplicate sky-component names: {names}")

    @property
    def n_blocks(self) -> int:
        """Number of sky blocks (one per component)."""
        return len(self.components)

    @property
    def names(self) -> list[str]:
        """Ordered list of component names (one per sky block)."""
        return [c.name for c in self.components]

    @property
    def aux_requirements(self) -> tuple[str, ...]:
        """Ordered union of all components' aux requirements."""
        seen = []
        for c in self.components:
            for k in getattr(c, 'aux_requirements', ()):
                if k not in seen:
                    seen.append(k)
        return tuple(seen)

    # --- factories matching the two legacy configurations ---
    @classmethod
    def continuum_only(cls) -> SkyModel:
        """Reproduces ``num_sky_blocks == 1`` (continuum-only).

        Returns
        -------
        SkyModel
            A model with a single :class:`ContinuumComponent`.
        """
        return cls((ContinuumComponent(),))

    @classmethod
    def continuum_plus_gaussian_line(
        cls, name: str, center_um: float, sigma_um: float, *, wavelength_key: str,
        width_key: str | None = None, fwhm_to_sigma: float = 2.355,
        intrinsic_var_um2: float = 0.0,
    ) -> SkyModel:
        """Continuum + one Gaussian emission line (two sky blocks).

        The line's per-pixel sigma comes from the instrument's band-width map
        ``width_key`` (FWHM / ``fwhm_to_sigma``, in quadrature with the
        intrinsic width ``intrinsic_var_um2``) when that map is supplied to the
        solve, else the scalar ``sigma_um``. Instrument catalogues wrap this
        with their constants (SPHEREx: ``instruments.spherex.line_catalog``).
        """
        from .profiles import GaussianProfile, QuadratureSigma
        sigma_source = None
        if width_key is not None:
            sigma_source = QuadratureSigma(fwhm_key=width_key, fwhm_to_sigma=fwhm_to_sigma,
                                           intrinsic_var_um2=intrinsic_var_um2)
        profile = GaussianProfile(center_um=center_um, sigma_um=sigma_um, sigma_source=sigma_source)
        return cls((ContinuumComponent(),
                    SpectralComponent(name=name, profile=profile, wavelength_key=wavelength_key)))
