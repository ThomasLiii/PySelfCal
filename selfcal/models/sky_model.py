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
``num_sky_blocks==1`` emission and ``SkyModel.continuum_plus_pah_gaussian()``
reproduces ``num_sky_blocks==2`` (component order [continuum, line], same
interleave, same float ops). The row assembly (``selfcal.core.assembly``)
consumes this model when emitting the per-observation sky coefficients; this
module holds no assembly logic itself.

Components are small frozen dataclasses (scalars + a profile holding at most a
small template array) so they pickle cleanly into the multiprocessing task
dicts; large arrays travel via SHM, never inside a component.
"""
from dataclasses import dataclass, field


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

    def coefficients(self, aux):  # pragma: no cover - interface
        raise NotImplementedError


@dataclass(frozen=True)
class ContinuumComponent(SkyComponent):
    name: str = 'continuum'
    aux_requirements: tuple = ()
    damp_weight: float = None

    def coefficients(self, aux):
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

    def coefficients(self, aux):
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
    def n_blocks(self):
        return len(self.components)

    @property
    def names(self):
        return [c.name for c in self.components]

    @property
    def aux_requirements(self):
        """Ordered union of all components' aux requirements."""
        seen = []
        for c in self.components:
            for k in getattr(c, 'aux_requirements', ()):
                if k not in seen:
                    seen.append(k)
        return tuple(seen)

    # --- factories matching the two legacy configurations ---
    @classmethod
    def continuum_only(cls):
        """Reproduces ``num_sky_blocks == 1`` (continuum-only)."""
        return cls((ContinuumComponent(),))

    @classmethod
    def continuum_plus_pah_gaussian(cls, line_center=None, line_sigma=None):
        """Reproduces ``num_sky_blocks == 2`` (continuum + PAH 3.29 µm Gaussian).

        Defaults pull the SPHEREx PAH constants; per-pixel σ uses the BW map via
        QuadratureSigma(2.355, 2.890e-4) when BW is supplied, else scalar
        line_sigma — matching the legacy spectral_fit behavior exactly.
        """
        from ..instruments.spherex.spherex_utility import PAH_LINE_CENTER_UM, LINE_SIGMA_UM
        from .profiles import GaussianProfile, QuadratureSigma
        center = PAH_LINE_CENTER_UM if line_center is None else line_center
        sigma = LINE_SIGMA_UM if line_sigma is None else line_sigma
        profile = GaussianProfile(
            center_um=center, sigma_um=sigma,
            sigma_source=QuadratureSigma(fwhm_key='BW', fwhm_to_sigma=2.355,
                                         intrinsic_var_um2=2.890e-4))
        return cls((ContinuumComponent(),
                    SpectralComponent(name='pah_3p29', profile=profile, wavelength_key='BC')))
