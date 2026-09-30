"""SkyModel — the per-pixel sky of the self-calibration equation.

Every sky term is a map times a multiplicative coefficient. The data row for
one observation ``i`` of reference pixel ``P`` is::

    data_i = w_i * Σ_j c_j(v_i) * S_j[P]  +  offsets  +  scalar

``S_j`` (one value per reference pixel) is solved; ``c_j`` is a KNOWN function
of *data variables* ``v_i`` — named per-observation quantities from any source
(:mod:`selfcal.models.variables`): detector maps sampled at each observation
(SPHEREx: its wavelength map ``BC`` and band-width map ``BW``), per-frame
values (time, filter, a half-wave-plate angle), reference-grid maps, planes
stored with the frames, the built-in coordinates, or functions of those. A
term without a coefficient (``c = 1``) is a constant sky. Nothing here knows
what a variable means: a spectral template of the wavelength makes ``S_j`` the
map of an emission feature, ``sin(2πt/P)`` of the time an annual modulation,
``cos 2ψ`` of a polariser angle a Stokes map.

A term's coefficient is a :class:`Coefficient`: the names of the variables it
reads and the function applied to them. The function is any picklable
callable ``f(*arrays) -> array`` (``ImportedFunction`` references one by
import path), or an object with an ``evaluate(x, obs)`` method — the
tabulated / Gaussian / linear shapes of :mod:`selfcal.models.profiles`.

Bit-identity: a term without a coefficient reproduces the legacy continuum
emission (identity coefficient, the assembly stores ``w_i`` directly), and a
coefficient built from a profile evaluates exactly the historical
``profile.evaluate(aux[key], aux)`` (``SpectralComponent`` is kept as that
spelling). The row assembly (``selfcal.core.assembly``) consumes this model;
this module holds no assembly logic.

Components are small frozen dataclasses (scalars, a function reference, at most
a small tabulated array) so they pickle cleanly into the multiprocessing task
dicts; large arrays travel via SHM, never inside a component.
"""
from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np

__all__ = [
    'Coefficient',
    'ImportedFunction',
    'SkyComponent',
    'ContinuumComponent',
    'SpectralComponent',
    'LineComponent',
    'SkyModel',
]


# ---------------------------------------------------------------------------
# coefficients
# ---------------------------------------------------------------------------
_IMPORTED = {}


@dataclass(frozen=True)
class ImportedFunction:
    """A coefficient function referenced by import path — ``"package.module:name"`` —
    called as ``f(*variables, **params)``. Stores only the path and the
    parameters, so it pickles into the worker processes, which import the
    function themselves."""

    path: str
    params: tuple = ()          # ((name, value), ...)

    def _resolve(self):
        fn = _IMPORTED.get(self.path)
        if fn is None:
            module, sep, attr = self.path.partition(':')
            if not sep or not module or not attr:
                raise ValueError(f"function path {self.path!r} must look like 'package.module:name'")
            fn = getattr(importlib.import_module(module), attr)
            _IMPORTED[self.path] = fn
        return fn

    def __call__(self, *arrays):
        return self._resolve()(*arrays, **dict(self.params))


@dataclass(frozen=True)
class Coefficient:
    """The multiplicative coefficient of a sky term: ``c = function(obs[v0], obs[v1], ...)``.

    ``variable``: the name of the data variable the function reads (a str), or
    several names (a tuple) for a function of several variables.
    ``function``: any picklable callable taking one array per variable (the
    per-observation values) and returning the coefficient per observation (a
    scalar is broadcast) — or an object with ``evaluate(x, obs)`` (the shapes
    of :mod:`selfcal.models.profiles`), called with the first variable and the
    full variable dict, so it may read further variables by name.
    """

    variable: str | tuple
    function: object

    @property
    def main_variables(self) -> tuple[str, ...]:
        """The variables the function is applied to (always read)."""
        return (self.variable,) if isinstance(self.variable, str) else tuple(self.variable)

    @property
    def variables(self) -> tuple[str, ...]:
        """Every variable the coefficient may read: the main ones plus any the
        function reads by name (e.g. a per-observation width map)."""
        out = list(self.main_variables)
        for k in getattr(self.function, 'aux_requirements', None) or getattr(self.function, 'variables', None) or ():
            if k not in out:
                out.append(k)
        return tuple(out)

    def describe(self) -> str:
        """``function(variables)`` — how the coefficient reads, for logs and product metadata."""
        f = self.function
        if isinstance(f, ImportedFunction):
            fname = f.path
        else:
            g = getattr(f, 'fn', f)
            fname = getattr(g, '__qualname__', None) or type(g).__name__
        return f"{fname}({', '.join(self.main_variables)})"

    def evaluate(self, obs: dict) -> np.ndarray:
        """The coefficient at every observation of ``obs`` (a mapping of equal-length arrays)."""
        f = self.function
        names = self.main_variables
        if hasattr(f, 'evaluate'):
            return f.evaluate(obs[names[0]], obs)
        x = [obs[v] for v in names]
        c = np.asarray(f(*x), dtype=np.float32)
        n = np.shape(x[0])
        if c.shape != n:
            c = np.broadcast_to(c, n).copy()
        return c


# ---------------------------------------------------------------------------
# components (one per sky term)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SkyComponent:
    """One sky term: the map ``S`` (solved) times ``coefficient`` (known).

    ``coefficient`` None means ``c = 1`` (a constant sky; the assembly stores the
    pixel weight directly). ``damp_weight``: this term's Tikhonov prior (None =
    the solver default: ``damp_weight`` for the first term, ``damp_weight_line``
    for the others).
    """

    name: str
    coefficient: Coefficient | None = None
    damp_weight: float | None = None

    @property
    def aux_requirements(self) -> tuple[str, ...]:
        """Every data variable the coefficient may read (empty for a constant term)."""
        return () if self.coefficient is None else self.coefficient.variables

    variables = aux_requirements

    @property
    def required_variables(self) -> tuple[str, ...]:
        """The variables the coefficient cannot do without."""
        return () if self.coefficient is None else self.coefficient.main_variables

    def coefficients(self, aux: dict) -> np.ndarray | None:
        """The coefficient at the observations of ``aux`` (the data variables sampled at
        a subframe's valid pixels, by name), or ``None`` for ``c = 1``."""
        if self.coefficient is None:
            return None
        return self.coefficient.evaluate(aux)


@dataclass(frozen=True)
class ContinuumComponent(SkyComponent):
    """A constant sky term (``c = 1``), named ``continuum`` by default."""

    name: str = 'continuum'


@dataclass(frozen=True)
class SpectralComponent(SkyComponent):
    """Historical spelling of a term whose coefficient is a profile of one variable:
    ``SpectralComponent(name, profile, wavelength_key)`` ≡
    ``SkyComponent(name, Coefficient(wavelength_key, profile))``."""

    name: str = 'spectral'
    profile: object = None
    wavelength_key: str = 'BC'

    def __post_init__(self):
        object.__setattr__(self, 'coefficient', Coefficient(self.wavelength_key, self.profile))


# Back-compat alias.
LineComponent = SpectralComponent


# ---------------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SkyModel:
    """Ordered tuple of :class:`SkyComponent` (default: one constant term)."""

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
        """Ordered union of every data variable the components may read."""
        seen = []
        for c in self.components:
            for k in getattr(c, 'aux_requirements', ()):
                if k not in seen:
                    seen.append(k)
        return tuple(seen)

    variables = aux_requirements

    @property
    def required_variables(self) -> tuple[str, ...]:
        """Ordered union of the variables the components cannot do without."""
        seen = []
        for c in self.components:
            for k in getattr(c, 'required_variables', getattr(c, 'aux_requirements', ())):
                if k not in seen:
                    seen.append(k)
        return tuple(seen)

    def damp_weights(self, damp_weight, damp_weight_line=None) -> list[float]:
        """Per-term Tikhonov weights: a term's own ``damp_weight`` when set, else
        ``damp_weight`` for the first term and ``damp_weight_line`` for the others
        (0 when that is None). The one rule the LSQR damping rows, the
        closed-form sky solve and the N-pass SKY pass share."""
        out = []
        for j, comp in enumerate(self.components):
            w = getattr(comp, 'damp_weight', None)
            if w is None:
                w = damp_weight if j == 0 else damp_weight_line
            out.append(0.0 if w is None else float(w))
        return out

    # --- factories -----------------------------------------------------------------
    @classmethod
    def continuum_only(cls) -> SkyModel:
        """One constant term (``continuum``)."""
        return cls((ContinuumComponent(),))

    @classmethod
    def continuum_plus_gaussian_line(
        cls, name: str, center_um: float, sigma_um: float, *, wavelength_key: str,
        width_key: str | None = None, fwhm_to_sigma: float = 2.355,
        intrinsic_var_um2: float = 0.0,
    ) -> SkyModel:
        """A constant term + one term whose coefficient is a Gaussian of the
        variable ``wavelength_key``; its per-observation sigma comes from the
        variable ``width_key`` (FWHM / ``fwhm_to_sigma``, in quadrature with
        ``intrinsic_var_um2``) when that variable is supplied to the solve,
        else the scalar ``sigma_um``. Instrument catalogues wrap this with
        their constants (SPHEREx: ``instruments.spherex.line_catalog``).
        """
        from .profiles import GaussianProfile, QuadratureSigma
        sigma_source = None
        if width_key is not None:
            sigma_source = QuadratureSigma(fwhm_key=width_key, fwhm_to_sigma=fwhm_to_sigma,
                                           intrinsic_var_um2=intrinsic_var_um2)
        profile = GaussianProfile(center_um=center_um, sigma_um=sigma_um, sigma_source=sigma_source)
        return cls((ContinuumComponent(),
                    SpectralComponent(name=name, profile=profile, wavelength_key=wavelength_key)))
