"""Spectral presets — a constant sky term plus terms whose coefficients are
functions of the instrument's wavelength variable (maps of emission features).

These are presets of the general model (``mode = "model"``, where a sky term's
coefficient is any function of any data variable). Three recipes, each a step
on the previous one; the sky terms are the same (see
:func:`~.base.spectral_sky_terms`), the offset term differs:

``spectral`` (preset ``pahfit``)
    the standard free offset term of the continuum mode.
``spectral_softpoly`` (presets ``pahfit_subch``, ``pahfit_lvf``, ``tiled``)
    + a SOFT polynomial constraint along the chunk map's spectral axis (degree
    ``spectral_poly_degree`` over ``[spectral_poly_lo, spectral_poly_hi]``,
    weight ``spectral_poly_weight``). What kept the full-NEP line/continuum
    split clean at Fisher >= 10; without it a single SEP cal leaks.
``spectral_polybasis`` (presets ``pahfit_lvf_polybasis``, ``multiline``)
    the offset IS a degree-D Chebyshev in the spectral axis, one independent
    polynomial per value of the group axis (coefficients solved directly, no
    weight knob — the knob let the offset grow a spurious line-shaped bump and
    diverge under iteration); the per-frame scalar owns the DC. The production
    spectral recipe and the INIT pass of the N-pass task. Optional
    ``spectral_poly_segments`` = ``[[lo, hi], ...]``: an independent polynomial
    per segment (see ``offset_basis.piecewise_cheb_shape_basis``).

The sky
    ``[[params.lines]]`` — one term per entry: ``name`` + either
    ``template_npz`` (a tabulated coefficient of the wavelength: ``center_um``
    / ``G_peaknorm`` — a realistic template; preferred) or ``center_um`` +
    ``sigma_um`` (a Gaussian of the wavelength) or ``center_um`` +
    ``intrinsic_var_um2`` (a Gaussian whose per-observation sigma comes from
    the band-width variable in quadrature); per-entry ``damp_weight``
    overrides ``[calibration].damp_weight_line``.
    Without ``lines``: ``line_template_npz`` (one tabulated term named
    ``pah_3p29``), else the catalogue coefficient ``[params].line`` (default
    ``pah_3p29``) of the instrument's ``coefficient_catalog`` with the optional
    ``line_center`` / ``line_sigma`` overrides — the historical ``pahfit`` sky.
    Line profiles are used as-is (not orthogonalised against the continuum)
    and the continuum is flat per pixel; both alternatives were tested and
    gave no measurable benefit. With ``lines``, a pre-flight prints the profile
    Gram matrix over the window and warns at |r| > 0.7 (overlapping profiles
    are mutually degenerate per pixel).

Historical parameter spellings (``subch_poly_degree`` / ``_lo`` / ``_hi`` /
``_weight`` / ``subch_poly_segments``) are accepted everywhere the generic
``spectral_poly_*`` names are. Any of these models can also be spelled out in
full with ``mode = "model"`` (see ``modes/model.py``).
"""
import numpy as np

from selfcal.models.spec import ModelSpec, OffsetTerm, PolyConstraint

from .base import register_mode, param, spectral_window, standard_offset_term, spectral_sky_terms
from .continuum import Continuum

GRAM_WARN = 0.7


def print_gram(cfg, geom, model, tag):
    """Pre-flight: the Gram matrix of the terms' coefficients, evaluated on the mean
    of every data variable over each spectral-axis value of the window."""
    p = cfg.params
    cm = geom.chunk_map
    if cm.spectral_axis is None or geom.wavelength_key is None or model.n_blocks < 2:
        return
    lo, hi = spectral_window(p)
    wl = np.asarray(geom.aux[geom.wavelength_key], dtype=np.float64)
    axis = cm.axes[cm.spectral_axis]
    sub_of_pix = axis.of_chunk[np.maximum(cm.det, 0)]
    valid = np.isfinite(wl) & (wl > 0) & (cm.det >= 0)
    cnts = np.bincount(sub_of_pix[valid].ravel(), minlength=axis.size)
    mean = {}
    for key, m in geom.aux.items():
        m = np.asarray(m, dtype=np.float64)
        sums = np.bincount(sub_of_pix[valid].ravel(), weights=m[valid].ravel(), minlength=axis.size)
        mean[key] = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    grid = np.arange(lo, hi + 1)
    ok = np.isfinite(mean[geom.wavelength_key][grid])
    aux = {k: v[grid][ok] for k, v in mean.items()}
    vecs, names = [], []
    for comp in model.components:
        try:
            c = comp.coefficients(aux)
        except KeyError as e:
            print(f"[{tag}] Gram check skipped: term {comp.name!r} reads {e} (not a detector-plane "
                  f"variable)", flush=True)
            return
        vecs.append(np.ones(ok.sum()) if c is None else np.asarray(c, float))
        names.append(comp.name)
    V = np.stack(vecs)
    norm = np.sqrt((V ** 2).sum(axis=1))
    C = (V @ V.T) / np.maximum(np.outer(norm, norm), 1e-300)
    w = max(len(n) for n in names)
    print(f"[{tag}] profile Gram over window values ({ok.sum()} of {grid.size} with valid "
          f"{geom.wavelength_key}):", flush=True)
    print(" " * (w + 2) + "  ".join(f"{n:>8s}" for n in names), flush=True)
    for i, n in enumerate(names):
        print(f"  {n:<{w}s}" + "  ".join(f"{C[i, j]:8.3f}" for j in range(len(names))), flush=True)
    bad = [(names[i], names[j], C[i, j])
           for i in range(len(names)) for j in range(i + 1, len(names)) if abs(C[i, j]) > GRAM_WARN]
    for a, b, r in bad:
        print(f"[{tag}] WARNING: |Gram({a},{b})| = {r:.3f} > {GRAM_WARN} — strongly degenerate per "
              f"pixel; expect cross-talk between their maps and a shorter semi-convergence plateau.",
              flush=True)
    if not bad:
        print(f"[{tag}] Gram check OK (all off-diagonals <= {GRAM_WARN}).", flush=True)


@register_mode("spectral", "pahfit")
class Spectral(Continuum):
    """Standard offset term + spectral sky terms."""
    mosaic_mode = "full"
    requires = ("wavelength",)

    def model_spec(self, cfg, inst, geom):
        """Return a spec: the spectral sky terms, the standard offset term, the per-frame scalar.

        The sky terms come from :func:`spectral_sky_terms`, the offset term from
        :func:`standard_offset_term` (as in the ``continuum`` mode).
        """
        return ModelSpec(sky=tuple(spectral_sky_terms(cfg, geom)),
                         offset=(standard_offset_term(cfg, geom),), scalar=True)

    def build_sky_model(self, cfg, inst, geom):
        """Build the base class's sky model and, when ``[params].lines`` is set, print its Gram check.

        :func:`print_gram` prints the normalised Gram matrix of the terms'
        coefficients over the ``[params]`` spectral window and warns about pairs
        with ``|r| > GRAM_WARN`` (0.7). It needs that window (``spectral_poly_lo``
        / ``spectral_poly_hi``) and raises ``ValueError`` without it, unless the
        chunk map has no spectral axis or the instrument no wavelength map, which
        skips the check.
        """
        model = super().build_sky_model(cfg, inst, geom)
        if cfg.params.get('lines'):
            print_gram(cfg, geom, model, self.requested_name or self.name)
        return model


def spectral_poly_constraint(cfg, geom, *, required=False):
    """The soft polynomial constraint along the spectral axis, or None when
    ``spectral_poly_weight`` is not set (and not required)."""
    p = cfg.params
    weight = param(p, 'spectral_poly_weight', 'subch_poly_weight')
    if weight is None:
        if required:
            raise ValueError("[params] needs spectral_poly_weight (subch_poly_weight)")
        return None
    cm = geom.chunk_map
    if cm.spectral_axis is None:
        raise ValueError("the spectral polynomial constraint needs a chunk map with a spectral axis")
    lo, hi = spectral_window(p)
    return PolyConstraint(axis=cm.spectral_axis, degree=int(param(p, 'spectral_poly_degree', 'subch_poly_degree')),
                          weight=weight, lo=lo, hi=hi)


@register_mode("spectral_softpoly", "pahfit_subch", "pahfit_lvf")
class SpectralSoftPoly(Spectral):
    """Standard offset term + soft polynomial along the spectral axis."""
    requires = ("wavelength", "spectral_axis")
    spectral_poly_required = False
    column_poly_default_weight = None

    def model_spec(self, cfg, inst, geom):
        """Return the :class:`Spectral` spec plus a soft polynomial along the spectral axis.

        :func:`spectral_poly_constraint` builds the constraint from ``[params]``
        ``spectral_poly_weight`` / ``_degree`` / ``_lo`` / ``_hi`` (historical
        ``subch_poly_*``), and it is appended to the standard offset term's
        constraints. Without ``spectral_poly_weight`` it is left out or, when the
        class sets ``spectral_poly_required`` (the ``tiled`` preset), a
        ``ValueError`` is raised. ``column_poly_default_weight`` (0.5 for
        ``tiled``) turns on the standard term's own polynomial when
        ``poly_weight`` is not given.
        """
        pc = spectral_poly_constraint(cfg, geom, required=self.spectral_poly_required)
        off = standard_offset_term(cfg, geom, extra_poly=(pc,) if pc else (),
                                   column_poly_default_weight=self.column_poly_default_weight)
        return ModelSpec(sky=tuple(spectral_sky_terms(cfg, geom)), offset=(off,), scalar=True)


@register_mode("tiled")
class TiledPreset(SpectralSoftPoly):
    """The region-partitioned production preset: column polynomial always on
    (weight 0.5 unless ``poly_weight`` is given), spectral polynomial required,
    no mosaic (run with ``[tiling]``: per-tile cal + Fisher stitch)."""
    mosaic_mode = "none"
    spectral_poly_required = True
    column_poly_default_weight = 0.5


@register_mode("spectral_polybasis", "pahfit_lvf_polybasis", "multiline")
class SpectralPolyBasis(Spectral):
    """Hard polynomial-basis offset term (no weight knob) + per-frame scalar."""
    requires = ("wavelength", "spectral_axis")

    def model_spec(self, cfg, inst, geom):
        """Return a spec: the spectral sky terms, a polynomial-basis offset, the per-frame scalar.

        The offset term (``kind='polybasis'``, on the primary map) is a mean-zero
        Chebyshev shape of degrees 1 to ``spectral_poly_degree`` in the map's
        spectral axis, one per value of its group axis, over the window
        ``[spectral_poly_lo, spectral_poly_hi]``. ``spectral_poly_segments``
        (``[[lo, hi], ...]``) makes it piecewise: a shape per segment, plus a level
        for each segment after the first. The historical ``subch_poly_*`` names are
        read too. There is no weight knob and no smoothness or mean-zero row: the
        per-frame scalar carries each frame's DC.
        """
        p = cfg.params
        lo, hi = spectral_window(p)
        off = OffsetTerm(kind='polybasis', degree=int(param(p, 'spectral_poly_degree', 'subch_poly_degree')),
                         lo=lo, hi=hi, segments=param(p, 'spectral_poly_segments', 'subch_poly_segments'))
        return ModelSpec(sky=tuple(spectral_sky_terms(cfg, geom)), offset=(off,), scalar=True)
