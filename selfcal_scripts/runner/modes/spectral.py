"""Spectral modes — continuum + emission-line amplitudes per pixel.

Three recipes, each a step on the previous one; the sky is the same
(continuum + N line blocks, §"the sky"), the offset structure differs:

``spectral`` (preset ``pahfit``)
    the standard block of the continuum mode.
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
    ``[[params.lines]]`` — one block per entry: ``name`` + either
    ``template_npz`` (a realistic template: ``center_um`` / ``G_peaknorm``
    tabulated against the instrument's wavelength map; preferred) or
    ``center_um`` + ``sigma_um`` (analytic Gaussian) or ``center_um`` +
    ``intrinsic_var_um2`` (per-pixel sigma from the instrument's band-width
    map in quadrature); per-line ``damp_weight`` overrides
    ``[calibration].damp_weight_line``.
    Without ``lines``: ``line_template_npz`` (a single template line named
    ``pah_3p29``), else the named catalogue line ``[params].line`` (default
    ``pah_3p29``) from the instrument's ``line_catalog`` with the optional
    ``line_center`` / ``line_sigma`` overrides — the historical ``pahfit`` sky.
    Line profiles are used as-is (not orthogonalised against the continuum)
    and the continuum is flat per pixel; both alternatives were tested and
    gave no measurable benefit. With ``lines``, a pre-flight prints the profile
    Gram matrix over the window and warns at |r| > 0.7 (overlapping profiles
    are mutually degenerate per pixel).

Historical parameter spellings (``subch_poly_degree`` / ``_lo`` / ``_hi`` /
``_weight`` / ``subch_poly_segments``) are accepted everywhere the generic
``spectral_poly_*`` names are.
"""
import os

import numpy as np

from .base import CalMode, register_mode, standard_block, param, spectral_window
from .continuum import Continuum

GRAM_WARN = 0.7


# ---------------------------------------------------------------------------
# the sky
# ---------------------------------------------------------------------------
def _line_profile(spec, geom):
    """One ``[[params.lines]]`` entry -> a SpectralProfile."""
    from selfcal.models.profiles import TemplateProfile, GaussianProfile, QuadratureSigma
    if 'template_npz' in spec:
        d = np.load(spec['template_npz'])
        key = 'G_peaknorm' if spec.get('template_norm', 'peak') == 'peak' else 'G'
        return TemplateProfile(wave_um=np.asarray(d['center_um'], float),
                               values=np.asarray(d[key], float))
    if 'center_um' in spec:
        if spec.get('sigma_um') is not None:
            return GaussianProfile(center_um=float(spec['center_um']),
                                   sigma_um=float(spec['sigma_um']))
        if geom.width_key is None:
            raise ValueError(f"line {spec.get('name')!r}: a per-pixel sigma needs an instrument "
                             f"band-width map; give sigma_um instead")
        return GaussianProfile(
            center_um=float(spec['center_um']),
            sigma_source=QuadratureSigma(
                fwhm_key=geom.width_key, fwhm_to_sigma=2.355,
                intrinsic_var_um2=float(spec.get('intrinsic_var_um2', 0.0))))
    raise ValueError(f"line spec needs 'template_npz' or 'center_um': {spec}")


def build_sky(cfg, inst, geom, tag):
    """The spectral sky model of a config (see the module docstring)."""
    from selfcal.models.sky_model import SkyModel, ContinuumComponent, SpectralComponent
    from selfcal.models.profiles import TemplateProfile
    p = cfg.params
    if geom.wavelength_key is None:
        raise ValueError(f"mode {tag!r} needs an instrument wavelength map")
    lines = p.get('lines')
    if lines:
        components = [ContinuumComponent()]
        for spec in lines:
            name = spec['name']
            dw = spec.get('damp_weight')
            components.append(SpectralComponent(
                name=name, profile=_line_profile(spec, geom),
                wavelength_key=geom.wavelength_key,
                damp_weight=None if dw is None else float(dw)))
            src = os.path.basename(spec.get('template_npz', '')) or \
                f"Gaussian@{spec.get('center_um')}um"
            print(f"[{tag}] line {name!r}: {src} damp_weight="
                  f"{dw if dw is not None else '(fallback damp_weight_line)'}", flush=True)
        model = SkyModel(tuple(components))
        _print_gram(cfg, geom, model, tag)
        return model
    npz = p.get('line_template_npz')
    if npz:
        d = np.load(npz)
        key = 'G_peaknorm' if p.get('line_template_norm', 'peak') == 'peak' else 'G'
        profile = TemplateProfile(wave_um=np.asarray(d['center_um'], float),
                                  values=np.asarray(d[key], float))
        print(f"[{tag}] line template {os.path.basename(npz)} [{key}]: "
              f"{d['center_um'][0]:.3f}-{d['center_um'][-1]:.3f} um, "
              f"peak coeff at BC={float(d['center_um'][np.argmax(d[key])]):.4f} um, "
              f"FWHM={1e3*float(d['fwhm_conv']):.1f} nm", flush=True)
        return SkyModel((ContinuumComponent(),
                         SpectralComponent(name='pah_3p29', profile=profile,
                                           wavelength_key=geom.wavelength_key)))
    catalog = inst.line_catalog()
    name = p.get('line', 'pah_3p29')
    if name not in catalog:
        raise ValueError(f"[params].line = {name!r} is not in the instrument's line catalogue "
                         f"{sorted(catalog)}; give [[params.lines]] instead")
    return catalog[name](p.get('line_center'), p.get('line_sigma'))


def _print_gram(cfg, geom, model, tag):
    """Pre-flight: the profile Gram matrix over the window's spectral-axis values."""
    p = cfg.params
    cm = geom.chunk_map
    if cm.spectral_axis is None or geom.width_key is None:
        return
    lo, hi = spectral_window(p)
    wl = np.asarray(geom.aux[geom.wavelength_key], dtype=np.float64)
    bw = np.asarray(geom.aux[geom.width_key], dtype=np.float64)
    axis = cm.axes[cm.spectral_axis]
    sub_of_pix = axis.of_chunk[np.maximum(cm.det, 0)]
    valid = np.isfinite(wl) & (wl > 0) & (cm.det >= 0)
    cnts = np.bincount(sub_of_pix[valid].ravel(), minlength=axis.size)
    mean = {}
    for key, m in ((geom.wavelength_key, wl), (geom.width_key, bw)):
        sums = np.bincount(sub_of_pix[valid].ravel(), weights=m[valid].ravel(), minlength=axis.size)
        mean[key] = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    grid = np.arange(lo, hi + 1)
    ok = np.isfinite(mean[geom.wavelength_key][grid])
    aux = {k: v[grid][ok] for k, v in mean.items()}
    vecs, names = [], []
    for comp in model.components:
        c = comp.coefficients(aux)
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


# ---------------------------------------------------------------------------
# the recipes
# ---------------------------------------------------------------------------
@register_mode("spectral", "pahfit")
class Spectral(Continuum):
    """Standard offset block + spectral sky."""
    mosaic_mode = "full"
    requires = ("wavelength",)

    def build_sky_model(self, cfg, inst, geom):
        return build_sky(cfg, inst, geom, self.requested_name or self.name)

    def aux_maps(self, cfg, inst, geom):
        return dict(geom.aux)

    def configure(self, cfg, cc):
        cc.line_fisher_threshold = cfg.params.get('line_fisher_threshold', 10.0)


def spectral_poly_group(cfg, geom, *, required=False):
    """The soft polynomial constraint along the spectral axis, or None when
    ``spectral_poly_weight`` is not set (and not required)."""
    from selfcal.models.offset_structure import poly_chains_along
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
    degree = int(param(p, 'spectral_poly_degree', 'subch_poly_degree'))
    chains, stencil = poly_chains_along(cm.axes, cm.spectral_axis, degree, lo, hi)
    return {'chains': chains, 'stencil': stencil, 'weight': weight}


@register_mode("spectral_softpoly", "pahfit_subch", "pahfit_lvf")
class SpectralSoftPoly(Spectral):
    """Standard block + soft polynomial along the spectral axis."""
    requires = ("wavelength", "spectral_axis")
    spectral_poly_required = False
    column_poly_default_weight = None

    def build_offset_model(self, cfg, inst, geom, jobgeom, job, n_frames):
        from selfcal.models.offset_model import OffsetModel
        grp = spectral_poly_group(cfg, geom, required=self.spectral_poly_required)
        block = standard_block(cfg, geom, n_frames, extra_poly_groups=[grp] if grp else (),
                               column_poly_default_weight=self.column_poly_default_weight)
        return OffsetModel([block], use_per_frame_scalar=True)


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
    """Hard polynomial-basis offset (no weight knob) + per-frame scalar."""
    requires = ("wavelength", "spectral_axis")

    def build_offset_model(self, cfg, inst, geom, jobgeom, job, n_frames):
        from selfcal.models.offset_model import OffsetModel, OffsetBlock
        from selfcal.models.offset_basis import n_coef
        from selfcal.models.offset_structure import poly_basis_along
        p = cfg.params
        cm = geom.chunk_map
        if cm.spectral_axis is None or cm.group_axis is None:
            raise ValueError("the polynomial-basis offset needs a chunk map with spectral and group axes")
        lo, hi = spectral_window(p)
        segments = param(p, 'spectral_poly_segments', 'subch_poly_segments')
        degree = int(param(p, 'spectral_poly_degree', 'subch_poly_degree'))
        poly_basis = poly_basis_along(cm.axes, cm.spectral_axis, cm.group_axis, degree, lo, hi,
                                      segments=segments)
        ncf = n_coef(poly_basis)
        tag = self.requested_name or self.name
        print(f"[{tag}] hard poly-basis offset: degree={poly_basis['degree']}"
              + (f" on {len(segments)} segments {segments}" if segments else "")
              + f" -> {ncf} coeffs/{cm.group_axis} x {poly_basis['num_groups']} {cm.group_axis} "
              f"= {ncf * poly_basis['num_groups']} coeffs/frame "
              f"(no penalty weight, no profile orthogonalization; "
              f"the DC term is carried by the per-frame scalar)", flush=True)
        return OffsetModel([OffsetBlock(chunk_map=cm.det, poly_basis=poly_basis)],
                           use_per_frame_scalar=True)
