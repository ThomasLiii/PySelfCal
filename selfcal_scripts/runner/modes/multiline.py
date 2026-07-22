"""Multi-line spectral mode — continuum + N emission-line blocks + hard
poly-basis offset.

The J-block-generic production recipe (validated on the SPHEREx NEP field: PAH
3.29 aromatic + 3.40 aliphatic + 3.47 plateau, Detector 4). Generalizes ``pahfit``
(a single 3.29um line) to an arbitrary, config-driven list of spectral blocks,
each an arbitrary profile's per-pixel amplitude, over the shared continuum.

Sky model = ``ContinuumComponent`` + one ``SpectralComponent`` per
``[[params.lines]]`` entry. Each line is either a *realistic template*
(``template_npz`` with peak-normalized ``center_um`` / ``G_peaknorm``, built from
an intrinsic profile convolved with the measured spectral response — the
preferred form) or an analytic Gaussian (``center_um`` + scalar ``sigma_um``, or
per-pixel via BW quadrature with ``intrinsic_var_um2``). Per-line ``damp_weight``
overrides the shared ``[calibration] damp_weight_line``.

Offset = a single hard poly-basis block: a degree-``subch_poly_degree`` Chebyshev
in subchannel, one independent polynomial per column, plus the per-frame scalar
(the DC). The SPHEREx chunk encoding lives in the adapter
(``inst.subchannel_poly_basis``); the core sees only abstract coord/group arrays.
No adjacency, no penalty weight, NO line-orthogonalization (abandoned — never
demonstrated a win), NO water-filling, NO sloped continuum.

Pre-flight, prints the profile **Gram matrix** (continuum + every line sampled
at the window subchannels' mean BC) and warns at |r| > 0.7 — overlapping
profiles are mutually degenerate per pixel. Both ``task = cal`` (a compact probe
region) and ``task = tiled`` (the full-field production, Fisher-stitched) route
through this one mode.
"""
import os

import numpy as np

from .base import register_mode
from .pahfit import PAHfit

GRAM_WARN = 0.7


def _build_line_profile(spec):
    """One ``[[params.lines]]`` entry -> a SpectralProfile."""
    from selfcal.models.profiles import (TemplateProfile, GaussianProfile,
                                         QuadratureSigma)
    if 'template_npz' in spec:
        d = np.load(spec['template_npz'])
        key = 'G_peaknorm' if spec.get('template_norm', 'peak') == 'peak' else 'G'
        return TemplateProfile(wave_um=np.asarray(d['center_um'], float),
                               values=np.asarray(d[key], float))
    if 'center_um' in spec:
        if spec.get('sigma_um') is not None:
            return GaussianProfile(center_um=float(spec['center_um']),
                                   sigma_um=float(spec['sigma_um']))
        # per-pixel sigma from the BW map (quadrature with an intrinsic width)
        return GaussianProfile(
            center_um=float(spec['center_um']),
            sigma_source=QuadratureSigma(
                fwhm_key='BW', fwhm_to_sigma=2.355,
                intrinsic_var_um2=float(spec.get('intrinsic_var_um2', 0.0))))
    raise ValueError(f"line spec needs 'template_npz' or 'center_um': {spec}")


@register_mode("multiline")
class Multiline(PAHfit):
    """Continuum + N config-driven line blocks + hard poly-basis offset."""

    requires = ("wavelength", "subchannel")

    # ---- sky: continuum + N spectral blocks --------------------------------
    def build_sky_model(self, cfg, inst, det_inputs):
        from selfcal.models.sky_model import (SkyModel, ContinuumComponent,
                                              SpectralComponent)
        lines = cfg.params.get('lines')
        if not lines:
            raise ValueError("multiline mode needs [[params.lines]] entries")
        components = [ContinuumComponent()]
        for spec in lines:
            name = spec['name']
            dw = spec.get('damp_weight')
            components.append(SpectralComponent(
                name=name, profile=_build_line_profile(spec),
                wavelength_key='BC',
                damp_weight=None if dw is None else float(dw)))
            src = os.path.basename(spec.get('template_npz', '')) or \
                f"Gaussian@{spec.get('center_um')}um"
            print(f"[multiline] line {name!r}: {src} damp_weight="
                  f"{dw if dw is not None else '(fallback damp_weight_line)'}",
                  flush=True)
        model = SkyModel(tuple(components))
        self._print_gram(cfg, inst, det_inputs, model)
        return model

    # ---- offset: hard poly-basis (degree D Chebyshev in subchannel, per col) -
    def build_offset_model(self, cfg, inst, det_inputs, ch_inputs, job, n_frames):
        from selfcal.models.offset_model import OffsetModel, OffsetBlock
        from selfcal.models.offset_basis import n_coef
        p = cfg.params
        cm = det_inputs['det_chunk_map']
        poly_basis = inst.subchannel_poly_basis(
            cm, int(cfg.instrument_cfg['num_col']),
            degree=int(p['subch_poly_degree']),
            lo=int(p['subch_poly_lo']), hi=int(p['subch_poly_hi']))
        ncf = n_coef(poly_basis)
        print(f"[multiline] hard poly-basis offset: degree={poly_basis['degree']} "
              f"-> {ncf} coeffs/col x {poly_basis['num_groups']} col "
              f"= {ncf * poly_basis['num_groups']} coeffs/frame "
              f"(no weight knob, no ortho; DC in the per-frame scalar)", flush=True)
        return OffsetModel([OffsetBlock(chunk_map=cm, poly_basis=poly_basis)],
                           use_per_frame_scalar=True)

    # ---- pre-flight profile Gram over the fit window ------------------------
    def _print_gram(self, cfg, inst, det_inputs, model):
        p = cfg.params
        ncol = int(cfg.instrument_cfg['num_col'])
        cm = det_inputs['det_chunk_map']
        bc_map, bw_map = (np.asarray(a, dtype=np.float64) for a in inst.aux(det_inputs))
        sub_of_pix = cm // ncol
        valid = np.isfinite(bc_map) & (bc_map > 0)
        TOT = int(cm.max()) // ncol + 1
        cnts = np.bincount(sub_of_pix[valid].ravel(), minlength=TOT)
        mean = {}
        for key, m in (('BC', bc_map), ('BW', bw_map)):
            sums = np.bincount(sub_of_pix[valid].ravel(),
                               weights=m[valid].ravel(), minlength=TOT)
            mean[key] = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
        grid = np.arange(int(p['subch_poly_lo']), int(p['subch_poly_hi']) + 1)
        ok = np.isfinite(mean['BC'][grid])
        aux = {'BC': mean['BC'][grid][ok], 'BW': mean['BW'][grid][ok]}
        vecs, names = [], []
        for comp in model.components:
            c = comp.coefficients(aux)
            vecs.append(np.ones(ok.sum()) if c is None else np.asarray(c, float))
            names.append(comp.name)
        V = np.stack(vecs)
        norm = np.sqrt((V ** 2).sum(axis=1))
        C = (V @ V.T) / np.maximum(np.outer(norm, norm), 1e-300)
        w = max(len(n) for n in names)
        print(f"[multiline] profile Gram over window subchannels "
              f"({ok.sum()} of {grid.size} with valid BC):", flush=True)
        print(" " * (w + 2) + "  ".join(f"{n:>8s}" for n in names), flush=True)
        for i, n in enumerate(names):
            print(f"  {n:<{w}s}" + "  ".join(f"{C[i, j]:8.3f}"
                                             for j in range(len(names))), flush=True)
        bad = [(names[i], names[j], C[i, j])
               for i in range(len(names)) for j in range(i + 1, len(names))
               if abs(C[i, j]) > GRAM_WARN]
        for a, b, r in bad:
            print(f"[multiline] WARNING: |Gram({a},{b})| = {r:.3f} > {GRAM_WARN} — "
                  f"strongly degenerate per pixel; expect cross-talk between their "
                  f"maps and a shorter semi-convergence plateau.", flush=True)
        if not bad:
            print(f"[multiline] Gram check OK (all off-diagonals <= {GRAM_WARN}).",
                  flush=True)
