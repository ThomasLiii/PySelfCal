"""PAHfit variants with the measured SPHEREx LVF line template.

Three modes, each a small step on the previous one (all keep ``pipeline="cal"``
so they run as a single cal + full mosaic, or with ``reproj_override`` on a
staged tile):

``pahfit_subch``
    PAHfit sky (continuum + Gaussian 3.29um line) + the production offset model:
    column poly + cubic subchannel poly-constraint + per-frame scalar. The
    subchannel constraint is what kept the full-NEP line/continuum split clean
    at Fisher>=10 (Pearson ~-0.01); without it a single SEP cal leaks (+0.40).

``pahfit_lvf``
    Same offset model, but the line profile is the MEASURED template
    G(lambda_c) = Drude intrinsic PAH line (x) SPHEREx Band-4 LVF response,
    tabulated vs channel centre (BC) in ``[params].line_template_npz`` (built by
    ``selfcal_scripts/spectral_4pass/build_pah_template.py``). Dropped in via
    :class:`selfcal.models.profiles.TemplateProfile`; replaces the too-narrow
    Gaussian that let PAH leak into the offset term.

``pahfit_lvf_polybasis``
    Same sky, but the offset is a HARD degree-D Chebyshev polynomial-basis in
    subchannel per column (coefficients solved directly, no soft weight knob —
    the knob let the offset grow a spurious PAH bump and diverge under
    iteration). The per-frame scalar owns the DC. This is the pass-1 recipe of
    the spectral 4-pass chain (``selfcal_scripts/spectral_4pass``).

``[params]`` keys: ``subch_poly_degree``, ``subch_poly_lo``, ``subch_poly_hi``
(all three), ``subch_tot`` / ``subch_poly_weight`` / ``poly_degree`` /
``poly_weight`` (subch, lvf), ``line_template_npz`` + optional
``line_template_norm`` (``"peak"`` default or ``"area"``) (lvf, polybasis).
"""
import os

import numpy as np

from .base import register_mode
from .pahfit import PAHfit


@register_mode("pahfit_subch")
class PAHfitSubch(PAHfit):
    """PAHfit with the production cubic-subchannel offset constraint."""

    requires = ("wavelength", "subchannel")

    def build_offset_model(self, cfg, inst, det_inputs, ch_inputs, job, n_frames):
        from selfcal.models.offset_model import OffsetModel, OffsetBlock
        p = cfg.params
        ncol = cfg.instrument_cfg['num_col']
        cm = det_inputs['det_chunk_map']
        adj = inst.column_adjacency(cm, ncol)
        poly_groups = []
        col_deg = p.get('poly_degree', 1)
        if p.get('poly_weight') is not None and ncol >= col_deg + 2:
            pc, ps = inst.column_poly_chains(cm, ncol, degree=col_deg)
            poly_groups.append({'chains': pc, 'stencil': ps, 'weight': p['poly_weight']})
        if p.get('subch_poly_weight') is not None:
            scn, sst = inst.subchannel_poly_chains(
                p['subch_tot'], ncol, p['subch_poly_degree'],
                p['subch_poly_lo'], p['subch_poly_hi'])
            poly_groups.append({'chains': scn, 'stencil': sst, 'weight': p['subch_poly_weight']})
        return OffsetModel([
            OffsetBlock(chunk_map=cm, adj_info=adj, reg_weight=p.get('reg_weight', 0.1),
                        poly_constraints=poly_groups or None, mean_offset=np.zeros(n_frames)),
        ], use_per_frame_scalar=True)


@register_mode("pahfit_lvf")
class PAHfitLVF(PAHfitSubch):
    """Subch-poly offset model + measured Drude x LVF line template."""

    def build_sky_model(self, cfg, inst, det_inputs):
        from selfcal.models.sky_model import (
            SkyModel, ContinuumComponent, SpectralComponent)
        from selfcal.models.profiles import TemplateProfile
        p = cfg.params
        npz = p['line_template_npz']
        d = np.load(npz)
        key = 'G_peaknorm' if p.get('line_template_norm', 'peak') == 'peak' else 'G'
        profile = TemplateProfile(wave_um=np.asarray(d['center_um'], float),
                                  values=np.asarray(d[key], float))
        print(f"[pahfit_lvf] line template {os.path.basename(npz)} [{key}]: "
              f"{d['center_um'][0]:.3f}-{d['center_um'][-1]:.3f} um, "
              f"peak coeff at BC={float(d['center_um'][np.argmax(d[key])]):.4f} um, "
              f"FWHM={1e3*float(d['fwhm_conv']):.1f} nm", flush=True)
        return SkyModel((ContinuumComponent(),
                         SpectralComponent(name='pah_3p29', profile=profile,
                                           wavelength_key='BC')))


@register_mode("pahfit_lvf_polybasis")
class PAHfitLVFPolyBasis(PAHfitLVF):
    """Measured LVF template sky + hard poly-basis offset (no weight knob)."""

    requires = ("wavelength", "subchannel")

    def build_offset_model(self, cfg, inst, det_inputs, ch_inputs, job, n_frames):
        from selfcal.models.offset_model import OffsetModel, OffsetBlock
        from selfcal.models.offset_basis import n_coef
        p = cfg.params
        ncol = int(cfg.instrument_cfg['num_col'])
        cm = det_inputs['det_chunk_map']
        # SPHEREx chunk encoding lives in the adapter (chunk = subch*num_col+col);
        # the core sees only the abstract coord/group arrays.
        poly_basis = inst.subchannel_poly_basis(
            cm, ncol, degree=int(p['subch_poly_degree']),
            lo=int(p['subch_poly_lo']), hi=int(p['subch_poly_hi']))
        ncf = n_coef(poly_basis)
        print(f"[pahfit_lvf_polybasis] hard poly-basis offset: degree={poly_basis['degree']} "
              f"-> {ncf} coeffs/col x {ncol} col = {ncol*ncf} coeffs/frame "
              f"(unconstrained poly; no weight knob)", flush=True)
        return OffsetModel([
            OffsetBlock(chunk_map=cm, poly_basis=poly_basis),
        ], use_per_frame_scalar=True)
