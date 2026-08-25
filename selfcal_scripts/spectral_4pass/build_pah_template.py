"""Build the PAH 3.29um line template: Drude intrinsic profile convolved with
the MEASURED SPHEREx Band-4 LVF response.

Produces ``pah_template.npz`` (``center_um``, ``G``, ``G_peaknorm``, ``lam0``,
``gamma``, ``fwhm_conv``) — the ``[params].line_template_npz`` input of the
``pahfit_lvf*`` modes — plus a diagnostic figure showing (1) the Drude line,
(2) the LVF response, (3) the convolved template G(lambda_c) against the
Gaussian it replaced.

Needs the external ``lvf_response`` module from the SPHEREx spectral
calibration repo (``--lvf-dir``, default ``$SPHEREX_SPECTRAL_CAL_DIR`` or
``~/spherex/SPHEREx_Spectral_Calibration``). The shipped
``data/pah_template.npz`` was built with the defaults below.

Drude (Draine): I(lam) = gamma^2 / [(lam/lam0 - lam0/lam)^2 + gamma^2],
lam0 = 3.289 um, FWHM = gamma*lam0 (``--fwhm-um`` 0.0423 -> gamma = 0.01286).
"""
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

LAM0 = 3.289
GAUSS_SIGMA = 0.0156             # the Gaussian this template replaced (BW/2.355)
GAUSS_INTRINSIC = 2.890e-4       # PAH intrinsic var added in quadrature (sky_model)


def drude(lam, lam0, gamma):
    x = lam / lam0 - lam0 / lam
    return gamma**2 / (x**2 + gamma**2)


def fwhm_of(x, y):
    y = y / y.max()
    above = x[y >= 0.5]
    return above[-1] - above[0] if above.size else np.nan


def main(out_dir, lvf_dir, band=4, fwhm_um=0.0423):
    sys.path.insert(0, lvf_dir)
    import lvf_response as L  # noqa: E402  (external)
    gamma = fwhm_um / LAM0
    os.makedirs(out_dir, exist_ok=True)
    lam = np.linspace(3.05, 3.55, 4001)
    D = drude(lam, LAM0, gamma)
    print(f"[Drude] lam0={LAM0} um, gamma={gamma:.5f} (FWHM {fwhm_um} um) -> "
          f"numeric FWHM={1e3*fwhm_of(lam, D):.1f} nm", flush=True)

    print(f"[LVF] building Band-{band} response surface ...", flush=True)
    m = L.LVFResponse(band=band, normalize='area')
    for c in (3.20, LAM0, 3.36):
        print(f"[LVF]   FWHM at lambda_c={c:.3f}: {1e3*m.lsf_width(c):.1f} nm", flush=True)
    lsf_line = m.lsf(LAM0)(lam)

    cen = np.linspace(3.10, 3.47, 260)
    G = np.array([m.convolve(lam, D, c) for c in cen])
    Gn = G / G.max()
    fwhm_conv = fwhm_of(cen, G)
    print(f"[template] G(lambda_c): peak at {cen[np.argmax(G)]:.4f} um, "
          f"FWHM={1e3*fwhm_conv:.1f} nm", flush=True)

    sig = np.sqrt(GAUSS_SIGMA**2 + GAUSS_INTRINSIC)
    gauss = np.exp(-0.5 * ((cen - LAM0) / sig)**2)

    fig, ax = plt.subplots(1, 3, figsize=(21, 6))
    ax[0].plot(lam, D, 'b-', lw=2); ax[0].axvline(LAM0, color='r', ls=':', lw=1)
    ax[0].set_title(f"(1) Drude intrinsic PAH line\nlam0={LAM0}um, FWHM={1e3*fwhm_of(lam, D):.1f}nm")
    ax[0].set_xlabel(r"$\lambda$ [$\mu$m]"); ax[0].set_ylabel("I [peak=1]")
    ax[1].plot(lam, lsf_line / lsf_line.max(), 'g-', lw=2,
               label=f'R($\\lambda$;$\\lambda_c$={LAM0})  FWHM={1e3*m.lsf_width(LAM0):.1f}nm')
    for c, col in [(3.20, '0.6'), (3.36, '0.4')]:
        y = m.lsf(c)(lam)
        ax[1].plot(lam, y / y.max(), color=col, lw=1, ls='--',
                   label=f'$\\lambda_c$={c}  FWHM={1e3*m.lsf_width(c):.1f}nm')
    ax[1].axvline(LAM0, color='r', ls=':', lw=1); ax[1].legend(fontsize=8)
    ax[1].set_title(f"(2) Measured SPHEREx Band-{band} LVF response"); ax[1].set_xlabel(r"$\lambda$ [$\mu$m]")
    ax[2].plot(cen, Gn, 'm-', lw=2.5, label=f'Drude$\\otimes$LVF  FWHM={1e3*fwhm_conv:.1f}nm')
    ax[2].plot(cen, gauss, 'k--', lw=1.6, label=f'Gaussian  FWHM={1e3*2.355*sig:.1f}nm')
    ax[2].axvline(LAM0, color='r', ls=':', lw=1); ax[2].legend(fontsize=9)
    ax[2].set_title("(3) template G($\\lambda_c$) vs the Gaussian"); ax[2].set_xlabel(r"$\lambda_c$ = BC [$\mu$m]")
    for a in ax:
        a.grid(alpha=0.25); a.set_xlim(3.1, 3.48)
    fig.suptitle("PAH 3.29um line template: Drude x measured SPHEREx LVF response", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(out_dir, "pah_template.png"), dpi=200, bbox_inches='tight')
    npz = os.path.join(out_dir, "pah_template.npz")
    np.savez(npz, center_um=cen, G=G, G_peaknorm=Gn, lam0=LAM0, gamma=gamma, fwhm_conv=fwhm_conv)
    print(f"[saved] {npz}", flush=True)
    return npz


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))
    ap.add_argument("--lvf-dir", default=os.environ.get(
        "SPHEREX_SPECTRAL_CAL_DIR", os.path.expanduser("~/spherex/SPHEREx_Spectral_Calibration")))
    ap.add_argument("--band", type=int, default=4)
    ap.add_argument("--fwhm-um", type=float, default=0.0423)
    a = ap.parse_args()
    main(a.out_dir, a.lvf_dir, a.band, a.fwhm_um)
