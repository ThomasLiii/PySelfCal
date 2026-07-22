"""Build a realistic line template for selfcal: intrinsic profile convolved with
the MEASURED SPHEREx spectral response (band-general LVFResponse kernel).

G(lambda_c) = int L(lambda) R(lambda; lambda_c) dlambda, tabulated vs channel
center (BC) — what selfcal uses as the per-pixel line coefficient. Saves npz
with center_um / G / G_peaknorm / fwhm_conv (same schema as pah_template.npz)
plus a diagnostic PNG (intrinsic | response | template).

Intrinsic shapes:
  drude  --center-um C --fwhm-um F     (Draine Drude; PAH-like broad features)
  gauss  --center-um C --fwhm-um F     (resolved Gaussian line)
  delta  --center-um C                 (unresolved line: G(c) = R(C; c), the
                                        response row itself — atomic lines)

Usage:
  python build_line_template.py --band 4 --name pah_3p29 --intrinsic drude \
      --center-um 3.289 --fwhm-um 0.0423 --bc-lo 3.10 --bc-hi 3.47 \
      --out-dir templates/
"""
import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, "/home/thomasli/spherex/SPHEREx_Spectral_Calibration")
import lvf_response as L


def drude(lam, lam0, gamma):
    x = lam / lam0 - lam0 / lam
    return gamma ** 2 / (x ** 2 + gamma ** 2)


def fwhm_of(x, y):
    y = y / y.max()
    above = x[y >= 0.5]
    return above[-1] - above[0] if above.size else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--band", type=int, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--intrinsic", choices=("drude", "gauss", "delta"),
                    required=True)
    ap.add_argument("--center-um", type=float, required=True)
    ap.add_argument("--fwhm-um", type=float, default=None,
                    help="intrinsic FWHM (drude/gauss)")
    ap.add_argument("--bc-lo", type=float, required=True,
                    help="template BC range low (um) — cover the fit window")
    ap.add_argument("--bc-hi", type=float, required=True)
    ap.add_argument("--n-centers", type=int, default=260)
    ap.add_argument("--out-dir", default=os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "templates"))
    a = ap.parse_args()
    if a.intrinsic in ("drude", "gauss") and a.fwhm_um is None:
        ap.error(f"--fwhm-um required for intrinsic={a.intrinsic}")

    os.makedirs(a.out_dir, exist_ok=True)
    lam0 = a.center_um
    pad = 4 * (a.fwhm_um or 0.02)
    lam = np.linspace(min(a.bc_lo, lam0 - pad), max(a.bc_hi, lam0 + pad), 4001)
    if a.intrinsic == "drude":
        gamma = a.fwhm_um / lam0
        D = drude(lam, lam0, gamma)
    elif a.intrinsic == "gauss":
        sig = a.fwhm_um / 2.355
        D = np.exp(-0.5 * ((lam - lam0) / sig) ** 2)
    else:
        D = None
    if D is not None:
        print(f"[{a.name}] intrinsic {a.intrinsic}: lam0={lam0} um, "
              f"numeric FWHM={1e3 * fwhm_of(lam, D):.1f} nm", flush=True)

    print(f"[{a.name}] building Band-{a.band} response surface "
          f"(cached after first use)...", flush=True)
    m = L.LVFResponse(band=a.band, normalize='area')
    print(f"[{a.name}] LSF FWHM at line center: "
          f"{1e3 * m.lsf_width(lam0):.1f} nm", flush=True)

    cen = np.linspace(a.bc_lo, a.bc_hi, a.n_centers)
    if a.intrinsic == "delta":
        G = np.array([m.lsf(c)(np.array([lam0]))[0] for c in cen])
    else:
        G = np.array([m.convolve(lam, D, c) for c in cen])
    Gn = G / G.max()
    fwhm_conv = fwhm_of(cen, G)
    print(f"[{a.name}] template G(lambda_c): peak at "
          f"lambda_c={cen[np.argmax(G)]:.4f} um, "
          f"FWHM={1e3 * fwhm_conv:.1f} nm", flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(21, 6))
    if D is not None:
        ax[0].plot(lam, D, 'b-', lw=2)
    else:
        ax[0].axvline(lam0, color='b', lw=2)
    ax[0].axvline(lam0, color='r', ls=':', lw=1)
    ax[0].set_title(f"(1) intrinsic {a.intrinsic} @ {lam0} um")
    ax[0].set_xlabel(r"$\lambda$ [$\mu$m]"); ax[0].grid(alpha=0.25)
    lsf_line = m.lsf(lam0)(lam)
    ax[1].plot(lam, lsf_line / lsf_line.max(), 'g-', lw=2)
    ax[1].axvline(lam0, color='r', ls=':', lw=1)
    ax[1].set_title(f"(2) measured Band-{a.band} response at line center\n"
                    f"FWHM={1e3 * m.lsf_width(lam0):.1f} nm")
    ax[1].set_xlabel(r"$\lambda$ [$\mu$m]"); ax[1].grid(alpha=0.25)
    ax[2].plot(cen, Gn, 'm-', lw=2.5)
    ax[2].axvline(lam0, color='r', ls=':', lw=1)
    ax[2].set_title(f"(3) template G($\\lambda_c$), FWHM={1e3 * fwhm_conv:.1f} nm")
    ax[2].set_xlabel(r"channel center $\lambda_c$ = BC [$\mu$m]")
    ax[2].set_ylabel("line coefficient [peak=1]"); ax[2].grid(alpha=0.25)
    fig.suptitle(f"{a.name}: {a.intrinsic} x measured SPHEREx Band-{a.band} "
                 f"response", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    png = os.path.join(a.out_dir, f"{a.name}.png")
    fig.savefig(png, dpi=300, bbox_inches='tight')

    npz = os.path.join(a.out_dir, f"{a.name}.npz")
    np.savez(npz, center_um=cen, G=G, G_peaknorm=Gn, lam0=lam0,
             intrinsic=a.intrinsic, fwhm_intrinsic=a.fwhm_um or 0.0,
             fwhm_conv=fwhm_conv, band=a.band)
    print(f"[saved] {png}\n[saved] {npz}", flush=True)


if __name__ == "__main__":
    main()
