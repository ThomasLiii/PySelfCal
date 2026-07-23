"""Integrated-flux triptych for any v3 multi-line cal: non-aromatic sum flux |
aromatic flux | non-aromatic/aromatic ratio. Renders BOTH a linear-symmetric
(RdBu_r, +/-p95) and a viridis-log (positive-only; grey=<=0, white=unmasked)
version.

flux_i = A_i * W_i, W_i = int G_i,peaknorm dlambda (um) from the template.
Non-aromatic = aliphatic + plateau (the degenerate pair; their SUM is the
well-constrained observable). Ratio denominator floored: pixels with aromatic
amplitude < --arom-floor MJy/sr are masked (removes the ~<=0 tail where the
ratio blows up / flips sign).

All panels: Fisher>=10 everywhere; line blocks additionally I_P >= --ip.
Units: flux 1e-3 MJy/sr um; ratio dimensionless.

Usage:
  python render_flux_ratio.py --cal PATH --tag NAME [--ip 25] [--arom-floor 1e-4]
"""
import argparse
import os

import numpy as np
import hdf5plugin  # noqa: F401
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from astropy.io import fits
from astropy.wcs import WCS
import warnings
warnings.filterwarnings("ignore")

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm",
    "axes.titlesize": 13, "axes.labelsize": 11,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.dpi": 300,
})

WS = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(WS, "figures")
REF = "/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D4_6p2arcsec/ref.fits"
FISH_MIN = 10.0
TPL = {"aromatic": "aromatic_3p289", "aliphatic": "aliphatic_3p400",
       "plateau": "plateau_3p470"}
FLUX_LABEL = r"$10^{-3}\,\mathrm{MJy\,sr^{-1}\,\mu m}$"

ap = argparse.ArgumentParser()
ap.add_argument("--cal", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--ip", type=float, default=25.0)
ap.add_argument("--arom-floor", type=float, default=1e-4,
                help="mask ratio where aromatic AMPLITUDE (MJy/sr) below this")
a = ap.parse_args()

W = {}
for nm, base in TPL.items():
    d = np.load(os.path.join("/home/thomasli/selfcal-project/selfcal/selfcal/instruments/spherex/data/line_templates", f"{base}.npz"))
    W[nm] = float(np.trapezoid(d["G_peaknorm"], d["center_um"]))

with h5py.File(a.cal, "r") as f:
    cov0 = f["sky_coverage"]["continuum"][:] > 0
    rows = np.where(cov0.any(axis=1))[0]
    cols = np.where(cov0.any(axis=0))[0]
    m = 20
    SL = (slice(max(0, rows[0] - m), min(cov0.shape[0], rows[-1] + 1 + m)),
          slice(max(0, cols[0] - m), min(cov0.shape[1], cols[-1] + 1 + m)))
    A, joint = {}, np.ones(cov0[SL].shape, bool)
    for nm in ("aromatic", "aliphatic", "plateau"):
        A[nm] = f["sky"][nm][SL].astype(np.float64)
        msk = (f["sky_fisher"][nm][SL] >= FISH_MIN) & np.isfinite(A[nm])
        msk &= f["sky_separability"][nm][SL] >= a.ip
        joint &= msk
del cov0

wcs_crop = WCS(fits.getheader(REF))[SL]

F_na = np.where(joint, A["aliphatic"] * W["aliphatic"]
                + A["plateau"] * W["plateau"], np.nan) * 1e3
F_a = np.where(joint, A["aromatic"] * W["aromatic"], np.nan) * 1e3
ratio_ok = joint & (A["aromatic"] >= a.arom_floor)
ratio = np.where(ratio_ok, (A["aliphatic"] * W["aliphatic"]
                            + A["plateau"] * W["plateau"])
                 / (A["aromatic"] * W["aromatic"]), np.nan)
n_floor = int(joint.sum() - ratio_ok.sum())
print(f"joint {int(joint.sum()):,} px; ratio floor aromatic>={a.arom_floor:g} "
      f"MJy/sr masks {n_floor:,} ({100*n_floor/joint.sum():.1f}%)")

panels = [
    (F_na, r"Non-aromatic $\Sigma$ flux (aliphatic + plateau)", FLUX_LABEL, False),
    (F_a, "Aromatic flux", FLUX_LABEL, False),
    (ratio, "Non-aromatic / aromatic", "ratio", True),
]


def draw(style):
    fig = plt.figure(figsize=(16.5, 5.9))
    for k, (v, title, lab, is_ratio) in enumerate(panels):
        ax = fig.add_subplot(1, 3, k + 1, projection=wcs_crop)
        if style == "linear":
            vmax = float(np.nanpercentile(np.abs(v), 95))
            im = ax.imshow(v, origin="lower", cmap="RdBu_r",
                           norm=Normalize(vmin=-vmax, vmax=vmax))
        else:
            vmax = float(np.nanpercentile(np.abs(v), 95))
            vmin = vmax / 100.0
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_under("lightgrey")
            cmap.set_bad("white")
            vv = v.copy()
            vv[(vv > 0) & (vv < vmin)] = vmin
            vv[vv <= 0] = vmin / 10.0
            im = ax.imshow(vv, origin="lower", cmap=cmap,
                           norm=LogNorm(vmin=vmin, vmax=vmax))
        ax.coords.grid(color="gray", ls=":", lw=0.6, alpha=0.7)
        ax.coords[0].set_axislabel("RA")
        ax.coords[1].set_axislabel("Dec")
        ax.coords[0].set_ticks(number=3)
        ax.coords[1].set_ticks(number=4)
        ax.coords[0].set_ticks_position("b")
        ax.coords[0].set_ticklabel_position("b")
        ax.coords[0].set_axislabel_position("b")
        ax.set_title(title, fontsize=11, pad=10)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02,
                          extend="min" if style == "log" else "neither")
        cb.set_label(lab)
    sty = "linear symmetric" if style == "linear" else \
        "viridis log (grey = value<=0, white = unmasked)"
    fig.suptitle(f"NEP D4 multi-line iter300 — integrated flux + ratio  [{sty}; "
                 f"Fisher>={FISH_MIN:g} + I_P>={a.ip:g}; ratio floor aromatic "
                 f"$\\geq$ {a.arom_floor:g} MJy sr$^{{-1}}$]", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    suffix = "linsym" if style == "linear" else "logviridis"
    out = os.path.join(FIG, f"sci_flux_ratio_{a.tag}_{suffix}.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")


draw("linear")
draw("log")
