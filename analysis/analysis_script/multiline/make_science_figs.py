"""Science-style figures for the NEP D4 multi-line probe (iter300, I_P-masked).

Outputs (figures/, dpi 300, serif + mathtext, WCS graticule where applicable):
  sci_components_iter300.png  2x2: continuum + 3 line-amplitude maps,
                              linear symmetric +/- p95(|v|), RA/Dec grid.
                              Also writes/prints the pairwise map-correlation
                              table (joint mask) -> sci_components_corr.md
  sci_templates.png           the 3 realistic templates G(BC) overlaid,
                              window + line centers marked.
  sci_flux_ratio.png          1x3: non-aromatic integrated flux (aliphatic +
                              plateau), aromatic integrated flux, their ratio.
  sci_ip_fisher.png           2x3: per-line I_P (log) and I_P/Fisher (0-1).

All map panels: Fisher>=10; line blocks additionally I_P>=25 (read-time mask).
Units: amplitudes 1e-3 MJy/sr; integrated flux 1e-3 MJy/sr um.
"""
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
CAL = ("/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D4_6p2arcsec/"
       "calibration/cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_"
       "multiline3_probe1k_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2.h5")
REF = "/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D4_6p2arcsec/ref.fits"
FISH_MIN, IP_MIN = 10.0, 25.0
SL = (slice(4847, 8173), slice(4696, 7949))
TITLES = {"continuum": "Continuum", "aromatic": r"Aromatic $3.289\,\mu$m",
          "aliphatic": r"Aliphatic $3.400\,\mu$m", "plateau": r"Plateau $3.470\,\mu$m"}
TPL = {"aromatic": "aromatic_3p289", "aliphatic": "aliphatic_3p400",
       "plateau": "plateau_3p470"}
AMP_LABEL = r"$10^{-3}\,\mathrm{MJy\,sr^{-1}}$"
FLUX_LABEL = r"$10^{-3}\,\mathrm{MJy\,sr^{-1}\,\mu m}$"

wcs_crop = WCS(fits.getheader(REF))[SL]

with h5py.File(CAL) as f:
    names = [n.decode() for n in f.attrs["sky_components"]]
    sky, fish, ip, masks = {}, {}, {}, {}
    for nm in names:
        sky[nm] = f["sky"][nm][SL].astype(np.float64) * 1e3
        fish[nm] = f["sky_fisher"][nm][SL].astype(np.float64)
        m = (fish[nm] >= FISH_MIN) & np.isfinite(sky[nm])
        if nm != "continuum":
            ip[nm] = f["sky_separability"][nm][SL].astype(np.float64)
            m &= ip[nm] >= IP_MIN
        masks[nm] = m

joint = np.logical_and.reduce([masks[nm] for nm in names])


def wcs_panel(ax):
    ax.coords.grid(color="gray", ls=":", lw=0.6, alpha=0.7)
    ax.coords[0].set_axislabel("RA")
    ax.coords[1].set_axislabel("Dec")
    ax.coords[0].set_ticks(number=4)
    ax.coords[1].set_ticks(number=4)


# ---------------- Fig 1: component maps + correlation table -----------------
fig = plt.figure(figsize=(11.5, 10.5))
for k, nm in enumerate(names):
    ax = fig.add_subplot(2, 2, k + 1, projection=wcs_crop)
    v = np.where(masks[nm], sky[nm], np.nan)
    vmax = float(np.nanpercentile(np.abs(v), 95))
    im = ax.imshow(v, origin="lower", cmap="RdBu_r",
                   norm=Normalize(vmin=-vmax, vmax=vmax))
    wcs_panel(ax)
    ax.set_title(TITLES[nm])
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(AMP_LABEL)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sci_components_iter300.png"), bbox_inches="tight")
plt.close(fig)
print("[saved] sci_components_iter300.png")

hdr = "| | " + " | ".join(n
                          for n in names) + " |"
lines = [hdr, "|---" * (len(names) + 1) + "|"]
print(f"\npairwise Pearson over joint mask ({joint.sum():,} px):")
for i, a in enumerate(names):
    row = []
    for b in names:
        r = np.corrcoef(sky[a][joint], sky[b][joint])[0, 1]
        row.append(f"{r:+.3f}")
    lines.append("| **" + a + "** | " + " | ".join(row) + " |")
    print(f"  {a:<10s} " + "  ".join(row))
with open(os.path.join(FIG, "sci_components_corr.md"), "w") as fh:
    fh.write(f"Pairwise Pearson, iter300, joint mask (Fisher>={FISH_MIN:g} all"
             f" blocks + I_P>={IP_MIN:g} lines), {joint.sum():,} px\n\n")
    fh.write("\n".join(lines) + "\n")
print("[saved] sci_components_corr.md")

# ---------------- Fig 2: templates overlaid ---------------------------------
fig, ax = plt.subplots(figsize=(7.5, 4.6))
colors = {"aromatic": "tab:red", "aliphatic": "tab:blue", "plateau": "tab:green"}
W = {}
for nm, base in TPL.items():
    d = np.load(os.path.join("/home/thomasli/selfcal-project/selfcal/selfcal/instruments/spherex/data/line_templates", f"{base}.npz"))
    W[nm] = float(np.trapezoid(d["G_peaknorm"], d["center_um"]))
    ax.plot(d["center_um"], d["G_peaknorm"], color=colors[nm], lw=2,
            label=TITLES[nm] + f"  (FWHM {1e3*float(d['fwhm_conv']):.0f} nm)")
    ax.axvline(float(d["lam0"]), color=colors[nm], ls=":", lw=1, alpha=0.7)
ax.axvspan(3.159, 3.716, color="0.92", zorder=0)
ax.text(3.72, 0.97, "fit window", fontsize=9, color="0.4",
        ha="right", va="top")
ax.set_xlabel(r"channel center $\lambda_c$ = BC [$\mu$m]")
ax.set_ylabel(r"$G(\lambda_c)$ [peak = 1]")
ax.set_xlim(3.08, 3.82)
ax.set_ylim(0, 1.05)
ax.legend(fontsize=9, loc="upper left")
ax.set_title("Realistic line templates (Drude $\\otimes$ measured Band-4 response)")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sci_templates.png"), bbox_inches="tight")
plt.close(fig)
print("[saved] sci_templates.png  W =",
      {k: round(v, 4) for k, v in W.items()}, "um")

# ---------------- Fig 3: non-aromatic vs aromatic integrated flux -----------
F_na = np.where(joint, sky["aliphatic"] * W["aliphatic"]
                + sky["plateau"] * W["plateau"], np.nan)
F_a = np.where(joint, sky["aromatic"] * W["aromatic"], np.nan)
ratio = np.where(F_a != 0, F_na / F_a, np.nan)   # unrestricted (user request)

fig = plt.figure(figsize=(16.5, 5.6))
panels = [
    (F_na, r"Non-aromatic $\Sigma$ flux (aliphatic + plateau)", FLUX_LABEL),
    (F_a, "Aromatic flux", FLUX_LABEL),
    (ratio, "Non-aromatic / aromatic", "ratio"),
]
for k, (v, title, lab) in enumerate(panels):
    ax = fig.add_subplot(1, 3, k + 1, projection=wcs_crop)
    vmax = float(np.nanpercentile(np.abs(v), 95))
    im = ax.imshow(v, origin="lower", cmap="RdBu_r",
                   norm=Normalize(vmin=-vmax, vmax=vmax))
    wcs_panel(ax)
    ax.set_title(title, fontsize=11)
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(lab)
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sci_flux_ratio.png"), bbox_inches="tight")
plt.close(fig)
print("[saved] sci_flux_ratio.png  (ratio unrestricted)")

# ---------------- Fig 4: I_P and I_P / Fisher --------------------------------
line_names = names[1:]
ip_all = np.concatenate([ip[nm][masks["continuum"] & (fish[nm] > 0)]
                         for nm in line_names])
ip_vmax = float(np.percentile(ip_all, 99))
fig = plt.figure(figsize=(16.5, 10.0))
for k, nm in enumerate(line_names):
    cov = fish[nm] > 0
    ax = fig.add_subplot(2, 3, k + 1, projection=wcs_crop)
    v = np.where(cov, np.maximum(ip[nm], 1e-2), np.nan)
    im = ax.imshow(v, origin="lower", cmap="viridis",
                   norm=LogNorm(vmin=1.0, vmax=ip_vmax))
    wcs_panel(ax)
    ax.set_title(TITLES[nm] + r":  $I_P$")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(r"$I_P$ [arb. units]")

    ax = fig.add_subplot(2, 3, k + 4, projection=wcs_crop)
    frac = np.where(cov & (fish[nm] > 0), ip[nm] / fish[nm], np.nan)
    im = ax.imshow(frac, origin="lower", cmap="viridis",
                   norm=Normalize(vmin=0, vmax=1))
    wcs_panel(ax)
    ax.set_title(TITLES[nm] + r":  $I_P\,/\,$Fisher")
    cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cb.set_label("fraction")
fig.tight_layout()
fig.savefig(os.path.join(FIG, "sci_ip_fisher.png"), bbox_inches="tight")
plt.close(fig)
print("[saved] sci_ip_fisher.png")
