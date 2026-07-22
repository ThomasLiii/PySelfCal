"""Science-style component maps + correlation table for ANY v3 multi-line cal
(per-tile, partial stitch, or the final stitched field).

Same style as make_science_figs.py fig 1: 2x2 panels (continuum + 3 line
amplitudes), linear symmetric +/- p95(|v|), WCS graticule, serif/mathtext,
dpi 300, units 1e-3 MJy/sr. Fisher>=10 everywhere; line blocks additionally
I_P >= --ip (default 25). Crop is auto-derived from the continuum coverage.

Usage:
  python render_components.py --cal PATH --tag NAME [--ip TAU]
"""
import argparse
import os

import numpy as np
import hdf5plugin  # noqa: F401
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
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
TITLES = {"continuum": "Continuum", "aromatic": r"Aromatic $3.289\,\mu$m",
          "aliphatic": r"Aliphatic $3.400\,\mu$m", "plateau": r"Plateau $3.470\,\mu$m"}
AMP_LABEL = r"$10^{-3}\,\mathrm{MJy\,sr^{-1}}$"

ap = argparse.ArgumentParser()
ap.add_argument("--cal", required=True)
ap.add_argument("--tag", required=True)
ap.add_argument("--ip", type=float, default=25.0)
a = ap.parse_args()

with h5py.File(a.cal, "r") as f:
    names = [n.decode() if isinstance(n, bytes) else str(n)
             for n in f.attrs["sky_components"]]
    cov0 = f["sky_coverage"][names[0]][:] > 0
    rows = np.where(cov0.any(axis=1))[0]
    cols = np.where(cov0.any(axis=0))[0]
    m = 20
    SL = (slice(max(0, rows[0] - m), min(cov0.shape[0], rows[-1] + 1 + m)),
          slice(max(0, cols[0] - m), min(cov0.shape[1], cols[-1] + 1 + m)))
    print(f"crop y[{SL[0].start}:{SL[0].stop}] x[{SL[1].start}:{SL[1].stop}]")
    sky, masks = {}, {}
    for nm in names:
        sky[nm] = f["sky"][nm][SL].astype(np.float64) * 1e3
        msk = (f["sky_fisher"][nm][SL] >= FISH_MIN) & np.isfinite(sky[nm])
        if nm != names[0]:
            msk &= f["sky_separability"][nm][SL] >= a.ip
        masks[nm] = msk
del cov0

wcs_crop = WCS(fits.getheader(REF))[SL]
joint = np.logical_and.reduce([masks[nm] for nm in names])

fig = plt.figure(figsize=(11.5, 10.5))
for k, nm in enumerate(names):
    ax = fig.add_subplot(2, 2, k + 1, projection=wcs_crop)
    v = np.where(masks[nm], sky[nm], np.nan)
    vmax = float(np.nanpercentile(np.abs(v), 95))
    im = ax.imshow(v, origin="lower", cmap="RdBu_r",
                   norm=Normalize(vmin=-vmax, vmax=vmax))
    ax.coords.grid(color="gray", ls=":", lw=0.6, alpha=0.7)
    ax.coords[0].set_axislabel("RA")
    ax.coords[1].set_axislabel("Dec")
    ax.coords[0].set_ticks(number=4)
    ax.coords[1].set_ticks(number=4)
    # RA ticks/labels on the BOTTOM (default WCS puts them on top, colliding
    # with the panel title on a near-square N-up frame).
    ax.coords[0].set_ticks_position("b")
    ax.coords[0].set_ticklabel_position("b")
    ax.coords[0].set_axislabel_position("b")
    ax.set_title(TITLES.get(nm, nm), pad=10)
    cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(AMP_LABEL)
out = os.path.join(FIG, f"sci_components_{a.tag}.png")
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {out}")

lines = ["| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
print(f"\npairwise Pearson over joint mask ({joint.sum():,} px):")
for x in names:
    row = [f"{np.corrcoef(sky[x][joint], sky[y][joint])[0, 1]:+.3f}" for y in names]
    lines.append("| **" + x + "** | " + " | ".join(row) + " |")
    print(f"  {x:<10s} " + "  ".join(row))
corr = os.path.join(FIG, f"sci_components_{a.tag}_corr.md")
with open(corr, "w") as fh:
    fh.write(f"Pairwise Pearson — {a.tag}, joint mask (Fisher>={FISH_MIN:g} + "
             f"I_P>={a.ip:g} lines), {joint.sum():,} px\n\n" + "\n".join(lines) + "\n")
print(f"[saved] {corr}")
