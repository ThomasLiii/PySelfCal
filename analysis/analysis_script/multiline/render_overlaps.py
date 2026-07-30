"""Tile-overlap diagnostics for a stitched multi-line cal.

Two figures, to check whether the low-S/N edge fringes sit on tile boundaries:
  sci_components_<tag>_tileedges.png : the 2x2 component maps (same style as
      render_components) with the tile ASSIGNMENT bboxes overlaid as outlines.
  sci_ncontrib_<tag>.png : the n_contrib map = number of tiles contributing
      (Fisher>0) per pixel. The assignment bbox is a hard cut, but frames are
      center-assigned with footprints spilling past it, so the real blend zones
      are the n_contrib LEVEL TRANSITIONS (1->2->3...). Fringes that track those
      steps are stitch/coverage edges, not sky.

Usage:
  python render_overlaps.py --cal PATH --tag NAME --tiles prod_tiles.npz [--ip 25]
"""
import argparse
import os

import numpy as np
import hdf5plugin  # noqa: F401
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm
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
ap.add_argument("--tiles", required=True)
ap.add_argument("--ip", type=float, default=25.0)
a = ap.parse_args()

bboxes = np.load(a.tiles)["bboxes"]        # (16, 4) = (y0, y1, x0, x1) full grid

with h5py.File(a.cal, "r") as f:
    names = [n.decode() if isinstance(n, bytes) else str(n)
             for n in f.attrs["sky_components"]]
    cov0 = f["sky_coverage"][names[0]][:] > 0
    rows = np.where(cov0.any(axis=1))[0]
    cols = np.where(cov0.any(axis=0))[0]
    m = 20
    r0 = max(0, rows[0] - m); r1 = min(cov0.shape[0], rows[-1] + 1 + m)
    c0 = max(0, cols[0] - m); c1 = min(cov0.shape[1], cols[-1] + 1 + m)
    SL = (slice(r0, r1), slice(c0, c1))
    sky, masks = {}, {}
    for nm in names:
        sky[nm] = f["sky"][nm][SL].astype(np.float64) * 1e3
        msk = (f["sky_fisher"][nm][SL] >= FISH_MIN) & np.isfinite(sky[nm])
        if nm != names[0]:
            msk &= f["sky_separability"][nm][SL] >= a.ip
        masks[nm] = msk
    ncontrib = f["n_contrib"][names[0]][SL].astype(np.float64)
del cov0

wcs_crop = WCS(fits.getheader(REF))[SL]


def axfmt(ax, title):
    ax.coords.grid(color="gray", ls=":", lw=0.6, alpha=0.7)
    ax.coords[0].set_axislabel("RA")
    ax.coords[1].set_axislabel("Dec")
    ax.coords[0].set_ticks(number=4)
    ax.coords[1].set_ticks(number=4)
    ax.coords[0].set_ticks_position("b")
    ax.coords[0].set_ticklabel_position("b")
    ax.coords[0].set_axislabel_position("b")
    ax.set_title(title, pad=10)


def draw_tile_edges(ax):
    """Tile assignment bboxes as outlines, in crop-array pixel coords."""
    for (y0, y1, x0, x1) in bboxes:
        xs = [x0 - c0, x1 - c0, x1 - c0, x0 - c0, x0 - c0]
        ys = [y0 - r0, y0 - r0, y1 - r0, y1 - r0, y0 - r0]
        ax.plot(xs, ys, color="lime", lw=0.8, alpha=0.9)


# --- Figure 1: components with tile-edge overlay ---
fig = plt.figure(figsize=(11.5, 10.5))
for k, nm in enumerate(names):
    ax = fig.add_subplot(2, 2, k + 1, projection=wcs_crop)
    v = np.where(masks[nm], sky[nm], np.nan)
    vmax = float(np.nanpercentile(np.abs(v), 95))
    ax.imshow(v, origin="lower", cmap="RdBu_r",
              norm=Normalize(vmin=-vmax, vmax=vmax))
    draw_tile_edges(ax)
    ax.set_xlim(-0.5, c1 - c0 - 0.5)
    ax.set_ylim(-0.5, r1 - r0 - 0.5)
    axfmt(ax, TITLES.get(nm, nm))
    cb = fig.colorbar(ax.images[0], ax=ax, shrink=0.9, pad=0.02)
    cb.set_label(AMP_LABEL)
out1 = os.path.join(FIG, f"sci_components_{a.tag}_tileedges.png")
fig.savefig(out1, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {out1}")

# --- Figure 2: n_contrib map ---
nmax = int(np.nanmax(ncontrib))
nc = np.where(ncontrib >= 1, ncontrib, np.nan)
fig = plt.figure(figsize=(8.2, 7.6))
ax = fig.add_subplot(1, 1, 1, projection=wcs_crop)
cmap = plt.get_cmap("turbo", nmax).copy()
cmap.set_bad("white")
norm = BoundaryNorm(np.arange(0.5, nmax + 1.5), cmap.N)
im = ax.imshow(nc, origin="lower", cmap=cmap, norm=norm)
draw_tile_edges(ax)
ax.set_xlim(-0.5, c1 - c0 - 0.5)
ax.set_ylim(-0.5, r1 - r0 - 0.5)
axfmt(ax, "Number of tiles contributing per pixel (n_contrib)")
cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02, ticks=range(1, nmax + 1))
cb.set_label("tiles (Fisher > 0)")
out2 = os.path.join(FIG, f"sci_ncontrib_{a.tag}.png")
fig.savefig(out2, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {out2}")

# --- Figure 3: line maps with n_contrib boundary contours overlaid ---
# Direct test: do the radial spoke fringes sit on coverage-count transitions?
# Contour n_contrib at half-integer levels -> clean lines between 1|2, 2|3, ...
from scipy import ndimage as ndi
ncf = ndi.median_filter(ncontrib, size=9)   # de-speckle the integer field
# Only the LOW-count wedge boundaries (1|2|3|4|5) — the deep center (6-9) is
# not where the low-S/N spokes live and its fine transitions just clutter.
levels = np.arange(1.5, 5.5)
fig = plt.figure(figsize=(16.0, 7.4))
for k, nm in enumerate(("aliphatic", "plateau")):
    ax = fig.add_subplot(1, 2, k + 1, projection=wcs_crop)
    v = np.where(masks[nm], sky[nm], np.nan)
    vmax = float(np.nanpercentile(np.abs(v), 95))
    ax.imshow(v, origin="lower", cmap="RdBu_r",
              norm=Normalize(vmin=-vmax, vmax=vmax))
    ax.contour(ncf, levels=levels, colors="k", linewidths=0.5, alpha=0.6)
    ax.set_xlim(-0.5, c1 - c0 - 0.5)
    ax.set_ylim(-0.5, r1 - r0 - 0.5)
    axfmt(ax, TITLES[nm] + "  +  n_contrib boundaries")
    cb = fig.colorbar(ax.images[0], ax=ax, shrink=0.85, pad=0.02)
    cb.set_label(AMP_LABEL)
out3 = os.path.join(FIG, f"sci_lines_ncontrib_contours_{a.tag}.png")
fig.savefig(out3, bbox_inches="tight")
plt.close(fig)
print(f"[saved] {out3}")
