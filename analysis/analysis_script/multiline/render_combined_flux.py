"""Integrated-flux combination of the degenerate aliphatic + plateau blocks.

The fitted amplitude A_i multiplies the PEAK-NORMALIZED template G_i(lambda),
so the per-pixel integrated line flux is A_i * W_i with W_i = int G_i dlambda
(um) from the template table (trapezoid). The aliphatic<->plateau null space
trades amplitude between the two profiles while the data pin their sum where
they overlap — the flux-weighted sum A_ali*W_ali + A_plat*W_plat is the
well-constrained observable, and should be stable across iterations even when
the individual maps swing.

Renders a 3 x n_iter grid (aliphatic flux, plateau flux, SUM) in viridis log
(grey = value<=0, white = below Fisher cut), one shared scale per row fixed at
the first iteration, units 1e-3 MJy/sr um. Also prints per-iteration stats of
the sum (median, frac_pos, Pearson vs continuum) and the iter-to-iter Pearson
of each row vs its first-iteration map — the stability metric.

Usage: python render_combined_flux.py [iter ...]   (default: 25 50 100)
"""
import os, sys
import numpy as np
import hdf5plugin  # noqa: F401
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

WS = os.path.dirname(os.path.abspath(__file__))
CAL_DIR = "/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration"
BASE = "cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_multiline3_probe1k"
TAIL = "_polybasisD2noortho_NumCol3_outThresh5_sigma2"
FISH_MIN = 10.0
PAIR = ("aliphatic", "plateau")
TPL = {"aliphatic": "aliphatic_3p400", "plateau": "plateau_3p470"}

args = sys.argv[1:]
LINEAR = "--linear" in args
iters = [int(a) for a in args if a != "--linear"] or [25, 50, 100]

W = {}
for nm, base in TPL.items():
    d = np.load(os.path.join("/home/thomasli/selfcal-project/selfcal/selfcal/instruments/spherex/data/line_templates", f"{base}.npz"))
    W[nm] = float(np.trapz(d["G_peaknorm"], d["center_um"]))
    print(f"W({nm}) = int G_peaknorm dlambda = {W[nm]:.4f} um")

paths = {it: os.path.join(CAL_DIR, f"{BASE}_iter{it}{TAIL}.h5") for it in iters}
with h5py.File(paths[iters[0]], "r") as f:
    cov = f["sky_coverage"]["continuum"][:] > 0
rows_i = np.where(cov.any(axis=1))[0]; cols_i = np.where(cov.any(axis=0))[0]
m = 20
sl = (slice(max(0, rows_i[0] - m), rows_i[-1] + 1 + m),
      slice(max(0, cols_i[0] - m), cols_i[-1] + 1 + m))

row_names = [f"aliphatic x {W['aliphatic']:.3f}", f"plateau x {W['plateau']:.3f}",
             "SUM (integrated flux)"]
maps = {}
stats = {}
for it in iters:
    with h5py.File(paths[it], "r") as f:
        mask = np.ones(cov[sl].shape, bool)
        blocks = {}
        for nm in ("continuum",) + PAIR:
            v = f["sky"][nm][sl].astype(np.float64) * 1e3
            fish = f["sky_fisher"][nm][sl]
            mask &= (fish >= FISH_MIN) & np.isfinite(v)
            blocks[nm] = v
    fa = blocks[PAIR[0]] * W[PAIR[0]]
    fp = blocks[PAIR[1]] * W[PAIR[1]]
    s = fa + fp
    for arr in (fa, fp, s):
        arr[~mask] = np.nan
    maps[(0, it)], maps[(1, it)], maps[(2, it)] = fa, fp, s
    sm, cm_ = s[mask], blocks["continuum"][mask]
    stats[it] = dict(med=np.median(sm), fpos=100 * np.mean(sm > 0),
                     p999=np.percentile(sm, 99.9),
                     r_cont=np.corrcoef(sm, cm_)[0, 1])

print(f"\nSUM stats (units 1e-3 MJy/sr um, joint Fisher>={FISH_MIN:g} mask):")
for it in iters:
    st = stats[it]
    print(f"  iter{it:>3d}: median={st['med']:+.3f}  frac_pos={st['fpos']:.1f}%  "
          f"p99.9={st['p999']:.1f}  Pearson(vs cont)={st['r_cont']:+.3f}")
print("\nstability: Pearson of each map vs its iter%d version:" % iters[0])
for i, rn in enumerate(row_names):
    ref = maps[(i, iters[0])]
    ok0 = np.isfinite(ref)
    line = []
    for it in iters[1:]:
        v = maps[(i, it)]
        ok = ok0 & np.isfinite(v)
        line.append(f"iter{it}: {np.corrcoef(ref[ok], v[ok])[0,1]:+.3f}")
    print(f"  {rn:<24s} " + "   ".join(line))

vmax, vmin = {}, {}
for i in range(3):
    a = maps[(i, iters[0])]
    vmax[i] = float(np.nanpercentile(np.abs(a), 99))
    vmin[i] = vmax[i] / 100.0

nC = len(iters)
fig, axes = plt.subplots(3, nC, figsize=(4.6 * nC + 1.2, 4.4 * 3),
                         squeeze=False, constrained_layout=True)
for i, rn in enumerate(row_names):
    if LINEAR:
        norm = Normalize(vmin=-vmax[i], vmax=vmax[i])
        cmap = plt.get_cmap("RdBu_r").copy()
        cmap.set_bad("white")
    else:
        norm = LogNorm(vmin=vmin[i], vmax=vmax[i])
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_under("lightgrey"); cmap.set_bad("white")
    for j, it in enumerate(iters):
        ax = axes[i][j]
        v = maps[(i, it)].copy()
        if not LINEAR:
            v[(v > 0) & (v < vmin[i])] = vmin[i]
            v[v <= 0] = vmin[i] / 10.0
        im = ax.imshow(v, origin="lower", cmap=cmap, norm=norm)
        if i == 0:
            ax.set_title(f"iter {it}", fontsize=13)
        if j == 0:
            ax.set_ylabel(rn, fontsize=12)
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes[i], shrink=0.85, pad=0.01, extend="neither" if LINEAR else "min")
    cb.set_label("1e-3 MJy/sr um")
fig.suptitle("NEP D4 multi-line probe — aliphatic/plateau INTEGRATED FLUX and their sum\n"
             f"flux_i = A_i x int G_i dlambda; Fisher>={FISH_MIN:g} mask; "
             + ("linear symmetric" if LINEAR else "viridis log (grey = <=0, white = unmasked)")
             + f"; row scale fixed at iter{iters[0]}", fontsize=13)
tag = "_".join(str(i) for i in iters)
out = os.path.join(WS, "figures", f"combined_flux_iter{tag}" + ("_linsym" if LINEAR else "") + ".png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300)
print(f"\n[saved] {out}")
