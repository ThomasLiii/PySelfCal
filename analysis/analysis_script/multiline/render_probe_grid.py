"""Component x iteration map grid for the NEP 1k multi-line probe cals.

Rows = sky blocks (continuum, aromatic, aliphatic, plateau), cols = iterations.
Fisher>=10 mask, crop to covered bbox, LINEAR symmetric RdBu_r with ONE shared
scale per row (fixed from the first iteration's |p99| so later-iteration
degradation shows as saturation, not rescaling). Colorbar per row.
Units 1e-3 MJy/sr. (Symlog abandoned per user 2026-07-13.)

Usage: python render_probe_grid.py [--log] [--ipmask TAU] [iter ...]
  --log : viridis + LogNorm (positive values only; negatives shown grey,
          no-coverage white) instead of the default linear symmetric RdBu_r.
  --ipmask TAU : additionally mask spectral blocks where separability
          I_P < TAU (read-time mask; kills the pole convergence-zone hole
          where coverage is deep but wavelength diversity ~0).
"""
import os, sys
import numpy as np
import hdf5plugin  # noqa: F401
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm

WS = os.path.dirname(os.path.abspath(__file__))
CAL_DIR = "/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration"
BASE = "cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_multiline3_probe1k"
TAIL = "_polybasisD2noortho_NumCol3_outThresh5_sigma2"
FISH_MIN = 10.0

args = sys.argv[1:]
LOG_STYLE = "--log" in args
IP_TAU = None
if "--ipmask" in args:
    k = args.index("--ipmask")
    IP_TAU = float(args[k + 1])
    del args[k:k + 2]
iters = [int(a) for a in args if a != "--log"] or [25, 50, 100]
paths = {it: os.path.join(CAL_DIR, f"{BASE}_iter{it}{TAIL}.h5") for it in iters}
for it, p in paths.items():
    if not os.path.exists(p):
        sys.exit(f"missing cal for iter{it}: {p}")

# read all blocks, masked + cropped
with h5py.File(paths[iters[0]], "r") as f:
    names = [n.decode() if isinstance(n, bytes) else str(n)
             for n in f.attrs["sky_components"]]
    cov = f["sky_coverage"][names[0]][:] > 0
rows = np.where(cov.any(axis=1))[0]; cols = np.where(cov.any(axis=0))[0]
m = 20
r0, r1 = max(0, rows[0] - m), min(cov.shape[0], rows[-1] + 1 + m)
c0, c1 = max(0, cols[0] - m), min(cov.shape[1], cols[-1] + 1 + m)
sl = (slice(r0, r1), slice(c0, c1))
print(f"crop bbox y[{r0}:{r1}] x[{c0}:{c1}]  ({r1-r0}x{c1-c0})")

maps = {}   # (name, iter) -> masked 1e-3 MJy/sr map
for it in iters:
    with h5py.File(paths[it], "r") as f:
        for nm in names:
            v = f["sky"][nm][sl].astype(np.float64) * 1e3
            fish = f["sky_fisher"][nm][sl]
            bad = ~((fish >= FISH_MIN) & np.isfinite(v))
            if IP_TAU is not None and nm in f.get("sky_separability", {}):
                bad |= f["sky_separability"][nm][sl] < IP_TAU
            v[bad] = np.nan
            maps[(nm, it)] = v

# shared per-row scale from the FIRST iteration
vmax, vmin_log = {}, {}
for nm in names:
    a = maps[(nm, iters[0])]
    vmax[nm] = float(np.nanpercentile(np.abs(a), 99))
    vmin_log[nm] = vmax[nm] / 100.0
    print(f"row {nm}: vmax = {vmax[nm]:.1f}, log vmin = {vmin_log[nm]:.2f} "
          f"1e-3 MJy/sr (p99 |v| at iter{iters[0]}, 2-decade range)")

nR, nC = len(names), len(iters)
fig, axes = plt.subplots(nR, nC, figsize=(4.6 * nC + 1.2, 4.4 * nR),
                         squeeze=False, constrained_layout=True)
for i, nm in enumerate(names):
    if LOG_STYLE:
        norm = LogNorm(vmin=vmin_log[nm], vmax=vmax[nm])
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_under("lightgrey")   # non-positive values
        cmap.set_bad("white")         # no coverage / below Fisher cut
    else:
        norm = Normalize(vmin=-vmax[nm], vmax=vmax[nm])
        cmap = "RdBu_r"
    for j, it in enumerate(iters):
        ax = axes[i][j]
        v = maps[(nm, it)]
        if LOG_STYLE:
            v = v.copy()
            v[(v > 0) & (v < vmin_log[nm])] = vmin_log[nm]  # keep low positives in-range
            v[v <= 0] = vmin_log[nm] / 10.0   # -> 'under' grey, distinct from bad
        im = ax.imshow(v, origin="lower", cmap=cmap, norm=norm)
        if i == 0:
            ax.set_title(f"iter {it}", fontsize=13)
        if j == 0:
            ax.set_ylabel(nm, fontsize=13)
        ax.set_xticks([]); ax.set_yticks([])
    cb = fig.colorbar(im, ax=axes[i], shrink=0.85, pad=0.01, extend="min" if LOG_STYLE else "neither")
    cb.set_label("1e-3 MJy/sr")
style = ("viridis log (grey = value<=0, white = unmasked)" if LOG_STYLE
         else "linear symmetric")
ipnote = f" + I_P>={IP_TAU:g} (lines)" if IP_TAU is not None else ""
fig.suptitle(f"NEP D4 multi-line probe (1k deep-center frames) — sky blocks vs LSQR iterations\n"
             f"Fisher>={FISH_MIN:g}{ipnote} mask; {style}; "
             f"row scale fixed at iter{iters[0]}", fontsize=13)
tag = ("_".join(str(i) for i in iters) + ("_logviridis" if LOG_STYLE else "_linsym")
       + (f"_ip{IP_TAU:g}" if IP_TAU is not None else ""))
out = os.path.join(WS, "figures", f"probe_grid_iter{tag}.png")
os.makedirs(os.path.dirname(out), exist_ok=True)
fig.savefig(out, dpi=300)
print(f"[saved] {out}")
