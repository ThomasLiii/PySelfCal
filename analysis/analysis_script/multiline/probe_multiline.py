"""Multi-block conditioning diagnostics for v3 spectral cals (label=path args).

Per sky block (over that block's Fisher>=10 covered pixels): percentiles,
median, frac_pos, blowup fractions. Then ALL pairwise Pearson correlations
(continuum + every line) over the joint mask — the campaign degeneracy metric
(near 0 = decoupled; strongly negative = per-pixel null-space cross-talk) —
plus each line's separability I_P quantiles and the Pearson restricted to the
I_P>=Q50 half (degeneracy lives in the low-I_P tail).
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
import numpy as np
import hdf5plugin  # noqa: F401
import h5py
from scipy import ndimage as ndi

FISH_MIN = 10.0


def probe(tag, path):
    if not os.path.exists(path):
        print(f"\n===== {tag}: MISSING {path}")
        return
    with h5py.File(path, "r") as f:
        names = [n.decode() if isinstance(n, bytes) else str(n)
                 for n in f.attrs["sky_components"]]
        sky = {nm: f["sky"][nm][:] for nm in names}
        fish = {nm: f["sky_fisher"][nm][:] for nm in names}
        cov = {nm: f["sky_coverage"][nm][:] > 0 for nm in names}
        sep = {}
        if "sky_separability" in f:
            for nm in names:
                if nm in f["sky_separability"]:
                    sep[nm] = f["sky_separability"][nm][:]
    joint = np.ones_like(cov[names[0]])
    for nm in names:
        joint &= cov[nm] & (fish[nm] >= FISH_MIN) & np.isfinite(sky[nm])
    print(f"\n===== {tag}  ({joint.sum():,} joint Fisher>={FISH_MIN:g} px, "
          f"grid {sky[names[0]].shape}) =====")
    vals = {nm: sky[nm][joint] * 1e3 for nm in names}
    for nm in names:
        a = vals[nm]
        p = np.percentile(a, [0.1, 1, 50, 99, 99.9])
        extra = (f"  frac_pos={100*np.mean(a > 0):5.1f}%" if nm != names[0] else "")
        print(f"  {nm:<10s} 1e-3 MJy/sr pct[.1,1,50,99,99.9]="
              f"{p[0]:9.1f}{p[1]:8.1f}{p[2]:8.2f}{p[3]:8.1f}{p[4]:9.1f}"
              f"  |max|={np.abs(a).max():7.0f}{extra}")
    c = vals[names[0]]
    print(f"  blowup |cont|>100: {100*np.mean(np.abs(c) > 100):.2f}%   "
          f">1000: {100*np.mean(np.abs(c) > 1000):.3f}%")
    cf = np.where(cov[names[0]] & np.isfinite(sky[names[0]]), sky[names[0]], 0.0)
    hi = (sky[names[0]] - ndi.median_filter(cf, size=3))[joint]
    print(f"  cont high-freq frac: {np.std(hi)/(np.std(c/1e3)+1e-12):.2f}"
          f"   (>~0.7 = high-freq null-space noise)")
    print("  pairwise Pearson (joint mask):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            r = np.corrcoef(vals[names[i]], vals[names[j]])[0, 1]
            print(f"    {names[i]:<10s} x {names[j]:<10s}: {r:+.3f}")
    for nm in names[1:]:
        if nm not in sep:
            continue
        s = sep[nm][joint]
        q = np.percentile(s, [10, 50, 90])
        rich = s >= q[1]
        r_cont = np.corrcoef(vals[names[0]][rich], vals[nm][rich])[0, 1]
        print(f"  I_P({nm}): q10/50/90 = {q[0]:.3g}/{q[1]:.3g}/{q[2]:.3g}; "
              f"cont-Pearson on I_P>=Q50 half: {r_cont:+.3f}")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        tag, path = arg.split("=", 1)
        probe(tag, path)
