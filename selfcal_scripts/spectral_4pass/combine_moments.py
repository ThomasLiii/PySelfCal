"""Combine per-tile moment dumps into ONE seam-free full-field sky solve.

A per-pixel closed-form sky solve that uses only the frames staged in one tile
solves boundary pixels from a biased subset of their observations, which
prints as a seam. The closed form makes tiles exactly ADDITIVE, though: each
pixel's normal equations are sums over observations, so summing the per-tile
moments (``closed_form_pass4 --dump-moments``) and solving once is identical
to a single full-field solve. Tiling becomes pure memory bookkeeping.

Requires each frame to belong to exactly ONE tile (``make_frame_lists``).

    python -m selfcal_scripts.spectral_4pass.combine_moments <out_cal.h5> \\
        <template_cal.h5> <moments.npz>... [--damp-weight 0.1] [--damp-weight-line 0.005]
"""
import os
import sys

import numpy as np
import hdf5plugin  # noqa: F401
import h5py

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from selfcal.core.solution import solve_sky_closed_form  # noqa: E402


def main(out_cal, template_cal, dumps, damp_weight=0.1, damp_weight_line=0.005):
    acc = None
    n_frames = 0
    for i, d in enumerate(dumps):
        z = np.load(d)
        if acc is None:
            acc = {k: z[k].astype(np.float64).copy()
                   for k in ("pixel_counts", "pixel_fisher", "pixel_cross", "pixel_rhs")}
            J = int(z["num_sky_blocks"])
            ref_shape = tuple(int(v) for v in z["ref_shape"])
        else:
            for k in acc:
                acc[k] += z[k]
        n_frames += int(z["n_frames"]) if "n_frames" in z else 0
        del z
        print(f"[combine] +{os.path.basename(d)} ({i+1}/{len(dumps)})", flush=True)
    num_sky = ref_shape[0] * ref_shape[1]
    print(f"[combine] solving {num_sky:,} pixels, J={J} ...", flush=True)
    x = solve_sky_closed_form(acc["pixel_fisher"], acc["pixel_cross"], acc["pixel_rhs"],
                              acc["pixel_counts"], num_sky, J,
                              damp_weights=[damp_weight, damp_weight_line][:J])
    cont = x[:num_sky].reshape(ref_shape).astype(np.float32)
    line = x[num_sky:2 * num_sky].reshape(ref_shape).astype(np.float32)
    F = [acc["pixel_fisher"][j * num_sky:(j + 1) * num_sky] for j in range(J)]
    cov = [acc["pixel_counts"][j * num_sky:(j + 1) * num_sky] for j in range(J)]
    I_P = np.maximum(F[1] - np.where(F[0] > 0, acc["pixel_cross"] ** 2 /
                                     np.maximum(F[0], 1e-300), 0.0), 0.0)
    with h5py.File(template_cal, "r") as t, h5py.File(out_cal, "w") as f:
        f.create_dataset("skymap", data=cont, **hdf5plugin.Blosc())
        f.create_dataset("skymap_line", data=line, **hdf5plugin.Blosc())
        f.create_dataset("skymap_fisher", data=F[0].reshape(ref_shape).astype(np.float32),
                         **hdf5plugin.Blosc())
        f.create_dataset("skymap_line_fisher", data=F[1].reshape(ref_shape).astype(np.float32),
                         **hdf5plugin.Blosc())
        f.create_dataset("skymap_coverage", data=cov[0].reshape(ref_shape).astype(np.float32),
                         **hdf5plugin.Blosc())
        f.create_dataset("skymap_line_coverage", data=cov[1].reshape(ref_shape).astype(np.float32),
                         **hdf5plugin.Blosc())
        g = f.create_group("sky_separability")
        g.create_dataset("pah_3p29", data=I_P.reshape(ref_shape).astype(np.float32),
                         **hdf5plugin.Blosc())
        for k, v in t.attrs.items():
            if k not in f.attrs:
                f.attrs[k] = v
        f.attrs["recipe"] = ("per-tile moment dumps summed, then ONE per-pixel closed-form "
                             "solve -- exactly equivalent to a single full-field solve")
        f.attrs["n_tiles"] = len(dumps)
        f.attrs["n_frames"] = n_frames
        f.attrs["damp_weight"] = damp_weight
        f.attrs["damp_weight_line"] = damp_weight_line
    m = np.isfinite(line) & (F[1].reshape(ref_shape) >= 10)
    print(f"[combine] line median {np.median(line[m])*1e3:+.2f}e-3, "
          f"{100*np.mean(line[m] > 0):.1f}% positive | cont median "
          f"{np.median(cont[m])*1e3:+.2f}e-3", flush=True)
    print(f"[combine] saved {out_cal}", flush=True)
    return out_cal


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out_cal")
    ap.add_argument("template_cal", help="a per-tile cal whose attrs are copied (mode metadata)")
    ap.add_argument("dumps", nargs="+")
    ap.add_argument("--damp-weight", type=float, default=0.1)
    ap.add_argument("--damp-weight-line", type=float, default=0.005)
    a = ap.parse_args()
    main(os.path.abspath(a.out_cal), os.path.abspath(a.template_cal),
         [os.path.abspath(d) for d in a.dumps], a.damp_weight, a.damp_weight_line)
