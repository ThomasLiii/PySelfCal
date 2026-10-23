"""The SKY pass of the N-pass solve: the sky terms given the offsets.

Given the offsets, the sky is block diagonal: each pixel's J sky values are an independent J x J
least-squares problem. A SKY pass subtracts the solved offsets from every frame
(:class:`OffsetModelSubtractor`, a frame hook), streams the per-pixel sums of the normal equations of the
rest (``setup_lsqr(..., sky_rhs_moments=True, moments_only=True)``: no matrix) and solves every pixel once
(:func:`solve_pixels`). The sums are additive over disjoint frame sets, so the tiles' dumps
(:func:`dump_moments`) are summed before anything is solved (:func:`combine_moments`): exactly a
full-field solve, with no seam.

The hook runs inside ``setup_lsqr``'s workers as a POSTprocess function: ``_prep_subframe`` computes
the Poisson weight ``1/sqrt(|data|)`` between the pre- and post-hooks, so subtracting before the
weights would bias every fit toward zero amplitude.
"""
from __future__ import annotations

import os

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from ..geometry.map_helper import compute_chunk_contrib
from ..io.atomic import atomic_path
from ..io.cal_writer import write_sky_groups
from ..io.calfile import CalFile
from ..pipeline.model_eval import frame_observations

__all__ = ["OffsetModelSubtractor", "solve_pixels", "dump_moments", "combine_moments", "write_sky_cal",
           "sky_monitors"]


def _basename(p):
    return os.path.basename(p.decode() if isinstance(p, bytes) else str(p))


# --------------------------------------------------------------------------- #
# the offsets subtracted
# --------------------------------------------------------------------------- #
class OffsetModelSubtractor:
    """A frame hook subtracting a solved offset model at every observation of each frame: every
    offset term of the model.

    ``cal_paths``: one calibration or several (disjoint frame sets, e.g. the tiles of a run) whose
    ``offsets/map_<m>`` hold, per frame, each term's solved values: one per chunk, or ``n_chunks * n``
    coefficients (chunk-major) for a term that multiplies ``n`` known functions of data variables.
    ``terms``: ``[(m, basis, det_chunk_map)]``, the model's offset terms (``basis`` None: a term
    without functions, ``φ ≡ 1``); ``variables``: the data variables, their frame values aligned
    with ``frame_names`` (default: the cals' frame list). The offset at observation ``i`` of chunk
    weights ``w_c(i)`` (the solve's bilinear weights) is ``Σ_m Σ_c w_c(i) Σ_k a_m[c, k] φ_{m,k}(v_i)``
    -- exactly the offset rows of the solve's design matrix times the solved values, whatever the
    terms are. Frames in none of the cals pass through unchanged. A cal of the older layout with a
    separate per-frame scalar is refused (:meth:`~selfcal.io.calfile.CalFile.check_offsets_complete`).
    """

    def __init__(self, cal_paths, terms, variables=None, oversample_factor=1, frame_names=None):
        if isinstance(cal_paths, (str, bytes, os.PathLike)):
            cal_paths = [cal_paths]
        self.terms = [(int(m), b, np.asarray(cm)) for m, b, cm in terms]
        self.variables = variables
        self.oversample_factor = oversample_factor
        self.rows, self.index = {}, {}
        names = []
        for path in cal_paths:
            with CalFile(path) as cal:
                cal.check_offsets_complete()
                offs = cal.offsets
                cal_names = [os.path.basename(p) for p in cal.reproj_list]
            for m, _, _ in self.terms:
                if m >= len(offs):
                    raise ValueError(f"{os.path.basename(os.fspath(path))} has {len(offs)} offset maps; the model's "
                                     f"term {m} has none there")
                block = np.asarray(offs[m])
                self.rows[m] = block if m not in self.rows else np.concatenate([self.rows[m], block])
            for i, n in enumerate(cal_names):
                self.index[n] = len(names) + i
            names += cal_names
        fn = names if frame_names is None else [os.path.basename(p) for p in frame_names]
        self.var_index = {n: i for i, n in enumerate(fn)}

    def __call__(self, ctx):
        sub_data = ctx.sub_data
        name = os.path.basename(ctx.file)
        i = self.index.get(name)
        if i is None:
            return sub_data
        shapes = {cm.shape for _, _, cm in self.terms}
        for shape in shapes:
            obs, interp = frame_observations(ctx, i, self.variables, oversample_factor=self.oversample_factor,
                                             det_shape=shape, frame_row=self.var_index.get(name, i))
            if obs.n == 0:
                continue
            total = np.zeros(obs.n, dtype=np.float64)
            for m, basis, cm in self.terms:
                if cm.shape != shape:
                    continue
                phi = (np.ones((obs.n, 1)) if basis is None
                       else np.asarray(basis.evaluate(obs), dtype=np.float64))        # (n_obs, n)
                nb = phi.shape[1]
                n_chunks = int(cm.max()) + 1
                a = self.rows[m][i].reshape(n_chunks, nb)
                contrib = compute_chunk_contrib(cm, interp)                       # (n_chunks, n_obs)
                off = np.asarray(contrib.T @ a)                                   # (n_obs, n)
                total += np.nan_to_num((off * phi).sum(axis=1))
            rows, cols = obs.pixels
            sub_data[rows, cols] -= total.astype(sub_data.dtype)
        return sub_data


# --------------------------------------------------------------------------- #
# every pixel solved
# --------------------------------------------------------------------------- #
def solve_pixels(pixel_fisher, pixel_cross, pixel_rhs, pixel_counts, num_sky, num_sky_blocks, damp_weights=None,
                 rcond=1e-12, chunk=4_000_000):
    """Every pixel's sky terms from its normal equations, the per-pixel sums ``setup_lsqr`` streams
    (``pixel_fisher`` = Σw²c_j², ``pixel_cross`` = Σw²c_ic_j, a dict ``{(i, j): array}`` or, for J = 2,
    the bare (0, 1) array, ``pixel_rhs`` = Σw²c_j·v): each pixel's J x J system ``M x = r``, the
    diagonal damped as the matrix solve damps it (``damp_weights[j]`` times the term's coverage
    ``pixel_counts``), solved in the minimum-norm sense -- the eigen-directions of ``M`` weaker than
    ``rcond`` times its strongest are left out.

    Without damping, a pixel whose terms the data cannot tell apart (a continuum and a line seen at
    one wavelength) has a normal matrix singular to machine precision: solving it as it is returns
    rounding noise, which raises the objective the pass is meant to lower. The minimum-norm solution
    is what an iterative solve from zero converges to there; a well-determined pixel is solved exactly.

    Returns the block-stacked solution, ``(J*num_sky,)`` float64 (block j at
    ``[j*num_sky:(j+1)*num_sky]``), 0 where a pixel has no data."""
    J = int(num_sky_blocks)
    if not isinstance(pixel_cross, dict):
        pixel_cross = {(0, 1): np.asarray(pixel_cross)} if J > 1 else {}
    dw = list(damp_weights) if damp_weights is not None else [0.0] * J
    M = np.zeros((num_sky, J, J), dtype=np.float64)
    for j in range(J):
        M[:, j, j] = np.asarray(pixel_fisher[j * num_sky:(j + 1) * num_sky], dtype=np.float64)
        if dw[j] and dw[j] > 0:
            M[:, j, j] += float(dw[j]) * np.asarray(pixel_counts[j * num_sky:(j + 1) * num_sky], dtype=np.float64)
    for (i, j), c in pixel_cross.items():
        M[:, i, j] = c
        M[:, j, i] = c
    rhs = np.stack([np.asarray(pixel_rhs[j * num_sky:(j + 1) * num_sky], dtype=np.float64) for j in range(J)],
                   axis=1)
    x = np.zeros((num_sky, J), dtype=np.float64)
    for s in range(0, num_sky, chunk):
        e = min(num_sky, s + chunk)
        idx = s + np.nonzero(np.any(M[s:e] != 0, axis=(1, 2)))[0]
        if idx.size == 0:
            continue
        w, V = np.linalg.eigh(M[idx])                               # ascending eigenvalues, V[:, :, k]
        keep = w > rcond * w[:, -1:]
        inv = np.where(keep, 1.0 / np.where(keep, w, 1.0), 0.0)
        proj = np.einsum('nik,ni->nk', V, rhs[idx])                 # V^T rhs
        x[idx] = np.einsum('nik,nk->ni', V, inv * proj)
    return x.T.reshape(-1)


# --------------------------------------------------------------------------- #
# the tiles' sums: dumps, combined
# --------------------------------------------------------------------------- #
def dump_moments(cc, out_npz):
    """Save a Calibrator's per-pixel normal-equation moments (after
    ``setup_lsqr(chunk_maps=[], ..., sky_rhs_moments=True)``) — additive across
    disjoint frame sets."""
    cc._materialize_pixel_state()
    J = int(cc.num_sky_blocks)
    pc = cc.pixel_cross
    cross = {(0, 1): np.asarray(pc)} if not isinstance(pc, dict) else pc
    payload = dict(pixel_counts=np.asarray(cc.pixel_counts, dtype=np.float64),
                   pixel_fisher=np.asarray(cc.pixel_fisher, dtype=np.float64),
                   pixel_rhs=np.asarray(cc.pixel_rhs, dtype=np.float64),
                   num_sky_blocks=J, ref_shape=np.asarray(cc.ref_shape),
                   reproj_list=np.array([str(p).encode() for p in cc.reproj_list]))
    for (i, j), c in cross.items():
        payload[f"cross_{i}_{j}"] = np.asarray(c, dtype=np.float64)
    with atomic_path(out_npz) as tmp:
        np.savez(tmp, **payload)
    print(f"[npass] moments dumped ({len(cc.reproj_list)} frames, J={J}) -> {out_npz}", flush=True)
    return out_npz


def combine_moments(dumps, out_cal, *, sky_names, damp_weights, line_fisher_threshold=None,
                    attrs=None):
    """Sum per-tile moment dumps and solve every pixel once (exactly a
    full-field solve). Writes a v3 sky-only cal via :func:`write_sky_cal`."""
    acc = None
    reproj = []
    for i, d in enumerate(dumps):
        z = np.load(d)
        keys = [k for k in z.files if k in ("pixel_counts", "pixel_fisher", "pixel_rhs")
                or k.startswith("cross_")]
        legacy = "pixel_cross" in z.files          # older dumps: bare (0,1) cross array
        if acc is None:
            acc = {k: z[k].astype(np.float64) for k in keys}
            if legacy:
                acc["cross_0_1"] = z["pixel_cross"].astype(np.float64)
            J = int(z["num_sky_blocks"])
            ref_shape = tuple(int(v) for v in z["ref_shape"])
        else:
            for k in keys:
                acc[k] += z[k]
            if legacy:
                acc["cross_0_1"] += z["pixel_cross"]
        if "reproj_list" in z.files:
            reproj.extend(_basename(p) for p in z["reproj_list"])
        del z
        print(f"[npass] combine: +{os.path.basename(d)} ({i+1}/{len(dumps)})", flush=True)
    if len(set(reproj)) != len(reproj):
        raise ValueError("combine_moments: a frame appears in more than one dump "
                         "(moments would be double-counted)")
    if len(sky_names) != J:
        raise ValueError(f"{len(sky_names)} sky names for J={J} blocks")
    cross = {tuple(int(t) for t in k.split("_")[1:]): v for k, v in acc.items()
             if k.startswith("cross_")}
    num_sky = ref_shape[0] * ref_shape[1]
    print(f"[npass] combine: solving {num_sky:,} pixels, J={J}, damp {damp_weights} ...", flush=True)
    x = solve_pixels(acc["pixel_fisher"], cross, acc["pixel_rhs"], acc["pixel_counts"], num_sky, J,
                     damp_weights=damp_weights)
    maps = [x[j * num_sky:(j + 1) * num_sky].reshape(ref_shape).astype(np.float32)
            for j in range(J)]
    counts = [acc["pixel_counts"][j * num_sky:(j + 1) * num_sky].reshape(ref_shape)
              for j in range(J)]
    fishers = [acc["pixel_fisher"][j * num_sky:(j + 1) * num_sky].reshape(ref_shape)
               for j in range(J)]
    write_sky_cal(out_cal, ref_shape=ref_shape, sky_names=list(sky_names), sky_maps=maps,
                  sky_counts=counts, sky_fishers=fishers, pixel_cross=cross,
                  pixel_fisher=acc["pixel_fisher"], reproj_list=reproj,
                  line_fisher_threshold=line_fisher_threshold,
                  attrs=dict({"recipe": "per-tile moment dumps summed, one per-pixel "
                                        "closed-form solve (== full-field solve)",
                              "n_tiles": len(dumps), "damp_weights": np.asarray(damp_weights)},
                             **(attrs or {})))
    for j, n in enumerate(sky_names):
        m = np.isfinite(maps[j]) & (fishers[j] >= (line_fisher_threshold or 10.0))
        if m.any():
            print(f"[npass] combine: {n:12s} median {np.median(maps[j][m])*1e3:+.2f}e-3, "
                  f"{100*np.mean(maps[j][m] > 0):.1f}% positive ({m.sum():,} px)", flush=True)
    print(f"[npass] combine: saved {out_cal}", flush=True)
    return out_cal


def write_sky_cal(path, *, ref_shape, sky_names, sky_maps, sky_counts, sky_fishers,
                  pixel_cross, pixel_fisher, reproj_list, line_fisher_threshold=None,
                  attrs=None):
    """Write a sky-only (``num_maps = 0``) cal in the v3 layout
    ``Calibrator.save_calibration`` produces: ``sky/<name>``, ``sky_coverage/``,
    ``sky_fisher/``, ``sky_separability/<name>`` for every spectral block, and
    the ``skymap`` / ``skymap_line`` hard-link aliases.
    """
    J = len(sky_names)
    with atomic_path(path) as tmp, h5py.File(tmp, "w") as f:
        f.attrs["num_maps"] = 0
        write_sky_groups(f, sky_names=list(sky_names),
                         sky_maps=[np.asarray(m, np.float32) for m in sky_maps],
                         sky_coverages=[np.asarray(c) for c in sky_counts],
                         sky_fishers=[np.asarray(fi, np.float32) for fi in sky_fishers],
                         pixel_cross=pixel_cross, pixel_fisher=pixel_fisher,
                         ref_shape=tuple(ref_shape), num_sky_blocks=J,
                         line_fisher_threshold=line_fisher_threshold)
        f.create_dataset("reproj_list", data=np.array([str(p).encode() for p in reproj_list]))
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
    return path


# --------------------------------------------------------------------------- #
# monitors
# --------------------------------------------------------------------------- #
def sky_monitors(cal_path, prev_cal_path=None, fisher_min=10.0):
    """Per-block gauge indicators (median, % positive at Fisher >= fisher_min)
    and, given the previous SKY product, the RMS step ``|ΔS_j|``."""
    out = {}
    with h5py.File(cal_path, "r") as f:
        names = _sky_block_names(f)
        prev = h5py.File(prev_cal_path, "r") if prev_cal_path else None
        try:
            pnames = _sky_block_names(prev) if prev is not None else []
            for j, n in enumerate(names):
                m = _sky_block(f, n, j)
                fi = _sky_block(f, n, j, kind="fisher")
                ok = np.isfinite(m) & (fi >= fisher_min) if fi is not None else np.isfinite(m)
                d = {"median": float(np.median(m[ok])) if ok.any() else None,
                     "frac_positive": float(np.mean(m[ok] > 0)) if ok.any() else None,
                     "n_px": int(ok.sum())}
                if prev is not None and j < len(pnames):
                    pm = _sky_block(prev, pnames[j], j)
                    both = ok & np.isfinite(pm)
                    d["step_rms"] = float(np.sqrt(np.mean((m[both] - pm[both]) ** 2))) if both.any() else None
                out[n] = d
        finally:
            if prev is not None:
                prev.close()
    return out


def _sky_block_names(f):
    """Sky block names of a v3 (``sky/<name>``) or v2 (``skymap`` [+ ``skymap_line``]) cal."""
    if "sky_components" in f.attrs:
        return [n.decode() if isinstance(n, bytes) else str(n) for n in f.attrs["sky_components"]]
    return ["continuum"] + (["line"] if "skymap_line" in f else [])


def _sky_block(f, name, j, kind="map"):
    """Block ``j`` (named ``name``) of a v3 or v2 cal; ``kind`` in map|fisher."""
    if "sky" in f and name in f["sky"]:
        grp = f["sky"] if kind == "map" else f.get("sky_fisher")
        return grp[name][:] if grp is not None and name in grp else None
    key = ("skymap" if j == 0 else "skymap_line") + ("" if kind == "map" else "_fisher")
    return f[key][:] if key in f else None
