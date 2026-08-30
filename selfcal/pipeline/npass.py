"""Primitives of the N-pass alternating solve (runner task ``npass``).

Model, per frame *k* and reference pixel *p*::

    d_k(p) = Σ_j S_j(p) c_j(λ_k(p)) + Σ_d a_{k,g(p),d} B_d(u(p)) + s_k

with J sky blocks ``S_j`` (block 0 = continuum, ``c_0 ≡ 1``; the others are
:class:`~selfcal.models.sky_model.SpectralComponent` amplitudes with profile
coefficients ``c_j`` evaluated at the observation's wavelength), a per-frame
offset that is a mean-zero Chebyshev shape ``B_d`` in an abstract per-chunk
coordinate ``u`` (SPHEREx: the subchannel) with one polynomial per group ``g``
(SPHEREx: the detector column), and a per-frame scalar ``s_k``.

The joint problem is solved by alternating least squares. Given the offsets the
sky is block-diagonal — one J×J normal system per pixel, solved exactly by
:func:`selfcal.core.solution.solve_sky_closed_form` from the per-pixel moments
that ``setup_lsqr(..., sky_rhs_moments=True)`` streams. Those moments are sums
over observations, so per-tile dumps are ADDITIVE: :func:`combine_moments`
sums them and solves once, which is exactly a full-field solve (tiling is pure
memory bookkeeping, no seam can exist). Given the sky, the offsets are
independent per frame — :func:`refit_offsets_per_frame` is a dense least
squares per frame over the whole field, no tiles and no gauge to connect.

Hooks (:class:`OffsetSubtractor`, :class:`SkySubtractor`) run inside
``setup_lsqr``'s workers as POSTprocess functions: ``_prep_subframe`` computes
the Poisson weight ``1/sqrt(|data|)`` between the pre- and post-hooks, so
subtracting before the weights would bias every fit toward zero amplitude.

Nothing here knows an instrument: the chunk→(coordinate, group) encoding comes
in as a ``poly_basis`` spec (see :mod:`selfcal.models.offset_basis`) and the
per-subchannel clip edges as an array (see
``SPHERExInstrument.subchannel_bc_edges``).
"""
from __future__ import annotations

import os
import time
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import h5py
import hdf5plugin  # noqa: F401
from scipy.ndimage import map_coordinates

from ..core.subframe import _prep_subframe
from ..core.solution import solve_sky_closed_form
from ..geometry.map_helper import chunk_to_det, find_outliers_grouped
from ..models.offset_basis import cheb_shape_basis

__all__ = [
    "group_wavelength_edges", "sky_damp_weights",
    "OffsetSubtractor", "SkySubtractor",
    "refit_offsets_per_frame", "dump_moments", "combine_moments", "write_sky_cal",
    "sky_monitors", "offset_monitors",
]


# --------------------------------------------------------------------------- #
# small helpers
# --------------------------------------------------------------------------- #
def group_wavelength_edges(det_wavelength, det_chunk_map, group_of_chunk, min_pixels=50):
    """Wavelength bin edges (midpoints between consecutive per-group mean
    wavelengths) for the per-group outlier clip (``outlier_subchannel_edges``).

    ``group_of_chunk[chunk]`` maps a chunk id to its clip group (SPHEREx: the
    subchannel). Groups with fewer than ``min_pixels`` valid pixels are skipped.
    """
    w = np.asarray(det_wavelength, dtype=np.float64)
    cm = np.asarray(det_chunk_map)
    grp = np.where(cm >= 0, np.asarray(group_of_chunk)[np.maximum(cm, 0)], -1)
    ngrp = int(grp.max()) + 1
    mean = np.full(ngrp, np.nan)
    valid = np.isfinite(w) & (w > 0) & (grp >= 0)
    cnt = np.bincount(grp[valid].ravel(), minlength=ngrp)
    sums = np.bincount(grp[valid].ravel(), weights=w[valid].ravel(), minlength=ngrp)
    ok = cnt >= min_pixels
    mean[ok] = sums[ok] / cnt[ok]
    ws = np.sort(mean[ok])
    return 0.5 * (ws[:-1] + ws[1:])


def sky_damp_weights(sky_model, damp_weight, damp_weight_line=None):
    """Per-block damping weights, mirroring ``Calibrator.solve_sky_closed_form``:
    block 0 gets ``damp_weight``; each spectral component its own ``damp_weight``
    if set, else ``damp_weight_line`` (0 when neither is given)."""
    dws = [float(damp_weight or 0.0)]
    for comp in sky_model.components[1:]:
        w = getattr(comp, "damp_weight", None)
        if w is not None:
            dws.append(float(w))
        elif damp_weight_line is not None:
            dws.append(float(damp_weight_line))
        else:
            dws.append(0.0)
    return dws


def _basename(p):
    return os.path.basename(p.decode() if isinstance(p, bytes) else str(p))


# --------------------------------------------------------------------------- #
# hooks
# --------------------------------------------------------------------------- #
class OffsetSubtractor:
    """Subtract solved per-frame offsets (per-chunk offset + frame scalar) from
    each frame's ``sub_data``, keyed by reproj filename.

    Accepts one or several cal files (e.g. the per-tile pass-1 cals, whose
    frame sets are disjoint) and the offsets fragments written by
    :func:`refit_offsets_per_frame`. Uses the same ``chunk_to_det`` + bilinear
    det→sub sampling the solver applied, so it reproduces exactly the offset
    the joint solve would have subtracted. Frames absent from every cal pass
    through unchanged.
    """

    def __init__(self, cal_paths):
        if isinstance(cal_paths, (str, bytes)):
            cal_paths = [cal_paths]
        self.by_file = {}
        self.cm = None
        n_dup = 0
        for path in cal_paths:
            with h5py.File(path, "r") as f:
                off = f["offsets/map_0"][:]
                sc = f["frame_scalar"][:] if "frame_scalar" in f else np.zeros(off.shape[0])
                cm = f["chunk_maps/map_0"][:]
                names = [_basename(r) for r in f["reproj_list"][:]]
                ok = f["fit_ok"][:] if "fit_ok" in f else np.ones(off.shape[0], bool)
            if self.cm is None:
                self.cm = cm
            elif self.cm.shape != cm.shape or not np.array_equal(self.cm, cm):
                raise ValueError(f"chunk map of {path} differs from the first cal's")
            total = off + sc[:, None]
            for i, n in enumerate(names):
                if not ok[i]:
                    continue
                if n in self.by_file:
                    n_dup += 1
                self.by_file[n] = total[i]
        vals = np.stack(list(self.by_file.values())) if self.by_file else np.zeros((1, 1))
        print(f"[npass] offset subtractor: {len(self.by_file)} frames from {len(cal_paths)} cal(s)"
              f"{f', {n_dup} duplicates (last wins)' if n_dup else ''}; "
              f"|median| {np.median(np.abs(vals))*1e3:.2f}, |max| {np.max(np.abs(vals))*1e3:.1f} "
              f"(1e-3 MJy/sr)", flush=True)

    def __call__(self, loc):
        sub_data = loc["sub_data"]
        row = self.by_file.get(os.path.basename(loc["file"]))
        if row is None:
            return sub_data
        grid_off = chunk_to_det(self.cm, chunk_data=row)
        sm = np.asarray(loc["sub_mapping"]).reshape(2, -1)      # [x, y] det coords
        sub_off = map_coordinates(grid_off, sm[::-1], order=1, mode="constant",
                                  cval=0.0).reshape(sub_data.shape)
        return sub_data - sub_off


class SkySubtractor:
    """Subtract a solved sky ``Σ_j S_j(p) c_j(λ)`` at each observation, for any
    number of sky blocks.

    The J maps are exported once by the parent as ``.npy`` files and memory-
    mapped in each worker (page cache shared across the pool), so J large maps
    cost one copy, not one per worker. ``sky_model`` supplies the component
    names (``sky/<name>`` in the cal) and their coefficient functions;
    ``aux_keys`` names the entries of the per-pixel aux list
    (SPHEREx: ``('BC', 'BW')``).
    """

    def __init__(self, sky_cal, sky_model, export_dir, aux_keys=("BC", "BW")):
        self.sky_cal = sky_cal
        self.sky_model = sky_model
        self.names = list(sky_model.names)
        self.aux_keys = tuple(aux_keys)
        self.export_dir = export_dir
        self.paths = {n: os.path.join(export_dir, f"{n}.npy") for n in self.names}
        self._maps = None
        os.makedirs(export_dir, exist_ok=True)
        with h5py.File(sky_cal, "r") as f:
            for n in self.names:
                src = f["sky"][n] if "sky" in f else (f["skymap"] if n == self.names[0]
                                                        else f["skymap_line"])
                if not os.path.exists(self.paths[n]):
                    np.save(self.paths[n], np.nan_to_num(src[:].astype(np.float32)))
            self.shape = tuple(f["skymap"].shape)

    # --- worker side ---------------------------------------------------------
    def _load(self):
        if self._maps is None:
            self._maps = [np.load(self.paths[n], mmap_mode="r") for n in self.names]
        return self._maps

    def window(self, rc, shape):
        """The J sky maps on the subframe grid, plus an on-map mask.

        The subframe's ``ref_coords`` may overhang ANY map edge. A plain
        ``map[y0:y1, x0:x1]`` handles only the high side: a NEGATIVE y0/x0 is a
        Python negative slice start and silently returns an EMPTY window — for
        the 94 SEP frames overhanging the bottom/left edge that meant NO sky
        subtracted, and their refit offsets absorbed the LMC emission (the
        white-streak bug, fixed 2026-08-24). Off-map pixels get sky 0 and
        ``on_map=False``; fits against the residual must exclude them.
        """
        maps = self._load()
        y0, x0 = int(rc[0]), int(rc[2])
        sh, sw = shape
        H, W = self.shape
        on_map = np.zeros(shape, bool)
        out = [np.zeros(shape, np.float32) for _ in maps]
        ys, ye = max(0, -y0), min(sh, H - y0)
        xs, xe = max(0, -x0), min(sw, W - x0)
        if ye > ys and xe > xs:
            for o, m in zip(out, maps):
                o[ys:ye, xs:xe] = m[y0 + ys:y0 + ye, x0 + xs:x0 + xe]
            on_map[ys:ye, xs:xe] = True
        return out, on_map

    def coefficients(self, sub_aux, shape):
        """``c_j`` per subframe pixel for every block (block 0 is all ones)."""
        aux = {k: np.asarray(sub_aux[i]) for i, k in enumerate(self.aux_keys)
               if sub_aux is not None and i < len(sub_aux)}
        out = [None]                     # block 0: continuum, coefficient 1
        for comp in self.sky_model.components[1:]:
            c = comp.coefficients(aux)
            out.append(np.asarray(c, dtype=np.float64).reshape(shape))
        return out

    def predict(self, rc, shape, sub_aux):
        """Total modelled sky on the subframe grid (float64), and the on-map mask."""
        maps, on_map = self.window(rc, shape)
        coef = self.coefficients(sub_aux, shape)
        pred = maps[0].astype(np.float64)
        for m, c in zip(maps[1:], coef[1:]):
            pred = pred + m * c
        return pred, on_map

    def __call__(self, loc):
        sub_data = loc["sub_data"]
        pred, _ = self.predict(loc["ref_coords"], sub_data.shape, loc.get("sub_aux"))
        return sub_data - pred


# --------------------------------------------------------------------------- #
# OFFSET pass: per-frame refit against a fixed sky
# --------------------------------------------------------------------------- #
_W = {}   # per-worker state


def _refit_init(state):
    _W.clear()
    _W.update(state)
    pb = state["poly_basis"]
    coord = np.asarray(pb["chunk_coord"], dtype=float)
    _W["Bc"] = cheb_shape_basis(coord, int(pb["degree"]), pb["coord_lo"], pb["coord_hi"])
    _W["grp"] = np.asarray(pb["chunk_group"], dtype=int)
    _W["deg"] = int(pb["degree"])
    _W["ngroups"] = int(pb["num_groups"])


def _refit_frame(path):
    """Min-norm least squares of (data − sky) on [per-group shape coeffs, DC].

    Returns ``(basename, (offsets_per_chunk, scalar) | None, resid_rms, n_used)``
    with ``n_used < 0`` flagging the all-pixels fallback.
    """
    try:
        g = _W
        ref_coords, sub_data, sub_weight, contribs, sub_aux = _prep_subframe(
            file=path, chunk_offsets=None, for_lsqr=True, det_offset_funcs=None,
            det_aux=g["det_aux"], chunk_maps=[g["cm"]],
            apply_weight=True, apply_mask=True, ignore_list=g["ignore_list"],
            grid_valid_weight=g["grid_valid"], oversample_factor=1,
            valid_threshold=0.5, postprocess_func=g["sky"], preprocess_func=None)
        sub_h, sub_w = sub_data.shape
        valid = sub_weight > 0
        if g["edges"] is not None:
            masked = np.where(valid, sub_data, np.nan)
            groups = np.digitize(sub_aux[0], g["edges"])
            valid &= ~find_outliers_grouped(masked, groups, threshold=g["thresh"])
        pred, on_map = g["sky"].predict(ref_coords, sub_data.shape, sub_aux)
        valid &= on_map
        used_all = False
        if g["bright_cut"] is not None:
            faint = valid & (pred < g["bright_cut"])
            used_all = int(faint.sum()) < g["min_pix"]
            if not used_all:
                valid = faint
        vc = np.nonzero(valid)
        v = sub_data[vc]
        w = sub_weight[vc]
        n = v.size
        if n < 200:
            return os.path.basename(path), None, np.nan, n
        pix = vc[0] * sub_w + vc[1]
        cc = contribs[0][:, pix].tocoo()
        nz = cc.data != 0
        chunk_i, obs_i, cval = cc.row[nz], cc.col[nz], cc.data[nz]
        deg, ngroups = g["deg"], g["ngroups"]
        D = np.zeros((n, ngroups * deg + 1))
        wc = w[obs_i] * cval
        for d in range(deg):
            np.add.at(D, (obs_i, g["grp"][chunk_i] * deg + d), wc * g["Bc"][chunk_i, d])
        D[:, -1] = w
        a, *_ = np.linalg.lstsq(D, w * v, rcond=None)
        off = np.zeros(len(g["grp"]), dtype=np.float32)
        for d in range(deg):
            off += (a[g["grp"] * deg + d] * g["Bc"][:, d]).astype(np.float32)
        resid = w * v - D @ a
        return (os.path.basename(path), (off, np.float32(a[-1])), float(np.std(resid)),
                -n if used_all else n)
    except Exception:
        return os.path.basename(path), None, np.nan, -1


def refit_offsets_per_frame(frames, sky, *, det_chunk_map, grid_valid, det_aux, poly_basis,
                            edges=None, ignore_list=(), thresh=2.5, bright_cut=0.05,
                            min_pix=5000, out_h5, max_workers=48, attrs=None):
    """OFFSET pass: refit every frame's offset against the fixed sky ``sky``
    (a :class:`SkySubtractor`). Independent per frame, hence over the whole
    field at once. Writes an offsets fragment (``offsets/map_0``,
    ``frame_scalar``, ``chunk_maps/map_0``, ``reproj_list``, ``fit_ok``,
    ``resid_rms``) that :class:`OffsetSubtractor` consumes. Returns
    ``(out_h5, monitor_dict)``.
    """
    state = dict(sky=sky, cm=np.asarray(det_chunk_map), grid_valid=grid_valid,
                 det_aux=det_aux, poly_basis=poly_basis, edges=edges,
                 ignore_list=list(ignore_list), thresh=float(thresh),
                 bright_cut=bright_cut, min_pix=int(min_pix))
    n_chunks = len(poly_basis["chunk_group"])
    offsets = np.zeros((len(frames), n_chunks), dtype=np.float32)
    scalars = np.zeros(len(frames), dtype=np.float32)
    rms = np.full(len(frames), np.nan, dtype=np.float32)
    ok = np.zeros(len(frames), dtype=bool)
    fell_back = 0
    t0 = time.time()
    print(f"[npass] OFFSET refit: {len(frames)} frames, deg {poly_basis['degree']} x "
          f"{poly_basis['num_groups']} groups + DC, clip {thresh}, bright cut {bright_cut}",
          flush=True)
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_refit_init,
                             initargs=(state,)) as ex:
        for i, (name, fit, r, n) in enumerate(ex.map(_refit_frame, frames, chunksize=8)):
            if fit is not None:
                offsets[i], scalars[i] = fit
                ok[i] = True
                rms[i] = r
                if n < 0:
                    fell_back += 1
            if (i + 1) % 2000 == 0:
                print(f"[npass]   {i+1}/{len(frames)}  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[npass] OFFSET refit done in {time.time()-t0:.0f}s; fitted {ok.sum()}/{len(frames)}; "
          f"{fell_back} bright-cut fallbacks", flush=True)
    with h5py.File(out_h5, "w") as f:
        f.create_dataset("offsets/map_0", data=offsets, **hdf5plugin.Blosc())
        f.create_dataset("frame_scalar", data=scalars)
        f.create_dataset("chunk_maps/map_0", data=np.asarray(det_chunk_map), **hdf5plugin.Blosc())
        f.create_dataset("reproj_list", data=np.array([str(p).encode() for p in frames]))
        f.create_dataset("fit_ok", data=ok)
        f.create_dataset("resid_rms", data=rms)
        f.attrs["num_maps"] = 1
        f.attrs["model"] = (f"OFFSET pass: per-frame min-norm LSQ of (data - sky) on deg-"
                            f"{poly_basis['degree']} Chebyshev per group + DC scalar")
        f.attrs["sky_cal"] = sky.sky_cal
        f.attrs["bright_cut"] = -1.0 if bright_cut is None else float(bright_cut)
        for k, v in (attrs or {}).items():
            f.attrs[k] = v
    mon = dict(n_frames=len(frames), n_fit=int(ok.sum()), n_fallback=int(fell_back),
               resid_rms_median=float(np.nanmedian(rms)) if ok.any() else None,
               scalar_median=float(np.median(scalars[ok])) if ok.any() else None,
               wall_s=time.time() - t0)
    return out_h5, mon


# --------------------------------------------------------------------------- #
# SKY pass: moment dumps + combine + writer
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
    np.savez(out_npz, **payload)
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
    x = solve_sky_closed_form(acc["pixel_fisher"], cross, acc["pixel_rhs"],
                              acc["pixel_counts"], num_sky, J, damp_weights=damp_weights)
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
    from ..io.cal_writer import write_sky_groups
    J = len(sky_names)
    with h5py.File(path, "w") as f:
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


def offset_monitors(off_path, prev_off_path=None):
    """Offset DC median / spread and, given the previous offsets product
    (any cal with ``offsets/map_0``), the RMS step ``|Δa|`` over shared frames."""
    with h5py.File(off_path, "r") as f:
        off = f["offsets/map_0"][:] + f["frame_scalar"][:][:, None]
        names = [_basename(r) for r in f["reproj_list"][:]]
        ok = f["fit_ok"][:] if "fit_ok" in f else np.ones(off.shape[0], bool)
        rms = f["resid_rms"][:] if "resid_rms" in f else None
    out = {"n_frames": int(off.shape[0]), "n_fit": int(ok.sum()),
           "dc_median": float(np.median(off[ok].mean(axis=1))) if ok.any() else None,
           "dc_p16_p84": [float(v) for v in np.percentile(off[ok].mean(axis=1), [16, 84])] if ok.any() else None,
           "resid_rms_median": float(np.nanmedian(rms[ok])) if rms is not None and ok.any() else None}
    if prev_off_path:
        with h5py.File(prev_off_path, "r") as f:
            poff = f["offsets/map_0"][:] + (f["frame_scalar"][:][:, None] if "frame_scalar" in f else 0.0)
            pnames = [_basename(r) for r in f["reproj_list"][:]]
        idx = {n: i for i, n in enumerate(pnames)}
        pairs = [(i, idx[n]) for i, n in enumerate(names) if n in idx and ok[i]]
        if pairs and poff.shape[1] == off.shape[1]:
            a = np.array([i for i, _ in pairs]); b = np.array([j for _, j in pairs])
            out["step_rms"] = float(np.sqrt(np.mean((off[a] - poff[b]) ** 2)))
            out["n_shared"] = len(pairs)
    return out


def append_monitor(path, record):
    """Append one pass record to the JSON monitor file (list of dicts)."""
    data = []
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
    data = [r for r in data if r.get("pass") != record.get("pass")] + [record]
    data.sort(key=lambda r: r.get("pass", 0))
    with open(path, "w") as f:
        json.dump(data, f, indent=1, default=float)
    return data
