"""Per-subframe hooks shared by every step of the 4-pass chain.

All three run inside ``setup_lsqr``'s worker processes, so they must be
picklable module-level objects (no closures).

Weight-hook rule: ``OffsetSubtractor`` and ``SkySubtractor`` are POSTprocess
hooks. ``_prep_subframe`` computes the Poisson weight ``1/sqrt(|data|)``
BETWEEN the pre- and post-hooks; subtracting in preprocess would build weights
from near-zero subtracted values (anti-correlated with |v|) and shrink every
fit toward zero amplitude. Subtracting AFTER the weights keeps them identical
to the baseline's.
"""
import os

import numpy as np
import hdf5plugin  # noqa: F401
import h5py
from scipy.ndimage import map_coordinates

from selfcal.geometry.map_helper import chunk_to_det

__all__ = ["subchannel_bc_edges", "OffsetSubtractor", "SkySubtractor"]


def subchannel_bc_edges(inst, cfg):
    """BC bin edges (midpoints between consecutive subchannel mean-BCs) for the
    per-subchannel outlier grouping (``outlier_subchannel_edges``).

    Instrument-specific, so computed here from the adapter's aux map rather
    than in the generic core.
    """
    di = inst.detector_inputs(cfg.instrument_cfg, cfg.oversample)
    det_BC = np.asarray(inst.aux(di)[0], dtype=np.float64)
    cm = np.asarray(di["det_chunk_map"])
    ncol = int(cfg.instrument_cfg["num_col"])
    sub = cm // ncol
    nsub = int(sub.max()) + 1
    bc_sub = np.full(nsub, np.nan)
    for s in range(nsub):
        m = (sub == s) & np.isfinite(det_BC) & (det_BC > 0)
        if m.sum() > 50:
            bc_sub[s] = np.nanmean(det_BC[m])
    good = np.isfinite(bc_sub)
    bcs = np.sort(bc_sub[good])
    edges = 0.5 * (bcs[:-1] + bcs[1:])
    print(f"[4pass] subchannel BC edges: {len(edges)} edges spanning "
          f"{bcs[0]:.4f}-{bcs[-1]:.4f} um ({good.sum()} subchannels)", flush=True)
    return edges


class OffsetSubtractor:
    """Subtract a solved per-frame offset (per-chunk offset + frame scalar)
    from each frame's ``sub_data``, keyed by reproj filename.

    Uses the same ``chunk_to_det`` + bilinear det->sub sampling the solver
    applied, so it reproduces exactly the offset the joint solve would have
    subtracted. Frames absent from the cal are passed through unchanged.
    """

    def __init__(self, cal_path):
        with h5py.File(cal_path, "r") as f:
            det_off = f["offsets/map_0"][:]          # (n_frames, num_chunks)
            fscal = (f["frame_scalar"][:] if "frame_scalar" in f
                     else np.zeros(det_off.shape[0]))
            self.cm = f["chunk_maps/map_0"][:]       # (det_h, det_w)
            reproj = [r.decode() if isinstance(r, bytes) else str(r)
                      for r in f["reproj_list"][:]]
        total = det_off + fscal[:, None]
        self.by_file = {os.path.basename(r): total[i] for i, r in enumerate(reproj)}
        print(f"[4pass] offset subtractor: {len(self.by_file)} frames, "
              f"total-offset |median|={np.median(np.abs(total))*1e3:.2f}, "
              f"|max|={np.max(np.abs(total))*1e3:.1f} (1e-3 MJy/sr)", flush=True)

    def __call__(self, loc):
        sub_data = loc["sub_data"]
        row = self.by_file.get(os.path.basename(loc["file"]))
        if row is None:
            return sub_data
        grid_off = chunk_to_det(self.cm, chunk_data=row)       # (det_h, det_w)
        sm = np.asarray(loc["sub_mapping"]).reshape(2, -1)     # [x, y] det coords
        sub_off = map_coordinates(grid_off, sm[::-1], order=1, mode="constant",
                                  cval=0.0).reshape(sub_data.shape)
        return sub_data - sub_off


class SkySubtractor:
    """Subtract a solved sky (``cont + line * G(BC)``) at each observation.

    Lazy per-worker load of the two ref-grid maps (~1.2 GB once per worker);
    the pickled instance carries only paths.
    """

    def __init__(self, sky_cal, tpl_path):
        self.sky_cal = sky_cal
        self.tpl_path = tpl_path
        self._loaded = False

    def _load(self):
        with h5py.File(self.sky_cal, "r") as f:
            self.cont = np.nan_to_num(f["skymap"][:].astype(np.float32))
            self.line = np.nan_to_num(f["skymap_line"][:].astype(np.float32))
        t = np.load(self.tpl_path)
        self.tw = np.asarray(t["center_um"], float)
        self.tg = np.asarray(t["G_peaknorm"], float)
        self._loaded = True

    def window(self, rc, shape):
        """Sky (cont, line) on the subframe grid, plus an on-map mask.

        The subframe's ``ref_coords`` may overhang ANY map edge. A plain
        ``self.cont[y0:y1, x0:x1]`` handles only the high side: a NEGATIVE
        y0/x0 is a Python negative slice start and silently returns an EMPTY
        window, so a zero-padded fallback subtracted NO sky at all for the 94
        SEP frames overhanging the bottom/left edge — pass 3 then fitted the
        raw data and its offset absorbed the LMC emission (the white-streak
        bug, fixed 2026-08-24). Pixels off the map get sky 0 and
        ``on_map=False``; callers fitting against the residual must exclude
        them.
        """
        if not self._loaded:
            self._load()
        y0, x0 = int(rc[0]), int(rc[2])
        sh, sw = shape
        H, W = self.cont.shape
        cont_w = np.zeros(shape, np.float32)
        line_w = np.zeros(shape, np.float32)
        on_map = np.zeros(shape, bool)
        ys, ye = max(0, -y0), min(sh, H - y0)
        xs, xe = max(0, -x0), min(sw, W - x0)
        if ye > ys and xe > xs:
            cont_w[ys:ye, xs:xe] = self.cont[y0 + ys:y0 + ye, x0 + xs:x0 + xe]
            line_w[ys:ye, xs:xe] = self.line[y0 + ys:y0 + ye, x0 + xs:x0 + xe]
            on_map[ys:ye, xs:xe] = True
        return cont_w, line_w, on_map

    def line_coeff(self, sub_aux):
        """G(BC) per subframe pixel (0 outside the template's range)."""
        return np.interp(sub_aux[0], self.tw, self.tg, left=0.0, right=0.0)

    def __call__(self, loc):
        if not self._loaded:
            self._load()
        sub_data = loc["sub_data"]
        aux = loc.get("sub_aux")
        cont_w, line_w, _ = self.window(loc["ref_coords"], sub_data.shape)
        G = self.line_coeff(aux) if aux is not None else 0.0
        return sub_data - (cont_w + line_w * G)
