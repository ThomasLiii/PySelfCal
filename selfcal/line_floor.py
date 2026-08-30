"""Line-floor anchor — the external convention for the uniform line level.

Why this exists
---------------
In a spectral (continuum + line) self-calibration, a detector-static pattern
shaped like the line profile G(BC) and a UNIFORM sky-line floor are *exactly*
degenerate: G(BC) is itself a detector map, so no dither or scan geometry ever
separates "instrument pedestal shaped like G" from "constant PAH floor on the
sky". Every offset model tried on SEP D4 differed from every other only along
this one gauge direction (line medians from -18 to +64 x1e-3 MJy/sr for maps
whose STRUCTURE was the same). The uniform level is therefore not measurable
from the data and must be set by an external convention.

This module does that the same way the zodi anchor does absolute DC:
**non-mutating**. The cal file stays pristine; the floor lands in a sidecar
``<cal>.floor.h5`` and is applied at read time by ``apply_floor``.

Convention implemented here (``fit_floor_reference_region``): the median line
value inside a user-designated emission-free REFERENCE REGION is defined to be
zero (or a supplied physical value ``ref_level``). The region is any boolean
mask on the reference grid — typically a hand-chosen high-latitude patch, or
an external-catalog-based mask. Only pixels above ``fisher_min`` count.

Because the floor is a single scalar per line map, applying it shifts the map
uniformly and cannot alter structure — which is exactly why it is safe to
decide post hoc, and why it belongs outside the solver.
"""
from __future__ import annotations

import os
import numpy as np
import h5py

__all__ = ["fit_floor_reference_region", "write_floor", "load_floor",
           "apply_floor", "floor_sidecar_path"]

FLOOR_VERSION = 1


def floor_sidecar_path(cal_path: str) -> str:
    """Sidecar file next to the cal: ``<cal>.floor.h5``."""
    return cal_path + ".floor.h5"


def fit_floor_reference_region(cal_path: str, ref_mask: np.ndarray, *,
                               line_key: str = "skymap_line",
                               fisher_key: str = "skymap_line_fisher",
                               fisher_min: float = 10.0,
                               ref_level: float = 0.0,
                               stat: str = "median") -> dict:
    """Measure the floor of ``line_key`` in a reference region.

    Returns a dict with ``floor`` (the value that maps the region's statistic
    to ``ref_level``, i.e. ``floor = stat(region) - ref_level``; SUBTRACT it),
    the region statistic itself, its robust scatter, and the pixel count.
    """
    with h5py.File(cal_path, "r") as f:
        line = f[line_key][:]
        fish = f[fisher_key][:] if fisher_key in f else None
    ref_mask = np.asarray(ref_mask, dtype=bool)
    if ref_mask.shape != line.shape:
        raise ValueError(f"ref_mask shape {ref_mask.shape} != map shape {line.shape}")
    sel = ref_mask & np.isfinite(line)
    if fish is not None:
        sel &= fish >= fisher_min
    n = int(sel.sum())
    if n == 0:
        raise ValueError("reference region has no valid pixels")
    v = line[sel].astype(np.float64)
    if stat == "median":
        level = float(np.median(v))
    elif stat == "mean":
        level = float(np.mean(v))
    else:
        raise ValueError(f"unknown stat {stat!r}")
    mad = float(1.4826 * np.median(np.abs(v - np.median(v))))
    return dict(floor=level - float(ref_level), region_stat=level, region_nmad=mad,
                n_pixels=n, stat=stat, ref_level=float(ref_level),
                fisher_min=float(fisher_min), line_key=line_key)


def write_floor(cal_path: str, result: dict, *, region_name: str = "",
                out_path: str | None = None) -> str:
    """Write the floor sidecar for ``cal_path``. Never touches the cal."""
    out = out_path or floor_sidecar_path(cal_path)
    with h5py.File(out, "w") as f:
        f.attrs["floor_version"] = FLOOR_VERSION
        f.attrs["source_cal"] = os.path.abspath(cal_path)
        f.attrs["region_name"] = region_name
        for k, v in result.items():
            f.attrs[k] = v
    return out


def load_floor(cal_path_or_sidecar: str) -> dict:
    """Read a floor sidecar (pass the cal path or the sidecar path)."""
    p = cal_path_or_sidecar
    if not p.endswith(".floor.h5"):
        p = floor_sidecar_path(p)
    with h5py.File(p, "r") as f:
        return {k: (v.item() if hasattr(v, "item") else v) for k, v in f.attrs.items()}


def apply_floor(line_map: np.ndarray, floor: float | dict) -> np.ndarray:
    """Return ``line_map - floor`` (floor may be the dict from load_floor).
    Uniform shift only; NaNs preserved; input untouched."""
    fl = float(floor["floor"]) if isinstance(floor, dict) else float(floor)
    return line_map - fl
