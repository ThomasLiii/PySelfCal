"""
DEPRECATED: superseded by selfcal.pipeline.tiled.stitch (TiledCalibration.stitch),
driven by run_cal_tiled_NEP.py. The package stitcher is byte-equal to this on the
same inputs (gated by cache/refactor_gate/verify_stitch.py). Kept for reference.

Stitch 4 disjoint-frame quadrant cal h5 files (NW/NE/SW/SE) into a single
cal-shaped h5 with Fisher-weighted inverse-variance averages.

Schema preservation (matches per-quadrant cal h5):
    skymap                (12676, 12672) float32  -- Fisher-weighted sky_cont
    skymap_line           (12676, 12672) float32  -- Fisher-weighted sky_line
    skymap_fisher         (12676, 12672) float32  -- summed continuum Fisher
    skymap_line_fisher    (12676, 12672) float32  -- summed line Fisher
    skymap_coverage       (12676, 12672) int64    -- summed n_obs
    skymap_line_coverage  (12676, 12672) int64    -- summed n_obs

Diagnostic extras (not in the per-quadrant cal h5):
    n_contrib_cont        (12676, 12672) uint8    -- count of chunks with Fisher>0
    n_contrib_line        (12676, 12672) uint8

Dropped (not composable across disjoint frame subsets):
    offsets/, offset_coverage/, offset_coverage_frac/, chunk_maps/,
    frame_scalar, reproj_list.  num_maps is written as 0.

Attrs preserved / added:
    line_fisher_threshold   = 10.0   (passthrough; verified identical across inputs)
    num_sky_blocks          = 2      (passthrough; verified identical across inputs)
    num_maps                = 0      (per-frame quantities are dropped)
    stitched_from           = [4 input cal paths]
    stitched_method         = 'fisher_weighted_inverse_variance'
    stitcher_version        = 'fisher-stream-v1'

Streaming pattern: process one chunk at a time, free arrays immediately after
accumulation.  Accumulators stay in float64 for the weighted-sum step; cast
back to float32 at write time to match the cal-h5 dtype contract.

Per-quadrant DC-offset estimation across overlap regions is NOT applied here.
The 4 quadrants share a single ref WCS and were each solved with the same
LSQR damping; the frame_scalar zero-point folded into each quadrant's skymap
is determined by the same regularization, so any residual quadrant-DC
mismatch is small.  The seam diagnostic at the end quantifies it.

Usage:
    python stitch_cals.py [NW.h5 NE.h5 SW.h5 SE.h5] [-o OUT.h5]

If no args given, defaults to the 4 hardcoded NEP 2026W17 D4 quadrant paths.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import h5py
import numpy as np


REF_SHAPE = (12676, 12672)
STITCHER_VERSION = "fisher-stream-v1"

CAL_DIR = (
    "/data3/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D4_6p2arcsec/calibration"
)
_BASENAME = (
    "cal_Detector4_NumSub10_NumCh34_NumCol5_Aromatic_PAHfit_damp0p1_reg0p1_"
    "applyWt_PAHfit_dampL5e-3_subch60_nosrcmask_NumCol5_full_{chunk}_iter200_"
    "subchPoly3_w100_outThresh5_sigma2_polyK1.h5"
)
DEFAULT_INPUTS = [
    os.path.join(CAL_DIR, _BASENAME.format(chunk=c)) for c in ("NW", "NE", "SW", "SE")
]
DEFAULT_OUTPUT = os.path.join(
    CAL_DIR,
    _BASENAME.format(chunk="NWNESWSE_stitched"),
)


def _bbox_nonzero(cov: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.where(cov.any(axis=1))[0]
    cols = np.where(cov.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return None
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def _check_consistency(paths: list[str]) -> dict:
    keys_to_check = ("line_fisher_threshold", "num_sky_blocks")
    ref = {}
    for i, p in enumerate(paths):
        with h5py.File(p, "r") as f:
            if f["skymap"].shape != REF_SHAPE:
                raise ValueError(
                    f"{p}: skymap shape {f['skymap'].shape} != {REF_SHAPE}"
                )
            for k in keys_to_check:
                if k not in f.attrs:
                    raise ValueError(f"{p}: missing required attr {k!r}")
                v = f.attrs[k]
                if i == 0:
                    ref[k] = v
                elif np.asarray(v) != np.asarray(ref[k]):
                    raise ValueError(
                        f"{p}: attr {k}={v!r} disagrees with first input {ref[k]!r}"
                    )
    return ref


def stitch(input_paths, output_path, seam_y_range=(5800, 6900)):
    if len(input_paths) < 1:
        raise ValueError(f"need at least 1 input cal file, got {len(input_paths)}")
    if len(input_paths) < 4:
        print(f"[stitch] WARNING: only {len(input_paths)} input(s) -- result is a PARTIAL stitch (some sky regions will be uncovered)", flush=True)
    for p in input_paths:
        if not os.path.isfile(p):
            raise FileNotFoundError(p)

    print(f"[stitch] {STITCHER_VERSION}", flush=True)
    print(f"[stitch] inputs ({len(input_paths)}):", flush=True)
    for p in input_paths:
        print(f"    {p}", flush=True)
    print(f"[stitch] output: {output_path}", flush=True)

    ref_attrs = _check_consistency(input_paths)
    print(f"[stitch] consistency OK; ref attrs = {ref_attrs}", flush=True)

    H, W = REF_SHAPE
    num_cont = np.zeros((H, W), dtype=np.float64)
    num_line = np.zeros((H, W), dtype=np.float64)
    den_cont = np.zeros((H, W), dtype=np.float64)
    den_line = np.zeros((H, W), dtype=np.float64)
    cov_cont = np.zeros((H, W), dtype=np.int64)
    cov_line = np.zeros((H, W), dtype=np.int64)
    n_cont = np.zeros((H, W), dtype=np.uint8)
    n_line = np.zeros((H, W), dtype=np.uint8)

    y0_seam, y1_seam = seam_y_range
    seam_cache = {}
    chunk_tags = []

    for p in input_paths:
        tag = _infer_chunk_tag(p)
        chunk_tags.append(tag)
        t0 = time.time()
        print(f"[stitch] [{tag}] reading {os.path.basename(p)} ...", flush=True)
        with h5py.File(p, "r") as f:
            cov_c_full = f["skymap_coverage"][:]
            bb = _bbox_nonzero(cov_c_full)
            if bb is None:
                print(f"[stitch] [{tag}] WARNING: empty coverage; skipping", flush=True)
                continue
            r0, r1, c0, c1 = bb
            sky_c = f["skymap"][r0:r1, c0:c1]
            fish_c = f["skymap_fisher"][r0:r1, c0:c1]
            cov_c = cov_c_full[r0:r1, c0:c1]
            sky_l = f["skymap_line"][r0:r1, c0:c1]
            fish_l = f["skymap_line_fisher"][r0:r1, c0:c1]
            cov_l = f["skymap_line_coverage"][r0:r1, c0:c1]
            seam_cache[tag] = (
                f["skymap_line"][y0_seam:y1_seam, :].astype(np.float64, copy=False),
                f["skymap_line_fisher"][y0_seam:y1_seam, :].astype(np.float64, copy=False),
                f["skymap_fisher"][y0_seam:y1_seam, :].astype(np.float64, copy=False),
            )
        del cov_c_full

        m_c = fish_c > 0
        if m_c.any():
            sl = (slice(r0, r1), slice(c0, c1))
            fc64 = fish_c.astype(np.float64, copy=False)
            sc64 = sky_c.astype(np.float64, copy=False)
            num_cont[sl] += fc64 * sc64 * m_c
            den_cont[sl] += fc64 * m_c
            cov_cont[sl] += cov_c
            n_cont[sl] += m_c.astype(np.uint8)
            del fc64, sc64

        m_l = fish_l > 0
        if m_l.any():
            sl = (slice(r0, r1), slice(c0, c1))
            fl64 = fish_l.astype(np.float64, copy=False)
            sl64 = sky_l.astype(np.float64, copy=False)
            num_line[sl] += fl64 * sl64 * m_l
            den_line[sl] += fl64 * m_l
            cov_line[sl] += cov_l
            n_line[sl] += m_l.astype(np.uint8)
            del fl64, sl64

        del sky_c, fish_c, cov_c, sky_l, fish_l, cov_l
        print(
            f"[stitch] [{tag}] accumulated in {time.time()-t0:.1f}s  "
            f"(bbox y[{r0}:{r1}] x[{c0}:{c1}])",
            flush=True,
        )

    print("[stitch] computing Fisher-weighted ratios ...", flush=True)
    m_c = den_cont > 0.0
    m_l = den_line > 0.0
    sky_cont = np.zeros((H, W), dtype=np.float32)
    sky_line = np.zeros((H, W), dtype=np.float32)
    np.divide(num_cont, den_cont, out=sky_cont, where=m_c, casting="unsafe")
    np.divide(num_line, den_line, out=sky_line, where=m_l, casting="unsafe")
    skymap_fisher = den_cont.astype(np.float32)
    skymap_line_fisher = den_line.astype(np.float32)

    _seam_diagnostic(seam_cache, chunk_tags, seam_y_range)

    print(f"[stitch] writing {output_path} ...", flush=True)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp = output_path + ".tmp"
    with h5py.File(tmp, "w") as f:
        f.create_dataset("skymap", data=sky_cont)
        f.create_dataset("skymap_line", data=sky_line)
        f.create_dataset("skymap_fisher", data=skymap_fisher)
        f.create_dataset("skymap_line_fisher", data=skymap_line_fisher)
        f.create_dataset("skymap_coverage", data=cov_cont)
        f.create_dataset("skymap_line_coverage", data=cov_line)
        f.create_dataset("n_contrib_cont", data=n_cont)
        f.create_dataset("n_contrib_line", data=n_line)
        f.attrs["num_sky_blocks"] = ref_attrs["num_sky_blocks"]
        f.attrs["num_maps"] = np.int64(0)
        f.attrs["line_fisher_threshold"] = ref_attrs["line_fisher_threshold"]
        f.attrs["stitched_from"] = np.array(input_paths, dtype="S")
        f.attrs["stitched_method"] = "fisher_weighted_inverse_variance"
        f.attrs["stitcher_version"] = STITCHER_VERSION
    os.replace(tmp, output_path)

    n_pix_cont = int(m_c.sum())
    n_pix_line = int(m_l.sum())
    n_overlap_cont = int((n_cont >= 2).sum())
    n_overlap_line = int((n_line >= 2).sum())
    print(
        f"[stitch] done.  cont pixels covered: {n_pix_cont:,}  "
        f"(overlap n>=2: {n_overlap_cont:,})",
        flush=True,
    )
    print(
        f"[stitch]       line pixels covered: {n_pix_line:,}  "
        f"(overlap n>=2: {n_overlap_line:,})",
        flush=True,
    )


def _infer_chunk_tag(path):
    base = os.path.basename(path)
    for tag in ("NW", "NE", "SW", "SE"):
        if f"_full_{tag}_" in base:
            return tag
    return base[:24]


def _seam_diagnostic(seam_cache, chunk_tags, seam_y_range):
    """Pairwise chi-squared seam quality on sky_line near the y=6313 seam.

    For each overlapping pair on pixels where both have skymap_line_fisher > 1:
        chi2_px = (sky_l_A - sky_l_B)^2 / (1/F_l_A + 1/F_l_B)
    chi2 ~ 1 means seam values agree within Fisher-implied 1-sigma;
    chi2 >> 1 indicates a real DC mismatch Fisher-averaging cannot reconcile.
    """
    print(
        f"[seam ] sky_line seam quality in y[{seam_y_range[0]}:{seam_y_range[1]}]  "
        f"(pairwise chi2 = dy^2 / (1/F_A + 1/F_B))",
        flush=True,
    )
    pairs = [(a, b) for i, a in enumerate(chunk_tags) for b in chunk_tags[i + 1 :]]
    any_overlap = False
    for a, b in pairs:
        if a not in seam_cache or b not in seam_cache:
            continue
        sky_a, fish_a, _ = seam_cache[a]
        sky_b, fish_b, _ = seam_cache[b]
        ok = (fish_a > 1.0) & (fish_b > 1.0)
        n_ok = int(ok.sum())
        if n_ok == 0:
            print(f"[seam ]   {a} vs {b}: no overlap with F_line > 1", flush=True)
            continue
        any_overlap = True
        da = sky_a[ok]; db = sky_b[ok]
        fa = fish_a[ok]; fb = fish_b[ok]
        var_sum = (1.0 / fa) + (1.0 / fb)
        chi2 = (da - db) ** 2 / var_sum
        print(
            f"[seam ]   {a} vs {b}: n_overlap_px={n_ok:>10,}  "
            f"mean_chi2={float(np.mean(chi2)):8.3f}  "
            f"p99_chi2={float(np.percentile(chi2, 99.0)):10.3f}  "
            f"mean|dy|={float(np.mean(np.abs(da - db))):.4g}",
            flush=True,
        )
    if not any_overlap:
        print("[seam ]   no pairwise overlap found in seam band -- check seam_y_range", flush=True)


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Stitch 4 quadrant cal h5 -> single cal-shaped h5")
    p.add_argument("inputs", nargs="*", default=None,
                   help="4 input cal h5 paths (NW NE SW SE order recommended).  "
                        "Defaults to the hardcoded NEP 2026W17 D4 quadrant paths.")
    p.add_argument("-o", "--output", default=DEFAULT_OUTPUT,
                   help=f"output cal h5 path (default: {DEFAULT_OUTPUT})")
    p.add_argument("--seam-y0", type=int, default=5800)
    p.add_argument("--seam-y1", type=int, default=6900)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    inputs = args.inputs if args.inputs else DEFAULT_INPUTS
    if len(inputs) < 1:
        print(f"error: need at least 1 input cal h5 path, got {len(inputs)}", file=sys.stderr)
        sys.exit(2)
    stitch(input_paths=list(inputs), output_path=args.output,
           seam_y_range=(args.seam_y0, args.seam_y1))
