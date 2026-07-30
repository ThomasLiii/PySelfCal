"""Prototype pedestal-subtraction anchor: smooth C(lambda) + per-detector P_D.

Joint fit across all three detectors (D3, D4, D5):

    full_DC[D, k, c] = (1 + amp_D) * zodi_pred[D, k, c] + C(lambda_{D,c}) + P_D

with:
  * ``amp_D`` per-detector multiplicative amplification (3 params).
  * ``C(lambda) = sum_n c_n * lambda_norm^n`` a cubic polynomial in
    ``lambda_norm = (lambda - 2.5) / 1.5`` -- shared across all detectors
    (default degree 3 -> 4 params).
  * ``P_D`` per-detector additive pedestal. ``P_{D_ref}`` is fixed to 0
    (gauge fix vs. the polynomial's constant term); 2 free P params for
    a 3-detector fit.

Total free parameters: 3 (amp) + (C-degree + 1) (poly) + (N_det - 1)
(pedestal) = 9 for the default 3 detectors with cubic C.

This replaces the per-channel anchor's 68 free params per detector
(2 * 34 channels = slope + C) with a strongly-structured 9-param fit
that separates a SMOOTH wavelength-dependent component (real diffuse
sky + smooth instrumental response) from per-detector additive
thermal pedestals.

Pristine cal/anchor files are *not* mutated.

Example::

    python prototype_pedestal_anchor.py \\
        --run-dir /mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D3_6p2arcsec \\
                  /mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D4_6p2arcsec \\
                  /mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D5_6p2arcsec \\
        --out figures/zodi_anchor/prototype_pedestal_anchor.png \\
        --out-data /tmp/prototype_pedestal_data.npz
"""
import argparse
import glob
import os
import re
import sys
import warnings

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from selfcal.zodi_anchor import compute_full_dc, load_anchor


# Detector color map (matches the convention used elsewhere).
DET_COLORS = {3: 'tab:green', 4: 'tab:blue', 5: 'tab:red'}
# Detector wavelength boundaries to mark on the spectrum (um).
DET_BOUNDARIES_UM = (2.42, 3.81)
# Wavelength normalization for the polynomial basis.
WL_CENTER_UM = 2.5
WL_SCALE_UM = 1.5


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--run-dir', nargs='+', required=True,
                   help='One or more SPHEREx run directories. Each must '
                        'contain calibration/, zodi_preds/, and '
                        'zodi_anchor/anchor_D*.h5.')
    p.add_argument('--C-degree', type=int, default=3,
                   help='Degree of the shared C(lambda) polynomial '
                        '(default 3 -> 4 free poly coeffs).')
    p.add_argument('--out', default='figures/zodi_anchor/prototype_pedestal_anchor.png',
                   help='Output PNG path '
                        '(default: figures/zodi_anchor/prototype_pedestal_anchor.png).')
    p.add_argument('--out-data', default='/tmp/prototype_pedestal_data.npz',
                   help='Output .npz path with fitted params + residuals '
                        '(default: /tmp/prototype_pedestal_data.npz).')
    p.add_argument('--sigma', type=float, default=3.0,
                   help='Per-channel sigma-clip threshold on residuals '
                        '(default 3.0).')
    p.add_argument('--n-iter', type=int, default=2,
                   help='Number of sigma-clip refit iterations (default 2).')
    p.add_argument('--reference-detector', type=int, default=3,
                   help='Detector whose P_D is fixed to 0 as the gauge '
                        '(default 3).')
    p.add_argument('--cal-glob-pat', default='cal_*polyK1.h5',
                   help="Glob pattern for cal files inside <run>/calibration "
                        "(default 'cal_*polyK1.h5' -- cal files whose "
                        "filename suffix marks the linear K=1 "
                        "polynomial-constraint solve; set to match your "
                        "run's cal-file suffix).")
    p.add_argument('--joint-data', default='/tmp/joint_resid_data.npz',
                   help='Optional .npz from diag_joint_amp_fit.py for the '
                        'panel (d) overlay of the joint-amp model residual std. '
                        'If missing, the overlay is skipped.')
    return p.parse_args()


# ------------------------------------------------------------------
# I/O helpers (same conventions as diag_joint_amp_fit.py)
# ------------------------------------------------------------------
def parse_detector_from_filename(path):
    m = re.search(r'Detector(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def parse_channel_from_filename(path):
    m = re.search(r'_Ch(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def matching_npz(cal_path, npz_dir):
    base = os.path.basename(cal_path)
    tag = base[len('cal_'):-len('.h5')]
    return os.path.join(npz_dir, f'zodi_pred_{tag}.npz')


def find_anchor_file(run_dir):
    cand = sorted(glob.glob(os.path.join(
        run_dir, 'zodi_anchor', 'anchor_D*.h5')))
    if not cand:
        raise SystemExit(f"no anchor_D*.h5 in {run_dir}/zodi_anchor/")
    if len(cand) > 1:
        raise SystemExit(f"ambiguous anchor files in {run_dir}/zodi_anchor/: "
                         f"{cand}")
    path = cand[0]
    m = re.search(r'anchor_D(\d+)\.h5', os.path.basename(path))
    if m is None:
        raise SystemExit(f"cannot parse detector from {path}")
    return path, int(m.group(1))


def load_cal_and_pred(cal_path, npz_path):
    """Return (full_DC, zodi_pred, wavelength_um) for one (det, ch)."""
    with h5py.File(cal_path, 'r') as f:
        frame_scalar = f['frame_scalar'][:].astype(np.float64)
        offsets_m0 = f['offsets/map_0'][:].astype(np.float64)
        cov_m0 = f['offset_coverage/map_0'][:].astype(np.float64)
    full_dc = compute_full_dc(frame_scalar, offsets_m0, cov_m0)
    with np.load(npz_path) as z:
        zodi_pred = z['zodi_pred'].astype(np.float64)
        wavelength_um = float(z['wavelength_um'])
    if zodi_pred.shape != full_dc.shape:
        raise SystemExit(f"shape mismatch for {cal_path} vs {npz_path}: "
                         f"{full_dc.shape} vs {zodi_pred.shape}")
    return full_dc, zodi_pred, wavelength_um


def process_run(run_dir, cal_glob_pat):
    """Load all 34 channels of one run; return FDC, ZP, WL, anchor info."""
    anchor_path, detector = find_anchor_file(run_dir)
    cal_dir = os.path.join(run_dir, 'calibration')
    npz_dir = os.path.join(run_dir, 'zodi_preds')
    cals = sorted(glob.glob(os.path.join(cal_dir, cal_glob_pat)))
    if not cals:
        raise SystemExit(f"no cal files in {cal_dir} matching {cal_glob_pat}")

    by_ch = {}
    for c in cals:
        if parse_detector_from_filename(c) != detector:
            continue
        ch = parse_channel_from_filename(c)
        if ch is None:
            continue
        by_ch[ch] = c
    chs = sorted(by_ch)
    if not chs:
        raise SystemExit(f"no channels parsed from cals in {cal_dir}")

    first_fdc, first_zp, first_wl = load_cal_and_pred(
        by_ch[chs[0]], matching_npz(by_ch[chs[0]], npz_dir))
    n_frames = first_fdc.shape[0]
    n_channels = len(chs)
    FDC = np.full((n_frames, n_channels), np.nan, dtype=np.float64)
    ZP = np.full((n_frames, n_channels), np.nan, dtype=np.float64)
    WL = np.full(n_channels, np.nan, dtype=np.float64)
    FDC[:, 0] = first_fdc
    ZP[:, 0] = first_zp
    WL[0] = first_wl

    for i, ch in enumerate(chs[1:], start=1):
        cal = by_ch[ch]
        npz = matching_npz(cal, npz_dir)
        if not os.path.exists(npz):
            print(f"  [D{detector} Ch{ch}] npz missing: {npz}; skipping",
                  file=sys.stderr)
            continue
        fdc, zp, wl = load_cal_and_pred(cal, npz)
        if fdc.shape[0] != n_frames:
            raise SystemExit(f"frame-count mismatch for {cal}: "
                             f"{fdc.shape[0]} vs {n_frames}")
        FDC[:, i] = fdc
        ZP[:, i] = zp
        WL[i] = wl

    return dict(
        detector=detector,
        anchor_path=anchor_path,
        channels=np.asarray(chs, dtype=np.int32),
        WL=WL, FDC=FDC, ZP=ZP,
    )


# ------------------------------------------------------------------
# Joint pedestal-anchor fit
# ------------------------------------------------------------------
def lambda_norm(wl_um):
    return (np.asarray(wl_um, dtype=np.float64) - WL_CENTER_UM) / WL_SCALE_UM


def build_design_matrix(per_det_arrays, det_order, c_degree, ref_det):
    """Stack per-detector (FDC, ZP, WL) into a flat LS problem.

    Returns (A, y, finite, det_idx_flat, ch_idx_flat) where each row of A
    is one (frame, channel) observation across all detectors.

    Column layout (n_cols = N_det + (c_degree+1) + (N_det - 1)):
      * 0 .. N_det-1            : amp_D columns (zodi_pred for that det)
      * N_det .. N_det+c_degree : polynomial columns (lambda_norm**n)
      * remaining N_det-1       : pedestal columns (one per non-ref det,
                                  in det_order excluding ref_det)
    """
    n_det = len(det_order)
    nonref_dets = [d for d in det_order if d != ref_det]
    n_poly = c_degree + 1
    n_cols = n_det + n_poly + len(nonref_dets)
    # First count total rows so we can pre-allocate.
    total = 0
    for d in det_order:
        info = per_det_arrays[d]
        total += info['FDC'].size
    A = np.zeros((total, n_cols), dtype=np.float64)
    y = np.zeros(total, dtype=np.float64)
    finite = np.zeros(total, dtype=bool)
    det_idx_flat = np.full(total, -1, dtype=np.int32)
    ch_idx_flat = np.full(total, -1, dtype=np.int32)

    pos = 0
    for di, d in enumerate(det_order):
        info = per_det_arrays[d]
        FDC = info['FDC']
        ZP = info['ZP']
        WL = info['WL']
        n_frames, n_channels = FDC.shape
        n_rows = FDC.size
        # Column index for this det's amp
        amp_col = di
        # Column index for this det's pedestal (if not ref)
        ped_col = (
            n_det + n_poly + nonref_dets.index(d)
            if d != ref_det else None
        )

        # Flatten frame-major over (frame, channel)
        # row order: for frame f, channel c -> idx = f * n_channels + c
        fdc_flat = FDC.reshape(-1)
        zp_flat = ZP.reshape(-1)
        # Build per-row wavelength (broadcast WL[c] across frames).
        wl_per_row = np.broadcast_to(WL[None, :], FDC.shape).reshape(-1)
        ln_per_row = lambda_norm(wl_per_row)
        # y = FDC - ZP  (so model is y = amp*ZP + sum c_n*lambda_norm^n + P_D)
        y_block = fdc_flat - zp_flat
        finite_block = np.isfinite(fdc_flat) & np.isfinite(zp_flat) & \
            np.isfinite(wl_per_row)

        sl = slice(pos, pos + n_rows)
        y[sl] = y_block
        finite[sl] = finite_block
        # amp column: ZP for this det only (zero elsewhere is the default).
        A[sl, amp_col] = zp_flat
        # poly columns
        for n in range(n_poly):
            A[sl, n_det + n] = ln_per_row ** n
        # pedestal column (1.0 for this det, 0 for ref)
        if ped_col is not None:
            A[sl, ped_col] = 1.0

        # det_idx / ch_idx tracking
        det_idx_flat[sl] = di
        # row -> channel index within this det
        ch_for_rows = np.broadcast_to(
            np.arange(n_channels, dtype=np.int32)[None, :],
            FDC.shape,
        ).reshape(-1)
        ch_idx_flat[sl] = ch_for_rows

        pos += n_rows

    return dict(
        A=A, y=y, finite=finite,
        det_idx=det_idx_flat,
        ch_idx=ch_idx_flat,
        det_order=det_order,
        nonref_dets=nonref_dets,
        n_det=n_det,
        n_poly=n_poly,
        ref_det=ref_det,
        c_degree=c_degree,
    )


def joint_pedestal_fit(per_det_arrays, det_order, c_degree=3, ref_det=3,
                       sigma=3.0, n_iter=2):
    """Solve the stacked LS with iterative per-(det, channel) sigma-clip.

    Returns a dict with parameters, per-channel residual std, etc.
    """
    design = build_design_matrix(per_det_arrays, det_order, c_degree, ref_det)
    A = design['A']
    y = design['y']
    finite = design['finite']
    det_idx = design['det_idx']
    ch_idx = design['ch_idx']
    n_det = design['n_det']
    n_poly = design['n_poly']
    nonref_dets = design['nonref_dets']

    inlier = finite.copy()
    # Build a (det, ch) pair key for per-channel clipping; channel
    # indexes are within each detector but are also globally bounded.
    # We use a flat key: det_idx * max_channels + ch_idx.
    max_ch = 1 + int(ch_idx[finite].max()) if finite.any() else 1
    pair_key = det_idx.astype(np.int64) * max_ch + ch_idx.astype(np.int64)

    params = None
    for it in range(int(n_iter) + 1):
        mask = inlier
        if mask.sum() < A.shape[1]:
            raise SystemExit("not enough inlier rows for the joint fit")
        Ai = A[mask]
        yi = y[mask]
        # Solve with lstsq; A has 9 columns, so this is tiny.
        params, *_ = np.linalg.lstsq(Ai, yi, rcond=None)
        resid = y - A @ params

        # Per (det, ch) sigma-clip.
        new_inlier = finite.copy()
        # Compute per-pair std on current inliers.
        # We sort by pair_key once to do this in one pass.
        order = np.argsort(pair_key, kind='stable')
        pk_s = pair_key[order]
        r_s = resid[order]
        in_s = inlier[order]
        fin_s = finite[order]
        # Find pair boundaries.
        change = np.concatenate(
            ([True], pk_s[1:] != pk_s[:-1])
        )
        starts = np.flatnonzero(change)
        ends = np.append(starts[1:], pk_s.size)
        for s, e in zip(starts, ends):
            in_slice = in_s[s:e] & fin_s[s:e]
            if in_slice.sum() < 5:
                continue
            r_slice = r_s[s:e]
            std_p = float(np.std(r_slice[in_slice]))
            if std_p == 0:
                continue
            keep = np.abs(r_slice) < sigma * std_p
            new_in_slice = fin_s[s:e] & keep
            # Map back to original ordering
            idx_orig = order[s:e]
            new_inlier[idx_orig] = new_in_slice
        inlier = new_inlier

    # Final residual + per-(det, ch) std.
    resid = y - A @ params

    # Extract parameter blocks
    amp = params[:n_det]
    c_poly = params[n_det:n_det + n_poly]
    p_nonref = params[n_det + n_poly:]
    P = {d: 0.0 for d in det_order}
    for d, val in zip(nonref_dets, p_nonref):
        P[d] = float(val)
    amp_dict = {d: float(amp[i]) for i, d in enumerate(det_order)}

    # Per-channel residual std per detector.
    per_det_resid_std = {}
    per_det_resid_std_inlier = {}
    for di, d in enumerate(det_order):
        info = per_det_arrays[d]
        n_channels = info['FDC'].shape[1]
        rs = np.full(n_channels, np.nan, dtype=np.float64)
        for c in range(n_channels):
            sel = (det_idx == di) & (ch_idx == c) & inlier
            if sel.sum() >= 2:
                rs[c] = float(np.std(resid[sel]))
        per_det_resid_std[d] = rs

    return dict(
        params=params,
        amp=amp_dict,
        c_poly=np.asarray(c_poly, dtype=np.float64),
        P=P,
        inlier=inlier,
        resid=resid,
        det_idx=det_idx,
        ch_idx=ch_idx,
        det_order=det_order,
        nonref_dets=nonref_dets,
        n_det=n_det,
        n_poly=n_poly,
        per_det_resid_std=per_det_resid_std,
    )


def eval_C(wl_um, c_poly):
    ln = lambda_norm(wl_um)
    n = c_poly.size
    out = np.zeros_like(np.asarray(wl_um, dtype=np.float64))
    for k in range(n):
        out = out + c_poly[k] * ln ** k
    return out


# ------------------------------------------------------------------
# main / plotting
# ------------------------------------------------------------------
def boundary_jumps(per_det_arrays, C_per_det, det_order):
    """C_eff jumps across detector boundaries (D3->D4, D4->D5)."""
    # Sort by detector
    pairs = list(zip(det_order[:-1], det_order[1:]))
    out = {}
    for d_low, d_hi in pairs:
        wl_low = per_det_arrays[d_low]['WL']
        wl_hi = per_det_arrays[d_hi]['WL']
        # Channel 34 of low det, channel 1 of high det.
        idx_low = np.nanargmax(wl_low)
        idx_hi = np.nanargmin(wl_hi)
        c_low = C_per_det[d_low][idx_low]
        c_hi = C_per_det[d_hi][idx_hi]
        out[(d_low, d_hi)] = float(c_hi - c_low)
    return out


def load_joint_resid_overlay(path, dets):
    """Load joint-amp resid_std arrays from diag_joint_amp_fit.py output."""
    out = {}
    if not os.path.exists(path):
        print(f"  [info] {path} not found; skipping joint-amp overlay")
        return out
    try:
        with np.load(path, allow_pickle=False) as z:
            for d in dets:
                key_rs = f'resid_std_D{d}'
                key_wl = f'WL_D{d}'
                if key_rs in z.files and key_wl in z.files:
                    out[d] = dict(
                        WL=np.asarray(z[key_wl], dtype=np.float64),
                        resid_std=np.asarray(z[key_rs], dtype=np.float64),
                    )
    except Exception as e:
        print(f"  [warn] could not read {path}: {e}")
    return out


def sigma_perch_est(sigma_joint, slope_old, amp, var_zp):
    """Per-channel-equivalent residual std estimate.

    Same formula as in diag_resid_features.py."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        delta = slope_old - (1.0 + amp)
        excess = delta * delta * var_zp
        residsq = sigma_joint * sigma_joint - excess
        residsq = np.where(np.isfinite(residsq), residsq, np.nan)
        residsq = np.where(residsq < 0.0, 0.0, residsq)
        return np.sqrt(residsq)


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out_data)) or '.', exist_ok=True)

    per_det_arrays = {}
    per_det_anchor = {}  # det -> {ch: dict with slope_final, C_final}
    per_det_anchor_arrays = {}  # det -> (slope_old[N_ch], C_old[N_ch])
    detectors = []
    for run_dir in args.run_dir:
        print(f"\n=== run-dir: {run_dir} ===")
        info = process_run(run_dir, args.cal_glob_pat)
        d = info['detector']
        per_det_arrays[d] = info
        detectors.append(d)
        anchor = load_anchor(info['anchor_path'])
        per_det_anchor[d] = anchor
        chs = info['channels']
        slope_old = np.full(len(chs), np.nan, dtype=np.float64)
        C_old = np.full(len(chs), np.nan, dtype=np.float64)
        for i, ch in enumerate(chs):
            if int(ch) in anchor.channels:
                slope_old[i] = float(anchor.channels[int(ch)]['slope_final'])
                C_old[i] = float(anchor.channels[int(ch)]['C_final'])
        per_det_anchor_arrays[d] = dict(slope_old=slope_old, C_old=C_old)
        print(f"  D{d}: {len(chs)} channels, "
              f"WL = [{np.nanmin(info['WL']):.3f}, {np.nanmax(info['WL']):.3f}] um")

    det_order = sorted(detectors)
    if args.reference_detector not in det_order:
        raise SystemExit(f"--reference-detector {args.reference_detector} "
                         f"not in loaded detectors {det_order}")

    # ------- Joint pedestal fit -------
    print(f"\n=== Joint pedestal-anchor fit (ref det = D{args.reference_detector}, "
          f"C poly degree = {args.C_degree}) ===")
    fit = joint_pedestal_fit(
        per_det_arrays, det_order,
        c_degree=args.C_degree,
        ref_det=args.reference_detector,
        sigma=args.sigma, n_iter=args.n_iter,
    )
    amp = fit['amp']
    c_poly = fit['c_poly']
    P = fit['P']
    per_det_resid_std = fit['per_det_resid_std']

    # Effective C per channel: C(lambda_c) + P_D
    C_eff_per_det = {}
    for d in det_order:
        wl = per_det_arrays[d]['WL']
        C_eff_per_det[d] = eval_C(wl, c_poly) + P[d]

    # Old C jumps (per-channel anchor) vs new C_eff jumps.
    C_old_per_det = {d: per_det_anchor_arrays[d]['C_old'] for d in det_order}
    jumps_old = boundary_jumps(per_det_arrays, C_old_per_det, det_order)
    jumps_new = boundary_jumps(per_det_arrays, C_eff_per_det, det_order)

    # ------- Print results -------
    print()
    print("--- amplification ---")
    for d in det_order:
        print(f"  amp_D{d:<1}                  = {amp[d]:+.5f}  "
              f"(1 + amp = {1 + amp[d]:.5f})")
    print()
    print("--- C(lambda) polynomial (lambda_norm = (lambda - "
          f"{WL_CENTER_UM:.2f}) / {WL_SCALE_UM:.2f}) ---")
    for k, c in enumerate(c_poly):
        print(f"  c_{k}                       = {c:+.5g}  (MJy/sr per "
              f"lambda_norm^{k})")
    print()
    print("--- per-detector pedestal (P_D, MJy/sr) ---")
    for d in det_order:
        tag = '  [ref, fixed=0]' if d == args.reference_detector else ''
        print(f"  P_D{d:<1}                    = {P[d]:+.5g}{tag}")
    print()
    print("--- per-detector residual std median (MJy/sr) ---")
    resid_std_median = {}
    for d in det_order:
        med = float(np.nanmedian(per_det_resid_std[d]))
        resid_std_median[d] = med
        print(f"  D{d} resid_std median   = {med:.4g}")
    print()
    print("--- boundary C jumps (D_low ch_max -> D_hi ch_min) ---")
    for (d_lo, d_hi), j in jumps_old.items():
        jn = jumps_new[(d_lo, d_hi)]
        print(f"  D{d_lo}->D{d_hi}: old C_c jump        = {j:+.5g} MJy/sr")
        print(f"  D{d_lo}->D{d_hi}: new C_eff jump      = {jn:+.5g} MJy/sr")
    # Reference: inter-detector boundary jumps measured on raw (pre-selfcal)
    # exposure stacks by diag_raw_stack_spectrum.py (same directory), on the
    # NEP 2026W17 runs. The ~+0.017 / ~+0.006 MJy/sr values below are
    # field-specific — re-measure for other fields.
    print()
    print("--- pedestal vs raw-stack reference jumps "
          "(diag_raw_stack_spectrum.py, NEP 2026W17; "
          "re-measure for other fields) ---")
    if 4 in P:
        print(f"  P_D4                     = {P[4]:+.5g} MJy/sr   "
              f"(raw-stack D3->D4 jump  ~ +0.017 MJy/sr)")
    if 4 in P and 5 in P:
        print(f"  P_D5 - P_D4              = {P[5] - P[4]:+.5g} MJy/sr   "
              f"(raw-stack D4->D5 jump  ~ +0.006 MJy/sr)")
    if 5 in P:
        print(f"  P_D5                     = {P[5]:+.5g} MJy/sr")

    # ------- joint-amp overlay data (panel d) -------
    joint_overlay = load_joint_resid_overlay(args.joint_data, det_order)

    # ------- save data -------
    npz_payload = {}
    npz_payload['detectors'] = np.asarray(det_order, dtype=np.int32)
    npz_payload['ref_detector'] = np.int32(args.reference_detector)
    npz_payload['C_poly_coeffs'] = c_poly.astype(np.float64)
    npz_payload['C_degree'] = np.int32(args.C_degree)
    npz_payload['wl_center_um'] = np.float64(WL_CENTER_UM)
    npz_payload['wl_scale_um'] = np.float64(WL_SCALE_UM)
    for d in det_order:
        npz_payload[f'amp_D{d}'] = np.float64(amp[d])
        npz_payload[f'P_D{d}'] = np.float64(P[d])
        npz_payload[f'WL_D{d}'] = per_det_arrays[d]['WL']
        npz_payload[f'resid_std_pedestal_D{d}'] = per_det_resid_std[d]
        npz_payload[f'C_eff_D{d}'] = C_eff_per_det[d]
        npz_payload[f'C_old_D{d}'] = per_det_anchor_arrays[d]['C_old']
        npz_payload[f'slope_old_D{d}'] = per_det_anchor_arrays[d]['slope_old']
    np.savez(args.out_data, **npz_payload)
    print(f"\nSaved data: {args.out_data}")

    # ------- Plot -------
    fig = plt.figure(figsize=(13, 14))
    # Layout: 4 panels stacked.
    gs = fig.add_gridspec(4, 1, height_ratios=[0.5, 1.2, 1.2, 1.2], hspace=0.4)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[1])
    ax_c = fig.add_subplot(gs[2], sharex=ax_b)
    ax_d = fig.add_subplot(gs[3], sharex=ax_b)

    # (a) amp bar chart
    bar_x = np.arange(len(det_order))
    bar_h = [amp[d] for d in det_order]
    colors = [DET_COLORS.get(d, 'tab:gray') for d in det_order]
    ax_a.bar(bar_x, bar_h, color=colors,
             edgecolor='k', linewidth=0.5)
    for x, h, d in zip(bar_x, bar_h, det_order):
        ax_a.annotate(f'{h:+.4f}', xy=(x, h),
                      xytext=(0, 3 if h >= 0 else -10),
                      textcoords='offset points', ha='center',
                      fontsize=9, color='k')
    ax_a.axhline(0, color='k', lw=0.5)
    ax_a.set_xticks(bar_x)
    ax_a.set_xticklabels([f'D{d}\namp' for d in det_order])
    ax_a.set_ylabel('amp_D')
    ax_a.set_title(f'(a) Per-detector amp_D '
                   f'(joint pedestal-anchor fit, ref=D{args.reference_detector})')
    ax_a.grid(alpha=0.3, axis='y')

    # (b) C(lambda) smooth curve + 3 shifted curves with pedestal, plus old C_c
    wl_min = min(np.nanmin(per_det_arrays[d]['WL']) for d in det_order)
    wl_max = max(np.nanmax(per_det_arrays[d]['WL']) for d in det_order)
    wl_grid = np.linspace(wl_min, wl_max, 400)
    C_grid = eval_C(wl_grid, c_poly)
    ax_b.plot(wl_grid, C_grid, '-', color='k', lw=1.3,
              label=r'$C(\lambda)$ shared smooth (P_D3 fixed = 0)')
    for d in det_order:
        color = DET_COLORS.get(d, 'tab:gray')
        if d == args.reference_detector:
            continue  # already drawn as black
        ax_b.plot(wl_grid, C_grid + P[d], '--', color=color, lw=1.1,
                  alpha=0.85,
                  label=f'$C(\\lambda)+P_{{D{d}}}$ = {P[d]:+.4g} MJy/sr')
    for d in det_order:
        color = DET_COLORS.get(d, 'tab:gray')
        wl = per_det_arrays[d]['WL']
        C_old = per_det_anchor_arrays[d]['C_old']
        order = np.argsort(wl)
        ax_b.plot(wl[order], C_old[order], 'o', mfc='none',
                  ms=5, mew=1.0, color=color, alpha=0.9,
                  label=f'D{d} per-channel C_old')
    ax_b.axhline(0.0, color='k', lw=0.5, alpha=0.4)
    for bx in DET_BOUNDARIES_UM:
        ax_b.axvline(bx, color='gray', lw=0.6, ls='--', alpha=0.5)
    ax_b.set_ylabel('C  (MJy/sr)')
    ax_b.set_title(rf'(b) Shared smooth $C(\lambda)$ (deg {args.C_degree}) + per-detector pedestal '
                   r'$P_D$  vs old per-channel $C_c$')
    ax_b.legend(loc='best', fontsize=7, ncol=2)
    ax_b.grid(alpha=0.3)

    # (c) C_eff per channel vs old C
    for d in det_order:
        color = DET_COLORS.get(d, 'tab:gray')
        wl = per_det_arrays[d]['WL']
        order = np.argsort(wl)
        C_eff = C_eff_per_det[d][order]
        C_old = per_det_anchor_arrays[d]['C_old'][order]
        ax_c.plot(wl[order], C_eff, '-s', ms=4, lw=1.0, color=color,
                  label=f'D{d} $C_{{\\rm eff}} = C(\\lambda_c) + P_{{D{d}}}$')
        ax_c.plot(wl[order], C_old, 'o', mfc='none', ms=5, lw=0,
                  color=color, alpha=0.9,
                  label=f'D{d} per-channel $C_{{\\rm old}}$')
    ax_c.axhline(0.0, color='k', lw=0.5, alpha=0.4)
    for bx in DET_BOUNDARIES_UM:
        ax_c.axvline(bx, color='gray', lw=0.6, ls='--', alpha=0.5)
    ax_c.set_ylabel('C  (MJy/sr)')
    ax_c.set_title('(c) Effective C per channel: pedestal model (markers) '
                   'vs per-channel anchor (open circles)')
    ax_c.legend(loc='best', fontsize=7, ncol=2)
    ax_c.grid(alpha=0.3)

    # (d) Per-channel residual std comparison
    for d in det_order:
        color = DET_COLORS.get(d, 'tab:gray')
        wl = per_det_arrays[d]['WL']
        order = np.argsort(wl)
        rs_ped = per_det_resid_std[d][order]
        ax_d.plot(wl[order], rs_ped, '-o', ms=4, lw=1.2, color=color,
                  label=f'D{d} pedestal model')
        # joint-amp overlay
        if d in joint_overlay:
            wlj = joint_overlay[d]['WL']
            rsj = joint_overlay[d]['resid_std']
            ord_j = np.argsort(wlj)
            ax_d.plot(wlj[ord_j], rsj[ord_j], '--', lw=1.0, color=color,
                      alpha=0.85,
                      label=f'D{d} joint-amp')
            # per-channel-fit estimate via misfit subtraction
            slope_old = per_det_anchor_arrays[d]['slope_old']
            # var_zp from FDC, ZP: use ZP only.
            zp = per_det_arrays[d]['ZP']
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                var_zp = np.nanvar(zp, axis=0)
            # amp from joint-amp file: need to read from joint-amp npz.
            # The joint-amp resid_std uses (1+amp_joint_amp_D); since we
            # don't store that here, we re-read it from the joint-amp file.
            amp_joint_amp = None
            try:
                with np.load(args.joint_data, allow_pickle=False) as z:
                    if f'amp_D{d}' in z.files:
                        amp_joint_amp = float(np.asarray(z[f'amp_D{d}']))
            except Exception:
                pass
            if amp_joint_amp is not None:
                rs_perch_est = sigma_perch_est(
                    joint_overlay[d]['resid_std'],
                    slope_old, amp_joint_amp, var_zp,
                )
                wl_jo = joint_overlay[d]['WL']
                # assume same channel ordering as our wl array
                ord_je = np.argsort(wl_jo)
                ax_d.plot(wl_jo[ord_je], rs_perch_est[ord_je],
                          ':', lw=1.0, color=color, alpha=0.85,
                          label=f'D{d} per-channel-fit est.')
    for bx in DET_BOUNDARIES_UM:
        ax_d.axvline(bx, color='gray', lw=0.6, ls='--', alpha=0.5)
    ax_d.set_xlabel(r'Channel mean wavelength ($\mu$m)')
    ax_d.set_ylabel('residual std  (MJy/sr)')
    ax_d.set_title('(d) Per-channel residual std: pedestal model (solid) vs '
                   'joint-amp (dashed) vs per-channel-fit estimate (dotted)')
    ax_d.legend(loc='best', fontsize=7, ncol=3)
    ax_d.grid(alpha=0.3)

    fig.suptitle(f'Prototype pedestal anchor: '
                 f'(1+amp_D)*ZP + C(lambda) + P_D    '
                 f'[deg={args.C_degree}, ref=D{args.reference_detector}, '
                 f'{sum(1 for _ in det_order)*1} amp + {args.C_degree+1} '
                 f'poly + {len(det_order)-1} P = '
                 f'{len(det_order)+args.C_degree+1+len(det_order)-1} '
                 f'free params]',
                 y=0.995, fontsize=11)
    plt.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f"Saved plot: {args.out}")


if __name__ == '__main__':
    main()
