"""Per-channel residual-std diagnostic with annotated sky/instrument features.

Loads the joint-fit per-detector arrays produced by
``diag_joint_amp_fit.py`` (the ``.npz`` it writes alongside its plot) and
draws the per-channel residual std vs wavelength, annotated with the
nearest-channel position of a handful of known sky / instrument features
(PAH 3.29, CO2 ice 4.27, dichroic edge at 2.42, etc.).

The per-channel residual std from the joint fit is effectively a
"non-zodi sky variance spectrum + instrumental noise floor"; bumps mark
wavelengths with real sky structure that zodipy does not predict.

The joint fit forces a single amplification (``1 + amp_D``) for all 34
channels of a detector, so individual channels are slightly mis-sloped
vs the true per-channel slope ``slope_old_c`` from the per-channel
anchor. The induced extra residual variance is roughly
``(slope_old_c - (1 + amp_D))**2 * var_zp_c``, where ``var_zp_c`` is the
per-frame variance of ``zodi_pred`` in channel c. Subtracting this gives
an "as if we had refit per channel" residual std estimate::

    sigma_perch_est = sqrt(max(0, sigma_joint^2
                               - (slope_old_c - (1+amp_D))^2 * var_zp_c))

If a feature bump survives in ``sigma_perch_est``, it is real (not a
joint-fit slope artifact).

Example::

    python diag_resid_features.py \\
        --joint-data figures/zodi_anchor/diag_joint_amp_fit.npz \\
        --run-dir /mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D3_6p2arcsec \\
                  /mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D4_6p2arcsec \\
                  /mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D5_6p2arcsec \\
        --out figures/zodi_anchor/diag_resid_features.png
"""
import argparse
import glob
import os
import re
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


DET_COLORS = {1: 'tab:purple', 2: 'tab:orange',
              3: 'tab:green', 4: 'tab:blue', 5: 'tab:red'}
# D1|D2 boundary is ~1.107 um (bands overlap slightly), D2|D3 ~1.65 um,
# D3|D4 = 2.42 um, D4|D5 = 3.81 um.
DET_BOUNDARIES_UM = (1.107, 1.65, 2.42, 3.81)

DEFAULT_FEATURES = [
    (0.8446, 'OI 8446 (airglow)'),
    (1.083, 'He I 1083 (airglow)'),
    (2.058, 'He I'),
    (2.166, 'Br gamma'),
    (2.30, 'CO band'),
    (2.42, 'dichroic'),
    (2.625, 'Br beta'),
    (3.05, 'H2O ice'),
    (3.29, 'PAH 3.29'),
    (3.40, 'PAH 3.4'),
    (4.052, 'Br alpha'),
    (4.27, 'CO2 ice'),
]


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--joint-data', required=True,
                   help='Path to the .npz produced by diag_joint_amp_fit.py.')
    p.add_argument('--run-dir', nargs='*', default=None,
                   help='Optional one-or-more run directories. If given, '
                        'per-channel var(zodi_pred) is read from '
                        '<run>/zodi_preds/zodi_pred_*.npz and used to overlay '
                        'the per-channel-fit-equivalent residual std estimate.')
    p.add_argument('--out', default='figures/zodi_anchor/diag_resid_features.png',
                   help='Output PNG path '
                        '(default: figures/zodi_anchor/diag_resid_features.png).')
    p.add_argument('--features', default='default',
                   help='Either "default" (the built-in feature list) or a '
                        'path to a whitespace-separated text file with one '
                        '"<wavelength_um> <name...>" entry per line.')
    return p.parse_args()


def load_features(spec):
    if spec == 'default' or spec is None:
        return list(DEFAULT_FEATURES)
    out = []
    with open(spec, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            parts = ln.split(None, 1)
            if len(parts) < 2:
                continue
            try:
                wl = float(parts[0])
            except ValueError:
                continue
            out.append((wl, parts[1].strip()))
    return out


def load_joint_payload(path):
    """Return dict: det -> {'WL', 'resid_std', 'amp', 'slope_old'}."""
    payload = {}
    with np.load(path, allow_pickle=False) as z:
        keys = list(z.files)
        if 'detectors' in keys:
            dets = [int(d) for d in np.asarray(z['detectors']).ravel()]
        else:
            dets = sorted({int(m.group(1))
                           for k in keys
                           for m in [re.match(r'WL_D(\d+)$', k)] if m})
        for d in dets:
            try:
                payload[d] = dict(
                    WL=np.asarray(z[f'WL_D{d}'], dtype=np.float64),
                    resid_std=np.asarray(z[f'resid_std_D{d}'], dtype=np.float64),
                    amp=float(np.asarray(z[f'amp_D{d}'])),
                    slope_old=np.asarray(z[f'slope_old_D{d}'], dtype=np.float64),
                )
            except KeyError as e:
                raise SystemExit(
                    f"--joint-data {path} missing key {e!r} for D{d}; "
                    "is this really a diag_joint_amp_fit.py output?")
    return payload


def parse_detector_from_filename(path):
    m = re.search(r'Detector(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def parse_channel_from_filename(path):
    m = re.search(r'_Ch(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def collect_var_zp_by_det(run_dirs):
    """Walk run_dirs/<zodi_preds>/zodi_pred_*.npz; return {det: {ch: var_zp}}."""
    out = {}
    for run in run_dirs:
        npz_dir = os.path.join(run, 'zodi_preds')
        files = sorted(glob.glob(os.path.join(npz_dir, 'zodi_pred_*.npz')))
        if not files:
            print(f"[warn] no zodi_pred_*.npz in {npz_dir}; skipping run")
            continue
        for path in files:
            det = parse_detector_from_filename(path)
            ch = parse_channel_from_filename(path)
            if det is None or ch is None:
                continue
            try:
                with np.load(path) as z:
                    zp = np.asarray(z['zodi_pred'], dtype=np.float64)
            except (KeyError, OSError) as e:
                print(f"[warn] cannot read {path}: {e}")
                continue
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', RuntimeWarning)
                v = float(np.nanvar(zp))
            out.setdefault(det, {})[ch] = v
    return out


def build_var_zp_arrays(payload, var_by_det):
    """For each det in payload, build a (N_channels,) var_zp array aligned to
    the joint-data channel index. If we have no var_by_det entry for a det,
    return all-NaN for that det."""
    out = {}
    for det, info in payload.items():
        n = info['WL'].shape[0]
        arr = np.full(n, np.nan, dtype=np.float64)
        ch_map = var_by_det.get(det, {})
        if ch_map:
            # The joint-fit npz drops channels (resp. WL) in channel order;
            # assume index i corresponds to channel i+1 (Ch1..ChN). The
            # joint-fit driver only filters channels by --cal-glob-pat /
            # parse_channel_from_filename then sorts ascending, so the
            # canonical 1..34 layout is the safe default.
            for i in range(n):
                ch = i + 1
                if ch in ch_map:
                    arr[i] = ch_map[ch]
        out[det] = arr
    return out


def compute_sigma_perch_est(sigma_joint, slope_old, amp, var_zp):
    """sqrt(max(0, sigma_joint^2 - (slope_old - (1+amp))^2 * var_zp))."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        delta = slope_old - (1.0 + amp)
        excess = delta * delta * var_zp
        residsq = sigma_joint * sigma_joint - excess
        residsq = np.where(np.isfinite(residsq), residsq, np.nan)
        residsq = np.where(residsq < 0.0, 0.0, residsq)
        return np.sqrt(residsq)


def find_nearest_channel(wl_pool):
    """Return a function that, given a target wl, returns (det, idx, wl_found,
    dist) for the global nearest entry across wl_pool: list of (det, WL)."""
    flat = []
    for det, wl in wl_pool:
        for idx, w in enumerate(wl):
            if np.isfinite(w):
                flat.append((det, idx, float(w)))
    arr_wl = np.asarray([t[2] for t in flat])

    def query(target):
        if arr_wl.size == 0:
            return None
        k = int(np.argmin(np.abs(arr_wl - target)))
        det, idx, w = flat[k]
        return det, idx, w, abs(w - target)

    return query


def baseline_for(payload, det, idx_feat, n_window=5):
    """Median of resid_std at the nearest n_window channels on the same
    detector, excluding the feature channel itself."""
    wl = payload[det]['WL']
    rs = payload[det]['resid_std']
    n = wl.shape[0]
    target = wl[idx_feat]
    if not np.isfinite(target):
        return np.nan
    dists = np.abs(wl - target)
    dists[idx_feat] = np.inf  # exclude self
    finite = np.isfinite(wl) & np.isfinite(rs)
    finite[idx_feat] = False
    if not finite.any():
        return np.nan
    dists = np.where(finite, dists, np.inf)
    order = np.argsort(dists)
    keep = order[:min(n_window, int(finite.sum()))]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return float(np.nanmedian(rs[keep]))


def verdict(enh):
    if not np.isfinite(enh):
        return 'n/a'
    if enh > 1.2:
        return 'clear'
    if enh >= 1.05:
        return 'weak'
    return 'none'


def main():
    args = parse_args()
    features = load_features(args.features)

    payload = load_joint_payload(args.joint_data)
    if not payload:
        raise SystemExit(f"no detectors found in {args.joint_data}")
    dets_sorted = sorted(payload)

    var_by_det = {}
    if args.run_dir:
        var_by_det = collect_var_zp_by_det(args.run_dir)
        if not var_by_det:
            print("[warn] --run-dir gave no var_zp data; sigma_perch_est "
                  "overlay will be omitted")
    var_zp_arrays = build_var_zp_arrays(payload, var_by_det)

    # sigma_perch_est per detector (NaN where var_zp is NaN)
    sigma_perch = {}
    has_any_perch = False
    for det in dets_sorted:
        info = payload[det]
        var_zp = var_zp_arrays.get(det, np.full(info['WL'].shape, np.nan))
        est = compute_sigma_perch_est(info['resid_std'], info['slope_old'],
                                      info['amp'], var_zp)
        if np.any(np.isfinite(est) & np.isfinite(var_zp)):
            has_any_perch = True
        sigma_perch[det] = est

    # ---- feature matching ----
    wl_pool = [(det, payload[det]['WL']) for det in dets_sorted]
    nearest = find_nearest_channel(wl_pool)

    feature_matches = []  # list of dict
    print()
    print(f"=== Feature enhancement (resid_std at feature / baseline median "
          f"of 5 nearest channels on same det) ===")
    hdr = (f"{'feature':>12}  {'wl_um':>6}  {'det':>3}  {'idx':>3}  "
           f"{'wl_match':>8}  {'dist_um':>7}  "
           f"{'sigma_joint':>11}  {'baseline':>9}  "
           f"{'sigma_perch':>11}  {'enh':>6}  verdict")
    print(hdr)
    print('-' * len(hdr))
    for wl_feat, name in features:
        q = nearest(wl_feat)
        if q is None:
            continue
        det, idx, wl_match, dist = q
        rs_at = float(payload[det]['resid_std'][idx])
        sp_at = float(sigma_perch[det][idx])
        baseline = baseline_for(payload, det, idx, n_window=5)
        if np.isfinite(baseline) and baseline > 0 and np.isfinite(rs_at):
            enh = rs_at / baseline
        else:
            enh = np.nan
        v = verdict(enh)
        print(f"{name:>12}  {wl_feat:>6.3f}  D{det:<2}  {idx:>3d}  "
              f"{wl_match:>8.4f}  {dist:>7.4f}  "
              f"{rs_at:>11.4g}  {baseline:>9.4g}  "
              f"{sp_at:>11.4g}  {enh:>6.3f}  {v}")
        feature_matches.append(dict(
            name=name, wl_feat=wl_feat,
            det=det, idx=idx, wl_match=wl_match, dist=dist,
            sigma_joint=rs_at, sigma_perch=sp_at,
            baseline=baseline, enhancement=enh, verdict=v,
        ))

    # ---- plot ----
    n_panels = 2 if has_any_perch else 1
    if n_panels == 2:
        fig, axes = plt.subplots(2, 1, figsize=(16, 7), sharex=True,
                                 gridspec_kw=dict(height_ratios=[3, 1]))
        ax_main, ax_diff = axes
    else:
        fig, ax_main = plt.subplots(1, 1, figsize=(16, 7))
        ax_diff = None

    for det in dets_sorted:
        info = payload[det]
        color = DET_COLORS.get(det, 'tab:gray')
        wl = info['WL']
        order = np.argsort(wl)
        wls = wl[order]
        rs = info['resid_std'][order]
        ax_main.plot(wls, rs, '-o', ms=4, lw=1.2, color=color,
                     label=f'D{det} sigma_joint')
        if has_any_perch:
            sp = sigma_perch[det][order]
            if np.any(np.isfinite(sp)):
                ax_main.plot(wls, sp, '--s', ms=3, lw=1.0, color=color,
                             alpha=0.85,
                             label=f'D{det} sigma_perch_est')
                if ax_diff is not None:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore', RuntimeWarning)
                        diff = rs - sp
                    ax_diff.plot(wls, diff, '-o', ms=3, lw=1.0, color=color,
                                 label=f'D{det} sigma_joint - sigma_perch_est')

    # Feature lines + labels on main panel.
    ymin, ymax = ax_main.get_ylim()
    label_y = ymax
    for wl_feat, name in features:
        ax_main.axvline(wl_feat, color='gray', lw=0.6, ls=':', alpha=0.55)
        ax_main.annotate(name, xy=(wl_feat, label_y),
                         xytext=(0, 3), textcoords='offset points',
                         ha='center', va='bottom', fontsize=7,
                         color='dimgray', rotation=60, clip_on=False)

    # Detector boundaries.
    for bx in DET_BOUNDARIES_UM:
        ax_main.axvline(bx, color='k', lw=0.8, ls='--', alpha=0.4)
        if ax_diff is not None:
            ax_diff.axvline(bx, color='k', lw=0.8, ls='--', alpha=0.4)

    ax_main.set_ylabel('residual std  (MJy/sr)')
    ax_main.set_title('Per-channel residual std with annotated features '
                      '(joint fit solid, per-channel-equivalent estimate dashed)')
    ax_main.legend(loc='best', fontsize=8, ncol=2)
    ax_main.grid(alpha=0.3)

    if ax_diff is not None:
        ax_diff.axhline(0.0, color='k', lw=0.5, alpha=0.4)
        ax_diff.set_xlabel('Channel mean wavelength (um)')
        ax_diff.set_ylabel('sigma_joint - sigma_perch_est')
        ax_diff.set_title('Excess residual std attributed to joint-fit slope '
                          'mismatch (small => bump is real)')
        ax_diff.legend(loc='best', fontsize=8, ncol=2)
        ax_diff.grid(alpha=0.3)
    else:
        ax_main.set_xlabel('Channel mean wavelength (um)')

    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=130, bbox_inches='tight')
    print(f"\nSaved plot: {args.out}")


if __name__ == '__main__':
    main()
