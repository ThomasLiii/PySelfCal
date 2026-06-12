"""Joint zodi-spectrum diagnostic across ALL five SPHEREx detectors.

Stacked 5-panel comparison (sharex on channel mean wavelength in um):

  (a) mean(full_DC), mean(zodi_pred), slope*mean(zodi_pred) per channel,
      color-coded by detector. Pulled from the anchor file.
  (b) C  (anchor non-zodi DC) vs lambda, color-coded by detector.
  (c) slope vs lambda, color-coded by detector; horizontal reference at 1.0.
  (d) Pearson r vs lambda, color-coded by detector; reference at 1.0.
  (e) per-channel inlier residual std (mMJy/sr) vs lambda. Computed by
      re-running ``fit_with_clip`` on the cached ``cal`` + ``zodi_preds``
      (same window/sigma/iters the anchor used), then::

        full_DC = compute_full_dc(frame_scalar, offsets_m0, cov_m0)
        slope_fit, C_fit, _, inlier = fit_with_clip(zp, full_DC, mjds)
        resid_std = std(full_DC[inlier] - (slope_fit * zp[inlier] + C_fit))

      Uses the re-fitted slope/C from ``fit_with_clip`` (which reproduces
      the anchor's raw slope/intercept exactly, including for D1/D2 where
      ``slope_final``/``C_final`` were later overwritten by smoothing).
      This guarantees the residual std is the inlier scatter the solver
      actually anchored against.

Color map:   D1=tab:purple, D2=tab:orange, D3=tab:green, D4=tab:blue, D5=tab:red.
Detector boundaries (vertical light gray, solid): 1.10, 1.65, 2.42, 3.81 um.
Spectral features (vertical dashed, labeled along the top of panel (e)):
  OI 8446 = 0.8446, He I 1083 = 1.083, Pa gamma = 1.094, Pa beta = 1.282,
  He I 2.058, Br gamma = 2.166, dichroic = 2.42, Br beta = 2.625,
  H2O ice = 3.05, PAH = 3.29 + 3.40, Br alpha = 4.052, CO2 ice = 4.27.

Outputs (overwritten):
  figures/zodi_anchor/zodi_spectrum_all_detectors.png
  figures/zodi_anchor/zodi_spectrum_all_detectors.npz

Read-only on cal / anchor / zodi-pred files; no commits; no mosaic render.

Example::

    python selfcal_scripts/zodi_anchor/diag_joint_zodi_spectrum.py \\
        --run-dir /mnt/md124/.../SPHEREx_NEP_2026W17_D1_6p2arcsec \\
                  /mnt/md124/.../SPHEREx_NEP_2026W17_D2_6p2arcsec \\
                  /mnt/md124/.../SPHEREx_NEP_2026W17_D3_6p2arcsec \\
                  /mnt/md124/.../SPHEREx_NEP_2026W17_D4_6p2arcsec \\
                  /mnt/md124/.../SPHEREx_NEP_2026W17_D5_6p2arcsec
"""
import argparse
import glob
import os
import re
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from SelfCal.ZodiAnchor import (
    compute_full_dc,
    fit_with_clip,
    load_anchor,
)


# ---------------------------------------------------------------------
# Conventions
# ---------------------------------------------------------------------

DET_COLORS = {
    1: 'tab:purple',
    2: 'tab:orange',
    3: 'tab:green',
    4: 'tab:blue',
    5: 'tab:red',
}

# Detector-band boundaries in microns (sharp dichroic / band edges).
DET_BOUNDARIES_UM = (1.10, 1.65, 2.42, 3.81)

# Spectral features highlighted on panel (e). (label, wavelength_um).
FEATURE_LINES = [
    ('OI 8446',   0.8446),
    ('He I 1083', 1.083),
    ('Pa gamma',  1.094),
    ('Pa beta',   1.282),
    ('He I 2.06', 2.058),
    ('Br gamma',  2.166),
    ('dichroic',  2.42),
    ('Br beta',   2.625),
    ('H2O ice',   3.05),
    ('PAH 3.29',  3.29),
    ('PAH 3.40',  3.40),
    ('Br alpha',  4.052),
    ('CO2 ice',   4.27),
]


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--anchor', nargs='+',
                     help='One or more anchor_D{N}.h5 paths (any order).')
    src.add_argument('--run-dir', nargs='+',
                     help='One or more SPHEREx run dirs; each must contain '
                          'zodi_anchor/anchor_D*.h5, calibration/, and '
                          'zodi_preds/.')
    p.add_argument('--cal-glob-pat', default='cal_*polyK1.h5',
                   help="Glob inside <run>/calibration (default: "
                        "'cal_*polyK1.h5').")
    p.add_argument('--out-plot', default=None,
                   help='Output PNG. Default: '
                        'figures/zodi_anchor/zodi_spectrum_all_detectors.png')
    p.add_argument('--out-data', default=None,
                   help='Output .npz. Default: same stem as --out-plot.')
    p.add_argument('--max-ch', type=int, default=34,
                   help='Only plot channels <= this (default 34).')
    p.add_argument('--no-resid-std', action='store_true',
                   help='Skip panel (e) recomputation (anchor-only run).')
    return p.parse_args()


# ---------------------------------------------------------------------
# Anchor + cal resolution
# ---------------------------------------------------------------------

def _detector_of_cal(path):
    m = re.search(r'Detector(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def _channel_of_cal(path):
    m = re.search(r'_Ch(\d+)_', os.path.basename(path))
    return int(m.group(1)) if m else None


def _matching_npz(cal_path, npz_dir):
    base = os.path.basename(cal_path)
    tag = base[len('cal_'):-len('.h5')]
    return os.path.join(npz_dir, f'zodi_pred_{tag}.npz')


def resolve_anchor_paths(args):
    """Return list of (anchor_path, run_dir_or_None) tuples in arbitrary
    order. run_dir is None when --anchor was used directly (then we infer it
    as the parent of the anchor's parent dir, since the conventional layout
    is <run>/zodi_anchor/anchor_D*.h5)."""
    out = []
    if args.anchor:
        for ap in args.anchor:
            ap = os.path.abspath(ap)
            run_dir = os.path.dirname(os.path.dirname(ap))
            out.append((ap, run_dir))
        return out
    for rd in args.run_dir:
        rd = os.path.abspath(rd)
        cand = sorted(glob.glob(os.path.join(rd, 'zodi_anchor',
                                             'anchor_D*.h5')))
        if not cand:
            raise SystemExit(f"no anchor_D*.h5 in {rd}/zodi_anchor/")
        for ap in cand:
            out.append((ap, rd))
    return out


# ---------------------------------------------------------------------
# Per-channel residual std (re-fit, matches anchor exactly for raw method)
# ---------------------------------------------------------------------

def _anchor_clip_defaults(anchor_path):
    with h5py.File(anchor_path, 'r') as f:
        wd = float(f.attrs.get('clip_window_days', 7.0))
        sg = float(f.attrs.get('clip_sigma', 3.0))
        it = int(f.attrs.get('clip_iters', 2))
    return wd, sg, it


def compute_resid_std_for_detector(anchor, run_dir, cal_glob_pat,
                                   max_ch=34):
    """Recompute the inlier residual std (MJy/sr) per channel for one
    detector by re-running ``fit_with_clip`` on the cached cal + zodi-pred.

    Returns dict {ch: resid_std_MJy}, missing channels mapped to NaN. Uses
    the anchor's stored clip window/sigma/iters so the inlier set matches
    the original fit; the slope/C used in the residual are the *re-fitted*
    raw (free) values (which equal anchor.slope/intercept for raw method).
    """
    det = anchor.detector
    cal_dir = os.path.join(run_dir, 'calibration')
    npz_dir = os.path.join(run_dir, 'zodi_preds')

    by_ch = {}
    for c in sorted(glob.glob(os.path.join(cal_dir, cal_glob_pat))):
        if _detector_of_cal(c) != det:
            continue
        ch = _channel_of_cal(c)
        if ch is not None and ch <= max_ch:
            by_ch[ch] = c

    wd, sg, it = _anchor_clip_defaults(anchor.path)

    resid_std = {}
    n_inl = {}
    for ch in sorted(by_ch):
        cal_path = by_ch[ch]
        npz_path = _matching_npz(cal_path, npz_dir)
        if not os.path.exists(npz_path):
            print(f"  [D{det} Ch{ch}] npz missing: {npz_path}; skipping",
                  file=sys.stderr)
            resid_std[ch] = float('nan')
            n_inl[ch] = 0
            continue
        try:
            with h5py.File(cal_path, 'r') as f:
                fs = f['frame_scalar'][:].astype(np.float64)
                om0 = f['offsets/map_0'][:].astype(np.float64)
                cm0 = f['offset_coverage/map_0'][:].astype(np.float64)
            with np.load(npz_path, allow_pickle=False) as z:
                zp = z['zodi_pred'].astype(np.float64).ravel()
                mjds = z['mjds'].astype(np.float64).ravel()
        except Exception as exc:
            print(f"  [D{det} Ch{ch}] read failed: {exc}", file=sys.stderr)
            resid_std[ch] = float('nan')
            n_inl[ch] = 0
            continue

        fdc = compute_full_dc(fs, om0, cm0)
        if zp.shape != fdc.shape or mjds.shape != fdc.shape:
            print(f"  [D{det} Ch{ch}] shape mismatch fdc={fdc.shape} "
                  f"zp={zp.shape} mjds={mjds.shape}", file=sys.stderr)
            resid_std[ch] = float('nan')
            n_inl[ch] = 0
            continue

        slope_f, C_f, _r, inlier = fit_with_clip(
            zp, fdc, mjds,
            window_days=wd, sigma=sg, iters=it)
        if int(inlier.sum()) >= 2 and np.isfinite(slope_f) and np.isfinite(C_f):
            r = fdc[inlier] - (slope_f * zp[inlier] + C_f)
            resid_std[ch] = float(np.std(r))
        else:
            resid_std[ch] = float('nan')
        n_inl[ch] = int(inlier.sum())
    return resid_std, n_inl


# ---------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------

def _draw_det_boundaries(ax):
    for w in DET_BOUNDARIES_UM:
        ax.axvline(w, color='lightgray', lw=0.8, alpha=0.7, zorder=0)


def _draw_feature_lines(ax, label=False, label_y=None):
    for name, w in FEATURE_LINES:
        ax.axvline(w, color='gray', lw=0.6, ls='--', alpha=0.55, zorder=0)
        if label and label_y is not None:
            ax.annotate(name, xy=(w, label_y),
                        xytext=(0, 2), textcoords='offset points',
                        ha='center', va='bottom',
                        rotation=70, fontsize=6, color='gray', alpha=0.9)


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    args = parse_args()
    out_plot = args.out_plot or os.path.join(
        'figures', 'zodi_anchor', 'zodi_spectrum_all_detectors.png')
    out_data = args.out_data or (os.path.splitext(out_plot)[0] + '.npz')
    os.makedirs(os.path.dirname(os.path.abspath(out_plot)) or '.',
                exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_data)) or '.',
                exist_ok=True)

    # ----- Resolve + load every anchor -----
    anchor_specs = resolve_anchor_paths(args)
    print(f"Loading {len(anchor_specs)} anchor file(s) ...")
    detectors = []
    for ap, rd in anchor_specs:
        a = load_anchor(ap)
        detectors.append(dict(anchor=a, anchor_path=ap, run_dir=rd))
        print(f"  D{a.detector}: {ap}  (method={a.anchor_method}, "
              f"{len(a.channels)} channels)")

    detectors.sort(key=lambda d: d['anchor'].detector)

    # ----- Pull per-channel arrays from each anchor file -----
    per_det = []
    for d in detectors:
        a = d['anchor']
        chs = sorted(c for c in a.channels if c <= args.max_ch)
        rows = [a.channels[c] for c in chs]
        wl = np.array([float(r['wavelength_um']) for r in rows])
        mean_fs = np.array([float(r['mean_full_dc']) for r in rows])
        mean_zp = np.array([float(r['mean_pred']) for r in rows])
        slope = np.array([float(r['slope_final']) for r in rows])
        C = np.array([float(r['C_final']) for r in rows])
        rval = np.array([float(r['pearson_r']) for r in rows])
        fit_zodi = slope * mean_zp
        per_det.append(dict(
            detector=a.detector,
            anchor=a,
            anchor_path=d['anchor_path'],
            run_dir=d['run_dir'],
            chs=np.asarray(chs, dtype=np.int32),
            wl=wl,
            mean_full_dc=mean_fs,
            mean_pred=mean_zp,
            slope=slope,
            C=C,
            pearson_r=rval,
            fit_zodi=fit_zodi,
        ))

    # ----- Panel (e): residual std via re-fit, per detector -----
    if not args.no_resid_std:
        print("\nRecomputing per-channel inlier residual std ...")
        for pd_ in per_det:
            if pd_['run_dir'] is None or not os.path.isdir(
                    os.path.join(pd_['run_dir'], 'calibration')):
                print(f"  D{pd_['detector']}: cal dir missing; skip",
                      file=sys.stderr)
                pd_['resid_std_mMJy'] = np.full(pd_['chs'].size, np.nan)
                pd_['n_inliers'] = np.zeros(pd_['chs'].size, dtype=np.int32)
                continue
            rs, ni = compute_resid_std_for_detector(
                pd_['anchor'], pd_['run_dir'], args.cal_glob_pat,
                max_ch=args.max_ch)
            pd_['resid_std_mMJy'] = np.array(
                [rs.get(int(c), float('nan')) * 1e3 for c in pd_['chs']])
            pd_['n_inliers'] = np.array(
                [ni.get(int(c), 0) for c in pd_['chs']], dtype=np.int32)
            ok = np.isfinite(pd_['resid_std_mMJy'])
            if ok.any():
                med = float(np.median(pd_['resid_std_mMJy'][ok]))
            else:
                med = float('nan')
            print(f"  D{pd_['detector']}: median resid_std = "
                  f"{med:.2f} mMJy/sr  ({int(ok.sum())} valid channels)")
    else:
        for pd_ in per_det:
            pd_['resid_std_mMJy'] = np.full(pd_['chs'].size, np.nan)
            pd_['n_inliers'] = np.zeros(pd_['chs'].size, dtype=np.int32)

    # ----- Figure -----
    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

    # (a) mean(full_DC) [solid], mean(zp) [dashed], slope*mean(zp) [dotted]
    ax = axes[0]
    for pd_ in per_det:
        det = pd_['detector']
        c = DET_COLORS.get(det, 'k')
        ax.plot(pd_['wl'], pd_['mean_full_dc'], '-o', ms=4, lw=1.0, c=c,
                label=f'D{det} mean(full_DC)')
        ax.plot(pd_['wl'], pd_['mean_pred'], '--s', ms=3, lw=0.9, c=c,
                alpha=0.7, label=f'D{det} mean(zp)')
        ax.plot(pd_['wl'], pd_['fit_zodi'], ':^', ms=3, lw=0.9, c=c,
                alpha=0.7, label=f'D{det} slope*mean(zp)')
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax.set_ylabel('MJy/sr')
    ax.set_title('(a) Per-channel mean DC: solid=mean(full_DC), '
                 'dashed=mean(zp), dotted=slope*mean(zp)')
    ax.legend(loc='best', fontsize=6, ncol=5)
    ax.grid(alpha=0.3)
    _draw_det_boundaries(ax)
    _draw_feature_lines(ax)

    # (b) C
    ax = axes[1]
    ax.axhline(0.0, color='k', lw=0.5, alpha=0.5)
    for pd_ in per_det:
        det = pd_['detector']
        c = DET_COLORS.get(det, 'k')
        ax.plot(pd_['wl'], pd_['C'], '-^', ms=4, lw=1.0, c=c,
                label=f'D{det}')
    ax.set_ylabel('C  (MJy/sr)')
    ax.set_title('(b) Anchor constant C  (non-zodi uniform DC added to mosaic)')
    ax.legend(loc='best', fontsize=8, ncol=5)
    ax.grid(alpha=0.3)
    _draw_det_boundaries(ax)
    _draw_feature_lines(ax)

    # (c) slope
    ax = axes[2]
    ax.axhline(1.0, color='k', lw=0.7, alpha=0.5)
    for pd_ in per_det:
        det = pd_['detector']
        c = DET_COLORS.get(det, 'k')
        ax.plot(pd_['wl'], pd_['slope'], '-o', ms=4, lw=1.0, c=c,
                label=f'D{det}')
    ax.set_ylabel('slope')
    ax.set_title('(c) Fitted slope per channel  (=1 if zodipy captures '
                 'temporal shape)')
    ax.legend(loc='best', fontsize=8, ncol=5)
    ax.grid(alpha=0.3)
    _draw_det_boundaries(ax)
    _draw_feature_lines(ax)

    # (d) Pearson r
    ax = axes[3]
    ax.axhline(1.0, color='k', lw=0.5, alpha=0.4)
    ax.axhline(0.0, color='k', lw=0.5, alpha=0.4)
    for pd_ in per_det:
        det = pd_['detector']
        c = DET_COLORS.get(det, 'k')
        ax.plot(pd_['wl'], pd_['pearson_r'], '-o', ms=4, lw=1.0, c=c,
                label=f'D{det}')
    ax.set_ylabel('Pearson r')
    ax.set_title('(d) Per-frame correlation of full_DC vs zodi_pred')
    ax.set_ylim(-0.3, 1.05)
    ax.legend(loc='lower left', fontsize=8, ncol=5)
    ax.grid(alpha=0.3)
    _draw_det_boundaries(ax)
    _draw_feature_lines(ax)

    # (e) residual std (mMJy/sr), with feature labels along the top
    ax = axes[4]
    for pd_ in per_det:
        det = pd_['detector']
        c = DET_COLORS.get(det, 'k')
        ax.plot(pd_['wl'], pd_['resid_std_mMJy'], '-o', ms=4, lw=1.0, c=c,
                label=f'D{det}')
    ax.set_ylabel('resid std  (mMJy/sr)')
    ax.set_title('(e) Inlier residual std per channel  '
                 '(refit with anchor clip params)')
    ax.set_xlabel('Channel mean wavelength (um)')
    ax.legend(loc='best', fontsize=8, ncol=5)
    ax.grid(alpha=0.3)
    _draw_det_boundaries(ax)
    # Label features along the top of panel (e). We need a label y-coord;
    # use a value slightly above the current ymax after plotting.
    finite_resid = np.concatenate(
        [pd_['resid_std_mMJy'][np.isfinite(pd_['resid_std_mMJy'])]
         for pd_ in per_det] + [np.array([0.0])])
    if finite_resid.size > 1:
        ymax = float(np.nanmax(finite_resid)) * 1.05 + 1e-9
    else:
        ymax = 1.0
    ymin = ax.get_ylim()[0]
    ax.set_ylim(ymin, ymax * 1.18)  # headroom for labels
    _draw_feature_lines(ax, label=True, label_y=ymax)

    plt.tight_layout()
    plt.savefig(out_plot, dpi=130)
    plt.close(fig)
    print(f"\nSaved {out_plot}")

    # ----- Save .npz companion -----
    npz_kw = {}
    for pd_ in per_det:
        d = pd_['detector']
        npz_kw[f'D{d}_channels']      = pd_['chs']
        npz_kw[f'D{d}_wavelength_um'] = pd_['wl']
        npz_kw[f'D{d}_mean_full_dc']  = pd_['mean_full_dc']
        npz_kw[f'D{d}_mean_pred']     = pd_['mean_pred']
        npz_kw[f'D{d}_slope']         = pd_['slope']
        npz_kw[f'D{d}_C']             = pd_['C']
        npz_kw[f'D{d}_pearson_r']     = pd_['pearson_r']
        npz_kw[f'D{d}_resid_std_mMJy'] = pd_['resid_std_mMJy']
        npz_kw[f'D{d}_n_inliers']      = pd_['n_inliers']
    npz_kw['detectors'] = np.asarray([pd_['detector'] for pd_ in per_det],
                                      dtype=np.int32)
    npz_kw['det_boundaries_um'] = np.asarray(DET_BOUNDARIES_UM, dtype=float)
    npz_kw['feature_names'] = np.asarray([n for n, _ in FEATURE_LINES])
    npz_kw['feature_wavelengths_um'] = np.asarray(
        [w for _, w in FEATURE_LINES], dtype=float)
    np.savez(out_data, **npz_kw)
    print(f"Saved {out_data}")

    # ----- Stdout summary -----
    print("\nPer-detector summary:")
    print(f"{'Det':>4} {'NCh':>4}  {'med_slope':>10} {'med_C(MJy)':>12} "
          f"{'med_r':>8} {'med_resid(mMJy)':>16}")
    for pd_ in per_det:
        ok = pd_['chs'] > 0
        med_slope = float(np.nanmedian(pd_['slope']))
        med_C     = float(np.nanmedian(pd_['C']))
        med_r     = float(np.nanmedian(pd_['pearson_r']))
        rs_finite = pd_['resid_std_mMJy'][np.isfinite(pd_['resid_std_mMJy'])]
        med_rs    = float(np.median(rs_finite)) if rs_finite.size else float('nan')
        print(f"  D{pd_['detector']} {pd_['chs'].size:>4d}  "
              f"{med_slope:>10.4f} {med_C:>12.4f} "
              f"{med_r:>8.4f} {med_rs:>16.3f}")


if __name__ == '__main__':
    main()
