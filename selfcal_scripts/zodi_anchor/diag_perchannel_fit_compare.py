"""Per-channel diagnostic: per-channel anchor fit vs smooth-global-slope refit.

For a curated set of representative channels (default: D1-D5 x
Ch{1,17,34}), pull the per-frame (full_DC, zodi_pred, mjd) using the
same loader as ``refit_smooth_slope.py``, build a common inlier mask
with the same moving MJD-window sigma-clip, then evaluate two fit
variants on those *same* inliers:

* ``per-ch``  -- (slope_final, C_final) from ``anchor_D{N}.h5``
* ``smooth``  -- (polyval(coef, lambda_c), C_c) from
                 ``figures/zodi_anchor/refit_smooth_slope.npz``

Outputs a grid (rows = detectors, cols = channels) of (zp, fdc)
scatter + both fit lines, and prints a single comparison table to
stdout.

Read-only on calibration/anchor/data files; only emits the figure.
"""
import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# selfcal_scripts/zodi_anchor lives next to refit_smooth_slope.py,
# so a direct relative import works once we add this dir to sys.path.
_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from refit_smooth_slope import (  # type: ignore
    load_detector,
    per_channel_from_anchor,
)
from selfcal.zodi_anchor import moving_sigma_clip_mask


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

DEFAULT_CHANNELS = [
    (1, 1), (1, 17), (1, 34),
    (2, 1), (2, 17), (2, 34),
    (3, 1), (3, 17), (3, 34),
    (4, 1), (4, 17), (4, 34),
    (5, 1), (5, 17), (5, 34),
]

DET_COLORS = {1: 'tab:purple', 2: 'tab:orange',
              3: 'tab:green', 4: 'tab:blue', 5: 'tab:red'}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--run-dir', nargs='+', required=True,
                   help='SPHEREx run directories (one per detector). '
                        'Same convention as refit_smooth_slope.py.')
    p.add_argument('--smooth-npz', default=os.path.join(
                       'figures', 'zodi_anchor', 'refit_smooth_slope.npz'),
                   help='Path to the .npz produced by '
                        'refit_smooth_slope.py (default: '
                        'figures/zodi_anchor/refit_smooth_slope.npz).')
    p.add_argument('--channels', default=None,
                   help='Comma-separated D:Ch list (e.g. '
                        '"3:1,3:17,3:34,4:1,4:17,4:34,5:1,5:17,5:34"). '
                        'Default: Ch 1, 17, 34 for each of D1-D5 '
                        '(band-edge + mid-band channels).')
    p.add_argument('--cal-glob-pat', default='cal_*polyK1.h5',
                   help="Glob inside <run>/calibration "
                        "(default: 'cal_*polyK1.h5').")
    p.add_argument('--sigma', type=float, default=3.0,
                   help='Sigma-clip threshold for the moving clip '
                        '(default 3.0, matches anchor + smooth refit).')
    p.add_argument('--window-days', type=float, default=7.0,
                   help='Moving MJD window (default 7.0).')
    p.add_argument('--out-plot', default=os.path.join(
                       'figures', 'zodi_anchor',
                       'perchannel_fit_compare.png'),
                   help='Output PNG path.')
    return p.parse_args()


def _parse_channel_spec(spec):
    if spec is None:
        return list(DEFAULT_CHANNELS)
    out = []
    for tok in spec.split(','):
        tok = tok.strip()
        if not tok:
            continue
        d_s, c_s = tok.split(':')
        out.append((int(d_s), int(c_s)))
    return out


# ----------------------------------------------------------------------
# Inlier mask: identical to per-channel anchor's moving sigma-clip
# ----------------------------------------------------------------------

def common_inlier_mask(fdc, zp, mjds, slope_pc, C_pc,
                       window_days, sigma):
    """Build the per-channel inlier mask used by both variants.

    Mirrors ``_refresh_moving_clip`` in refit_smooth_slope.py: start
    from the finite mask, compute residuals against the per-channel
    fit (slope_pc * zp + C_pc), then apply the moving MJD-window
    sigma-clip. We use the per-channel fit's residual (not smooth)
    because that is the closer-to-zero residual; this is the same
    mask the anchor itself converged on.
    """
    init = np.isfinite(fdc) & np.isfinite(zp) & np.isfinite(mjds)
    if init.sum() < 10:
        return init.copy()
    resid = fdc - (slope_pc * zp + C_pc)
    resid_for_clip = np.where(init, resid, np.inf)
    keep = moving_sigma_clip_mask(mjds, resid_for_clip,
                                  window_days, sigma)
    out = init & keep
    if out.sum() < 10:
        out = init.copy()
    return out


# ----------------------------------------------------------------------
# Smooth-fit lookup
# ----------------------------------------------------------------------

class SmoothLookup:
    def __init__(self, npz_path):
        with np.load(npz_path, allow_pickle=False) as z:
            self.coef = z['smooth_coef_high_to_low'].astype(np.float64)
            self.WL = z['smooth_WL'].astype(np.float64)
            self.C = z['smooth_C'].astype(np.float64)
            self.slope = z['smooth_slope'].astype(np.float64)
            self.det_of_ch = z['smooth_det_of_ch'].astype(np.int32)
            self.ch_id = z['smooth_ch_id'].astype(np.int32)
            self.poly_degree = int(z['poly_degree'])

    def get(self, detector, channel):
        mask = (self.det_of_ch == detector) & (self.ch_id == channel)
        if not mask.any():
            return None
        i = int(np.where(mask)[0][0])
        return dict(
            slope=float(self.slope[i]),
            C=float(self.C[i]),
            WL=float(self.WL[i]),
        )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()
    channel_list = _parse_channel_spec(args.channels)

    os.makedirs(os.path.dirname(os.path.abspath(args.out_plot)) or '.',
                exist_ok=True)

    # ---- Load detectors --------------------------------------------------
    print("Loading detectors ...")
    detectors_by_id = {}
    perch_by_det = {}
    for run_dir in args.run_dir:
        print(f"  {run_dir}")
        det_data = load_detector(run_dir, args.cal_glob_pat)
        d = det_data['detector']
        detectors_by_id[d] = det_data
        perch_by_det[d] = per_channel_from_anchor(det_data['anchor_path'])

    smooth = SmoothLookup(args.smooth_npz)
    print(f"Loaded smooth fit from {args.smooth_npz}  "
          f"(K={smooth.poly_degree}, n_ch={len(smooth.WL)})")

    # ---- Per-channel comparison -----------------------------------------
    rows = []  # for the printed table

    # Group requested channels by detector (preserve user order within det)
    by_det_order = {}
    for d, c in channel_list:
        by_det_order.setdefault(d, []).append(c)
    det_ids_in_plot = sorted(by_det_order)

    # Build the grid figure. Rows = detectors, cols = channels per det.
    n_rows = len(det_ids_in_plot)
    n_cols = max(len(by_det_order[d]) for d in det_ids_in_plot)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.4 * n_cols, 3.8 * n_rows),
                             squeeze=False)

    for r, d in enumerate(det_ids_in_plot):
        det_data = detectors_by_id.get(d)
        if det_data is None:
            print(f"  [D{d}] not loaded (no --run-dir for it); "
                  f"skipping its row")
            for c in range(n_cols):
                axes[r, c].set_visible(False)
            continue
        pc = perch_by_det[d]
        # Lookup helpers: channel -> array index in det_data
        ch_to_idx = {int(ch): i for i, ch in enumerate(det_data['channels'])}
        pc_ch_to_idx = {int(ch): i for i, ch in enumerate(pc['channels'])}

        cols_for_d = by_det_order[d]
        for c_col, ch in enumerate(cols_for_d):
            ax = axes[r, c_col]
            color = DET_COLORS.get(d, 'tab:purple')

            if ch not in ch_to_idx or ch not in pc_ch_to_idx:
                ax.text(0.5, 0.5, f"D{d} Ch{ch}\nMISSING",
                        ha='center', va='center', transform=ax.transAxes)
                ax.set_xticks([])
                ax.set_yticks([])
                continue
            i_data = ch_to_idx[ch]
            i_pc = pc_ch_to_idx[ch]

            fdc = det_data['FDC'][i_data]
            zp = det_data['ZP'][i_data]
            mjds = det_data['MJD'][i_data]
            WL_ch = float(det_data['WL'][i_data])

            slope_pc = float(pc['slope'][i_pc])
            C_pc = float(pc['C'][i_pc])

            sm = smooth.get(d, ch)
            if sm is None:
                ax.text(0.5, 0.5, f"D{d} Ch{ch}\nNO SMOOTH",
                        ha='center', va='center', transform=ax.transAxes)
                continue
            slope_sm = sm['slope']
            C_sm = sm['C']

            # Common inlier mask (anchor-style moving clip on per-ch resid)
            mask = common_inlier_mask(fdc, zp, mjds,
                                      slope_pc, C_pc,
                                      args.window_days, args.sigma)
            n_in = int(mask.sum())

            if n_in < 5:
                ax.text(0.5, 0.5, f"D{d} Ch{ch}\nn_inlier={n_in}",
                        ha='center', va='center', transform=ax.transAxes)
                continue

            zp_in = zp[mask]
            fdc_in = fdc[mask]

            # Predictions on the same inlier set
            pred_pc = slope_pc * zp_in + C_pc
            pred_sm = slope_sm * zp_in + C_sm
            r_pc = fdc_in - pred_pc
            r_sm = fdc_in - pred_sm
            std_pc = float(np.std(r_pc))
            std_sm = float(np.std(r_sm))

            # Pearson r of (fdc, slope * zp): the slope-only modelled DC
            # component, NOT the (slope*zp + C) model line (adding a
            # constant leaves r unchanged anyway, so this is the same
            # number; computed against slope*zp to make the compared
            # quantity explicit).
            def _pearson(a, b):
                a = np.asarray(a, dtype=np.float64)
                b = np.asarray(b, dtype=np.float64)
                if a.std() == 0 or b.std() == 0:
                    return float('nan')
                return float(np.corrcoef(a, b)[0, 1])

            r_per_ch = _pearson(fdc_in, slope_pc * zp_in)
            r_smooth = _pearson(fdc_in, slope_sm * zp_in)

            rows.append(dict(
                det=d, ch=ch, wl=WL_ch,
                slope_pc=slope_pc, slope_sm=slope_sm,
                C_pc=C_pc, C_sm=C_sm,
                r_pc=r_per_ch, r_sm=r_smooth,
                std_pc=std_pc, std_sm=std_sm,
                n_in=n_in,
            ))

            # --- Plot ---
            ax.scatter(zp_in, fdc_in, s=8, alpha=0.3,
                       color=color, edgecolors='none', zorder=1)
            # Optional: faint outliers (clipped) in light grey
            out_mask = (~mask) & np.isfinite(fdc) & np.isfinite(zp)
            if out_mask.any():
                ax.scatter(zp[out_mask], fdc[out_mask],
                           s=6, alpha=0.15, color='lightgray',
                           edgecolors='none', zorder=0,
                           label=f'clipped ({int(out_mask.sum())})')

            x_lo = float(np.nanmin(zp_in))
            x_hi = float(np.nanmax(zp_in))
            x_ln = np.linspace(x_lo, x_hi, 200)
            ax.plot(x_ln, slope_pc * x_ln + C_pc,
                    color=color, lw=1.6, ls='-',
                    label=f'per-ch (s={slope_pc:.4f}, C={C_pc:+.4g})')
            ax.plot(x_ln, slope_sm * x_ln + C_sm,
                    color='black', lw=1.4, ls='--',
                    label=f'smooth (s={slope_sm:.4f}, C={C_sm:+.4g})')

            # in-axis text
            txt = (f"r_perch  = {r_per_ch:.6f}\n"
                   f"r_smooth = {r_smooth:.6f}\n"
                   f"std_pc   = {std_pc:.4g}\n"
                   f"std_sm   = {std_sm:.4g}\n"
                   f"n_in     = {n_in}")
            ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                    va='top', ha='left', fontsize=7,
                    family='monospace',
                    bbox=dict(facecolor='white', alpha=0.85, lw=0.4))

            ax.set_title(f'D{d} Ch{ch}   lambda={WL_ch:.3f} um',
                         fontsize=9)
            ax.set_xlabel('zodi_pred  (MJy/sr)', fontsize=8)
            ax.set_ylabel('full_DC  (MJy/sr)', fontsize=8)
            ax.grid(alpha=0.3)
            ax.legend(loc='lower right', fontsize=6.5)

        # Hide unused columns in this row, if any
        for c_col in range(len(cols_for_d), n_cols):
            axes[r, c_col].set_visible(False)

    fig.suptitle(
        'Per-channel anchor vs smooth-poly slope refit  '
        '(same inlier mask, anchor moving sigma-clip)',
        y=0.999, fontsize=11)
    plt.tight_layout(rect=(0, 0, 1, 0.985))
    plt.savefig(args.out_plot, dpi=130, bbox_inches='tight')
    print(f"Saved plot: {args.out_plot}")

    # ---- Print table ----------------------------------------------------
    hdr = (f"{'Det':>3s} {'Ch':>3s} {'WL[um]':>8s}  "
           f"{'slope_pc':>10s} {'slope_sm':>10s}  "
           f"{'C_pc':>10s} {'C_sm':>10s}  "
           f"{'r_pc':>10s} {'r_sm':>10s}  "
           f"{'std_pc':>10s} {'std_sm':>10s}  "
           f"{'n_in':>6s}")
    sep = '-' * len(hdr)
    print()
    print(sep)
    print(hdr)
    print(sep)
    for row in rows:
        print(f"{row['det']:>3d} {row['ch']:>3d} {row['wl']:>8.3f}  "
              f"{row['slope_pc']:>10.6f} {row['slope_sm']:>10.6f}  "
              f"{row['C_pc']:>+10.5f} {row['C_sm']:>+10.5f}  "
              f"{row['r_pc']:>10.6f} {row['r_sm']:>10.6f}  "
              f"{row['std_pc']:>10.5g} {row['std_sm']:>10.5g}  "
              f"{row['n_in']:>6d}")
    print(sep)


if __name__ == '__main__':
    main()
