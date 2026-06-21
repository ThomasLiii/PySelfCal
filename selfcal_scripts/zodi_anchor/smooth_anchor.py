"""Phase-1 slope smoothing of contaminated anchor channels.

Some channels' per-channel anchor fit is unreliable because a bright,
time-variable non-zodi signal (e.g. He I 1083 nm / OI 8446 nm airglow on
D1) overwhelms the zodi correlation: low Pearson r, wild slope, and a C
blown far off the smooth trend. Leaving those C values in place puts a
large wrong uniform offset into the affected channels' mosaics.

This script fits a Pearson-r-weighted smoothing spline over slope(λ) and
C(λ) across a detector's channels (contaminated channels carry ~zero
weight, so the curve follows the clean trend), then REPLACES only the
flagged (r < --r-threshold) channels' slope_final / C_final with the
spline value. Clean channels keep their raw fit. The result is written
back into the SAME anchor file (raw slope/intercept stay untouched;
consumers read slope_final / C_final).

    # inspect first (writes a before/after plot, no file changes):
    python smooth_anchor.py --run-dir /mnt/.../D1_... --dry-run --plot

    # apply in-place:
    python smooth_anchor.py --run-dir /mnt/.../D1_...

See selfcal.zodi_anchor.rweighted_slope_smooth for the core math and
workspace/zodi_anchor_refactor/refactor.md for context.
"""
import argparse
import glob
import os

import numpy as np

from selfcal.zodi_anchor import smooth_anchor_file


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--anchor', help='Path to anchor_D{N}.h5.')
    src.add_argument('--run-dir',
                     help='Run dir; uses <run>/zodi_anchor/anchor_D{N}.h5 '
                          '(needs --detector if >1 present).')
    p.add_argument('--detector', type=int, default=None)
    p.add_argument('--r-threshold', type=float, default=0.9,
                   help='Channels with Pearson r below this are smoothed '
                        '(default 0.9; lower to 0.5 for only hard blowouts).')
    p.add_argument('--s-factor', type=float, default=1.0,
                   help='Spline smoothing strength; ~1 targets reduced-chi^2 '
                        '1, larger = smoother (default 1.0).')
    p.add_argument('--spline-k', type=int, default=3,
                   help='Spline degree (default 3).')
    p.add_argument('--dry-run', action='store_true',
                   help='Report + plot but do not modify the anchor file.')
    p.add_argument('--plot', nargs='?', const='auto', default=None,
                   help='Write a before/after PNG. With no value, saves '
                        'next to the anchor file as anchor_D{N}_smooth.png.')
    return p.parse_args()


def resolve_anchor_path(args):
    if args.anchor:
        return args.anchor
    cand = sorted(glob.glob(os.path.join(args.run_dir, 'zodi_anchor',
                                         'anchor_D*.h5')))
    if args.detector:
        want = os.path.join(args.run_dir, 'zodi_anchor',
                            f'anchor_D{args.detector}.h5')
        if os.path.exists(want):
            return want
    if len(cand) == 1:
        return cand[0]
    raise SystemExit(f"ambiguous/missing anchor file in {args.run_dir}: "
                     f"{cand}; pass --anchor or --detector.")


def make_plot(out_png, det, wl, slope_raw, C_raw, r, res):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    contam = res['contaminated']
    order = np.argsort(wl)
    wls = wl[order]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    def panel(ax, raw, final, label, curve=None, hline=None):
        if curve is not None:
            ax.plot(wls, curve[order], '-', c='tab:green', lw=1.2,
                    label='r-weighted spline', zorder=1)
        ax.scatter(wl[~contam], raw[~contam], s=18, c='tab:blue',
                   label='clean (kept raw)', zorder=3)
        ax.scatter(wl[contam], raw[contam], s=30, c='tab:red', marker='x',
                   label='contaminated (raw)', zorder=3)
        ax.scatter(wl[contam], final[contam], s=40, facecolors='none',
                   edgecolors='tab:red', linewidths=1.5,
                   label='contaminated (smoothed)', zorder=4)
        if hline is not None:
            ax.axhline(hline, color='k', lw=0.5, alpha=0.4)
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc='best')

    panel(axes[0], slope_raw, res['slope_final'], 'slope',
          curve=res['slope_curve'], hline=1.0)
    axes[0].set_title(f'(a) D{det} slope: raw vs r-weighted spline '
                      f'(smoothed = open red)')
    # C is NOT smoothed: recomputed C = mean_full_dc - slope_final*mean_pred,
    # which keeps the non-zodi/airglow content. No spline curve here.
    panel(axes[1], C_raw, res['C_final'], 'C (MJy/sr)', hline=0.0)
    axes[1].set_title('(b) C: raw vs recomputed from smoothed slope '
                      '(keeps non-zodi signal; not smoothed)')
    ax = axes[2]
    ax.scatter(wl[~contam], r[~contam], s=18, c='tab:blue', label='clean')
    ax.scatter(wl[contam], r[contam], s=30, c='tab:red', marker='x',
               label='contaminated')
    ax.axhline(0.0, color='k', lw=0.5, alpha=0.4)
    ax.set_ylabel('Pearson r')
    ax.set_xlabel('Channel mean wavelength (um)')
    ax.set_ylim(-0.3, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc='lower left')
    axes[2].set_title('(c) Pearson r (flag = below threshold)')

    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=130)
    plt.close(fig)
    print(f"  wrote plot {out_png}")


def main():
    args = parse_args()
    path = resolve_anchor_path(args)
    summary = smooth_anchor_file(
        path, r_threshold=args.r_threshold, s_factor=args.s_factor,
        spline_k=args.spline_k, dry_run=args.dry_run)
    chs = summary['chs']
    wl = summary['wl']
    slope_raw = summary['slope']
    C_raw = summary['intercept']
    r = summary['pearson_r']
    res = summary['result']
    det = summary['detector']
    contam = res['contaminated']
    n_rep = int(contam.sum())

    print(f"{path}")
    print(f"  detector D{det}, {len(chs)} channels, r_threshold="
          f"{args.r_threshold}, s_factor={args.s_factor}")
    print(f"  {n_rep} channel(s) flagged for smoothing:")
    for i, c in enumerate(chs):
        if not contam[i]:
            continue
        ex = '  [EXTRAPOLATED]' if res['extrapolated'][i] else ''
        print(f"    Ch{c:>2} (wl={wl[i]:.3f}, r={r[i]:+.3f}): "
              f"slope {slope_raw[i]:+.3f} -> {res['slope_final'][i]:+.3f}, "
              f"C {C_raw[i]:+.4g} -> {res['C_final'][i]:+.4g}{ex}")
    if res['extrapolated'].any():
        print("  WARNING: extrapolated smoothing(s) above — flagged channel "
              "outside the clean wavelength span; value is a spline "
              "extrapolation, inspect the plot.")
    if n_rep == 0:
        print("  nothing to smooth (all channels above threshold).")

    if args.plot is not None:
        out_png = (args.plot if args.plot != 'auto'
                   else os.path.join(os.path.dirname(path),
                                     f'anchor_D{det}_smooth.png'))
        make_plot(out_png, det, wl, slope_raw, C_raw, r, res)

    if args.dry_run:
        print("  --dry-run: anchor file NOT modified.")
    else:
        print(f"  updated {n_rep} channel(s) in-place; raw slope/intercept "
              f"preserved. anchor_method -> rweighted_spline.")


if __name__ == '__main__':
    main()
