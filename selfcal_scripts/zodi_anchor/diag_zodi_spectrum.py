"""Per-detector zodi spectrum diagnostic, read entirely from the anchor file.

Four panels vs channel-mean wavelength:
 (a) mean(full_DC), mean(zodi_pred), slope*mean(zodi_pred)
     - mean(full_DC) = what the solver saw (zodi + non-zodi uniform DC)
     - mean(zodi_pred) = pure zodipy/Kelsall
     - slope*mean(zodi_pred) = anchor-attributed zodi
 (b) C (the anchor constant = non-zodi uniform DC, added to mosaic)
 (c) fitted slope (=1 if zodipy captures the temporal shape)
 (d) Pearson r of full_DC vs zodi_pred

Pure anchor-file read — no cal / npz access, so it's instant. Everything
is already stored per channel by build_anchor.py.

    python diag_zodi_spectrum.py --anchor <run>/zodi_anchor/anchor_D1.h5 \\
        --out figures/zodi_anchor/D1_multichannel_34/D1_zodi_spectrum.png

With --run-dir the anchor file + a default out path are inferred.
"""
import argparse
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from selfcal.ZodiAnchor import load_anchor


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument('--anchor', help='Path to anchor_D{N}.h5.')
    src.add_argument('--run-dir',
                     help='Run dir; uses <run>/zodi_anchor/anchor_D{N}.h5. '
                          'Needs --detector if >1 anchor file present.')
    p.add_argument('--detector', type=int, default=None)
    p.add_argument('--out', default=None)
    p.add_argument('--max-ch', type=int, default=34,
                   help='Only plot channels <= this (e.g. 30 to drop the '
                        'airglow-blown D1 Ch31-34).')
    return p.parse_args()


def resolve_anchor_path(args):
    if args.anchor:
        return args.anchor
    import glob
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


def main():
    args = parse_args()
    anchor = load_anchor(resolve_anchor_path(args))
    det = anchor.detector

    chs = sorted(c for c in anchor.channels if c <= args.max_ch)
    rows = [anchor.channels[c] for c in chs]
    wl = np.array([r['wavelength_um'] for r in rows])
    mean_fs = np.array([r['mean_full_dc'] for r in rows])
    mean_zp = np.array([r['mean_pred'] for r in rows])
    slope = np.array([r['slope_final'] for r in rows])
    C = np.array([r['C_final'] for r in rows])
    rval = np.array([r['pearson_r'] for r in rows])
    fit_zodi = slope * mean_zp
    ch_arr = np.array(chs)

    fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

    ax = axes[0]
    ax.plot(wl, mean_fs, '-o', ms=4, lw=1, c='tab:blue',
            label='mean(full_DC) - solver saw (zodi + non-zodi uniform DC)')
    ax.plot(wl, mean_zp, '-o', ms=4, lw=1, c='tab:orange',
            label='mean(zodi_pred) - pure zodipy/Kelsall')
    ax.plot(wl, fit_zodi, '-s', ms=4, lw=1, c='tab:green',
            label='slope * mean(zodi_pred) - anchor-attributed zodi')
    ax.axhline(0, color='k', lw=0.5, alpha=0.4)
    ax.set_ylabel('MJy/sr')
    ax.set_title(f'(a) D{det}: per-channel mean DC. '
                 f'mean(full_DC) = slope*mean(zp) + C  (C in panel b)')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.axhline(0.0, color='k', lw=0.5, alpha=0.5)
    ax.plot(wl, C, '-^', ms=4, lw=1, c='tab:red',
            label='C - anchor-attributed non-zodi uniform DC (added to mosaic)')
    ax.set_ylabel('C  (MJy/sr)')
    ax.set_title('(b) Per-channel anchor constant C')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.axhline(1.0, color='k', lw=0.7, alpha=0.5)
    ax.plot(wl, slope, '-o', ms=4, lw=1, c='tab:orange',
            label='fitted slope (=1 if zodipy captures temporal shape)')
    ax.set_ylabel('slope')
    ax.set_title('(c) Fitted slope per channel')
    ax.legend(loc='best', fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[3]
    ax.axhline(1.0, color='k', lw=0.5, alpha=0.4)
    ax.axhline(0.0, color='k', lw=0.5, alpha=0.4)
    ax.plot(wl, rval, '-o', ms=4, lw=1, c='tab:blue',
            label=f'D{det} Pearson r')
    ax.set_ylabel('Pearson r')
    ax.set_title('(d) Per-frame correlation of full_DC vs zodi_pred')
    ax.set_xlabel('Channel mean wavelength (um)')
    ax.set_ylim(-0.3, 1.05)
    ax.legend(loc='lower left', fontsize=8)
    ax.grid(alpha=0.3)

    for a in axes:
        ymax = a.get_ylim()[1]
        for w, c in zip(wl, ch_arr):
            a.annotate(f'{c}', xy=(w, ymax),
                       xytext=(0, -2), textcoords='offset points',
                       ha='center', va='top', fontsize=5, color='gray',
                       alpha=0.6)

    out = args.out
    if out is None and args.run_dir:
        out = os.path.join(args.run_dir, 'zodi_anchor',
                           f'D{det}_zodi_spectrum.png')
    if out is None:
        out = f'D{det}_zodi_spectrum.png'
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"Saved {out}  (D{det}, {len(chs)} channels, method={anchor.anchor_method})")


if __name__ == '__main__':
    main()
