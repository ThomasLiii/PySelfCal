"""Per-subchannel raw-exposure stacked spectrum across detectors.

For each detector run passed via --run-dir, sample N exposures uniformly
across the survey and compute the per-subchannel median raw brightness of
the reprojected L2b data (pre self-cal). The result is a spectrum vs
wavelength at the *subchannel* resolution (NUM_COL=10 -> 342 subchannels
per detector vs 34 channels), which reveals inter-detector boundary jumps
in the raw data.

Inter-detector jumps are quantified by linearly extrapolating the spectrum
from each side toward the boundary wavelength (the midpoint between the
max wavelength of the lower detector and the min of the higher detector).

    python diag_raw_stack_spectrum.py --run-dir <RUN_D3> <RUN_D4> <RUN_D5> \\
        --n-exposures 600 \\
        --out figures/zodi_anchor/diag_raw_stack_spectrum.png

Output: PNG with a top panel (full spectrum, one curve per detector with
shaded +/- SE band) and a bottom row of zoom panels (one per adjacent
boundary, with the two extrapolation fit lines + jump annotation).
"""
import argparse
import glob
import os
import warnings

import h5py
import hdf5plugin  # noqa: F401  -- registers compression plugins for h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as ndi

from selfcal.geometry.map_helper import bit_to_bool
from selfcal.instruments.spherex.spherex_utility import load_lvf_params, make_stripped_chunk_map


NUM_SUB = 10
NUM_CH = 34
NUM_COL = 10
TOT_SUB = NUM_SUB * NUM_CH + 2  # 342
DET_COLORS = {
    1: 'tab:orange',
    2: 'tab:purple',
    3: 'tab:green',
    4: 'tab:blue',
    5: 'tab:red',
    6: 'tab:brown',
}


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--run-dir', nargs='+', required=True,
                   help='One or more run directories (each contains a '
                        'reprojected/ subdir with exp_*_det_*.h5 files).')
    p.add_argument('--n-exposures', type=int, default=600,
                   help='Number of exposures to sample uniformly per run '
                        '(default: 600).')
    p.add_argument('--out',
                   default='figures/zodi_anchor/diag_raw_stack_spectrum.png',
                   help='Output PNG path.')
    p.add_argument('--lvf-params-dir', default=None,
                   help='Override the lvf_params directory (defaults to the '
                        'package-relative data/lvf_params/).')
    return p.parse_args()


def detector_from_reproj_dir(run_dir):
    """Infer the detector id from any exp_*_det_*.h5 in run_dir/reprojected.

    The reprojected files are named exp_NNNN_det_MM.h5 where MM is the
    *zero-based* detector index in the SPHEREx convention. We rely on the
    run dir being single-detector (the production convention).
    """
    reproj_dir = os.path.join(run_dir, 'reprojected')
    files = sorted(glob.glob(os.path.join(reproj_dir, 'exp_*_det_*.h5')))
    if not files:
        raise SystemExit(f"No exp_*_det_*.h5 in {reproj_dir}")
    base = os.path.basename(files[0])
    # exp_0000_det_02.h5  ->  det index 02 -> detector 3 (1-based)
    det_str = base.split('_det_')[1].split('.')[0]
    return int(det_str) + 1, files


def build_subchannel_map(detector, lvf_params_dir):
    kwargs = {}
    if lvf_params_dir is not None:
        kwargs['input_dir'] = lvf_params_dir
    lvf = load_lvf_params(f'lvf_params_D{detector}.npy', **kwargs)
    if lvf is None:
        raise SystemExit(f"Could not load lvf_params_D{detector}.npy")
    chunk_map, _, _, _ = make_stripped_chunk_map(
        detector,
        num_subchannels=NUM_SUB,
        num_channels=NUM_CH,
        num_columns=NUM_COL,
        oversample_factor=1,
        lvf_params=lvf,
    )
    sub_map = (chunk_map // NUM_COL).astype(np.int32)
    return sub_map, lvf


def per_exposure_subchannel_medians(reproj_path, sub_map):
    """Per-subchannel median raw brightness for a single reprojected file.

    Returns an (TOT_SUB,) float64 array of medians (NaN where empty).
    """
    H, W = sub_map.shape
    with h5py.File(reproj_path, 'r') as f:
        sub_data = f['sub_data'][:].astype(np.float32, copy=False)
        sub_foot = f['sub_foot'][:].astype(np.float32, copy=False)
        sub_mapping = f['sub_mapping'][:].astype(np.float32, copy=False)
        sub_bitmask = f['sub_bitmask'][:]
    good = bit_to_bool(sub_bitmask, [], invert=True)

    col = np.round(np.nan_to_num(sub_mapping[0], nan=-1.0)).astype(np.intp)
    row = np.round(np.nan_to_num(sub_mapping[1], nan=-1.0)).astype(np.intp)

    in_bounds = (row >= 0) & (row < H) & (col >= 0) & (col < W)
    mask = in_bounds & (sub_foot > 0) & np.isfinite(sub_data) & (good > 0)
    if not np.any(mask):
        return np.full(TOT_SUB, np.nan, dtype=np.float64)

    r_sel = row[mask]
    c_sel = col[mask]
    vals = sub_data[mask].astype(np.float64, copy=False)
    subchan = sub_map[r_sel, c_sel]
    keep = (subchan >= 0) & (subchan < TOT_SUB)
    if not np.any(keep):
        return np.full(TOT_SUB, np.nan, dtype=np.float64)

    subchan = subchan[keep]
    vals = vals[keep]

    medians = ndi.median(vals, labels=subchan, index=np.arange(TOT_SUB))
    medians = np.asarray(medians, dtype=np.float64)
    # subchannels with no labelled pixels show up as 0 from ndi.median; mark
    # them NaN by checking which labels were present.
    present = np.zeros(TOT_SUB, dtype=bool)
    present[np.unique(subchan)] = True
    medians[~present] = np.nan
    return medians


def stack_detector(run_dir, n_exposures, lvf_params_dir):
    detector, files = detector_from_reproj_dir(run_dir)
    sub_map, lvf = build_subchannel_map(detector, lvf_params_dir)

    total = len(files)
    n = min(n_exposures, total)
    if n < total:
        idx = np.linspace(0, total - 1, n).round().astype(int)
        idx = np.unique(idx)  # dedupe in case rounding collides
        sampled = [files[i] for i in idx]
    else:
        sampled = files
    print(f"  D{detector}: sampling {len(sampled)}/{total} exposures")

    M = np.full((len(sampled), TOT_SUB), np.nan, dtype=np.float64)
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        for i, fp in enumerate(sampled):
            try:
                M[i] = per_exposure_subchannel_medians(fp, sub_map)
            except Exception as exc:
                print(f"    skip {os.path.basename(fp)}: {exc}")

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        spec = np.nanmedian(M, axis=0)
        std = np.nanstd(M, axis=0)
    cnt = np.sum(np.isfinite(M), axis=0).astype(np.int64)
    se = 1.2533 * std / np.sqrt(np.maximum(cnt, 1))

    # Wavelength axis: 342 subchannels, wave_edges has length 341.
    # WL[0] is the upper padding subchannel (NaN), WL[1:342] = wave_edges.
    wl = np.full(TOT_SUB, np.nan, dtype=np.float64)
    wl[1:1 + len(lvf['wave_edges'])] = np.asarray(lvf['wave_edges'])

    return {
        'detector': detector,
        'wl': wl,
        'spec': spec,
        'se': se,
        'n_per_sub': cnt,
        'n_exposures_used': len(sampled),
        'n_exposures_total': total,
    }


def linfit_window(wl, y, lo, hi):
    """Linear OLS fit of y vs wl in [lo, hi]; returns (a, b, n_used) where
    y_pred = a * wl + b. (a, b) are NaN if fewer than 2 finite points.
    """
    sel = np.isfinite(wl) & np.isfinite(y) & (wl >= lo) & (wl <= hi)
    if np.sum(sel) < 2:
        return np.nan, np.nan, int(np.sum(sel))
    x = wl[sel]
    yy = y[sel]
    a, b = np.polyfit(x, yy, 1)
    return float(a), float(b), int(np.sum(sel))


def main():
    args = parse_args()

    print(f"Stacking subchannel spectra for {len(args.run_dir)} run(s)...")
    detectors = []
    for run in args.run_dir:
        if not os.path.isdir(run):
            raise SystemExit(f"Not a directory: {run}")
        detectors.append(stack_detector(run, args.n_exposures,
                                        args.lvf_params_dir))

    # Sort detectors by wavelength (lower-mean first) so adjacent pairs are
    # boundary neighbours along the spectral axis.
    detectors.sort(key=lambda d: np.nanmin(d['wl']))

    # ----- Boundary jumps -----
    half = 0.15  # +/- um window around each boundary
    boundaries = []
    for lo, hi in zip(detectors[:-1], detectors[1:]):
        lo_max = float(np.nanmax(lo['wl']))
        hi_min = float(np.nanmin(hi['wl']))
        wb = 0.5 * (lo_max + hi_min)
        a_lo, b_lo, n_lo = linfit_window(
            lo['wl'], lo['spec'], wb - half, lo_max)
        a_hi, b_hi, n_hi = linfit_window(
            hi['wl'], hi['spec'], hi_min, wb + half)
        lo_extrap = a_lo * wb + b_lo
        hi_extrap = a_hi * wb + b_hi
        jump = hi_extrap - lo_extrap
        denom = 0.5 * (abs(lo_extrap) + abs(hi_extrap))
        pct = 100.0 * jump / denom if denom > 0 else np.nan
        boundaries.append({
            'lo_det': lo['detector'],
            'hi_det': hi['detector'],
            'wb': wb,
            'lo_max': lo_max,
            'hi_min': hi_min,
            'a_lo': a_lo, 'b_lo': b_lo, 'n_lo': n_lo,
            'a_hi': a_hi, 'b_hi': b_hi, 'n_hi': n_hi,
            'lo_extrap': lo_extrap,
            'hi_extrap': hi_extrap,
            'jump': jump,
            'jump_pct': pct,
        })
        print(f"D{lo['detector']}->D{hi['detector']} boundary "
              f"~{wb:.4f}um: lo_extrap={lo_extrap:.4f}, "
              f"hi_extrap={hi_extrap:.4f}, jump={jump:.4f}, "
              f"jump_pct={pct:.2f}%")

    # ----- Plot -----
    K = max(len(boundaries), 1)
    fig = plt.figure(figsize=(15, 11), dpi=130)
    gs = fig.add_gridspec(2, K, height_ratios=[2.4, 1.0], hspace=0.32,
                          wspace=0.30)
    ax_top = fig.add_subplot(gs[0, :])

    for d in detectors:
        color = DET_COLORS.get(d['detector'], 'tab:gray')
        wl = d['wl']
        spec = d['spec']
        se = d['se']
        ok = np.isfinite(wl) & np.isfinite(spec)
        order = np.argsort(wl[ok])
        wl_o = wl[ok][order]
        spec_o = spec[ok][order]
        se_o = se[ok][order]
        label = (f"D{d['detector']}  "
                 f"(N={d['n_exposures_used']}/{d['n_exposures_total']} exp)")
        ax_top.plot(wl_o, spec_o, '-', color=color, lw=1.2, label=label)
        ax_top.fill_between(wl_o, spec_o - se_o, spec_o + se_o,
                            color=color, alpha=0.18, linewidth=0)

    for b in boundaries:
        ax_top.axvline(b['wb'], color='k', ls=':', lw=0.7, alpha=0.6)

    ax_top.set_xlabel('Wavelength (um)')
    ax_top.set_ylabel('Per-subchannel median raw brightness (MJy/sr)')
    ax_top.set_title(
        'Raw L2b per-subchannel stacked spectrum '
        f'(NumSub={NUM_SUB}, NumCh={NUM_CH}, NumCol={NUM_COL}; '
        f'342 subchannels/det)'
    )
    ax_top.grid(alpha=0.3)
    ax_top.legend(loc='best', fontsize=9)

    # ----- Bottom row: zoom panels -----
    if not boundaries:
        ax_dummy = fig.add_subplot(gs[1, 0])
        ax_dummy.text(0.5, 0.5,
                      'No inter-detector boundaries (only one run passed).',
                      ha='center', va='center', transform=ax_dummy.transAxes,
                      fontsize=11)
        ax_dummy.set_axis_off()
    else:
        det_by_id = {d['detector']: d for d in detectors}
        for j, b in enumerate(boundaries):
            ax = fig.add_subplot(gs[1, j])
            wb = b['wb']
            xlo = wb - half * 1.1
            xhi = wb + half * 1.1

            for det_id, color in ((b['lo_det'], DET_COLORS.get(b['lo_det'],
                                                               'tab:gray')),
                                  (b['hi_det'], DET_COLORS.get(b['hi_det'],
                                                               'tab:gray'))):
                d = det_by_id[det_id]
                wl = d['wl']
                spec = d['spec']
                se = d['se']
                ok = (np.isfinite(wl) & np.isfinite(spec)
                      & (wl >= xlo) & (wl <= xhi))
                order = np.argsort(wl[ok])
                wl_o = wl[ok][order]
                spec_o = spec[ok][order]
                se_o = se[ok][order]
                ax.plot(wl_o, spec_o, '-o', ms=3, lw=1.0, color=color,
                        label=f'D{det_id}')
                ax.fill_between(wl_o, spec_o - se_o, spec_o + se_o,
                                color=color, alpha=0.18, linewidth=0)

            # Extrapolation lines
            xs_lo = np.linspace(wb - half, wb, 25)
            xs_hi = np.linspace(wb, wb + half, 25)
            if np.isfinite(b['a_lo']):
                ax.plot(xs_lo, b['a_lo'] * xs_lo + b['b_lo'],
                        '--', color=DET_COLORS.get(b['lo_det'], 'tab:gray'),
                        lw=1.0, alpha=0.8)
                ax.plot([wb], [b['lo_extrap']], 'v',
                        color=DET_COLORS.get(b['lo_det'], 'tab:gray'),
                        ms=7, mec='k', mew=0.6)
            if np.isfinite(b['a_hi']):
                ax.plot(xs_hi, b['a_hi'] * xs_hi + b['b_hi'],
                        '--', color=DET_COLORS.get(b['hi_det'], 'tab:gray'),
                        lw=1.0, alpha=0.8)
                ax.plot([wb], [b['hi_extrap']], '^',
                        color=DET_COLORS.get(b['hi_det'], 'tab:gray'),
                        ms=7, mec='k', mew=0.6)

            ax.axvline(wb, color='k', ls=':', lw=0.7, alpha=0.6)
            ax.set_xlabel('Wavelength (um)')
            ax.set_ylabel('MJy/sr')
            ax.grid(alpha=0.3)
            ax.set_title(
                f"D{b['lo_det']}->D{b['hi_det']} @ {wb:.3f}um\n"
                f"jump = {b['jump']:+.3f} MJy/sr "
                f"({b['jump_pct']:+.1f}%)",
                fontsize=10,
            )
            ax.legend(loc='best', fontsize=8)

    fig.suptitle(
        f"Raw stacked subchannel spectrum across "
        f"{len(detectors)} detector(s) "
        f"(D{','.join(str(d['detector']) for d in detectors)})",
        fontsize=13, y=0.995,
    )

    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    plt.savefig(out, dpi=130, bbox_inches='tight')
    print(f"Saved {out}")


if __name__ == '__main__':
    main()
