"""Overlap-subchannel sanity check for the v2 pedestal model.

The SelfCal pipeline runs each channel as an independent LSQR solve, but
each channel's solve is padded by ``subchannel_padding=1`` so that
adjacent channels (c, c+1) share exactly two global subchannels:

    shared_subch(c, c+1) = { c*10, c*10 + 1 }   (NUM_SUB=10)

With NumCol=10, that is 20 *shared* chunk_ids per adjacent pair: the
SAME physical detector pixels are estimated in both channel c's cal
file and channel c+1's cal file, by two independent solves.

The two solves have independent per-channel additive gauge ambiguities;
the post-hoc anchor C_c absorbs that ambiguity by shifting the recovered
sky by +C_c (and the per-frame scalar by -C_c) - leaving the data model
``data = sky + scalar + offset_chunk`` invariant. The chunk offsets
themselves are NOT mutated by the anchor.

So the operational continuity statement at a shared chunk q is::

    < offset_c[k, q] >_k  +  C_c   ==   < offset_{c+1}[k, q] >_k  +  C_{c+1}

(both sides represent the chunk-q DC level relative to the anchored
absolute-zero sky). Rearranging:

    dC_overlap(c, c+1)  :=  C_c - C_{c+1}
                        =  < offset_{c+1}[k, q] - offset_c[k, q] >_k

averaged over the 20 shared chunks per pair, weighted by per-frame
min-coverage (frames where both cals see q) for robustness.

This script then compares for each adjacent pair (c, c+1), per detector
D in {3,4,5}:

  (emp)    dC_overlap from the cal files (this script's measurement)
  (anchor) dC_anchor   = intercept_c - intercept_{c+1}   from anchor_D{N}.h5
  (v2)     dC_v2       = C_corr_D[c-1] - C_corr_D[c]    from the
           prototype_pedestal_anchor_v2.py output. Within a single
           detector, v2's per-detector pedestal P_D cancels in
           differences so dC_v2 must equal dC_anchor identically - the
           empirical leg is the independent test of whether the
           per-channel anchor C's reproduce the photon-level continuity
           that v2 inherits.

NOTE: this test only validates WITHIN-detector v2 logic. Adjacent-
detector pairs (e.g. D3-Ch34 vs D4-Ch1) live on different focal-plane
arrays and do not share physical chunks, so they cannot be tested this
way.

Usage::

    python selfcal_scripts/zodi_anchor/diag_overlap_subchannel_continuity.py
"""
import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from SelfCal.ZodiAnchor import load_anchor


# Production constants (match selfcal_scripts/drivers/run_cal_v2.py).
NUM_SUB = 10
NUM_CH = 34
NUM_COL = 10
SUBCHANNEL_PADDING = 1
CAL_SUFFIX = 'damp0p1_reg0p1_outThresh5_sigma2_polyK1'

DEFAULT_RUN_TEMPLATE = (
    '/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D{D}_6p2arcsec'
)
CAL_TEMPLATE = ('cal_Detector{D}_NumSub{NS}_NumCh{NCH}_NumCol{NC}_Ch{ch}'
                f'_{CAL_SUFFIX}.h5')

DET_COLORS = {1: 'tab:purple', 2: 'tab:orange',
              3: 'tab:green', 4: 'tab:blue', 5: 'tab:red'}


def shared_chunk_ids(c):
    """Chunk_ids shared between channels c and c+1 with padding=1.

    Adjacent channels with subchannel_padding=1 share exactly the two
    global subchannels ``c*NUM_SUB`` and ``c*NUM_SUB + 1``; each
    subchannel owns NUM_COL contiguous chunk_ids.
    """
    s0 = c * NUM_SUB
    s1 = c * NUM_SUB + 1
    return np.array(
        [s0 * NUM_COL + j for j in range(NUM_COL)] +
        [s1 * NUM_COL + j for j in range(NUM_COL)],
        dtype=np.int64,
    )


def cal_path(run_dir, D, ch):
    return os.path.join(
        run_dir, 'calibration',
        CAL_TEMPLATE.format(D=D, NS=NUM_SUB, NCH=NUM_CH, NC=NUM_COL, ch=ch),
    )


def empirical_dC(run_dir, D, c):
    """Return (mean, sem, per_chunk_mean, per_chunk_n_frames) for dC_c - dC_{c+1}.

    Per shared chunk q we compute the per-frame-min-coverage-weighted
    mean of (off_{c+1}[k,q] - off_c[k,q]) over frames where BOTH cals
    see chunk q (cov_c[k,q] > 0 AND cov_{c+1}[k,q] > 0).

    Returns
    -------
    mean : float
        Mean across the 20 shared chunks (simple mean of per-chunk means).
    sem : float
        Standard error of the mean = std(per_chunk) / sqrt(n_finite).
    per_chunk : (20,) float
        Per-chunk weighted means (NaN if a chunk had no co-covered frame).
    per_chunk_n_frames : (20,) int
        Number of co-covered frames per shared chunk.
    """
    q = shared_chunk_ids(c)
    pc = cal_path(run_dir, D, c)
    pn = cal_path(run_dir, D, c + 1)
    with h5py.File(pc, 'r') as f:
        off_c = f['offsets/map_0'][:, q].astype(np.float64)
        cov_c = f['offset_coverage/map_0'][:, q].astype(np.int64)
    with h5py.File(pn, 'r') as f:
        off_n = f['offsets/map_0'][:, q].astype(np.float64)
        cov_n = f['offset_coverage/map_0'][:, q].astype(np.int64)

    finite = np.isfinite(off_c) & np.isfinite(off_n)
    both = (cov_c > 0) & (cov_n > 0) & finite
    # min-coverage weight, zeroed where !both
    w = np.minimum(cov_c, cov_n).astype(np.float64)
    w[~both] = 0.0
    diff = off_n - off_c  # (Nf, 20)
    num = (diff * w).sum(axis=0)
    den = w.sum(axis=0)
    per_chunk = np.where(den > 0, num / den, np.nan)  # (20,)
    n_frames = both.sum(axis=0)  # (20,)

    good = np.isfinite(per_chunk)
    n_good = int(good.sum())
    if n_good == 0:
        return float('nan'), float('nan'), per_chunk, n_frames
    mean = float(np.nanmean(per_chunk))
    if n_good >= 2:
        sem = float(np.nanstd(per_chunk, ddof=1) / np.sqrt(n_good))
    else:
        sem = float('nan')
    return mean, sem, per_chunk, n_frames


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--detector', type=int, nargs='+', default=[3, 4, 5],
                   help='Detectors to test. Default: 3 4 5.')
    p.add_argument('--run-template', default=DEFAULT_RUN_TEMPLATE,
                   help='Python format string with {D}. Default: '
                        f'{DEFAULT_RUN_TEMPLATE}')
    p.add_argument('--v2-npz', default='/tmp/prototype_pedestal_v2_data.npz',
                   help='v2 prototype output (C_old_D{N}, C_corr_D{N}).')
    p.add_argument('--out', default='figures/zodi_anchor/overlap_subchannel_continuity.png',
                   help='Output plot path.')
    p.add_argument('--out-data', default=None,
                   help='Optional .npz output with per-pair arrays.')
    return p.parse_args()


def main():
    args = parse_args()
    detectors = sorted(args.detector)

    # Load v2 npz (per-detector C_old and C_corr arrays, channel-ordered 1..34)
    if not os.path.exists(args.v2_npz):
        print(f"WARN: v2 npz {args.v2_npz} not found; v2 trace will be NaN.",
              file=sys.stderr)
        v2 = None
    else:
        v2 = np.load(args.v2_npz)

    # Per-detector anchor + per-detector run dir
    run_dirs = {D: args.run_template.format(D=D) for D in detectors}
    anchors = {}
    for D in detectors:
        ap = os.path.join(run_dirs[D], 'zodi_anchor', f'anchor_D{D}.h5')
        if not os.path.exists(ap):
            raise SystemExit(f"missing anchor file: {ap}")
        anchors[D] = load_anchor(ap)
        print(f"D{D}: anchor loaded ({ap}, method={anchors[D].anchor_method})")

    # Per-detector compute
    results = {}  # D -> dict of arrays length 33
    for D in detectors:
        run_dir = run_dirs[D]
        anchor = anchors[D]
        # Pre-extract per-channel anchor intercept
        intercept = {ch: float(anchor.channels[ch]['intercept'])
                     for ch in anchor.channels}
        wl = {ch: float(anchor.channels[ch]['wavelength_um'])
              for ch in anchor.channels}

        n_pairs = NUM_CH - 1  # 33
        emp = np.full(n_pairs, np.nan, dtype=np.float64)
        sem = np.full(n_pairs, np.nan, dtype=np.float64)
        ach = np.full(n_pairs, np.nan, dtype=np.float64)
        v2d = np.full(n_pairs, np.nan, dtype=np.float64)
        wl_mid = np.full(n_pairs, np.nan, dtype=np.float64)
        per_chunk_all = np.full((n_pairs, 2 * NUM_COL), np.nan,
                                dtype=np.float64)

        if v2 is not None:
            C_corr = v2[f'C_corr_D{D}']  # (34,)
            C_old = v2[f'C_old_D{D}']    # (34,)
            P_D = float(v2[f'P_D{D}'])
        else:
            C_corr = C_old = None
            P_D = float('nan')

        print(f"\n=== D{D}: {n_pairs} adjacent-channel pairs "
              f"(run_dir={run_dir}) ===")
        print(f"  P_D (v2 per-detector pedestal) = {P_D:+.6f} MJy/sr")
        header = (f"  pair  | wl_mid | shared_chunks | emp dC    +/- sem      | "
                  f"anchor dC | v2 dC (=anchor for within-det) | resid(emp-anchor)")
        print(header)
        print('  ' + '-' * (len(header) - 2))

        for c in range(1, NUM_CH):  # c=1..33 -> pair (c, c+1)
            i = c - 1  # 0..32 array index
            m, s, pc, nfr = empirical_dC(run_dir, D, c)
            emp[i] = m
            sem[i] = s
            per_chunk_all[i] = pc
            ach[i] = intercept[c] - intercept[c + 1]
            if v2 is not None:
                v2d[i] = float(C_corr[c - 1] - C_corr[c])
            wl_mid[i] = 0.5 * (wl[c] + wl[c + 1])
            shared_n = int(np.isfinite(pc).sum())
            resid = m - ach[i]
            print(f"  {c:2d}-{c+1:2d} | {wl_mid[i]:6.3f} | "
                  f"{shared_n:2d}/20         | "
                  f"{m:+.5f} +/- {s:.5f} | "
                  f"{ach[i]:+.5f}   | {v2d[i]:+.5f}                          | "
                  f"{resid:+.5f}")

        # Summary statistics
        resid_arr = emp - ach
        print(f"  D{D} SUMMARY: residual (emp - anchor) mean={np.nanmean(resid_arr):+.5f}, "
              f"std={np.nanstd(resid_arr):.5f}, "
              f"max|resid|={np.nanmax(np.abs(resid_arr)):.5f}")
        # Check v2 == anchor identity (within detector)
        v2_anchor_resid = v2d - ach
        print(f"  D{D} v2-vs-anchor max|diff| = "
              f"{np.nanmax(np.abs(v2_anchor_resid)):.2e} "
              f"(should be ~0 within detector since P_D cancels)")

        results[D] = dict(emp=emp, sem=sem, ach=ach, v2=v2d, wl_mid=wl_mid,
                          per_chunk=per_chunk_all,
                          C_old=C_old, C_corr=C_corr, P_D=P_D)

    # Plot
    fig, axes = plt.subplots(len(detectors), 1, figsize=(13, 3.2 * len(detectors)),
                             sharex=False)
    if len(detectors) == 1:
        axes = [axes]

    for ax, D in zip(axes, detectors):
        r = results[D]
        x = np.arange(1, NUM_CH)  # pair label c = 1..33
        ax.axhline(0.0, color='k', lw=0.5, alpha=0.4)
        # anchor trace (per-channel anchor C differences)
        ax.plot(x, r['ach'], '-o', ms=4, lw=1, c='tab:orange',
                label='anchor: C_c - C_{c+1}  (intercept from anchor file)')
        # v2 trace (within-detector equal to anchor; plot to make that visible)
        ax.plot(x, r['v2'], '--x', ms=5, lw=1, c='tab:purple', alpha=0.7,
                label='v2: C_corr,c - C_corr,c+1  (== anchor within det; P_D cancels)')
        # empirical trace + error bars (the independent measurement)
        ax.errorbar(x, r['emp'], yerr=r['sem'],
                    fmt='-s', ms=4, lw=1, c=DET_COLORS.get(D, 'k'),
                    ecolor=DET_COLORS.get(D, 'k'), capsize=2,
                    label='empirical: <off_{c+1} - off_c>_q  (overlap chunks)')
        ax.set_ylabel('dC = C_c - C_{c+1}  (MJy/sr)')
        ax.set_title(f'D{D}: adjacent-pair C-difference  '
                     f'(P_D = {r["P_D"]:+.4f} MJy/sr, cancels in differences)')
        ax.set_xlabel('Adjacent channel pair index c  (pair = c, c+1)')
        ax.set_xticks(x)
        ax.set_xticklabels([f'{c}-{c+1}' for c in x], rotation=60,
                           fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(loc='best', fontsize=7)

    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"\nSaved {out}")

    if args.out_data:
        save = {}
        save['detectors'] = np.array(detectors, dtype=np.int64)
        for D in detectors:
            r = results[D]
            save[f'emp_D{D}'] = r['emp']
            save[f'sem_D{D}'] = r['sem']
            save[f'anchor_D{D}'] = r['ach']
            save[f'v2_D{D}'] = r['v2']
            save[f'wl_mid_D{D}'] = r['wl_mid']
            save[f'per_chunk_D{D}'] = r['per_chunk']
        os.makedirs(os.path.dirname(os.path.abspath(args.out_data)),
                    exist_ok=True)
        np.savez(args.out_data, **save)
        print(f"Saved {args.out_data}")


if __name__ == '__main__':
    main()
