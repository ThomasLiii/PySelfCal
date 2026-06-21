"""Bandpass-sky null test for the overlap-subchannel residual.

The companion script ``diag_overlap_subchannel_continuity.py`` computes,
for every adjacent channel pair (c, c+1) sharing 20 physical chunks under
``subchannel_padding=1``::

    emp_dC(c, c+1) = < off_{c+1}[k, q] - off_c[k, q] >_{q, k in inlier}

and finds residuals 3-8 mMJy/sr per pair that survive subtracting
``anchor_dC = intercept_c - intercept_{c+1}``.

The verifier (rightly) noted that ``emp_dC`` is NOT a pure
C-difference. The two channels' independent LSQR solves write the same
data through different gauge fixings, so at a shared physical pixel:

    fdc_c[k, q]   = sky(lambda_c, pixel_q, t_k) + scalar_c[k] + off_c[k,q]
    fdc_{c+1}[..] = sky(lambda_{c+1}, ..)      + scalar_{c+1}[k] + off_{c+1}[..]

Differencing chunks ONLY (which is what diag does) gives::

    < off_{c+1} - off_c >  =
        - (anchor_C_c - anchor_C_{c+1})              # anchor delta
        - (sky(lam_{c+1}) - sky(lam_c))              # bandpass-sky delta
        - (scalar_{c+1} - scalar_c)                  # frame-scalar drift

i.e. the empirical overlap delta absorbs two physical effects that have
nothing to do with anchor incompleteness:

  (a) Bandpass-sky -- the SAME physical pixel sees a DIFFERENT photon
      flux in channel c vs c+1 because the LVF tunes the bandpass at
      shared edges to slightly different effective wavelengths. SPHEREx
      zodi flux varies smoothly in wavelength (~MJy/sr-scale spectrum),
      so even one-subchannel-width separation in lambda imprints a
      sub-mMJy/sr to mMJy/sr photon delta at the same physical pixel.
  (b) Frame-scalar drift -- the per-frame DC term ``frame_scalar`` in
      each cal file is an independent gauge for that channel's solve;
      its mean over inlier frames also enters the chunk-difference.

This script subtracts both, leaving an *unexplained* residual that, if
small, is consistent with the anchor (no incompleteness signal). The
first-cut bandpass-sky uses the per-channel-mean ZodiPy prediction
already cached in ``zodi_preds/zodi_pred_*.npz``. This is conservative
and OVER-estimates the bandpass effect (the shared chunks are at the
EDGES of each band, so the true wavelength gap is ~1 subchannel rather
than 1 channel) -- so a small "unexplained" after this correction is a
robust pass.

Usage::

    python selfcal_scripts/zodi_anchor/diag_overlap_bandpass_null.py
"""
import argparse
import os
import sys

import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from selfcal.zodi_anchor import load_anchor

# Sibling-script import (selfcal_scripts/zodi_anchor is not a package).
_HERE = os.path.abspath(os.path.dirname(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

from diag_overlap_subchannel_continuity import (  # type: ignore
    NUM_SUB,
    NUM_CH,
    NUM_COL,
    CAL_SUFFIX,
    DEFAULT_RUN_TEMPLATE,
    CAL_TEMPLATE,
    DET_COLORS,
    cal_path,
    shared_chunk_ids,
)


def zodi_pred_path(run_dir, D, ch):
    """Path to per-channel zodi-pred .npz (per-frame channel-mean
    bandpass integral)."""
    fname = (f'zodi_pred_Detector{D}_NumSub{NUM_SUB}_NumCh{NUM_CH}_'
             f'NumCol{NUM_COL}_Ch{ch}_{CAL_SUFFIX}.npz')
    return os.path.join(run_dir, 'zodi_preds', fname)


def pair_budget(run_dir, D, c):
    """Compute the four-term overlap budget for the pair (c, c+1).

    Returns dict with keys (all in MJy/sr):
        emp        scalar : < off_{c+1} - off_c >_{q, k} (matches diag)
        sem        scalar : SEM across the 20 shared chunks
        anchor     scalar : intercept_c - intercept_{c+1}
        bandpass   scalar : < zp_c - zp_{c+1} >_{k in inlier}
                            (per-channel-mean wavelength, first cut)
        scalar     scalar : < frame_scalar_c - frame_scalar_{c+1} >
        unexplained scalar: emp - (anchor + bandpass + scalar)
        n_shared   int    : number of shared chunks with finite per-chunk mean
        n_frames   int    : number of inlier frames (any-q co-cov)
    """
    q = shared_chunk_ids(c)
    pc = cal_path(run_dir, D, c)
    pn = cal_path(run_dir, D, c + 1)

    with h5py.File(pc, 'r') as f:
        off_c = f['offsets/map_0'][:, q].astype(np.float64)
        cov_c = f['offset_coverage/map_0'][:, q].astype(np.int64)
        fs_c = f['frame_scalar'][:].astype(np.float64)
    with h5py.File(pn, 'r') as f:
        off_n = f['offsets/map_0'][:, q].astype(np.float64)
        cov_n = f['offset_coverage/map_0'][:, q].astype(np.int64)
        fs_n = f['frame_scalar'][:].astype(np.float64)

    finite = np.isfinite(off_c) & np.isfinite(off_n)
    both = (cov_c > 0) & (cov_n > 0) & finite          # (Nf, 20)

    # --- empirical (chunk-difference) ---
    w = np.minimum(cov_c, cov_n).astype(np.float64)
    w[~both] = 0.0
    diff = off_n - off_c                                # (Nf, 20)
    num = (diff * w).sum(axis=0)
    den = w.sum(axis=0)
    per_chunk = np.where(den > 0, num / den, np.nan)    # (20,)
    good = np.isfinite(per_chunk)
    n_good = int(good.sum())
    if n_good == 0:
        emp = float('nan')
        sem = float('nan')
    else:
        emp = float(np.nanmean(per_chunk))
        sem = (float(np.nanstd(per_chunk, ddof=1) / np.sqrt(n_good))
               if n_good >= 2 else float('nan'))

    # --- frame-level inlier mask (any q co-covered in that frame) ---
    frame_inlier = both.any(axis=1)
    n_frames = int(frame_inlier.sum())

    # --- bandpass-sky (first cut: per-channel mean wavelength) ---
    zp_c_path = zodi_pred_path(run_dir, D, c)
    zp_n_path = zodi_pred_path(run_dir, D, c + 1)
    if not (os.path.exists(zp_c_path) and os.path.exists(zp_n_path)):
        bandpass = float('nan')
    else:
        with np.load(zp_c_path, allow_pickle=False) as zc, \
             np.load(zp_n_path, allow_pickle=False) as zn:
            zp_c_arr = zc['zodi_pred'].astype(np.float64)
            zp_n_arr = zn['zodi_pred'].astype(np.float64)
        # SIGN: emp ~ < off_{c+1} - off_c > = C_c - C_{c+1}
        #       (higher chunk DC in c+1 means the c+1 solve put more
        #        photons into the offset because its sky term sees less,
        #        i.e. its zodi-pred was smaller).
        # Bandpass-sky absorbed into emp has the SAME sign as
        # < zp_c - zp_{c+1} > over the inlier frames.
        finite_zp = (np.isfinite(zp_c_arr) & np.isfinite(zp_n_arr)
                     & frame_inlier)
        if finite_zp.sum() == 0:
            bandpass = float('nan')
        else:
            bandpass = float(np.mean(zp_c_arr[finite_zp]
                                     - zp_n_arr[finite_zp]))

    # --- frame-scalar drift ---
    finite_fs = (np.isfinite(fs_c) & np.isfinite(fs_n) & frame_inlier)
    if finite_fs.sum() == 0:
        scalar = float('nan')
    else:
        scalar = float(np.mean(fs_c[finite_fs] - fs_n[finite_fs]))

    return dict(
        emp=emp, sem=sem,
        bandpass=bandpass, scalar=scalar,
        per_chunk=per_chunk,
        n_shared=n_good,
        n_frames=n_frames,
    )


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--detector', type=int, nargs='+', default=[3, 4, 5],
                   help='Detectors to test. Default: 3 4 5.')
    p.add_argument('--run-template', default=DEFAULT_RUN_TEMPLATE,
                   help='Python format string with {D}. '
                        f'Default: {DEFAULT_RUN_TEMPLATE}')
    p.add_argument('--out', default='figures/zodi_anchor/overlap_bandpass_null.png',
                   help='Output plot path. Default: figures/zodi_anchor/'
                        'overlap_bandpass_null.png')
    p.add_argument('--out-data', default=None,
                   help='Optional .npz output with per-pair budget arrays.')
    return p.parse_args()


def main():
    args = parse_args()
    detectors = sorted(args.detector)

    # Resolve run dirs and load per-detector anchors
    run_dirs = {D: args.run_template.format(D=D) for D in detectors}
    anchors = {}
    intercepts = {}
    for D in detectors:
        ap = os.path.join(run_dirs[D], 'zodi_anchor', f'anchor_D{D}.h5')
        if not os.path.exists(ap):
            raise SystemExit(f"missing anchor file: {ap}")
        anchors[D] = load_anchor(ap)
        intercepts[D] = {ch: float(anchors[D].channels[ch]['intercept'])
                         for ch in anchors[D].channels}
        print(f"D{D}: anchor loaded ({ap}, method={anchors[D].anchor_method})")

    n_pairs = NUM_CH - 1  # 33

    results = {}
    for D in detectors:
        run_dir = run_dirs[D]
        emp = np.full(n_pairs, np.nan, dtype=np.float64)
        sem = np.full(n_pairs, np.nan, dtype=np.float64)
        ach = np.full(n_pairs, np.nan, dtype=np.float64)
        bp = np.full(n_pairs, np.nan, dtype=np.float64)
        sc = np.full(n_pairs, np.nan, dtype=np.float64)
        unx = np.full(n_pairs, np.nan, dtype=np.float64)
        # 'phys' = the physically-correct budget when chunks are NOT
        # anchor-mutated (Anchor.apply_to_cal_scalar shifts ONLY scalar
        # by -C, leaving offsets/map_0 untouched). In that gauge the
        # chunk-difference equation is
        #   < off_{c+1} - off_c > = (scalar_c - scalar_{c+1})
        #                         + (sky_c - sky_{c+1}).
        # phys_unx = emp - bandpass - scalar  -- if this is small, the
        # original diag's "anchor residual" was actually scalar+bandpass.
        phys_unx = np.full(n_pairs, np.nan, dtype=np.float64)

        print(f"\n=== D{D}: {n_pairs} adjacent-channel pairs "
              f"(run_dir={run_dir}) ===")
        header = (f"  pair  | shared/Nframe |   emp dC  +/- sem    |"
                  f"  anchor   bandpass    scalar   |  unx(+anc)  unx_phys")
        print(header)
        print('  ' + '-' * (len(header) - 2))

        for c in range(1, NUM_CH):
            i = c - 1
            try:
                b = pair_budget(run_dir, D, c)
            except FileNotFoundError as e:
                print(f"  {c:2d}-{c+1:2d}: missing cal: {e}", file=sys.stderr)
                continue
            emp[i] = b['emp']
            sem[i] = b['sem']
            ach[i] = intercepts[D][c] - intercepts[D][c + 1]
            bp[i] = b['bandpass']
            sc[i] = b['scalar']
            unx[i] = emp[i] - (ach[i] + bp[i] + sc[i])
            phys_unx[i] = emp[i] - (bp[i] + sc[i])
            print(f"  {c:2d}-{c+1:2d} | "
                  f"{b['n_shared']:2d}/{b['n_frames']:5d}      | "
                  f"{emp[i]*1e3:+7.2f} +/- {sem[i]*1e3:5.2f} | "
                  f"{ach[i]*1e3:+7.2f} {bp[i]*1e3:+8.2f} {sc[i]*1e3:+8.2f} | "
                  f"{unx[i]*1e3:+8.2f}  {phys_unx[i]*1e3:+8.2f}   (mMJy/sr)")

        # Summary
        emp_m = np.nanmean(emp); emp_max = np.nanmax(np.abs(emp))
        ach_m = np.nanmean(ach)
        bp_m = np.nanmean(bp); bp_s = np.nanstd(bp)
        sc_m = np.nanmean(sc); sc_s = np.nanstd(sc)
        unx_m = np.nanmean(unx); unx_s = np.nanstd(unx)
        unx_max = np.nanmax(np.abs(unx))
        phys_m = np.nanmean(phys_unx); phys_s = np.nanstd(phys_unx)
        phys_max = np.nanmax(np.abs(phys_unx))
        sem_m = np.nanmean(sem)
        print(f"\n  D{D} SUMMARY (mMJy/sr):")
        print(f"    empirical    : mean={emp_m*1e3:+6.2f}  max|.|={emp_max*1e3:5.2f}"
              f"  mean_sem={sem_m*1e3:5.2f}")
        print(f"    anchor       : mean={ach_m*1e3:+6.2f}")
        print(f"    bandpass-sky : mean={bp_m*1e3:+6.2f}  std={bp_s*1e3:5.2f}")
        print(f"    scalar drift : mean={sc_m*1e3:+6.2f}  std={sc_s*1e3:5.2f}")
        print(f"    unexp +anc   : mean={unx_m*1e3:+6.2f}  std={unx_s*1e3:5.2f}"
              f"  max|.|={unx_max*1e3:5.2f}  "
              f"(emp - anchor - bandpass - scalar; matches task spec)")
        print(f"    unexp phys   : mean={phys_m*1e3:+6.2f}  std={phys_s*1e3:5.2f}"
              f"  max|.|={phys_max*1e3:5.2f}  "
              f"(emp - bandpass - scalar; chunks are NOT anchor-mutated)")
        # Verdict on the physical budget (chunks not anchored => no C in eqn)
        pass_thresh = max(sem_m, 0.5e-3)
        verdict_phys = ("CONSISTENT" if abs(phys_m) < pass_thresh
                        else "RESIDUAL")
        verdict_task = ("CONSISTENT" if abs(unx_m) < pass_thresh
                        else "RESIDUAL")
        print(f"    Verdict (phys budget): |unexp_phys mean| = "
              f"{abs(phys_m)*1e3:.2f} mMJy/sr vs threshold "
              f"{pass_thresh*1e3:.2f} -> {verdict_phys}")
        print(f"    Verdict (task budget): |unexp_+anc mean| = "
              f"{abs(unx_m)*1e3:.2f} mMJy/sr vs threshold "
              f"{pass_thresh*1e3:.2f} -> {verdict_task}")
        print(f"    Bandpass-sky absorbs {bp_m*1e3:+.2f} mMJy/sr (mean) of the "
              f"{emp_m*1e3:+.2f} mMJy/sr empirical mean; scalar drift absorbs "
              f"{sc_m*1e3:+.2f}; physical unexplained = {phys_m*1e3:+.2f} mMJy/sr.")

        results[D] = dict(emp=emp, sem=sem, ach=ach, bp=bp, sc=sc, unx=unx,
                          phys_unx=phys_unx,
                          unx_m=unx_m, unx_s=unx_s, unx_max=unx_max,
                          phys_m=phys_m, phys_s=phys_s, phys_max=phys_max,
                          emp_m=emp_m, emp_max=emp_max,
                          ach_m=ach_m, bp_m=bp_m, sc_m=sc_m, sem_m=sem_m,
                          verdict_phys=verdict_phys, verdict_task=verdict_task,
                          pass_thresh=pass_thresh)

    # --------------------------- PLOT ---------------------------
    fig, axes = plt.subplots(len(detectors), 1,
                             figsize=(14, 3.4 * len(detectors)),
                             sharex=False)
    if len(detectors) == 1:
        axes = [axes]

    bar_w = 0.14
    bar_colors = dict(
        emp='black',
        anchor='tab:orange',
        bandpass='tab:cyan',
        scalar='tab:purple',
        unx_task='tab:red',
        unx_phys='tab:olive',
    )

    for ax, D in zip(axes, detectors):
        r = results[D]
        x = np.arange(1, NUM_CH)              # pair index, 1..33
        scale = 1e3                            # plot in mMJy/sr

        # Grouped bars (6 per pair). Order: emp, anchor, bandpass, scalar,
        # unx_task (with anchor), unx_phys (without anchor).
        ax.bar(x - 2.5 * bar_w, r['emp'] * scale, bar_w, label='empirical',
               color=bar_colors['emp'], alpha=0.85)
        ax.bar(x - 1.5 * bar_w, r['ach'] * scale, bar_w,
               label='anchor (intercept_c - intercept_{c+1})',
               color=bar_colors['anchor'], alpha=0.85)
        ax.bar(x - 0.5 * bar_w, r['bp'] * scale, bar_w,
               label='bandpass-sky <zp_c - zp_{c+1}> (UPPER BOUND, '
                     'per-channel mean lambda)',
               color=bar_colors['bandpass'], alpha=0.85)
        ax.bar(x + 0.5 * bar_w, r['sc'] * scale, bar_w,
               label='frame-scalar drift <fs_c - fs_{c+1}>',
               color=bar_colors['scalar'], alpha=0.85)
        ax.bar(x + 1.5 * bar_w, r['unx'] * scale, bar_w,
               label='unexplained (task) = emp - anchor - bp - scalar',
               color=bar_colors['unx_task'], alpha=0.9)
        ax.bar(x + 2.5 * bar_w, r['phys_unx'] * scale, bar_w,
               label='unexplained (phys) = emp - bp - scalar  '
                     '(chunks NOT anchor-mutated)',
               color=bar_colors['unx_phys'], alpha=0.9)
        # Error bars on empirical
        ax.errorbar(x - 2.5 * bar_w, r['emp'] * scale, yerr=r['sem'] * scale,
                    fmt='none', ecolor='gray', capsize=2, lw=0.7)

        ax.axhline(0.0, color='k', lw=0.5, alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{c}-{c+1}' for c in x], rotation=60, fontsize=7)
        ax.set_xlabel('Adjacent channel pair (c, c+1)')
        ax.set_ylabel('Delta C  (mMJy/sr)')
        ax.set_title(
            f'D{D}: overlap-subchannel residual budget   '
            f'mean emp={r["emp_m"]*1e3:+.2f}, anchor={r["ach_m"]*1e3:+.2f}, '
            f'bp={r["bp_m"]*1e3:+.2f}, scalar={r["sc_m"]*1e3:+.2f}; '
            f'PHYS unx mean={r["phys_m"]*1e3:+.2f} std={r["phys_s"]*1e3:.2f}'
            f' -> {r["verdict_phys"]} ; '
            f'TASK unx mean={r["unx_m"]*1e3:+.2f} std={r["unx_s"]*1e3:.2f}'
            f' -> {r["verdict_task"]}    (mMJy/sr)'
        )
        ax.grid(alpha=0.25)
        if D == detectors[0]:
            ax.legend(loc='best', fontsize=7, ncol=2)

    out = args.out
    os.makedirs(os.path.dirname(os.path.abspath(out)) or '.', exist_ok=True)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    print(f"\nSaved plot: {out}")

    if args.out_data:
        payload = {'detectors': np.asarray(detectors, dtype=np.int64)}
        for D in detectors:
            r = results[D]
            for key in ('emp', 'sem', 'ach', 'bp', 'sc', 'unx', 'phys_unx'):
                payload[f'{key}_D{D}'] = r[key]
        os.makedirs(os.path.dirname(os.path.abspath(args.out_data)) or '.',
                    exist_ok=True)
        np.savez(args.out_data, **payload)
        print(f"Saved data: {args.out_data}")

    # Print one-line per-detector verdicts
    print("\n=== VERDICTS ===")
    for D in detectors:
        r = results[D]
        print(f"  D{D}: emp |dC|_max={r['emp_max']*1e3:.2f}  mean_emp="
              f"{r['emp_m']*1e3:+.2f}  bandpass={r['bp_m']*1e3:+.2f}  "
              f"scalar={r['sc_m']*1e3:+.2f}  PHYS_unx mean="
              f"{r['phys_m']*1e3:+.2f} std={r['phys_s']*1e3:.2f} max="
              f"{r['phys_max']*1e3:.2f}  -> {r['verdict_phys']}  "
              f"(task-spec budget: anchor={r['ach_m']*1e3:+.2f}, "
              f"unx mean={r['unx_m']*1e3:+.2f} -> {r['verdict_task']})  "
              f"[mMJy/sr]")


if __name__ == '__main__':
    main()
