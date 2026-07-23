"""Boundary jackknives for the slope-smoothness refit.

Question: do the residual ``smooth-poly`` boundary jumps — the jumps in
C(lambda) that remain after the smooth-slope refit (the ``smooth``
variant of refit_smooth_slope.py: slope(lambda) constrained to one
global degree-K polynomial across detectors, C free per channel) — at
the four SPHEREx detector seams (D1->D2 ~ 1.10 um, D2->D3 ~ 1.65 um,
D3->D4 ~ 2.42 um, D4->D5 ~ 3.81 um) come from a handful of dichroic-seam
channels (Ch1-3 / Ch32-34 on each side), or are they a bulk continuum
mismatch?

Strategy: re-run ``fit_smooth_global`` from refit_smooth_slope.py while
dropping increasingly many edge channels from the design matrix, and
report the post-drop boundary jumps two ways:

* ``data`` jump  -- delta_C measured at the *surviving* nearest channels
  on each side of the seam.
* ``ext`` jump   -- delta_C extrapolated back to the *original* boundary
  midpoint (so the value is comparable across drop specs) using a linear
  C(lambda) extrapolation per detector.

If the jumps collapse as edge channels are dropped, the residual seam is
an edge-channel artifact. If they persist (or grow because the
extrapolation has further to travel) the residual seam is bulk physics.

Example::

    python selfcal_scripts/zodi_anchor/diag_boundary_jackknife.py \\
        --run-dir /mnt/md124/.../SPHEREx_NEP_2026W17_D1_6p2arcsec \\
                  /mnt/md124/.../SPHEREx_NEP_2026W17_D2_6p2arcsec \\
                  /mnt/md124/.../SPHEREx_NEP_2026W17_D3_6p2arcsec \\
                  /mnt/md124/.../SPHEREx_NEP_2026W17_D4_6p2arcsec \\
                  /mnt/md124/.../SPHEREx_NEP_2026W17_D5_6p2arcsec
"""
import argparse
import copy
import json
import os
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Allow `import refit_smooth_slope` no matter where the script is invoked from
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
if _PKG_DIR not in sys.path:
    sys.path.insert(0, _PKG_DIR)

from refit_smooth_slope import (  # noqa: E402
    DET_COLORS,
    boundary_jump,
    fit_smooth_global,
    load_detector,
)


# ----------------------------------------------------------------------
# Drop specs (default; overridable via --drop-spec)
# ----------------------------------------------------------------------

# Spec names encode <seam>_b(oundary)_<n>ea = drop the n outermost edge
# channels on EAch side of that seam; e.g. D12_b_2ea drops D1 Ch33-34 and
# D2 Ch1-2. These names appear verbatim in the summary table, plot
# legends, and npz keys.
DEFAULT_DROP_SPECS = [
    ('baseline',  {}),
    # D1-D2 seam only
    ('D12_b_1ea', {1: {34},          2: {1}}),
    ('D12_b_2ea', {1: {33, 34},      2: {1, 2}}),
    ('D12_b_3ea', {1: {32, 33, 34},  2: {1, 2, 3}}),
    # D2-D3 seam only
    ('D23_b_1ea', {2: {34},          3: {1}}),
    ('D23_b_2ea', {2: {33, 34},      3: {1, 2}}),
    ('D23_b_3ea', {2: {32, 33, 34},  3: {1, 2, 3}}),
    # D3-D4 seam only
    ('D34_b_1ea', {3: {34},          4: {1}}),
    ('D34_b_2ea', {3: {33, 34},      4: {1, 2}}),
    ('D34_b_3ea', {3: {32, 33, 34},  4: {1, 2, 3}}),
    # D4-D5 seam only
    ('D45_b_1ea', {4: {34},          5: {1}}),
    ('D45_b_2ea', {4: {33, 34},      5: {1, 2}}),
    ('D45_b_3ea', {4: {32, 33, 34},  5: {1, 2, 3}}),
    # All four seams together. 'BOTH' is historical naming (originally
    # only the D3-D4 and D4-D5 seams were jackknifed); these specs now
    # drop edge channels at all four seams. The name is load-bearing:
    # the verdict grouping below does name.startswith('BOTH'), and it
    # appears in the npz keys, so it is kept for output stability.
    ('BOTH_2ea',  {1: {33, 34},
                   2: {1, 2, 33, 34},
                   3: {1, 2, 33, 34},
                   4: {1, 2, 33, 34},
                   5: {1, 2}}),
    ('BOTH_3ea',  {1: {32, 33, 34},
                   2: {1, 2, 3, 32, 33, 34},
                   3: {1, 2, 3, 32, 33, 34},
                   4: {1, 2, 3, 32, 33, 34},
                   5: {1, 2, 3}}),
]


def parse_drop_spec_arg(spec_str):
    """Parse a drop-spec argument into a {det_id: set([ch_id...])} dict.

    Accepts two formats:

    * JSON object, e.g. ``'{"3":[34],"4":[1]}'``
    * CSV of ``Dx:Chy`` tokens, e.g. ``'D3:Ch34,D4:Ch1,D4:Ch2'``
    """
    s = spec_str.strip()
    if not s:
        return {}
    if s.startswith('{'):
        raw = json.loads(s)
        out = {}
        for k, v in raw.items():
            d = int(re.sub(r'\D', '', str(k)))
            out[d] = {int(c) for c in v}
        return out
    out = {}
    for tok in s.split(','):
        tok = tok.strip()
        if not tok:
            continue
        m = re.match(r'D?(\d+)\s*:\s*(?:Ch)?(\d+)', tok)
        if not m:
            raise ValueError(f"cannot parse drop token: {tok!r}")
        d = int(m.group(1))
        ch = int(m.group(2))
        out.setdefault(d, set()).add(ch)
    return out


# ----------------------------------------------------------------------
# Drop application
# ----------------------------------------------------------------------

def apply_drop(detectors, drop_map):
    """Return per-detector dict copies with the requested (det, ch)
    pairs removed from channels / WL / FDC / ZP / MJD.

    detectors : list of dicts as returned by load_detector()
    drop_map  : {det_id: set([ch_id, ...])}
    """
    out = []
    for d in detectors:
        nd = copy.copy(d)  # shallow copy is enough; we rebind list/array refs
        chs = d['channels']
        drop = drop_map.get(d['detector'], set())
        keep = np.array([int(c) not in drop for c in chs], dtype=bool)
        nd['channels'] = chs[keep]
        nd['WL'] = d['WL'][keep]
        nd['FDC'] = [x for x, k in zip(d['FDC'], keep) if k]
        nd['ZP'] = [x for x, k in zip(d['ZP'], keep) if k]
        nd['MJD'] = [x for x, k in zip(d['MJD'], keep) if k]
        out.append(nd)
    return out


# ----------------------------------------------------------------------
# Boundary jump helpers
# ----------------------------------------------------------------------

def C_at_linear(wl_arr, C_arr, lam_target):
    """Linear C(lambda) extrapolation. Returns NaN if <2 surviving."""
    wl = np.asarray(wl_arr, dtype=np.float64)
    C = np.asarray(C_arr, dtype=np.float64)
    m = np.isfinite(wl) & np.isfinite(C)
    if m.sum() < 2:
        return np.nan
    coef = np.polyfit(wl[m], C[m], 1)
    return float(np.polyval(coef, lam_target))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument('--run-dir', nargs='+', required=True,
                   help='SPHEREx run directories (one per detector, '
                        'e.g. D1..D5).')
    p.add_argument('--poly-degree', type=int, default=3,
                   help='K for the smooth slope poly (default 3).')
    p.add_argument('--sigma', type=float, default=3.0,
                   help='Sigma-clip threshold (default 3.0).')
    p.add_argument('--window-days', type=float, default=7.0,
                   help='Moving MJD window for the sigma clip (default 7.0).')
    p.add_argument('--n-iter', type=int, default=2,
                   help='Sigma-clip refit iterations (default 2).')
    p.add_argument('--cal-glob-pat', default='cal_*polyK1.h5',
                   help="Cal glob inside <run>/calibration "
                        "(default 'cal_*polyK1.h5').")
    p.add_argument('--drop-spec', action='append', default=None,
                   metavar='NAME=SPEC',
                   help='Override default drop specs. Repeatable. '
                        'NAME=SPEC where SPEC is JSON or Dx:Chy CSV. '
                        'If any --drop-spec is given, the built-in list is '
                        'replaced entirely (a "baseline=" entry is auto-'
                        'prepended if missing).')
    p.add_argument('--out-plot', default=None,
                   help='Output PNG (default figures/zodi_anchor/'
                        'boundary_jackknife.png).')
    p.add_argument('--out-data', default=None,
                   help='Output .npz (default: swap .png -> .npz on '
                        '--out-plot).')
    return p.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    args = parse_args()

    repo_root = os.path.abspath(os.path.join(_PKG_DIR, '..', '..'))
    default_plot = os.path.join(
        repo_root, 'figures', 'zodi_anchor', 'boundary_jackknife.png')
    out_plot = args.out_plot or default_plot
    out_data = args.out_data or (os.path.splitext(out_plot)[0] + '.npz')
    os.makedirs(os.path.dirname(os.path.abspath(out_plot)) or '.',
                exist_ok=True)

    # ------------- Drop spec list -------------
    if args.drop_spec:
        specs = []
        names_seen = set()
        for raw in args.drop_spec:
            if '=' not in raw:
                raise SystemExit(
                    f"--drop-spec must be NAME=SPEC, got {raw!r}")
            name, spec_str = raw.split('=', 1)
            name = name.strip()
            dmap = parse_drop_spec_arg(spec_str)
            specs.append((name, dmap))
            names_seen.add(name)
        if 'baseline' not in names_seen:
            specs.insert(0, ('baseline', {}))
    else:
        specs = list(DEFAULT_DROP_SPECS)

    # ------------- Load detectors ONCE -------------
    print("Loading detectors ...")
    base_detectors = []
    for run_dir in args.run_dir:
        print(f"  {run_dir}")
        det_data = load_detector(run_dir, args.cal_glob_pat)
        det = det_data['detector']
        n_ch = len(det_data['channels'])
        n_frames_per_ch = [len(f) for f in det_data['FDC']]
        print(f"    D{det}: {n_ch} channels, "
              f"N_frames range = "
              f"[{min(n_frames_per_ch)}, {max(n_frames_per_ch)}]")
        base_detectors.append(det_data)
    base_detectors.sort(key=lambda d: d['detector'])
    det_ids = [d['detector'] for d in base_detectors]

    # Capture ORIGINAL per-detector WL extrema BEFORE any drops so the
    # "extrapolated" jump always refers to the same boundary lambda.
    orig_wl_max = {}
    orig_wl_min = {}
    for d in base_detectors:
        wl_finite = d['WL'][np.isfinite(d['WL'])]
        orig_wl_max[d['detector']] = float(np.nanmax(wl_finite))
        orig_wl_min[d['detector']] = float(np.nanmin(wl_finite))

    # Original boundary midpoints: the fixed seam wavelengths (e.g.
    # ~2.42 um for D3->D4, ~3.81 um for D4->D5) at which every drop
    # spec's extrapolated jump is evaluated, so values stay comparable
    # across specs even as edge channels are dropped.
    boundary_pairs = []  # (dA, dB, lam_mid)
    for i in range(len(det_ids) - 1):
        dA = det_ids[i]
        dB = det_ids[i + 1]
        lam_mid = 0.5 * (orig_wl_max[dA] + orig_wl_min[dB])
        boundary_pairs.append((dA, dB, lam_mid))
    print("\nOriginal boundary lambdas (midpoint):")
    for dA, dB, lam_mid in boundary_pairs:
        print(f"  D{dA} (max_lam={orig_wl_max[dA]:.4f}) -> "
              f"D{dB} (min_lam={orig_wl_min[dB]:.4f}): "
              f"lam_mid = {lam_mid:.4f} um")

    # ------------- Refit per drop spec -------------
    results = {}  # name -> dict
    print(f"\nRefitting smooth K={args.poly_degree} for "
          f"{len(specs)} drop specs ...")
    for name, drop_map in specs:
        dets = apply_drop(base_detectors, drop_map)
        # Per-detector surviving channel count
        n_keep = {d['detector']: len(d['channels']) for d in dets}
        drop_pretty = ', '.join(
            f"D{d}:{sorted(v)}" for d, v in sorted(drop_map.items())) \
            if drop_map else '(none)'
        print(f"  [{name}] drop = {drop_pretty}; "
              f"surviving per det = {n_keep}")
        sm = fit_smooth_global(
            dets, args.poly_degree,
            args.window_days, args.sigma, args.n_iter)
        per_det = {}
        for d_id in det_ids:
            msk = (sm['det_of_ch'] == d_id)
            per_det[d_id] = (sm['WL'][msk], sm['C'][msk],
                             sm['ch_id'][msk])
        # 'data' jump at surviving nearest channels
        # 'ext'  jump at the ORIGINAL boundary midpoint via linear C(lam)
        data_jumps = []
        ext_jumps = []
        for dA, dB, lam_mid in boundary_pairs:
            wlA, CA, _ = per_det[dA]
            wlB, CB, _ = per_det[dB]
            jD, lamA, lamB = boundary_jump(wlA, CA, wlB, CB)
            cA_at = C_at_linear(wlA, CA, lam_mid)
            cB_at = C_at_linear(wlB, CB, lam_mid)
            jE = cB_at - cA_at
            data_jumps.append(dict(dA=dA, dB=dB, lamA=lamA, lamB=lamB,
                                   dC_mMJy=jD * 1e3))
            ext_jumps.append(dict(dA=dA, dB=dB, lam_mid=lam_mid,
                                  dC_mMJy=jE * 1e3,
                                  cA=cA_at, cB=cB_at,
                                  n_keep_A=int(np.isfinite(wlA).sum()),
                                  n_keep_B=int(np.isfinite(wlB).sum())))
        results[name] = dict(
            drop_map=drop_map,
            smooth=sm,
            per_det=per_det,
            data_jumps=data_jumps,
            ext_jumps=ext_jumps,
            n_keep=n_keep,
        )

    # ------------- Print summary table -------------
    print("\n--- Boundary-jump table (mMJy/sr) ---")
    hdr = f"  {'name':<10s}  "
    for dA, dB, _ in boundary_pairs:
        hdr += f"{'D'+str(dA)+'->D'+str(dB)+' data':>18s}  " \
               f"{'D'+str(dA)+'->D'+str(dB)+' ext':>18s}  "
    print(hdr)
    for name, _ in specs:
        r = results[name]
        line = f"  {name:<10s}  "
        for j_d, j_e in zip(r['data_jumps'], r['ext_jumps']):
            line += f"{j_d['dC_mMJy']:+18.2f}  {j_e['dC_mMJy']:+18.2f}  "
        print(line)

    # Baseline reference for relative-change print
    base = results['baseline']
    base_ext = {(j['dA'], j['dB']): j['dC_mMJy'] for j in base['ext_jumps']}

    # ------------- Interpretation (extrapolated at original lambda) ----
    print("\n--- Interpretation (delta_C at ORIGINAL boundary lambda) ---")
    for k, (dA, dB, lam_mid) in enumerate(boundary_pairs):
        ref = base_ext[(dA, dB)]
        # Group drop specs that target this boundary
        tag = f'D{dA}{dB}_b'
        track = []
        for name, _ in specs:
            if name == 'baseline':
                continue
            if tag is not None and (name.startswith(tag)
                                    or name.startswith('BOTH')):
                track.append((name, results[name]['ext_jumps'][k]['dC_mMJy']))
        if not track:
            continue
        last_name, last_val = track[-1]
        change = last_val - ref
        # Shrink = |last| < 0.5 * |ref|; persists if |last| > 0.7 * |ref|
        abs_ref = abs(ref)
        abs_last = abs(last_val)
        if abs_last < 0.5 * max(abs_ref, 1e-6):
            verdict = (f"D{dA}->D{dB} jump SHRINKS from {ref:+.2f} to "
                       f"{last_val:+.2f} mMJy/sr when {last_name} edge "
                       f"channels dropped -> driven by edge artifacts.")
        elif abs_last > 0.7 * abs_ref:
            verdict = (f"D{dA}->D{dB} jump PERSISTS at {last_val:+.2f} "
                       f"mMJy/sr (baseline {ref:+.2f}) even after {last_name} "
                       f"-> bulk physics, not an edge artifact.")
        else:
            verdict = (f"D{dA}->D{dB} jump partially shrinks {ref:+.2f} -> "
                       f"{last_val:+.2f} mMJy/sr under {last_name} "
                       f"-> mixed edge + bulk contribution.")
        track_str = ', '.join(f"{n}={v:+.2f}" for n, v in track)
        print(f"  {verdict}  Track: baseline={ref:+.2f}; {track_str}.")

    # ------------- Save .npz ------------
    npz_payload = {
        'detectors': np.asarray(det_ids, dtype=np.int32),
        'poly_degree': np.int32(args.poly_degree),
        'spec_names': np.asarray([n for n, _ in specs], dtype=object),
        'boundary_lam_mid': np.asarray([lm for _, _, lm in boundary_pairs],
                                       dtype=np.float64),
        'boundary_dA': np.asarray([dA for dA, _, _ in boundary_pairs],
                                  dtype=np.int32),
        'boundary_dB': np.asarray([dB for _, dB, _ in boundary_pairs],
                                  dtype=np.int32),
        'orig_wl_max': np.asarray([orig_wl_max[d] for d in det_ids],
                                  dtype=np.float64),
        'orig_wl_min': np.asarray([orig_wl_min[d] for d in det_ids],
                                  dtype=np.float64),
    }
    for name, _ in specs:
        r = results[name]
        sm = r['smooth']
        safe = re.sub(r'\W+', '_', name)
        npz_payload[f'{safe}_coef'] = sm['coef']
        npz_payload[f'{safe}_WL'] = sm['WL']
        npz_payload[f'{safe}_C'] = sm['C']
        npz_payload[f'{safe}_slope'] = sm['slope']
        npz_payload[f'{safe}_det_of_ch'] = sm['det_of_ch']
        npz_payload[f'{safe}_ch_id'] = sm['ch_id']
        npz_payload[f'{safe}_data_jumps_mMJy'] = np.asarray(
            [j['dC_mMJy'] for j in r['data_jumps']], dtype=np.float64)
        npz_payload[f'{safe}_ext_jumps_mMJy'] = np.asarray(
            [j['dC_mMJy'] for j in r['ext_jumps']], dtype=np.float64)
    np.savez(out_data, **npz_payload, allow_pickle=True)
    print(f"\nSaved data: {out_data}")

    # ------------- Plot ------------
    _make_plot(results, specs, base_detectors, boundary_pairs,
               orig_wl_max, orig_wl_min, args.poly_degree, out_plot)
    print(f"Saved plot: {out_plot}")


# ----------------------------------------------------------------------
# Plot
# ----------------------------------------------------------------------

def _make_plot(results, specs, base_detectors, boundary_pairs,
               orig_wl_max, orig_wl_min, poly_degree, out_plot):
    spec_names = [n for n, _ in specs]
    n_specs = len(spec_names)
    n_bnd = len(boundary_pairs)

    # Layout: 2 wide bar panels (data + ext) span the top row;
    #         n_bnd C(lambda) zoom panels in a row beneath.
    n_C_rows = max(1, (n_bnd + 1) // 2)
    fig = plt.figure(figsize=(16, 6 + 2.6 * n_C_rows))
    gs = fig.add_gridspec(
        2 + n_C_rows, 2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0] + [1.0] * n_C_rows,
        hspace=0.55, wspace=0.25,
    )
    ax_bar_d = fig.add_subplot(gs[0, :])
    ax_bar_e = fig.add_subplot(gs[1, :])
    ax_C_list = []
    for i in range(n_bnd):
        row = 2 + i // 2
        col = i % 2
        ax_C_list.append(fig.add_subplot(gs[row, col]))

    # ---- bar charts (data + extrapolated) ----
    x = np.arange(n_specs)
    width = 0.8 / max(n_bnd, 1)
    palette = ['tab:blue', 'tab:green', 'tab:purple', 'tab:orange',
               'tab:red', 'tab:cyan']
    bnd_colors = {(dA, dB): palette[i % len(palette)]
                  for i, (dA, dB, _) in enumerate(boundary_pairs)}

    def _bar_panel(ax, jump_key, title):
        for i, (dA, dB, _) in enumerate(boundary_pairs):
            vals = np.array([
                results[name][jump_key][i]['dC_mMJy'] for name in spec_names
            ], dtype=np.float64)
            offset = (i - (n_bnd - 1) / 2.0) * width
            col = bnd_colors.get((dA, dB), 'tab:gray')
            ax.bar(x + offset, vals, width=width, color=col,
                   edgecolor='k', lw=0.4,
                   label=f'D{dA}->D{dB}')
            for xi, v in zip(x + offset, vals):
                ax.text(xi, v + (0.6 if v >= 0 else -0.6),
                        f"{v:+.1f}",
                        ha='center', va='bottom' if v >= 0 else 'top',
                        fontsize=5, rotation=0)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(spec_names, rotation=30, ha='right', fontsize=7)
        ax.set_ylabel('Delta C  (mMJy/sr)')
        ax.set_title(title)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(fontsize=7, loc='best', ncol=min(n_bnd, 4))

    _bar_panel(ax_bar_d, 'data_jumps',
               '(a) Boundary jump at SURVIVING nearest channels  '
               '("data" jump)')
    _bar_panel(ax_bar_e, 'ext_jumps',
               '(b) Boundary jump EXTRAPOLATED to original lambda_mid  '
               '("ext" jump)')

    # ---- C(lambda) overlay around each boundary ----
    # Build a cmap across drop specs for visual continuity
    cmap = plt.get_cmap('viridis', n_specs)
    spec_colors = {name: cmap(i) for i, name in enumerate(spec_names)}

    # Originally dropped channels per spec (for hollow circles)
    spec_drops = {name: dmap for name, dmap in specs}

    def _C_panel(ax, dA, dB, lam_mid):
        lam_lo = lam_mid - 0.35
        lam_hi = lam_mid + 0.35
        # Plot baseline channel WL ticks first as a reference
        for d in (dA, dB):
            base = next(b for b in base_detectors if b['detector'] == d)
            wls = base['WL']
            in_win = (wls >= lam_lo) & (wls <= lam_hi)
            for w in wls[in_win]:
                ax.axvline(w, color='lightgray', lw=0.4, alpha=0.6)
        ax.axvline(lam_mid, color='gray', lw=0.8, ls=':',
                   label=f'orig boundary {lam_mid:.3f} um')

        for name in spec_names:
            r = results[name]
            col = spec_colors[name]
            drop_for_dA = spec_drops[name].get(dA, set())
            drop_for_dB = spec_drops[name].get(dB, set())
            for d, drop_set in ((dA, drop_for_dA), (dB, drop_for_dB)):
                wl, C, ch = r['per_det'][d]
                if len(wl) == 0:
                    continue
                in_win = (wl >= lam_lo) & (wl <= lam_hi)
                if not in_win.any():
                    continue
                order = np.argsort(wl[in_win])
                wl_w = wl[in_win][order]
                C_w = C[in_win][order]
                ax.plot(wl_w, C_w, color=col, lw=1.0, alpha=0.9,
                        marker='o', ms=4, mfc=col, mec=col,
                        label=f'{name}' if d == dA else None)
            # Also draw the dropped (baseline) channel markers as hollow
            # circles at their original WL, evaluated on baseline C poly
            baseline_smooth = results['baseline']['smooth']
            for d, drop_set in ((dA, drop_for_dA), (dB, drop_for_dB)):
                if not drop_set:
                    continue
                base = next(b for b in base_detectors if b['detector'] == d)
                base_wls = base['WL']
                base_chs = base['channels']
                in_win = (base_wls >= lam_lo) & (base_wls <= lam_hi)
                for w, c in zip(base_wls[in_win], base_chs[in_win]):
                    if int(c) in drop_set:
                        # Plot a hollow circle at the baseline-fit C
                        msk_bs = ((baseline_smooth['det_of_ch'] == d)
                                  & (baseline_smooth['ch_id'] == int(c)))
                        if msk_bs.any():
                            C_bs = float(baseline_smooth['C'][msk_bs][0])
                            ax.plot(w, C_bs, marker='o', mfc='none',
                                    mec=col, ms=8, mew=1.0)

        ax.set_xlim(lam_lo, lam_hi)
        ax.set_xlabel('lambda  (um)')
        ax.set_ylabel('C  (MJy/sr)')
        ax.set_title(f'(zoom) D{dA} <-> D{dB} smooth-poly C(lambda); '
                     'hollow = dropped')
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc='best')

    for ax_C, (dA, dB, lam_mid) in zip(ax_C_list, boundary_pairs):
        _C_panel(ax_C, dA, dB, lam_mid)

    fig.suptitle(
        f'Boundary jackknives for the slope-smoothness refit  '
        f'(poly K={poly_degree})',
        y=0.995, fontsize=11)
    plt.savefig(out_plot, dpi=130, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()
