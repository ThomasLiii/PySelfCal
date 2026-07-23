"""Re-plot the all-detector anchor-spectrum summary written by
diag_joint_zodi_spectrum.py (figures/zodi_anchor/
zodi_spectrum_all_detectors.npz), with y-axis ranges clipped to exclude
the D1 airglow-contaminated channels (He I 1083 nm, OI 8446 nm) so the
bulk continuum is readable.

Reads:  figures/zodi_anchor/zodi_spectrum_all_detectors.npz
Writes: figures/zodi_anchor/zodi_spectrum_all_detectors_zoomed.png
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


DET_COLORS = {
    1: 'tab:purple',
    2: 'tab:orange',
    3: 'tab:green',
    4: 'tab:blue',
    5: 'tab:red',
}
DET_BOUNDARIES_UM = (1.10, 1.65, 2.42, 3.81)
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

# Wavelength windows around D1 airglow where channels should be masked
# from the y-range computation. Channels in these windows still get
# plotted (as out-of-range markers at the y-axis edges), but they don't
# squash the scale.
AIRGLOW_WINDOWS_UM = [
    (0.820, 0.870),   # OI 8446
    (1.045, 1.130),   # He I 1083 / Pa gamma
]


def in_any_window(wl, windows):
    m = np.zeros_like(wl, dtype=bool)
    for lo, hi in windows:
        m |= (wl >= lo) & (wl <= hi)
    return m


def yrange_clean(per_det_values, per_det_wl, pad=0.08):
    """Compute (ymin, ymax) across all detectors, excluding samples whose
    wavelength is inside any AIRGLOW_WINDOWS_UM window. Pads by `pad` of
    the range on each side."""
    vals = []
    for v, w in zip(per_det_values, per_det_wl):
        v = np.asarray(v, dtype=float)
        w = np.asarray(w, dtype=float)
        keep = np.isfinite(v) & ~in_any_window(w, AIRGLOW_WINDOWS_UM)
        vals.append(v[keep])
    vals = np.concatenate(vals) if vals else np.array([])
    if vals.size == 0:
        return None
    lo = float(np.min(vals))
    hi = float(np.max(vals))
    rng = hi - lo
    if rng <= 0:
        rng = max(abs(hi), 1e-12)
    return lo - pad * rng, hi + pad * rng


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


def main():
    in_npz = 'figures/zodi_anchor/zodi_spectrum_all_detectors.npz'
    out_png = 'figures/zodi_anchor/zodi_spectrum_all_detectors_zoomed.png'
    z = np.load(in_npz, allow_pickle=False)
    detectors = sorted(int(d) for d in z['detectors'])
    per_det = []
    for d in detectors:
        per_det.append(dict(
            detector=d,
            wl=z[f'D{d}_wavelength_um'],
            mean_full_dc=z[f'D{d}_mean_full_dc'],
            mean_pred=z[f'D{d}_mean_pred'],
            slope=z[f'D{d}_slope'],
            C=z[f'D{d}_C'],
            pearson_r=z[f'D{d}_pearson_r'],
            resid_std_mMJy=z[f'D{d}_resid_std_mMJy'],
            fit_zodi=z[f'D{d}_slope'] * z[f'D{d}_mean_pred'],
        ))

    fig, axes = plt.subplots(5, 1, figsize=(12, 16), sharex=True)

    # (a) Three series stacked. Compute a robust y-range from the
    #     non-airglow channels of ALL three series.
    ax = axes[0]
    series_vals, series_wls = [], []
    for pd_ in per_det:
        series_vals.extend([pd_['mean_full_dc'], pd_['mean_pred'],
                            pd_['fit_zodi']])
        series_wls.extend([pd_['wl'], pd_['wl'], pd_['wl']])
    yr = yrange_clean(series_vals, series_wls)
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
    if yr is not None:
        ax.set_ylim(*yr)
    ax.set_ylabel('MJy/sr')
    ax.set_title('(a) Per-channel mean DC: solid=mean(full_DC), '
                 'dashed=mean(zp), dotted=slope*mean(zp)  '
                 '[D1 airglow channels off-scale]')
    ax.legend(loc='best', fontsize=6, ncol=5)
    ax.grid(alpha=0.3)
    _draw_det_boundaries(ax)
    _draw_feature_lines(ax)

    # (b) C
    ax = axes[1]
    yr = yrange_clean([pd_['C'] for pd_ in per_det],
                      [pd_['wl'] for pd_ in per_det])
    ax.axhline(0.0, color='k', lw=0.5, alpha=0.5)
    for pd_ in per_det:
        det = pd_['detector']
        c = DET_COLORS.get(det, 'k')
        ax.plot(pd_['wl'], pd_['C'], '-^', ms=4, lw=1.0, c=c,
                label=f'D{det}')
    if yr is not None:
        ax.set_ylim(*yr)
    ax.set_ylabel('C  (MJy/sr)')
    ax.set_title('(b) Anchor constant C  (non-zodi DC) '
                 '[D1 airglow channels off-scale]')
    ax.legend(loc='best', fontsize=8, ncol=5)
    ax.grid(alpha=0.3)
    _draw_det_boundaries(ax)
    _draw_feature_lines(ax)

    # (c) slope
    ax = axes[2]
    yr = yrange_clean([pd_['slope'] for pd_ in per_det],
                      [pd_['wl'] for pd_ in per_det])
    ax.axhline(1.0, color='k', lw=0.7, alpha=0.5)
    for pd_ in per_det:
        det = pd_['detector']
        c = DET_COLORS.get(det, 'k')
        ax.plot(pd_['wl'], pd_['slope'], '-o', ms=4, lw=1.0, c=c,
                label=f'D{det}')
    if yr is not None:
        ax.set_ylim(*yr)
    ax.set_ylabel('slope')
    ax.set_title('(c) Fitted slope per channel  (=1 if zodipy captures '
                 'temporal shape) [D1 airglow channels off-scale]')
    ax.legend(loc='best', fontsize=8, ncol=5)
    ax.grid(alpha=0.3)
    _draw_det_boundaries(ax)
    _draw_feature_lines(ax)

    # (d) Pearson r: fixed zoom ylim(0.85, 1.005) so the airglow
    # channels' depressed r values fall off-scale instead of setting
    # the axis range.
    ax = axes[3]
    ax.axhline(1.0, color='k', lw=0.5, alpha=0.4)
    ax.axhline(0.0, color='k', lw=0.5, alpha=0.4)
    for pd_ in per_det:
        det = pd_['detector']
        c = DET_COLORS.get(det, 'k')
        ax.plot(pd_['wl'], pd_['pearson_r'], '-o', ms=4, lw=1.0, c=c,
                label=f'D{det}')
    ax.set_ylim(0.85, 1.005)
    ax.set_ylabel('Pearson r')
    ax.set_title('(d) Per-frame correlation of full_DC vs zodi_pred '
                 '[zoomed to 0.85-1.0]')
    ax.legend(loc='lower left', fontsize=8, ncol=5)
    ax.grid(alpha=0.3)
    _draw_det_boundaries(ax)
    _draw_feature_lines(ax)

    # (e) residual std (mMJy/sr)
    ax = axes[4]
    yr = yrange_clean([pd_['resid_std_mMJy'] for pd_ in per_det],
                      [pd_['wl'] for pd_ in per_det])
    for pd_ in per_det:
        det = pd_['detector']
        c = DET_COLORS.get(det, 'k')
        ax.plot(pd_['wl'], pd_['resid_std_mMJy'], '-o', ms=4, lw=1.0, c=c,
                label=f'D{det}')
    if yr is not None:
        # Floor at 0 and add a bit more headroom for feature labels
        ax.set_ylim(max(0.0, yr[0]), yr[1] * 1.18)
    ax.set_ylabel('resid std  (mMJy/sr)')
    ax.set_title('(e) Inlier residual std per channel '
                 '[D1 airglow channels off-scale]')
    ax.set_xlabel('Channel mean wavelength (um)')
    ax.legend(loc='best', fontsize=8, ncol=5)
    ax.grid(alpha=0.3)
    _draw_det_boundaries(ax)
    label_y = ax.get_ylim()[1] / 1.18
    _draw_feature_lines(ax, label=True, label_y=label_y)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close(fig)
    print(f"Saved {out_png}")


if __name__ == '__main__':
    main()
