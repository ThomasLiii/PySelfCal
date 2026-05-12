"""Render per-variant mean detector-frame offset for D5 Ch3 fixed calibrations.

Total offset at detector pixel p (averaged across frames):
    total[p] = mean_f(scalar[f]) + sum_k chunk_to_det_k(mean_f(offsets_k[f, :]))[p]

Plots a single row of 3 panels (poly_off_fixed / poly_k1_fixed / poly_k2_fixed)
and a 4-panel breakdown for poly_k2_fixed (scalar / map0 / map1 / sum).
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

CAL_DIR = '/mnt/md124/thomasli/selfcal/outputs/SPHEREx_nep_qr2_det5_6p2arcsec/calibration/'
FIG_DIR = '/home/thomasli/spherex/selfcal/figures/cc_figure/'

VARIANTS = [
    ('poly_off_fixed', 'cal_Detector5_NumSub10_NumCh34_NumCol3_Ch3_baseline_poly_off_fixed.h5'),
    ('poly_k1_fixed',  'cal_Detector5_NumSub10_NumCh34_NumCol10_Ch3_baseline_poly_k1_fixed.h5'),
    ('poly_k2_fixed',  'cal_Detector5_NumSub10_NumCh34_NumCol10_Ch3_baseline_poly_k2_fixed.h5'),
]


def total_mean_det(fp):
    """Return (scalar_mean, total_det_map, components_dict)."""
    with h5py.File(fp, 'r') as f:
        scalar_mean = float(np.mean(f['frame_scalar'][:]))
        num_maps = int(f.attrs['num_maps'])
        per_map = []  # list of (chunk_map, mean_per_chunk)
        for k in range(num_maps):
            chunk_map = f[f'chunk_maps/map_{k}'][:]
            off = f[f'offsets/map_{k}'][:]
            per_map.append((chunk_map, off.mean(axis=0)))
    total = np.full_like(per_map[0][0], scalar_mean, dtype=np.float64)
    comps = {'scalar': np.full_like(total, scalar_mean)}
    for k, (cm, mpc) in enumerate(per_map):
        comp = mpc[cm]
        total = total + comp
        comps[f'map_{k}'] = comp
    return scalar_mean, total, comps


def panel(ax, img, title, vmin=-1.5e-3, vmax=1.5e-3):
    med = np.nanmedian(img)
    im = ax.imshow(img - med, vmin=vmin, vmax=vmax, cmap='bwr', origin='lower')
    ax.set_title(f'{title}\nmed={med:+.4f}, std={np.nanstd(img - med):.2e}', fontsize=10)
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return im


# 3-panel total-offset row (one panel per variant) ---------------------------
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5))
totals = {}
all_comps = {}
for ax, (label, fname) in zip(axes, VARIANTS):
    fp = os.path.join(CAL_DIR, fname)
    sm, total, comps = total_mean_det(fp)
    totals[label] = total
    all_comps[label] = comps
    panel(ax, total, f'{label}  (scalar mean={sm:+.4f})')
    print(f'{label}: scalar_mean={sm:+.4f}, total std={np.std(total):.3e}, '
          f'p1/99={np.percentile(total, [1, 99])}')
fig.suptitle('D5 Ch3 mean detector-frame offset (median-subtracted, ±1.5e-3)',
             fontsize=13)
fig.tight_layout()
out = os.path.join(FIG_DIR, 'd5_ch3_det_offset_3way.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f'Wrote {out}')

# 4-panel K=2 component breakdown -------------------------------------------
k2 = all_comps['poly_k2_fixed']
fig, axes = plt.subplots(2, 2, figsize=(11, 11))
panel(axes[0, 0], k2['scalar'], 'scalar (broadcast)')
panel(axes[0, 1], k2['map_0'], 'map_0 (LVF chunk map)')
panel(axes[1, 0], k2['map_1'], 'map_1 (32-stripe det-fixed)')
panel(axes[1, 1], totals['poly_k2_fixed'], 'total = scalar + map_0 + map_1')
fig.suptitle('D5 Ch3 poly_k2_fixed components (median-subtracted, ±1.5e-3)',
             fontsize=13)
fig.tight_layout()
out2 = os.path.join(FIG_DIR, 'd5_ch3_polyk2_fixed_components.png')
fig.savefig(out2, dpi=300, bbox_inches='tight')
print(f'Wrote {out2}')
