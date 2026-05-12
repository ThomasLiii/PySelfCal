"""4-panel D5 Ch3 mosaic comparison: production vs the 3 fixed-cal variants."""
import os
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

MOS_DIR = '/mnt/md124/thomasli/selfcal/outputs/SPHEREx_nep_qr2_det5_6p2arcsec/mosaic/'
FIG_DIR = '/home/thomasli/spherex/selfcal/figures/cc_figure/'

PANELS = [
    ('production',    'mosaic_Detector5_NumSub10_NumCh34_NumCol3_Ch3_damp0p1_reg0p1_outThresh5_sigma2.fits'),
    ('poly_off_fixed', 'mosaic_Detector5_NumSub10_NumCh34_NumCol3_Ch3_baseline_poly_off_fixed.fits'),
    ('poly_k1_fixed',  'mosaic_Detector5_NumSub10_NumCh34_NumCol10_Ch3_baseline_poly_k1_fixed.fits'),
    ('poly_k2_fixed',  'mosaic_Detector5_NumSub10_NumCh34_NumCol10_Ch3_baseline_poly_k2_fixed.fits'),
]


def load_mean(fp):
    with fits.open(fp, memmap=True) as hdul:
        # PipelineWrapper.save_mosaic packs mean_map into hdul[1]
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim == 2:
                return np.asarray(hdu.data, dtype=np.float32)
    raise RuntimeError(f"no 2D image in {fp}")


fig, axes = plt.subplots(2, 2, figsize=(14, 14))
axes = axes.ravel()

for ax, (label, fname) in zip(axes, PANELS):
    fp = os.path.join(MOS_DIR, fname)
    if not os.path.exists(fp):
        ax.set_title(f'{label}\n(missing)')
        ax.axis('off')
        continue
    img = load_mean(fp)
    med = np.nanmedian(img)
    im = ax.imshow(img - med, vmin=-0.01, vmax=0.01, cmap='bwr', origin='lower')
    ax.set_title(f'{label}  (median={med:+.4f})')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    print(f'{label}: shape={img.shape}, median={med:+.5f}, '
          f'std={np.nanstd(img - med):.4e}, '
          f'p1/99={np.nanpercentile(img - med, [1, 99])}')

fig.suptitle('D5 Ch3 mosaic comparison (median-subtracted, ±0.01)', fontsize=14)
fig.tight_layout()

out = os.path.join(FIG_DIR, 'd5_ch3_fixed_vs_production.png')
fig.savefig(out, dpi=300, bbox_inches='tight')
print(f'Wrote {out}')
