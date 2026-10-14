"""Apply a post-solve gauge to a finished map instead of through the per-frame offsets: read the polynomial a gauged
cal moved into its sky term (apply_gauge_q.py: plane g and quadratic H about p0 = the mean covered pixel, the cal's
'gauge' attribute) and add the same polynomial -- with the same zero point, minus its mean over the solve's covered
pixels -- to a mosaic made from the UNGAUGED cal. Then mosaic_out - sky(gauged cal) = mosaic - sky(ungauged cal) on
every covered pixel, and no part of the polynomial passes through the offsets. Checks that the gauged and ungauged sky
terms differ by exactly this polynomial.
    python gauge_map.py UNGAUGED_CAL.h5 GAUGED_CAL.h5 MOSAIC_FROM_UNGAUGED.fits OUT.fits"""
import re
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from astropy.io import fits

cal0, calg, mos, out = sys.argv[1:5]
with h5py.File(calg, 'r') as f:
    note = str(f.attrs['gauge']); sky_g = f['sky/continuum'][:].astype(np.float64); cov = f['sky_coverage/continuum'][:]
with h5py.File(cal0, 'r') as f:
    sky_0 = f['sky/continuum'][:].astype(np.float64)
g = np.array([float(v) for v in re.search(r'plane g = \[([^\]]+)\]', note).group(1).split(',')])
Hs = re.search(r'quadratic H = \[\[([^\]]+)\], \[([^\]]+)\]\]', note)
H = (np.array([[float(v) for v in Hs.group(1).split(',')], [float(v) for v in Hs.group(2).split(',')]]) if Hs
     else np.zeros((2, 2)))                     # a linear gauge (apply_gauge.py --no-quadratic): the plane alone
m = cov > 0
pyy, pxx = np.nonzero(m)
p0 = np.array([pxx.mean(), pyy.mean()])


def poly(xx, yy):
    px, py = xx - p0[0], yy - p0[1]
    return g[0] * px + g[1] * py + 0.5 * (H[0, 0] * px ** 2 + 2 * H[0, 1] * px * py + H[1, 1] * py ** 2)


mean_m = poly(pxx.astype(np.float64), pyy.astype(np.float64)).mean()
# check: the gauged sky term = the ungauged one + (poly - its mean over the covered pixels)
sel = (slice(None, None, 7), slice(None, None, 7))
yy, xx = np.mgrid[0:sky_g.shape[0]:7, 0:sky_g.shape[1]:7]
dd = (sky_g[sel] - sky_0[sel]) - (poly(xx, yy) - mean_m)
ok = m[sel] & np.isfinite(dd)
print(f'gauge: g = {g}, H = {H.tolist()}, p0 = {p0}; check sky(gauged) - sky(ungauged) - polynomial: rms {np.std(dd[ok]):.1e}, max {np.abs(dd[ok]).max():.1e} MJy/sr')
with fits.open(mos) as hd:
    mm = np.asarray(hd['MEAN_MAP'].data, dtype=np.float64)
    H_, W_ = mm.shape
    yy, xx = np.mgrid[0:H_, 0:W_]
    add = (poly(xx.astype(np.float64), yy.astype(np.float64)) - mean_m).astype(np.float32)
    for ext in ('MEAN_MAP', 'SC_MEAN_MAP'):      # the clipped mean too, when the coadd made one (the std map is unchanged)
        if ext not in hd:
            continue
        mm = np.asarray(hd[ext].data, dtype=np.float32); w = np.asarray(hd[ext + '_WEIGHT'].data)
        hd[ext].data = np.where(w > 0, mm + add, mm).astype(np.float32)
        hd[ext].header['GAUGE'] = ('post-solve gauge applied to the map', 'gauge_map.py')
        print(f'gauged {ext}')
    hd.writeto(out, overwrite=True)
print('wrote', out)
