"""The transfer gate's check of the quadratic gauge (``sc.Gauge(degree=2)``), which the production recipe no longer
runs: the library's computation (``selfcal.core.gauge``) on the golden UNGAUGED cal and map must reproduce what the
campaign's apply_gauge_q.py (HDEG=2) and gauge_map.py (kept in the workspace) made from them -- the cal's sky term and the mosaic's mean and
clipped-mean maps, byte for byte. No solve: the gauge runs on a copy of the ungauged golden cal in SCRATCH, with the
gauge settings the engine passes for the SPHEREx slope term (map 1, constants map 0, centre 1019.5, half 1020).
Exit 0 iff everything matches.
    python quad_check.py UNGAUGED.h5 UNGAUGED.fits QUAD.h5 QUADMAP.fits FRAMES_DIR REF.fits SCRATCH_DIR"""
import os
import shutil
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from selfcal.core.gauge import apply_to_maps, gauge_cal, read_gauge

cal0, mos0, calq, mosq, frames_dir, ref, scratch = sys.argv[1:8]
ok = True


def same(name, a, b):
    global ok
    a, b = np.asarray(a), np.asarray(b)
    a, b = a.astype(a.dtype.newbyteorder('=')), b.astype(b.dtype.newbyteorder('='))   # FITS data are big-endian
    if a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a.view(np.uint8), b.view(np.uint8)):
        print(f"{name:32s} BYTE-EQUAL")
    else:
        ok = False
        if a.shape == b.shape:
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            print(f"{name:32s} DIFFERS  max|d|={np.nanmax(d):.4g}  n(d!=0)={int(np.count_nonzero(d))}")
        else:
            print(f"{name:32s} DIFFERS  {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")


os.makedirs(scratch, exist_ok=True)
work = os.path.join(scratch, 'quad_check.h5')
shutil.copyfile(cal0, work)
with h5py.File(work, 'r') as f:
    names = [n.decode() if isinstance(n, bytes) else str(n) for n in f['reproj_list'][:]]
frames = [os.path.join(frames_dir, os.path.basename(n)) for n in names]
gauge_cal(work, frames, WCS(fits.getheader(ref)), slope_map=1, constant_map=0, centre=(1019.5, 1019.5), half=1020.0,
          degree=2, position_degree=2, apply='map', geometry='tangent')
with h5py.File(work, 'r') as G, h5py.File(calq, 'r') as Q:
    for k in Q['sky']:
        same(f'sky/{k} (quad gauged)', G['sky'][k][...], Q['sky'][k][...])
    gauge = read_gauge(G)
with fits.open(mos0) as hd:
    maps = {e.lower(): {'data': hd[e].data, 'weight': hd[e + '_WEIGHT'].data} for e in ('MEAN_MAP', 'SC_MEAN_MAP') if e in hd}
    done = apply_to_maps(maps, gauge)
    with fits.open(mosq) as hq:
        for e in ('MEAN_MAP', 'SC_MEAN_MAP'):
            if e in hq:
                same(f'{e} (quad on the map)', maps[e.lower()]['data'], hq[e].data)
if len(done) != 2:
    ok = False
    print(f"the gauge reached {done}, not the mean and clipped-mean maps")
os.remove(work)
print("QUAD GAUGE BYTE-EQUAL" if ok else "QUAD GAUGE DIFFERS")
sys.exit(0 if ok else 1)
