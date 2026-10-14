"""The transfer gate's cal comparison: the gate cal (the library's weighted solve + the quad gauge applied on the
map) against two goldens of the archived workspace implementations -- its offsets, offset coverage, chunk maps,
frame scalars, sky coverage and frame list must equal the UNGAUGED golden's byte for byte (the library leaves the
offsets as solved), and its sky term the GAUGED golden's (the script gauged the sky the same way and moved the
offsets too). Exit 0 iff everything matches.
    python transfer_check.py GATE_CAL.h5 UNGAUGED_GOLDEN.h5 GAUGED_GOLDEN.h5"""
import sys

import h5py
import hdf5plugin  # noqa: F401
import numpy as np

gate, ungauged, gauged = sys.argv[1:4]
ok = True


def same(name, a, b):
    global ok
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype == b.dtype and a.shape == b.shape and np.array_equal(a.view(np.uint8), b.view(np.uint8)):
        print(f"{name:32s} BYTE-EQUAL")
    else:
        ok = False
        if a.shape == b.shape and a.dtype.kind in 'fiu' and b.dtype.kind in 'fiu':
            d = np.abs(a.astype(np.float64) - b.astype(np.float64))
            print(f"{name:32s} DIFFERS  max|d|={np.nanmax(d):.4g}  n(d!=0)={int(np.count_nonzero(d))}")
        else:
            print(f"{name:32s} DIFFERS  {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")


with h5py.File(gate, 'r') as G, h5py.File(ungauged, 'r') as U, h5py.File(gauged, 'r') as Q:
    if 'gauge' not in G or G['gauge'].attrs.get('applied_to') not in ('map', b'map'):
        ok = False
        print("the gate cal carries no gauge applied to the map")
    for grp in ('offsets', 'offset_coverage', 'offset_coverage_frac', 'chunk_maps', 'sky_coverage'):
        for k in U[grp]:
            same(f'{grp}/{k}', G[grp][k][...], U[grp][k][...])
    same('frame_scalar', G['frame_scalar'][...], U['frame_scalar'][...])
    same('reproj_list (basenames)', np.array([str(n).split('/')[-1] for n in G['reproj_list'][...].astype(str)]),
         np.array([str(n).split('/')[-1] for n in U['reproj_list'][...].astype(str)]))
    for k in Q['sky']:
        same(f'sky/{k} (gauged)', G['sky'][k][...], Q['sky'][k][...])
print("TRANSFER CAL BYTE-EQUAL" if ok else "TRANSFER CAL DIFFERS")
sys.exit(0 if ok else 1)
