"""Stitch whatever NEPprod tile cals exist so far -> a PARTIAL stitched cal.

Sanity check during the long production run (and an exercise of the new
multi-block `_stitch_multiblock` path on real data). Safe to run while the
production is solving another tile: read-only on the finished tile cals, and
writes to a distinct _PARTIAL_<tiles> filename that the run never touches.

Usage: python stitch_partial.py [tag]     (default tag: the tile list)
"""
import os
import sys
import glob
import re

import numpy as np  # noqa: F401  (h5py/np import order)
from selfcal.pipeline.tiled import stitch

CAL_DIR = ("/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D4_6p2arcsec/"
           "calibration")
BASE = "cal_Detector4_NumSub10_NumCh34_NumCol3_Multiline3_multiline3_NEPprod"
TAIL = "_iter300_polybasisD2noortho_NumCol3_outThresh5_sigma2"
REF_SHAPE = (12676, 12672)

paths = sorted(glob.glob(os.path.join(CAL_DIR, f"{BASE}_T*{TAIL}.h5")))
tiles = [re.search(r"NEPprod_(T\d+)_", p).group(1) for p in paths]
if not paths:
    sys.exit("no tile cals found yet")
tag = sys.argv[1] if len(sys.argv) > 1 else f"{tiles[0]}-{tiles[-1]}"
out = os.path.join(CAL_DIR, f"{BASE}_PARTIAL_{tag}{TAIL}.h5")
print(f"stitching {len(paths)} tiles {tiles} ->\n  {out}", flush=True)
stitch(paths, out, ref_shape=REF_SHAPE)
