"""Tiled-calibration tests: tile geometry + frame assignment (no SPHEREx data).

The Fisher stitch is verified separately against the archived NEP stitched cal
(cache/refactor_gate, real data). Here: make_tile_grid reproduces the NEP
quadrant bboxes, and the frame_select helpers assign synthetic frames correctly.
"""
import tempfile

import h5py
import numpy as np

from selfcal.pipeline.tiled import make_tile_grid, assign_frames, TileSpec
from selfcal.io.frame_select import (load_ref_coords_table, filter_by_center,
                                     compute_overlapping_frames,
                                     compute_overlapping_frames_from_cache)


def test_make_tile_grid_reproduces_nep_quadrants():
    tiles = make_tile_grid((12676, 12672), 2, 2, overlap_px=50,
                           names=['NW', 'NE', 'SW', 'SE'])
    bb = {t.name: t.bbox for t in tiles}
    assert bb['NW'] == (0, 6363, 0, 6361)
    assert bb['NE'] == (0, 6363, 6311, 12672)
    assert bb['SW'] == (6313, 12676, 0, 6361)
    assert bb['SE'] == (6313, 12676, 6311, 12672)


def test_make_tile_grid_default_names_and_no_overlap():
    tiles = make_tile_grid((100, 80), 2, 2, overlap_px=0)
    assert [t.name for t in tiles] == ['r0c0', 'r0c1', 'r1c0', 'r1c1']
    # no overlap -> tiles tile the grid exactly
    assert tiles[0].bbox == (0, 50, 0, 40)
    assert tiles[3].bbox == (50, 100, 40, 80)


def _write_frame(path, ref_coords):
    with h5py.File(path, 'w') as f:
        f.attrs['ref_coords'] = np.array(ref_coords, dtype=np.int32)


def test_frame_select_center_and_overlap():
    tmp = tempfile.mkdtemp()
    # frames: A center (50,50) small; B center (150,150); C straddles the seam.
    layout = {
        'A': (40, 60, 40, 60),     # center (50,50)
        'B': (140, 160, 140, 160), # center (150,150)
        'C': (90, 130, 90, 130),   # center (110,110)
    }
    files = []
    for name, rc in layout.items():
        p = f"{tmp}/{name}.h5"
        _write_frame(p, rc)
        files.append(p)
    rc_table = load_ref_coords_table(files)
    assert rc_table.shape == (3, 4)
    assert list(rc_table[0]) == [40, 60, 40, 60]

    # center filter: tile [0,100)x[0,100) -> only A (center 50,50)
    kept, idx = filter_by_center(files, rc_table, (0, 100, 0, 100), halo=0)
    assert [f.split('/')[-1] for f in kept] == ['A.h5']
    # overlap filter: same tile -> A and C (C footprint 90:130 overlaps [0,100))
    kept_o, _ = compute_overlapping_frames_from_cache(files, rc_table, (0, 100, 0, 100))
    assert sorted(f.split('/')[-1] for f in kept_o) == ['A.h5', 'C.h5']
    # the file-reading variant agrees with the cached variant
    kept_f, _ = compute_overlapping_frames(files, (0, 100, 0, 100))
    assert sorted(kept_f) == sorted(kept_o)


def test_assign_frames_partitions_by_center():
    tmp = tempfile.mkdtemp()
    files = []
    for i, rc in enumerate([(10, 30, 10, 30), (10, 30, 70, 90),
                            (70, 90, 10, 30), (70, 90, 70, 90)]):
        p = f"{tmp}/f{i}.h5"
        _write_frame(p, rc)
        files.append(p)
    tiles = make_tile_grid((100, 100), 2, 2, overlap_px=0)
    asg = assign_frames(files, tiles, frame_filter='center')
    # each quadrant frame lands in exactly one tile
    counts = {name: len(fl) for name, (fl, idx) in asg.items()}
    assert sorted(counts.values()) == [1, 1, 1, 1]
    assert sum(counts.values()) == 4


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")


if __name__ == '__main__':
    _run_all()
