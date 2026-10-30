"""The generic chunk-axes builders reproduce the SPHEREx builders exactly (adjacency pairs, group
edges) on the real stripped chunk maps."""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from selfcal.models.offset_structure import ChunkAxes, adjacency_along, group_edges_along  # noqa: E402


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def test_row_major_axes():
    ax = ChunkAxes.row_major(('subchannel', 'column'), (4, 3), ('y', 'x'))
    assert ax.n_chunks == 12 and ax['subchannel'].size == 4 and ax['column'].size == 3
    assert list(ax['subchannel'].of_chunk) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert list(ax['column'].of_chunk) == [0, 1, 2] * 4
    assert ax['subchannel'].scan == 'y' and ax['column'].scan == 'x'


def test_grid_map_adjacency():
    """A 4x4 (row, col) grid map: adjacency along col within a row, along row within a column."""
    from selfcal.geometry.map_helper import make_grid_chunk_map
    cm = make_grid_chunk_map((30, 40), 4)[:, :]          # 4x4 chunks on a 30x40 detector
    n = int(cm.max()) + 1
    ax = ChunkAxes.row_major(('row', 'col'), (4, 4), ('y', 'x'))
    i, j = adjacency_along(cm, ax, 'col')
    row, col = ax['row'].of_chunk, ax['col'].of_chunk
    assert len(i) == 4 * 3 and all(row[i] == row[j]) and all(col[j] - col[i] == 1)
    i, j = adjacency_along(cm, ax, 'row', step=1)
    assert len(i) == 4 * 3 and all(col[i] == col[j])


@pytest.mark.parametrize('det,ncol', [(4, 3), (4, 5)])
def test_reproduces_spherex_builders(det, ncol, monkeypatch):
    from selfcal.instruments.spherex import spherex_utility as su
    from selfcal.npass.offsets import group_wavelength_edges
    from selfcal import _state
    _state.set_progress(False)
    # The chunk maps come from the shipped LVF fit and the shipped channel table. The
    # calibration files exist only on the processing host; with a fit given, the
    # band-centre map only sets the shape, so a blank one stands in (identical maps for
    # D1-D6, checked on the processing host).
    blank = np.zeros((2040, 2040), dtype=np.float32)
    monkeypatch.setattr(su, 'load_calibration', lambda band, calibration_dir=None: (blank, blank))

    def legacy_edges(det_bc, det_chunk_map, num_columns):
        # SPHERExInstrument.subchannel_bc_edges before the instrument-agnostic engine, verbatim
        n_chunks = int(det_chunk_map.max()) + 1
        return group_wavelength_edges(det_bc, det_chunk_map, np.arange(n_chunks) // int(num_columns))
    lvf = su.load_lvf_params(f'lvf_params_D{det}.npy')
    cm, _, _, _ = su.make_stripped_chunk_map(det, num_subchannels=10, num_channels=34, num_columns=ncol,
                                             oversample_factor=1, lvf_params=lvf)
    nsub = (int(cm.max()) + 1) // ncol
    axes = ChunkAxes.row_major(('subchannel', 'column'), (nsub, ncol), ('y', 'x'))
    a = su.compute_column_adjacency(cm, ncol); g = adjacency_along(cm, axes, 'column')
    assert _same(a[0], g[0]) and _same(a[1], g[1]) and len(g[0]) > 0
    a = su.compute_subchannel_adjacency(cm, ncol); g = adjacency_along(cm, axes, 'subchannel', step=1)
    assert _same(a[0], g[0]) and _same(a[1], g[1]) and len(g[0]) > 0
    bc = np.linspace(3.0, 4.0, cm.size).reshape(cm.shape)
    e1 = legacy_edges(bc, cm, ncol)
    e2 = group_edges_along(bc, cm, axes, 'subchannel')
    assert _same(e1, e2)
