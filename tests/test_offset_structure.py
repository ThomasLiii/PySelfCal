"""The generic chunk-axes builders reproduce the SPHEREx builders exactly (pairs, chain
order, stencils, basis descriptors, group edges) on the real stripped chunk maps."""
import os
import sys

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from selfcal.models.offset_structure import (ChunkAxes, adjacency_along, poly_chains_along,   # noqa: E402
                                             poly_basis_along, group_edges_along, fd_stencil)


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)


def test_row_major_axes():
    ax = ChunkAxes.row_major(('subchannel', 'column'), (4, 3), ('y', 'x'))
    assert ax.n_chunks == 12 and ax['subchannel'].size == 4 and ax['column'].size == 3
    assert list(ax['subchannel'].of_chunk) == [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
    assert list(ax['column'].of_chunk) == [0, 1, 2] * 4
    assert ax['subchannel'].scan == 'y' and ax['column'].scan == 'x'


def test_fd_stencil():
    assert list(fd_stencil(0)) == [1, -1]
    assert list(fd_stencil(1)) == [1, -2, 1]
    assert list(fd_stencil(2)) == [1, -3, 3, -1]


def test_grid_map_adjacency_and_chains():
    """A 3x4 (row, col) grid map: adjacency along col within a row, chains along col."""
    from selfcal.geometry.map_helper import make_grid_chunk_map
    cm = make_grid_chunk_map((30, 40), 4)[:, :]          # 4x4 chunks on a 30x40 detector
    n = int(cm.max()) + 1
    ax = ChunkAxes.row_major(('row', 'col'), (4, 4), ('y', 'x'))
    i, j = adjacency_along(cm, ax, 'col')
    row, col = ax['row'].of_chunk, ax['col'].of_chunk
    assert len(i) == 4 * 3 and all(row[i] == row[j]) and all(col[j] - col[i] == 1)
    i, j = adjacency_along(cm, ax, 'row', step=1)
    assert len(i) == 4 * 3 and all(col[i] == col[j])
    chains, st = poly_chains_along(ax, 'col', 1)
    assert chains.shape == (4 * 2, 3) and list(st) == [1, -2, 1]
    assert (chains[:, 0] == np.sort(chains[:, 0])).all()


@pytest.mark.parametrize('det,ncol', [(4, 3), (4, 5)])
def test_reproduces_spherex_builders(det, ncol, monkeypatch):
    from selfcal.instruments.spherex import spherex_utility as su
    from selfcal.pipeline.npass import group_wavelength_edges
    from selfcal import _state
    _state.set_progress(False)
    # The chunk maps come from the shipped LVF fit. The calibration files and the SPHEREx
    # channel table exist only on the processing host, so stand-ins replace them: with a
    # fit given the band-centre map only sets the shape (a blank one), and the band's 18
    # channel edges are every 20th of the fit's 341 subchannel edges. Both give maps
    # identical to the real files' for D1-D6 (checked on the processing host).
    blank = np.zeros((2040, 2040), dtype=np.float32)
    monkeypatch.setattr(su, 'load_calibration', lambda band, calibration_dir=None: (blank, blank))
    edges = np.asarray(su.load_lvf_params(f'lvf_params_D{det}.npy')['wave_edges'])[::20]
    monkeypatch.setattr(su, 'extract_spherex_channel_edges', lambda band, channel_file=None: edges.copy())

    def legacy_poly_basis(det_chunk_map, num_columns, degree, lo, hi, segments=None):
        # the pre-S2 SPHERExInstrument.subchannel_poly_basis, verbatim
        n_chunks = int(det_chunk_map.max()) + 1
        chunk_ids = np.arange(n_chunks)
        pb = {'degree': int(degree), 'num_groups': int(num_columns), 'coord_lo': int(lo), 'coord_hi': int(hi),
              'chunk_coord': chunk_ids // int(num_columns), 'chunk_group': chunk_ids % int(num_columns)}
        if segments:
            pb['segments'] = [(int(a), int(b)) for a, b in segments]
        return pb

    def legacy_edges(det_bc, det_chunk_map, num_columns):
        # the pre-S2 SPHERExInstrument.subchannel_bc_edges, verbatim
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
    for deg in (1, 2):
        if ncol >= deg + 2:
            c1, s1 = su.compute_column_polynomial_chains(cm, ncol, degree=deg)
            c2, s2 = poly_chains_along(axes, 'column', deg)
            assert _same(c1, c2) and _same(s1, s2)
    for deg, lo, hi in ((2, 200, 320), (3, 210, 249), (1, None, None)):
        c1, s1 = su.compute_subchannel_polynomial_chains(nsub, ncol, degree=deg, subch_lo=lo, subch_hi=hi)
        c2, s2 = poly_chains_along(axes, 'subchannel', deg, lo, hi)
        assert _same(c1, c2) and _same(s1, s2)
    for segs in (None, [[200, 260], [261, 320]]):
        p1 = legacy_poly_basis(cm, ncol, 2, 200, 320, segments=segs)
        p2 = poly_basis_along(axes, 'subchannel', 'column', 2, 200, 320, segments=segs)
        assert set(p1) == set(p2)
        for k in p1:
            assert _same(p1[k], p2[k]) if isinstance(p1[k], np.ndarray) else p1[k] == p2[k]
    bc = np.linspace(3.0, 4.0, cm.size).reshape(cm.shape)
    e1 = legacy_edges(bc, cm, ncol)
    e2 = group_edges_along(bc, cm, axes, 'subchannel')
    assert _same(e1, e2)
