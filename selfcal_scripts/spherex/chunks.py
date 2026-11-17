"""The partitions of a SPHEREx detector a model may solve on, as chunk-map functions.

A recipe names them in its model, ``sc.Model(chunk_maps={'subchannel': subchannel_map(num_sub=10, num_ch=17,
num_col=1), 'detector_column': sc.ChunkMap(detector_columns)})``; the engine evaluates each on the detector
geometry of ``sc.SPHEREx(d)`` (:func:`subchannels` reads its LVF arc fit from ``geometry.extra``). The first map
is the primary one: the offset terms' default, the jobs' window (``jobs.channel(c)`` names its subchannels) and
the mosaic's; its tag joins the product name, ``cal_Detector<d>_NumSub10_NumCh17_NumCol1_Ch9_<recipe>.h5``.
"""
from functools import partial

import numpy as np

import selfcal as sc
from selfcal.instruments.spherex.adapter import make_column_chunk_map, make_readout_chunk_map
from selfcal.instruments.spherex.spherex_utility import make_spherex_stripped_offset_map, make_stripped_chunk_map
from selfcal.models.offset_structure import ChunkAxes

__all__ = ['subchannels', 'subchannel_map', 'columns', 'column_map', 'detector_columns', 'readouts', 'pixels',
           'pixel_map', 'arc_renderer']


def arc_renderer(chunks, jobgeom):
    """The mosaic's renderer of the subchannel map's offsets for one job: a mean-preserving spline
    along the arcs (:func:`~selfcal.instruments.spherex.spherex_utility.make_spherex_stripped_offset_map`)
    over the job's strict subchannels, filling a frame's zeroed chunks from its others."""
    x = chunks.extra
    renderer = partial(make_spherex_stripped_offset_map, chunk_valid_mask=jobgeom.chunk_valid_strict,
                       lvf_params=x['lvf_params'], r_edges=x['r_edges'], x_edges=x['x_edges'],
                       tot_subchannels=x['num_sub'] * x['num_ch'] + 2, num_columns=x['num_col'], fill_invalid=True)
    renderer.fills_invalid = True
    return renderer


def subchannels(geometry, num_sub=10, num_ch=17, num_col=1):
    """The LVF arcs cut into ``num_ch`` channels of ``num_sub`` subchannels each (the channel table
    sets the channel edges) and into ``num_col`` columns: chunk id ``subchannel * num_col + column``, the
    axes ``subchannel`` (changing along y, across the arcs; the spectral axis) and ``column`` (along x; the
    group axis), the chunks smoothed along ``column`` by default. Subchannels 0 and ``num_sub * num_ch + 1``
    lie beyond the band's edges. The map on the mosaic's grid is traced at its resolution
    (:func:`~selfcal.instruments.spherex.spherex_utility.make_stripped_chunk_map`). ``extra`` carries the arc
    radii and column edges of the partition (``r_edges``, ``x_edges``) for the terms and the renderer."""
    det, lvf_params = geometry.extra['detector'], geometry.extra['lvf_params']
    grid_map, _, _, _ = make_stripped_chunk_map(
        det, num_subchannels=num_sub, num_channels=num_ch, num_columns=num_col,
        oversample_factor=geometry.oversample, lvf_params=lvf_params)
    det_map, _, r_edges, x_edges = make_stripped_chunk_map(
        det, num_subchannels=num_sub, num_channels=num_ch, num_columns=num_col,
        oversample_factor=1, lvf_params=lvf_params)
    n_sub = (int(det_map.max()) + 1) // int(num_col)      # == num_sub * num_ch + 2 (padding subchannels)
    axes = ChunkAxes.row_major(('subchannel', 'column'), (n_sub, int(num_col)), ('y', 'x'))
    return sc.Chunks(det_map, axes, grid=grid_map, adjacency_axes=('column',), spectral_axis='subchannel',
                     renderers={'arcs': arc_renderer}, render='arcs',
                     extra={'lvf_params': lvf_params, 'r_edges': r_edges, 'x_edges': x_edges,
                            'num_sub': int(num_sub), 'num_ch': int(num_ch), 'num_col': int(num_col)})


def subchannel_map(num_sub=10, num_ch=17, num_col=1) -> sc.ChunkMap:
    """:func:`subchannels` as a model's chunk map, tagged ``NumSub<num_sub>_NumCh<num_ch>_NumCol<num_col>``."""
    return sc.ChunkMap(subchannels, tag=f'NumSub{num_sub}_NumCh{num_ch}_NumCol{num_col}',
                       num_sub=int(num_sub), num_ch=int(num_ch), num_col=int(num_col))


def columns(geometry, num_sub=10, num_ch=17, num_col=1):
    """The ``num_col`` columns of the subchannel partition (:func:`subchannels` with the same arguments), one chunk
    each: chunk id = the pixel's column index of that partition (``subchannel id % num_col``), -1 where the
    partition has no subchannel; the axis ``column``, along x. An offset term on it per frame varies across the
    columns and not across the arcs: the map of the multi-line fits' polynomial of the arc position per column."""
    det, lvf_params = geometry.extra['detector'], geometry.extra['lvf_params']
    det_map, _, _, _ = make_stripped_chunk_map(
        det, num_subchannels=num_sub, num_channels=num_ch, num_columns=num_col,
        oversample_factor=1, lvf_params=lvf_params)
    col = np.where(det_map >= 0, det_map % int(num_col), -1).astype(det_map.dtype)
    return sc.Chunks(col, ChunkAxes.row_major(('column',), (int(num_col),), ('x',)))


def column_map(num_sub=10, num_ch=17, num_col=1) -> sc.ChunkMap:
    """:func:`columns` as a model's chunk map (not tagged: never the primary map)."""
    return sc.ChunkMap(columns, num_sub=int(num_sub), num_ch=int(num_ch), num_col=int(num_col))


def detector_columns(geometry):
    """One chunk per detector column (chunk id = the column index; the axis ``detector_column``, along
    x). An offset term on it shared by every frame is a detector-fixed column pattern."""
    det, n = make_column_chunk_map(tuple(geometry.shape))
    return sc.Chunks(det, ChunkAxes.row_major(('detector_column',), (n,), ('x',)))


def readouts(geometry):
    """One chunk per H2RG readout channel (64 columns; chunk 0 the reference columns; the axis
    ``readout``, along x)."""
    det, n = make_readout_chunk_map(tuple(geometry.shape))
    return sc.Chunks(det, ChunkAxes.row_major(('readout',), (n,), ('x',)))


def pixels(geometry, first, last, num_sub=10, num_ch=17, num_col=1):
    """One chunk per detector pixel of the subchannels ``first`` to ``last`` (inclusive) of :func:`subchannels`
    (``num_sub`` x ``num_ch`` x ``num_col``); every other pixel has none. Chunk ids run over the selected pixels in
    row-major order (one axis, ``pixel``). An offset term on it shared by every frame is a detector-fixed pattern
    of any shape over those subchannels."""
    det_map, _, _, _ = make_stripped_chunk_map(
        geometry.extra['detector'], num_subchannels=num_sub, num_channels=num_ch, num_columns=num_col,
        oversample_factor=1, lvf_params=geometry.extra['lvf_params'])
    sub = np.asarray(det_map) // int(num_col)
    inside = (det_map >= 0) & (sub >= int(first)) & (sub <= int(last))
    ids = np.full(det_map.shape, -1, dtype=np.int32)
    ids[inside] = np.arange(int(inside.sum()), dtype=np.int32)
    return sc.Chunks(ids, ChunkAxes.row_major(('pixel',), (int(inside.sum()),), ('both',)))


def pixel_map(first, last, num_sub=10, num_ch=17, num_col=1) -> sc.ChunkMap:
    """:func:`pixels` as a model's chunk map."""
    return sc.ChunkMap(pixels, first=int(first), last=int(last), num_sub=int(num_sub), num_ch=int(num_ch),
                       num_col=int(num_col))
