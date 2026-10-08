"""Parallel gzip'd HDF5 dataset writes with byte-identical stored chunks.

``create_dataset(..., compression='gzip')`` compresses every chunk serially
inside the HDF5 filter pipeline — the dominant cost of ``save_calibration``
(~15 GB of maps through zlib on one core). zlib is fully deterministic, so
compressing the chunks ourselves in threads (zlib releases the GIL) and
handing the payloads to ``write_direct_chunk`` produces the same stored
bytes as the serial path: same auto-guessed chunk shape (a function of
shape/dtype only), same level-4 zlib stream per chunk, same zero fill in
edge-chunk padding (verified chunk-for-chunk against h5py's own output).
Only the wall time changes.
"""
import itertools
import zlib

import numpy as np
from concurrent.futures import ThreadPoolExecutor


def create_gzip_dataset_parallel(group, name, data, workers=8, level=4):
    """Drop-in for ``group.create_dataset(name, data=data, compression='gzip')``."""
    data = np.ascontiguousarray(data)
    d = group.create_dataset(name, shape=data.shape, dtype=data.dtype,
                             chunks=True, compression='gzip',
                             compression_opts=level)
    ch = d.chunks
    if ch is None or data.size == 0:
        if data.size:
            d[...] = data
        return d

    origins = list(itertools.product(
        *[range(0, s, c) for s, c in zip(data.shape, ch)]))

    def _compress(origin):
        sl = tuple(slice(o, min(o + c, s))
                   for o, c, s in zip(origin, ch, data.shape))
        blk = data[sl]
        if blk.shape != ch:
            # Edge chunk: HDF5 compresses the FULL chunk buffer with the
            # (default, zero) fill value in the padding — replicate it.
            full = np.zeros(ch, dtype=data.dtype)
            full[tuple(slice(0, e) for e in blk.shape)] = blk
            blk = full
        return origin, zlib.compress(np.ascontiguousarray(blk).tobytes(), level)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        # Compression runs in threads; the (single-threaded) HDF5 writes
        # happen here in deterministic origin order.
        for origin, payload in ex.map(_compress, origins, chunksize=4):
            d.id.write_direct_chunk(origin, payload)
    return d


def create_gzip_dataset_rows(group, name, shape, dtype, rows, workers=8, level=4):
    """:func:`create_gzip_dataset_parallel` for data read a band of rows at a time.

    ``rows(r0, r1)`` returns rows ``r0:r1`` of the data (shape ``(r1 - r0,) + shape[1:]``, dtype
    ``dtype``). The dataset gets the same chunk shape (h5py's guess for ``shape`` and ``dtype``)
    and the same stored chunks, written in the same order, as
    ``create_gzip_dataset_parallel(group, name, data)``; only one band of chunk rows is in memory
    at a time (the snapshots of a solve write their sky maps so)."""
    shape = tuple(int(s) for s in shape)
    dtype = np.dtype(dtype)
    d = group.create_dataset(name, shape=shape, dtype=dtype, chunks=True, compression='gzip',
                             compression_opts=level)
    ch = d.chunks
    size = int(np.prod(shape))
    if ch is None or size == 0:
        if size:
            d[...] = rows(0, shape[0])
        return d

    def _compress(item):
        origin, blk = item
        if blk.shape != ch:
            full = np.zeros(ch, dtype=dtype)          # edge chunk: zero fill, as HDF5 pads it
            full[tuple(slice(0, e) for e in blk.shape)] = blk
            blk = full
        return origin, zlib.compress(np.ascontiguousarray(blk).tobytes(), level)

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for r0 in range(0, shape[0], ch[0]):
            r1 = min(r0 + ch[0], shape[0])
            band = np.ascontiguousarray(rows(r0, r1))
            if band.dtype != dtype or band.shape != (r1 - r0,) + shape[1:]:
                raise ValueError(f"{name}: rows({r0}, {r1}) gave {band.dtype} {band.shape}, not {dtype} "
                                 f"{(r1 - r0,) + shape[1:]}")
            items = []
            for rest in itertools.product(*[range(0, s, c) for s, c in zip(shape[1:], ch[1:])]):
                sl = (slice(0, r1 - r0),) + tuple(slice(o, min(o + c, s)) for o, c, s in zip(rest, ch[1:], shape[1:]))
                items.append(((r0,) + rest, band[sl]))
            for origin, payload in ex.map(_compress, items):
                d.id.write_direct_chunk(origin, payload)
            del band, items
    return d
