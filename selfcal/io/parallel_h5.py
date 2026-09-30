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
