"""Compare two products: byte-identical, equal values, or different, and why.

``sc.compare(a, b)`` (``selfcal compare A B``) takes two cal files (``.h5``) or two mosaics
(``.fits``). The verdict is ``"identical"`` (the same bytes), ``"equal values"`` (every dataset
or map holds the same values; the files differ in layout, compression or metadata) or
``"different"``, with each differing dataset and its largest difference. When both products have
sidecars (:mod:`selfcal.run.products`), the differences between their inputs say why.
"""
from __future__ import annotations

import filecmp
import os
from dataclasses import dataclass, field

import numpy as np

__all__ = ['Comparison', 'compare']


@dataclass
class Comparison:
    """The result of :func:`compare`: ``verdict``, the per-dataset ``differences`` (``(name,
    what)``), the datasets only one side has, and the differences of the two products' inputs."""
    a: str
    b: str
    verdict: str
    differences: list = field(default_factory=list)
    only_a: list = field(default_factory=list)
    only_b: list = field(default_factory=list)
    inputs: list | None = None

    def __str__(self):
        lines = [f"{self.verdict.upper()}: {os.path.basename(self.a)} vs {os.path.basename(self.b)}"]
        for name, what in self.differences[:20]:
            lines.append(f"  {name}: {what}")
        if len(self.differences) > 20:
            lines.append(f"  ... {len(self.differences) - 20} more")
        for side, names in (('only in the first', self.only_a), ('only in the second', self.only_b)):
            if names:
                lines.append(f"  {side}: {', '.join(names[:10])}" + (' ...' if len(names) > 10 else ''))
        if self.inputs is not None:
            lines.append('  inputs: ' + ('the same' if not self.inputs else ''))
            lines += [f"    {d}" for d in self.inputs[:20]]
        return '\n'.join(lines)


def _kind(path) -> str:
    """``"hdf5"``, ``"fits"`` or ``"other"``, from the file's first bytes (not its name)."""
    with open(path, 'rb') as f:
        head = f.read(8)
    if head == b'\x89HDF\r\n\x1a\n':
        return 'hdf5'
    if head.startswith(b'SIMPLE  '):
        return 'fits'
    return 'gzip' if head[:2] == b'\x1f\x8b' else 'other'


def _array_difference(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.shape != y.shape or x.dtype != y.dtype:
        return f"shape/dtype {x.shape} {x.dtype} vs {y.shape} {y.dtype}"
    if x.tobytes() == y.tobytes():
        return None
    if x.dtype.kind in 'fc':
        with np.errstate(invalid='ignore'):
            both = np.isfinite(x) & np.isfinite(y)
            # NaN on one side only, and an infinity on either side not matched by the same one
            nan_mismatch = int((np.isnan(x) != np.isnan(y)).sum()
                               + ((np.isinf(x) | np.isinf(y)) & ~(x == y) & ~(np.isnan(x) | np.isnan(y))).sum())
            d = np.abs(x[both].astype(np.float64) - y[both].astype(np.float64))
            scale = np.nanmax(np.abs(y[both])) if both.any() else 0.0
        if not d.size or d.max() == 0:
            return f"{nan_mismatch} NaN positions differ" if nan_mismatch else "same values, different bytes"
        return (f"max |a - b| = {d.max():.3g} ({d.max() / scale:.2g} of max |b|), {int((d > 0).sum())} of "
                f"{d.size} values" + (f", {nan_mismatch} NaN positions" if nan_mismatch else ''))
    return f"{int((x != y).sum())} of {x.size} values differ"


def _h5_items(path):
    import h5py
    out = {}
    with h5py.File(path, 'r') as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                out[name] = obj[()]
        f.visititems(visit)
        out['/attrs'] = {k: np.asarray(v).tolist() for k, v in f.attrs.items()}
    return out


def _fits_items(path):
    from astropy.io import fits
    out = {}
    with fits.open(path) as h:
        for i, hdu in enumerate(h):
            name = hdu.name or f'HDU{i}'
            if hdu.data is not None:
                out[name] = np.asarray(hdu.data)
    return out


def compare(a, b) -> Comparison:
    """Compare the products ``a`` and ``b`` (see the module docstring)."""
    a, b = os.fspath(a), os.fspath(b)
    for p in (a, b):
        if not os.path.exists(p):
            raise FileNotFoundError(p)
    inputs = None
    from .products import diff_inputs, read_sidecar
    sa, sb = read_sidecar(a), read_sidecar(b)
    if sa is not None and sb is not None:
        inputs = diff_inputs(sa.get('inputs', {}), sb.get('inputs', {}))
    if filecmp.cmp(a, b, shallow=False):
        return Comparison(a, b, 'identical', inputs=inputs)
    kinds = _kind(a), _kind(b)
    if kinds[0] != kinds[1]:
        return Comparison(a, b, 'different', differences=[('file', f'{kinds[0]} vs {kinds[1]}')], inputs=inputs)
    if kinds[0] not in ('hdf5', 'fits'):
        raise ValueError(f"compare: {a} is neither an HDF5 nor a FITS file")
    reader = _fits_items if kinds[0] == 'fits' else _h5_items
    ia, ib = reader(a), reader(b)
    diffs = []
    for name in sorted(set(ia) & set(ib)):
        if name == '/attrs':
            if ia[name] != ib[name]:
                keys = sorted(k for k in set(ia[name]) | set(ib[name]) if ia[name].get(k) != ib[name].get(k))
                diffs.append(('attributes', f"differ: {', '.join(keys)}"))
            continue
        what = _array_difference(ia[name], ib[name])
        if what is not None and what != "same values, different bytes":
            diffs.append((name, what))
    only_a, only_b = sorted(set(ia) - set(ib)), sorted(set(ib) - set(ia))
    verdict = 'different' if diffs or only_a or only_b else 'equal values'
    return Comparison(a, b, verdict, diffs, only_a, only_b, inputs)
