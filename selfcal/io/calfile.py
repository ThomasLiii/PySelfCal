"""``CalFile`` — the one reader of every calibration product.

The pipeline writes several HDF5 layouts (documented in PIPELINE.md):

* **v3** (``Calibrator.save_calibration``, the N-pass SKY products): named sky
  blocks ``sky/<name>`` with ``sky_coverage/<name>``, ``sky_fisher/<name>`` and
  (spectral blocks) ``sky_separability/<name>``; per-map offsets
  ``offsets/map_<m>`` with ``offset_coverage/map_<m>``,
  ``offset_coverage_frac/map_<m>`` and ``chunk_maps/map_<m>``; the per-frame
  ``frame_scalar``; ``reproj_list``. The v2 names below exist as hard links.
* **v2**: ``skymap`` (+ ``skymap_line`` for a 2-block sky), the same
  ``offsets/map_<m>`` groups.
* **v1**: a single top-level ``offset``.
* **stitched** cals and **N-pass OFFSET products** carry a subset (no offsets
  / no sky respectively).

``CalFile`` hides those differences: ``sky(name)``, ``offsets`` (a list per
map, expanded per frame), ``frame_scalar``, ``reproj_list`` ... resolve on any
of them, and the analysis scripts, the mosaicker and the N-pass readers all
go through it. Reads are lazy; use it as a context manager or call ``close()``.
"""
from __future__ import annotations

import os

import h5py
import hdf5plugin  # noqa: F401  (Blosc/Zstd-compressed datasets)
import numpy as np

__all__ = ['CalFile', 'open_cal']


def _decode(v):
    return v.decode() if isinstance(v, bytes) else v


class CalFile:
    """A calibration product (``cal_*.h5``) opened for reading."""

    def __init__(self, path):
        self.path = str(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        self._f = h5py.File(self.path, 'r')
        self.attrs = {k: _decode(v) for k, v in self._f.attrs.items()}

    # ---- lifecycle ---------------------------------------------------------------
    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f'CalFile({self.path!r}, schema v{self.schema_version}, sky={self.sky_names}, maps={self.num_maps})'

    # ---- schema ----------------------------------------------------------------------
    @property
    def schema_version(self) -> int:
        f = self._f
        if 'sky' in f:
            return 3
        if 'offsets' in f or 'skymap' in f:
            return 2
        return 1

    @property
    def has_sky(self) -> bool:
        return 'sky' in self._f or 'skymap' in self._f

    @property
    def sky_names(self) -> list[str]:
        f = self._f
        if 'sky' in f:
            if 'sky_components' in f.attrs:
                return [_decode(n) for n in f.attrs['sky_components']]
            return list(f['sky'].keys())
        if 'skymap' not in f:
            return []
        names = ['continuum']
        if 'skymap_line' in f:
            names.append('line')
        return names

    @property
    def num_sky_blocks(self) -> int:
        return int(self.attrs.get('num_sky_blocks', len(self.sky_names)))

    @property
    def ref_shape(self) -> tuple[int, int] | None:
        f = self._f
        if 'sky' in f:
            return tuple(f['sky'][self.sky_names[0]].shape)
        if 'skymap' in f:
            return tuple(f['skymap'].shape)
        return None

    # ---- sky blocks -------------------------------------------------------------------
    def _sky_dataset(self, group_v3, alias_cont, alias_line, which):
        f = self._f
        name = self.sky_names[which] if isinstance(which, int) else which
        if group_v3 in f and name in f[group_v3]:
            return f[group_v3][name]
        names = self.sky_names
        if name == names[0] and alias_cont in f:
            return f[alias_cont]
        if len(names) > 1 and name == names[-1] and alias_line in f:
            return f[alias_line]
        return None

    def sky(self, which=0) -> np.ndarray:
        """The sky map of block ``which`` (a name or an index; 0 = continuum)."""
        ds = self._sky_dataset('sky', 'skymap', 'skymap_line', which)
        if ds is None:
            raise KeyError(f'{self.path}: no sky block {which!r} (has {self.sky_names})')
        return ds[()]

    def sky_coverage(self, which=0) -> np.ndarray | None:
        ds = self._sky_dataset('sky_coverage', 'skymap_coverage', 'skymap_line_coverage', which)
        return None if ds is None else ds[()]

    def sky_fisher(self, which=0) -> np.ndarray | None:
        ds = self._sky_dataset('sky_fisher', 'skymap_fisher', 'skymap_line_fisher', which)
        return None if ds is None else ds[()]

    def sky_separability(self, which) -> np.ndarray | None:
        f = self._f
        name = self.sky_names[which] if isinstance(which, int) else which
        if 'sky_separability' in f and name in f['sky_separability']:
            return f['sky_separability'][name][()]
        return None

    @property
    def line_fisher_threshold(self) -> float | None:
        v = self.attrs.get('line_fisher_threshold')
        return None if v is None else float(v)

    # ---- frames ---------------------------------------------------------------------------
    @property
    def reproj_list(self) -> list[str]:
        f = self._f
        if 'reproj_list' not in f:
            return []
        return [_decode(r) for r in f['reproj_list'][()]]

    @property
    def n_frames(self) -> int:
        f = self._f
        if 'reproj_list' in f:
            return int(f['reproj_list'].shape[0])
        offs = self.offsets
        return int(offs[0].shape[0]) if offs else 0

    @property
    def frame_scalar(self) -> np.ndarray | None:
        f = self._f
        return f['frame_scalar'][()] if 'frame_scalar' in f else None

    @property
    def fit_ok(self) -> np.ndarray | None:
        """Per-frame fit flag of an N-pass OFFSET product (None elsewhere)."""
        f = self._f
        return f['fit_ok'][()] if 'fit_ok' in f else None

    # ---- offset blocks -------------------------------------------------------------------
    @property
    def num_maps(self) -> int:
        f = self._f
        if 'offsets' in f:
            return int(self.attrs.get('num_maps', len(f['offsets'])))
        return 1 if 'offset' in f else 0

    def _per_map(self, group, legacy):
        f = self._f
        if group in f:
            return [f[group][f'map_{m}'][()] for m in range(self.num_maps)]
        if legacy is not None and legacy in f:
            return [f[legacy][()]]
        return []

    @property
    def offsets(self) -> list[np.ndarray]:
        """Per-map ``(n_frames, n_chunks)`` offsets, expanded per frame (the per-frame
        scalar is NOT included; see :meth:`total_offsets`)."""
        return self._per_map('offsets', 'offset')

    @property
    def offset_coverage(self) -> list[np.ndarray]:
        return self._per_map('offset_coverage', 'offset_coverage')

    @property
    def offset_coverage_frac(self) -> list[np.ndarray]:
        return self._per_map('offset_coverage_frac', 'offset_coverage_frac')

    @property
    def chunk_maps(self) -> list[np.ndarray]:
        """Per-map chunk maps stored with the cal (empty for the legacy schemas)."""
        return self._per_map('chunk_maps', None)

    def total_offsets(self) -> list[np.ndarray]:
        """``offsets`` with the per-frame scalar folded into map 0 — the total
        per-(frame, chunk) bias a single-map consumer subtracts."""
        offs = self.offsets
        sc = self.frame_scalar
        if offs and sc is not None:
            offs = list(offs)
            offs[0] = offs[0] + sc[:, np.newaxis]
        return offs

    # ---- summary ------------------------------------------------------------------------------
    def describe(self) -> str:
        lines = [f'{os.path.basename(self.path)}: schema v{self.schema_version}',
                 f'  sky blocks: {self.sky_names} on {self.ref_shape}',
                 f'  offset maps: {self.num_maps}, frames: {self.n_frames}, '
                 f'frame scalar: {"yes" if self.frame_scalar is not None else "no"}']
        extra = {k: v for k, v in self.attrs.items()
                 if k not in ('sky_components',) and not isinstance(v, np.ndarray)}
        if extra:
            lines.append('  attrs: ' + ', '.join(f'{k}={v}' for k, v in sorted(extra.items())))
        return '\n'.join(lines)


def open_cal(path) -> CalFile:
    return CalFile(path)
