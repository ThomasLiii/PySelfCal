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
        """Open the ``cal_*.h5`` file at ``path`` (a ``str`` or path-like) read-only.

        ``FileNotFoundError`` is raised when ``path`` is not an existing file. The
        instance keeps ``path`` as a ``str`` and ``attrs``, a dict of the file's root
        attributes with ``bytes`` values decoded to ``str``. The file stays open until
        :meth:`close` or the end of a ``with`` block. Apart from ``attrs`` nothing is
        cached: every property access reads the file again.
        """
        self.path = str(path)
        if not os.path.isfile(self.path):
            raise FileNotFoundError(self.path)
        self._f = h5py.File(self.path, 'r')
        self.attrs = {k: _decode(v) for k, v in self._f.attrs.items()}

    # ---- lifecycle ---------------------------------------------------------------
    def close(self):
        """Close the HDF5 file; calling it again does nothing."""
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
        """Layout version (3, 2 or 1), inferred from the datasets the file holds.

        3 if there is a ``sky`` group, 2 if there is an ``offsets`` group or a
        ``skymap`` dataset, else 1; the ``schema_version`` attribute is not read.
        """
        f = self._f
        if 'sky' in f:
            return 3
        if 'offsets' in f or 'skymap' in f:
            return 2
        return 1

    @property
    def has_sky(self) -> bool:
        """Whether the file holds a sky map (a ``sky`` group or a ``skymap`` dataset)."""
        return 'sky' in self._f or 'skymap' in self._f

    @property
    def sky_names(self) -> list[str]:
        """Names of the sky blocks in block order; block 0 is the continuum.

        v3: the ``sky_components`` attribute, else the keys of the ``sky`` group.
        v2: ``['continuum']``, plus ``'line'`` when ``skymap_line`` exists. ``[]``
        when the file holds no sky.
        """
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
        """Number of sky blocks: the ``num_sky_blocks`` attribute, else ``len(sky_names)``."""
        return int(self.attrs.get('num_sky_blocks', len(self.sky_names)))

    @property
    def ref_shape(self) -> tuple[int, int] | None:
        """Reference-grid shape ``(ref_h, ref_w)`` of the sky maps; None when there is no sky."""
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
        """The number of observations of each reference pixel in sky block ``which``.

        A ``ref_shape`` array, or None when the file stores no coverage for the block.
        """
        ds = self._sky_dataset('sky_coverage', 'skymap_coverage', 'skymap_line_coverage', which)
        return None if ds is None else ds[()]

    def sky_fisher(self, which=0) -> np.ndarray | None:
        """The Fisher information of each reference pixel in sky block ``which``.

        The sum over the pixel's observations of the squared weighted coefficient
        (``Σ w² c²``; ``c = 1`` for the continuum), as a ``ref_shape`` array, or None
        when the file stores none for the block.
        """
        ds = self._sky_dataset('sky_fisher', 'skymap_fisher', 'skymap_line_fisher', which)
        return None if ds is None else ds[()]

    def sky_separability(self, which) -> np.ndarray | None:
        """The separability ``I_P`` of spectral sky block ``which``, or None when absent.

        ``I_P`` is the block's Fisher information left after the other sky blocks are
        profiled out of each pixel (a Schur complement). It measures the diversity of
        the coefficients a pixel is observed with: a pixel seen many times at one
        wavelength has a large Fisher information but ``I_P = 0``, so line maps are
        masked on ``I_P``. Only spectral blocks have one; there is no v2 alias.
        """
        f = self._f
        name = self.sky_names[which] if isinstance(which, int) else which
        if 'sky_separability' in f and name in f['sky_separability']:
            return f['sky_separability'][name][()]
        return None

    @property
    def line_fisher_threshold(self) -> float | None:
        """The recommended read-time Fisher threshold for the line map, or None.

        The ``line_fisher_threshold`` root attribute. The stored line maps are raw;
        :func:`~selfcal.core.system.apply_line_fisher_mask` applies a threshold.
        """
        v = self.attrs.get('line_fisher_threshold')
        return None if v is None else float(v)

    @property
    def solve(self) -> dict | None:
        """The record of the solve that made the file, or None (a cal made before records existed, a
        stitched cal, an N-pass pass product).

        The attributes of the ``solve`` group (:class:`~selfcal.core.solve_record.SolveRecord`):
        the method, the iterations run and in total, ``istop`` and its meaning (``stop``), the
        solver's final estimates, the true residual ``|b - A x|``, the tolerances and ``conlim``,
        and ``history_file``, the NPZ of its history per iteration (the wall time is in the
        action's record only: the file stays byte-identical from run to run). A snapshot of a solve
        (:mod:`selfcal.core.snapshots`) has ``snapshot = True``, ``iteration`` (cumulative) and the
        solver's estimates at that iteration instead of the final ones (no ``istop``).
        """
        from ..core.solve_record import read
        return read(self._f)

    # ---- frames ---------------------------------------------------------------------------
    @property
    def reproj_list(self) -> list[str]:
        """The frame files of the solve, in the row order of every per-frame dataset.

        ``[]`` when the file has no ``reproj_list`` dataset.
        """
        f = self._f
        if 'reproj_list' not in f:
            return []
        return [_decode(r) for r in f['reproj_list'][()]]

    @property
    def n_frames(self) -> int:
        """The number of frames: the length of ``reproj_list``, else map 0's rows, else 0."""
        f = self._f
        if 'reproj_list' in f:
            return int(f['reproj_list'].shape[0])
        offs = self.offsets
        return int(offs[0].shape[0]) if offs else 0

    @property
    def frame_scalar(self) -> np.ndarray | None:
        """The per-frame scalar offset, shape ``(n_frames,)``, or None when the solve had none."""
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
        """The number of offset maps; 1 for a legacy single-``offset`` file, 0 without offsets.

        The ``num_maps`` root attribute when present, else the size of the
        ``offsets`` group.
        """
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
        """Per-map ``(n_frames, n_chunks)`` counts of the observations of each offset.

        Template and polynomial-basis maps, whose unknowns are not one per chunk,
        store ones. A map whose offsets multiply ``n_basis > 1`` functions
        (:meth:`offset_basis`) has ``n_chunks * n_basis`` columns, one per unknown.
        """
        return self._per_map('offset_coverage', 'offset_coverage')

    @property
    def offset_coverage_frac(self) -> list[np.ndarray]:
        """Per-map ``(n_frames, n_chunks)`` coverage as a fraction of each chunk's pixels.

        :attr:`offset_coverage` divided by the chunk's size in detector pixels.
        Template and polynomial-basis maps store ones; a map with ``n_basis > 1``
        functions stores its :attr:`offset_coverage` counts undivided, as floats.
        """
        return self._per_map('offset_coverage_frac', 'offset_coverage_frac')

    def offset_basis(self, m) -> tuple[int, str] | None:
        """``(n_basis, description)`` when map ``m``'s offsets are coefficients of
        ``n_basis`` known functions of data variables per chunk (columns
        ``chunk * n_basis + k``), else None (plain per-chunk offsets)."""
        f = self._f
        if 'offsets' not in f or f'map_{m}' not in f['offsets']:
            return None
        a = f['offsets'][f'map_{m}'].attrs
        if 'n_basis' not in a:
            return None
        return int(a['n_basis']), _decode(a.get('basis', ''))

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
        """A short text summary: schema, sky blocks and grid, offset maps, frames, attributes."""
        lines = [f'{os.path.basename(self.path)}: schema v{self.schema_version}',
                 f'  sky blocks: {self.sky_names} on {self.ref_shape}',
                 f'  offset maps: {self.num_maps}, frames: {self.n_frames}, '
                 f'frame scalar: {"yes" if self.frame_scalar is not None else "no"}']
        extra = {k: v for k, v in self.attrs.items()
                 if k not in ('sky_components',) and not isinstance(v, np.ndarray)}
        if extra:
            lines.append('  attrs: ' + ', '.join(f'{k}={v}' for k, v in sorted(extra.items())))
        solve = self.solve
        if solve is not None and solve.get('snapshot'):
            lines.append(f"  snapshot of a {solve.get('method')} solve at iteration {solve.get('iteration')} "
                         f"({solve.get('iterations')} of this solve), r1norm = {solve.get('r1norm')}")
        elif solve is not None:
            lines.append(f"  solve: {solve.get('method')}, {solve.get('iterations')} iterations "
                         f"(istop {solve.get('istop')}: {solve.get('stop')}), |b - A x| = {solve.get('true_residual')}")
        return '\n'.join(lines)


def open_cal(path) -> CalFile:
    """Open a cal file for reading; the same as ``CalFile(path)``."""
    return CalFile(path)
