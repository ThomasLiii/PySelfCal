"""Continuing a solve: the identity of a system, and a starting vector read back from a cal file.

A solve can start from the solution of an earlier one (``field.calibrate(recipe, start=<cal>)``).
:class:`WarmStart` reads that cal's solution as the solver's starting vector ``x0``, the exact
inverse of how :meth:`~selfcal.pipeline.pipeline_wrapper.Calibrator.save_calibration` writes it,
after checking that the cal is a solution of the same system (:class:`System`): the same frames
in the same order, the same model (sky terms, chunk maps, how frames share offsets, basis
functions), the same reference grid and the same column counts.

The starting vector is in the solver's full (uncompacted) column layout and in physical units, as
:func:`~selfcal.core.solve.apply_lsqr` takes it (it compacts and scales it itself):

* each sky term's map ``sky/<name>``, in the model's order, row-major; pixels the source did not
  solve start at 0;
* each offset term's block ``offsets/map_<m>``, frame-major. A term shared by groups of frames
  (``per="all"`` or a frame variable) stores its group's offsets on every frame of the group: they
  are read from the group's first frame, after checking that every frame of the group holds them.
  A term with ``n`` basis functions holds ``n`` coefficients per chunk, as stored;
* the per-frame scalar ``frame_scalar``.

Columns the solve has now but the source left at zero (it did not solve them) start at 0; source
values of columns this solve does not have (no data now) are dropped by the compaction;
:meth:`WarmStart.vector` logs both counts. An offset term with a polynomial basis
(``Offsets(polynomial=...)``) cannot be continued: its cal holds the offsets the polynomial
expands to, not its coefficients.

A continuation restarts the solver's Krylov space: LSQR and LSMR solve ``A dx = b - A x0`` from
``dx = 0``, so a solve of ``N`` iterations continued for ``M`` more is not one solve of ``N + M``
iterations (the search directions start again from the residual of ``x0``).

Every solve records the identity of its system in the cal's ``solve`` group (``system``, its
fingerprint, and ``system_identity``, the JSON it is the fingerprint of; :class:`System`), and a
continuation its source (``start_from``, ``start_identity``) and the cumulative iteration count
(``iterations_total``). A source with a recorded identity is checked against it as well as against
its contents; one without (a cal solved before 2026-10-08) by its contents only: its frames, sky
terms, chunk maps, column counts and grid shape (not the grid's WCS, the sky terms' coefficients or
the job).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field

import numpy as np

from ..config.base import ConfigError

__all__ = ['System', 'WarmStart', 'IDENTITY_VERSION', 'contents_problems']

logger = logging.getLogger(__name__)

#: The layout of :meth:`System.identity` (its ``version``).
IDENTITY_VERSION = 1

# The pieces the active-column counts are taken over (bool temporaries of 50 MB).
_COUNT_BLOCK = 50_000_000


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _array_digest(a) -> str:
    """The sha256 of an array's dtype, shape and bytes."""
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f'{a.dtype.str}{a.shape}'.encode())
    h.update(a.tobytes())
    return h.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), default=str)


def _plain(value):
    """A polynomial-basis descriptor (or any dict with arrays) in JSON form, arrays by digest."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return {'array': _array_digest(value)}
    if isinstance(value, np.generic):
        return value.item()
    return value


def _diff(a, b, path=''):
    """The differences between two identities, as ``path: a -> b`` lines."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _diff(a.get(k, '<absent>'), b.get(k, '<absent>'), f'{path}.{k}' if path else k)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += _diff(x, y, f'{path}[{i}]')
        return out
    if a == b:
        return []

    def short(v):
        text = str(v)
        return text[:40] + '...' if len(text) > 40 else text
    return [f'{path}: {short(a)} (the source) vs {short(b)} (this solve)']


def frames_identity(frames) -> dict:
    """The frames of a solve in its order, by name: their number and the hash of their names."""
    names = [os.path.basename(os.fspath(f)) for f in frames]
    return {'n': len(names), 'sha256': _digest('\n'.join(names).encode())}


def grid_identity(shape, wcs=None) -> dict:
    """The reference grid: its shape and, with its ``wcs`` (an ``astropy.wcs.WCS``), the hash of the
    WCS parameters that place it (``crval``, ``crpix``, the pixel scale and rotation, the projection)."""
    out = {'shape': [int(shape[0]), int(shape[1])]}
    if wcs is not None:
        w = wcs.wcs
        params = {'ctype': [str(c) for c in w.ctype], 'crval': [float(v) for v in w.crval],
                  'crpix': [float(v) for v in w.crpix], 'cdelt': [float(v) for v in w.get_cdelt()],
                  'pc': [[float(v) for v in row] for row in w.get_pc()]}
        out['wcs'] = _digest(_canonical(params).encode())
    return out


@dataclass
class System:
    """The unknowns of one solve: what a starting vector must be a solution of.

    ``layout``: the column layout (:class:`~selfcal.core.layout.SystemLayout`); ``frames``: the
    frame files, in the solve's order (by name); ``sky``: ``(name, coefficient)`` of each sky term
    (the coefficient's description, None for a constant term); ``chunk_maps``: each offset term's
    chunk map; ``basis``: each offset term's ``(n, description)`` of its known functions, or None;
    ``wcs``: the reference grid's WCS (None: its shape only); ``extra``: more of the identity
    (the run engine adds the job)."""
    layout: object
    frames: tuple
    sky: tuple
    chunk_maps: list
    basis: list
    wcs: object = None
    extra: dict = field(default_factory=dict)
    _identity: dict | None = field(default=None, init=False, repr=False, compare=False)

    @classmethod
    def of(cls, layout, *, frames, sky_model, chunk_maps, basis_list=None, wcs=None, extra=None) -> System:
        """The system of a set-up solve: ``sky_model`` (a :class:`~selfcal.models.sky_model.SkyModel`)
        and ``basis_list`` (per offset term an :class:`~selfcal.models.offset_model.Basis` or None) as
        the solver holds them."""
        sky = tuple((c.name, None if c.coefficient is None else c.coefficient.describe())
                    for c in sky_model.components)
        basis = [None if b is None else (int(b.n), b.coefficient.describe())
                 for b in (basis_list or [None] * len(chunk_maps))]
        return cls(layout, tuple(os.path.basename(os.fspath(f)) for f in frames), sky, list(chunk_maps), basis,
                   wcs, dict(extra or {}))

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def identity(self) -> dict:
        """The identity as plain data: the frames (in order), the grid, the sky terms, each offset
        term (its chunk map, chunks and columns, its groups of frames, its basis, template or
        polynomial basis), the per-frame scalar columns, the total column count, and ``extra``
        (computed once)."""
        if self._identity is not None:
            return self._identity
        L = self.layout
        offsets = []
        for m in range(L.num_maps):
            cm = np.asarray(self.chunk_maps[m])
            pb = (L.poly_basis_list or [None] * L.num_maps)[m]
            tmpl = L.det_template_arr_list[m]
            n, desc = self.basis[m] if self.basis[m] is not None else (None, None)
            offsets.append({
                'chunk_map': _array_digest(cm.astype(np.int64)), 'chunks': int(cm.max()) + 1,
                'columns': int(L.col_bases[m + 1] - L.col_bases[m]), 'groups': int(L.num_offset_groups_list[m]),
                'grouping': _array_digest(np.asarray(L.frame_to_group_list[m], dtype=np.int64)),
                'basis': None if n is None else {'n': n, 'function': desc},
                'template': None if tmpl is None else _array_digest(tmpl),
                'polybasis': None if pb is None else _plain(pb)})
        out = {'version': IDENTITY_VERSION, 'frames': frames_identity(self.frames),
               'grid': grid_identity(L.ref_shape, self.wcs),
               'sky': [{'name': name, 'coefficient': coeff} for name, coeff in self.sky],
               'offsets': offsets, 'scalar': int(L.num_scalar_cols), 'columns': int(L.total_cols)}
        out.update(self.extra)
        self._identity = out
        return out

    def identity_json(self) -> str:
        """:meth:`identity` as canonical JSON (what the cal's ``solve`` group records)."""
        return _canonical(self.identity())

    def fingerprint(self) -> str:
        """The sha256 of :meth:`identity_json`."""
        return _digest(self.identity_json().encode())

    def stored_columns(self, m) -> int:
        """The columns of offset map ``m`` as a cal stores them per frame: its unknowns per group,
        except a template's or a polynomial basis's map, stored per chunk."""
        L = self.layout
        pb = (L.poly_basis_list or [None] * L.num_maps)[m]
        if L.det_template_arr_list[m] is not None or pb is not None:
            return int(np.asarray(self.chunk_maps[m]).max()) + 1
        return int(L.num_chunks_list[m])


class WarmStart:
    """The cal a solve continues from (see the module docstring).

    ``path``: the source cal; ``identity``: what identifies it among products (the run engine
    gives ``"fingerprint:<sha256>"``, its sidecar's fingerprint, or ``"sha256:<sha256>"``, the hash
    of its bytes). The source's record is read when it is made: :attr:`iterations_total` (None when
    it has no record of its iterations) and the identity of its system (:attr:`recorded`, None for
    a cal solved before systems were recorded)."""

    def __init__(self, path, identity=None):
        from ..io.calfile import CalFile
        self.path = os.path.abspath(os.fspath(path))
        self.identity = identity
        with CalFile(self.path) as cal:
            solve = cal.solve or {}
        total = solve.get('iterations_total')
        self.iterations_total = None if total is None or int(total) < 0 else int(total)
        text = solve.get('system_identity')
        self.recorded = json.loads(text) if text else None
        self.system = None              # the system it was checked against (check())
        self.started_zero = self.dropped = None     # the counts of vector()

    def __repr__(self):
        return f"WarmStart({self.path!r})"

    # ---- the checks ------------------------------------------------------------------------
    def problems(self, system: System) -> list[str]:
        """What makes the source not a solution of ``system`` (empty: nothing): its contents
        (frames, sky terms and grid shape, offset maps, chunk maps, basis functions, column counts,
        the per-frame scalar) and, when the source records the identity of its system, that
        identity."""
        from ..io.calfile import CalFile
        L = system.layout
        out = contents_problems(self.path, frames=system.frames, sky_names=[name for name, _ in system.sky],
                                n_maps=L.num_maps, ref_shape=L.ref_shape, chunk_maps=system.chunk_maps)
        with CalFile(self.path) as cal:
            f = cal._f
            frames = cal.reproj_list
            K = cal.num_maps if 'offsets' in f else 0
            if K == L.num_maps:
                for m in range(K):
                    shape = tuple(f['offsets'][f'map_{m}'].shape)
                    want_shape = (system.n_frames, system.stored_columns(m))
                    if frames and shape != want_shape:
                        out.append(f"offset map {m}: {shape} values in the source, {want_shape} in this solve")
                    basis = cal.offset_basis(m)
                    want_basis = system.basis[m]
                    if (basis is None) != (want_basis is None) or (basis is not None and tuple(basis) != tuple(want_basis)):
                        out.append(f"offset map {m}: basis functions {basis} in the source, {want_basis} here")
            scalar = 'frame_scalar' in f
            if scalar != bool(L.num_scalar_cols):
                out.append(f"the per-frame scalar: {'present' if scalar else 'absent'} in the source, "
                           f"{'present' if L.num_scalar_cols else 'absent'} in this solve")
            elif scalar and frames and f['frame_scalar'].shape != (system.n_frames,):
                out.append(f"the per-frame scalar: {f['frame_scalar'].shape} values, {system.n_frames} frames")
        if self.recorded is not None:
            out += _diff(self.recorded, system.identity())
        return out

    def check(self, system: System):
        """Refuse (:class:`~selfcal.config.base.ConfigError`) a source that is not a solution of
        ``system`` (:meth:`problems`), and a model whose offsets cannot be read back (a polynomial
        basis, a template); otherwise bind the start to ``system``."""
        L = system.layout
        for m in range(L.num_maps):
            if (L.poly_basis_list or [None] * L.num_maps)[m] is not None:
                raise ConfigError(f"start={self.path}: offset term {m} has a polynomial basis "
                                  f"(Offsets(polynomial=...)): its cal holds the offsets the polynomial expands "
                                  f"to, not the polynomial's coefficients, so the solve cannot continue from it")
            if L.det_template_arr_list[m] is not None:
                raise ConfigError(f"start={self.path}: offset term {m} is a template amplitude: its cal holds "
                                  f"the offsets the template expands to, not the amplitudes")
        problems = self.problems(system)
        if problems:
            shown = '; '.join(problems[:8]) + (f' (+{len(problems) - 8} more)' if len(problems) > 8 else '')
            raise ConfigError(f"start={self.path} is not a solution of this system: {shown}. A solve continues "
                              f"only from a cal of the same frames (in the same order), model, reference grid and "
                              f"job")
        if self.recorded is None:
            logger.warning(f"warm start from {self.path}: the source records no system identity (solved before "
                           f"2026-10-08); checked by its contents: frames, sky terms, chunk maps, column counts and "
                           f"grid shape (not the grid's WCS, the sky coefficients or the job)")
        self.system = system

    # ---- the starting vector ---------------------------------------------------------------
    def vector(self, active_mask=None) -> np.ndarray:
        """The starting vector: the source's solution in the full column layout of the system it was
        checked against (:meth:`check`), float64, the exact inverse of how a cal is written (see the
        module docstring). ``active_mask`` (the solve's active columns, or None: every column)
        only serves the counts logged: columns active now that start at 0 (the source left them at
        zero or unsolved), and nonzero source values of columns inactive now (dropped)."""
        if self.system is None:
            raise ValueError("WarmStart.vector(): check() the start against the system first")
        from ..io.calfile import CalFile
        system = self.system
        L = system.layout
        n = system.n_frames
        x0 = np.zeros(int(L.total_cols), dtype=np.float64)
        with CalFile(self.path) as cal:
            f = cal._f
            for j, (name, _) in enumerate(system.sky):
                ds = cal._sky_dataset('sky', 'skymap', 'skymap_line', name)
                ds.read_direct(x0[L.sky_block_slice(j)].reshape(L.ref_shape))
            for m in range(L.num_maps):
                ds = f['offsets'][f'map_{m}']
                ng, nc = int(L.num_offset_groups_list[m]), int(L.num_chunks_list[m])
                block = x0[L.offset_slice(m)].reshape(ng, nc)
                ftg = np.asarray(L.frame_to_group_list[m])
                if ng == n and np.array_equal(ftg, np.arange(n)):
                    ds.read_direct(block)                   # one row per frame, as stored
                    continue
                stored = ds[()]
                first = np.empty(ng, dtype=np.int64)
                first[ftg[::-1]] = np.arange(n)[::-1]          # each group's first frame
                rows = stored[first]
                if not np.array_equal(stored, rows[ftg]):
                    raise ConfigError(f"start={self.path}: offset map {m} of the source does not share its "
                                      f"offsets over this solve's groups of frames (another grouping)")
                block[...] = rows
            if L.num_scalar_cols:
                f['frame_scalar'].read_direct(x0[L.scalar_slice()])
        # Unsolved (NaN) values read as 0, and the counts, piece by piece (no full-size temporary).
        unsolved = started_zero = dropped = 0
        active_mask = None if active_mask is None else np.asarray(active_mask, dtype=bool)
        for s in range(0, x0.size, _COUNT_BLOCK):
            piece = x0[s:s + _COUNT_BLOCK]
            nan = np.isnan(piece)
            if nan.any():
                unsolved += int(np.count_nonzero(nan))
                piece[nan] = 0.0
            nz = piece != 0
            act = np.ones(piece.size, dtype=bool) if active_mask is None else active_mask[s:s + _COUNT_BLOCK]
            started_zero += int(np.count_nonzero(act & ~nz))
            dropped += int(np.count_nonzero(~act & nz))
        n_active = int(x0.size if active_mask is None else np.count_nonzero(active_mask))
        logger.info(f"warm start from {self.path}: {n_active} active columns, {started_zero} of them zero or "
                    f"unsolved in the source (start at 0); {dropped} source values of columns inactive now "
                    f"(dropped)" + (f"; {unsolved} unsolved (NaN) source values read as 0" if unsolved else ''))
        self.started_zero, self.dropped = started_zero, dropped
        return x0


def contents_problems(path, *, frames=None, sky_names=None, n_maps=None, ref_shape=None, chunk_maps=None,
                      job=None) -> list[str]:
    """What a cal's contents show is not a solution of a solve of ``frames`` (names, in order),
    ``sky_names`` on a grid of ``ref_shape``, ``n_maps`` offset terms on ``chunk_maps`` (detector
    chunk maps) and, when the cal records its system, ``job`` (as the identity holds it); each None:
    not checked. The run engine's plan checks a warm start with it before the solve is set up; the
    whole system is checked when it is (:meth:`WarmStart.check`)."""
    from ..io.calfile import CalFile
    out = []
    with CalFile(path) as cal:
        f = cal._f
        source = [os.path.basename(p) for p in cal.reproj_list]
        if not source:
            out.append("frames: the source lists none (a stitched cal or an N-pass product is no start)")
        elif frames is not None and tuple(source) != tuple(os.path.basename(x) for x in frames):
            out.append('frames: ' + _frames_problem(source, [os.path.basename(x) for x in frames]))
        names = list(cal.sky_names)
        if sky_names is not None and names != list(sky_names):
            out.append(f"sky terms: {names} (the source) vs {list(sky_names)} (this solve)")
        elif ref_shape is not None:
            for name in names:
                shape = tuple(cal._sky_dataset('sky', 'skymap', 'skymap_line', name).shape)
                if shape != tuple(ref_shape):
                    out.append(f"the grid: sky {name!r} is {shape} in the source, {tuple(ref_shape)} here")
                    break
        k = cal.num_maps if 'offsets' in f else 0
        if n_maps is not None and k != n_maps:
            out.append(f"offset terms: {k} in the source, {n_maps} in this solve")
        elif chunk_maps is not None and 'chunk_maps' in f:
            for m in range(min(k, len(chunk_maps))):
                if f'map_{m}' in f['chunk_maps'] and _array_digest(
                        np.asarray(f['chunk_maps'][f'map_{m}'][()], dtype=np.int64)) != \
                        _array_digest(np.asarray(chunk_maps[m], dtype=np.int64)):
                    out.append(f"offset map {m}: another chunk map")
        text = (cal.solve or {}).get('system_identity')
        if job is not None and text:
            recorded = json.loads(text).get('job')
            if recorded != job:
                out.append(f"the job: {recorded} (the source) vs {job} (this solve)")
    return out


def _frames_problem(source, here) -> str:
    if len(source) != len(here):
        return f"the source was solved on {len(source)} frames, this solve has {len(here)}"
    if sorted(source) == sorted(here):
        return f"the same {len(here)} frames in another order"
    k = sum(a != b for a, b in zip(source, here))
    first = next((a, b) for a, b in zip(source, here) if a != b)
    return f"{k} of the {len(here)} frames differ (the first: {first[0]} in the source, {first[1]} here)"
