"""The run engine's input: one engine run, resolved from the Python API's objects.

:func:`selfcal.run.lower.lower` resolves a field, a recipe, jobs and the options of an action into
one :class:`RunSpec` per group of jobs the instrument runs together, and the engine
(:mod:`selfcal.run.engine`, :mod:`selfcal.run.pipelines`, :mod:`selfcal.run.npass`) reads nothing
else. Every value is resolved: the library keywords of the solve, the solver and the coadd, the
frames and how they are staged, the tiles, the passes. The settings objects it was resolved from
ride along for the record and the products' book; nothing here is fingerprinted (the products
are fingerprinted from the settings, :mod:`selfcal.run.products`).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from dataclasses import field as dc_field

from ..config.base import Config, encode

__all__ = ['RunSpec', 'FrameSource', 'TilingSpec', 'PassClip', 'PassesSpec', 'ReprojectSpec']


@dataclass(frozen=True)
class FrameSource:
    """Where a run's frames come from: read in place from ``in_place``, or copied into
    ``stage_dir`` (``stage="copy"``; ``"reuse"``: a copy another run made), ``io_limit`` reads at a
    time, and deleted after the run unless ``keep``. ``files``: the frames to solve, by name, in
    this order (None: every frame of the directory, sorted); ``first_n``: only the first ``n``."""
    in_place: str | None = None
    files: tuple[str, ...] | None = None
    first_n: int | None = None
    stage: str = 'copy'
    stage_dir: str | None = None
    keep: bool = False
    io_limit: int = 20


@dataclass(frozen=True)
class TilingSpec:
    """The tiles of a tiled calibration on a reference grid of ``ref_shape``: explicit ``tiles``
    (``((name, (y0, y1, x0, x1)), ...)``, in order) or a ``grid`` ``(n_y, n_x)`` with ``overlap``
    pixels and tile ``names`` (None: the engine names them); ``only``: solve these tiles only (no
    stitch). Every frame of ``frames_dir`` goes to the tile holding its footprint's centre
    (``assign="center"``) or to every tile it overlaps, each tile grown by ``halo`` pixels, and is
    staged into ``stage_dir``. ``stitched_suffix``: the stitched cal's suffix; ``stitch_line``: the
    stitch weighs the line terms; ``memory_guard``: the RSS guard thread runs."""
    ref_shape: tuple[int, int]
    frames_dir: str
    stage_dir: str
    stitched_suffix: str
    tiles: tuple | None
    grid: tuple[int, int] | None
    overlap: int
    names: tuple[str, ...] | None
    only: tuple[str, ...] | None
    assign: str
    halo: int
    stitch_line: bool
    memory_guard: bool


@dataclass(frozen=True)
class PassClip:
    """A clip of the N-pass schedule at ``sigma``, per frame or ``grouped`` along the primary chunk
    map's spectral axis; ``ignore_flags`` (the first pass only): the data-quality bits that do not
    flag a pixel (None: the fit's)."""
    sigma: float
    grouped: bool
    ignore_flags: tuple[int, ...] | None = None


@dataclass(frozen=True)
class PassesSpec:
    """The N-pass schedule: ``n`` passes, ``order`` saying which of the SKY and OFFSET passes comes
    first after the INIT; ``init_clip``: the INIT's clip (None: the fit's); ``sky_clip``; the OFFSET
    refit's polynomial ``refit_degree`` (piecewise on ``segments``), its clip ``refit_clip``,
    ``bright_cut``, ``min_pixels`` and ``ridge``; ``stop_tol``, ``sky_merge`` and ``keep_moments``
    as :class:`~selfcal.run.schedule.Passes` says."""
    n: int
    order: str
    init_clip: PassClip | None
    sky_clip: PassClip
    refit_degree: int
    refit_clip: PassClip
    bright_cut: float | None
    min_pixels: int
    segments: tuple | None
    ridge: float
    stop_tol: float
    sky_merge: str
    keep_moments: bool


@dataclass(frozen=True)
class ReprojectSpec:
    """A reprojection: the ``exposures`` (glob patterns or files) onto the reference grid, made
    from ``reference`` (a FITS file whose WCS it takes) or fitted to the exposures when there is
    none, ``padding`` pixels and ``padding_fraction`` wider; ``method``; ``replace`` existing
    frames; ``verify`` every frame afterwards; ``workers`` processes."""
    exposures: tuple[str, ...]
    reference: str | None
    method: str
    padding: int
    padding_fraction: float
    replace: bool
    verify: bool
    workers: int


@dataclass
class RunSpec:
    """One engine run: its ``task`` (``"cal"``, ``"mosaic"``, ``"npass"`` or ``"reproject"``) on the
    ``field`` (``<output_dir>/<run_name>``, reference grid at ``resolution_arcsec``) for the
    ``jobs`` of its ``instrument``, everything resolved.

    ``recipe``: the settings it was resolved from. ``scratch``: the scratch area (staged tiles,
    the solver's spill files, the coadd's cache, the N-pass work; ends with ``/``); ``suffix``:
    of every product name (``{tile}`` in a tiled run); ``oversample``: of the coadd. ``model``:
    the ``[model]`` table of the recipe's model (:meth:`~selfcal.models.model.Model.lower`), which
    the engine reads as a :class:`~selfcal.models.spec.ModelSpec`. ``setup``, ``lsqr`` and
    ``mosaic``: the keywords of ``setup_lsqr`` (the model's own options aside), ``apply_lsqr`` and
    ``make_mosaic``; ``pre_cal`` / ``post_cal`` / ``post_mosaic``: the frame hooks.
    ``line_fisher_threshold`` and ``spectral_window`` (the model's polynomial window, the N-pass
    refit's) as the recipe says. ``make_mosaic``: the run coadds; ``instrument_maps``: with the
    instrument's maps; ``reuse_mosaics``: an existing mosaic is kept (the plan checked it);
    ``cal_override``: the cal a mosaic task coadds. ``on_product`` is told of every product written
    (the action's book, :class:`~selfcal.run.products.Book`).
    """
    task: str
    field: object
    instrument: object
    output_dir: str
    run_name: str
    resolution_arcsec: float | None
    recipe: object = None
    jobs: tuple = ()
    scratch: str | None = None
    suffix: str = ''
    oversample: int = 1
    frames: FrameSource = dc_field(default_factory=FrameSource)
    model: dict | None = None
    line_fisher_threshold: float = 10.0
    spectral_window: tuple[int, int] | None = None
    setup: dict = dc_field(default_factory=dict)
    lsqr: dict = dc_field(default_factory=dict)
    mosaic: dict = dc_field(default_factory=dict)
    pre_cal: object = None
    post_cal: object = None
    post_mosaic: object = None
    make_mosaic: bool = False
    instrument_maps: bool = False
    reuse_mosaics: bool = False
    cal_override: str | None = None
    tiling: TilingSpec | None = None
    passes: PassesSpec | None = None
    reproject: ReprojectSpec | None = None
    on_product: object = None

    def replace(self, **changes) -> RunSpec:
        """A copy with ``changes``."""
        return dataclasses.replace(self, **changes)

    def describe(self) -> dict:
        """The run in JSON form (what a record keeps under ``lowered``): every value but the settings
        it was resolved from (the record holds them) and the product book."""
        return {f.name: _plain(getattr(self, f.name)) for f in dataclasses.fields(self)
                if f.name not in ('field', 'recipe', 'on_product')}


def _plain(value):
    if dataclasses.is_dataclass(value) and not isinstance(value, (type, Config)):
        return {f.name: _plain(getattr(value, f.name)) for f in dataclasses.fields(value)}
    return encode(value)
