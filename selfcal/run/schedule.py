"""Big fields and long solves: tiles, the N-pass alternating solve, snapshots and monitors of a solve.

:class:`Tiles` splits the reference grid into tiles solved one at a time and stitched (each
pixel the Fisher-weighted mean of the tiles that cover it). :class:`Passes` alternates exact
sky and per-frame offset solves after a first joint solve (the N-pass schedule of
``PIPELINE.md``); :class:`Refit` is its offset pass. :class:`Snapshots` writes a solve's
solution every ``k`` iterations as a cal file (:mod:`selfcal.core.snapshots`); :class:`Monitor`
checks a solve every ``m`` iterations (:mod:`selfcal.core.monitor`).
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from typing import Literal

from ..config.base import Config, ConfigError
from .recipe import Clip, as_clip

__all__ = ['Tiles', 'Refit', 'Passes', 'Snapshots', 'Monitor', 'schedule', 'as_snapshots', 'as_monitor']


@dataclass(frozen=True)
class Tiles(Config):
    """Tiles of the reference grid, solved one at a time and stitched.

    A uniform ``grid=(n_y, n_x)`` with ``overlap`` pixels and optional tile ``names``, or explicit
    ``boxes={name: (y0, y1, x0, x1)}`` (which may overlap). ``assign``: a frame goes to the tile
    holding its footprint's centre (``"center"``), or to every tile it overlaps (``"overlap"``);
    ``halo``: grow each tile by this many pixels for the assignment. ``only``: solve only these
    tiles (no stitch). ``tile_name`` / ``stitched_name``: the products' names, formed from the
    recipe's ``{name}`` and the tile's ``{tile}``.
    """
    grid: tuple[int, int] | None = None
    _: KW_ONLY
    overlap: int = 0
    names: tuple[str, ...] | None = None
    boxes: dict[str, tuple[int, int, int, int]] | None = None
    assign: Literal['center', 'overlap'] = 'center'
    halo: int = 0
    only: tuple[str, ...] | None = None
    tile_name: str = '{name}_{tile}'
    stitched_name: str = '{name}_stitched'

    def _validate(self):
        if (self.grid is None) == (self.boxes is None):
            raise ConfigError("Tiles: give grid=(n_y, n_x) or boxes={name: (y0, y1, x0, x1)}")
        if self.boxes is not None and (self.overlap or self.names):
            raise ConfigError("Tiles: overlap= and names= belong to grid=; explicit boxes carry their own")
        if self.grid is not None and min(self.grid) < 1:
            raise ConfigError(f"Tiles(grid={self.grid}): at least one tile per axis")
        if self.names is not None and self.grid is not None and len(self.names) != self.grid[0] * self.grid[1]:
            raise ConfigError(f"Tiles: {len(self.names)} names for a {self.grid[0]} x {self.grid[1]} grid")
        if '{tile}' not in self.tile_name:
            raise ConfigError(f"Tiles(tile_name={self.tile_name!r}): needs {{tile}}, or every tile writes the "
                              f"same file")
        if self.boxes is not None and self.only is not None:
            bad = [t for t in self.only if t not in self.boxes]
            if bad:
                raise ConfigError(f"Tiles(only=...): no tiles {bad} (tiles: {sorted(self.boxes)})")

    def tile_names(self):
        """The tile names, in order (None for an unnamed grid: the engine names it)."""
        return list(self.boxes) if self.boxes is not None else (list(self.names) if self.names else None)


@dataclass(frozen=True)
class Refit(Config):
    """The N-pass offset pass: each frame's offsets refitted against the fixed sky, as a
    degree-``degree`` polynomial along the spectral axis per group (optionally piecewise on
    ``segments``) plus a level. ``clip``: of the residual (per frame unless chunk groups are
    given). ``bright_cut``: only pixels whose modelled sky is below it (all pixels when fewer than
    ``min_pixels`` remain). ``ridge``: Tikhonov on the shape coefficients (0: plain least
    squares)."""
    degree: int = 4
    _: KW_ONLY
    clip: Clip = field(default_factory=lambda: Clip(2.5))
    bright_cut: float | None = 0.05
    min_pixels: int = 5000
    segments: tuple[tuple[int, int], ...] | None = None
    ridge: float = 0.0

    def _validate(self):
        object.__setattr__(self, 'clip', as_clip(self.clip, 'Refit(clip=...)'))
        if self.clip is None:
            raise ConfigError("Refit(clip=None): the offset refit always clips; give a sigma")
        if self.degree < 0 or self.min_pixels < 0 or self.ridge < 0:
            raise ConfigError("Refit: degree, min_pixels and ridge are at least 0")


def schedule(n, order='offset_first'):
    """The pass types of an ``n``-pass schedule: ``init``, then sky and offset alternately."""
    if order not in ('sky_first', 'offset_first'):
        raise ConfigError(f"order is 'sky_first' or 'offset_first', got {order!r}")
    first, second = ('offset', 'sky') if order == 'offset_first' else ('sky', 'offset')
    return ['init'] + [first if i % 2 == 0 else second for i in range(2, n + 1)]


@dataclass(frozen=True)
class Passes(Config):
    """The N-pass alternating solve: pass 1 is the recipe's joint solve (INIT), then exact SKY
    and per-frame OFFSET passes alternate, ``order`` deciding which comes first.

    ``init_clip``: the first pass's clip (None: the recipe's ``Fit.clip``); ``sky_clip``: the SKY
    passes'; ``offset``: the OFFSET pass (:class:`Refit`). Clips are per frame unless chunk groups
    are given (``sc.Clip(5.0, per=SUBCHANNEL)``). ``stop_tol``: stop early once the sky changes
    less than this. ``sky_merge``: how tiled SKY passes are merged (``"combine"``: summed normal
    equations, exact; ``"stitch"``: per-tile solves, Fisher-stitched). ``keep_moments``: keep
    those intermediates. A schedule ending on an OFFSET pass is refused (its last sky would be a
    pass older) unless ``ends_on_offset=True``.
    """
    n: int = 3
    _: KW_ONLY
    order: Literal['sky_first', 'offset_first'] = 'offset_first'
    init_clip: float | Clip | None = None
    sky_clip: float | Clip = field(default_factory=lambda: Clip(5.0))
    offset: Refit = field(default_factory=Refit)
    stop_tol: float = 0.0
    sky_merge: Literal['combine', 'stitch'] = 'combine'
    keep_moments: bool = False
    ends_on_offset: bool = False

    def _validate(self):
        if self.n < 1:
            raise ConfigError(f"Passes(n={self.n}): at least 1")
        object.__setattr__(self, 'init_clip', as_clip(self.init_clip, 'Passes(init_clip=...)'))
        object.__setattr__(self, 'sky_clip', as_clip(self.sky_clip, 'Passes(sky_clip=...)'))
        if self.sky_clip is None:
            raise ConfigError("Passes(sky_clip=None): the SKY passes always clip; give a sigma")
        if self.n > 1 and schedule(self.n, self.order)[-1] == 'offset' and not self.ends_on_offset:
            raise ConfigError(f"Passes(n={self.n}, order={self.order!r}) ends on an OFFSET pass, which leaves the "
                              f"sky of the pass before; use n={self.n + 1} (or the other order), or say "
                              f"ends_on_offset=True")
        if self.stop_tol < 0:
            raise ConfigError("Passes(stop_tol=...): at least 0")


@dataclass(frozen=True)
class Snapshots(Config):
    """Snapshots of a solve: its solution after every ``every``-th iteration, written as a cal file.

    ``field.calibrate(recipe, snapshots=sc.Snapshots(every=50, keep=3))`` writes
    ``calibration/snapshots/<cal stem>_it<NNNN>.h5`` after iterations 50, 100, ... (``NNNN``: the
    cumulative iteration, counted from a warm start's total); not after the iteration the solve
    stops at (the cal is written then). Each is a complete cal file (the cal's schema, its
    ``solve`` group marked ``snapshot = True`` with the ``iteration``), so it can be mosaicked
    (``field.mosaic(recipe, cal=...)``), continued (``calibrate(start=...)``) and read like any
    cal. ``keep``: how many of the solve's snapshots to keep, the latest (None: all). Snapshots
    do not change the solve, so they are an action's setting, recorded and replayed by a rerun,
    and never part of a product's inputs; they are not products (no sidecar). Plain
    calibrations only (no ``tiles`` or ``passes``). See :mod:`selfcal.core.snapshots`.
    """
    every: int
    _: KW_ONLY
    keep: int | None = None

    def _validate(self):
        if self.every < 1:
            raise ConfigError(f"Snapshots(every={self.every}): at least 1 iteration")
        if self.keep is not None and self.keep < 1:
            raise ConfigError(f"Snapshots(keep={self.keep}): at least 1 (None: keep them all)")


def as_snapshots(value) -> Snapshots | None:
    """``calibrate(snapshots=...)`` as a :class:`Snapshots`: None, a :class:`Snapshots`, or a number of
    iterations (``snapshots=50`` is ``Snapshots(every=50)``)."""
    if value is None or isinstance(value, Snapshots):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return Snapshots(value)
    raise ConfigError(f"calibrate(snapshots=...): a sc.Snapshots(every=..., keep=...) or a number of iterations; "
                      f"got {value!r}")


@dataclass(frozen=True)
class Monitor(Config):
    """Convergence monitors of a solve: checks every ``every`` iterations, saved in its history.

    ``field.calibrate(recipe, monitor=sc.Monitor(every=10))`` checks each solve at iterations 0,
    10, 20, ...: ``residual``, the true ``|b - A x|`` (one product ``A x``) and the ratio of the
    solver's estimate to it; ``gradient``, the true ``|A^T r|`` too (one product ``A^T r`` more)
    and its estimate's ratio; ``large_scale``, the smooth fit of each sky term (``True``: degree 2,
    or the Fit's large-scale stop rule's; a number: that degree; ``False``: none) on every
    ``step``-th row and column of its covered pixels (None: 4, or the rule's), its coefficients
    and their relative change since the previous check. The values go to the solve's history file
    (``<field>/records/<cal stem>_history.npz``: ``check_itn``, ``check_iteration`` counted across
    warm starts, ``true_residual``, ``residual_ratio``, ``true_gradient``, ``gradient_ratio``,
    ``large_scale_*``), the log, and the last check to the action's record.

    Monitors do not change the solve: an action's setting, recorded and replayed by a rerun, never
    part of a product's inputs; the cal is the same, byte for byte, with or without them. A stop
    rule that needs a check (:class:`~selfcal.run.recipe.Stop`) runs it at its own cadence, with or
    without a monitor. See :mod:`selfcal.core.monitor`.
    """
    every: int = 10
    _: KW_ONLY
    residual: bool = True
    gradient: bool = False
    large_scale: bool | int = True
    step: int | None = None

    def _validate(self):
        if self.every < 1:
            raise ConfigError(f"Monitor(every={self.every}): at least 1 iteration")
        if not isinstance(self.large_scale, bool) and not 1 <= self.large_scale <= 6:
            raise ConfigError(f"Monitor(large_scale={self.large_scale}): a degree from 1 to 6, True or False")
        if self.step is not None and self.step < 1:
            raise ConfigError(f"Monitor(step={self.step}): at least 1 (None: 4, or the stop rule's)")
        if not (self.residual or self.gradient or self.large_scale is not False):
            raise ConfigError("Monitor: nothing to check (residual, gradient and large_scale are off)")


def as_monitor(value) -> Monitor | None:
    """``calibrate(monitor=...)`` as a :class:`Monitor`: None, a :class:`Monitor`, ``True`` (the
    default monitor) or a number of iterations (``monitor=20`` is ``Monitor(every=20)``)."""
    if value is None or isinstance(value, Monitor):
        return value
    if value is True:
        return Monitor()
    if isinstance(value, int) and not isinstance(value, bool):
        return Monitor(value)
    raise ConfigError(f"calibrate(monitor=...): a sc.Monitor(every=..., ...), True or a number of iterations; "
                      f"got {value!r}")
