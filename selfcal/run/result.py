"""What an action made: the products of a calibration or a mosaic, and readers for them."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

import numpy as np

__all__ = ['Result', 'MosaicFile']


class MosaicFile:
    """A mosaic FITS file: its maps by name (``MEAN_MAP``, ``STD_MAP``, ``SC_MEAN_MAP``, ...), their
    weights, the WCS and the unit. A context manager; arrays are read on access."""

    def __init__(self, path):
        from astropy.io import fits
        self.path = path
        self._hdul = fits.open(path, memmap=True)

    def close(self):
        self._hdul.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __repr__(self):
        return f"MosaicFile({self.path!r}: {', '.join(self.names)})"

    @property
    def names(self) -> list[str]:
        """The maps the file holds (weight maps excluded)."""
        return [h.name for h in self._hdul[1:] if not h.name.endswith('_WEIGHT')]

    def __getitem__(self, name) -> np.ndarray:
        return np.asarray(self._hdul[name.upper()].data)

    def weight(self, name) -> np.ndarray:
        """The weight map of ``name``."""
        return np.asarray(self._hdul[f'{name.upper()}_WEIGHT'].data)

    @property
    def mean(self) -> np.ndarray:
        """The weighted mean of the calibrated frames."""
        return self['MEAN_MAP']

    @property
    def std(self) -> np.ndarray:
        """The per-pixel standard deviation."""
        return self['STD_MAP']

    @property
    def clipped(self) -> np.ndarray:
        """The sigma-clipped mean (the mosaic to use when the coadd clips)."""
        return self['SC_MEAN_MAP']

    @property
    def instrument_maps(self) -> dict:
        """The instrument's coadded per-pixel maps (SPHEREx: ``WAV_MEAN_MAP``, ``WAV_STD_MAP``)."""
        return {n: self[n] for n in self.names if n.startswith('WAV_')}

    @property
    def wcs(self):
        from astropy.wcs import WCS
        return WCS(self._hdul['MEAN_MAP'].header)

    @property
    def unit(self) -> str:
        return str(self._hdul['MEAN_MAP'].header.get('BUNIT', ''))


@dataclass
class Result:
    """The products of an action, per job: ``cal_paths`` and ``mosaic_paths`` (in ``jobs`` order),
    a tiled run's ``tile_cals`` (``{tile: path}``), an N-pass run's ``passes`` (``{pass: path}``),
    ``final`` (the cal holding the field's final sky: the stitched cal, the last SKY pass, or the
    only cal) and the run ``record``.

    ``cal()`` / ``mosaic()`` open a product (context managers); ``sky(name)`` and
    ``offsets(name)`` read arrays by term name; ``result[job]`` is the result of one job.
    """
    field: object
    recipe: object
    jobs: tuple
    cal_paths: list
    mosaic_paths: list = field(default_factory=list)
    tile_cals: dict | None = None
    passes: dict | None = None
    final: str | None = None
    record: str | None = None

    def __repr__(self):
        lines = [f"Result of {os.path.basename(self.field.path)}: {len(self.jobs)} job(s)"]
        for i, j in enumerate(self.jobs):
            cal = self.cal_paths[i] if i < len(self.cal_paths) else None
            mos = self.mosaic_paths[i] if i < len(self.mosaic_paths) else None
            lines.append(f"  {j.name}: cal {cal}" + (f"\n  {' ' * len(j.name)}  mosaic {mos}" if mos else ''))
        if self.final and self.final not in self.cal_paths:
            lines.append(f"  final sky: {self.final}")
        if self.record:
            lines.append(f"  record: {self.record}")
        return '\n'.join(lines)

    def _index(self, job):
        if job is None:
            if len(self.jobs) != 1:
                raise ValueError(f"the result has {len(self.jobs)} jobs; name one ({[j.name for j in self.jobs]})")
            return 0
        names = [j.name for j in self.jobs]
        key = getattr(job, 'name', job)
        if key not in names:
            raise KeyError(f"no job {key!r} in the result ({names})")
        return names.index(key)

    def __getitem__(self, job) -> Result:
        i = self._index(job)
        return Result(self.field, self.recipe, (self.jobs[i],), [self.cal_paths[i]],
                      [self.mosaic_paths[i]] if i < len(self.mosaic_paths) else [],
                      final=self.cal_paths[i] if len(self.jobs) > 1 else self.final, record=self.record)

    # ---- products ----------------------------------------------------------------------------
    def cal(self, job=None):
        """The job's cal file (the final one: stitched or last pass), a
        :class:`~selfcal.io.calfile.CalFile` (use it in a ``with`` block)."""
        from ..io.calfile import CalFile
        path = self.final if (job is None and self.final) else self.cal_paths[self._index(job)]
        return CalFile(path)

    def mosaic(self, job=None) -> MosaicFile:
        """The job's mosaic, a :class:`MosaicFile` (use it in a ``with`` block)."""
        if not self.mosaic_paths:
            raise FileNotFoundError("the action made no mosaic (Recipe(coadd=None), tiles or passes)")
        return MosaicFile(self.mosaic_paths[self._index(job)])

    def sky(self, name=None, job=None) -> np.ndarray:
        """The sky map of the sky term ``name`` (default the first) from the final cal."""
        with self.cal(job) as cal:
            return np.asarray(cal.sky(0 if name is None else name))

    def offsets(self, name=None, job=None) -> np.ndarray:
        """The offsets of the offset term ``name`` (default the first), per frame:
        ``(frames, chunks)``, or ``(frames, chunks, n)`` for a term with ``n`` basis functions."""
        terms = list(self.recipe.model.offsets)
        if not terms:
            raise ValueError("the model has no offset terms")
        names = [t.term_name for t in terms]
        m = 0 if name is None else (names.index(name) if name in names else None)
        if m is None:
            raise KeyError(f"no offset term {name!r} (terms: {names})")
        with self.cal(job) as cal:
            off = np.asarray(cal.offsets[m])
            basis = cal.offset_basis(m)
        if basis is not None and basis[0] > 1:
            n = basis[0]
            off = off.reshape(off.shape[0], -1, n)
        return off

    def show(self, job=None, name='SC_MEAN_MAP', *, vmin=None, vmax=None, path=None, dpi=300):
        """Draw one mosaic map with a colour bar (the full array; matplotlib resamples it). Saves
        to ``path`` when given; returns the figure."""
        import matplotlib.pyplot as plt
        with self.mosaic(job) as mos:
            data = mos[name if name in mos.names else 'MEAN_MAP']
            unit = mos.unit
        if vmin is None or vmax is None:
            lo, hi = np.nanpercentile(data, [1, 99])
            vmin = lo if vmin is None else vmin
            vmax = hi if vmax is None else vmax
        fig, ax = plt.subplots(figsize=(8, 7), dpi=dpi)
        im = ax.imshow(data, origin='lower', vmin=vmin, vmax=vmax, cmap='gray')
        fig.colorbar(im, ax=ax, label=unit)
        ax.set_title(os.path.basename(self.mosaic_paths[self._index(job)]), fontsize=8)
        if path is not None:
            fig.savefig(path, dpi=dpi, bbox_inches='tight')
        return fig
