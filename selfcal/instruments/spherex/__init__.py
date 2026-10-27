"""SPHEREx: the instrument ``sc.SPHEREx`` and its linear variable filter (LVF) geometry.

On each SPHEREx detector an LVF sets the wavelength a pixel sees: its band centre
``BC`` (band width ``BW``) is nearly constant along concentric circular arcs and
changes across them. The chunk maps therefore cut the detector into subchannels
(strips between neighbouring arcs), and those into columns.

- :mod:`~selfcal.instruments.spherex.settings`: SPHEREx as settings of the Python API:
  :class:`~selfcal.instruments.spherex.settings.SPHEREx`, which implements the instrument
  contract (the exposure layout, the band-centre and band-width maps, the LVF arc fit, the
  mosaic's wavelength maps), :func:`~selfcal.instruments.spherex.settings.precompute_lvf` and
  :func:`~selfcal.instruments.spherex.settings.zodi_anchor` (available from this package). The
  chunk maps on the arcs, the jobs and the terms a model applies are built beside the recipes,
  in ``selfcal_scripts/spherex/``.
- :mod:`~selfcal.instruments.spherex.exposures`: the L2b exposure reader
  (:func:`~selfcal.instruments.spherex.exposures.read_l2b`: ``IMAGE``, ``FLAGS`` and the
  ``VARIANCE`` as the frame layer ``variance``) and
  :func:`~selfcal.instruments.spherex.exposures.l2b_variance`.
- :mod:`~selfcal.instruments.spherex.adapter`: the detector-column and readout-channel maps.
- :mod:`~selfcal.instruments.spherex.spherex_utility`: the LVF geometry: the ``BC`` /
  ``BW`` maps, the arc fit (``lvf_params``), the stripped chunk maps and valid masks,
  the smooth arc offset renderer, adjacency and polynomial chains.
- :mod:`~selfcal.instruments.spherex.wavemap`: the standalone coadd of the mosaic's
  wavelength maps.

Package data: the per-detector arc fits ``data/lvf_params/lvf_params_D<n>.npy`` and
the line templates ``data/line_templates/*.npz``.
"""
_SETTINGS = ('SPHEREx', 'LINE_TEMPLATES', 'precompute_lvf', 'zodi_anchor')


def __getattr__(name):
    # The settings load on first use, so worker processes importing the instrument stay light.
    if name in _SETTINGS:
        from . import settings
        return getattr(settings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

