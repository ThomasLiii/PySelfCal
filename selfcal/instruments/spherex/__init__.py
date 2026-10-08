"""SPHEREx: the instrument ``sc.SPHEREx`` and its linear variable filter (LVF) geometry.

On each SPHEREx detector an LVF sets the wavelength a pixel sees: its band centre
``BC`` (band width ``BW``) is nearly constant along concentric circular arcs and
changes across them. The chunk maps therefore cut the detector into subchannels
(strips between neighbouring arcs), and those into columns.

- :mod:`~selfcal.instruments.spherex.settings`: SPHEREx as settings of the Python API:
  :class:`~selfcal.instruments.spherex.settings.SPHEREx`, which implements the instrument
  contract (the run's geometry, a job's masks and weights, the mosaic's renderer and
  wavelength maps, the exposure layout), its jobs
  (:func:`~selfcal.instruments.spherex.settings.channel`,
  :func:`~selfcal.instruments.spherex.settings.channels`,
  :func:`~selfcal.instruments.spherex.settings.group`,
  :func:`~selfcal.instruments.spherex.settings.window`),
  :func:`~selfcal.instruments.spherex.settings.line`,
  :func:`~selfcal.instruments.spherex.settings.precompute_lvf` and
  :func:`~selfcal.instruments.spherex.settings.zodi_anchor` (available from this package:
  ``spherex.channel(17)``).
- :mod:`~selfcal.instruments.spherex.adapter`: the named subchannel windows and the
  readout-channel map.
- :mod:`~selfcal.instruments.spherex.spherex_utility`: the LVF geometry: the ``BC`` /
  ``BW`` maps, the arc fit (``lvf_params``), the stripped chunk maps and valid masks,
  the smooth arc offset renderer, adjacency and polynomial chains.
- :mod:`~selfcal.instruments.spherex.line_catalog`: named sky-term coefficients
  (``pah_3p29``).
- :mod:`~selfcal.instruments.spherex.wavemap`: the standalone coadd of the mosaic's
  wavelength maps.

Package data: the per-detector arc fits ``data/lvf_params/lvf_params_D<n>.npy`` and
the line templates ``data/line_templates/*.npz``.
"""
_SETTINGS = ('SPHEREx', 'channel', 'channels', 'group', 'window', 'line', 'LINE_TEMPLATES', 'precompute_lvf',
             'zodi_anchor')


def __getattr__(name):
    # The settings load on first use, so worker processes importing the instrument stay light.
    if name in _SETTINGS:
        from . import settings
        return getattr(settings, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

