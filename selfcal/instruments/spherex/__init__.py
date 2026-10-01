"""SPHEREx: the ``spherex`` instrument and its linear variable filter (LVF) geometry.

On each SPHEREx detector an LVF sets the wavelength a pixel sees: its band centre
``BC`` (band width ``BW``) is nearly constant along concentric circular arcs and
changes across them. The chunk maps therefore cut the detector into subchannels
(strips between neighbouring arcs), and those into columns.

- :mod:`~selfcal.instruments.spherex.adapter`: the instrument,
  :class:`~selfcal.instruments.spherex.adapter.SPHERExInstrument`
  (``[instrument].name = "spherex"``).
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
