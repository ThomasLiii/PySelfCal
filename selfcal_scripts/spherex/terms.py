"""Model terms for SPHEREx, built from the public API (``sc.Offsets``, ``sc.Sky``, ``sc.Function``,
``sc.DetectorMap``, ``sc.template``). They are choices a recipe makes, not properties of the instrument: a recipe
imports them (``from selfcal_scripts.spherex import terms as T``) and composes its model with them. In every call a
data variable appears only inside ``of=`` (the built-in ``det_x``; a registered map such as ``arc_position``); every
other keyword is a bound parameter. The functions must stay importable by the worker processes (module level here).
"""
import os

import numpy as np

import selfcal as sc
from selfcal.instruments.spherex.settings import DETECTOR_SIZE, LINE_TEMPLATES, _DATA as LINE_TEMPLATE_DIR
from selfcal.instruments.spherex.spherex_utility import PAH_LINE_CENTER_UM

__all__ = ['tilt', 'frame_slope', 'arc_position', 'hats', 'spectral_offsets', 'chebyshev', 'spectral_polynomial', 'line',
           'pah_3p29']

#: The intrinsic width of the PAH 3.29 um feature, as a variance in um^2, added in quadrature to the band width.
PAH_INTRINSIC_VAR_UM2 = 2.890e-4


def tilt(x, centre, half):
    """A linear basis of a coordinate: ``(x - centre) / half``, -1 at ``centre - half`` and +1 at ``centre + half``."""
    return (np.asarray(x, dtype=np.float64) - centre) / half


def frame_slope(name='xs_term', on='subchannel', per='frame', size=DETECTOR_SIZE, exclude=None):
    """An offset term that is a free gradient along detector x per frame (``per``) and chunk of the map ``on``:
    one coefficient times :func:`tilt` of the built-in variable ``det_x`` over a detector ``size`` pixels wide
    (centre ``(size - 1) / 2``, half width ``size / 2``). ``exclude`` is the term's hard constraint
    (``sc.SkySignature('all')`` in the channel recipes: the slopes carry no sky pattern)."""
    return sc.Offsets(name, on=on, per=per, exclude=exclude,
                      times=sc.Function(tilt, of='det_x', centre=(size - 1) / 2.0, half=size / 2.0))


def arc_position(geometry, map='subchannel'):
    """Each detector pixel's position across the LVF arcs, in subchannels of the chunk map ``map``: a
    detector map for ``sc.DetectorMap``. The subchannel edges are the concentric circles of the arc fit
    (``geometry.extra['lvf_params']``: centre ``xc``, ``yc``; the map's ``extra['r_edges']``: the edges'
    radii, :func:`selfcal_scripts.spherex.chunks.subchannels`), so the position is the pixel's radius about
    the centre interpolated on the edges' radii against the edge index: subchannel ``s`` spans ``s - 1`` to
    ``s``, and beyond the first and the last edge the position continues linearly at the end intervals'
    rate."""
    lvf = geometry.extra['lvf_params']
    r_edges = np.asarray(geometry.chunk_maps[map].extra['r_edges'], dtype=np.float64)
    H, W = geometry.shape
    y, x = np.mgrid[0:H, 0:W].astype(np.float64)
    r = np.hypot(x - lvf['xc'], y - lvf['yc'])
    k = np.arange(r_edges.size, dtype=np.float64)
    u = np.interp(r, r_edges, k)
    lo, hi = r < r_edges[0], r > r_edges[-1]
    u[lo] = (r[lo] - r_edges[0]) / (r_edges[1] - r_edges[0])
    u[hi] = k[-1] + (r[hi] - r_edges[-1]) / (r_edges[-1] - r_edges[-2])
    return u


def hats(position, first, last):
    """The hat functions on the subchannel edges ``first`` to ``last`` of a position across the arcs: each is 1
    at its edge and falls linearly to 0 at the neighbouring edges, so their span is the continuous, piecewise
    linear functions of the position with the edges as nodes."""
    return [np.clip(1.0 - np.abs(position - k), 0.0, None) for k in range(first, last + 1)]


def spectral_offsets(subchannels, position='arc_position', name='spectral', per='frame'):
    """An offset term that is a continuous, piecewise-linear function of the position across the LVF arcs per
    frame (``per``): one coefficient per frame and subchannel edge of ``subchannels`` (a range of subchannel
    indices, the job's: a channel job covers the channel's subchannels and one on each side), times the hat
    function of that edge (:func:`hats`), on the whole detector. In place of per-subchannel constants it follows a
    background that varies within a subchannel (an airglow line's edge) with no step between subchannels;
    continuity holds by the basis, with no constraint and no weight. ``position`` names the model's variable
    holding the position: ``variables={position: sc.DetectorMap(arc_position)}``."""
    first, last = int(subchannels.start) - 1, int(subchannels.stop) - 1
    return sc.Offsets(name, on='detector', per=per, basis=sc.Function(hats, of=position, first=first, last=last),
                      n=last - first + 1)


def chebyshev(position, degree, lo, hi):
    """The Chebyshev polynomials ``T_0 .. T_degree`` of a position across the LVF arcs over the subchannels ``lo``
    to ``hi`` (inclusive indices): the position is mapped to ``x`` in [-1, 1] from the first subchannel's start edge
    (``lo - 1``; subchannel ``s`` spans the positions ``s - 1`` to ``s``, :func:`arc_position`) to the last one's end
    edge (``hi``), and the polynomials run on past the window. ``degree + 1`` functions, the constant first."""
    from numpy.polynomial import chebyshev as C
    x = 2.0 * (np.asarray(position, dtype=np.float64) - (lo - 1)) / (hi - (lo - 1)) - 1.0
    return C.chebvander(x, int(degree))


def spectral_polynomial(subchannels, degree=2, position='arc_position', name='poly', on='detector', per='frame'):
    """An offset term that is a polynomial of degree ``degree`` in the position across the LVF arcs over the
    subchannels ``subchannels`` (a range of subchannel indices, the fit's window), per frame (``per``) and per chunk
    of the map ``on`` (``'detector'``: one polynomial per frame; ``'detector_column'``: one per frame and column):
    ``degree + 1`` coefficients each, of the Chebyshev polynomials :func:`chebyshev` (the constant among them, so the
    term holds the frame's level too). A continuous function of the position, exact by its basis: no constraint and
    no weight. ``position`` names the model's variable holding the position:
    ``variables={position: sc.DetectorMap(arc_position)}``."""
    lo, hi = int(subchannels.start), int(subchannels.stop) - 1
    return sc.Offsets(name, on=on, per=per, basis=sc.Function(chebyshev, of=position, degree=int(degree), lo=lo, hi=hi),
                      n=int(degree) + 1)


def line(name, damping=None, *, file=None):
    """A sky term times one of the shipped line templates of the wavelength (``LINE_TEMPLATES`` of the instrument
    module: ``aromatic``, ``aliphatic``, ``plateau``), or the template ``file``."""
    if file is None:
        if name not in LINE_TEMPLATES:
            raise ValueError(f"line({name!r}): no shipped template ({sorted(LINE_TEMPLATES)}); give file=")
        file = os.path.join(LINE_TEMPLATE_DIR, LINE_TEMPLATES[name])
    return sc.Sky(name, times=sc.template(file), damping=damping)


def pah_3p29(center=None):
    """The PAH 3.29 um feature as a sky coefficient: a Gaussian of the wavelength map centred on ``center``
    (default 3.290 um) whose width per observation is the band width in quadrature with the feature's intrinsic
    width (a fixed width instead: ``sc.gaussian(center, sigma=...)``)."""
    return sc.gaussian(PAH_LINE_CENTER_UM if center is None else center, width='bandwidth',
                       intrinsic_var=PAH_INTRINSIC_VAR_UM2, fwhm_to_sigma=2.355)
