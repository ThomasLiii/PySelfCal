"""Data variables — the named per-observation quantities a model's functions read.

An *observation* is one sample of one frame: a value on a reference (sky) pixel,
seen at a detector position, in a frame. Every function of a self-calibration
model — a sky term's coefficient, an offset term's basis, the observation
weight, a frame grouping, a clip grouping — reads **data variables**: named
arrays with one value per observation. Where a variable comes from is declared
once, by its *source*; the functions never know::

    source           one value per ...      examples
    ---------------  ---------------------  ---------------------------------------------
    built-in         observation            det_x, det_y (detector position),
                                            sky_x, sky_y (reference pixel), frame (index)
    detector         detector pixel         SPHEREx band-centre map BC, a pixel polariser angle
    frame            frame                  time, filter, half-wave-plate angle, temperature
    sky              reference pixel        ecliptic latitude, a dust template, a previous sky
    layer            observation            a variance plane stored with each frame file
    derived          observation            any function of other variables
    frame function   observation            any function of the whole frame (its data,
                                            coordinates, header, other variables)

A detector variable is sampled (bilinearly) at each observation's detector
position; a frame variable is broadcast over the frame's observations; a sky
variable is read at the observation's reference pixel; a layer is read from the
frame file. Everything else is a function, so any quantity a future instrument
defines — the time of each row of a drift-scan detector, the wavelength of a
tunable filter, the signal of a mirrored amplifier — is a variable without a
change here.

:class:`VariableSet` declares the sources of a solve; the row assembly, the
mosaic and the N-pass evaluate them per frame through
:class:`ObservationVariables`, a lazy mapping (a variable is computed only
when a function reads it, then cached for the frame).
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np

__all__ = ['BUILTIN_VARIABLES', 'Derived', 'FrameFunction', 'VariableSet', 'FrameObservations',
           'ObservationVariables', 'identity']


def identity(x):
    """The variable itself (the coefficient of a term that is proportional to a variable)."""
    return x

BUILTIN_VARIABLES = {
    'det_x': 'detector column of the observation (detector pixels)',
    'det_y': 'detector row of the observation (detector pixels)',
    'sky_x': 'reference-grid column of the observation',
    'sky_y': 'reference-grid row of the observation',
    'frame': 'index of the frame in the solve',
}


@dataclass(frozen=True)
class Derived:
    """A variable computed per observation from other variables:
    ``function(*[obs[name] for name in inputs])``. ``function`` is any picklable
    callable (an :class:`~selfcal.models.sky_model.ImportedFunction` for one
    referenced by import path)."""
    inputs: tuple
    function: object

    def __post_init__(self):
        inputs = (self.inputs,) if isinstance(self.inputs, str) else tuple(self.inputs)
        object.__setattr__(self, 'inputs', inputs)


@dataclass(frozen=True)
class FrameFunction:
    """A variable computed from the whole frame: ``function(frame)`` with ``frame``
    a :class:`FrameObservations`. Returns one value per observation, a value per
    subframe pixel (the subframe grid), or a scalar (broadcast)."""
    function: object


@dataclass(frozen=True)
class VariableSet:
    """The data-variable sources of one solve (all optional).

    ``detector``: ``{name: map on the detector grid}``; ``frame``:
    ``{name: (n_frames,) values}``; ``sky``: ``{name: map on the reference grid}``;
    ``layers``: names of per-observation planes stored in the frame files
    (``layers/<name>``); ``derived``: ``{name: Derived}``; ``frame_functions``:
    ``{name: FrameFunction}``. The built-ins are always available.
    """
    detector: dict = field(default_factory=dict)
    frame: dict = field(default_factory=dict)
    sky: dict = field(default_factory=dict)
    layers: tuple = ()
    derived: dict = field(default_factory=dict)
    frame_functions: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, 'layers', tuple(self.layers))
        seen = {}
        for scope in ('detector', 'frame', 'sky', 'derived', 'frame_functions'):
            for name in getattr(self, scope):
                if name in seen or name in BUILTIN_VARIABLES:
                    raise ValueError(f"data variable {name!r} is defined twice "
                                     f"({seen.get(name, 'built-in')} and {scope})")
                seen[name] = scope
        for name in self.layers:
            if name in seen or name in BUILTIN_VARIABLES:
                raise ValueError(f"data variable {name!r} is defined twice ({seen.get(name, 'built-in')} and layers)")

    @property
    def names(self) -> tuple[str, ...]:
        """Every variable the set provides, built-ins included."""
        return (tuple(BUILTIN_VARIABLES) + tuple(self.detector) + tuple(self.frame) + tuple(self.sky)
                + self.layers + tuple(self.derived) + tuple(self.frame_functions))

    def provides(self, name) -> bool:
        return name in self.names

    def scope(self, name) -> str:
        """Where ``name`` comes from: built-in, detector, frame, sky, layer, derived, frame_function."""
        if name in BUILTIN_VARIABLES:
            return 'built-in'
        for scope in ('detector', 'frame', 'sky', 'derived', 'frame_functions'):
            if name in getattr(self, scope):
                return 'frame_function' if scope == 'frame_functions' else scope
        if name in self.layers:
            return 'layer'
        raise KeyError(name)

    def merged(self, **more) -> VariableSet:
        """A copy with ``more`` sources added (same keyword names as the fields)."""
        kw = {'detector': dict(self.detector), 'frame': dict(self.frame), 'sky': dict(self.sky),
              'layers': tuple(self.layers), 'derived': dict(self.derived),
              'frame_functions': dict(self.frame_functions)}
        for k, v in more.items():
            if k == 'layers':
                kw[k] = kw[k] + tuple(n for n in v if n not in kw[k])
            else:
                kw[k].update(v)
        return VariableSet(**kw)

    @property
    def is_empty(self) -> bool:
        return not (self.detector or self.frame or self.sky or self.layers or self.derived
                    or self.frame_functions)


# ---------------------------------------------------------------------------
# per-frame evaluation
# ---------------------------------------------------------------------------
@dataclass
class FrameObservations:
    """What a frame function receives: the frame and its observations.

    ``pixels`` are the ``(rows, cols)`` of the observations on the subframe grid
    (the frame's box ``ref_coords = [y0, y1, x0, x1]`` on the reference grid);
    ``sub_data`` / ``sub_weight`` / ``sub_mapping`` are the prepared subframe
    arrays (``sub_mapping`` holds the detector ``(x, y)`` of every subframe
    pixel); ``variables`` the frame's other variables. :meth:`raw` reads the
    frame's stored values, :meth:`header` its stored header."""
    file: str
    index: int
    pixels: tuple
    ref_coords: object
    sub_data: np.ndarray = None
    sub_weight: np.ndarray = None
    sub_mapping: np.ndarray = None
    variables: object = None

    def raw(self):
        """The frame's stored values on the subframe grid, before any hook or
        offset subtraction — the same in the solve, the mosaic and the N-pass
        (use it, not ``sub_data``, for a variable computed from the data)."""
        from ..io.reproj import load_reproj_file
        v = load_reproj_file(self.file, fields=['sub_data'])['sub_data']
        return np.asarray(v, dtype=np.float64)

    def header(self):
        """The frame's stored detector header (an ``astropy.io.fits.Header``)."""
        import h5py
        from astropy.io import fits
        with h5py.File(self.file, 'r') as f:
            h = f.attrs.get('det_header', b'')
        return fits.Header.fromstring(h.decode() if isinstance(h, bytes) else str(h))

    def at_observations(self, sub_grid_array):
        """``sub_grid_array`` (one value per subframe pixel) at the observations."""
        return np.asarray(sub_grid_array)[self.pixels]


class ObservationVariables(Mapping):
    """The data variables of one frame's observations, evaluated on demand.

    ``pixels``: the observations' ``(rows, cols)`` on the subframe grid.
    ``detector``: ``{name: subframe-grid array}`` (detector maps already sampled
    onto the subframe) or ``{name: per-observation array}``; ``frame_values``:
    ``{name: scalar}`` (this frame's value
    of each frame variable); ``sky``: ``{name: reference-grid map}``; ``layers``:
    ``{name: subframe-grid array}``; ``derived`` / ``frame_functions``: the
    function sources; ``context``: the :class:`FrameObservations` handed to frame
    functions. Values are float arrays of length ``n`` (the number of
    observations), computed on first access and cached.
    """

    def __init__(self, pixels, *, index=0, ref_coords=None, sub_mapping=None, detector=None,
                 frame_values=None, sky=None, layers=None, derived=None, frame_functions=None,
                 context=None):
        self.pixels = (np.asarray(pixels[0]), np.asarray(pixels[1]))
        self.n = int(self.pixels[0].shape[0])
        self.index = int(index)
        self.ref_coords = ref_coords
        self.sub_mapping = sub_mapping
        self._detector = detector or {}
        self._frame = frame_values or {}
        self._sky = sky or {}
        self._layers = layers or {}
        self._derived = derived or {}
        self._functions = frame_functions or {}
        self._context = context
        if context is not None and getattr(context, 'variables', None) is None:
            context.variables = self
        self._cache = {}
        self._busy = set()

    # ---- Mapping ---------------------------------------------------------------------
    def _names(self):
        out = list(BUILTIN_VARIABLES)
        if self.sub_mapping is None:
            out = [n for n in out if n not in ('det_x', 'det_y')]
        if self.ref_coords is None:
            out = [n for n in out if n not in ('sky_x', 'sky_y')]
        for src in (self._detector, self._frame, self._sky, self._layers, self._derived, self._functions):
            out.extend(src)
        return out

    def __contains__(self, name):
        return name in self._names()

    def __iter__(self):
        return iter(self._names())

    def __len__(self):
        return len(self._names())

    def get(self, name, default=None):
        return self[name] if name in self else default

    def __getitem__(self, name):
        hit = self._cache.get(name)
        if hit is not None:
            return hit
        if name not in self:
            raise KeyError(f"data variable {name!r} is not available for this solve "
                           f"(available: {sorted(self._names())})")
        if name in self._busy:
            raise ValueError(f"data variable {name!r} depends on itself")
        self._busy.add(name)
        try:
            value = self._compute(name)
        finally:
            self._busy.discard(name)
        self._cache[name] = value
        return value

    # ---- evaluation --------------------------------------------------------------------
    def _broadcast(self, name, v):
        v = np.asarray(v)
        if v.ndim == 0:
            return np.full(self.n, v, dtype=np.float64 if v.dtype.kind in 'biuf' else v.dtype)
        if v.shape == (self.n,):
            return v
        sub = self.pixels
        if v.ndim == 2 and self.n and v.shape[0] > int(sub[0].max()) and v.shape[1] > int(sub[1].max()):
            return v[sub]
        raise ValueError(f"data variable {name!r}: expected {self.n} values (one per observation), "
                         f"a subframe-grid array or a scalar; got shape {v.shape}")

    def _compute(self, name):
        rows, cols = self.pixels
        if name in self._detector or name in self._layers:
            # a subframe-grid array (indexed at the observations), or already
            # one value per observation
            a = self._detector[name] if name in self._detector else np.asarray(self._layers[name])
            return a if a.ndim == 1 else a[rows, cols]
        if name in self._frame:
            return self._broadcast(name, self._frame[name])
        if name in self._sky:
            m = self._sky[name]
            y = rows + int(self.ref_coords[0])
            x = cols + int(self.ref_coords[2])
            inside = (y >= 0) & (y < m.shape[0]) & (x >= 0) & (x < m.shape[1])
            out = np.full(self.n, np.nan, dtype=np.float64)
            out[inside] = m[y[inside], x[inside]]
            return out
        if name in self._derived:
            d = self._derived[name]
            return self._broadcast(name, d.function(*[self[i] for i in d.inputs]))
        if name in self._functions:
            if self._context is None:
                raise ValueError(f"frame function {name!r} needs the frame context")
            return self._broadcast(name, self._functions[name].function(self._context))
        if name == 'det_x':
            return np.asarray(self.sub_mapping[0])[rows, cols]
        if name == 'det_y':
            return np.asarray(self.sub_mapping[1])[rows, cols]
        if name == 'sky_x':
            return (cols + int(self.ref_coords[2])).astype(np.float64)
        if name == 'sky_y':
            return (rows + int(self.ref_coords[0])).astype(np.float64)
        if name == 'frame':
            return np.full(self.n, float(self.index))
        raise KeyError(name)
