"""Product fingerprints (selfcal.run.products) against a committed golden, ``tests/data/fingerprints.json``.

Every product the Python API writes (a cal, a tile cal, a stitched cal, an N-pass product, a mosaic)
gets a sidecar holding the canonical inputs it was made from and their fingerprint, and an existing
product is reused only while that fingerprint is what the action would make now. Any change to a
canonical input (a settings class moved or renamed, a setting added, removed or given another
default, another job key, tile box or product name, another encoding of a value) turns every existing
sidecar it touches into ``different``: the product is refused until it is made again. Nothing else
pins these values; this test does. For a set of cases covering every fingerprinted object, the
canonical inputs and the fingerprint of each product must equal the golden:

* from the settings (``products.cal_inputs``, ``mosaic_inputs``, ``stitched_inputs`` and
  ``pass_inputs`` with the ``Book``'s tile key, chained by fingerprint as the ``Book`` chains them),
  on fixed frame lists and jobs: the instruments ``sc.SPHEREx``, ``sc.Euclid``, ``sc.Camera`` and
  :class:`Prism` (an ``sc.Instrument`` subclass of this module); the models ``sc.continuum``,
  ``sc.spectral`` (template, Gaussian, catalogue, tabulated and linear lines, damping set and left
  unset), ``sc.two_block`` and an ``sc.Model`` with every kind of term, data-variable source,
  function (by name and ``sc.by_value``), weight and prior; ``sc.Fit`` with every kind of clip and
  with hooks; ``sc.Coadd``; ``sc.Numerics``; ``sc.Passes`` with ``sc.Refit``; ``sc.Tiles`` (a grid
  and boxes, both assignments, a halo);
* through the plan (``field.plan(...).products``) on synthetic fields: the engine's share of the
  inputs (its job keys, the tile boxes and the frames assigned to each tile) and the product names,
  for a camera (plain, and with detector maps and header variables), Euclid and :class:`Prism`, with
  tiles and passes. (SPHEREx needs its calibration maps to plan, which CI has not.)

Machine independence. The fields' ``ref.fits`` enters by the sha256 of its bytes: it is written byte
by byte here (no FITS writer whose output could change with its version). Files the settings name (a
detector and a sky map, a solved sky's cal and its sidecar) are written with fixed bytes under the
test's temporary directory and enter by content; a shipped line template enters by content too;
frames enter by name. :func:`test_no_path_of_this_machine_enters_a_fingerprint` checks that neither
the temporary directory nor the checkout's path appears in any canonical input. Functions and classes
of this module are recorded as ``test_fingerprints:<name>``, the name pytest imports it under (any
other import name is mapped to it). The golden holds on any checkout path, on Python 3.11 and 3.12.

Regenerating the golden::

    SELFCAL_WRITE_FINGERPRINT_GOLDEN=1 pytest tests/test_fingerprints.py

declares that every existing product sidecar whose inputs changed is now ``different`` (those
products are refused until made again): do it only on purpose, for a change meant to change what
decides the products' bytes, and say so in the commit.
"""
import difflib
import json
import os
import struct
import sys
from dataclasses import dataclass

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import selfcal as sc  # noqa: E402
from selfcal.instruments import spherex  # noqa: E402
from selfcal.io.frames import standard_frame_path, write_frame  # noqa: E402
from selfcal.run import products  # noqa: E402

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fingerprints.json')
WRITE_VARIABLE = 'SELFCAL_WRITE_FINGERPRINT_GOLDEN'
MODULE = 'test_fingerprints'           # the name this module's functions and classes are recorded under

FRAMES = tuple(f'exp_{k:04d}_det_{d:02d}.h5' for k in range(1, 13) for d in (1, 2))    # 24 frames, by name
CAL_FINGERPRINT = '5ca1ab1e' * 8       # the cal of every mosaic of the first part
SOLVED_FINGERPRINT = '0ddba11' + '0' * 57    # the sidecar fingerprint of the solved sky's cal
REF_SHAPE = (120, 160)                 # the reference grid of every field
PIXEL_SCALE = 6.2
SUBCHANNEL = sc.ChunkGroups.along('subchannel')


# ============================================================================
# functions, a hook and an instrument the settings name (module level: recorded by name)
# ============================================================================
def ramp(det_x, scale=1.0):
    """A sky coefficient: the detector column, scaled."""
    return scale * np.asarray(det_x, dtype=np.float64)


def season(time, t0=60000.0, period=365.25):
    """A sky coefficient of a frame variable: a seasonal modulation."""
    return np.sin(2 * np.pi * (np.asarray(time, dtype=np.float64) - t0) / period)


def bump(det_y, centre=16.0, width=8.0):
    """A sky coefficient sent by value (sc.by_value): a bump across the detector rows."""
    return np.exp(-0.5 * ((np.asarray(det_y, dtype=np.float64) - centre) / width) ** 2)


def centred(temp, t0=80.0):
    """An offset coefficient: the temperature above t0."""
    return np.asarray(temp, dtype=np.float64) - t0


def plane(det_x, det_y):
    """An offset basis of two functions: a plane over the detector."""
    return [np.asarray(det_x, dtype=np.float64), np.asarray(det_y, dtype=np.float64)]


def inverse_sigma(variance):
    """A derived variable and the observation weight: 1 / sigma."""
    return 1.0 / np.sqrt(np.maximum(np.asarray(variance, dtype=np.float64), 1e-12))


def qe_map(geometry, power=1.0):
    """A detector map computed from the geometry."""
    return np.ones(geometry.shape) ** power


def frame_phase(frames, period=10.0):
    """A per-frame value computed from the frame list."""
    return np.arange(len(frames)) % period


def night(exposure, length=4):
    """A per-frame value computed from a frame variable."""
    return np.asarray(exposure) // length


def quiet(frame, level=0.0):
    """A per-observation variable computed from the whole frame."""
    return np.full(np.shape(frame.raw()), level)


def known_sky(ref_wcs, ref_shape, scale=1.0):
    """A reference-grid map computed from the grid."""
    return np.full(ref_shape, scale)


def gains_sum_to_zero(gain, target=0.0):
    """A prior's rows: the term's unknowns sum to ``target``."""
    cols = gain.col_base + np.arange(gain.size)
    return np.zeros(cols.size, dtype=np.int64), cols, np.ones(cols.size), np.full(1, target)


def subtract_dark(ctx):
    """A frame hook (a function)."""
    return ctx.sub_data


class Scale:
    """A frame hook with state (an object)."""

    def __init__(self, factor):
        self.factor = factor

    def __call__(self, ctx):
        return ctx.sub_data * self.factor


@dataclass(frozen=True, kw_only=True)
class Prism(sc.Instrument):
    """A spectral imager of this module: a 32 x 48 detector cut into ``bands`` bands up the detector
    (the chunk map's spectral axis) x ``cols`` columns (its group axis), with a wavelength map; its
    exposures' data-quality bit 7 is ignored unless a recipe says otherwise."""
    bands: int = 8
    cols: int = 2
    tag: str = 'Prism'
    unit: str = 'MJy/sr'
    capabilities = ('wavelength', 'spectral_axis')

    def geometry(self, oversample=1):
        rows = np.arange(32) * self.bands // 32
        cols = np.arange(48) * self.cols // 48
        det = (rows[:, None] * self.cols + cols[None, :]).astype(np.int32)
        axes = sc.ChunkAxes.row_major(('band', 'col'), (self.bands, self.cols), ('y', 'x'))
        lvf = sc.ChunkMap('lvf', det, det, axes=axes, adjacency_axes=('col',), spectral_axis='band',
                          group_axis='col')
        wave = np.broadcast_to(np.linspace(1.0, 1.5, 32, dtype=np.float32)[:, None], (32, 48)).copy()
        return sc.Geometry((32, 48), oversample, maps=[lvf], aux={'wave': wave}, wavelength='wave')

    def layout(self):
        return sc.ExposureLayout(sci_ext=[1], dq_ext=[2], detector_ids=[0], ref_use_ext=(1,),
                                 default_ignore_bits=(7,), cache_tag='headers_Prism')


# The settings record this module's functions and classes as "test_fingerprints:<name>", the name
# pytest's default import mode gives it. Imported under another name (--import-mode=importlib, a
# tests/__init__.py), it is registered under that one too and they are renamed into it, so the
# fingerprints, chained ones included, stay those of the golden.
if __name__ != MODULE:
    sys.modules[MODULE] = sys.modules[__name__]
    for _obj in (ramp, season, bump, centred, plane, inverse_sigma, qe_map, frame_phase, night, quiet, known_sky,
                 gains_sum_to_zero, subtract_dark, Scale, Prism):
        _obj.__module__ = MODULE


# ============================================================================
# files with fixed bytes
# ============================================================================
def write_reference(path, shape):
    """A reference grid ``ref.fits`` written byte by byte: a FITS primary header (a TAN WCS and the
    grid's shape) and float64 zeros, the layout ``wcs_helper.save_to_fits`` writes."""
    ny, nx = shape
    cards = [('SIMPLE', 'T'), ('BITPIX', '-64'), ('NAXIS', '2'), ('NAXIS1', str(nx)), ('NAXIS2', str(ny)),
             ('WCSAXES', '2'), ('CRPIX1', f'{nx / 2 + 0.5}'), ('CRPIX2', f'{ny / 2 + 0.5}'),
             ('CDELT1', f'{-PIXEL_SCALE / 3600:.12f}'), ('CDELT2', f'{PIXEL_SCALE / 3600:.12f}'),
             ('CUNIT1', "'deg'"), ('CUNIT2', "'deg'"), ('CTYPE1', "'RA---TAN'"), ('CTYPE2', "'DEC--TAN'"),
             ('CRVAL1', '270.0'), ('CRVAL2', '66.5')]
    header = ''.join((f"{k:<8}= {v:<20}" if v.startswith("'") else f"{k:<8}= {v:>20}").ljust(80)
                     for k, v in cards) + 'END'.ljust(80)
    header = header.ljust(-(-len(header) // 2880) * 2880).encode('ascii')
    data = bytes(8 * ny * nx)
    data += bytes(-len(data) % 2880)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'wb') as f:
        f.write(header + data)


def write_npy(path, array):
    """A ``.npy`` file written byte by byte (format 1.0, little-endian float32)."""
    array = np.ascontiguousarray(array, dtype='<f4')
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': %r, }" % (tuple(array.shape),)
    header += ' ' * (-(10 + len(header) + 1) % 64) + '\n'
    with open(path, 'wb') as f:
        f.write(b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header.encode('latin1') + array.tobytes())


def write_settings_files(directory):
    """The files the settings name: a detector map, a sky map, and a solved sky's cal (any bytes)
    with the sidecar that names its fingerprint."""
    os.makedirs(directory, exist_ok=True)
    files = {'flat': os.path.join(directory, 'flat.npy'), 'known': os.path.join(directory, 'known.npy'),
             'prior': os.path.join(directory, 'cal_prior.h5')}
    write_npy(files['flat'], np.linspace(0.9, 1.1, 32 * 48).reshape(32, 48))
    write_npy(files['known'], np.linspace(0.0, 1.0, REF_SHAPE[0] * REF_SHAPE[1]).reshape(REF_SHAPE))
    with open(files['prior'], 'wb') as f:
        f.write(b'a solved sky')
    with open(files['prior'] + '.json', 'w') as f:
        json.dump({'fingerprint': SOLVED_FINGERPRINT}, f)
    return files


# ============================================================================
# the settings
# ============================================================================
QE = np.linspace(0.8, 1.2, 64 * 64, dtype=np.float32).reshape(64, 64)
CAMERA = sc.Camera((64, 64), chunks=4, dq_ext=2, tag='Toy')
SPECTRAL = sc.spectral([sc.Sky('line', times=sc.gaussian(1.25, sigma=0.08))], polynomial=sc.Poly(2, window=range(0, 8)))


def lines():
    """Sky terms of every kind of coefficient of the wavelength, damping set and left unset."""
    return [spherex.line('aromatic'),                                              # a shipped template
            spherex.line('aliphatic', 5e-3),
            sc.Sky('broad', times=sc.gaussian(3.40, width='bandwidth', intrinsic_var=2.89e-4)),
            sc.Sky('narrow', times=sc.gaussian(3.30, sigma=0.02), damping=0.0),
            sc.Sky('pah', times=sc.catalog('pah_3p29', center=3.29)),
            sc.Sky('tabulated', times=sc.template(x=[3.2, 3.3, 3.4], y=[0.0, 1.0, 0.0])),
            sc.Sky('tilt', times=sc.linear(3.3, 0.1))]


def model_with_everything(files):
    """An ``sc.Model`` with every kind of sky and offset term, data-variable source, weight and prior."""
    return sc.Model(
        sky=[sc.Sky(damping=1e-3),
             sc.Sky('ramp', times=sc.Function(ramp, of='det_x', scale=2.0)),
             sc.Sky('season', times=season),
             sc.Sky('bump', times=sc.by_value(bump)),
             sc.Sky('scaled', times='qe', damping=0.5)],
        offsets=[sc.Offsets('chunks', smooth=0.05, smooth_along=('band',), smooth_step=1, mean_zero=True,
                            poly_prior=(sc.Poly(1, along='band', window=range(0, 8), weight=0.2),)),
                 sc.Offsets('thermal', per='all', times=centred, mean_zero=True, exact_group_rows=True),
                 sc.Offsets('gradient', on='detector', basis=plane, n=2, damping=1e-3),
                 sc.Offsets('nightly', per='night', smooth=0.1, render='block')],
        scalar=False,
        variables={'qe': sc.DetectorMap(QE[:32, :48].copy()), 'flat': sc.DetectorMap(files['flat']),
                   'gain': sc.DetectorMap(qe_map, power=2.0), 'temp': sc.Header('TEMP', default=80.0),
                   'phase': sc.PerFrame(frame_phase, period=10.0), 'night': sc.PerFrame(night, of='exposure', length=4),
                   'known': sc.SkyMap(files['known']), 'model_sky': sc.SkyMap(known_sky, scale=2.0),
                   'prior': sc.SolvedSky(files['prior'], term='continuum'), 'variance': sc.Layer(),
                   'sigma': sc.Derived(inverse_sigma), 'quiet': sc.FrameFunction(quiet, level=0.5)},
        weight=sc.Function(inverse_sigma, of='variance'),
        priors=[sc.Prior(gains_sum_to_zero, 'gradient', weight=10.0, name='scale', target=0.0),
                sc.priors.frame_smoothness('chunks', 'temp', power=0.5, weight=0.2),
                sc.priors.sky_smoothness('ramp', weight=0.1),
                sc.priors.toward('scaled', 'known', weight=0.5)])


@dataclass(frozen=True)
class Case:
    """A recipe on an instrument for a job, optionally with tiles (``tile_frames``: each tile's frames)
    and passes: its cal, and the mosaic, tile cals, stitched cal and pass products it implies."""
    name: str
    instrument: object
    recipe: object
    job: object = None
    passes: object = None
    tiles: object = None
    tile_frames: tuple = ()


def cases(files):
    polybasis = sc.spectral([spherex.line('aromatic', 5e-3)], polynomial=sc.Poly(2, window=range(200, 321)))
    aromatic = spherex.window('Aromatic')
    return [
        # the instruments, each with its default job where it has one
        Case('spherex_channel', sc.SPHEREx(4), sc.Recipe(sc.continuum(), name='c'), spherex.channel(17)),
        Case('spherex_settings_window', sc.SPHEREx(3, num_col=10, calib_dir='/data/spherex/calibration',
                                                   lvf_dir='/data/spherex/lvf'),
             sc.Recipe(sc.continuum(), coadd=sc.Coadd(oversample=2), name='w'), aromatic),
        Case('spherex_group_two_block', sc.SPHEREx(4), sc.Recipe(sc.two_block(), name='k2'), spherex.group(17, 18)),
        Case('euclid', sc.Euclid(band='Y', chunks=40, strips=40), sc.Recipe(sc.continuum(), name='e')),
        Case('camera_maps', sc.Camera((64, 64), chunks=4, dq_ext=2, tag='Toy', detector_maps={'qe': QE},
                                      headers={'temp': 'TEMP'}), sc.Recipe(sc.continuum(), name='c')),
        Case('user_instrument', Prism(), sc.Recipe(sc.continuum(), name='p')),
        # the models
        Case('continuum_smooth_poly', CAMERA,
             sc.Recipe(sc.continuum(smooth=0.3, poly_prior=sc.Poly(1, weight=0.5)), name='cp')),
        Case('spectral_lines', sc.SPHEREx(4),
             sc.Recipe(sc.spectral(lines(), polynomial=sc.Poly(2, window=range(200, 321))), coadd=None,
                       name='lines'), spherex.window('Multiline3', subchannels=range(200, 321))),
        Case('spectral_soft_poly', sc.SPHEREx(4),
             sc.Recipe(sc.spectral([spherex.line('aromatic', 5e-3)], smooth=0.2,
                                   poly_prior=sc.Poly(3, along='subchannel', window=range(200, 321)), damping=0.05),
                       name='soft'), spherex.window('W', subchannels=range(200, 321))),
        Case('two_block_settings', sc.SPHEREx(4),
             sc.Recipe(sc.two_block('readout', smooth=0.2, second_smooth=0.05, along='subchannel', damping=0.2),
                       coadd=None, name='k2s'), spherex.channel(3)),
        Case('model_with_everything', Prism(), sc.Recipe(model_with_everything(files), name='all')),
        # the fit
        Case('fit_settings', sc.SPHEREx(4),
             sc.Recipe(sc.continuum(), fit=sc.Fit(120, clip=sc.Clip(4.0, per=SUBCHANNEL), ignore_flags=(3, 21),
                                                  tolerance=(1e-7, 1e-9), method='lsmr', damp=1e-6, precondition=False,
                                                  float32=False, shot_noise_weights=True, use_mask=False,
                                                  line_fisher_threshold=5.0, frame_hook=subtract_dark,
                                                  raw_frame_hook=Scale(2.0)),
                       coadd=None, name='fit'), spherex.channel(17)),
        Case('fit_no_clip', sc.Euclid(band='H'), sc.Recipe(sc.continuum(), fit=sc.Fit(clip=None), name='nc')),
        Case('fit_clip_per_chunk', CAMERA,
             sc.Recipe(sc.continuum(), fit=sc.Fit(clip=sc.Clip(3.0, per='chunk')), coadd=None, name='pc')),
        Case('fit_clip_mapping', CAMERA,
             sc.Recipe(sc.continuum(), fit=sc.Fit(clip=sc.Clip(3.0, per=sc.ChunkGroups.mapping([0, 0, 1, 1] * 4,
                                                                                               map='grid'))),
                       coadd=None, name='pm')),
        Case('fit_clip_edges', sc.SPHEREx(4),
             sc.Recipe(sc.continuum(), fit=sc.Fit(clip=sc.Clip(3.0, edges=(3.2, 3.3, 3.4), variable='wavelength')),
                       coadd=None, name='pe'), spherex.channel(17)),
        # the coadd and the numerics
        Case('coadd_settings', CAMERA,
             sc.Recipe(sc.continuum(), coadd=sc.Coadd(None, std=False, use_mask=False, ignore_flags=(1,),
                                                      shot_noise_weights=True, oversample=2, instrument_maps=False,
                                                      min_chunk_coverage=0.05, subtract_offsets=False,
                                                      normalize_offsets=True, frame_hook=Scale(0.5)),
                       name='co')),
        Case('numerics', CAMERA,
             sc.Recipe(sc.continuum(), numerics=sc.Numerics(4, batch=10, mosaic_batch=20, coadd_batch=30,
                                                            rmatvec_threads=2), name='n')),
        # the passes
        Case('passes_default', sc.SPHEREx(4), sc.Recipe(polybasis, coadd=None, name='np'), aromatic,
             passes=sc.Passes(3)),
        Case('passes_settings', sc.SPHEREx(4), sc.Recipe(polybasis, coadd=None, name='np'), aromatic,
             passes=sc.Passes(3, order='sky_first', ends_on_offset=True,
                              init_clip=sc.Clip(2.5, per=SUBCHANNEL, ignore_flags=(21,)),
                              sky_clip=sc.Clip(4.0, per=SUBCHANNEL),
                              offset=sc.Refit(2, clip=sc.Clip(2.0, per=SUBCHANNEL), bright_cut=None, min_pixels=100,
                                              segments=((200, 260), (261, 320)), ridge=0.1),
                              stop_tol=1e-4, sky_merge='stitch', keep_moments=True)),
        Case('passes_init_sigma', sc.SPHEREx(4), sc.Recipe(polybasis, coadd=None, name='np'), aromatic,
             passes=sc.Passes(5, init_clip=4.0)),
        # the tiles
        Case('tiles_grid', CAMERA, sc.Recipe(sc.continuum(), coadd=None, name='t'),
             tiles=sc.Tiles((1, 2), overlap=10, names=('W', 'E'), assign='overlap', halo=5),
             tile_frames=(FRAMES[:16], FRAMES[8:])),
        Case('tiles_grid_named_by_the_engine', CAMERA, sc.Recipe(sc.continuum(), coadd=None, name='t4'),
             tiles=sc.Tiles((2, 2), overlap=20), tile_frames=tuple(FRAMES[i::4] for i in range(4))),
        Case('tiles_boxes_passes', sc.SPHEREx(4), sc.Recipe(polybasis, coadd=None, name='tp'), aromatic,
             passes=sc.Passes(3),
             tiles=sc.Tiles(boxes={'north': (0, 70, 0, 160), 'south': (50, 120, 0, 160)}, tile_name='{name}_t{tile}',
                            stitched_name='{name}_all'),
             tile_frames=(FRAMES[:14], FRAMES[10:])),
    ]


# ============================================================================
# the canonical inputs and fingerprints
# ============================================================================
def _entry(inputs, kind=None, path=None):
    """A product's golden entry: its canonical inputs (as products.write_sidecar records them) and
    their fingerprint (as products.fingerprint computes it)."""
    canonical = json.loads(json.dumps(products._canonical(inputs), sort_keys=True, default=str))
    fp = products.fingerprint(inputs)
    assert products.fingerprint(canonical) == fp           # the recorded inputs give the same fingerprint
    out = {'fingerprint': fp, 'inputs': canonical}
    if kind is not None:
        out.update(kind=kind, path=path)
    return out


def _tile_specs(tiles):
    """The tiles as the engine resolves them on ``REF_SHAPE``: a grid by ``sc.make_tile_grid``, boxes as given."""
    if tiles.boxes is not None:
        return [sc.TileSpec(name, tuple(box)) for name, box in tiles.boxes.items()]
    return sc.make_tile_grid(REF_SHAPE, tiles.grid[0], tiles.grid[1], overlap_px=tiles.overlap,
                             names=None if tiles.names is None else list(tiles.names))


def _pass_types(passes):
    first, second = ('sky', 'off') if passes.order == 'sky_first' else ('off', 'sky')
    return [first if i % 2 == 0 else second for i in range(2, passes.n + 1)]


def settings_entries(field_dir, files):
    """The golden entries of the first part, ``"<case>/<product>"``: a case's cal (or its tile cals,
    with the ``Book``'s tile key, and their stitched cal), its mosaic, its pass products (made from
    the cal or the stitched cal), chained by fingerprint as the ``Book`` chains them."""
    out = {}
    for case in cases(files):
        field = sc.Field(field_dir, case.instrument, PIXEL_SCALE)
        job = case.job or case.instrument.default_jobs()[0]
        if case.tiles is None:
            init = products.cal_inputs(field, case.recipe, job, FRAMES, passes=case.passes)
            out[f'{case.name}/cal'] = _entry(init)
            if case.recipe.coadd is not None and case.passes is None:
                out[f'{case.name}/mosaic'] = _entry(products.mosaic_inputs(field, case.recipe, job, CAL_FINGERPRINT,
                                                                           FRAMES))
        else:
            book = products.Book(field, case.recipe, passes=case.passes, tiles=case.tiles)
            tiles = {}
            for spec, frames in zip(_tile_specs(case.tiles), case.tile_frames):
                inputs = products.cal_inputs(field, case.recipe, job, frames, passes=case.passes,
                                             tile=book.tile_key(spec))
                out[f'{case.name}/tile {spec.name}'] = _entry(inputs)
                tiles[spec.name] = products.fingerprint(inputs)
            init = products.stitched_inputs(tiles, len(case.recipe.model.sky) > 1)
            out[f'{case.name}/stitched'] = _entry(init)
        if case.passes is not None:
            for i, t in enumerate(_pass_types(case.passes), start=2):
                out[f'{case.name}/pass {i} {t}'] = _entry(products.pass_inputs(products.fingerprint(init),
                                                                               case.passes, i, t))
    return out


def write_frames(directory, boxes, shape):
    """Frame files (zeros) on ``boxes`` (their top-left corners) of the reference grid."""
    h, w = shape
    det_y, det_x = np.mgrid[0:h, 0:w].astype(np.float32)
    os.makedirs(directory, exist_ok=True)
    for k, (y0, x0) in enumerate(boxes):
        write_frame(standard_frame_path(directory, k, 0), np.zeros(shape), [y0, y0 + h, x0, x0 + w],
                    np.stack([det_x, det_y]))


def plan_entries(root):
    """The golden entries of the second part, ``"<plan>/<product path in the field>"``: every product
    of ``field.plan(...)`` (a mosaic by its name only: its inputs read its cal)."""
    camera = sc.Field(os.path.join(root, 'camera'), CAMERA, PIXEL_SCALE)
    prism = sc.Field(os.path.join(root, 'prism'), Prism(), PIXEL_SCALE)
    euclid = sc.Field(os.path.join(root, 'euclid'), sc.Euclid(band='J'), PIXEL_SCALE)
    for field, boxes, shape in ((camera, [(0, 0), (0, 48), (0, 96), (28, 20), (28, 70), (56, 0), (56, 48), (56, 96)],
                                 (64, 64)),
                                (prism, [(0, 0), (0, 56), (0, 112), (44, 30), (44, 86), (88, 0), (88, 56), (88, 112)],
                                 (32, 48)),
                                (euclid, [(0, 0), (40, 60), (80, 120)], (40, 40))):
        write_reference(os.path.join(field.path, 'ref.fits'), REF_SHAPE)
        write_frames(os.path.join(field.path, 'reprojected'), boxes, shape)
    # the camera with per-pixel maps and header variables, on the camera's field
    camera_maps = camera.replace(instrument=sc.Camera((64, 64), chunks=4, dq_ext=2, tag='Toy', detector_maps={'qe': QE},
                                                      headers={'temp': 'TEMP'}))
    plans = [
        ('plan camera', camera, camera.plan(sc.Recipe(sc.continuum(), name='c'))),
        ('plan camera tiles', camera, camera.plan(
            sc.Recipe(sc.continuum(), coadd=None, name='t'),
            tiles=sc.Tiles((1, 2), overlap=10, names=('W', 'E'), assign='overlap', halo=4))),
        ('plan camera maps', camera_maps, camera_maps.plan(sc.Recipe(sc.continuum(), coadd=None, name='m'))),
        ('plan euclid', euclid, euclid.plan(sc.Recipe(sc.continuum(), name='e'))),
        ('plan prism passes', prism, prism.plan(sc.Recipe(SPECTRAL, coadd=None, name='np'), passes=sc.Passes(3))),
        ('plan prism tiles passes', prism, prism.plan(
            sc.Recipe(SPECTRAL, coadd=None, name='tp'), passes=sc.Passes(4, order='sky_first', sky_merge='stitch'),
            tiles=sc.Tiles(boxes={'north': (0, 70, 0, 160), 'south': (50, 120, 0, 160)}))),
    ]
    out = {}
    for name, field, plan in plans:
        for prod in plan.products:
            path = os.path.relpath(prod.path, field.path)
            if prod.kind == 'mosaic':
                out[f'{name}/{path}'] = {'kind': 'mosaic', 'path': path}
            else:
                out[f'{name}/{path}'] = _entry(prod.inputs(), kind=prod.kind, path=path)
    return out


@pytest.fixture(scope='module')
def computed(tmp_path_factory):
    """Every golden entry, computed now (and the temporary directory they were computed in)."""
    sc.set_progress(False)
    root = str(tmp_path_factory.mktemp('fingerprints'))
    field_dir = os.path.join(root, 'field')
    write_reference(os.path.join(field_dir, 'ref.fits'), REF_SHAPE)
    entries = settings_entries(field_dir, write_settings_files(os.path.join(root, 'files')))
    entries.update(plan_entries(os.path.join(root, 'plans')))
    return root, entries


def _text(entry):
    return json.dumps(entry, indent=1, sort_keys=True)


def test_product_fingerprints_are_the_golden(computed):
    _, entries = computed
    if os.environ.get(WRITE_VARIABLE):
        os.makedirs(os.path.dirname(GOLDEN), exist_ok=True)
        with open(GOLDEN, 'w') as f:
            json.dump({'about': "tests/test_fingerprints.py: the canonical inputs and fingerprint of each product. "
                                "Regenerating this file makes the sidecar of every product whose inputs changed "
                                "'different': do it only on purpose.",
                       'products': entries}, f, indent=1, sort_keys=True)
            f.write('\n')
    with open(GOLDEN) as f:
        golden = json.load(f)['products']
    for key, entry in golden.items():                       # the golden is consistent with itself
        if 'inputs' in entry:
            assert products.fingerprint(entry['inputs']) == entry['fingerprint'], key
    missing, extra = sorted(set(golden) - set(entries)), sorted(set(entries) - set(golden))
    changed = [k for k in sorted(set(golden) & set(entries)) if _text(golden[k]) != _text(entries[k])]
    if missing or extra or changed:
        lines = []
        if missing:
            lines.append(f"products of the golden no longer made: {missing}")
        if extra:
            lines.append(f"products not in the golden: {extra}")
        for key in changed[:6]:
            diff = difflib.unified_diff(_text(golden[key]).splitlines(), _text(entries[key]).splitlines(),
                                        f'golden {key}', f'now {key}', lineterm='', n=2)
            lines.append('\n'.join(diff))
        if len(changed) > 6:
            lines.append(f"... and {len(changed) - 6} more changed: {changed[6:]}")
        pytest.fail(f"product fingerprints differ from the golden ({len(changed)} changed, {len(missing)} no longer "
                    f"made, {len(extra)} new; {WRITE_VARIABLE}=1 regenerates it, and makes every such existing "
                    f"sidecar 'different'):\n" + '\n'.join(lines), pytrace=False)


def test_no_path_of_this_machine_enters_a_fingerprint(computed):
    root, entries = computed
    text = json.dumps(entries)
    for path in (root, os.path.realpath(root), _REPO, os.path.realpath(_REPO)):
        assert path not in text, f"{path} appears in the canonical inputs"
    for key, entry in entries.items():                    # the reference grid, by content (not its path)
        reference = entry.get('inputs', {}).get('reference')
        if reference is not None:
            assert isinstance(reference, dict) and set(reference) == {'sha256'}, (key, reference)
    assert 'object at 0x' not in text                      # no repr of an object (a memory address)
