"""The N-pass scheduler end to end (``Field.calibrate(passes=...)``) on a toy spectral instrument.

The instrument: a 64 x 64 detector cut into 16 bands (the spectral axis of the chunk map, 4 rows
each) x 2 columns (the group axis), with a wavelength map rising smoothly up the detector (and
curving a little across it): an ``sc.Instrument`` subclass whose geometry is :func:`toy_geometry`
(everything else is the contract's defaults).

The data: 20 frames written directly on a 96 x 96 reference grid (no reprojection), each the sky
``S_0 + S_1 * gauss(wavelength)`` (a continuum and one line) seen at a random pointing, plus a
per-frame offset that is a quadratic in the band per column and a per-frame scalar (what the
model's polynomial-basis offset term and scalar can represent).

Each schedule is solved and its products checked: the products and the monitor of every pass, the
damping every SKY pass records (each sky term's, as in the INIT pass). The schedules, n = 3 each:

=========  ===========  =======  ===================================================================
name       order        merge    damping
=========  ===========  =======  ===================================================================
skyfirst   sky_first    combine  the model's default (0.1 / 0.3)
offfirst   offset_first combine  each term its own (0.02 / 0.05); grouped clips along the band
                                 (init, sky and offset)
stitch     sky_first    stitch   the model's default
=========  ===========  =======  ===================================================================

Until October 2026 every schedule was also run from a TOML config and its products had to be
identical; their digests were recorded then (``workspace/api-redesign/evidence/stage4``). Before and
after a change to the run engine: ``SELFCAL_NPASS_TOY_DIGESTS=<file> pytest tests/test_npass_toy.py``
writes the digest of every product (the INIT cals and the pass products of every schedule) to
``<file>`` (JSON), to compare with the recorded ones. A product records the paths of the run (its
frames, the sky cal it was refitted against, the pieces of a stitch), which lie in a temporary
directory, so the file bytes differ from run to run: the digest is over every dataset (name, dtype,
shape and bytes, storage layout) and every attribute, with the temporary directory replaced by
``<run>``. Two runs give identical files.
"""
import hashlib
import json
import os
import shutil
import sys
from dataclasses import dataclass

import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import h5py  # noqa: E402
import hdf5plugin  # noqa: E402,F401  (the frames and products are compressed)
from astropy.wcs import WCS  # noqa: E402

import selfcal as sc  # noqa: E402
from selfcal.geometry import wcs_helper  # noqa: E402
from selfcal.io.frames import standard_frame_path, write_frame  # noqa: E402

DET = 64                            # detector side (px)
N_BAND, N_COL = 16, 2               # chunk grid: bands (spectral axis, 4 rows each) x columns (groups)
REF = 96                            # reference grid side (px)
PIX = 20.0                          # arcsec per pixel
N_FRAMES = 20
LINE_CENTER, LINE_SIGMA = 1.25, 0.08     # the line: a Gaussian of the wavelength (um)
WINDOW = range(0, N_BAND)           # the polynomial offset spans every band
TAG = 'ToySpec'
DIGESTS_VARIABLE = 'SELFCAL_NPASS_TOY_DIGESTS'


# ============================================================================
# the instrument
# ============================================================================
def toy_wavelength():
    """The wavelength (um) of every detector pixel: 1.0 at the bottom row to 1.5 at the top,
    plus a slight curvature across the columns."""
    y, x = np.mgrid[0:DET, 0:DET].astype(np.float64)
    return (1.0 + 0.5 * y / (DET - 1) + 0.02 * ((x - 31.5) / 31.5) ** 2).astype(np.float32)


def toy_chunk_ids():
    """Chunk id ``band * N_COL + col`` of every pixel: band = row // 4, column = col // 32."""
    rows = np.arange(DET) * N_BAND // DET
    cols = np.arange(DET) * N_COL // DET
    return (rows[:, None] * N_COL + cols[None, :]).astype(np.int32)


def toy_geometry(oversample):
    """The detector geometry: one chunk map whose axes are the band (spectral) and the column
    (group), the wavelength map and a constant width map."""
    det = toy_chunk_ids()
    axes = sc.ChunkAxes.row_major(('band', 'col'), (N_BAND, N_COL), ('y', 'x'))
    lvf = sc.ChunkMap('lvf', det, det, axes=axes, adjacency_axes=('col',), spectral_axis='band',
                      group_axis='col')
    return sc.Geometry((DET, DET), oversample, maps=[lvf],
                       aux={'wave': toy_wavelength(), 'width': np.full((DET, DET), 0.02, np.float32)},
                       wavelength='wave', width='width')


@dataclass(frozen=True, kw_only=True)
class ToySpectrograph(sc.Instrument):
    """The geometry; jobs, layout, job geometry and frame variables are the contract's defaults."""
    tag: str = TAG
    capabilities = ('wavelength', 'spectral_axis')

    def geometry(self, oversample=1):
        return toy_geometry(oversample)


# ============================================================================
# the data
# ============================================================================
def band_basis():
    """The mean-zero Chebyshev shapes T1, T2 of the band over the window, ``(N_BAND, 2)``."""
    lo, hi = WINDOW.start, WINDOW.stop - 1
    x = 2.0 * (np.arange(N_BAND) - lo) / (hi - lo) - 1.0
    shapes = np.stack([x, 2.0 * x ** 2 - 1.0], axis=1)
    return shapes - shapes.mean(axis=0)


def write_field(field_dir, rng):
    """``ref.fits`` and ``N_FRAMES`` frame files in ``<field_dir>/reprojected``: a continuum plus
    a line times a Gaussian of the wavelength, at random pointings, with per-frame offsets (a
    quadratic in the band per column) and scalars."""
    y, x = np.mgrid[0:REF, 0:REF].astype(np.float64)
    continuum = 5.0 + 0.01 * x + 0.005 * y + 0.3 * np.sin(x / 9.0) * np.cos(y / 11.0)
    line = 0.6 + 0.3 * np.sin(x / 7.0 + 1.0) * np.cos(y / 8.0)
    gauss = np.exp(-0.5 * ((toy_wavelength().astype(np.float64) - LINE_CENTER) / LINE_SIGMA) ** 2)
    chunk = toy_chunk_ids()
    band, col = chunk // N_COL, chunk % N_COL
    shapes = band_basis()[band]                                   # (DET, DET, 2)
    det_y, det_x = np.mgrid[0:DET, 0:DET].astype(np.float32)
    w = WCS(naxis=2)
    w.wcs.ctype = ['RA---TAN', 'DEC--TAN']
    w.wcs.crpix = [REF / 2 + 0.5, REF / 2 + 0.5]
    w.wcs.crval = [180.0, 30.0]
    w.wcs.cdelt = [-PIX / 3600.0, PIX / 3600.0]
    wcs_helper.save_to_fits(w, (REF, REF), os.path.join(field_dir, 'ref.fits'))
    frames = os.path.join(field_dir, 'reprojected')
    os.makedirs(frames)
    for k in range(N_FRAMES):
        oy, ox = int(rng.integers(0, REF - DET + 1)), int(rng.integers(0, REF - DET + 1))
        coeffs = rng.normal(0.0, 0.1, (N_COL, 2))
        offset = (shapes * coeffs[col]).sum(axis=-1) + rng.normal(0.0, 0.5)
        values = (continuum[oy:oy + DET, ox:ox + DET] + line[oy:oy + DET, ox:ox + DET] * gauss + offset
                  + rng.normal(0.0, 0.01, (DET, DET)))
        write_frame(standard_frame_path(frames, k, 0), values, [oy, oy + DET, ox, ox + DET], np.stack([det_x, det_y]))


# ============================================================================
# the schedules
# ============================================================================
BAND = sc.ChunkGroups.along('band')
FIT = sc.Fit(100, tolerance=1e-8, line_fisher_threshold=1.0)
NUMERICS = sc.Numerics(2, batch=4)


def _line(damping=None):
    return sc.Sky('line', times=sc.gaussian(LINE_CENTER, sigma=LINE_SIGMA), damping=damping)


def _polynomial():
    return sc.Poly(2, window=WINDOW)


@dataclass(frozen=True)
class Schedule:
    """One N-pass run: its name (the recipe's name, the products' suffix), its model and passes."""
    name: str
    model: object
    passes: object

    @property
    def recipe(self):
        return sc.Recipe(self.model, fit=FIT, coadd=None, numerics=NUMERICS, name=self.name)

    @property
    def pass_types(self):
        n, order = self.passes.n, self.passes.order
        first, second = ('sky', 'off') if order == 'sky_first' else ('off', 'sky')
        return [first if i % 2 == 0 else second for i in range(2, n + 1)]


SCHEDULES = (
    Schedule('skyfirst', sc.spectral([_line()], polynomial=_polynomial()),
             sc.Passes(3, order='sky_first', ends_on_offset=True, offset=sc.Refit(2))),
    Schedule('offfirst', sc.spectral([_line(0.05)], polynomial=_polynomial(), damping=0.02),
             sc.Passes(3, order='offset_first', init_clip=sc.Clip(4.0, per=BAND), sky_clip=sc.Clip(5.0, per=BAND),
                       offset=sc.Refit(2, clip=sc.Clip(2.5, per=BAND), min_pixels=500))),
    Schedule('stitch', sc.spectral([_line()], polynomial=_polynomial()),
             sc.Passes(3, order='sky_first', ends_on_offset=True, sky_merge='stitch',
                       offset=sc.Refit(2, min_pixels=500))),
)


# ============================================================================
# digesting products
# ============================================================================
def _plain(value, run_root):
    """An HDF5 value as JSON: text with the run's temporary directory replaced by ``<run>`` (its
    fixed-length dtype holds the length of that path, so it is left out); numbers by dtype, shape
    and the sha256 of their bytes, and their values when there are a few (attributes)."""
    arr = np.asarray(value)
    if arr.dtype.kind in 'SUO':
        def text(v):
            v = v.decode() if isinstance(v, bytes) else str(v)
            return v.replace(run_root, '<run>')
        return {'text': [text(v) for v in arr.ravel().tolist()], 'shape': list(arr.shape)}
    out = {'dtype': arr.dtype.str, 'shape': list(arr.shape),
           'sha256': hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()}
    if arr.size <= 16:
        out['values'] = arr.ravel().tolist()
    return out


def describe(path, run_root):
    """Every link of an HDF5 file in name order (hard-link aliases included): groups with their
    attributes, datasets with their values, storage layout and attributes, as JSON."""
    out = {}

    def attrs(obj):
        return {k: _plain(obj.attrs[k], run_root) for k in sorted(obj.attrs)}

    def filters(ds):
        plist = ds.id.get_create_plist()
        return [[int(code), [int(v) for v in values]]
                for code, _, values, _ in (plist.get_filter(i) for i in range(plist.get_nfilters()))]

    def walk(group, prefix):
        for name in sorted(group):
            obj = group[name]
            key = prefix + name
            if isinstance(obj, h5py.Group):
                out[key] = {'group': True, 'attrs': attrs(obj)}
                walk(obj, key + '/')
            else:
                out[key] = {'value': _plain(obj[()], run_root), 'chunks': list(obj.chunks or ()),
                            'filters': filters(obj), 'attrs': attrs(obj)}

    with h5py.File(path, 'r') as f:
        out['/'] = {'group': True, 'attrs': attrs(f)}
        walk(f, '')
    return out


def digest(path, run_root):
    """The sha256 of :func:`describe`'s JSON."""
    text = json.dumps(describe(path, run_root), sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()


# ============================================================================
# the test
# ============================================================================
@pytest.fixture(scope='module')
def toy(tmp_path_factory):
    """The toy field (``<root>/out/toy``: ref.fits and the frames) and the digests of the products,
    written at the end when ``SELFCAL_NPASS_TOY_DIGESTS`` names a file."""
    sc.set_progress(False)
    root = str(tmp_path_factory.mktemp('npass_toy'))
    write_field(os.path.join(root, 'out', 'toy'), np.random.default_rng(2026))
    digests = {}
    yield root, digests
    target = os.environ.get(DIGESTS_VARIABLE)
    if target and len(digests) == len(SCHEDULES):
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        with open(target, 'w') as f:
            json.dump({'about': "tests/test_npass_toy.py: the products of each N-pass schedule, by the sha256 of "
                                "every dataset and attribute (paths of the run's temporary directory written <run>)",
                       'schedules': {k: digests[k] for k in sorted(digests)}}, f, indent=1, sort_keys=True)
            f.write('\n')
    shutil.rmtree(root, ignore_errors=True)


@pytest.mark.parametrize('schedule', SCHEDULES, ids=[s.name for s in SCHEDULES])
def test_npass_schedule(toy, schedule):
    root, digests = toy
    field_dir = os.path.join(root, 'out', 'toy')
    stem = f'cal_{TAG}_All_{schedule.name}'
    expected = [f'{stem}.h5'] + [f'{stem}_pass{i}{t}.h5' for i, t in enumerate(schedule.pass_types, start=2)]
    if schedule.passes.sky_merge == 'stitch':        # the per-tile solves (one tile untiled) the stitch merges
        expected += [f'{stem}_pass{i}sky_all.h5' for i, t in enumerate(schedule.pass_types, start=2) if t == 'sky']

    field = sc.Field(field_dir, ToySpectrograph(), PIX)
    result = field.calibrate(schedule.recipe, passes=schedule.passes,
                             compute=sc.Compute(os.path.join(root, f'scratch_py_{schedule.name}'), workers=2, stage=None))
    assert sorted(result.passes) == list(range(1, schedule.passes.n + 1))
    cal_dir = os.path.join(field_dir, 'calibration')
    made = sorted(n for n in os.listdir(cal_dir) if n.startswith(f'{stem}.') or n.startswith(f'{stem}_pass'))
    assert [n for n in made if n.endswith('.h5')] == sorted(expected), made
    with open(os.path.join(cal_dir, f'{stem}_npass_monitor.json')) as f:
        monitor = json.load(f)
    assert [(r['pass'], r['type']) for r in monitor] == \
        [(1, 'init')] + [(i, 'sky' if t == 'sky' else 'offset') for i, t in enumerate(schedule.pass_types, start=2)]
    # the SKY passes damp each term as the INIT pass did
    want = [0.02, 0.05] if schedule.name == 'offfirst' else [0.1, 0.3]
    for i, t in enumerate(schedule.pass_types, start=2):
        if t == 'sky' and schedule.passes.sky_merge == 'combine':
            with h5py.File(os.path.join(cal_dir, f'{stem}_pass{i}sky.h5'), 'r') as f:
                assert f.attrs['damp_weights'].tolist() == want
    digests[schedule.name] = {'schedule': ['init'] + ['sky' if t == 'sky' else 'offset' for t in schedule.pass_types],
                              'products': {name: digest(os.path.join(cal_dir, name), root) for name in sorted(expected)}}
