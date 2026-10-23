"""The N-pass passes with any offset model (``Field.calibrate(passes=...)``): the SKY pass subtracts
the offsets of every term the model has, and the OFFSET pass solves the model's own offset system with
the sky held fixed, every frame directly from its own rows -- nothing in the passes knows the offsets'
form.

A toy spectral instrument (a 64 x 64 detector of 16 bands x 2 columns, a wavelength rising up the
detector, and a second chunk map of the bands alone) and 20 noise-free frames on a 96 x 96 grid, the
sky a continuum plus a line times a Gaussian of the wavelength. Two offset models, each with data
it can represent exactly:

=========== =======================================================================================
plain       a free offset per chunk with no constant over a frame's chunks (``exclude=Poly(0)``) and
            the per-frame scalar (the term of one chunk)
poly_slope  a quadratic along the bands per column, a free x-slope per band (``times`` a function of
            the detector column, as SPHEREx's ``frame_slope``) and the scalar
=========== =======================================================================================

For each, the joint objective (the weighted residual of the data against the sky and offsets the
passes hold, computed here from the products independently of the library) never increases from pass
to pass (each pass is an exact block minimisation), and from a deliberately unconverged first solve
the passes bring the sky much closer to the truth. A third model, a detector-fixed pattern shared by
every frame (``per="all"``), would tie the frames of an OFFSET pass together; the plan refuses it.
"""
import json
import os
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
import hdf5plugin  # noqa: E402,F401
from astropy.wcs import WCS  # noqa: E402

import selfcal as sc  # noqa: E402
from selfcal.geometry import wcs_helper  # noqa: E402
from selfcal.io.frames import standard_frame_path, write_frame  # noqa: E402
from selfcal.models.bases import tilt  # noqa: E402

DET, N_BAND, N_COL, REF, PIX = 64, 16, 2, 96, 20.0
N_FRAMES = 20
LINE_CENTER, LINE_SIGMA = 1.25, 0.08
TAG = 'ToyAny'
CENTRE, HALF = (DET - 1) / 2.0, DET / 2.0


def wavelength():
    y, x = np.mgrid[0:DET, 0:DET].astype(np.float64)
    return (1.0 + 0.5 * y / (DET - 1) + 0.02 * ((x - 31.5) / 31.5) ** 2).astype(np.float32)


def chunk_ids():
    rows = np.arange(DET) * N_BAND // DET
    cols = np.arange(DET) * N_COL // DET
    return (rows[:, None] * N_COL + cols[None, :]).astype(np.int32)


def band_ids():
    return np.repeat((np.arange(DET) * N_BAND // DET)[:, None], DET, axis=1).astype(np.int32)


def geometry(oversample):
    det = chunk_ids()
    lvf = sc.ChunkMap('lvf', det, det, axes=sc.ChunkAxes.row_major(('band', 'col'), (N_BAND, N_COL), ('y', 'x')),
                      adjacency_axes=('col',), spectral_axis='band', group_axis='col')
    bands = band_ids()
    band = sc.ChunkMap('band', bands, bands, axes=sc.ChunkAxes.row_major(('band',), (N_BAND,), ('y',)),
                       adjacency_axes=('band',), spectral_axis='band')
    return sc.Geometry((DET, DET), oversample, maps=[lvf, band],
                       aux={'wave': wavelength(), 'width': np.full((DET, DET), 0.02, np.float32)},
                       wavelength='wave', width='width')


@dataclass(frozen=True, kw_only=True)
class ToyInstrument(sc.Instrument):
    tag: str = TAG
    capabilities = ('wavelength', 'spectral_axis')

    def geometry(self, oversample=1):
        return geometry(oversample)


def sky_truth():
    y, x = np.mgrid[0:REF, 0:REF].astype(np.float64)
    continuum = 5.0 + 0.01 * x + 0.005 * y + 0.3 * np.sin(x / 9.0) * np.cos(y / 11.0)
    line = 0.6 + 0.3 * np.sin(x / 7.0 + 1.0) * np.cos(y / 8.0)
    return continuum, line


def offsets_of(kind, rng):
    """One frame's offset on the detector, of the form ``kind``'s model represents."""
    chunk, band = chunk_ids(), band_ids()
    col = chunk % N_COL
    det_x = np.mgrid[0:DET, 0:DET][1].astype(np.float64)
    scalar = rng.normal(0.0, 0.5)
    if kind == 'plain':
        per_chunk = rng.normal(0.0, 0.1, N_BAND * N_COL)
        return per_chunk[chunk] - per_chunk.mean() + scalar
    if kind == 'poly_slope':
        u = 2.0 * np.arange(N_BAND) / (N_BAND - 1) - 1.0
        shapes = np.stack([u, 2.0 * u ** 2 - 1.0], axis=1)
        shapes -= shapes.mean(axis=0)
        coeffs = rng.normal(0.0, 0.1, (N_COL, 2))
        poly = (shapes[band] * coeffs[col]).sum(axis=-1)
        slope = rng.normal(0.0, 0.05, N_BAND)[band] * tilt(det_x, CENTRE, HALF)
        return poly + slope + scalar
    if kind == 'shared':
        return scalar                          # + the fixed pattern, added by the caller
    raise ValueError(kind)


def write_field(field_dir, kind, rng):
    continuum, line = sky_truth()
    gauss = np.exp(-0.5 * ((wavelength().astype(np.float64) - LINE_CENTER) / LINE_SIGMA) ** 2)
    pattern = rng.normal(0.0, 0.1, N_BAND * N_COL)
    pattern = (pattern - pattern.mean())[chunk_ids()]
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
        offset = offsets_of(kind, rng) + (pattern if kind == 'shared' else 0.0)
        values = continuum[oy:oy + DET, ox:ox + DET] + line[oy:oy + DET, ox:ox + DET] * gauss + offset
        write_frame(standard_frame_path(frames, k, 0), values, [oy, oy + DET, ox, ox + DET], np.stack([det_x, det_y]))


def model_of(kind):
    sky = [sc.Sky(damping=0.0), sc.Sky('line', times=sc.gaussian(LINE_CENTER, sigma=LINE_SIGMA), damping=0.0)]
    if kind == 'plain':
        offsets = [sc.Offsets(exclude=sc.Poly(0))]
    elif kind == 'poly_slope':
        offsets = [sc.Offsets(polynomial=sc.Poly(2, window=range(0, N_BAND))),
                   sc.Offsets('slope', on='band', times=sc.Function(tilt, of='det_x', centre=CENTRE, half=HALF))]
    else:
        offsets = [sc.Offsets(per='all', exclude=sc.Poly(0))]
    return sc.Model(sky=sky, offsets=offsets + [sc.Offsets('scalar', on='detector')])


NO_CLIP = sc.Clip(1e6)          # noise-free data: nothing is an outlier


def solve(root, kind, init_iterations, n=5):
    field_dir = os.path.join(root, kind, 'out', 'toy')
    if not os.path.exists(field_dir):
        os.makedirs(field_dir)
        write_field(field_dir, kind, np.random.default_rng(7))
    name = f'{kind}_it{init_iterations}'
    recipe = sc.Recipe(model_of(kind), fit=sc.Fit(init_iterations, clip=None, line_fisher_threshold=0.0),
                       coadd=None, numerics=sc.Numerics(2, batch=4), name=name)
    passes = sc.Passes(n, order='offset_first', sky_clip=NO_CLIP,
                       offset=sc.Refit(clip=NO_CLIP, min_pixels=0))
    field = sc.Field(field_dir, ToyInstrument(), PIX)
    field.calibrate(recipe, passes=passes,
                    compute=sc.Compute(os.path.join(root, kind, f'scratch_{name}'), workers=2, stage=None))
    cal_dir = os.path.join(field_dir, 'calibration')
    stem = f'cal_{TAG}_All_{name}'
    products = [os.path.join(cal_dir, f'{stem}.h5')] + [
        os.path.join(cal_dir, f'{stem}_pass{i}{"off" if i % 2 == 0 else "sky"}.h5') for i in range(2, n + 1)]
    return products


def sky_of(path):
    with h5py.File(path, 'r') as f:
        return f['sky']['continuum'][()].astype(np.float64), f['sky']['line'][()].astype(np.float64), \
            f['sky_fisher']['line'][()]


def offsets_on_detector(path):
    """``{frame name: detector offset}`` of a cal or an OFFSET product, rendered here (the toy's frames
    map detector pixels one to one): each map's stored values on its chunk map (the bands' map for 16
    values per frame, the band x column map for 32, the one-chunk map of the scalar for 1), times
    ``tilt(det_x)`` for a map of known functions."""
    det_x = np.mgrid[0:DET, 0:DET][1].astype(np.float64)
    maps = {N_BAND: band_ids(), N_BAND * N_COL: chunk_ids(), 1: np.zeros((DET, DET), dtype=np.int64)}
    out = {}
    with h5py.File(path, 'r') as f:
        names = [os.path.basename(r.decode() if isinstance(r, bytes) else r) for r in f['reproj_list'][()]]
        rows = [(f['offsets'][f'map_{m}'][()].astype(np.float64), 'n_basis' in f['offsets'][f'map_{m}'].attrs)
                for m in range(int(f.attrs['num_maps']))]
    for i, name in enumerate(names):
        total = np.zeros((DET, DET))
        for values, with_function in rows:
            term = values[i][maps[values.shape[1]]]
            total += term * tilt(det_x, CENTRE, HALF) if with_function else term
        out[name] = total
    return out


def rounding_floor(field_dir):
    """The objective's rounding floor: the weighted norm of the float32 rounding of the data (the
    products are float32), below which a change of the objective is rounding."""
    total = 0.0
    for name in sorted(os.listdir(os.path.join(field_dir, 'reprojected'))):
        with h5py.File(os.path.join(field_dir, 'reprojected', name), 'r') as g:
            data = g['sub_data'][()].astype(np.float64)
        total += float(np.sum((np.finfo(np.float32).eps * data) ** 2 / np.abs(data)))
    return np.sqrt(total)


def non_increasing(obj, field_dir):
    """Whether the objectives never increase by more than four times the rounding floor."""
    tol = 4.0 * rounding_floor(field_dir)
    return all(b <= a + tol for a, b in zip(obj, obj[1:]))


def objective(field_dir, sky_path, offsets_path):
    """The joint objective: the shot-noise-weighted residual norm of every frame against the sky of
    ``sky_path`` and the offsets of ``offsets_path``."""
    continuum, line, _ = sky_of(sky_path)
    continuum, line = np.nan_to_num(continuum), np.nan_to_num(line)
    gauss = np.exp(-0.5 * ((wavelength().astype(np.float64) - LINE_CENTER) / LINE_SIGMA) ** 2)
    offsets = offsets_on_detector(offsets_path)
    total = 0.0
    for name in sorted(os.listdir(os.path.join(field_dir, 'reprojected'))):
        with h5py.File(os.path.join(field_dir, 'reprojected', name), 'r') as g:
            data = g['sub_data'][()].astype(np.float64)
            y0, y1, x0, x1 = (int(v) for v in g.attrs['ref_coords'])
        model = continuum[y0:y1, x0:x1] + line[y0:y1, x0:x1] * gauss + offsets[name]
        total += float(np.sum((data - model) ** 2 / np.abs(data)))
    return np.sqrt(total)


def objectives(products):
    """The joint objective after the first solve and after every pass (the sky and offsets it holds)."""
    field_dir = os.path.dirname(os.path.dirname(products[0]))
    sky, off = products[0], products[0]
    out = [objective(field_dir, sky, off)]
    for p in products[1:]:
        if p.endswith('off.h5'):
            off = p
        else:
            sky = p
        out.append(objective(field_dir, sky, off))
    return out


KINDS = ('plain', 'poly_slope')


@pytest.fixture(scope='module')
def root(tmp_path_factory):
    sc.set_progress(False)
    return str(tmp_path_factory.mktemp('npass_any'))


@pytest.mark.parametrize('kind', KINDS)
def test_passes_never_increase_the_objective(root, kind):
    """From a long first solve, no pass raises the joint objective (beyond the float32 rounding of
    the data), whatever the offset terms: the subtraction and the fixed-sky solve are the joint
    solve's own rows. The OFFSET product holds every term of the model (the scalar among them), for
    every frame, and no sky."""
    products = solve(root, kind, init_iterations=400, n=5)
    field_dir = os.path.dirname(os.path.dirname(products[0]))
    obj = objectives(products)
    assert non_increasing(obj, field_dir), (obj, rounding_floor(field_dir))
    with h5py.File(products[1], 'r') as f:
        assert f.attrs['num_maps'] == len(model_of(kind).offsets)
        assert f['reproj_list'].shape[0] == N_FRAMES and 'frame_scalar' not in f and 'sky' not in f
    with h5py.File(products[1].replace('_pass2off.h5', '_pass2off_all.h5'), 'r') as f:
        solved = json.loads(f.attrs['solve'])
    assert solved['method'] == 'direct' and solved['blocks'] == N_FRAMES and solved['left_out'] == 0, solved


@pytest.mark.parametrize('kind', KINDS)
def test_passes_improve_an_unconverged_start(root, kind):
    """From a deliberately unconverged first solve, every pass lowers the joint objective and the
    passes take it well below the first solve's."""
    products = solve(root, kind, init_iterations=3, n=5)
    field_dir = os.path.dirname(os.path.dirname(products[0]))
    obj = objectives(products)
    assert non_increasing(obj, field_dir), obj
    assert obj[-1] < 0.7 * obj[0], obj


def test_shared_terms_are_refused(root):
    """An offset term shared across frames ties every frame of an OFFSET pass together; the plan
    refuses it before anything runs."""
    from selfcal.config.base import ConfigError
    with pytest.raises(ConfigError, match="shared across frames"):
        solve(root, 'shared', init_iterations=3, n=3)
