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
poly_slope  the first two Chebyshev shapes of the band index per column (a ``basis`` of two known
            functions of a detector map), a free x-slope per band (``times`` a function of the
            detector column, as SPHEREx's ``frame_slope``) and the scalar
=========== =======================================================================================

For each, the joint objective (the weighted residual of the data against the sky and offsets the
passes hold, computed here from the products independently of the library) never increases from pass
to pass (each pass is an exact block minimisation), and from a deliberately unconverged first solve
the passes bring the sky much closer to the truth. A third model, a detector-fixed pattern shared by
every frame (``per="all"``), would tie the frames of an OFFSET pass together; the plan refuses it. A term
that excludes a sky signature is tied across frames by that constraint: the plan refuses it for a schedule
with an OFFSET pass and takes it for the joint solve followed by a SKY pass alone. A SKY pass weighs every
observation with the model's observation weight, as the joint solve does, and a prior on a sky term, which a
SKY pass cannot hold, is refused.
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
from selfcal.run.schedule import schedule  # noqa: E402

DET, N_BAND, N_COL, REF, PIX = 64, 16, 2, 96, 20.0
N_FRAMES = 20
LINE_CENTER, LINE_SIGMA = 1.25, 0.08
TAG = 'ToyAny'

def tilt(x, centre, half):
    """A linear basis of a coordinate, -1 to +1 over centre +- half (defined here, above the API call)."""
    return (np.asarray(x, dtype=np.float64) - centre) / half


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


def band_columns(geometry):
    """The primary chunk map: a chunk per band and column (:func:`chunk_ids`)."""
    return sc.Chunks(chunk_ids(), sc.ChunkAxes.row_major(('band', 'col'), (N_BAND, N_COL), ('y', 'x')),
                     adjacency_axes=('col',), spectral_axis='band')


def bands(geometry):
    """A second chunk map: a chunk per band (:func:`band_ids`)."""
    return sc.Chunks(band_ids(), sc.ChunkAxes.row_major(('band',), (N_BAND,), ('y',)), adjacency_axes=('band',),
                     spectral_axis='band')


def columns(geometry):
    """A third chunk map: a chunk per column, the map of the polynomial offset term."""
    return sc.Chunks(chunk_ids() % N_COL, sc.ChunkAxes.row_major(('col',), (N_COL,), ('x',)))


def band_position(geometry):
    """The band index of every detector pixel: the coordinate the polynomial offsets are in (a detector map)."""
    return band_ids().astype(np.float32)


def band_shapes(band, degree, lo, hi):
    """The Chebyshev shapes ``T_1 .. T_degree`` of the band over ``lo .. hi``, each minus its mean over the
    bands (no constant: the frame's level is the scalar's)."""
    from numpy.polynomial import chebyshev as C
    x = 2.0 * (np.asarray(band, dtype=np.float64) - lo) / (hi - lo) - 1.0
    centre = C.chebvander(2.0 * (np.arange(lo, hi + 1) - lo) / (hi - lo) - 1.0, int(degree))[:, 1:].mean(axis=0)
    return C.chebvander(x, int(degree))[..., 1:] - centre


def geometry(oversample):
    return sc.Geometry((DET, DET), oversample,
                       aux={'wave': wavelength(), 'width': np.full((DET, DET), 0.02, np.float32)},
                       wavelength='wave', width='width')


@dataclass(frozen=True, kw_only=True)
class ToyInstrument(sc.Instrument):
    tag: str = TAG
    capabilities = ('wavelength', 'spectral_axis')

    def geometry(self, oversample=1):
        return geometry(oversample)

    def default_chunk_maps(self):
        return {'lvf': sc.ChunkMap(band_columns), 'band': sc.ChunkMap(bands), 'column': sc.ChunkMap(columns)}


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


def variance_by_column(det_x):
    """An observation variance for the tests of the model's weight: rising with the detector column, and undefined
    (NaN: the observation carries no weight) on column 5."""
    v = (1.0 + np.asarray(det_x, dtype=np.float64) / 32.0) ** 2
    return np.where(np.rint(det_x) == 5, np.nan, v)


def sky_prior(line):
    """A prior on the line term; never evaluated (the plan refuses it with a SKY pass before anything runs)."""
    raise AssertionError("not evaluated")


def model_of(kind, slope_exclude=None, weighted=False, prior=False):
    """The model of ``kind``; ``slope_exclude``: what the x-slopes of ``poly_slope`` may not carry; ``weighted``: the
    observation weight ``1 / variance_by_column``; ``prior``: a prior on the line term."""
    sky = [sc.Sky(damping=0.0), sc.Sky('line', times=sc.gaussian(LINE_CENTER, sigma=LINE_SIGMA), damping=0.0)]
    if kind == 'plain':
        offsets = [sc.Offsets(exclude=sc.Poly(0))]
    elif kind == 'poly_slope':
        offsets = [sc.Offsets('poly', on='column', n=2,
                              basis=sc.Function(band_shapes, of='band_pos', degree=2, lo=0, hi=N_BAND - 1)),
                   sc.Offsets('slope', on='band', times=sc.Function(tilt, of='det_x', centre=CENTRE, half=HALF),
                              exclude=slope_exclude)]
    else:
        offsets = [sc.Offsets(per='all', exclude=sc.Poly(0))]
    return sc.Model(sky=sky, offsets=offsets + [sc.Offsets('scalar', on='detector')],
                    variables={'band_pos': sc.DetectorMap(band_position)} if kind == 'poly_slope' else {},
                    **({'weight': sc.Variance(variance_by_column, of='det_x')} if weighted else {}),
                    **({'priors': [sc.Prior(sky_prior, 'line')]} if prior else {}))


def solve(root, kind, init_iterations, n=5, order='offset_first', tied=False, floor=None, weighted=False, prior=False):
    """The products of an ``n``-pass solve of ``kind``'s field: the first solve's cal, then each pass's. ``tied``: the
    x-slopes of ``poly_slope`` exclude every sky signature of a deflation of order 2 (a constraint across frames).
    ``floor``: the fit's coverage floor (``Fit(min_observations=)``). ``weighted``, ``prior``: as :func:`model_of`."""
    field_dir = os.path.join(root, kind, 'out', 'toy')
    if not os.path.exists(field_dir):
        os.makedirs(field_dir)
        write_field(field_dir, kind, np.random.default_rng(7))
    name = (f'{kind}{"_tied" if tied else ""}{"" if floor is None else f"_floor{floor}"}'
            f'{"_weighted" if weighted else ""}{"_prior" if prior else ""}_it{init_iterations}')
    model = model_of(kind, slope_exclude=sc.SkySignature('all') if tied else None, weighted=weighted, prior=prior)
    fit = sc.Fit(init_iterations, clip=None, line_fisher_threshold=0.0, min_observations=floor,
                 **({'deflate': sc.Deflate(2)} if tied else {}))
    recipe = sc.Recipe(model, fit=fit, coadd=None, numerics=sc.Numerics(2, batch=4), name=name)
    passes = sc.Passes(n, order=order, offset=sc.Refit(min_pixels=0))      # no pass clips, as by default
    field = sc.Field(field_dir, ToyInstrument(), PIX)
    field.calibrate(recipe, jobs=sc.Job('All'), passes=passes,
                    compute=sc.Compute(os.path.join(root, kind, f'scratch_{name}'), workers=2, stage=None))
    cal_dir = os.path.join(field_dir, 'calibration')
    stem = f'cal_{TAG}_All_{name}'
    products = [os.path.join(cal_dir, f'{stem}.h5')] + [
        os.path.join(cal_dir, f'{stem}_pass{i}{"off" if t == "offset" else "sky"}.h5')
        for i, t in enumerate(schedule(n, order)[1:], start=2)]
    return products


def sky_of(path):
    with h5py.File(path, 'r') as f:
        return f['sky']['continuum'][()].astype(np.float64), f['sky']['line'][()].astype(np.float64), \
            f['sky_fisher']['line'][()]


def offsets_on_detector(path):
    """``{frame name: detector offset}`` of a cal or an OFFSET product, rendered here (the toy's frames
    map detector pixels one to one): each map's stored values on its chunk map (the bands' map for 16
    chunks, the band x column map for 32, the columns' map for 2, the one-chunk map of the scalar), and
    for a map of known functions (``n_basis`` coefficients per chunk, columns ``chunk * n_basis + k``)
    the coefficients times the functions: ``tilt(det_x)`` on the bands' map, the band shapes on the
    columns' map."""
    det_x = np.mgrid[0:DET, 0:DET][1].astype(np.float64)
    band = band_ids()
    maps = {N_BAND: band, N_BAND * N_COL: chunk_ids(), N_COL: chunk_ids() % N_COL,
            1: np.zeros((DET, DET), dtype=np.int64)}
    functions = {N_BAND: tilt(det_x, CENTRE, HALF)[..., None], N_COL: band_shapes(band, 2, 0, N_BAND - 1)}
    out = {}
    with h5py.File(path, 'r') as f:
        names = [os.path.basename(r.decode() if isinstance(r, bytes) else r) for r in f['reproj_list'][()]]
        rows = [(f['offsets'][f'map_{m}'][()].astype(np.float64),
                 int(f['offsets'][f'map_{m}'].attrs.get('n_basis', 0))) for m in range(int(f.attrs['num_maps']))]
    for i, name in enumerate(names):
        total = np.zeros((DET, DET))
        for values, n_basis in rows:
            if n_basis:
                chunks = values.shape[1] // n_basis
                total += (values[i].reshape(chunks, n_basis)[maps[chunks]] * functions[chunks]).sum(axis=-1)
            else:
                total += values[i][maps[values.shape[1]]]
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


def test_a_term_tied_across_frames_takes_sky_passes_only(root):
    """An offset term that excludes a sky signature is tied across frames by that constraint. A schedule with an
    OFFSET pass, which solves every frame on its own, is refused before anything runs. The joint solve followed by a
    SKY pass alone holds the joint solve's offsets: it runs, its product is a sky, and that sky does not raise the
    joint objective."""
    from selfcal.config.base import ConfigError
    with pytest.raises(ConfigError, match="exclude a sky signature"):
        solve(root, 'poly_slope', init_iterations=400, n=3, tied=True)
    products = solve(root, 'poly_slope', init_iterations=400, n=2, order='sky_first', tied=True)
    assert [os.path.basename(p).rsplit('_', 1)[-1] for p in products[1:]] == ['pass2sky.h5']
    field_dir = os.path.dirname(os.path.dirname(products[0]))
    obj = objectives(products)
    assert non_increasing(obj, field_dir), (obj, rounding_floor(field_dir))
    with h5py.File(products[1], 'r') as f:
        assert 'sky' in f and 'offsets' not in f


def coverage_of(path):
    with h5py.File(path, 'r') as f:
        return f['sky_coverage']['continuum'][()]


def test_a_sky_pass_applies_the_coverage_floor_to_its_own_pixels(root):
    """With ``Fit(min_observations=k)``, a SKY pass leaves out the pixels its own observations cover fewer than k
    times, as the joint solve does: they have no coverage and no value in its product, and every other pixel keeps
    the coverage it has without the floor."""
    k = 4
    plain = coverage_of(solve(root, 'plain', init_iterations=400, n=2, order='sky_first')[1])
    floored_path = solve(root, 'plain', init_iterations=400, n=2, order='sky_first', floor=k)[1]
    floored = coverage_of(floored_path)
    below = (plain > 0) & (plain < k)
    assert below.any() and (plain >= k).any()
    assert not floored[below].any() and np.array_equal(floored[~below], plain[~below])
    continuum, line, _ = sky_of(floored_path)
    assert not np.nan_to_num(continuum)[below].any() and not np.nan_to_num(line)[below].any()
    assert np.all(np.isfinite(continuum[plain >= k])) and np.any(continuum[plain >= k] != 0)


def test_the_coverage_floor_on_a_pass_sums():
    """``apply_coverage_floor`` zeroes, for the pixels the first term's count puts below the floor, every term's count,
    Fisher and right-hand side and the cross moments, and nothing else; a pixel without data is not counted."""
    from selfcal.npass.sky import apply_coverage_floor
    counts = np.array([0, 1, 3, 5, 0, 1, 3, 5], dtype=np.float64)        # two terms on four pixels
    fisher, rhs = np.arange(1.0, 9.0), np.arange(11.0, 19.0)
    cross = {(0, 1): np.array([0.5, 0.6, 0.7, 0.8])}
    assert apply_coverage_floor(counts, fisher, rhs, cross, 4, 2, 3) == 1
    assert counts.tolist() == [0, 0, 3, 5, 0, 0, 3, 5]
    assert fisher.tolist() == [1, 0, 3, 4, 5, 0, 7, 8] and rhs.tolist() == [11, 0, 13, 14, 15, 0, 17, 18]
    assert cross[(0, 1)].tolist() == [0.5, 0.0, 0.7, 0.8]


def test_a_sky_pass_weighs_every_observation_with_the_models_weight(root):
    """With ``sc.Model(weight=sc.Variance(...))`` a SKY pass weighs every observation as the joint solve does: each
    pixel's stored information on the continuum (whose coefficient is one) is the sum of 1/variance over the
    observations that see it, computed here from the frames, in the joint solve's cal and in the SKY pass's product
    alike; the observations of detector column 5, whose variance is undefined, carry no weight in either."""
    products = solve(root, 'plain', init_iterations=50, n=2, order='sky_first', weighted=True)
    field_dir = os.path.dirname(os.path.dirname(products[0]))
    expected = np.zeros((REF, REF))
    for name in sorted(os.listdir(os.path.join(field_dir, 'reprojected'))):
        with h5py.File(os.path.join(field_dir, 'reprojected', name), 'r') as g:
            det_x = g['sub_mapping'][0].astype(np.float64)
            y0, y1, x0, x1 = (int(v) for v in g.attrs['ref_coords'])
        expected[y0:y1, x0:x1] += np.nan_to_num(1.0 / variance_by_column(det_x), nan=0.0)
    assert np.any(expected > 0)
    for path in products:
        with h5py.File(path, 'r') as f:
            fisher = f['sky_fisher']['continuum'][()].astype(np.float64)
        np.testing.assert_allclose(fisher, expected, rtol=1e-5, atol=0.0, err_msg=os.path.basename(path))


def test_a_prior_on_a_sky_term_is_refused_with_a_sky_pass(root):
    """A SKY pass solves every pixel from that pixel's own sums, which have no place for a prior's rows on the sky:
    the plan refuses a model with a prior on a sky term for a schedule with a SKY pass, before anything runs."""
    from selfcal.config.base import ConfigError
    with pytest.raises(ConfigError, match="act on sky terms"):
        solve(root, 'plain', init_iterations=3, n=2, order='sky_first', prior=True)
