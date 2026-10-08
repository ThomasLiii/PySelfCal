"""Continuing a solve from a cal file (selfcal.core.warm_start; ``field.calibrate(recipe, start=...)``).

The starting vector read back from a cal is the exact inverse of how the cal is written: a solve whose
solution IS its starting vector writes a cal whose every dataset equals the source's (a model with two
offset terms, constants and a ``times=`` term; one with a shared term and a two-function basis; a
spectral model with two sky blocks). A warm start from the least-squares solution stays there. A cal
of another system is refused: other frames, another frame order, another model, another grid, other
column counts (and, without a recorded identity, by its contents). The cumulative iteration count
accumulates over continuations, the source is recorded (cal, action record, a rerun replays it), the
start enters the cal's fingerprint only when given, and tiled and N-pass runs take no start.
"""
import json
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np
import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import selfcal as sc  # noqa: E402
from selfcal import _state  # noqa: E402
from selfcal.config import ConfigError  # noqa: E402
from selfcal.core.solve_record import SolveHistory, SolveRecord  # noqa: E402
from selfcal.core.warm_start import WarmStart  # noqa: E402
from selfcal.io.calfile import CalFile  # noqa: E402
from selfcal.pipeline import pipeline_wrapper  # noqa: E402
from selfcal.run import products  # noqa: E402
from selfcal.run.lower import lower  # noqa: E402
from tests.synthetic_exposures import (  # noqa: E402
    DET,
    N_CHUNK_SIDE,
    REF_ARCSEC,
    write_exposures,
    x_ramp,
    xtilde,
)

NUMERICS = sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4)


def plane(det_x, det_y):
    """Two known functions of the detector coordinates (a per-frame gradient)."""
    return [(np.asarray(det_x, dtype=np.float64) - DET / 2) / DET, (np.asarray(det_y, dtype=np.float64) - DET / 2) / DET]


#: The models of the round trip: two offset terms (constants and a ``times=`` term); a shared term
#: (``per="all"``) and a two-function basis; a spectral model with two sky blocks.
MODELS = {
    'times': sc.Model(offsets=[sc.Offsets(smooth=0.1, mean_zero=True), sc.Offsets('xslope', times=xtilde)]),
    'shared': sc.Model(offsets=[sc.Offsets(smooth=0.1, mean_zero=True),
                                sc.Offsets('pattern', per='all', mean_zero=True),
                                sc.Offsets('gradient', on='detector', basis=plane, n=2)]),
    'spectral': sc.Model(sky=[sc.Sky(), sc.Sky('ramp', times=x_ramp)], offsets=[sc.Offsets(smooth=0.1, mean_zero=True)]),
}


def _recipe(model, iterations=20, name='w', **fit):
    return sc.Recipe(model, fit=sc.Fit(iterations, tolerance=0, **fit), coadd=None, numerics=NUMERICS, name=name)


def _field(tmp, name='toy', padding=8):
    field = sc.Field(os.path.join(tmp, 'out', name), sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy'),
                     REF_ARCSEC, compute=sc.Compute(os.path.join(tmp, 'cache'), workers=2, io_limit=4))
    field.reproject(os.path.join(tmp, 'exposures', 'toy_exp_*_D0.fits'), method='interp', padding=padding)
    return field


@pytest.fixture(scope='module')
def toy():
    """A field of 12 frames, and a second one of the same frames on another (wider) reference grid."""
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_warm_')
    write_exposures(os.path.join(tmp, 'exposures'), 12, np.random.default_rng(5))
    yield _field(tmp), _field(tmp, 'wide', padding=12)
    shutil.rmtree(tmp, ignore_errors=True)


def _no_solve(monkeypatch):
    """The solver replaced by: the solution IS the starting vector (in the solver's precision, as
    ``apply_lsqr`` returns it), with an empty record."""
    def apply_lsqr(self, x0=None, use_float32=False, **kw):
        self.A = self.b = self.active_mask = self.num_cols_full = None
        self.x = np.asarray(x0).astype(np.float32 if use_float32 else np.float64)
        self.solve_record = SolveRecord.from_result(
            'lsqr', (self.x, 7, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), history=SolveHistory(), true_residual=0.0,
            bnorm=0.0, atol=0.0, btol=0.0, conlim=0.0, damp=0.0, iteration_limit=1, shape=(0, 0), wall_s=0.0)
    monkeypatch.setattr(pipeline_wrapper.Calibrator, 'apply_lsqr', apply_lsqr)


def _items(path):
    """Every dataset of a cal (dtype, shape, bytes) and every attribute outside its ``solve`` group."""
    out = {}
    with h5py.File(path, 'r') as f:
        out['/'] = sorted((k, repr(f.attrs[k])) for k in f.attrs)

        def visit(name, obj):
            if name == 'solve' or name.startswith('solve/'):
                return
            attrs = sorted((k, repr(obj.attrs[k])) for k in obj.attrs)
            if isinstance(obj, h5py.Dataset):
                a = obj[()]
                out[name] = (a.dtype.str, a.shape, a.tobytes(), attrs)
            else:
                out[name] = attrs
        f.visititems(visit)
    return out


def _solve(path):
    with CalFile(path) as cal:
        return cal.solve


# =================================================================== the round trip
@pytest.mark.parametrize('float32', [True, False])
@pytest.mark.parametrize('model', sorted(MODELS))
def test_a_cal_reads_back_as_its_solution(toy, monkeypatch, model, float32):
    """cal -> x0 -> cal reproduces every dataset exactly: the solve's solution is its starting vector."""
    field, _ = toy
    tag = f'rt_{model}_{int(float32)}'
    source = field.calibrate(_recipe(MODELS[model], 15, name=tag, float32=float32)).cal_paths[0]   # a real solve
    _no_solve(monkeypatch)                                                  # then the solution is the start
    again = field.calibrate(_recipe(MODELS[model], 15, name=tag + '_again', float32=float32), start=source)
    a, b = _items(source), _items(again.cal_paths[0])
    assert sorted(a) == sorted(b)
    for name in a:
        assert a[name] == b[name], name
    assert any(name.startswith('offsets/') for name in a) and any(name.startswith('sky/') for name in a)
    with CalFile(source) as cal:
        assert len(cal.sky_names) == (2 if model == 'spectral' else 1)
        assert cal.num_maps == {'times': 2, 'shared': 3, 'spectral': 1}[model]


# =================================================================== a start at the solution stays there
def test_a_warm_start_from_the_least_squares_solution_stays_there(toy):
    field, _ = toy
    exact = field.calibrate(_recipe(MODELS['times'], 400, name='exact'))
    first = _solve(exact.cal_paths[0])
    # the least-squares solution: |A^T r| / (|A| |r|) far below float32 precision
    assert np.load(first['history_file'])['test2'][-1] < 1e-8
    more = field.calibrate(_recipe(MODELS['times'], 20, name='exact_more'), start=exact)
    then = _solve(more.cal_paths[0])
    h = np.load(then['history_file'])
    eps = np.finfo(np.float32).eps
    assert abs(h['r1norm'][0] - first['true_residual']) <= 10 * eps * first['true_residual']   # starts there
    assert abs(then['true_residual'] - first['true_residual']) <= 10 * eps * first['true_residual']   # stays
    with CalFile(exact.cal_paths[0]) as a, CalFile(more.cal_paths[0]) as b:
        sky_a, sky_b = a.sky(0), b.sky(0)
        assert np.nanmax(np.abs(sky_b - sky_a)) <= 1e-5 * np.nanmax(np.abs(sky_a))


def test_a_variant_starts_from_an_earlier_solution(toy, caplog):
    """Another fit of the same system (the data-quality mask off, then on again): the columns with data
    now that the source left unsolved start at 0, the source's columns without data now are dropped."""
    import logging
    field, _ = toy
    model = MODELS['times']
    masked = field.calibrate(_recipe(model, 10, name='masked'))
    with caplog.at_level(logging.INFO, logger='selfcal.core.warm_start'):
        unmasked = field.calibrate(_recipe(model, 10, name='unmasked', use_mask=False), start=masked)
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith('warm start from')]
    started_zero = int(line.split(' of them zero')[0].rsplit(' ', 1)[1])
    dropped = int(line.split(' source values of columns inactive now')[0].rsplit(' ', 1)[1])
    assert started_zero > 0 and dropped == 0, line           # more pixels have data without the mask
    caplog.clear()
    with caplog.at_level(logging.INFO, logger='selfcal.core.warm_start'):
        field.calibrate(_recipe(model, 10, name='masked_again'), start=unmasked)
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith('warm start from')]
    assert int(line.split(' source values of columns inactive now')[0].rsplit(' ', 1)[1]) > 0, line


# =================================================================== another system is refused
def test_a_cal_of_another_system_is_refused(toy):
    field, wide = toy
    model = MODELS['times']
    source = field.calibrate(_recipe(model, 10, name='src')).cal_paths[0]
    frames = field.frames
    # other frames, another frame order: before the solve is set up (the plan)
    with pytest.raises(ConfigError, match='solved on 12 frames, this solve has 8'):
        field.calibrate(_recipe(model, 5, name='r_frames'), frames=8, start=source)
    with pytest.raises(ConfigError, match='the same 12 frames in another order'):
        field.calibrate(_recipe(model, 5, name='r_order'), frames=frames[::-1], start=source)
    with pytest.raises(ConfigError, match=r"sky terms: \['continuum'\] \(the source\) vs \['continuum', 'ramp'\]"):
        field.calibrate(_recipe(MODELS['spectral'].replace(offsets=model.offsets), 5, name='r_sky'), start=source)
    # another model (the same number of terms, the slope term on another chunk map), another grid,
    # other column counts: when the solve is set up
    other_map = model.replace(offsets=(model.offsets[0], sc.Offsets('xslope', on='detector', times=xtilde)))
    with pytest.raises(ConfigError, match='offset map 1: another chunk map'):
        field.calibrate(_recipe(other_map, 5, name='r_model'), start=source)
    with pytest.raises(ConfigError, match='the grid: sky'):
        wide.calibrate(_recipe(model, 5, name='r_grid'), start=source)
    two = model.replace(offsets=(model.offsets[0], sc.Offsets('xslope', basis=plane, n=2)))
    with pytest.raises(ConfigError, match=r'offset map 1: \(12, 16\) values in the source, \(12, 32\) in this solve'):
        field.calibrate(_recipe(two, 5, name='r_columns'), start=source)
    # none of them left a cal behind
    assert not [p for p in os.listdir(os.path.join(field.path, 'calibration')) if p.startswith('cal_') and '_r_' in p]

    # a cal without a recorded system identity (solved before 2026-10-08): checked by its contents
    legacy = os.path.join(os.path.dirname(source), 'legacy', os.path.basename(source))
    os.makedirs(os.path.dirname(legacy))
    shutil.copy(source, legacy)
    with h5py.File(legacy, 'a') as f:
        for k in ('system', 'system_identity'):
            del f['solve'].attrs[k]
    assert WarmStart(legacy).recorded is None and WarmStart(source).recorded is not None
    with pytest.raises(ConfigError, match='the grid: sky'):
        wide.calibrate(_recipe(model, 5, name='r_grid_legacy'), start=legacy)
    ok = field.calibrate(_recipe(model, 5, name='from_legacy'), start=legacy)
    assert _solve(ok.cal_paths[0])['start_from'] == legacy
    # its contents cannot show how frames share offsets: the starting vector checks it
    shared = model.replace(offsets=(model.offsets[0], sc.Offsets('xslope', per='all', times=xtilde)))
    with pytest.raises(ConfigError, match='offset map 1 of the source does not share its offsets'):
        field.calibrate(_recipe(shared, 5, name='r_grouping_legacy'), start=legacy)
    with pytest.raises(ConfigError, match=r'offsets\[1\]\.groups: 12 \(the source\) vs 1 \(this solve\)'):
        field.calibrate(_recipe(shared, 5, name='r_grouping'), start=source)

    # the cal this calibration writes is no start
    with pytest.raises(ConfigError, match='a cal this calibration writes'):
        field.calibrate(_recipe(model, 10, name='src'), start=source, overwrite=True)


def test_a_model_whose_offsets_cannot_be_read_back_is_refused(toy):
    field, _ = toy
    source = field.calibrate(_recipe(sc.continuum(), 5, name='poly_src')).cal_paths[0]
    poly = sc.Model(offsets=[sc.Offsets(polynomial=sc.Poly(1, along='row', each='col', window=range(0, 4)))])
    with pytest.raises(ConfigError, match='polynomial basis'):
        field.plan(_recipe(poly, 5, name='poly'), start=source)


# =================================================================== continuations, provenance, reruns
@pytest.mark.filterwarnings('ignore:.*recorded with code:UserWarning')
def test_iterations_accumulate_and_the_start_is_recorded(toy):
    field, _ = toy
    model = MODELS['times']
    first = field.calibrate(_recipe(model, 12, name='it12'))
    second = field.calibrate(_recipe(model, 8, name='it20'), start=first)
    third = field.calibrate(_recipe(model, 5, name='it25'), start=second.cal_paths[0])
    s1, s2, s3 = (_solve(r.cal_paths[0]) for r in (first, second, third))
    assert (s1['iterations'], s1['iterations_total']) == (12, 12) and 'start_from' not in s1
    assert (s2['iterations'], s2['iterations_total']) == (8, 20)
    assert (s3['iterations'], s3['iterations_total']) == (5, 25)
    for s, source in ((s2, first.cal_paths[0]), (s3, second.cal_paths[0])):
        assert s['start_from'] == source
        assert s['start_identity'] == 'fingerprint:' + products.read_sidecar(source)['fingerprint']
        assert s['system'] == s1['system'] and s['system_identity'] == s1['system_identity']
    # the action's record: the start among the settings, the provenance among the solves
    record = json.load(open(third.record))
    assert record['settings']['start'] == {third.jobs[0].name: second.cal_paths[0]}
    (entry,) = record['solves']
    assert {k: entry[k] for k in ('start_from', 'start_identity', 'iterations', 'iterations_total')} == \
        {k: s3[k] for k in ('start_from', 'start_identity', 'iterations', 'iterations_total')}
    # the plan says where each job starts
    assert f"start       {third.jobs[0].name}: {second.cal_paths[0]}" in str(
        field.plan(_recipe(model, 5, name='it25'), start=second))
    # a rerun replays the start: the same cal, byte for byte
    saved = open(third.cal_paths[0], 'rb').read()
    redone = sc.rerun(third.record, overwrite=True)
    assert redone.cal_paths == third.cal_paths and open(redone.cal_paths[0], 'rb').read() == saved
    assert _solve(redone.cal_paths[0])['iterations_total'] == 25


def test_tiles_and_passes_take_no_start(toy):
    field, _ = toy
    source = field.calibrate(_recipe(sc.continuum(), 5, name='nostart')).cal_paths[0]
    recipe = _recipe(sc.continuum(), 5, name='tiled')
    with pytest.raises(ConfigError, match='a tiled or an N-pass calibration takes no start'):
        field.plan(recipe, tiles=sc.Tiles((1, 2), overlap=10), start=source)
    with pytest.raises(ConfigError, match='a tiled or an N-pass calibration takes no start'):
        field.plan(recipe, passes=sc.Passes(3), start=source)
    with pytest.raises(ConfigError, match='a tiled or an N-pass calibration takes no start'):
        lower(field, recipe, tiles=sc.Tiles((1, 2), overlap=10), start=source)
    with pytest.raises(ConfigError, match='only a calibration starts from a cal'):
        from selfcal.run.plan import make_plan
        make_plan(field, 'mosaic', recipe.replace(coadd=sc.Coadd()), start=source)


def test_the_fingerprint_holds_the_start_only_when_given(toy, tmp_path):
    field, _ = toy
    recipe = _recipe(sc.continuum(), 5, name='fp')
    source = field.calibrate(recipe.replace(name='fp_src')).cal_paths[0]
    (job,) = field.instrument.default_jobs()
    plain = products.cal_inputs(field, recipe, job, field.frames)
    assert 'start' not in plain
    ident = products.start_identity(source)
    assert ident == {'fingerprint': products.read_sidecar(source)['fingerprint']}
    started = products.cal_inputs(field, recipe, job, field.frames, start=ident)
    assert started == {**plain, 'start': ident}
    assert products.fingerprint(started) != products.fingerprint(plain)
    # the plan's book: the start only for a calibration that has one
    assert 'start' not in field.plan(recipe).book.cal(job, field.frames)
    assert field.plan(recipe, start=source).book.cal(job, field.frames)['start'] == ident
    # a copy (no sidecar) is the same cal by its bytes; a sidecar no longer current is not trusted
    copy = tmp_path / 'copy.h5'
    shutil.copy(source, copy)
    assert products.start_identity(copy) == {'sha256': products._file_digest(str(copy))['sha256']}
    shutil.copy(products.sidecar_path(source), products.sidecar_path(copy))
    assert 'sha256' in products.start_identity(copy)
    # the made cal's sidecar holds the start
    made = field.calibrate(recipe.replace(name='fp_made'), start=source).cal_paths[0]
    assert products.read_sidecar(made)['inputs']['start'] == ident


def test_one_cal_starts_one_job(toy):
    field, _ = toy
    from selfcal.run.lower import start_paths
    source = field.calibrate(_recipe(sc.continuum(), 5, name='jobs_src'))
    (job,) = source.jobs
    assert start_paths(source, source.jobs) == {job.name: source.cal_paths[0]}
    assert start_paths({job: source.cal_paths[0]}, source.jobs) == {job.name: source.cal_paths[0]}
    with CalFile(source.cal_paths[0]) as cal:
        assert start_paths(cal, source.jobs) == {job.name: source.cal_paths[0]}
    with pytest.raises(ConfigError, match='one cal for 2 jobs'):
        start_paths(source.cal_paths[0], (job, job.__class__('Other')))
    with pytest.raises(ConfigError, match=r"no start for the job\(s\) \['Other'\]"):
        start_paths(source, (job.__class__('Other'),))
