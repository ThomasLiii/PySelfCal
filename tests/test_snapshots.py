"""Snapshots of a solve (selfcal.core.snapshots; ``field.calibrate(recipe, snapshots=sc.Snapshots(...))``).

The solvers call back every ``k`` iterations while the solve goes on (not at the iteration it stops at)
and their iterates are unchanged; :class:`~selfcal.core.snapshots.Iterate` reads the iterate exactly as
the end of the solve converts it. Through the API: snapshots appear at k, 2k, ... (not at the last
iteration), named and marked with the cumulative iteration, also after a warm start; ``keep`` keeps the
last ones; a snapshot's datasets are those of a solve of exactly that many iterations, bit for bit (a
model with two offset terms, a spectral model with two sky blocks, LSQR and LSMR); a snapshot is a
valid start and mosaics like that cal; the final cal is the same with and without snapshots; tiled and
N-pass runs take none; the setting stays out of the fingerprints and a rerun replays it.
"""
import json
import logging
import os
import shutil
import sys
import tempfile

import h5py
import numpy as np
import pytest
import scipy.sparse as sp

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import selfcal as sc  # noqa: E402
from selfcal import _state  # noqa: E402
from selfcal.config import ConfigError  # noqa: E402
from selfcal.core import snapshots as snapshots_mod  # noqa: E402
from selfcal.core.lsmr import lsmr  # noqa: E402
from selfcal.core.lsqr_inplace import lsqr_inplace  # noqa: E402
from selfcal.core.snapshots import ActiveColumns, Iterate  # noqa: E402
from selfcal.core.solve import apply_lsqr  # noqa: E402
from selfcal.io.calfile import CalFile  # noqa: E402
from selfcal.io.parallel_h5 import (  # noqa: E402
    create_gzip_dataset_parallel,
    create_gzip_dataset_rows,
)
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

# scipy's LSMR (and so ours) overflows a float32 scalar once a float32 system has converged
pytestmark = pytest.mark.filterwarnings('ignore:overflow encountered in cast:RuntimeWarning')

NUMERICS = sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4)
MODELS = {
    'times': sc.Model(offsets=[sc.Offsets(smooth=0.1, mean_zero=True), sc.Offsets('xslope', times=xtilde)]),
    'spectral': sc.Model(sky=[sc.Sky(), sc.Sky('ramp', times=x_ramp)], offsets=[sc.Offsets(smooth=0.1, mean_zero=True)]),
}


def _system(dtype=np.float64, m=400, n=60, seed=0):
    A = sp.random(m, n, density=0.2, random_state=seed + 1, format='csr', dtype=np.float64).astype(dtype)
    b = np.random.default_rng(seed).normal(size=m).astype(dtype)
    return A, b


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


# =================================================================== the solvers' callback
@pytest.mark.parametrize('solver', ['lsqr', 'lsmr'])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_the_callback_sees_every_kth_iterate_and_changes_nothing(solver, dtype):
    A, b = _system(dtype)
    x0 = np.random.default_rng(3).normal(size=A.shape[1]).astype(dtype)
    run = lsqr_inplace if solver == 'lsqr' else lsmr
    limit = 'iter_lim' if solver == 'lsqr' else 'maxiter'
    kw = dict(atol=0.0, btol=0.0, conlim=0.0, damp=0.0)
    seen = {}
    got = run(A, b.copy(), x0=x0.copy(), callback=lambda itn, x: seen.setdefault(itn, x.copy()), callback_every=4,
              **{limit: 18}, **kw)
    plain = run(A, b.copy(), x0=x0.copy(), **{limit: 18}, **kw)
    assert _same(got[0], plain[0]) and got[1:3] == plain[1:3]
    assert sorted(seen) == [4, 8, 12, 16]                         # not 18: the solve stops there
    for itn, x in seen.items():                                   # the iterate of a solve of itn iterations
        assert _same(x, run(A, b.copy(), x0=x0.copy(), **{limit: itn}, **kw)[0])


@pytest.mark.parametrize('solver', ['lsqr', 'lsmr'])
def test_no_callback_after_a_tolerance_stop(solver):
    A, b = _system()
    run = lsqr_inplace if solver == 'lsqr' else lsmr
    seen = []
    res = run(A, b.copy(), atol=1e-6, btol=1e-6, callback=lambda itn, x: seen.append(itn), callback_every=1,
              **{'iter_lim' if solver == 'lsqr' else 'maxiter': 100})
    assert res[1] in (1, 2) and seen == list(range(1, res[2]))   # every iteration but the last


def test_active_columns_find_compact_indices(monkeypatch):
    monkeypatch.setattr(snapshots_mod, '_PREFIX_BLOCK', 7)
    mask = np.random.default_rng(1).random(100) < 0.6
    ac = ActiveColumns(mask)
    for start in range(0, 101):
        assert ac.compact_start(start) == int(np.count_nonzero(mask[:start]))


@pytest.mark.parametrize('precondition', [True, False])
@pytest.mark.parametrize('use_float32', [True, False])
@pytest.mark.parametrize('solver', ['lsqr', 'lsmr'])
def test_an_iterate_reads_as_the_solution_of_that_many_iterations(monkeypatch, solver, use_float32, precondition):
    """``apply_lsqr``'s snapshot at iteration i, read block by block (blocks across the compaction), is
    bit for bit the solution ``apply_lsqr`` returns after exactly i iterations."""
    monkeypatch.setattr(snapshots_mod, '_PREFIX_BLOCK', 5)
    A, b = _system()
    n_full = 90
    active = np.zeros(n_full, dtype=bool)
    active[np.sort(np.random.default_rng(2).choice(n_full, A.shape[1], replace=False))] = True
    x0 = np.random.default_rng(4).normal(size=n_full)
    kw = dict(ref_shape=(1, 1), atol=0.0, btol=0.0, conlim=0.0, damp=0.0, solver=solver, n_threads=1,
              use_float32=use_float32, precondition=precondition, active_mask=active, num_cols_full=n_full)
    read = {}

    def snap(it):
        assert it.num_cols == n_full
        cuts = [0, 3, 11, 12, 40, 41, 77, n_full]
        read[it.itn] = np.concatenate([it.read(a, z) for a, z in zip(cuts[:-1], cuts[1:])])
        assert it.estimates['itn'] == it.itn and it.settings['method'] == solver
    final = apply_lsqr(A.copy(), b.copy(), x0=x0.copy(), iter_lim=13, snapshot=snap, snapshot_every=3, **kw)
    assert _same(final, apply_lsqr(A.copy(), b.copy(), x0=x0.copy(), iter_lim=13, **kw))
    assert sorted(read) == [3, 6, 9, 12]
    for itn, x in read.items():
        assert _same(x, apply_lsqr(A.copy(), b.copy(), x0=x0.copy(), iter_lim=itn, **kw)), itn


def test_snapshots_need_the_csr_system():
    A, b = _system()
    with pytest.raises(ValueError, match='COO path'):
        apply_lsqr(A.tocoo(), b, (1, 1), snapshot=lambda it: None, snapshot_every=2)
    with pytest.raises(ValueError, match='at least 1 iteration'):
        apply_lsqr(A, b, (1, 1), snapshot=lambda it: None, snapshot_every=0)


def test_rows_written_band_by_band_store_the_same_chunks(tmp_path):
    data = np.random.default_rng(0).normal(size=(1003, 517)).astype(np.float32)
    with h5py.File(tmp_path / 'a.h5', 'w') as f:
        a = create_gzip_dataset_parallel(f, 'x', data)
        b = create_gzip_dataset_rows(f, 'y', data.shape, data.dtype, lambda r0, r1: data[r0:r1])
        assert a.chunks == b.chunks and a.compression == b.compression and a.compression_opts == b.compression_opts
        assert a.id.get_num_chunks() == b.id.get_num_chunks() > 4
        for i in range(a.id.get_num_chunks()):
            ia, ib = a.id.get_chunk_info(i), b.id.get_chunk_info(i)
            assert ia.chunk_offset == ib.chunk_offset
            assert a.id.read_direct_chunk(ia.chunk_offset) == b.id.read_direct_chunk(ib.chunk_offset)
        assert _same(b[()], data)
        with pytest.raises(ValueError, match='rows'):
            create_gzip_dataset_rows(f, 'z', data.shape, np.float64, lambda r0, r1: data[r0:r1])


def test_the_settings():
    assert sc.Snapshots(10) == sc.Snapshots(every=10) and sc.Snapshots(10).keep is None
    with pytest.raises(ConfigError, match='at least 1 iteration'):
        sc.Snapshots(0)
    with pytest.raises(ConfigError, match='keep them all'):
        sc.Snapshots(5, keep=0)
    from selfcal.run.schedule import as_snapshots
    assert as_snapshots(7) == sc.Snapshots(7) and as_snapshots(None) is None
    with pytest.raises(ConfigError, match='a sc.Snapshots'):
        as_snapshots('7')


# =================================================================== through the API
def _recipe(model, iterations=20, name='s', coadd=None, **fit):
    return sc.Recipe(model, fit=sc.Fit(iterations, tolerance=0, **fit), coadd=coadd, numerics=NUMERICS, name=name)


@pytest.fixture(scope='module')
def field():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_snap_')
    write_exposures(os.path.join(tmp, 'exposures'), 12, np.random.default_rng(5))
    f = sc.Field(os.path.join(tmp, 'out', 'toy'), sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy'),
                 REF_ARCSEC, compute=sc.Compute(os.path.join(tmp, 'cache'), workers=2, io_limit=4))
    f.reproject(os.path.join(tmp, 'exposures', 'toy_exp_*_D0.fits'), method='interp', padding=8)
    yield f
    shutil.rmtree(tmp, ignore_errors=True)


def _snapdir(cal):
    return os.path.join(os.path.dirname(cal), 'snapshots')


def _snaps(cal):
    stem = os.path.basename(cal)[:-len('.h5')]
    d = _snapdir(cal)
    return sorted(p for p in os.listdir(d) if p.startswith(stem + '_it')) if os.path.isdir(d) else []


def _items(path):
    """Every dataset (dtype, shape, bytes, chunks, filters, attributes) and group of a cal outside its
    ``solve`` group, and the root attributes."""
    out = {}
    with h5py.File(path, 'r') as f:
        out['/'] = sorted((k, repr(f.attrs[k])) for k in f.attrs)

        def visit(name, obj):
            if name == 'solve' or name.startswith('solve/'):
                return
            attrs = sorted((k, repr(obj.attrs[k])) for k in obj.attrs)
            if isinstance(obj, h5py.Dataset):
                a = obj[()]
                out[name] = (a.dtype.str, a.shape, a.tobytes(), obj.chunks, obj.compression, attrs)
            else:
                out[name] = attrs
        f.visititems(visit)
    return out


def _solve(path):
    with CalFile(path) as cal:
        return cal.solve


@pytest.mark.parametrize('method', ['lsqr', 'lsmr'])
@pytest.mark.parametrize('model', sorted(MODELS))
def test_a_snapshot_is_the_cal_of_that_many_iterations(field, model, method):
    """Snapshots at 4, 8, 12 of a 14-iteration solve (none at 14); each holds exactly the datasets a solve
    of that many iterations writes."""
    recipe = _recipe(MODELS[model], 14, name=f'sn_{model}_{method}', method=method)
    res = field.calibrate(recipe, snapshots=sc.Snapshots(4))
    cal = res.cal_paths[0]
    stem = os.path.basename(cal)[:-len('.h5')]
    assert _snaps(cal) == [f'{stem}_it0004.h5', f'{stem}_it0008.h5', f'{stem}_it0012.h5']
    assert not [p for p in os.listdir(_snapdir(cal)) if p.startswith('.')]          # the template is gone
    for it in (4, 12):
        snap = os.path.join(_snapdir(cal), f'{stem}_it{it:04d}.h5')
        exact = field.calibrate(recipe.replace(fit=recipe.fit.replace(iterations=it),
                                               name=f'sn_{model}_{method}_{it}')).cal_paths[0]
        a, b = _items(snap), _items(exact)
        assert sorted(a) == sorted(b)
        for name in a:
            assert a[name] == b[name], name
        assert any(n.startswith('sky/') for n in a) and any(n.startswith('offsets/') for n in a)
        s, e = _solve(snap), _solve(exact)
        assert s['snapshot'] is True and 'snapshot' not in e
        assert s['iteration'] == s['iterations'] == s['iterations_total'] == it == e['iterations']
        assert (s['system'], s['system_identity']) == (e['system'], e['system_identity'])
        assert s['iteration_limit'] == 14 and s['method'] == method and 'istop' not in s
        h = np.load(e['history_file'])                 # the solver's estimates at that iteration
        assert s['r1norm'] == h['r1norm'][it] and s['xnorm'] == h['xnorm'][it]
        with CalFile(snap) as c:
            assert 'snapshot of a' in c.describe()


def test_the_coverage_is_read_where_the_setup_parked_it(field, monkeypatch, tmp_path, caplog):
    """With the pixel state parked on scratch disk by the setup (here: any size is parked), the template
    reads it memory-mapped and leaves it parked; the snapshot still equals the cal of that many
    iterations (coverage, Fisher information, separability of a two-block sky)."""
    monkeypatch.setenv('SELFCAL_SPILL_MIN_GB', '0')
    monkeypatch.setenv('SELFCAL_SPILL_DIR', str(tmp_path))
    recipe = _recipe(MODELS['spectral'], 7, name='parked')
    with caplog.at_level(logging.INFO):
        cal = field.calibrate(recipe, snapshots=sc.Snapshots(5)).cal_paths[0]
    assert any('Spilling pixel state' in r.getMessage() and 'CSR build' in r.getMessage() for r in caplog.records)
    snap = os.path.join(_snapdir(cal), os.path.basename(cal)[:-len('.h5')] + '_it0005.h5')
    exact = field.calibrate(recipe.replace(fit=recipe.fit.replace(iterations=5), name='parked_5')).cal_paths[0]
    a, b = _items(snap), _items(exact)
    assert sorted(a) == sorted(b) and all(a[k] == b[k] for k in a)
    assert any(k.startswith('sky_separability/') for k in a)
    assert os.listdir(tmp_path) == []                    # every parked state was restored and removed


def test_a_warm_start_from_a_snapshot_and_cumulative_names(field):
    model = MODELS['times']
    first = field.calibrate(_recipe(model, 12, name='cu_first'), snapshots=sc.Snapshots(5))
    cal = first.cal_paths[0]
    stem = os.path.basename(cal)[:-len('.h5')]
    snap10 = os.path.join(_snapdir(cal), f'{stem}_it0010.h5')
    # a snapshot is a valid start (the warm start's checks accept it: frames, model, grid, identity)
    more = field.calibrate(_recipe(model, 12, name='cu_more'), start=snap10, snapshots=4)
    mcal = more.cal_paths[0]
    mstem = os.path.basename(mcal)[:-len('.h5')]
    assert _snaps(mcal) == [f'{mstem}_it0014.h5', f'{mstem}_it0018.h5']     # 10 + 4, 10 + 8 (not 10 + 12)
    s = _solve(os.path.join(_snapdir(mcal), f'{mstem}_it0018.h5'))
    assert (s['iteration'], s['iterations'], s['iterations_total']) == (18, 8, 18)
    assert s['start_from'] == snap10 and s['start_identity'].startswith('sha256:')       # no sidecar: its bytes
    final = _solve(mcal)
    assert (final['iterations'], final['iterations_total'], final['start_from']) == (12, 22, snap10)
    # the continuation from the snapshot is the continuation from a cal of 10 iterations
    ten = field.calibrate(_recipe(model, 10, name='cu_ten')).cal_paths[0]
    again = field.calibrate(_recipe(model, 12, name='cu_again'), start=ten).cal_paths[0]
    a, b = _items(mcal), _items(again)
    assert all(a[k] == b[k] for k in a) and sorted(a) == sorted(b)


def test_keep_the_last_snapshots(field):
    res = field.calibrate(_recipe(MODELS['times'], 20, name='keep'), snapshots=sc.Snapshots(3, keep=2))
    cal = res.cal_paths[0]
    stem = os.path.basename(cal)[:-len('.h5')]
    assert _snaps(cal) == [f'{stem}_it0015.h5', f'{stem}_it0018.h5']
    (entry,) = json.load(open(res.record))['solves']
    assert entry['snapshots']['written'] == [os.path.join(_snapdir(cal), p) for p in _snaps(cal)]
    assert entry['snapshots']['removed'] == 4 and entry['snapshots']['every'] == 3 and entry['snapshots']['keep'] == 2


def test_a_snapshot_mosaics_like_the_cal_of_that_many_iterations(field):
    coadd = sc.Coadd(clip=None, std=False)
    model = MODELS['times']
    res = field.calibrate(_recipe(model, 10, name='mos_src'), snapshots=sc.Snapshots(6))
    cal = res.cal_paths[0]
    snap = os.path.join(_snapdir(cal), os.path.basename(cal)[:-len('.h5')] + '_it0006.h5')
    from_snap = field.mosaic(_recipe(model, 10, name='mos_snap', coadd=coadd), cal=snap).mosaic_paths[0]
    six = field.calibrate(_recipe(model, 6, name='mos_six', coadd=coadd)).mosaic_paths[0]
    from astropy.io import fits
    with fits.open(from_snap) as a, fits.open(six) as b:
        assert len(a) == len(b) and len(a) > 1
        for x, y in zip(a, b):
            assert x.name == y.name and (x.data is None) == (y.data is None)
            if x.data is not None:
                assert _same(x.data, y.data), x.name


def test_the_final_cal_is_the_same_with_and_without_snapshots(field):
    recipe = _recipe(MODELS['spectral'], 9, name='same')
    with_snaps = field.calibrate(recipe, snapshots=sc.Snapshots(2)).cal_paths[0]
    saved = open(with_snaps, 'rb').read()
    side = products.read_sidecar(with_snaps)['fingerprint']
    shutil.rmtree(_snapdir(with_snaps))
    without = field.calibrate(recipe, overwrite=True).cal_paths[0]
    assert without == with_snaps and open(without, 'rb').read() == saved            # byte for byte
    assert products.read_sidecar(without)['fingerprint'] == side                    # the same inputs
    assert _snaps(without) == []
    # a current cal is reused, unsolved: it writes no snapshots
    field.calibrate(recipe, snapshots=sc.Snapshots(2))
    assert _snaps(without) == [] and open(without, 'rb').read() == saved


@pytest.mark.filterwarnings('ignore:.*recorded with code:UserWarning')
def test_not_in_the_fingerprint_and_replayed_by_a_rerun(field):
    recipe = _recipe(sc.continuum(), 6, name='fp_snap')
    (job,) = field.instrument.default_jobs()
    plan = field.plan(recipe, snapshots=sc.Snapshots(2, keep=1))
    assert plan.book.cal(job, field.frames) == field.plan(recipe).book.cal(job, field.frames)
    assert 'snapshots   every 2 iterations, the last 1 kept, in ' in str(plan)
    assert 'none: the solve runs 6' in str(field.plan(recipe, snapshots=6))
    res = field.calibrate(recipe, snapshots=sc.Snapshots(2, keep=1))
    cal = res.cal_paths[0]
    stem = os.path.basename(cal)[:-len('.h5')]
    assert _snaps(cal) == [f'{stem}_it0004.h5']
    record = json.load(open(res.record))
    assert record['settings']['snapshots'] == sc.Snapshots(2, keep=1).to_dict()
    assert record['lowered'][0]['snapshots'] == sc.Snapshots(2, keep=1).to_dict()
    assert products.read_sidecar(cal)['fingerprint'] == products.fingerprint(
        products.cal_inputs(field, recipe, job, field.frames))
    shutil.rmtree(_snapdir(cal))
    redone = sc.rerun(res.record, overwrite=True)
    assert redone.cal_paths == [cal] and _snaps(cal) == [f'{stem}_it0004.h5']


def test_tiles_passes_and_mosaics_take_no_snapshots(field):
    recipe = _recipe(sc.continuum(), 5, name='nosnap')
    for kw in (dict(tiles=sc.Tiles((1, 2), overlap=10)), dict(passes=sc.Passes(3))):
        with pytest.raises(ConfigError, match='a tiled or an N-pass calibration takes none'):
            field.plan(recipe, snapshots=sc.Snapshots(2), **kw)
        with pytest.raises(ConfigError, match='a tiled or an N-pass calibration takes none'):
            lower(field, recipe, snapshots=sc.Snapshots(2), **kw)
    from selfcal.run.plan import make_plan
    with pytest.raises(ConfigError, match="only a calibration's solve writes snapshots"):
        make_plan(field, 'mosaic', recipe.replace(coadd=sc.Coadd()), snapshots=sc.Snapshots(2))


def test_a_snapshot_that_cannot_be_written_is_skipped(field, monkeypatch, caplog):
    real = pipeline_wrapper.Calibrator.write_cal_solution

    def flaky(self, f, iterate):
        if iterate.itn == 4:
            raise OSError(28, 'No space left on device')
        return real(self, f, iterate)
    monkeypatch.setattr(pipeline_wrapper.Calibrator, 'write_cal_solution', flaky)
    with caplog.at_level(logging.ERROR, logger='selfcal.core.snapshots'):
        res = field.calibrate(_recipe(sc.continuum(), 7, name='flaky'), snapshots=sc.Snapshots(2))
    cal = res.cal_paths[0]
    stem = os.path.basename(cal)[:-len('.h5')]
    assert _snaps(cal) == [f'{stem}_it0002.h5', f'{stem}_it0006.h5']
    assert any('snapshot at iteration 4 not written' in r.getMessage() for r in caplog.records)
    assert not [p for p in os.listdir(_snapdir(cal)) if '.part-' in p or p.startswith('.')]
    (entry,) = json.load(open(res.record))['solves']
    assert entry['snapshots']['failed'][0][0] == 4


def test_an_iterate_without_compaction_or_scaling():
    x = np.arange(6, dtype=np.float32)
    it = Iterate(3, x)
    got = it.read(1, 4)
    assert _same(got, x[1:4]) and got.base is None                   # a copy, not the solver's buffer
    scale = np.full(6, 2.0, dtype=np.float32)
    assert _same(Iterate(3, x, scale=scale).read(0, 6), x * scale)


def test_names_and_iterations_of_a_writer(tmp_path):
    """The names count the cumulative iteration; a start that records no count names them by this
    solve's iterations and records iteration -1."""
    from types import SimpleNamespace

    from selfcal.core.snapshots import SnapshotWriter, snapshot_path
    cal = str(tmp_path / 'calibration' / 'cal_X_All_r.h5')
    assert snapshot_path(cal, 7) == str(tmp_path / 'calibration' / 'snapshots' / 'cal_X_All_r_it0007.h5')
    fresh = SnapshotWriter(cal, 5)
    assert (fresh.iteration(10), os.path.basename(fresh.path(10))) == (10, 'cal_X_All_r_it0010.h5')
    known = SnapshotWriter(cal, 5, start=SimpleNamespace(iterations_total=422, path='/a.h5', identity='sha256:x'))
    assert (known.iteration(100), os.path.basename(known.path(100))) == (522, 'cal_X_All_r_it0522.h5')
    unknown = SnapshotWriter(cal, 5, start=SimpleNamespace(iterations_total=None, path='/a.h5', identity=None))
    assert (unknown.iteration(10), os.path.basename(unknown.path(10))) == (-1, 'cal_X_All_r_it0010.h5')
    attrs = unknown.solve_attrs(Iterate(10, np.zeros(3), settings={'method': 'lsqr'}, estimates={'r1norm': 2.0}))
    assert (attrs['iteration'], attrs['iterations'], attrs['iterations_total']) == (-1, 10, -1)
    assert attrs['start_from'] == '/a.h5' and 'start_identity' not in attrs and attrs['r1norm'] == 2.0
    with pytest.raises(ValueError, match='at least 1'):
        SnapshotWriter(cal, 0)
    with pytest.raises(ValueError, match='bind'):
        fresh(Iterate(5, np.zeros(3)))
