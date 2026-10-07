"""Products and records of the Python API (selfcal.run.products, records, compare, convert, the CLI):
a product is reused only when it was made by the same inputs; records rerun byte-identically;
products are written atomically; conversion of TOML configs checks itself."""
import os
import shutil
import subprocess
import sys
import tempfile

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
from selfcal.io.atomic import atomic_path, is_partial  # noqa: E402
from selfcal.run import products  # noqa: E402
from tests.synthetic_exposures import DET, N_CHUNK_SIDE, REF_ARCSEC, write_exposures  # noqa: E402


def test_atomic_writes_leave_nothing_behind_on_failure(tmp_path):
    target = tmp_path / 'product.h5'
    with atomic_path(target) as tmp:
        assert is_partial(tmp) and tmp.endswith('.h5')
        open(tmp, 'w').write('complete')
    assert target.read_text() == 'complete'
    with pytest.raises(RuntimeError):
        with atomic_path(target) as tmp:
            open(tmp, 'w').write('half')
            raise RuntimeError('killed')
    assert target.read_text() == 'complete' and os.listdir(tmp_path) == ['product.h5']


def test_fingerprints_ignore_order_and_name_their_differences():
    a = {'fit': {'iterations': 50, 'clip': 5.0}, 'frames': {'n': 3}}
    b = {'frames': {'n': 3}, 'fit': {'clip': 5.0, 'iterations': 50}}
    assert products.fingerprint(a) == products.fingerprint(b)
    c = {'fit': {'iterations': 100, 'clip': 5.0}, 'frames': {'n': 3}}
    assert products.diff_inputs(a, c) == ['fit.iterations: 50 -> 100']
    assert products.frames_digest(['/a/exp_1.h5', '/b/exp_0.h5']) == products.frames_digest(['exp_0.h5', 'exp_1.h5'])


@pytest.fixture(scope='module')
def toy_field():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_products_')
    write_exposures(os.path.join(tmp, 'exposures'), 10, np.random.default_rng(11))
    field = sc.Field(os.path.join(tmp, 'out', 'toy'), sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy'),
                     REF_ARCSEC, compute=sc.Compute(os.path.join(tmp, 'cache'), workers=2))
    field.reproject(os.path.join(tmp, 'exposures', 'toy_exp_*_D0.fits'), method='interp', padding=8)
    yield field
    shutil.rmtree(tmp, ignore_errors=True)


def _recipe(iterations=40, name='p'):
    return sc.Recipe(sc.continuum(), fit=sc.Fit(iterations, tolerance=1e-8), coadd=sc.Coadd(clip=3.0),
                     numerics=sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4), name=name)


def test_products_are_reused_refused_adopted_and_rerun(toy_field):
    field, recipe = toy_field, _recipe()
    first = field.calibrate(recipe)
    cal, mosaic = first.cal_paths[0], first.mosaic_paths[0]
    for p in (cal, mosaic):
        assert products.read_sidecar(p)['fingerprint'] == products.fingerprint(products.read_sidecar(p)['inputs'])
    saved = open(cal, 'rb').read()

    # the same recipe: everything is reused (nothing written, no sidecar rewritten)
    again = field.calibrate(recipe)
    assert sc.rerun is not None and again.cal_paths == [cal]
    import json
    assert json.load(open(again.record))['products']['sidecars_written'] == []

    # other inputs under the same name: the plan says so; the action refuses, with the difference
    plan = field.plan(_recipe(50))
    assert {p.state for p in plan.refused} == {'different'} and 'fit.iterations: 40 -> 50' in str(plan)
    with pytest.raises(ConfigError, match=r'fit.iterations: 40 -> 50'):
        field.calibrate(_recipe(50))
    assert {p.state for p in field.plan(_recipe(50), overwrite=True).products} == {'replace'}
    # a product without a sidecar: refused until adopted
    os.remove(products.sidecar_path(cal))
    assert [p.path for p in field.plan(recipe).refused] == [cal]
    with pytest.raises(ConfigError, match='no record of how it was made'):
        field.calibrate(recipe)
    assert field.adopt(recipe) == [cal]
    assert products.read_sidecar(cal)['adopted'] is True
    assert {p.state for p in field.plan(recipe).products} == {'current'}

    # a record reruns byte-identically (overwrite: made again)
    redone = sc.rerun(first.record, overwrite=True)
    assert open(redone.cal_paths[0], 'rb').read() == saved
    assert sc.compare(cal, cal).verdict == 'identical'


def test_compare_explains_a_difference(toy_field):
    field = toy_field
    a = field.calibrate(_recipe(30, name='cmp_a')).cal_paths[0]
    b = field.calibrate(_recipe(31, name='cmp_b')).cal_paths[0]
    result = sc.compare(a, b)
    assert result.verdict == 'different'
    assert any(name.startswith('sky/') or name.startswith('offsets/') for name, _ in result.differences)
    assert 'fit.iterations: 30 -> 31' in result.inputs


def test_tuning_is_applied_for_the_action_and_recorded(toy_field):
    import json
    field = toy_field
    before = os.environ.get('SELFCAL_COADD_FLUSH_STRIPES')
    result = field.calibrate(_recipe(20, name='tuned'),
                             compute=field.compute.replace(tuning=sc.Tuning(flush_stripes=8, vector_threads=2)))
    record = json.load(open(result.record))
    assert record['tuning']['flush_stripes'] == '8' and record['tuning']['vector_threads'] == '2'
    assert os.environ.get('SELFCAL_COADD_FLUSH_STRIPES') == before
    # byte-neutral: the same products as without the knobs
    plain = field.calibrate(_recipe(20, name='untuned'))
    assert sc.compare(result.mosaic_paths[0], plain.mosaic_paths[0]).verdict in ('identical', 'equal values')


def test_by_value_runs_and_cannot_be_submitted(toy_field):
    field = toy_field
    ramp = sc.by_value(lambda det_x: (det_x - 32.0) / 64.0)
    recipe = sc.Recipe(sc.Model(sky=[sc.Sky(damping=0.001), sc.Sky('ramp', times=sc.Function(ramp, of='det_x'))],
                                offsets=[sc.Offsets(smooth=0.1, mean_zero=True)]),
                       fit=sc.Fit(20), coadd=None, numerics=sc.Numerics(2, batch=4), name='byvalue')
    result = field.calibrate(recipe)
    with result.cal() as cal:
        assert cal.sky_names == ['continuum', 'ramp']
    with pytest.raises(ConfigError, match='by value'):
        field.submit(recipe)


def _cli(*args, cwd=None):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, os.environ.get('PYTHONPATH')) if p))
    return subprocess.run([sys.executable, '-m', 'selfcal', *args], cwd=cwd, env=env, capture_output=True,
                          text=True, timeout=600)


def test_convert_writes_scripts_that_run_identically(tmp_path):
    for name in ('reproject', 'cal'):
        out = tmp_path / f'{name}.py'
        proc = _cli('convert', os.path.join(_REPO, 'examples', 'quickstart', f'{name}.toml'), '-o', str(out))
        assert proc.returncode == 0, proc.stderr[-3000:]
        text = out.read_text()
        assert 'FIELD = Field(' in text and 'if __name__ == "__main__":' in text
        compile(text, str(out), 'exec')


def test_cli_plans_and_adopts_a_run_script(toy_field, tmp_path):
    field, recipe = toy_field, _recipe(25, name='cli')
    cal = field.calibrate(recipe).cal_paths[0]
    os.remove(products.sidecar_path(cal))                 # as if made by a TOML run
    script = tmp_path / 'toy_run.py'
    script.write_text(f"from selfcal import *\nFIELD = {field!r}\nRECIPE = {recipe!r}\nRUN = {{}}\n\n"
                      "if __name__ == '__main__':\n    print(FIELD.calibrate(RECIPE, **RUN))\n")
    proc = _cli('plan', str(script))
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert '(refused: exists, unrecorded)' in proc.stdout and f'selfcal adopt {script}' in proc.stdout
    proc = _cli('adopt', str(script))
    assert proc.returncode == 0 and os.path.basename(cal) in proc.stdout, proc.stdout + proc.stderr[-3000:]
    assert products.read_sidecar(cal)['adopted'] is True
    assert '(refused' not in _cli('plan', str(script)).stdout


def test_submit_runs_detached_and_records(toy_field, tmp_path):
    import json
    field = toy_field
    job = field.submit(_recipe(15, name='detached'))
    assert job.wait(timeout=600) == 0, open(job.console).read()[-3000:]
    assert not job.running()
    request = json.load(open(job.request))
    assert request['status'] == 'submitted'
    made = field.result(_recipe(15, name='detached'))
    assert os.path.exists(made.cal_paths[0]) and products.read_sidecar(made.cal_paths[0]) is not None


def test_a_template_is_fingerprinted_by_content_not_path(tmp_path):
    from selfcal.instruments import spherex
    from selfcal.instruments.spherex.settings import _DATA, LINE_TEMPLATES
    copy = tmp_path / 'aromatic_copy.npz'
    shutil.copy(os.path.join(_DATA, LINE_TEMPLATES['aromatic']), copy)
    a = sc.spectral([spherex.line('aromatic')], polynomial=sc.Poly(2, window=range(200, 321)))
    b = sc.spectral([spherex.line('aromatic', file=str(copy))], polynomial=sc.Poly(2, window=range(200, 321)))
    from selfcal.config.base import encode
    ca, cb = (products.content_addressed(encode(products.resolved_model(m))) for m in (a, b))
    assert ca == cb
    open(copy, 'ab').write(b'\0')                       # another template: another fingerprint
    products._DIGESTS.clear()
    assert products.content_addressed(encode(products.resolved_model(b))) != ca


def test_a_plan_without_frames_says_so_and_the_action_refuses(toy_field, tmp_path):
    field = sc.Field(tmp_path / 'no_frames', toy_field.instrument, REF_ARCSEC, compute=toy_field.compute)
    os.makedirs(field.path)
    shutil.copy(os.path.join(toy_field.path, 'ref.fits'), os.path.join(field.path, 'ref.fits'))
    plan = field.plan(_recipe())
    assert any(n.startswith('no frames in') for n in plan.notes), plan.notes
    with pytest.raises(ConfigError, match='no frames in'):
        field.calibrate(_recipe())
    with pytest.raises(ConfigError, match='no frames in'):           # adopt records frames: it needs them
        field.adopt(_recipe())


def test_two_exposures_make_a_reference_grid(tmp_path):
    write_exposures(str(tmp_path / 'exposures'), 2, np.random.default_rng(3))
    field = sc.Field(tmp_path / 'two', sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy'), REF_ARCSEC,
                     compute=sc.Compute(str(tmp_path / 'cache'), workers=1))
    field.reproject(str(tmp_path / 'exposures' / 'toy_exp_*_D0.fits'), method='interp', padding=8)
    assert len(field.frames) == 2 and os.path.exists(os.path.join(field.path, 'ref.fits'))


def test_smoothing_on_a_chunk_map_without_axes_is_refused(toy_field):
    """A term smoothed on a map that declares no axes to smooth along would add no rows at all."""
    from tests.test_python_api import AmpsCamera
    field = toy_field.replace(instrument=AmpsCamera())
    unsmoothed = sc.Recipe(sc.Model(offsets=[sc.Offsets(on='amps', smooth=0.1)]), coadd=None, name='amps')
    with pytest.raises(ConfigError, match='declares no axes to smooth along'):
        field.plan(unsmoothed)
    field.plan(unsmoothed.replace(model=sc.Model(offsets=[sc.Offsets(on='amps', smooth=0.1, smooth_along=('amp',))])))
