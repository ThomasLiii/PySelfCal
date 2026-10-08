"""Products and records of the Python API (selfcal.run.products, records, compare, convert, the CLI):
a product is reused only when it was made by the same inputs; records rerun byte-identically;
products are written atomically; old TOML configs convert to run scripts, and the converter's rules
for settings the Python API has no switch for give the expected objects, which run."""
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
from tests.synthetic_exposures import (  # noqa: E402
    DET,
    N_CHUNK_SIDE,
    REF_ARCSEC,
    write_exposures,
    x_ramp,
    xtilde,
)


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


# An old TOML run config: the quickstart's (examples/quickstart/{reproject,cal}.toml until October 2026).
QUICKSTART_TOML = {'reproject': """task = "reproject"
output_dir = "quickstart_output"
run_name = "quickstart"
resolution_arcsec = 10.0

[instrument]
name = "grid"
tag = "Sim"
detector_shape = [64, 64]
chunks = [4, 4]
sci_ext = 1
dq_ext = 2

[reproject]
input_dirs = ["quickstart_output/exposures"]
file_pattern = "/sim_*.fits"
reproj_func = "interp"
padding_pixels = 8
max_workers = 2
""", 'cal': """task = "cal"
mode = "continuum"
output_dir = "quickstart_output"
run_name = "quickstart"
resolution_arcsec = 10.0
cache_dir = "quickstart_output/cache/"
suffix = "_quickstart"
apply_n_threads = 2

[instrument]
name = "grid"
tag = "Sim"
detector_shape = [64, 64]
chunks = [4, 4]
sci_ext = 1
dq_ext = 2

[params]
reg_weight = 0.1

[calibration]
apply_mask = true
apply_weight = false
ignore_list = []
outlier_thresh = 5.0
offset_regularization = true
weighted_damping = true
damp_weight = 0.001
batch_size = 4
max_workers = 2

[lsqr]
solver = "lsqr"
iter_lim = 200
atol = 1e-8
btol = 1e-8
damp = 0
precondition = true

[mosaic]
apply_mask = true
apply_weight = false
ignore_list = []
make_std_map = true
apply_sigma_clipping = true
sigma = 3.0
cache_intermediate = true
cache_batch_size = 4
coadd_batch_size = 4
max_workers = 2
"""}


def test_convert_writes_run_scripts(tmp_path):
    for name, text in QUICKSTART_TOML.items():
        config, out = tmp_path / f'{name}.toml', tmp_path / f'{name}.py'
        config.write_text(text)
        proc = _cli('convert', str(config), '-o', str(out))
        assert proc.returncode == 0 and 'not run' in proc.stdout, proc.stderr[-3000:]
        script = out.read_text()
        assert 'FIELD = Field(' in script and 'if __name__ == "__main__":' in script
        compile(script, str(out), 'exec')
    assert sorted(os.listdir(tmp_path)) == ['cal.py', 'cal.toml', 'reproject.py', 'reproject.toml']
    module = _import(tmp_path / 'cal.py')
    assert module.RECIPE == sc.Recipe(sc.continuum(damping=0.001), fit=sc.Fit(200, clip=5.0, ignore_flags=(),
                                                                             tolerance=1e-8),
                                      coadd=sc.Coadd(clip=3.0, ignore_flags=()),
                                      numerics=sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4), name='quickstart')
    assert module.FIELD.instrument == sc.Camera((64, 64), chunks=4, dq_ext=2, tag='Sim')


def _import(path):
    """The module of the script ``path``, imported without running its ``__main__`` block."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('_converted_' + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The converter's rules for TOML settings the Python API has no switch for, each written as what the
# engine built or read alike: the converted script holds the expected objects, and runs on the toy field.
_RULE_TOML = """task = "cal"
mode = "{mode}"
output_dir = "{out}"
run_name = "toy"
resolution_arcsec = {res}
cache_dir = "{cache}/"
suffix = "_{name}"
apply_n_threads = 2
skip_mosaic = {skip}

[instrument]
name = "grid"
tag = "Toy"
detector_shape = [{det}, {det}]
chunks = [{side}, {side}]
dq_ext = 2

[calibration]
apply_mask = true
apply_weight = false
ignore_list = []
outlier_thresh = 5.0
batch_size = 4
max_workers = 2
{cal}

[lsqr]
solver = "lsqr"
iter_lim = 30
atol = 1e-8
btol = 1e-8
damp = 0
precondition = true

[mosaic]
apply_mask = true
apply_weight = false
ignore_list = []
{mosaic}
cache_intermediate = true
cache_batch_size = 4
coadd_batch_size = 4
max_workers = 2
{rest}"""

_CLIPPED = 'make_std_map = true\napply_sigma_clipping = true\nsigma = 3.0'
_RULES = {
    # no smoothness or polynomial rows (the switch) -> every offset term smooth=0 and no poly_prior
    'noreg': dict(mode='continuum', cal='offset_regularization = false\nweighted_damping = true\ndamp_weight = 0.001',
                  rest='[params]\nreg_weight = 0.1\npoly_weight = 0.5\npoly_degree = 1\n',
                  python='offsets=(Offsets(mean_zero=True),)',
                  model=sc.Model(sky=[sc.Sky(damping=0.001)], offsets=[sc.Offsets(mean_zero=True)])),
    # no sky damping rows (the switch, despite damp_weight, damp_weight_line and a term's own) -> damping=0.0
    'nodamp': dict(mode='model', cal='offset_regularization = true\nweighted_damping = false\ndamp_weight = 0.01\n'
                                     'damp_weight_line = 0.02',
                   rest='[model]\nscalar = true\nmosaic = "none"\n\n[[model.sky]]\nname = "continuum"\n\n'
                        '[[model.sky]]\nname = "ramp"\ndamp_weight = 0.05\n'
                        'coefficient = { variable = "det_x", function = "tests.synthetic_exposures:x_ramp" }\n\n'
                        '[[model.offset]]\nkind = "free"\nreg_weight = 0.1\nmean_zero = true\n',
                   python="sky=(Sky(damping=0.0), Sky('ramp', times=Function(x_ramp, of='det_x'), damping=0.0))",
                   model=sc.Model(sky=[sc.Sky(damping=0.0), sc.Sky('ramp', times=sc.Function(x_ramp, of='det_x'),
                                                                   damping=0.0)],
                                  offsets=[sc.Offsets(smooth=0.1, mean_zero=True)])),
    # a basis of one function -> times= (the cal labels the map 'basis' alike); with the mosaic, which
    # subtracts the term through the same basis
    'basis1': dict(mode='model', cal='offset_regularization = true\nweighted_damping = true\ndamp_weight = 0.001',
                   rest='[model]\nscalar = true\nmosaic = "full"\n\n[[model.sky]]\nname = "continuum"\n\n'
                        '[[model.offset]]\nkind = "free"\nreg_weight = 0.1\nmean_zero = true\n\n'
                        '[[model.offset]]\nkind = "free"\nname = "xslope"\n'
                        'basis = { variable = "det_x", function = "tests.synthetic_exposures:xtilde", n = 1 }\n',
                   mosaic=_CLIPPED, python="Offsets('xslope', times=Function(xtilde, of='det_x'))",
                   model=sc.Model(sky=[sc.Sky(damping=0.001)],
                                  offsets=[sc.Offsets(smooth=0.1, mean_zero=True),
                                           sc.Offsets('xslope', times=sc.Function(xtilde, of='det_x'))])),
    # the mosaic's sigma without a sigma-clip pass (nor instrument maps) is read by nothing -> the Coadd's
    # default
    'nosigma': dict(mode='continuum', cal='weighted_damping = true\ndamp_weight = 0.001', rest='',
                    mosaic='make_std_map = false\napply_sigma_clipping = false\nsigma = 1.0',
                    python='coadd=Coadd(None, std=False, ignore_flags=())',
                    model=sc.Model(sky=[sc.Sky(damping=0.001)], offsets=[sc.Offsets(mean_zero=True)]),
                    coadd=sc.Coadd(None, std=False, ignore_flags=())),
}


@pytest.mark.parametrize('name', sorted(_RULES))
def test_converter_rules(toy_field, tmp_path, name):
    from selfcal.run.convert import convert_file
    case = _RULES[name]
    mosaic = case.get('mosaic')
    config = tmp_path / f'{name}.toml'
    config.write_text(_RULE_TOML.format(mode=case['mode'], out=os.path.dirname(toy_field.path), res=REF_ARCSEC,
                                        cache=tmp_path / 'cache', name=name, skip=str(mosaic is None).lower(),
                                        det=DET, side=N_CHUNK_SIDE, cal=case['cal'], mosaic=mosaic or _CLIPPED,
                                        rest=case['rest']))
    script = tmp_path / f'{name}.py'
    convert_file(str(config), str(script))
    text = script.read_text()
    assert case['python'] in text and 'basis=' not in text, text
    module = _import(script)
    coadd = case.get('coadd', sc.Coadd(clip=3.0, ignore_flags=()) if mosaic is not None else None)
    assert module.RECIPE == sc.Recipe(case['model'], fit=sc.Fit(30, clip=5.0, ignore_flags=(), tolerance=1e-8),
                                      coadd=coadd, numerics=sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4),
                                      name=name)
    assert module.FIELD.path == toy_field.path and module.FIELD.instrument == toy_field.instrument
    result = module.FIELD.calibrate(module.RECIPE, **module.RUN)
    assert [os.path.basename(p) for p in result.cal_paths] == [f'cal_Toy_Chunks{N_CHUNK_SIDE}x{N_CHUNK_SIDE}_All_{name}.h5']
    assert len(result.mosaic_paths) == (mosaic is not None)


def test_converter_refuses_what_it_cannot_translate(tmp_path):
    """The N-pass SKY pass of a TOML run ignored weighted_damping and defaulted damp_weight to 0.0: a config whose
    SKY passes damped otherwise than its joint solve has no Python form (a term carries one damping). Options the
    library no longer has are refused unless at the value it always uses."""
    from selfcal.run.convert import from_toml

    def converted(calibration, n=3, lsqr=''):
        passes = f'[passes]\nn = {n}\n[passes.sky]\nsubch_clip = false\n[passes.offset]\nsubch_clip = false\n'
        path = tmp_path / 'npass.toml'
        path.write_text(f'task = "npass"\nmode = "continuum"\noutput_dir = "{tmp_path}"\nrun_name = "toy"\n'
                        f'resolution_arcsec = 20.0\ncache_dir = "{tmp_path}/cache/"\n\n[instrument]\nname = "grid"\n'
                        f'detector_shape = [64, 64]\n\n[calibration]\n{calibration}\n\n[lsqr]\n{lsqr}\n\n{passes}')
        return from_toml(str(path))

    for refused in ('weighted_damping = false\ndamp_weight = 0.1',      # SKY pass 0.1, joint none
                    'weighted_damping = true'):                       # joint 0.1 (setup_lsqr), SKY pass 0.0
        with pytest.raises(ConfigError, match='N-pass SKY passes damp'):
            converted(refused)
    assert converted('weighted_damping = false').recipe.model.sky_dampings() == [0.0]
    assert converted('weighted_damping = true\ndamp_weight = 0.1').recipe.model.sky_dampings() == [0.1]
    assert converted('weighted_damping = false\ndamp_weight = 0.1', n=1).recipe.model.sky_dampings() == [0.0]  # no SKY pass
    with pytest.raises(ConfigError, match='compact_zero_columns'):
        converted('weighted_damping = false\ncompact_zero_columns = false')
    with pytest.raises(ConfigError, match='keep_state'):
        converted('weighted_damping = false', lsqr='keep_state = true')
    kept = converted('weighted_damping = false\ncompact_zero_columns = true', lsqr='resume = false')
    assert any('compact_zero_columns' in n for n in kept.notes) and any('resume' in n for n in kept.notes)


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
