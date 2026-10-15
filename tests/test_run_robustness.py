"""Products and records stay right when things go off the happy path: a product written again
after its sidecar, two actions in one second, a rerun with --overwrite, a mosaic whose cal is gone,
a run-script function in a worker process, fingerprints that must be the same in every process,
interrupted writes, refused settings for a detached run."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

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
from selfcal.config.base import encode  # noqa: E402
from selfcal.io.atomic import sweep_partials  # noqa: E402
from selfcal.run import products, records  # noqa: E402
from tests.synthetic_exposures import DET, N_CHUNK_SIDE, REF_ARCSEC, write_exposures  # noqa: E402


@pytest.fixture(scope='module')
def toy():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_robust_')
    write_exposures(os.path.join(tmp, 'exposures'), 8, np.random.default_rng(5))
    field = sc.Field(os.path.join(tmp, 'out', 'toy'), sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy'),
                     REF_ARCSEC, compute=sc.Compute(os.path.join(tmp, 'cache'), workers=2))
    field.reproject(os.path.join(tmp, 'exposures', 'toy_exp_*_D0.fits'), method='interp', padding=8)
    yield field
    shutil.rmtree(tmp, ignore_errors=True)


def _recipe(iterations=20, name='r'):
    return sc.Recipe(sc.continuum(), fit=sc.Fit(iterations, tolerance=1e-8), coadd=sc.Coadd(clip=3.0),
                     numerics=sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4), name=name)


# ------------------------------------------------------------------------------- sidecars
def test_a_product_written_again_after_its_sidecar_is_changed(tmp_path):
    product = tmp_path / 'p.h5'
    product.write_bytes(b'a' * 64)
    inputs = {'kind': 'cal', 'x': 1}
    products.write_sidecar(product, inputs)
    assert products.check(product, inputs) == ('current', [])
    product.write_bytes(b'b' * 64)                       # same size, other bytes (a TOML run's rewrite)
    st = os.stat(product)
    os.utime(product, ns=(st.st_atime_ns, st.st_mtime_ns + 10_000_000))
    state, why = products.check(product, inputs)
    assert state == 'changed' and 'written again' in why[0]


def test_a_mosaic_whose_cal_is_gone_is_made_again(toy):
    recipe = _recipe(15, name='gone')
    first = toy.calibrate(recipe)
    products.remove_product(first.cal_paths[0])
    plan = toy.plan(recipe)
    states = {p.kind: p.state for p in plan.products}
    assert states == {'cal': 'missing', 'mosaic': 'replace'}
    again = toy.calibrate(recipe)
    assert products.check(again.mosaic_paths[0], products.read_sidecar(again.mosaic_paths[0])['inputs'])[0] == 'current'


# ------------------------------------------------------------------------------- records and reruns
def test_two_actions_in_one_second_keep_two_records(toy):
    recipe = _recipe(15, name='twice')
    toy.calibrate(recipe)
    a = toy.mosaic(recipe)
    b = toy.mosaic(recipe)
    assert a.record != b.record and os.path.exists(a.record) and os.path.exists(b.record)


def test_rerun_overwrite_makes_a_mosaic_again_and_keeps_the_cwd(toy):
    recipe = _recipe(15, name='remosaic')
    made = toy.calibrate(recipe)
    mosaic = made.mosaic_paths[0]
    before_bytes = open(mosaic, 'rb').read()
    record = toy.mosaic(recipe).record
    assert json.load(open(record))['products']['sidecars_written'] == []      # reused, not written
    cwd = os.getcwd()
    redone = sc.rerun(record, overwrite=True)
    assert os.getcwd() == cwd
    data = json.load(open(redone.record))
    assert data['products']['sidecars_written'] == [redone.mosaic_paths[0]]  # written again ...
    assert open(redone.mosaic_paths[0], 'rb').read() == before_bytes     # ... to the same bytes
    assert data['script'] == json.load(open(record))['script']          # the original script, not selfcal's


def test_submit_refuses_settings_a_detached_run_cannot_rebuild(toy):
    model = sc.Model(sky=[sc.Sky(damping=0.001), sc.Sky('known', times='known')],
                     offsets=[sc.Offsets(smooth=0.1, mean_zero=True)],
                     variables={'known': sc.SkyMap(np.ones(toy.reference()[1]))})
    recipe = sc.Recipe(model, fit=sc.Fit(10), coadd=None, numerics=sc.Numerics(2, batch=4), name='arrays')
    with pytest.raises(ConfigError, match='could not rebuild'):
        toy.submit(recipe)
    assert not [f for f in os.listdir(os.path.join(toy.path, 'records')) if f.startswith('submit_')
                and 'arrays' in open(os.path.join(toy.path, 'records', f)).read()]


# ------------------------------------------------------------------------------- run scripts
def test_a_run_script_function_reaches_the_worker_processes(toy, tmp_path):
    """The documented pattern: a function at the top of the run script, used in settings built at
    the top; the worker processes import the script as __mp_main__."""
    script = tmp_path / 'ramp_run.py'
    script.write_text(
        "import selfcal as sc\n"
        f"from selfcal import *\nFIELD = {toy!r}\n\n"
        "def ramp(det_x):\n    return (det_x - 32.0) / 64.0\n\n"
        "RECIPE = sc.Recipe(sc.Model(sky=[sc.Sky(damping=0.001), sc.Sky('ramp', times=ramp)],\n"
        "                            offsets=[sc.Offsets(smooth=0.1, mean_zero=True)]),\n"
        "                   fit=sc.Fit(10), coadd=None, numerics=sc.Numerics(2, batch=4), name='script_fn')\n\n"
        "if __name__ == '__main__':\n    print(FIELD.calibrate(RECIPE))\n")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, os.environ.get('PYTHONPATH')) if p))
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=600, env=env)
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-3000:]
    assert 'script_fn' in proc.stdout


# ------------------------------------------------------------------------------- fingerprints
class _Hook:
    """A hook object whose repr names its memory address (no __repr__ of its own)."""

    def __init__(self, scale=1.0):
        self.scale = scale

    def __call__(self, ctx):
        return ctx.sub_data


def test_a_hook_object_has_the_same_fingerprint_in_every_instance():
    a = products.fingerprint(encode(sc.Fit(10, frame_hook=_Hook(2.0))))
    b = products.fingerprint(encode(sc.Fit(10, frame_hook=_Hook(2.0))))
    c = products.fingerprint(encode(sc.Fit(10, frame_hook=_Hook(3.0))))
    assert a == b != c


def test_a_function_by_value_counts_by_what_it_computes(tmp_path):
    source = "def ratio(wavelength, scale=2.0):\n    return wavelength / scale\n"
    fns = []
    for d in ('kernel_a', 'kernel_b'):                     # one cell, two notebook kernels
        (tmp_path / d).mkdir()
        (tmp_path / d / f'cell_{d}.py').write_text(source)
        sys.path.insert(0, str(tmp_path / d))
        try:
            fns.append(__import__(f'cell_{d}').ratio)
        finally:
            sys.path.remove(str(tmp_path / d))
    a, b = (sc.by_value(f) for f in fns)
    assert a.sha256 != b.sha256 and a.digest == b.digest and a == b


def test_catalog_overrides_left_at_none_are_the_defaults():
    assert sc.catalog('pah_3p29', center=None, sigma=None) == sc.catalog('pah_3p29')


# ------------------------------------------------------------------------------- interrupted writes, compare
def test_interrupted_writes_are_swept(tmp_path):
    target = tmp_path / 'cal_x.h5'
    dead = tmp_path / 'cal_x.part-999999999-0.h5'
    mine = tmp_path / f'cal_x.part-{os.getpid()}-1.h5'
    fresh_dead = tmp_path / 'cal_x.part-999999998-0.h5'
    for p in (dead, mine, fresh_dead):
        p.write_bytes(b'partial')
    old = time.time() - 2 * 3600
    os.utime(dead, (old, old))
    os.utime(mine, (old, old))
    assert sweep_partials(target) == [str(dead)]          # not this process's, not a fresh one
    assert mine.exists() and fresh_dead.exists()


def test_compare_sees_an_infinity_and_the_file_type(tmp_path):
    a, b, c = tmp_path / 'a.h5', tmp_path / 'b.h5', tmp_path / 'c.fits'
    for path, values in ((a, [1.0, np.inf]), (b, [1.0, 2.0])):
        with h5py.File(path, 'w') as f:
            f['x'] = np.array(values)
    assert sc.compare(a, b).verdict == 'different'
    c.write_bytes(b'SIMPLE  =                    T' + b' ' * 2850)
    assert sc.compare(a, c).verdict == 'different'


def test_convert_never_loses_an_existing_script(tmp_path):
    from tests.test_run_products import QUICKSTART_TOML
    toml = tmp_path / 'config' / 'cal.toml'
    toml.parent.mkdir()
    toml.write_text(QUICKSTART_TOML['cal'])
    out = tmp_path / 'cal.py'
    out.write_text('# mine\n')
    from selfcal.run.convert import convert_file
    with pytest.raises(ConfigError, match='exists'):
        convert_file(str(toml), str(out))
    assert out.read_text() == '# mine\n'
    convert_file(str(toml), str(out), force=True)
    assert 'FIELD = Field(' in out.read_text()
    assert sorted(p.name for p in tmp_path.iterdir()) == ['cal.py', 'config']


def test_record_names_are_claimed(tmp_path):
    a = records._claim(str(tmp_path), 'calibrate_x')
    b = records._claim(str(tmp_path), 'calibrate_x')
    assert a != b and os.path.basename(b) == 'calibrate_x_2.json'
