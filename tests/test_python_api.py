"""The Python API (selfcal.config, selfcal.models.model, selfcal.run.{recipe,schedule,compute,field,lower}).

Settings are checked when built; they lower to the run configs a TOML run makes; and a run
through the Python API makes the same bytes as the same run through a TOML config (a toy data
set with the built-in grid camera: calibrate + mosaic, the mosaic action, a tiled solve).
"""
import functools
import os
import pickle
import shutil
import sys
import tempfile
from dataclasses import KW_ONLY, dataclass

import h5py
import numpy as np
import pytest
from astropy.io import fits

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import selfcal as sc  # noqa: E402
from selfcal import _state  # noqa: E402
from selfcal.config import Config, ConfigError  # noqa: E402
from selfcal.instruments import spherex  # noqa: E402
from selfcal.run.lower import lower  # noqa: E402
from tests.synthetic_exposures import DET, N_CHUNK_SIDE, REF_ARCSEC, write_exposures  # noqa: E402


# ------------------------------------------------------------------- functions the tests reference
def above_80(temperature, t0=80.0):
    return temperature - t0


def gradient(det_x, det_y):
    return [(det_x - 32) / 64, (det_y - 24) / 48]


def per_frame_index(frames):
    return np.arange(len(frames), dtype=float)


# =================================================================== the settings base
@dataclass(frozen=True)
class _Probe(Config):
    count: int = 3
    _: KW_ONLY
    scale: float = 1.0
    names: tuple[str, ...] = ()
    choice: str | None = None


def test_settings_are_checked_converted_and_immutable():
    p = _Probe(5, scale=2, names=['a', 'b'])
    assert p.scale == 2.0 and isinstance(p.scale, float) and p.names == ('a', 'b')
    with pytest.raises(TypeError, match="Did you mean 'scale'"):
        _Probe(scael=2)
    with pytest.raises(ConfigError, match=r"_Probe\(count=...\): expected an integer"):
        _Probe(2.5)
    with pytest.raises(Exception):
        p.count = 7                                         # frozen
    q = p.replace(count=9)
    assert q.count == 9 and p.count == 5
    assert repr(p) == "_Probe(5, scale=2.0, names=('a', 'b'))"
    assert Config.from_dict(p.to_dict()) == p
    assert pickle.loads(pickle.dumps(p)) == p
    assert _Probe(names='single').names == ('single',)


def test_choices_suggest_the_closest():
    with pytest.raises(ConfigError, match="Did you mean 'lsqr'"):
        sc.Fit(method='lsq')


# =================================================================== the model
def test_functions_read_their_variables_from_the_signature():
    f = sc.Function(above_80)
    assert f.of == ('temperature',) and f.lower()['function'].endswith(':above_80')
    f = sc.Function(functools.partial(above_80, t0=70.0))
    assert f.of == ('temperature',) and f.lower()['params'] == {'t0': 70.0}
    assert sc.Function(np.cos, of='phase').lower() == {'variable': 'phase', 'function': 'numpy:cos'}
    with pytest.raises(ConfigError, match='has no parameter'):
        sc.Function(above_80, t1=3.0)


def test_functions_that_workers_cannot_import_are_refused():
    with pytest.raises(ConfigError, match='lambda'):
        sc.Sky('s', times=lambda wavelength: wavelength)

    def nested(wavelength):
        return wavelength
    with pytest.raises(ConfigError, match='inside another function'):
        sc.Sky('s', times=nested)


def test_model_lowers_to_the_model_table():
    model = sc.Model(sky=[sc.Sky(damping=1e-4), sc.Sky('line', times=sc.gaussian(3.3, sigma=0.01))],
                     offsets=[sc.Offsets('thermal', per='all', times=above_80, mean_zero=True),
                              sc.Offsets('gradient', on='detector', basis=gradient, n=2),
                              sc.Offsets('nights', per='night', smooth=0.1, smooth_along=())],
                     variables={'temperature': sc.Header('TEMP'), 'night': sc.PerFrame(per_frame_index)},
                     priors=[sc.priors.frame_smoothness('thermal', 'temperature', weight=2.0)])
    t = model.lower()
    assert [s['damp_weight'] for s in t['sky']] == [1e-4, 0.3]      # the second term's default
    assert t['sky'][1]['coefficient'] == {'variable': 'wavelength', 'function': 'gaussian', 'center': 3.3,
                                          'sigma': 0.01}
    kinds = [(o['kind'], o.get('groups')) for o in t['offset']]
    assert kinds == [('fixed', None), ('free', None), ('grouped', 'night')]
    assert t['offset'][1]['basis']['n'] == 2
    assert t['prior'][0] == {'terms': ['thermal'], 'function': 'frame_smoothness', 'weight': 2.0,
                             'params': {'variable': 'temperature', 'power': 0.0, 'covered_only': True}}
    spec = model.spec()
    assert [v.name for v in spec.variables] == ['temperature', 'night']


def test_presets_and_their_rules():
    m = sc.continuum(smooth=0.2, poly_prior=sc.Poly(1, weight=0.5), damping=0.001)
    assert m.lower()['offset'][0]['poly'] == [{'axis': None, 'degree': 1, 'weight': 0.5}]
    s = sc.spectral([spherex.line('aromatic', damping=5e-3)], polynomial=sc.Poly(2, window=range(200, 321)))
    assert s.spectral_window() == (200, 320)
    assert s.lower()['offset'][0]['kind'] == 'polybasis'
    with pytest.raises(ConfigError, match='takes no smooth'):
        sc.spectral([], polynomial=sc.Poly(2, window=range(200, 321)), smooth=0.1)
    with pytest.raises(ConfigError, match='needs Poly\\(window'):
        sc.Offsets(polynomial=sc.Poly(2))
    with pytest.raises(ConfigError, match='basis= needs n='):
        sc.Offsets(basis=gradient)


# =================================================================== recipe, schedule, compute
def test_recipe_parts():
    fit = sc.Fit(100, clip=sc.Clip(2.5, per=sc.ChunkGroups.along('subchannel')), tolerance=(1e-8, 1e-6))
    assert fit.atol_btol == (1e-8, 1e-6)
    assert sc.Fit(clip=3).clip == sc.Clip(3.0)
    with pytest.raises(ConfigError, match='needs the std map'):
        sc.Coadd(clip=2.0, std=False)
    assert sc.Recipe(name='x').model == sc.continuum() and sc.Recipe(name='x').suffix == '_x'
    with pytest.raises(ConfigError, match='ends on an OFFSET pass'):
        sc.Passes(n=2)
    assert sc.Passes(n=2, ends_on_offset=True).n == 2
    with pytest.raises(ConfigError, match='needs \\{tile\\}'):
        sc.Tiles((2, 2), tile_name='same')


# =================================================================== instruments
def test_spherex_jobs_lower_to_the_instrument_table():
    inst = sc.SPHEREx(3, num_col=3)
    assert inst.product_tag == 'Detector3_NumSub10_NumCh34_NumCol3'
    name, table = inst.engine(spherex.channels(16, 18))
    assert name == 'spherex' and table['channels'] == [[16], [17], [18]]
    _, table = inst.engine((spherex.window('Multiline3', subchannels=range(200, 321)),))
    assert table['windows'] == ['Multiline3'] and table['window_defs'] == {'Multiline3': [200, 321]}
    assert spherex.group(17, 18).name == 'Ch17-18'
    with pytest.raises(ConfigError, match='1 to 6'):
        sc.SPHEREx(7)
    with pytest.raises(ConfigError, match='name their maps'):
        inst.default_jobs()


def test_camera_and_euclid_tables():
    cam = sc.Camera((64, 48), chunks=3, dq_ext=2, tag='Toy')
    assert cam.engine(())[1] == {'name': 'grid', 'detector_shape': [64, 48], 'chunks': [3, 3], 'sci_ext': 1,
                                 'dq_ext': 2, 'tag': 'Toy'}
    assert cam.product_tag == 'Toy_Chunks3x3'
    eu = sc.Euclid(band='J', chunks=510)
    assert eu.default_jobs()[0].name == 'J' and eu.default_ignore_flags() == (11, 15)


@dataclass(frozen=True, kw_only=True)
class Owl(sc.Instrument):
    """A user-defined instrument: rectangular chunks and an amplifier map."""
    chunks: int = 4
    tag: str = 'Owl'
    unit: str = 'e-/s'

    def geometry(self, oversample):
        grid = sc.ChunkMap.rectangles('grid', (DET, DET), (self.chunks, self.chunks))
        amps = sc.ChunkMap.rectangles('amps', (DET, DET), (1, 4), axes=('all', 'amp'))
        return sc.Geometry((DET, DET), oversample, maps=[grid, amps])


@dataclass(frozen=True, kw_only=True)
class AmpsCamera(sc.Instrument):
    """A camera with a second chunk map that declares no adjacency axes (four amplifiers)."""
    tag: str = 'Amps'

    def geometry(self, oversample):
        grid = sc.ChunkMap.rectangles('grid', (DET, DET), (N_CHUNK_SIDE, N_CHUNK_SIDE))
        ids = np.repeat(np.arange(4, dtype=np.int32)[None, :], DET, axis=0).repeat(DET // 4, axis=1)
        amps = sc.ChunkMap('amps', ids, ids, axes=sc.ChunkAxes.row_major(('amp',), (4,), ('x',)))
        return sc.Geometry((DET, DET), oversample, maps=[grid, amps])


def test_a_new_instrument_is_a_subclass():
    from selfcal.run.engine import RunContext
    owl = Owl(chunks=2)
    field = sc.Field('/tmp/owl_field', owl, 20.0)
    (low,) = lower(field, sc.Recipe(coadd=None))
    ctx = RunContext.build(low.cfg)
    assert ctx.frame_tag == 'Owl' and [j.name for j in ctx.jobs()] == ['All']
    assert sorted(ctx.geom.chunk_maps) == ['amps', 'grid'] and ctx.geom.chunk_map.n_chunks == 4
    assert ctx.geom.chunk_maps['amps'].grid.shape == (DET, DET)


@dataclass(frozen=True, kw_only=True)
class RenderedOwl(Owl):
    """Owl with two of the engine's optional hooks."""

    def coefficient_catalog(self):
        return {'ramp': functools.partial(sc.linear, 0.0, 1.0)}

    def offset_renderer(self, geom, jobgeom, map_name=None, render=None):
        return _render_constant if map_name in (None, 'grid') else None


def _render_constant(chunk_map, offsets):
    return np.asarray(offsets)[chunk_map]


def test_an_instruments_optional_hooks_reach_the_engine():
    engine, _ = RenderedOwl(chunks=2).engine(())
    assert sorted(engine.coefficient_catalog()) == ['ramp']
    assert engine.offset_renderer({}, None, None) is _render_constant
    assert engine.offset_renderer({}, None, None, map_name='amps') is None
    assert engine.aux_coadds(None) is None
    plain, _ = Owl().engine(())
    assert plain.coefficient_catalog() == {} and plain.offset_renderer({}, None, None) is None


def test_built_in_instruments_give_their_geometry():
    geom = sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy').geometry(2)
    assert geom.chunk_map.n_chunks == N_CHUNK_SIDE ** 2 and geom.chunk_map.grid.shape == (2 * DET, 2 * DET)
    from selfcal.instruments.spherex.spherex_utility import DEFAULT_CALIBRATION_DIR
    if os.path.isdir(os.environ.get('SELFCAL_SPHEREX_CALIB_DIR', DEFAULT_CALIBRATION_DIR)):   # its maps (not on CI)
        lvf = sc.SPHEREx(4, num_col=3).geometry(1)
        assert lvf.chunk_map.axes is not None and 'subchannel' in lvf.chunk_map.axes.names
    with pytest.raises(NotImplementedError, match='a subclass of sc.Instrument implements'):
        sc.Instrument.geometry(_NoGeometry())


@dataclass(frozen=True, kw_only=True)
class _NoGeometry(sc.Instrument):
    tag: str = 'None'


def test_map_variables_take_arrays():
    """sc.SkyMap / sc.DetectorMap of an array reach the solve (they took only files and functions)."""
    from selfcal.instruments.base import get_instrument
    geom = get_instrument('grid').detector_geometry({'detector_shape': [8, 8], 'chunks': [2]}, 1)
    model = sc.Model(variables={'known': sc.SkyMap(np.full((4, 4), 2.0)), 'qe': sc.DetectorMap(np.ones((8, 8)))})
    variables = model.spec().build_variables(geom, [], ref_shape=(4, 4), log=lambda *a, **k: None)
    assert float(np.asarray(variables.sky['known']).mean()) == 2.0
    assert np.asarray(variables.detector['qe']).shape == (8, 8)


# =================================================================== the same bytes as a TOML run
def _datasets(path):
    out = {}
    with h5py.File(path, 'r') as f:
        f.visititems(lambda name, obj: out.__setitem__(name, (obj[()], dict(obj.attrs)))
                     if isinstance(obj, h5py.Dataset) else None)
        out['/'] = dict(f.attrs)
    return out


def _same_h5(a, b):
    da, db = _datasets(a), _datasets(b)
    assert sorted(da) == sorted(db)
    for k in da:
        if k == '/':
            assert {x: np.asarray(v).tolist() for x, v in da[k].items()} == \
                   {x: np.asarray(v).tolist() for x, v in db[k].items()}, k
            continue
        va, vb = da[k][0], db[k][0]
        assert np.asarray(va).dtype == np.asarray(vb).dtype and np.asarray(va).tobytes() == np.asarray(vb).tobytes(), k


def _same_fits(a, b):
    with fits.open(a) as ha, fits.open(b) as hb:
        assert [h.name for h in ha] == [h.name for h in hb]
        for x, y in zip(ha, hb):
            if x.data is not None:
                assert x.data.tobytes() == y.data.tobytes(), x.name


def _toml(path, text):
    with open(path, 'w') as f:
        f.write(text)
    from selfcal.run.config import load_config
    return load_config(path)


_TOML_COMMON = """
output_dir = "{out}"
run_name = "toy_run"
resolution_arcsec = {ref}
cache_dir = "{cache}/"
suffix = "_t"
apply_n_threads = 2
hdd_io_limit = 4
wavelength_coadd = false
[instrument]
name = "grid"
tag = "Toy"
detector_shape = [{det}, {det}]
chunks = [{nc}]
sci_ext = 1
dq_ext = 2
[params]
reg_weight = 0.1
[calibration]
apply_mask = true
apply_weight = false
outlier_thresh = 5.0
ignore_list = []
batch_size = 4
offset_regularization = true
weighted_damping = true
damp_weight = 0.1
max_workers = 2
[lsqr]
atol = 1e-8
btol = 1e-8
damp = 0
iter_lim = 60
precondition = true
solver = "lsqr"
[mosaic]
apply_mask = true
apply_weight = false
make_std_map = true
apply_sigma_clipping = true
sigma = 3.0
ignore_list = []
cache_batch_size = 4
coadd_batch_size = 4
cache_intermediate = true
max_workers = 2
"""


def test_a_python_run_makes_the_bytes_of_the_toml_run():
    from selfcal.run import pipelines
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_pyapi_')
    try:
        exp_dir = os.path.join(tmp, 'exposures')
        write_exposures(exp_dir, 12, np.random.default_rng(5))
        out, cache = os.path.join(tmp, 'out'), os.path.join(tmp, 'cache')
        fmt = dict(out=out, cache=cache, ref=REF_ARCSEC, det=DET, nc=N_CHUNK_SIDE)
        run_dir = os.path.join(out, 'toy_run')
        common = _TOML_COMMON.format(**fmt)

        # TOML: reproject, cal (+ mosaic), tiled cal
        rcfg = _toml(os.path.join(tmp, 'r.toml'), 'task = "reproject"' + common + f"""
[reproject]
input_dirs = ["{exp_dir}"]
file_pattern = "/toy_exp_*_D0.fits"
padding_pixels = 8
max_workers = 2
reproj_func = "interp"
""")
        pipelines.run(rcfg)
        res = pipelines.run(_toml(os.path.join(tmp, 'c.toml'), 'task = "cal"\nmode = "continuum"' + common))
        tiled = pipelines.run(_toml(os.path.join(tmp, 't.toml'),
                                    'task = "cal"\nmode = "continuum"' + common.replace('suffix = "_t"', 'suffix = "_t_{tile}"')
                                    + f"""
[tiling]
grid = [1, 2]
overlap_px = 10
tile_names = ["W", "E"]
ref_shape = {list(fits.getdata(os.path.join(run_dir, 'ref.fits')).shape)}
full_reproj_dir = "{run_dir}/reprojected"
nvme_subdir = "toy_tiles"
stitched_suffix = "_t_stitched"
line = false
rss_guardrail = false
"""))
        golden = os.path.join(tmp, 'golden')
        shutil.move(os.path.join(run_dir, 'calibration'), golden)
        shutil.move(os.path.join(run_dir, 'mosaic'), os.path.join(golden, 'mosaic'))

        # the Python API, same field folder
        field = sc.Field(run_dir, sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy'), REF_ARCSEC,
                         compute=sc.Compute(cache, workers=2, io_limit=4))
        recipe = sc.Recipe(sc.continuum(smooth=0.1), fit=sc.Fit(60, tolerance=1e-8), coadd=sc.Coadd(clip=3.0),
                           numerics=sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4), name='t')
        plan = field.plan(recipe)
        assert 'cal_Toy_Chunks4x4_All_t.h5' in str(plan) and plan.frames[0] == 12
        result = field.calibrate(recipe)
        _same_h5(result.cal_paths[0], os.path.join(golden, os.path.basename(res.cal_paths[0])))
        _same_fits(result.mosaic_paths[0], os.path.join(golden, 'mosaic', os.path.basename(res.mosaic_paths[0])))
        assert os.path.exists(result.record)
        assert result.offsets().shape == (12, N_CHUNK_SIDE ** 2) and result.sky().ndim == 2

        os.remove(result.mosaic_paths[0])                    # the mosaic action, from the existing cal
        again = field.mosaic(recipe)
        _same_fits(again.mosaic_paths[0], os.path.join(golden, 'mosaic', os.path.basename(res.mosaic_paths[0])))

        tiles = sc.Tiles((1, 2), overlap=10, names=('W', 'E'))
        t = field.calibrate(recipe.replace(coadd=None), tiles=tiles,
                            compute=field.compute.replace(stage_dir='toy_tiles', memory_guard=False))
        _same_h5(t.final, os.path.join(golden, os.path.basename(tiled.stitched)))
        for name, path in t.tile_cals.items():
            _same_h5(path, os.path.join(golden, os.path.basename(tiled.tiles[name])))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# =================================================================== the run script's rules
def _run_script(tmp_path, text):
    import subprocess
    script = tmp_path / 'run_script.py'
    script.write_text(text)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in (_REPO, os.environ.get('PYTHONPATH')) if p))
    return subprocess.run([sys.executable, str(script)], cwd=tmp_path, env=env, capture_output=True, text=True,
                          timeout=120)


def test_a_run_outside_the_main_guard_is_refused(tmp_path):
    proc = _run_script(tmp_path, "import selfcal as sc\n"
                                 "field = sc.Field('out/x', sc.Camera((8, 8)), 1.0)\n"
                                 "field.plan()\n")
    assert proc.returncode != 0 and 'if __name__ == "__main__"' in proc.stderr, proc.stderr[-2000:]


def test_a_function_defined_under_the_guard_is_refused(tmp_path):
    proc = _run_script(tmp_path, "import selfcal as sc\n"
                                 "if __name__ == '__main__':\n"
                                 "    def ratio(wavelength):\n"
                                 "        return wavelength\n"
                                 "    sc.Sky('line', times=ratio)\n")
    assert proc.returncode != 0 and 'Move the definition above the guard' in proc.stderr, proc.stderr[-2000:]


def test_a_script_function_is_recorded_under_the_script_name(tmp_path):
    proc = _run_script(tmp_path, "import selfcal as sc\n"
                                 "def ratio(wavelength, scale=2.0):\n"
                                 "    return wavelength / scale\n"
                                 "if __name__ == '__main__':\n"
                                 "    print(sc.Sky('line', times=ratio).lower(0.1)['coefficient'])\n")
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "'function': 'run_script:ratio'" in proc.stdout


# =================================================================== clips over chunk groups
def test_each_pixel_takes_the_group_of_its_dominant_chunk():
    import scipy.sparse as sp

    from selfcal.core.assembly import chunk_group_of_pixels
    # 3 chunks x 6 pixels: pixel 1 is mostly chunk 2, pixel 5 has no chunk
    contrib = sp.csr_matrix(np.array([[1.0, 0.2, 0.0, 0.0, 0.5, 0.0],
                                      [0.0, 0.0, 1.0, 0.0, 0.5, 0.0],
                                      [0.0, 0.8, 0.0, 1.0, 0.0, 0.0]]))
    groups = chunk_group_of_pixels(contrib, np.array([10, 11, 12]), (2, 3))
    assert groups.tolist() == [[10, 12, 11], [12, 10, -1]]


def test_clip_groups_resolve_on_the_primary_chunk_map():
    from selfcal.run.engine import RunContext, clip_groups
    field = sc.Field('/tmp/clip_field', sc.Camera((DET, DET), chunks=N_CHUNK_SIDE), 20.0)
    (low,) = lower(field, sc.Recipe(coadd=None))
    ctx = RunContext.build(low.cfg)
    n = N_CHUNK_SIDE ** 2
    assert clip_groups(ctx, {'chunk': True})['outlier_chunk_groups'].tolist() == list(range(n))
    rows = clip_groups(ctx, {'along': 'row'})['outlier_chunk_groups']
    assert rows.tolist() == [c // N_CHUNK_SIDE for c in range(n)]
    assert clip_groups(ctx, {'mapping': [c % 2 for c in range(n)]})['outlier_chunk_groups'].tolist() == \
        [c % 2 for c in range(n)]
    for bad in ({'mapping': [0, 1]}, {'along': 'subchannel'}, {'along': 'row', 'map': 'other'}):
        with pytest.raises(ValueError):
            clip_groups(ctx, bad)


def test_a_fit_clipped_per_chunk_group_runs():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_clip_')
    try:
        write_exposures(os.path.join(tmp, 'exposures'), 8, np.random.default_rng(7))
        field = sc.Field(os.path.join(tmp, 'out', 'clip_run'),
                         sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Clip'), REF_ARCSEC,
                         compute=sc.Compute(os.path.join(tmp, 'cache'), workers=2))
        field.reproject(os.path.join(tmp, 'exposures', 'toy_exp_*_D0.fits'), method='interp', padding=8)
        cals = {}
        for name, per in (('frame', 'frame'), ('chunk', 'chunk'), ('rows', sc.ChunkGroups.along('row'))):
            recipe = sc.Recipe(fit=sc.Fit(30, clip=sc.Clip(2.0, per=per)), coadd=None,
                               numerics=sc.Numerics(2, batch=4), name=name)
            cals[name] = field.calibrate(recipe).cal_paths[0]
        with h5py.File(cals['frame']) as a, h5py.File(cals['chunk']) as b, h5py.File(cals['rows']) as c:
            sky = [f['sky/continuum'][()] for f in (a, b, c)]
        assert all(np.isfinite(s).any() for s in sky)
        assert not np.array_equal(sky[0], sky[1], equal_nan=True)     # a different clip, a different solve
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
