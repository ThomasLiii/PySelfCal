"""The Python API (selfcal.config, selfcal.models.model, selfcal.run.{recipe,schedule,compute,field,lower,engine}).

Settings are checked when built (and a setting added later keeps the products' fingerprints); they
lower to the engine's runs, which call the instrument's contract; a run through the Python API on
a toy data set with the built-in camera: calibrate + mosaic (the injected offsets recovered), the
mosaic action, a tiled solve, a camera without a mask; the geometry is built once for a plan and its
action; a model grouped by a frame variable of its own is mosaicked.
"""
import dataclasses
import functools
import os
import pickle
import shutil
import sys
import tempfile
import threading
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
from selfcal.config.base import added, encode, type_name  # noqa: E402
from selfcal.instruments import spherex  # noqa: E402
from selfcal.io.calfile import CalFile  # noqa: E402
from selfcal.models.spec import ModelSpec  # noqa: E402
from selfcal.run.engine import RunContext, chunk_map_specs, with_chunk_maps  # noqa: E402
from selfcal.run.lower import lower  # noqa: E402
from selfcal.run.products import fingerprint, job_key  # noqa: E402
from selfcal_scripts.spherex import jobs  # noqa: E402
from selfcal_scripts.spherex import terms as T  # noqa: E402
from tests.synthetic_exposures import (  # noqa: E402
    DET,
    N_CHUNK_SIDE,
    REF_ARCSEC,
    write_exposures,
    x_ramp,
)


# ------------------------------------------------------------------- functions the tests reference
def above_80(temperature, t0=80.0):
    return temperature - t0


def gradient(det_x, det_y):
    return [(det_x - 32) / 64, (det_y - 24) / 48]


def per_frame_index(frames):
    return np.arange(len(frames), dtype=float)


def fortnight(exposure, length=8):
    return np.asarray(exposure) // length


def plane(det_x, det_y):
    return [(np.asarray(det_x, dtype=np.float64) - DET / 2) / DET, (np.asarray(det_y, dtype=np.float64) - DET / 2) / DET]


#: The job of a camera's runs: every chunk of the primary chunk map.
ALL = sc.Job('All')


def map_specs(inst, model=None):
    """The chunk maps a run of ``model`` on ``inst`` evaluates (a model naming none, the default: the
    instrument's)."""
    return chunk_map_specs(inst, ModelSpec.from_config((model or sc.Model()).lower()))


def detector_geometry(inst, oversample=1, model=None):
    """``inst``'s detector geometry with the chunk maps of ``model`` evaluated on it, as the run engine builds
    it."""
    return with_chunk_maps(inst.geometry(oversample), map_specs(inst, model))


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


@dataclass(frozen=True)
class _Versioned(Config):
    count: int = 3
    _: KW_ONLY
    snapshot_every: int | None = added(None, since='2026-10-08')


def test_a_setting_added_later_keeps_the_fingerprints():
    before = {'type': type_name(_Versioned), 'count': 3}        # encoded before the setting existed
    v = _Versioned()
    assert v.to_dict() == before and fingerprint(encode(v)) == fingerprint(before)
    w = _Versioned(snapshot_every=10)
    assert w.to_dict() == {**before, 'snapshot_every': 10} and fingerprint(encode(w)) != fingerprint(before)
    assert Config.from_dict(before) == v and Config.from_dict(w.to_dict()) == w
    assert repr(w) == '_Versioned(snapshot_every=10)'


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
                     offsets=[sc.Offsets('thermal', per='all', times=above_80, exclude=sc.Poly(0)),
                              sc.Offsets('gradient', on='detector', basis=gradient, n=2),
                              sc.Offsets('nights', per='night', smooth=0.1, smooth_along=())],
                     variables={'temperature': sc.Header('TEMP'), 'night': sc.PerFrame(per_frame_index)},
                     priors=[sc.priors.frame_smoothness('thermal', 'temperature', weight=2.0)])
    t = model.lower()
    assert [s['damp_weight'] for s in t['sky']] == [1e-4, 0.0]      # the second term's default: undamped
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
    s = sc.spectral([T.line('aromatic', damping=5e-3)], polynomial=sc.Poly(2, window=range(200, 321)))
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
def test_spherex_jobs_and_their_runs():
    from selfcal_scripts.recipes.spherex import maps
    inst = sc.SPHEREx(3)
    assert inst.product_tag == 'Detector3' and inst.default_chunk_maps() == {}
    chans = jobs.channels(16, 18)
    assert [j.name for j in chans] == ['Ch16', 'Ch17', 'Ch18']
    # a channel's window is its subchannels padded by one on each side, its strict set its own
    assert chans[1].window == {'subchannel': range(160, 172)} and chans[1].strict == {'subchannel': range(161, 171)}
    window = jobs.window('Multiline3', range(200, 321))
    assert window.window == {'subchannel': range(200, 321)} and window.strict is None
    assert jobs.window('Aromatic').window == {'subchannel': jobs.WINDOWS['Aromatic']}
    group = jobs.group(17, 18)
    assert group.name == 'Ch17-18' and group.strict == {'subchannel': range(161, 181)}
    with pytest.raises(ConfigError, match='consecutive channels'):
        jobs.group(17, 19)
    # channel jobs and window jobs run together, on the recipe's chunk maps, whose primary map tags the products
    recipe = sc.Recipe(sc.continuum().replace(chunk_maps=maps(num_ch=34, num_col=3)), coadd=None)
    (run,) = lower(sc.Field('/tmp/spherex_field', inst, 6.2), recipe, jobs=chans + (window,))
    assert [j.name for j in run.jobs] == ['Ch16', 'Ch17', 'Ch18', 'Multiline3']
    assert RunContext.build(run, need_geometry=False).frame_tag == 'Detector3_NumSub10_NumCh34_NumCol3'
    # SPHEREx offers no chunk maps: a model names its own
    (plain,) = lower(sc.Field('/tmp/spherex_field', inst, 6.2), sc.Recipe(coadd=None), jobs=chans)
    with pytest.raises(ConfigError, match='has no default chunk maps'):
        RunContext.build(plain, need_geometry=False)
    with pytest.raises(ConfigError, match='1 to 6'):
        sc.SPHEREx(7)
    with pytest.raises(TypeError):
        sc.SPHEREx(3, num_col=3)                        # the partition is the model's


def test_a_job_names_chunks_on_the_primary_map():
    """A job's window and strict set select the chunks of the primary map by their indices along its axes, and a
    job naming an axis the map lacks, reaching beyond it, or keeping chunks outside its window is refused; an
    action names its jobs."""
    from selfcal.instruments.contract import check_job, chunk_window
    cm = sc.Chunks.rectangles((DET, DET), (2, 3))              # chunk id = row * 3 + col
    assert chunk_window(cm, None).all() and chunk_window(cm, {'col': range(1, 3)}).tolist() == [0, 1, 1] * 2
    assert chunk_window(cm, {'row': range(1, 2), 'col': range(0, 1)}).nonzero()[0].tolist() == [3]
    check_job(sc.Job('Right', window={'col': range(1, 3)}, strict={'col': range(2, 3)}), cm)
    for job, msg in ((sc.Job('A', window={'band': range(0, 1)}), "no chunk axis 'band'"),
                     (sc.Job('B', window={'col': range(2, 4)}), 'lies outside'),
                     (sc.Job('C', window={'col': range(0, 1)}, strict={'col': range(1, 2)}), 'leaves its window')):
        with pytest.raises(ConfigError, match=msg):
            check_job(job, cm)
    with pytest.raises(ConfigError, match='consecutive chunk indices'):
        sc.Job('D', window={'col': range(0, 3, 2)})
    assert job_key(sc.Job('Right', window={'col': range(1, 3)})) == \
        {'name': 'Right', 'window': {'col': [1, 3]}, 'strict': None}
    with pytest.raises(ConfigError, match='jobs='):
        lower(sc.Field('/tmp/owl_field', sc.Camera((DET, DET)), 20.0), sc.Recipe(coadd=None))


def test_the_zodi_anchor_reads_the_channel_of_each_cal(tmp_path, monkeypatch):
    """spherex.zodi_anchor anchors each single-channel cal, its name ending in the job's (``..._Ch17.h5``, a recipe
    without a name) or followed by the recipe's (``..._Ch17_name.h5``); a group of channels is skipped."""
    import selfcal.zodi_anchor as za
    from selfcal.run.result import Result
    anchored = []
    monkeypatch.setattr(za, 'fit_anchor_for_channel', lambda cal, npz, **clip: os.path.basename(cal))
    monkeypatch.setattr(za, 'append_anchor_channel',
                        lambda path, det, run, channel, fit, clip, anchor_method: anchored.append(channel))
    field = sc.Field(str(tmp_path / 'field'), sc.SPHEREx(4), 6.2)
    run_jobs = (jobs.channel(17), jobs.channel(18), jobs.group(19, 20))
    predictions = tmp_path / 'predictions'
    predictions.mkdir()
    for suffix in ('', '_named'):
        # the products' tag: the instrument's, then the primary chunk map's (whose own 'NumCh17' names no channel)
        stems = [f'{field.instrument.product_tag}_NumSub10_NumCh17_NumCol1_{j.name}{suffix}' for j in run_jobs]
        for stem in stems:
            (predictions / f'zodi_pred_{stem}.npz').touch()
        anchored.clear()
        cals = [os.path.join(field.path, 'calibration', f'cal_{stem}.h5') for stem in stems]
        fits_by_job = spherex.zodi_anchor(Result(field, None, run_jobs, cals), predictions)
        assert anchored == [17, 18] and fits_by_job == {'Ch17': os.path.basename(cals[0]),
                                                        'Ch18': os.path.basename(cals[1])}


def test_camera_and_euclid():
    cam = sc.Camera((64, 48), chunks=3, dq_ext=2, tag='Toy')
    assert cam.chunks == (3, 3) and cam.product_tag == 'Toy'
    assert [(k, m.tag) for k, m in cam.default_chunk_maps().items()] == [('grid', 'Chunks3x3')]
    layout = cam.layout()
    assert layout.sci_ext == [1] and layout.dq_ext == [2] and layout.cache_tag == 'headers_Toy'
    cam = sc.Camera((30, 40), chunks=(3, 4), dq_ext=2)
    assert cam.geometry(2).chunk_maps == {}                     # the engine adds the model's maps
    geom = detector_geometry(cam, 2)
    cm = geom.chunk_map
    assert geom.shape == (30, 40) and cm.n_chunks == 12 and cm.grid.shape == (60, 80) and cm.name == 'grid'
    assert cm.axes.names == ['row', 'col'] and cm.adjacency_axes == ('row', 'col')
    assert cm.axes['col'].size == 4 and cm.axes['row'].scan == 'y'
    jg = cam.job_geometry(geom, ALL)
    assert jg.det_valid_weight.shape == (30, 40) and jg.grid_valid_weight.shape == (60, 80)
    assert jg.det_valid_weight.dtype == np.float64 and jg.grid_valid_weight.dtype == np.float32
    assert jg.det_valid_weight.min() == 1 and jg.chunk_valid.all() and jg.chunk_valid_strict.all()
    right = cam.job_geometry(geom, sc.Job('Right', window={'col': range(2, 4)}, strict={'col': range(3, 4)}))
    assert right.chunk_valid.tolist() == [False, False, True, True] * 3
    assert right.chunk_valid_strict.tolist() == [False, False, False, True] * 3
    assert np.array_equal(right.det_valid_weight, (cm.det % 4 >= 2).astype(np.float64))
    assert np.array_equal(right.grid_valid_weight, (cm.grid % 4 == 3).astype(np.float32))
    assert cam.product_tag == 'Grid30x40'
    (spec,) = lower(sc.Field('/tmp/camera_field', cam, 20.0), sc.Recipe(coadd=None), jobs=ALL)
    assert RunContext.build(spec, need_geometry=False).frame_tag == 'Grid30x40_Chunks3x4'
    eu = sc.Euclid(band='J', chunks=510)
    assert eu.product_tag == 'EDFN' and eu.default_ignore_flags() == (11, 15)
    assert list(eu.default_chunk_maps()) == ['grid', 'col_strips', 'row_strips', 'col_tilt', 'row_tilt']


def rectangles(geometry, chunks, axes=('row', 'col')):
    """A chunk map of the user instruments: ``chunks`` = (rows, cols) rectangles over the detector."""
    return sc.Chunks.rectangles(geometry.shape, chunks, axes=axes)


def amplifier_columns(geometry, n=4):
    """A chunk map of the user instruments: one chunk per amplifier, ``n`` vertical bands, along x, with no
    adjacency axes."""
    h, w = geometry.shape
    ids = np.repeat(np.arange(n, dtype=np.int32)[None, :], h, axis=0).repeat(w // n, axis=1)
    return sc.Chunks(ids, sc.ChunkAxes.row_major(('amp',), (n,), ('x',)))


@dataclass(frozen=True, kw_only=True)
class Owl(sc.Instrument):
    """A user-defined instrument: rectangular chunks and an amplifier map."""
    chunks: int = 4
    tag: str = 'Owl'
    unit: str = 'e-/s'

    def geometry(self, oversample):
        return sc.Geometry((DET, DET), oversample)

    def default_chunk_maps(self):
        return {'grid': sc.ChunkMap(rectangles, chunks=(self.chunks, self.chunks)),
                'amps': sc.ChunkMap(rectangles, chunks=(1, 4), axes=('all', 'amp'))}


@dataclass(frozen=True, kw_only=True)
class AmpsCamera(sc.Instrument):
    """A camera with a second chunk map that declares no adjacency axes (four amplifiers)."""
    tag: str = 'Amps'

    def geometry(self, oversample):
        return sc.Geometry((DET, DET), oversample)

    def default_chunk_maps(self):
        return {'grid': sc.ChunkMap(rectangles, chunks=(N_CHUNK_SIDE, N_CHUNK_SIDE)),
                'amps': sc.ChunkMap(amplifier_columns)}


def test_a_new_instrument_is_a_subclass():
    owl = Owl(chunks=2)
    field = sc.Field('/tmp/owl_field', owl, 20.0)
    (spec,) = lower(field, sc.Recipe(coadd=None), jobs=ALL)
    ctx = RunContext.build(spec)
    assert ctx.frame_tag == 'Owl' and [j.name for j in ctx.jobs()] == ['All']
    assert list(ctx.geom.chunk_maps) == ['grid', 'amps'] and ctx.geom.primary == 'grid'
    assert ctx.geom.chunk_map.n_chunks == 4 and ctx.geom.chunk_maps['amps'].name == 'amps'
    assert ctx.geom.chunk_maps['amps'].grid.shape == (DET, DET)            # replicated from the detector map
    # a model's own maps replace the instrument's defaults, the first one primary
    model = sc.Model(chunk_maps={'amps': sc.ChunkMap(amplifier_columns, tag='Amps4'),
                                 'grid': sc.ChunkMap(rectangles, chunks=(2, 2))})
    (spec,) = lower(field, sc.Recipe(model, coadd=None), jobs=ALL)
    ctx = RunContext.build(spec)
    assert ctx.frame_tag == 'Owl_Amps4' and ctx.geom.primary == 'amps' and ctx.geom.chunk_map.n_chunks == 4
    with pytest.raises(ConfigError, match="no chunk axis 'row'"):
        top = sc.Job('Top', window={'row': range(0, 1)})           # an axis of the instrument's map, not the model's
        RunContext.build(lower(field, sc.Recipe(model, coadd=None), jobs=top)[0])


class UndecoratedOwl(sc.Instrument):
    """An instrument without the dataclass decorator."""

    def geometry(self, oversample=1):
        return Owl().geometry(oversample)


@dataclass(kw_only=True)
class MutableOwl(sc.Instrument):
    """An instrument that is a dataclass, but not a frozen one."""
    chunks: int = 4

    def geometry(self, oversample=1):
        return Owl(chunks=self.chunks).geometry(oversample)


def test_an_instrument_is_a_frozen_dataclass():
    for inst in (UndecoratedOwl(), MutableOwl()):
        name = type(inst).__name__
        with pytest.raises(ConfigError, match=rf"Field\(instrument={name}\(\.\.\.\)\): an sc.Instrument subclass is a "
                                              rf"frozen dataclass; put @dataclass\(frozen=True\) above class {name}"):
            sc.Field('/tmp/owl_field', inst, 20.0)


def _render_constant(chunk_map, offsets):
    return np.asarray(offsets)[chunk_map]


def drawn(chunks, jobgeom):
    """A mosaic renderer of a chunk map: the renderer of one job's offsets."""
    return _render_constant


def drawn_rectangles(geometry, chunks):
    """:func:`rectangles` whose offsets the mosaic draws with :func:`drawn` unless a term names another way."""
    return dataclasses.replace(rectangles(geometry, chunks), renderers={'drawn': drawn}, render='drawn')


@dataclass(frozen=True, kw_only=True)
class RenderedOwl(Owl):
    """Owl whose grid map carries a renderer of its own."""

    def default_chunk_maps(self):
        grid = sc.ChunkMap(drawn_rectangles, chunks=(self.chunks, self.chunks))
        return {**super().default_chunk_maps(), 'grid': grid}


def test_an_instruments_optional_hooks_reach_the_engine():
    """A chunk map's renderers reach the mosaic: the map's default one, or the one an offset term names
    (``'constant'``: constant over each chunk); a map without renderers is drawn constant over each chunk."""
    def renderers(model):
        (spec,) = lower(sc.Field('/tmp/owl_field', RenderedOwl(chunks=2), 20.0), sc.Recipe(model, coadd=None),
                        jobs=ALL)
        ctx = RunContext.build(spec)
        return ctx.mosaic_geometry(ctx.job_geometry(ctx.jobs()[0]))[1]

    assert renderers(sc.Model(offsets=[sc.Offsets(), sc.Offsets(on='amps', smooth_along=())])) == [_render_constant,
                                                                                                    None]
    assert renderers(sc.Model(offsets=[sc.Offsets(render='constant')])) == [None]
    with pytest.raises(ConfigError, match="has no renderer 'spline'"):
        renderers(sc.Model(offsets=[sc.Offsets(render='spline')]))
    plain = Owl()
    assert plain.aux_coadds(None) is None and plain.geometry_files() == ()
    assert _NoGeometry().default_chunk_maps() == {}


def test_built_in_instruments_give_their_geometry():
    geom = detector_geometry(sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy'), 2)
    assert geom.chunk_map.n_chunks == N_CHUNK_SIDE ** 2 and geom.chunk_map.grid.shape == (2 * DET, 2 * DET)
    from selfcal.instruments.spherex.spherex_utility import DEFAULT_CALIBRATION_DIR
    if os.path.isdir(os.environ.get('SELFCAL_SPHEREX_CALIB_DIR', DEFAULT_CALIBRATION_DIR)):   # its maps (not on CI)
        from selfcal_scripts.recipes.spherex import maps
        lvf = detector_geometry(sc.SPHEREx(4), 1, sc.Model(chunk_maps=maps(num_ch=34, num_col=3)))
        assert list(lvf.chunk_maps) == ['subchannel', 'detector_column'] and lvf.primary == 'subchannel'
        assert lvf.chunk_map.axes is not None and 'subchannel' in lvf.chunk_map.axes.names
    with pytest.raises(NotImplementedError, match='a subclass of sc.Instrument implements'):
        sc.Instrument.geometry(_NoGeometry())


@dataclass(frozen=True, kw_only=True)
class _NoGeometry(sc.Instrument):
    tag: str = 'None'


def test_map_variables_take_arrays():
    """sc.SkyMap / sc.DetectorMap of an array reach the solve (they took only files and functions)."""
    geom = sc.Camera((8, 8), chunks=2).geometry(1)
    model = sc.Model(variables={'known': sc.SkyMap(np.full((4, 4), 2.0)), 'qe': sc.DetectorMap(np.ones((8, 8)))})
    variables = model.spec().build_variables(geom, [], ref_shape=(4, 4), log=lambda *a, **k: None)
    assert float(np.asarray(variables.sky['known']).mean()) == 2.0
    assert np.asarray(variables.detector['qe']).shape == (8, 8)


# =================================================================== a run through the API
def _same_fits(a, b):
    with fits.open(a) as ha, fits.open(b) as hb:
        assert [h.name for h in ha] == [h.name for h in hb]
        for x, y in zip(ha, hb):
            if x.data is not None:
                assert x.data.tobytes() == y.data.tobytes(), x.name


def _degauge(a):
    """The offsets without the solve's gauge freedom (a smooth sky gradient is degenerate with a
    detector ramp in the offsets plus per-frame scalars): each frame's mean and each chunk's mean
    over the frames removed."""
    a = a - a.mean(axis=1, keepdims=True)
    return a - a.mean(axis=0, keepdims=True)


def _recovered(cal_path, off_true, sc_true=None):
    """The cal's offsets (and scalars: the last term, of one chunk) follow the injected ones (exposure k is
    the cal's frame k)."""
    with CalFile(cal_path) as cal:
        off, scalar = cal.offsets[0], cal.offsets[-1][:, 0]
        assert np.isfinite(cal.sky(0)).any()
    assert off.shape == off_true.shape
    r_off = np.corrcoef(_degauge(off).ravel(), _degauge(off_true).ravel())[0, 1]
    r_sc = 1.0 if sc_true is None else np.corrcoef(scalar, sc_true)[0, 1]
    assert r_off > 0.9 and r_sc > 0.8, (r_off, r_sc)


def test_a_run_through_the_api():
    """Reproject, calibrate with a mosaic (the injected offsets recovered), the mosaic action from the
    existing cal (the same mosaic), and a tiled solve, on the built-in camera."""
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_pyapi_')
    try:
        exp_dir = os.path.join(tmp, 'exposures')
        _, off_true, sc_true = write_exposures(exp_dir, 12, np.random.default_rng(5))
        run_dir = os.path.join(tmp, 'out', 'toy_run')
        cache = os.path.join(tmp, 'cache')
        field = sc.Field(run_dir, sc.Camera((DET, DET), chunks=N_CHUNK_SIDE, dq_ext=2, tag='Toy'), REF_ARCSEC,
                         compute=sc.Compute(cache, workers=2, io_limit=4))
        assert field.reproject(os.path.join(exp_dir, 'toy_exp_*_D0.fits'), method='interp', padding=8) == \
            os.path.join(run_dir, 'reprojected')
        assert len(field.frames) == 12 and os.path.exists(os.path.join(run_dir, 'ref.fits'))
        # the sky damped (0.1): the toy's offsets are recovered in the 60 iterations given
        recipe = sc.Recipe(sc.continuum(smooth=0.1, damping=0.1), fit=sc.Fit(60, tolerance=1e-8), coadd=sc.Coadd(clip=3.0),
                           numerics=sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4), name='t')
        plan = field.plan(recipe, jobs=ALL)
        assert 'cal_Toy_Chunks4x4_All_t.h5' in str(plan) and plan.frames[0] == 12
        result = field.calibrate(recipe, jobs=ALL)
        assert os.path.exists(result.record)
        assert result.offsets().shape == (12, N_CHUNK_SIDE ** 2) and result.sky().ndim == 2
        _recovered(result.cal_paths[0], off_true, sc_true)
        with fits.open(result.mosaic_paths[0]) as h:
            names = [x.name for x in h]
        assert 'SC_MEAN_MAP' in names and not any('WAV' in n for n in names)
        assert not os.path.exists(os.path.join(cache, 'reproj_nvme_toy_run'))      # the staged frames are gone

        made = os.path.join(tmp, 'mosaic_made.fits')                 # the mosaic action, from the existing cal
        shutil.move(result.mosaic_paths[0], made)
        again = field.mosaic(recipe, jobs=ALL)
        _same_fits(again.mosaic_paths[0], made)

        tiles = sc.Tiles((1, 2), overlap=10, names=('W', 'E'))
        t = field.calibrate(recipe.replace(coadd=None), tiles=tiles,
                            compute=field.compute.replace(stage_dir='toy_tiles', memory_guard=False), jobs=ALL)
        assert sorted(t.tile_cals) == ['E', 'W']
        assert os.path.basename(t.final) == 'cal_Toy_Chunks4x4_All_t_stitched.h5'
        assert [os.path.basename(p) for p in sorted(t.tile_cals.values())] == \
            ['cal_Toy_Chunks4x4_All_t_E.h5', 'cal_Toy_Chunks4x4_All_t_W.h5']
        with CalFile(t.final) as cal:
            assert cal.ref_shape == fits.getdata(os.path.join(run_dir, 'ref.fits')).shape
            assert np.isfinite(cal.sky(0)).sum() > 0
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_a_nonsquare_camera_without_a_mask_runs(tmp_path):
    """A 48 x 80 imager whose exposures have no data-quality extension."""
    _state.set_progress(False)
    shape, chunks, n = (48, 80), (3, 5), 12
    paths, off_true, _ = write_exposures(str(tmp_path / 'exposures'), n, np.random.default_rng(11),
                                               det_shape=shape, chunks=chunks, with_dq=False)
    with fits.open(paths[0]) as h:
        assert len(h) == 2                                        # primary + science: no DQ HDU
    field = sc.Field(tmp_path / 'out' / 'wide', sc.Camera(shape, chunks=chunks, tag='Wide'), REF_ARCSEC,
                     compute=sc.Compute(str(tmp_path / 'cache'), workers=2))
    reprojected = field.reproject(str(tmp_path / 'exposures' / 'toy_exp_*_D0.fits'), method='interp', padding=8)
    with h5py.File(field.frames[0], 'r') as f:
        assert int(f['sub_bitmask'][()].max()) == 0               # no mask: nothing flagged
    assert len(field.frames) == n and reprojected == os.path.join(field.path, 'reprojected')
    result = field.calibrate(sc.Recipe(sc.continuum(damping=0.1), fit=sc.Fit(60, tolerance=1e-8), coadd=sc.Coadd(clip=3.0),
                                       numerics=sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4)), jobs=ALL)
    assert len(result.cal_paths) == 1 and len(result.mosaic_paths) == 1
    with CalFile(result.cal_paths[0]) as cal:
        assert cal.schema_version == 3 and cal.sky_names == ['continuum'] and cal.num_maps == 2   # + the scalar term
        assert cal.n_frames == n and cal.offsets[0].shape == (n, chunks[0] * chunks[1])
        assert cal.chunk_maps[0].shape == shape and cal.sky(0).shape == cal.ref_shape
    _recovered(result.cal_paths[0], off_true)


def test_frame_context_reads_as_a_mapping():
    from selfcal.core.subframe import FrameContext
    ctx = FrameContext(stage='post', file='/x/exp_0003_det_00.h5', exp_idx=3, det_idx=0,
                       ref_coords=np.array([0, 10, 0, 10]), sub_data=np.ones((2, 2)), sub_weight=np.ones((2, 2)),
                       sub_mapping=np.zeros((2, 2, 2)), sub_aux=None)
    assert ctx['sub_data'] is ctx.sub_data and ctx.get('sub_aux') is None and ctx.get('nope', 1) == 1


# =================================================================== the engine's resolution of a run
@dataclass(frozen=True, kw_only=True)
class CountedCamera(sc.Instrument):
    """The toy camera, counting the geometries it builds; ``files`` are the files its geometry reads,
    and nothing else: the engine may keep its geometry between actions."""
    files: tuple[str, ...] = ()
    tag: str = 'Counted'

    geometry_is_pure = True

    def geometry(self, oversample=1):
        GEOMETRIES.append(oversample)
        return sc.Camera((DET, DET), chunks=N_CHUNK_SIDE).geometry(oversample)

    def default_chunk_maps(self):
        return sc.Camera((DET, DET), chunks=N_CHUNK_SIDE).default_chunk_maps()

    def layout(self):
        return sc.Camera((DET, DET), dq_ext=2).layout()

    def geometry_files(self):
        return self.files


GEOMETRIES = []


def test_the_geometry_is_kept_and_copied(tmp_path):
    from selfcal.run.engine import GEOMETRIES_KEPT, instrument_geometry
    marker = tmp_path / 'geometry_input'
    marker.write_text('a')
    inst = CountedCamera(files=(str(marker),))
    specs = map_specs(inst)
    GEOMETRIES.clear()
    a = instrument_geometry(inst, 1, specs)
    a.chunk_map.det[0, 0] = 99                       # a run that writes into its geometry changes no other's
    b = instrument_geometry(inst, 1, specs)
    assert GEOMETRIES == [1] and b.chunk_map.det[0, 0] == 0
    instrument_geometry(inst, 2, specs)
    assert GEOMETRIES == [1, 2]
    os.utime(marker, ns=(0, 0))                      # a file the geometry reads changed: built again
    instrument_geometry(inst, 1, specs)
    assert GEOMETRIES == [1, 2, 1]
    assert GEOMETRIES_KEPT == 2
    instrument_geometry(inst, 3, specs)              # the last two are kept: 1 and 3; 2 is built again
    instrument_geometry(inst, 2, specs)
    assert GEOMETRIES == [1, 2, 1, 3, 2]
    other = map_specs(inst, sc.Model(chunk_maps={'grid': sc.ChunkMap(rectangles, chunks=(2, 2))}))
    assert instrument_geometry(inst, 2, other).chunk_map.n_chunks == 4     # other chunk maps: built again
    assert GEOMETRIES == [1, 2, 1, 3, 2, 2]


def test_the_geometry_is_built_once_for_a_plan_and_its_action(tmp_path):
    _state.set_progress(False)
    write_exposures(str(tmp_path / 'exposures'), 6, np.random.default_rng(13))
    field = sc.Field(tmp_path / 'out' / 'counted', CountedCamera(), REF_ARCSEC,
                     compute=sc.Compute(str(tmp_path / 'cache'), workers=2))
    field.reproject(str(tmp_path / 'exposures' / 'toy_exp_*_D0.fits'), method='interp', padding=8)
    recipe = sc.Recipe(fit=sc.Fit(10), coadd=sc.Coadd(clip=3.0), numerics=sc.Numerics(2, batch=4, mosaic_batch=4,
                                                                                     coadd_batch=4))
    GEOMETRIES.clear()
    field.plan(recipe, jobs=ALL)
    field.calibrate(recipe, jobs=ALL)
    assert GEOMETRIES == [1]


#: The chunks per side of :class:`ModuleStateCamera`: module state its geometry reads.
MODULE_CHUNKS = {'side': 2}


@dataclass(frozen=True, kw_only=True)
class ModuleStateCamera(sc.Instrument):
    """A user instrument whose geometry reads module state (and holds a lock, which cannot be copied):
    it does not declare its geometry pure."""
    tag: str = 'ModuleState'

    def geometry(self, oversample=1):
        GEOMETRIES.append(oversample)
        geom = sc.Geometry((DET, DET), oversample)
        geom.extra['lock'] = threading.Lock()
        return geom

    def default_chunk_maps(self):
        return {'grid': sc.ChunkMap(module_state_rectangles)}


def module_state_rectangles(geometry):
    """``MODULE_CHUNKS['side']`` x ``MODULE_CHUNKS['side']`` rectangles: a chunk map that reads module state."""
    side = MODULE_CHUNKS['side']
    return sc.Chunks.rectangles(geometry.shape, (side, side))

    def layout(self):
        return sc.Camera((DET, DET), dq_ext=2).layout()


def test_a_user_geometry_is_built_by_each_action_and_not_kept(tmp_path):
    """An instrument that does not declare its geometry pure has it built once per action (the plan and
    the run share it) and never kept: module state its geometry reads that changed between two actions
    takes effect, and the geometry is not copied."""
    _state.set_progress(False)
    write_exposures(str(tmp_path / 'exposures'), 6, np.random.default_rng(19))
    field = sc.Field(tmp_path / 'out' / 'module_state', ModuleStateCamera(), REF_ARCSEC,
                     compute=sc.Compute(str(tmp_path / 'cache'), workers=2))
    field.reproject(str(tmp_path / 'exposures' / 'toy_exp_*_D0.fits'), method='interp', padding=8)
    recipe = sc.Recipe(fit=sc.Fit(10), coadd=None, numerics=sc.Numerics(2, batch=4))
    GEOMETRIES.clear()
    MODULE_CHUNKS['side'] = 2
    try:
        assert field.plan(recipe, jobs=ALL).contexts[0].geom.chunk_map.n_chunks == 4
        MODULE_CHUNKS['side'] = 4
        result = field.calibrate(recipe, jobs=ALL)
    finally:
        MODULE_CHUNKS['side'] = 2
    assert GEOMETRIES == [1, 1]
    with CalFile(result.cal_paths[0]) as cal:
        assert cal.offsets[0].shape == (6, 16)


def test_every_pass_damps_the_sky_as_the_model_says():
    model = sc.Model(sky=[sc.Sky(), sc.Sky('ramp', times=x_ramp)])
    (spec,) = lower(sc.Field('/tmp/damping_field', sc.Camera((DET, DET)), 20.0), sc.Recipe(model, coadd=None),
                    jobs=ALL)
    ctx = RunContext.build(spec)
    assert ctx.sky_damping == model.sky_dampings() == [0.0, 0.0]          # Sky() leaves the sky undamped
    # the one rule (SkyModel.damp_weights): a term's own, else damp_weight for the first, damp_weight_line for the others
    from dataclasses import replace

    from selfcal.models.sky_model import SkyModel
    plain = SkyModel(tuple(replace(c, damp_weight=None) for c in ctx.sky_model().components))
    assert plain.damp_weights(0.1) == [0.1, 0.0] and plain.damp_weights(0.1, 0.02) == [0.1, 0.02]


@dataclass(frozen=True, kw_only=True)
class AmpCamera(sc.Instrument):
    """The toy camera with a second chunk map, one chunk per amplifier column."""
    tag: str = 'AmpCam'

    def geometry(self, oversample=1):
        return sc.Geometry((DET, DET), oversample)

    def default_chunk_maps(self):
        return {'grid': sc.ChunkMap(rectangles, chunks=(N_CHUNK_SIDE, N_CHUNK_SIDE)),
                'amps': sc.ChunkMap(rectangles, chunks=(1, 4), axes=('all', 'amp'))}


def test_a_model_grouped_by_its_own_frame_variable_is_mosaicked(tmp_path):
    """An offset term shared over the frames with equal values of a frame variable the MODEL defines
    (``sc.PerFrame``), with a term the mosaic subtracts per observation (a basis): the mosaic builds
    the offset model as the solve does."""
    _state.set_progress(False)
    write_exposures(str(tmp_path / 'exposures'), 16, np.random.default_rng(17))
    field = sc.Field(tmp_path / 'out' / 'amps', AmpCamera(), REF_ARCSEC,
                     compute=sc.Compute(str(tmp_path / 'cache'), workers=2))
    field.reproject(str(tmp_path / 'exposures' / 'toy_exp_*_D0.fits'), method='interp', padding=8)
    model = sc.Model(offsets=[sc.Offsets('chunks', smooth=0.05, exclude=sc.Poly(0)),
                              sc.Offsets('amps', on='amps', per='fortnight', smooth_along=()),
                              sc.Offsets('gradient', on='detector', basis=plane, n=2), sc.Offsets('scalar', on='detector')],
                     variables={'fortnight': sc.PerFrame(fortnight, of='exposure', length=8)})
    result = field.calibrate(sc.Recipe(model, fit=sc.Fit(30), coadd=sc.Coadd(clip=3.0),
                                       numerics=sc.Numerics(2, batch=4, mosaic_batch=4, coadd_batch=4)), jobs=ALL)
    with result.mosaic() as mosaic:
        assert np.isfinite(mosaic.mean).any()


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
                                 "field.plan(jobs=sc.Job('All'))\n")
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
    from selfcal.run.engine import clip_groups
    field = sc.Field('/tmp/clip_field', sc.Camera((DET, DET), chunks=N_CHUNK_SIDE), 20.0)
    (spec,) = lower(field, sc.Recipe(coadd=None), jobs=ALL)
    ctx = RunContext.build(spec)
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
            cals[name] = field.calibrate(recipe, jobs=ALL).cal_paths[0]
        with h5py.File(cals['frame']) as a, h5py.File(cals['chunk']) as b, h5py.File(cals['rows']) as c:
            sky = [f['sky/continuum'][()] for f in (a, b, c)]
        assert all(np.isfinite(s).any() for s in sky)
        assert not np.array_equal(sky[0], sky[1], equal_nan=True)     # a different clip, a different solve
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
