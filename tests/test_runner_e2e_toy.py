"""End-to-end through the RUN ENGINE with a telescope-free instrument.

Synthetic FITS exposures -> ``reproject`` task -> ``cal`` task (continuum mode, mosaic
with std + sigma-clip) -> ``mosaic`` task -> ``cal`` with ``[tiling]`` (two tiles +
Fisher stitch), all via ``selfcal_scripts.runner.pipelines.run`` on configs written
as TOML and loaded by ``load_config``. Proves the engine runs without SPHEREx and
that the recovered per-frame offsets track the injected ones.
Runnable as ``python tests/test_runner_e2e_toy.py`` or under pytest (~10 s).
"""
import os
import shutil
import sys
import tempfile

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import h5py                                                     # noqa: E402
from astropy.io import fits                                     # noqa: E402

from selfcal import _state                                      # noqa: E402
from selfcal_scripts.runner import config as runner_config      # noqa: E402
from selfcal_scripts.runner import engine, pipelines            # noqa: E402
from tests.toy_instrument import ToyInstrument, write_exposures, REF_ARCSEC, N_CHUNK_SIDE  # noqa: E402

N_EXP = 14


def _toml_value(v):
    if isinstance(v, bool):
        return 'true' if v else 'false'
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return '"' + v.replace('\\', '\\\\').replace('"', '\\"') + '"'
    if isinstance(v, (list, tuple)):
        return '[' + ', '.join(_toml_value(x) for x in v) + ']'
    if isinstance(v, dict):
        return '{ ' + ', '.join(f'{k} = {_toml_value(x)}' for k, x in v.items()) + ' }'
    raise TypeError(type(v))


def _write_config(path, task, out, cache, scalars=None, **tables):
    """Write a runner TOML exactly as a user would (the test goes through ``load_config``)."""
    top = dict(task=task, output_dir=out, run_name='toy_run', resolution_arcsec=REF_ARCSEC, cache_dir=cache + '/',
               suffix='_t', oversample=1, staging='copy', keep_nvme=False, hdd_io_limit=4, apply_n_threads=2,
               wavelength_coadd=False)
    if task in ('cal', 'tiled', 'mosaic'):
        top['mode'] = 'continuum'
    top.update(scalars or {})
    base = dict(instrument={'name': 'toy', 'detector': 0, 'num_col': N_CHUNK_SIDE},
                params={'reg_weight': 0.1},
                calibration=dict(apply_mask=True, apply_weight=False, outlier_thresh=5.0, ignore_list=[],
                                 batch_size=4, offset_regularization=True, weighted_damping=True,
                                 damp_weight=0.1, max_workers=2),
                lsqr=dict(atol=1e-8, btol=1e-8, damp=0, iter_lim=60, precondition=True, solver='lsqr'),
                mosaic=dict(apply_mask=True, apply_weight=False, make_std_map=True, apply_sigma_clipping=True,
                            sigma=3.0, ignore_list=[], cache_batch_size=4, coadd_batch_size=4,
                            cache_intermediate=True, max_workers=2))
    base.update(tables)
    lines = [f'{k} = {_toml_value(v)}' for k, v in top.items()]
    for t, d in base.items():
        lines += ['', f'[{t}]'] + [f'{k} = {_toml_value(v)}' for k, v in d.items()]
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    return runner_config.load_config(path)


def _degauge(a):
    """Project out the joint solve's gauge freedom: a smooth sky gradient is degenerate
    with a fixed detector ramp in the offsets plus per-frame scalars (that is what the
    polynomial constraints are for), so compare offsets without their per-frame mean and
    their per-chunk mean across frames."""
    a = a - a.mean(axis=1, keepdims=True)
    return a - a.mean(axis=0, keepdims=True)


def _check_cal(cal_path, off_true, sc_true, n_exp):
    with h5py.File(cal_path, 'r') as f:
        off = f['offsets/map_0'][()]
        sc = f['frame_scalar'][()]
        sky = f['sky/continuum'][()]
        assert off.shape == (n_exp, N_CHUNK_SIDE ** 2)
    r_off = np.corrcoef(_degauge(off).ravel(), _degauge(off_true).ravel())[0, 1]
    r_sc = np.corrcoef(sc, sc_true)[0, 1]
    assert r_off > 0.9, r_off
    assert r_sc > 0.8, r_sc
    assert np.isfinite(sky[np.isfinite(sky)]).any()


def test_toy_instrument_end_to_end():
    _state.set_progress(False)
    orig_get = runner_config.get_instrument
    runner_config.get_instrument = lambda name: ToyInstrument() if name == 'toy' else orig_get(name)
    engine.get_instrument = runner_config.get_instrument
    tmp = tempfile.mkdtemp(prefix='selfcal_toy_')
    try:
        rng = np.random.default_rng(3)
        exp_dir = os.path.join(tmp, 'exposures')
        paths, off_true, sc_true = write_exposures(exp_dir, N_EXP, rng)
        out, cache = os.path.join(tmp, 'out'), os.path.join(tmp, 'cache')
        os.makedirs(cache, exist_ok=True)
        # --- reproject task
        rcfg = _write_config(os.path.join(tmp, 'reproject.toml'), 'reproject', out, cache, reproject=dict(
            input_dirs=[exp_dir], file_pattern='/toy_exp_*_D{detector}.fits', use_ext=[1], sci_ext_list=[1],
            dq_ext_list=[2], padding_pixels=8, max_workers=2, inner_parallel=1, reproj_func='interp',
            padding_percentage=0.05, replace_existing=True, header_filter_workers=2))
        reproj_dir = pipelines.run(rcfg)
        assert reproj_dir == os.path.join(out, 'toy_run', 'reprojected')
        frames = sorted(f for f in os.listdir(reproj_dir) if f.endswith('.h5'))
        assert len(frames) == N_EXP, frames
        ref = os.path.join(out, 'toy_run', 'ref.fits')
        assert os.path.exists(ref)
        # --- cal task (+ mosaic)
        ccfg = _write_config(os.path.join(tmp, 'cal.toml'), 'cal', out, cache)
        assert ccfg.instrument == 'toy'
        res = pipelines.run(ccfg)
        assert len(res.cal_paths) == 1 and os.path.exists(res.cal_paths[0])
        assert res.sky_path == res.cal_paths[0]
        _check_cal(res.cal_paths[0], off_true, sc_true, N_EXP)
        tag = ToyInstrument().frame_tag(ccfg.instrument_cfg)
        mos = os.path.join(out, 'toy_run', 'mosaic', f"mosaic_{tag}_All_t.fits")
        assert res.mosaic_paths == [mos] and os.path.exists(mos), os.listdir(os.path.join(out, 'toy_run', 'mosaic'))
        with fits.open(mos) as h:
            names = [x.name for x in h]
            assert len(names) >= 5 and not any('WAV' in n for n in names), names   # mean/std/sc maps, no wavelength maps
            mean = h[1].data
            assert mean.ndim == 2 and np.isfinite(mean).sum() > 0.5 * mean.size, names
        assert not os.path.exists(os.path.join(cache, 'reproj_nvme_toy_run')), "NVMe staging dir should be cleaned up"
        # --- mosaic task: re-make the mosaic of the existing cal
        os.remove(mos)
        mcfg = _write_config(os.path.join(tmp, 'mosaic.toml'), 'mosaic', out, cache)
        mres = pipelines.run(mcfg)
        assert mres.cal_paths == res.cal_paths and mres.mosaic_paths == [mos] and os.path.exists(mos)
        # --- cal with [tiling]: two tiles (west/east halves) + Fisher stitch; 'tiled' is an alias task
        ny, nx = fits.getheader(ref)['NAXIS2'], fits.getheader(ref)['NAXIS1']
        tcfg = _write_config(os.path.join(tmp, 'tiled.toml'), 'tiled', out, cache, scalars={'suffix': '_tile_{tile}'},
                             tiling=dict(grid=[1, 2], overlap_px=0, tile_names=['W', 'E'], ref_shape=[ny, nx],
                                         full_reproj_dir=reproj_dir, frame_glob='exp_*_det_00.h5',
                                         frame_filter='center', halo=0, nvme_subdir='tiles_nvme',
                                         stitched_suffix='_tile_stitched', line=False, rss_guardrail=False))
        assert tcfg.task == 'cal' and tcfg.tiling and tcfg.tiled is tcfg.tiling
        tres = pipelines.run(tcfg)
        assert sorted(tres.tiles) == ['E', 'W'] and tres.stitched == tres.sky_path
        for p in list(tres.tiles.values()) + [tres.stitched]:
            assert os.path.exists(p), p
        with h5py.File(tres.stitched, 'r') as f:
            assert f['skymap'].shape == (ny, nx)
            assert np.isfinite(f['skymap'][()]).sum() > 0
    finally:
        runner_config.get_instrument = orig_get
        engine.get_instrument = orig_get
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == '__main__':
    test_toy_instrument_end_to_end()
    print("OK toy instrument: reproject -> cal -> mosaic -> mosaic task -> tiled cal through the run engine")
