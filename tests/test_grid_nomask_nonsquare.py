"""A non-square imager without a data-quality extension, through the run engine on the
built-in ``grid`` instrument: reproject (no mask, 48 x 80 detector) -> cal -> mosaic;
then the CalFile reader on the product and the typed hook context."""
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

from selfcal import _state                                      # noqa: E402
from selfcal.io.calfile import CalFile                          # noqa: E402
from selfcal.run import pipelines                    # noqa: E402
from tests.synthetic_exposures import write_exposures           # noqa: E402
from tests.test_runner_e2e_toy import _write_config, _degauge   # noqa: E402

N_EXP = 12
SHAPE = (48, 80)
CHUNKS = (3, 5)


def test_grid_nonsquare_without_mask():
    _state.set_progress(False)
    tmp = tempfile.mkdtemp(prefix='selfcal_grid_')
    try:
        rng = np.random.default_rng(11)
        exp_dir = os.path.join(tmp, 'exposures')
        paths, off_true, sc_true = write_exposures(exp_dir, N_EXP, rng, det_shape=SHAPE, chunks=CHUNKS,
                                                   with_dq=False)
        from astropy.io import fits
        with fits.open(paths[0]) as h:
            assert len(h) == 2                                        # primary + science: no DQ HDU
        out, cache = os.path.join(tmp, 'out'), os.path.join(tmp, 'cache')
        os.makedirs(cache, exist_ok=True)
        inst = {'name': 'grid', 'tag': 'Wide', 'detector_shape': list(SHAPE), 'chunks': list(CHUNKS),
                'sci_ext': 1, 'dq_ext': -1}
        rcfg = _write_config(os.path.join(tmp, 'reproject.toml'), 'reproject', out, cache, instrument=inst,
                             reproject=dict(input_dirs=[exp_dir], file_pattern='/toy_exp_*_D0.fits',
                                            padding_pixels=8, max_workers=2, inner_parallel=1,
                                            reproj_func='interp', padding_percentage=0.05, replace_existing=True))
        reproj_dir = pipelines.run(rcfg)
        frames = sorted(f for f in os.listdir(reproj_dir) if f.endswith('.h5'))
        assert len(frames) == N_EXP
        with h5py.File(os.path.join(reproj_dir, frames[0]), 'r') as f:
            assert int(f['sub_bitmask'][()].max()) == 0            # no mask -> nothing flagged
            assert f['sub_mapping'].shape[0] == 2
        ccfg = _write_config(os.path.join(tmp, 'cal.toml'), 'cal', out, cache, instrument=inst)
        res = pipelines.run(ccfg)
        assert len(res.cal_paths) == 1 and len(res.mosaic_paths) == 1
        with CalFile(res.cal_paths[0]) as cal:
            assert cal.schema_version == 3 and cal.sky_names == ['continuum'] and cal.num_maps == 1
            assert cal.n_frames == N_EXP and len(cal.reproj_list) == N_EXP
            off = cal.offsets[0]
            assert off.shape == (N_EXP, CHUNKS[0] * CHUNKS[1])
            assert cal.frame_scalar.shape == (N_EXP,)
            assert cal.total_offsets()[0].shape == off.shape
            assert cal.chunk_maps[0].shape == SHAPE
            assert cal.sky(0).shape == cal.ref_shape and cal.sky_coverage('continuum') is not None
            assert 'schema v3' in cal.describe()
        r_off = np.corrcoef(_degauge(off).ravel(), _degauge(off_true).ravel())[0, 1]
        assert r_off > 0.9, r_off
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_frame_context_hooks():
    from selfcal.core.subframe import FrameContext
    ctx = FrameContext(stage='post', file='/x/exp_0003_det_00.h5', exp_idx=3, det_idx=0,
                       ref_coords=np.array([0, 10, 0, 10]), sub_data=np.ones((2, 2)), sub_weight=np.ones((2, 2)),
                       sub_mapping=np.zeros((2, 2, 2)), sub_aux=None)
    assert ctx['sub_data'] is ctx.sub_data and ctx.get('sub_aux') is None and ctx.get('nope', 1) == 1
    from selfcal.run.postprocess import mask_bright_pixels
    ctx.sub_data = np.array([[1.0, 2.0], [3.0, 4.0]])
    out = mask_bright_pixels(ctx)
    assert np.isnan(out).sum() == 3 and out[0, 0] == 1.0


if __name__ == '__main__':
    test_grid_nonsquare_without_mask()
    test_frame_context_hooks()
    print('OK grid non-square without mask + CalFile + FrameContext')
