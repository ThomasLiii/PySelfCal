"""N>2 sky-component round-trip test (no SPHEREx data, no full solve).

The cal gates only exercise <=2 sky blocks, so this is the verification for the
N>2 path: a 3-component SkyModel (continuum + Gaussian line + numerical
template) driven through Calibrator.save_calibration (cal-schema v3) ->
load_calibration -> get_sky. Checks the v3 layout, the hard-link back-compat
aliases, and that all three sky maps round-trip exactly.

Runnable as ``python tests/test_nsky_roundtrip.py`` or under pytest.
"""
import os
import tempfile

import h5py
import numpy as np

from selfcal.pipeline.pipeline_wrapper import Calibrator
from selfcal.models.sky_model import SkyModel, ContinuumComponent, LineComponent
from selfcal.models.profiles import GaussianProfile, TemplateProfile


def _make_stub(x, ref_shape, num_frames, num_chunks, sky_model, pixel_counts, pixel_fisher):
    """A bare Calibrator with just the attributes save/load/get_sky use."""
    cc = Calibrator.__new__(Calibrator)
    cc.x = x
    cc.ref_shape = ref_shape
    cc.reproj_list = [f'/hdd/exp_{i:04d}_det_00.h5' for i in range(num_frames)]
    cc.chunk_maps = [np.tile(np.arange(num_chunks, dtype=np.int32), (2, 1))]  # max = num_chunks-1
    cc.num_offset_groups_list = [num_frames]   # per-frame
    cc.num_chunks_list = [num_chunks]
    cc.num_sky_blocks = sky_model.n_blocks
    cc.pixel_counts = pixel_counts
    cc.pixel_fisher = pixel_fisher
    cc.det_templates = [None]
    cc.frame_to_groups = [np.arange(num_frames)]
    cc.num_scalar_cols = num_frames
    cc.sky_model = sky_model
    cc.line_fisher_threshold = None
    return cc


def test_nsky3_save_load_roundtrip():
    ref_shape = (6, 7)
    num_sky = ref_shape[0] * ref_shape[1]
    num_frames, num_chunks = 4, 3
    J = 3
    sky_model = SkyModel((
        ContinuumComponent(),
        LineComponent(name='pah_3p29', profile=GaussianProfile(3.29, 0.04)),
        LineComponent(name='aliphatic', profile=TemplateProfile(
            np.linspace(3.3, 3.5, 6), np.ones(6))),
    ))
    rng = np.random.default_rng(7)
    sky_vals = rng.standard_normal(J * num_sky)
    off_vals = rng.standard_normal(num_frames * num_chunks)
    sc_vals = rng.standard_normal(num_frames)
    x = np.concatenate([sky_vals, off_vals, sc_vals]).astype(np.float64)
    pixel_counts = (rng.random(len(x)) * 12).astype(np.int64)
    pixel_fisher = rng.random(len(x)).astype(np.float64)

    cc = _make_stub(x, ref_shape, num_frames, num_chunks, sky_model, pixel_counts, pixel_fisher)
    tmp = tempfile.mkdtemp()
    path = cc.save_calibration(cal_dir=tmp, cal_file='nsky3.h5')

    # v3 structure: three named sky blocks + back-compat hard links.
    with h5py.File(path, 'r') as f:
        assert int(f.attrs['schema_version']) == 3
        names = [n.decode() for n in f.attrs['sky_components']]
        assert names == ['continuum', 'pah_3p29', 'aliphatic']
        assert set(f['sky'].keys()) == set(names)
        # hard-link aliases: skymap -> continuum (values equal)
        assert np.array_equal(f['skymap'][...], f['sky']['continuum'][...])
        assert np.array_equal(f['skymap_coverage'][...], f['sky_coverage']['continuum'][...])
        # With >1 spectral block, skymap_line aliases the LAST block (the
        # primary line; earlier extras are nuisance shapes like a continuum
        # slope) — matching save_calibration and the tiled stitch convention.
        assert np.array_equal(f['skymap_line'][...], f['sky']['aliphatic'][...])
        # each block has coverage + fisher
        for nm in names:
            assert nm in f['sky_coverage'] and nm in f['sky_fisher']

    # load round-trip via a fresh stub (load_calibration sets sky_component_names).
    cc2 = Calibrator.__new__(Calibrator)
    cc2.ref_shape = ref_shape
    cc2.reproj_list = cc.reproj_list
    cc2.load_calibration(cal_path=path)
    assert cc2.num_sky_blocks == 3
    assert cc2.sky_component_names == ['continuum', 'pah_3p29', 'aliphatic']

    # get_sky returns each component's map, matching the injected sky values.
    for j, nm in enumerate(['continuum', 'pah_3p29', 'aliphatic']):
        expected = sky_vals[j * num_sky:(j + 1) * num_sky].reshape(ref_shape)
        np.testing.assert_allclose(cc2.get_sky(nm), expected, rtol=0, atol=0)
    # get_skymap() is the continuum; get_skymap_line() is the first line (back-compat).
    np.testing.assert_allclose(cc2.get_skymap(), sky_vals[:num_sky].reshape(ref_shape))
    np.testing.assert_allclose(cc2.get_skymap_line(),
                               sky_vals[num_sky:2 * num_sky].reshape(ref_shape))


def test_single_line_keeps_legacy_skymap_line_alias():
    """J==2 must still write the legacy skymap_line hard-link alias (back-compat)."""
    ref_shape = (5, 5)
    num_sky = 25
    num_frames, num_chunks = 3, 2
    sky_model = SkyModel((ContinuumComponent(),
                          LineComponent(name='pah_3p29', profile=GaussianProfile(3.29, 0.04))))
    rng = np.random.default_rng(1)
    x = rng.standard_normal(2 * num_sky + num_frames * num_chunks + num_frames)
    pc = (rng.random(len(x)) * 5).astype(np.int64)
    cc = _make_stub(x, ref_shape, num_frames, num_chunks, sky_model, pc, rng.random(len(x)))
    tmp = tempfile.mkdtemp()
    path = cc.save_calibration(cal_dir=tmp, cal_file='nsky2.h5')
    with h5py.File(path, 'r') as f:
        assert 'skymap_line' in f  # single line -> alias present
        assert np.array_equal(f['skymap_line'][...], f['sky']['pah_3p29'][...])


def _run_all():
    fns = [v for k, v in sorted(globals().items())
           if k.startswith('test_') and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\nALL {len(fns)} TESTS PASSED")


if __name__ == '__main__':
    _run_all()
