"""The SPHEREx ``precompute`` task writes one LVF arc fit per detector.

Regression: the adapter unpacked two of the three values ``make_fiducial_chunk_map``
returns, so the task failed before writing anything. The calibration files are
replaced by a blank band-centre map, the SPHEREx channel table by the channel edges
of the shipped Detector 1 fit (every 20th of its 341 subchannel edges), and the arc
fit by that fit (``make_fiducial_chunk_map`` itself runs, so its real return value
is unpacked). Nothing reads files that exist only on the processing host.
Runnable as a script or under pytest.
"""
import os
import tempfile

import numpy as np

import selfcal.instruments.spherex.adapter as adapter
import selfcal.instruments.spherex.spherex_utility as su
from selfcal.instruments import get_instrument


def _run(out_dir, monkeypatch=None):
    shipped = su.load_lvf_params('lvf_params_D1.npy')
    real = su.make_fiducial_chunk_map
    calls = []

    def with_shipped_fit(band, BC_map, **kw):
        calls.append(band)
        return real(band, BC_map, lvf_params=dict(shipped), **kw)

    def blank_calibration(band, calibration_dir=None):
        return np.zeros((2040, 2040), dtype=np.float32), None

    edges = np.asarray(shipped['wave_edges'])[::20]

    def shipped_channel_edges(band, channel_file=None):
        return edges.copy()

    patches = [(su, 'make_fiducial_chunk_map', with_shipped_fit), (adapter, 'load_calibration', blank_calibration),
               (su, 'extract_spherex_channel_edges', shipped_channel_edges)]
    saved = [(mod, name, getattr(mod, name)) for mod, name, _ in patches]
    for mod, name, value in patches:
        if monkeypatch is not None:
            monkeypatch.setattr(mod, name, value)
        else:
            setattr(mod, name, value)
    try:
        get_instrument('spherex').precompute({'detectors': [1], 'lvf_output_dir': out_dir})
    finally:
        if monkeypatch is None:
            for mod, name, value in saved:
                setattr(mod, name, value)
    return shipped, calls


def _check(out_dir, shipped, calls):
    assert calls == [1]
    assert os.path.exists(os.path.join(out_dir, 'lvf_params_D1.npy'))
    got = su.load_lvf_params('lvf_params_D1.npy', input_dir=out_dir)
    assert got['filename'] == 'lvf_params_D1.npy'
    for key in ('xc', 'yc', 'R'):
        assert np.array_equal(np.asarray(got[key]), np.asarray(shipped[key])), key


def test_precompute_writes_lvf_params(tmp_path, monkeypatch):
    shipped, calls = _run(str(tmp_path), monkeypatch)
    _check(str(tmp_path), shipped, calls)


if __name__ == '__main__':
    with tempfile.TemporaryDirectory() as d:
        _check(d, *_run(d))
    print('OK spherex precompute')
