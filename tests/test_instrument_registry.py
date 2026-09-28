"""The instrument and mode registries: built-ins resolve, a locally registered
instrument resolves, presets are aliases of the structural recipes."""
import os
import sys

import numpy as np

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from selfcal.instruments import (get_instrument, available_instruments, register_instrument,   # noqa: E402
                                 Instrument, Job, DetectorGeometry, JobGeometry, ExposureLayout)
from selfcal.instruments.grid import GridInstrument                                          # noqa: E402
from selfcal_scripts.runner.modes import get_mode                                            # noqa: E402


def test_builtin_instruments():
    assert {'grid', 'spherex'} <= set(available_instruments())
    assert isinstance(get_instrument('grid'), GridInstrument)
    assert get_instrument('spherex').name == 'spherex'


def test_grid_geometry_from_config():
    cfg = {'name': 'grid', 'detector_shape': [30, 40], 'chunks': [3, 4], 'sci_ext': 1, 'dq_ext': 2}
    inst = get_instrument('grid')
    geom = inst.detector_geometry(cfg, 2)
    cm = geom.chunk_map
    assert geom.shape == (30, 40) and cm.n_chunks == 12 and cm.grid.shape == (60, 80)
    assert cm.axes.names == ['row', 'col'] and cm.adjacency_axes == ('row', 'col')
    assert cm.axes['col'].size == 4 and cm.axes['row'].scan == 'y'
    jg = inst.job_geometry(cfg, geom, inst.jobs(cfg)[0])
    assert jg.det_valid_weight.shape == (30, 40) and jg.grid_valid_weight.shape == (60, 80)
    layout = inst.exposure_layout(cfg)
    assert layout.sci_ext == [1] and layout.dq_ext == [2] and layout.detector_ids == [0]
    assert inst.frame_tag(cfg) == 'Grid30x40_Chunks3x4'


def test_local_registration():
    @register_instrument('unit_test_cam')
    class Cam(Instrument):
        def jobs(self, c): return [Job('all')]
        def frame_tag(self, c): return 'Cam'
        def exposure_layout(self, c): return ExposureLayout(sci_ext=[0], dq_ext=None, detector_ids=[0])
        def detector_geometry(self, c, o): raise NotImplementedError
        def job_geometry(self, c, g, j): raise NotImplementedError
    assert 'unit_test_cam' in available_instruments()
    assert get_instrument('unit_test_cam').frame_tag({}) == 'Cam'


def test_mode_presets_are_aliases():
    for alias, structural in [('pahfit', 'spectral'), ('pahfit_subch', 'spectral_softpoly'),
                              ('pahfit_lvf', 'spectral_softpoly'), ('tiled', 'tiled'),
                              ('pahfit_lvf_polybasis', 'spectral_polybasis'),
                              ('multiline', 'spectral_polybasis'), ('k2_readout', 'two_block_fixed')]:
        m = get_mode(alias)
        assert m.requested_name == alias
        assert m.name == structural, (alias, m.name)
    assert get_mode('tiled').mosaic_mode == 'none'
    assert type(get_mode('tiled')).__mro__[1].name == 'spectral_softpoly'
