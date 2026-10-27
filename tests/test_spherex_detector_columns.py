"""The SPHEREx per-detector-column chunk map: one chunk per column, id = the column, every row
(the reference columns included), at detector and grid resolution; a frame-shared term on it builds."""
import numpy as np
import pytest

import selfcal as sc
from selfcal.models.spec import ModelSpec
from selfcal.run.engine import chunk_map_specs, with_chunk_maps
from selfcal_scripts.recipes.spherex import maps
from selfcal_scripts.spherex import chunks as C


@pytest.fixture(autouse=True)
def _blank_calibration_maps(monkeypatch):
    # The engine builds every chunk map of the model, and the subchannel map reads the detector's band-centre
    # map from the calibration files, which exist only on the processing host. With the shipped LVF
    # fit given, that map only sets the shape, so a blank one stands in (as in
    # test_offset_structure.py). The per-column map does not depend on it.
    from selfcal.instruments.spherex import settings, spherex_utility
    blank = np.zeros((2040, 2040), dtype=np.float32)
    for module in (settings, spherex_utility):
        monkeypatch.setattr(module, 'load_calibration', lambda *args, **kwargs: (blank, blank))


def _geometry(oversample):
    """Detector 4 with the recipes' chunk maps of three columns (:func:`maps`) and the readout channels evaluated on
    it, as the run engine builds it."""
    inst = sc.SPHEREx(4)
    model = sc.Model(chunk_maps={**maps(num_ch=34, num_col=3), 'readout': sc.ChunkMap(C.readouts)})
    return with_chunk_maps(inst.geometry(oversample), chunk_map_specs(inst, ModelSpec.from_config(model.lower())))


def test_detector_column_map_is_one_chunk_per_column():
    geom = _geometry(1)
    cm = geom.chunk_maps['detector_column']
    H, W = geom.shape
    assert cm.n_chunks == W == 2040
    assert np.array_equal(cm.det, np.tile(np.arange(W, dtype=np.int32), (H, 1)))
    assert list(cm.axes.names) == ['detector_column']
    assert list(geom.chunk_maps) == ['subchannel', 'detector_column', 'readout'] and geom.primary == 'subchannel'
    readout = geom.chunk_maps['readout']
    assert list(readout.axes.names) == ['readout'] and readout.axes['readout'].size == readout.n_chunks == 32


def test_detector_column_map_follows_the_oversampling():
    geom = _geometry(2)
    cm = geom.chunk_maps['detector_column']
    assert cm.grid.shape == tuple(2 * v for v in geom.shape)
    assert np.array_equal(cm.grid[::2, ::2], cm.det) and np.array_equal(cm.grid[1::2, 1::2], cm.det)


def test_a_shared_column_term_builds_on_the_map():
    term = sc.Offsets('columns', on='detector_column', per='all')
    model = sc.Model(sky=[sc.Sky(damping=0.0)], offsets=[sc.Offsets(on='subchannel', exclude=sc.Poly(0)), term, sc.Offsets('scalar', on='detector')])
    assert model.offsets[1].on == 'detector_column' and model.offsets[1].per == 'all'
