"""The SPHEREx position across the LVF arcs and the spectral offset term on it: every pixel of a subchannel lies
within that subchannel's unit interval of the position, a channel job's hat functions sum to 1 on its pixels, and the
term builds in a model with the final recipe's other terms."""
import numpy as np
import pytest

import selfcal as sc
from selfcal.models.spec import ModelSpec
from selfcal.run.engine import chunk_map_specs, with_chunk_maps
from selfcal_scripts.recipes.spherex import maps
from selfcal_scripts.spherex import terms as T


@pytest.fixture(autouse=True)
def _blank_calibration_maps(monkeypatch):
    # The detector geometry and its subchannel map read the detector's band-centre map from the calibration files,
    # which exist only on the processing host; with the shipped LVF fit given, that map only sets the shape, so a
    # blank one stands in (as in test_spherex_detector_columns.py).
    from selfcal.instruments.spherex import settings, spherex_utility
    blank = np.zeros((2040, 2040), dtype=np.float32)
    for module in (settings, spherex_utility):
        monkeypatch.setattr(module, 'load_calibration', lambda *args, **kwargs: (blank, blank))


def _geometry():
    """Detector 1 with the channel recipes' chunk maps (17 channels of 10 subchannels, one column) evaluated on it,
    as the run engine builds it."""
    inst = sc.SPHEREx(1)
    specs = chunk_map_specs(inst, ModelSpec.from_config(sc.Model(chunk_maps=maps(10, 17, 1)).lower()))
    return with_chunk_maps(inst.geometry(1), specs)


def test_the_arc_position_places_every_pixel_in_its_subchannel():
    geom = _geometry()
    u = T.arc_position(geom)
    sub = geom.chunk_maps['subchannel'].det
    inner = (sub >= 1) & (sub <= 170)                   # the subchannels of the band; 0 and 171 lie beyond its edges
    assert u.shape == geom.shape and np.isfinite(u).all()
    assert np.all(u[inner] >= sub[inner] - 1 - 1e-6) and np.all(u[inner] <= sub[inner] + 1e-6)
    assert np.all(u[sub == 0] <= 1e-6) and np.all(u[sub == 171] >= 170 - 1e-6)


def test_a_channel_job_s_hat_functions_sum_to_one_on_its_pixels():
    geom = _geometry()
    u = T.arc_position(geom)
    sub = geom.chunk_maps['subchannel'].det
    term = T.spectral_offsets(range(150, 162))    # Ch16's job: subchannels 150-161
    assert term.n == 13 and term.on == 'detector' and term.basis.params == {'first': 149, 'last': 161}
    hats = T.hats(u, 149, 161)
    job = (sub >= 150) & (sub <= 161)
    total = np.sum(hats, axis=0)
    assert np.allclose(total[job], 1.0, atol=1e-9)
    assert np.all(np.sum(np.asarray(hats) > 0, axis=0)[job] <= 2)          # at most two nonzero at a pixel


def test_the_channel_recipe_replaces_the_subchannel_constants():
    from selfcal_scripts.recipes.spherex import SUBCH_VARWEIGHT_COLUMNS_FINAL, continuous_final
    recipe = continuous_final(16)
    terms = recipe.model.offsets
    assert recipe.name == SUBCH_VARWEIGHT_COLUMNS_FINAL.name + '_continuous'
    assert terms[0].name == 'spectral' and terms[0].n == 13 and terms[0].basis.params == {'first': 149, 'last': 161}
    assert not any(o.on == 'subchannel' and o.times is None and o.basis is None for o in terms)
    assert [o.name for o in terms[1:]] == [o.name for o in SUBCH_VARWEIGHT_COLUMNS_FINAL.model.offsets[1:]]
    assert 'arc_position' in dict(recipe.model.variables) and dict(recipe.model.variables)['arc_position'].source is T.arc_position
    assert recipe.fit == SUBCH_VARWEIGHT_COLUMNS_FINAL.fit and recipe.coadd == SUBCH_VARWEIGHT_COLUMNS_FINAL.coadd
