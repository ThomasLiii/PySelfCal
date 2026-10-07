"""The run scripts (``selfcal_scripts/runs/``) and the recipe module: each builds its settings and
lowers to engine runs without the data (the tiled ones need their field's reference grid). That
each lowers to what its TOML config ran is checked on the real geometry by
``selfcal_scripts/gates/config_equivalence.py runs``."""
import importlib
import os
import sys

import pytest

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from selfcal.run.lower import lower  # noqa: E402

RUNS = sorted(f[:-3] for f in os.listdir(os.path.join(_REPO, 'selfcal_scripts', 'runs'))
              if f.endswith('.py') and not f.startswith('_'))


@pytest.mark.parametrize('name', RUNS)
def test_a_run_script_builds_and_lowers(name, monkeypatch):
    module = importlib.import_module(f'selfcal_scripts.runs.{name}')
    if not hasattr(module, 'RUN'):
        assert hasattr(module, 'REPROJECT') or hasattr(module, 'PRECOMPUTE')
        return
    # a tiled run reads its grid's shape from the field's ref.fits: the NEP / SEP grids' own (no /mnt)
    from selfcal.run import lower as lower_module
    monkeypatch.setattr(lower_module, 'reference_shape', lambda field: (12676, 12672))
    run = {k: v for k, v in module.RUN.items() if k in ('jobs', 'tiles', 'passes', 'frames', 'compute')}
    for field in getattr(module, 'FIELDS', None) or [module.FIELD]:
        lowered = lower(field, module.RECIPE, **run)
        assert lowered and all(low.jobs for low in lowered)
        if run.get('tiles') is not None:
            assert all(low.cfg.tiling and low.cfg.tiling['ref_shape'] == [12676, 12672] for low in lowered)


def test_every_run_script_but_the_campaigns_has_its_config():
    configs = os.path.join(_REPO, 'selfcal_scripts', 'configs')
    missing = [n for n in RUNS if not os.path.exists(os.path.join(configs, f'{n}.toml'))]
    assert missing == ['numcol3_maps']


def test_the_transfer_function_recipe_is_its_toml():
    """transfer_function.py's FIDUCIAL makes what transfer_function.toml makes: the inputs a cal's and
    a mosaic's sidecars record are the same for both (the TOML's spelling of the defaults resolves
    to the same values)."""
    from selfcal.run.config import load_config
    from selfcal.run.convert import from_runconfig
    from selfcal.run.products import cal_inputs, fingerprint, mosaic_inputs
    from selfcal_scripts.transfer_function import transfer_function as tf
    toml = os.path.join(_REPO, 'selfcal_scripts', 'transfer_function', 'transfer_function.toml')
    converted = from_runconfig(load_config(toml))
    field, job = converted.field, converted.jobs[0]
    frames = [f'exp_{i:04d}_det_00.h5' for i in range(3)]
    for inputs in (lambda r: cal_inputs(field, r, job, frames),
                   lambda r: mosaic_inputs(field, r, job, 'the cal', frames)):
        assert fingerprint(inputs(tf.FIDUCIAL)) == fingerprint(inputs(converted.recipe))
