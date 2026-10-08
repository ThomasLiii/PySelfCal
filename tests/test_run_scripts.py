"""The run scripts (``selfcal_scripts/runs/``) and the recipe module: each builds its settings and
lowers to engine runs without the data (the tiled ones need their field's reference grid). What
the engine does with each is recorded on the real geometry by
``selfcal_scripts/gates/config_equivalence.py views``."""
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
        assert lowered and all(spec.jobs for spec in lowered)
        if run.get('tiles') is not None:
            assert all(spec.tiling is not None and spec.tiling.ref_shape == (12676, 12672) for spec in lowered)
