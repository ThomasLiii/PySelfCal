"""The run engine moved from ``selfcal_scripts.runner`` to ``selfcal.run``; the old import
path is an alias, so code written against it (scripts, out-of-tree modes) keeps working and
shares the engine's state (one mode registry, one set of module globals)."""
import importlib

import pytest

MODULES = ['config', 'engine', 'pipelines', 'npass', 'staging', 'runlog', 'postprocess',
           'modes', 'modes.base', 'modes.continuum', 'modes.spectral', 'modes.two_block_fixed',
           'modes.model']


@pytest.mark.parametrize('name', MODULES)
def test_old_path_is_the_same_module(name):
    old = importlib.import_module(f'selfcal_scripts.runner.{name}')
    new = importlib.import_module(f'selfcal.run.{name}')
    assert old is new


def test_from_imports_of_the_old_path():
    from selfcal_scripts.runner.config import RunConfig, load_config  # noqa: F401

    import selfcal.run
    from selfcal_scripts.runner import load_config as lc
    from selfcal_scripts.runner import pipelines, run, staging  # noqa: F401
    assert run is selfcal.run.run and lc is selfcal.run.load_config
    assert pipelines is selfcal.run.pipelines


def test_a_mode_registered_through_the_old_path_is_found_through_the_new():
    from selfcal_scripts.runner.modes.base import CalMode, register_mode

    from selfcal.run.modes import available_modes, get_mode

    @register_mode('shim_probe_mode')
    class _Probe(CalMode):
        pass

    try:
        assert 'shim_probe_mode' in available_modes()
        assert isinstance(get_mode('shim_probe_mode'), _Probe)
    finally:
        from selfcal.run.modes import base
        base._MODE_REGISTRY.pop('shim_probe_mode', None)
