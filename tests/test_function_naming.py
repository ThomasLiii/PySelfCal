"""The library states which data variables a function reads: every ``Function`` / ``Variance`` it builds passes
``of=``, no function it offers as a basis or coefficient takes a parameter named after a built-in variable, and
the model side of SPHEREx (its terms, chunk maps and jobs) is not in the instrument module."""
import ast
import importlib
import os

import pytest

import selfcal as sc
from selfcal.instruments import spherex
from selfcal.models.variables import BUILTIN_VARIABLES

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'selfcal')


def _library_sources():
    for base, _, files in os.walk(ROOT):
        for f in files:
            if f.endswith('.py'):
                path = os.path.join(base, f)
                yield path, ast.parse(open(path).read(), path)


def test_every_library_function_call_names_its_variables():
    missing = []
    for path, tree in _library_sources():
        if os.path.relpath(path, ROOT) == 'models/model.py':      # the API itself: it wraps a bare function the
            continue                                             # user passes (the implicit mode's definition)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, 'id', None)
                if name in ('Function', 'Variance') and not any(k.arg == 'of' for k in node.keywords):
                    missing.append(f'{os.path.relpath(path, ROOT)}:{node.lineno}')
    assert not missing, f'Function(...) without of= in the library: {missing}'


def test_no_library_function_has_a_parameter_named_after_a_coordinate_variable():
    """A parameter named like a built-in coordinate (det_x, ...) would bind it silently through sc.Function's
    implicit mode and bake an axis into a library function's signature."""
    coordinates = {v for v in BUILTIN_VARIABLES if v.startswith(('det_', 'sky_'))}
    offenders = []
    for path, tree in _library_sources():
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                for arg in node.args.args + node.args.kwonlyargs:
                    if arg.arg in coordinates:
                        offenders.append(f'{os.path.relpath(path, ROOT)}:{node.lineno} {node.name}({arg.arg})')
    assert not offenders, offenders


def test_the_spherex_instrument_module_holds_no_model_terms():
    from selfcal.instruments.spherex import settings
    assert settings.__all__ == ['SPHEREx', 'LINE_TEMPLATES', 'precompute_lvf', 'zodi_anchor']
    assert all(getattr(spherex, n) is getattr(settings, n) for n in settings.__all__)
    for name in ('frame_slope', 'spectral_offsets', 'arc_position', 'line', 'tilt', 'pah_3p29',
                 'channel', 'channels', 'group', 'window', 'SUBCH_WINDOWS'):
        assert not hasattr(spherex, name), name
    assert not hasattr(sc, 'catalog')
    for module in ('selfcal.models.bases', 'selfcal.instruments.spherex.line_catalog'):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module)
    from selfcal_scripts.spherex import chunks, jobs, terms
    assert all(callable(getattr(terms, n)) for n in ('tilt', 'frame_slope', 'arc_position', 'hats', 'spectral_offsets',
                                                     'line', 'pah_3p29'))
    assert all(callable(getattr(chunks, n)) for n in ('subchannels', 'subchannel_map', 'detector_columns', 'readouts',
                                                      'arc_renderer'))
    assert all(callable(getattr(jobs, n)) for n in ('channel', 'channels', 'group', 'window'))
