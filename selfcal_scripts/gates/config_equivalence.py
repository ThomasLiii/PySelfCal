"""What the run engine does with each run of the repository, normalised so that two trees that
run it identically compare equal: the evidence that a change to the engine (or to how the Python
API lowers onto it) leaves what it does unchanged.

``views`` records, for each producer, the engine view of every engine run of its action
(:func:`selfcal.run.equivalence.engine_view`: the library calls' keywords with defaults filled,
resolved per-term damping, offset rows, sky coefficients, jobs, product paths, frames, staging,
tiles, passes), one JSON file each:

* ``script__<name>.json``: a run script of ``selfcal_scripts/runs/`` (a list, one view per engine
  run of its action, in order; a campaign's, FIELDS: ``{'<i> <field name>': that list}``);
* ``example__quickstart.json``: ``examples/quickstart/quickstart.py``;
* ``tool__transfer_function.json``: ``selfcal_scripts/transfer_function/transfer_function.py`` with
  its default inputs;
* ``gate__<name>.json``: each gate of ``python_gates.py``, through a capture of its
  ``Field.calibrate`` call.

A view that cannot be made is stored as ``{"ERROR": "<type>: <message>"}``. The views read this
machine's files (frame lists, reference grids, calibration data) and its CPU count (the default
number of workers): compare views made on one machine, with ``compare-views`` of the directories
made before and after.

Usage:
  config_equivalence.py views <out_dir> [name ...]   # default: every producer
  config_equivalence.py compare-views <dir_a> <dir_b>
"""
import importlib
import json
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _norm(v):
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _norm(x) for k, x in sorted(v.items(), key=lambda kv: str(kv[0]))}
    if isinstance(v, np.generic):
        return v.item()
    if isinstance(v, np.ndarray):
        import hashlib
        return ['ndarray', list(v.shape), str(v.dtype), hashlib.sha1(np.ascontiguousarray(v).tobytes()).hexdigest()]
    if callable(v):
        return ['callable', f"{getattr(v, '__module__', '?')}:{getattr(v, '__qualname__', type(v).__name__)}"]
    return v


def diff(a, b, path=''):
    """The differences between two normalised structures, as 'path: a != b' lines."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff(a.get(k, '<absent>'), b.get(k, '<absent>'), f'{path}.{k}' if path else k)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f'{path}[{i}]')
        return out
    return [] if a == b else [f'{path}: {str(a)[:80]} != {str(b)[:80]}']


# The producers of the stored views: what they return is the file format (see the module docstring).
def _action_views(field, recipe, run):
    """The engine views of ``field.calibrate(recipe, **run)``, one per engine run, in order."""
    from selfcal.run.equivalence import engine_view
    from selfcal.run.lower import lower
    run = {k: v for k, v in run.items() if k != 'overwrite'}     # decides only whether products are made again
    return [engine_view(spec) for spec in lower(field, recipe, task='cal', **run)]


def _views_of_script(module):
    """The engine views of a run script: its action, ``FIELD.calibrate(RECIPE, **RUN)`` as ``selfcal plan``
    reads a script; a reproject script's is its reprojection, a precompute script's its settings; a
    campaign (FIELDS) gives ``{'<i> <field name>': views}`` per field."""
    from selfcal.run.equivalence import engine_view
    if hasattr(module, 'PRECOMPUTE'):
        return [{'precompute': module.PRECOMPUTE}]
    if hasattr(module, 'REPROJECT'):
        return [engine_view(module.FIELD.reprojection_spec(**module.REPROJECT))]
    recipe, run = getattr(module, 'RECIPE', None), getattr(module, 'RUN', {})
    if hasattr(module, 'FIELDS'):
        return {f'{i:02d} {field.name}': _action_views(field, recipe, run) for i, field in enumerate(module.FIELDS)}
    return _action_views(module.FIELD, recipe, run)


def _views_of_gate(fn):
    """The engine views of a gate of ``python_gates.py``: the ``Field.calibrate`` call it makes,
    captured (nothing runs)."""
    import selfcal as sc
    calls = []
    real = sc.Field.calibrate
    sc.Field.calibrate = lambda self, recipe=None, **kw: calls.append((self, recipe, kw))
    try:
        fn()
    finally:
        sc.Field.calibrate = real
    (field, recipe, run), = calls
    return _action_views(field, recipe, run)


def _producers():
    """``[(view-file stem, name, produce)]`` of every producer, in order."""
    folder = os.path.join(REPO, 'selfcal_scripts', 'runs')
    items = []
    for f in sorted(os.listdir(folder)):
        if f.endswith('.py') and not f.startswith('_'):
            name = f[:-3]
            items.append((f'script__{name}', name, lambda name=name: _views_of_script(
                importlib.import_module(f'selfcal_scripts.runs.{name}'))))

    def quickstart():
        module = importlib.import_module('examples.quickstart.quickstart')
        return _action_views(module.FIELD, module.RECIPE, {})

    def transfer_function():
        tf = importlib.import_module('selfcal_scripts.transfer_function.transfer_function')
        field, run = tf.setup(tf.arguments([]))
        return _action_views(field, tf.RECIPE, run)
    items += [('example__quickstart', 'quickstart', quickstart),
              ('tool__transfer_function', 'transfer_function', transfer_function)]
    from selfcal_scripts.gates import python_gates
    for name, fn in python_gates.GATES.items():
        items.append((f'gate__{name}', name, lambda fn=fn: _views_of_gate(fn)))
    return items


def views(out_dir, names=None):
    """Write the normalised engine views of every producer (or of those whose name or view-file stem
    is in ``names``) to ``out_dir``, one JSON file each. Returns the number of views that could not be
    made (stored as ERROR)."""
    from selfcal.run.compute import pin_threads
    pin_threads()               # one BLAS thread, as the engine runs: the views must not depend on the shell
    items = _producers()
    if names:
        for n in names:
            if not any(n in (stem, name) for stem, name, _ in items):
                print(f'NO MATCH {n}')
        items = [item for item in items if item[0] in names or item[1] in names]
    os.makedirs(out_dir, exist_ok=True)
    errors = 0
    for stem, _, produce in items:
        try:
            out = _norm(json.loads(json.dumps(produce(), default=str)))
        except Exception as e:
            out = {'ERROR': f'{type(e).__name__}: {e}'}
            errors += 1
        with open(os.path.join(out_dir, stem + '.json'), 'w') as f:
            json.dump(out, f, indent=1, sort_keys=True)
            f.write('\n')
        failed = isinstance(out, dict) and 'ERROR' in out
        print(f"{'ERROR' if failed else 'ok':<6}  {stem}" + (f"  {out['ERROR'][:160]}" if failed else ''))
    print(f'{len(items)} view files written to {out_dir}' + (f', {errors} of them ERROR' if errors else ''))
    return errors


def _typed(v):
    """Numbers tagged with their type, so that ``diff`` tells 0 from 0.0 and true from 1."""
    if isinstance(v, dict):
        return {k: _typed(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_typed(x) for x in v]
    return f'{type(v).__name__}:{v!r}' if isinstance(v, (bool, int, float)) else v


def compare_views(a, b):
    """The view files of two ``views`` directories, file by file: EQUAL (identical files), DIFFERS
    (with the first differing paths; values compared, then their types) or MISSING (in one
    directory only). Returns the number not equal."""
    names = sorted(n for n in set(os.listdir(a)) | set(os.listdir(b)) if n.endswith('.json'))
    if not names:
        print(f'no view files in {a} or {b}')
        return 1
    bad = 0
    for n in names:
        pa, pb = os.path.join(a, n), os.path.join(b, n)
        if not (os.path.exists(pa) and os.path.exists(pb)):
            print(f'MISSING {n}  (only in {a if os.path.exists(pa) else b})')
            bad += 1
            continue
        with open(pa) as fa, open(pb) as fb:
            ta, tb = fa.read(), fb.read()
        va, vb = json.loads(ta), json.loads(tb)
        d = [] if ta == tb else (diff(va, vb) or diff(_typed(va), _typed(vb)) or ['(formatting only)'])
        both_error = not d and isinstance(va, dict) and 'ERROR' in va
        print(('EQUAL   ' if not d else 'DIFFERS ') + n + ('  (ERROR in both)' if both_error else ''))
        for line in d[:12]:
            print('    ', line)
        bad += bool(d)
    print(f'ALL {len(names)} EQUAL' if not bad else f'{bad} of {len(names)} NOT EQUAL')
    return bad


if __name__ == '__main__':
    sys.path.insert(0, REPO)
    cmd = sys.argv[1] if len(sys.argv) > 1 else ''
    if cmd == 'views':
        sys.exit(1 if views(sys.argv[2], sys.argv[3:] or None) else 0)
    elif cmd == 'compare-views':
        sys.exit(1 if compare_views(sys.argv[2], sys.argv[3]) else 0)
    else:
        print(__doc__)
