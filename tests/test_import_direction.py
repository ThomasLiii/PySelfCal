"""Layering rule: the numerical layers never import the instrument layer, and the
instrument layer never imports the pipeline/runner layers. Scanned by AST (incl.
function-level imports). Known violations are listed with their audit reference and
must only ever shrink."""
import ast
import os

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG = os.path.join(_REPO, 'selfcal')

FORBIDDEN = {
    # importer prefix (relative to selfcal/) : forbidden imported module prefixes
    'core': ('selfcal.instruments', 'selfcal.pipeline', 'selfcal_scripts'),
    'models': ('selfcal.instruments', 'selfcal.pipeline', 'selfcal_scripts'),
    'geometry': ('selfcal.instruments', 'selfcal.pipeline', 'selfcal_scripts'),
    'io': ('selfcal.instruments', 'selfcal.pipeline', 'selfcal_scripts'),
    'pipeline': ('selfcal.instruments', 'selfcal_scripts'),
    'instruments': ('selfcal.pipeline', 'selfcal_scripts'),
}
# Audited violations still present (SYNTHESIS §0 / audits B R6, D R1); remove entries as they are fixed.
KNOWN = {
    ('models/sky_model.py', 'selfcal.instruments.spherex.spherex_utility'),
    ('instruments/spherex/adapter.py', 'selfcal.pipeline.npass'),
}


def _resolve(module, level, importer_rel):
    if level == 0:
        return module or ''
    parts = importer_rel.split('/')[:-1]          # package path of the importer inside selfcal/
    base = ['selfcal'] + parts[:len(parts) - (level - 1)] if level - 1 <= len(parts) else ['selfcal']
    return '.'.join(base + ([module] if module else []))


def _violations():
    out = []
    for root, _, files in os.walk(PKG):
        for fn in files:
            if not fn.endswith('.py'):
                continue
            path = os.path.join(root, fn)
            rel = os.path.relpath(path, PKG)
            top = rel.split('/')[0]
            if top not in FORBIDDEN:
                continue
            tree = ast.parse(open(path).read(), filename=path)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    mod = _resolve(node.module, node.level, rel)
                    targets = [mod]
                elif isinstance(node, ast.Import):
                    targets = [a.name for a in node.names]
                else:
                    continue
                for t in targets:
                    if any(t == f or t.startswith(f + '.') for f in FORBIDDEN[top]):
                        out.append((rel, t))
    return out


def test_import_direction():
    found = set(_violations())
    new = found - KNOWN
    assert not new, f"new layering violations: {sorted(new)}"
    stale = KNOWN - found
    assert not stale, f"KNOWN list has entries that no longer occur (remove them): {sorted(stale)}"


if __name__ == '__main__':
    test_import_direction()
    print("OK import direction (known violations:", sorted(KNOWN), ")")
