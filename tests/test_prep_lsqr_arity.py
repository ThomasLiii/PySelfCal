"""``_prep_lsqr`` must return the same number of values on every path.

Its callers (``_prep_lsqr_batch_worker`` and ``_table_batch``) unpack six values, so an early
return with fewer raises ``ValueError: not enough values to unpack`` inside a pool worker and takes
down the whole solve. That is what happened on the SEP offset-first run (2026-09-13): a frame whose
subframe had zero valid pixels after masking + the per-subchannel clip killed a pass-3 SKY tile. The
degenerate path is rare and data-dependent, so nothing else pins it down.
"""
import ast
import pathlib

ASSEMBLY = pathlib.Path(__file__).resolve().parents[1] / "selfcal" / "core" / "assembly.py"
WIDTH = 6


def _returns(fn_name):
    tree = ast.parse(ASSEMBLY.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    return [n.value for n in ast.walk(fn) if isinstance(n, ast.Return)]


def test_prep_lsqr_tuple_returns_all_have_six_values():
    widths = {len(r.elts) for r in _returns("_prep_lsqr") if isinstance(r, ast.Tuple)}
    assert widths == {WIDTH}, (
        f"_prep_lsqr returns tuples of widths {sorted(widths)}; its callers unpack exactly {WIDTH}")


def test_prep_lsqr_callers_unpack_six():
    tree = ast.parse(ASSEMBLY.read_text())
    for name in ("_prep_lsqr_batch_worker", "_table_batch"):
        fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)
        targets = [n.targets[0] for n in ast.walk(fn)
                   if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
                   and isinstance(n.value, ast.Name) and n.value.id == "result"]
        assert targets, f"expected `... = result` tuple unpacking in {name}"
        assert all(len(t.elts) == WIDTH for t in targets), name
