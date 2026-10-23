"""``_prep_lsqr`` must return the same number of values on every path.

Its caller (``_prep_lsqr_batch_worker``) unpacks six values, so an early return
with five raises ``ValueError: not enough values to unpack`` inside a pool
worker and takes down the whole solve. That is what happened on the SEP
offset-first run (2026-09-13): a frame whose subframe had zero valid pixels
after masking + the per-subchannel clip killed a pass-3 SKY tile. The degenerate
path is rare and data-dependent, so nothing else pins it down.
"""
import ast
import pathlib

ASSEMBLY = pathlib.Path(__file__).resolve().parents[1] / "selfcal" / "core" / "assembly.py"


def _returns(fn_name):
    tree = ast.parse(ASSEMBLY.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == fn_name)
    return [n.value for n in ast.walk(fn) if isinstance(n, ast.Return)]


def test_prep_lsqr_tuple_returns_all_have_six_values():
    widths = {len(r.elts) for r in _returns("_prep_lsqr") if isinstance(r, ast.Tuple)}
    assert widths == {6}, (
        f"_prep_lsqr returns tuples of widths {sorted(widths)}; the caller in "
        "_prep_lsqr_batch_worker unpacks exactly 6")


def test_prep_lsqr_batch_worker_unpacks_six():
    tree = ast.parse(ASSEMBLY.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "_prep_lsqr_batch_worker")
    targets = [n.targets[0] for n in ast.walk(fn)
               if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Tuple)
               and isinstance(n.value, ast.Name) and n.value.id == "result"]
    assert targets, "expected `... = result` tuple unpacking in _prep_lsqr_batch_worker"
    assert all(len(t.elts) == 6 for t in targets)


def test_zero_valid_pixel_return_is_skippable():
    """The empty return must be shaped so the caller's `len(sub_b) == 0` skip fires."""
    empties = [r for r in _returns("_prep_lsqr")
               if isinstance(r, ast.Tuple)
               and all(isinstance(e, (ast.Call, ast.Constant)) for e in r.elts)]
    assert empties, "expected a literal empty-arrays return in _prep_lsqr"
    for r in empties:
        assert isinstance(r.elts[4], ast.Constant) and r.elts[4].value == 0, \
            "num_rows must be 0 on the degenerate path"
        assert isinstance(r.elts[5], ast.Constant) and r.elts[5].value is None, \
            "off_counts must be None on the degenerate path"


if __name__ == "__main__":
    # pytest is not installed in the `selfcal` env; run the checks directly.
    failed = 0
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            try:
                _fn()
                print(f"  PASS {_name}")
            except Exception as exc:                      # noqa: BLE001
                failed += 1
                print(f"  FAIL {_name}: {exc}")
    raise SystemExit(1 if failed else 0)
