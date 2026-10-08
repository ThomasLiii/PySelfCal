# Settings and fingerprints

Every setting of the Python API is a frozen dataclass derived from
[`Config`][selfcal.config.base.Config], and a product is reused only while the settings that
decided its bytes are what the action would use now. A change to a settings class therefore
decides whether the products made before it stay current. This page says what enters a
product's fingerprint and how to add a setting without changing the fingerprints of existing
products.

## What a fingerprint holds

Next to each product it writes, an action writes a sidecar, `<product>.json`: the product's
inputs and their fingerprint, the SHA-256 of their canonical JSON
([`selfcal.run.products`](../reference/selfcal/run/products.md)).

| product | inputs |
| --- | --- |
| cal | the instrument, the reference grid (`ref.fits`, by content), the job, the model (template and map files by content), the fit, the solve's `Numerics`, the frames (by name) |
| tile cal | the same, plus the tile's box and how frames were assigned to it |
| stitched cal | the fingerprints of its tile cals |
| N-pass product | the fingerprint of the first pass, the pass number and type, the pass settings |
| mosaic | the fingerprint of its cal, the reference grid, the model, the `Coadd`, the coadd's `Numerics`, the frames |

A settings object enters through its encoding (`Config.to_dict()` and the `encode()` of
`selfcal.config.base`): its class's import path under `type`, then every field by name with its
value, defaults included; a function by its import string, or by what it computes when it is sent
by value (`sc.by_value`); a hook object by its class and state. So each of these changes the
fingerprint of every product it touches:

- a settings class moved to another module or renamed;
- a field renamed or removed, or its default changed;
- a field added, unless it is declared as added later (below);
- another encoding of a value (a list where a tuple was, a function under another import path).

The machine settings, `sc.Compute` and its `sc.Tuning`, never enter: they leave the products'
bytes alone. The code does not enter either: a record names the code's version, and a product
made by older code is current when its inputs are.

`tests/test_fingerprints.py` pins the inputs and the fingerprint of every kind of product, for
settings that cover every fingerprinted class, against `tests/data/fingerprints.json`, and fails on
any of the changes above. Regenerate that golden
(`SELFCAL_WRITE_FINGERPRINT_GOLDEN=1 pytest tests/test_fingerprints.py`) only for a change meant
to change what decides the products' bytes, and say so in the commit: every existing product whose
inputs changed is then refused until it is made again.

## Adding a setting

A new setting of a fingerprinted class (an option of `sc.Fit`, say) must leave the products made
before it current. Declare it as added later with
[`added`][selfcal.config.base.added] (also importable as `selfcal.config.added`), with a default
that keeps the behaviour of the code before it:

```python
from dataclasses import KW_ONLY, dataclass

from selfcal.config import Config, added


@dataclass(frozen=True)
class Fit(Config):
    iterations: int = 50
    _: KW_ONLY
    ...
    stop: Stop | None = added(None, since="2026-10")      # the opt-in stop rules
```

`added(default, *, since)` makes a dataclass field whose metadata holds the date it was added. Such
a field is left out of `to_dict()` and of the encoding while it equals its default, so every
product made before the field existed, and every product made since with the default, keeps its
fingerprint. Set to another value, the field is encoded and the fingerprint changes, as it should:
that product is made by other inputs. A record written before the field existed decodes with the
field at its default.

Rules for every change to a fingerprinted class (`sc.SPHEREx`, `sc.Euclid`, `sc.Camera`, the model's
classes, `sc.Fit`, `sc.Clip`, `sc.ChunkGroups`, `sc.Stop` and its rules (`sc.ResidualRule`,
`sc.GradientRule`, `sc.LargeScaleRule`), `sc.Coadd`, `sc.Numerics`, `sc.Tiles`, `sc.Passes`,
`sc.Refit`):

- a new field is declared with `added(default, since=...)`;
- a class is never renamed or moved, a field never renamed or removed, a default never changed:
  a new behaviour is a new field;
- a constant of an instrument that is not a setting (its data unit, its capabilities) is a class
  attribute without an annotation, or a property, so that the dataclass does not make it a field;
- a setting that never changes a byte belongs to `sc.Compute`, or to `sc.Tuning` when it is one of
  the library's `SELFCAL_*` environment variables; one that writes files beside the products
  without changing them (`sc.Snapshots`), or observes a solve without changing it (`sc.Monitor`),
  is an argument of the action (`calibrate(snapshots=..., monitor=...)`), recorded in the action's
  record and never in a fingerprint; one that can change where a solve ends (`sc.Fit(stop=...)`)
  is a fingerprinted setting, declared with `added`;
- `pytest tests/test_fingerprints.py` passes with the golden unchanged.

The byte-equality [regression gates](gates.md) check the other half: that the products themselves
did not change.
