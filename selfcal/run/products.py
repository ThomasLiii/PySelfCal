"""Products and their fingerprints: a product is reused only when it is complete and was made
by the same inputs.

Every product an action of the Python API writes (a cal file, a tile cal, a stitched cal, an
N-pass product, a mosaic) gets a sidecar ``<product>.json``, written after the product itself
(which is written atomically, :mod:`selfcal.io.atomic`). The sidecar holds the product's
*inputs*, exactly what decided its bytes and nothing else, and their fingerprint:

=============  ==============================================================================
cal            the instrument, the job, the model, the fit (the N-pass first pass: with its
               clip), the numerics of the solve, and the frames (by name); a solve continued
               from another's cal (``calibrate(start=...)``), that cal's identity
tile cal       the same, plus the tile's box and how frames were assigned to it
stitched cal   the fingerprints of its tile cals
pass product   the fingerprint of the first pass, the pass number and type, the pass settings
mosaic         the fingerprint of its cal, the coadd, the numerics of the coadd, the frames
=============  ==============================================================================

Machine settings (``Compute``) never enter: they leave products byte-identical. Two runs that
share a first pass (the same model, fit and frames) therefore share its products, and anything
downstream that differs is made again.

Before an action starts, :func:`check` compares each product it would reuse with what it would
make: a product with a matching sidecar is reused; one made by other inputs is refused with
the differences; one without a sidecar (made before records existed, or by a TOML run) is
refused until :meth:`~selfcal.run.field.Field.adopt` checks it and writes its sidecar.
"""
from __future__ import annotations

import datetime
import glob
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, field

import numpy as np

from ..config.base import ConfigError, encode
from ..io.atomic import atomic_path

__all__ = ['sidecar_path', 'read_sidecar', 'write_sidecar', 'fingerprint', 'check', 'cal_inputs',
           'mosaic_inputs', 'stitched_inputs', 'pass_inputs', 'frames_digest', 'diff_inputs', 'Product', 'Book',
           'expected_products', 'verify', 'remove_product', 'clear_stale_intermediates', 'start_identity',
           'start_label']

SCHEMA = 1


# --------------------------------------------------------------------------- sidecars
def sidecar_path(product) -> str:
    """The sidecar of ``product``: ``<product>.json``."""
    return os.fspath(product) + '.json'


def _canonical(x):
    """A JSON-ready, order-independent form; long strings by their hash. A function sent by value
    counts by what it computes (its digest), an object (a hook) by its class and state: their
    pickles and reprs change from process to process."""
    if isinstance(x, dict):
        if 'by_value' in x and 'digest' in x:
            x = {'by_value': x['by_value'], 'digest': x['digest']}
        elif 'object' in x and 'state' in x:
            x = {'object': x['object'], 'state': x['state']}
        return {str(k): _canonical(v) for k, v in sorted(x.items(), key=lambda kv: str(kv[0]))}
    if isinstance(x, (list, tuple)):
        return [_canonical(v) for v in x]
    if isinstance(x, str) and len(x) > 512:
        return 'sha256:' + hashlib.sha256(x.encode()).hexdigest()
    return x


def fingerprint(inputs) -> str:
    """The SHA-256 of the inputs' canonical JSON."""
    text = json.dumps(_canonical(inputs), sort_keys=True, separators=(',', ':'), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def read_sidecar(product) -> dict | None:
    """The sidecar of ``product``, or None when there is none."""
    path = sidecar_path(product)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def write_sidecar(product, inputs, *, record=None, adopted=False):
    """Write ``product``'s sidecar (atomically, after the product)."""
    st = os.stat(product)
    data = {'selfcal_product': SCHEMA, 'product': os.path.basename(os.fspath(product)),
            'fingerprint': fingerprint(inputs), 'inputs': _canonical(inputs), 'size': st.st_size,
            'mtime_ns': st.st_mtime_ns, 'record': record, 'adopted': bool(adopted),
            'written': datetime.datetime.now().isoformat(timespec='seconds')}
    with atomic_path(sidecar_path(product)) as tmp:
        with open(tmp, 'w') as f:
            json.dump(data, f, indent=1, sort_keys=True, default=str)


def diff_inputs(a, b, path='') -> list[str]:
    """The differences between two inputs, as ``path: a -> b`` lines."""
    a, b = _canonical(a), _canonical(b)
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += diff_inputs(a.get(k, '<absent>'), b.get(k, '<absent>'), f'{path}.{k}' if path else k)
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff_inputs(x, y, f'{path}[{i}]')
        return out
    if a == b:
        return []
    return [f'{path}: {_short(a)} -> {_short(b)}']


def _short(v):
    text = str(v)
    return text[:60] + '...' if len(text) > 60 else text


def check(product, inputs) -> tuple[str, list[str]]:
    """The state of an existing product against the inputs it would be made from:
    ``("missing", [])``, ``("current", [])``, ``("different", differences)``,
    ``("unrecorded", [])`` (no sidecar) or ``("changed", [why])`` (written again after its sidecar,
    e.g. by a TOML run: its size or its modification time is not the one recorded)."""
    product = os.fspath(product)
    if not os.path.exists(product):
        return 'missing', []
    side = read_sidecar(product)
    if side is None:
        return 'unrecorded', []
    st = os.stat(product)
    if side.get('size') != st.st_size:
        return 'changed', [f'its size is {st.st_size} bytes, the sidecar recorded {side.get("size")}']
    if side.get('mtime_ns') is not None and side['mtime_ns'] != st.st_mtime_ns:
        when = datetime.datetime.fromtimestamp(st.st_mtime_ns / 1e9).isoformat(timespec='seconds')
        return 'changed', [f'it was written again at {when}, after its sidecar']
    if side.get('fingerprint') == fingerprint(inputs):
        return 'current', []
    return 'different', diff_inputs(side.get('inputs', {}), inputs)


def remedy(state, *, input_cal=False, tile=False) -> str:
    """What to do about a refused product: ``input_cal`` for the cal of a mosaic action (which never
    makes cals), ``tile`` for a tile cal (named by the Tiles' ``tile_name``)."""
    if input_cal:
        return ("calibrate it again with this recipe (field.calibrate), or adopt it if this recipe made it"
                if state == 'unrecorded' else "calibrate it again with this recipe (field.calibrate)")
    again = "or pass overwrite=True to make it again"
    if state == 'unrecorded':
        return f"check it and adopt it (field.adopt(recipe, ...)), {again}"
    if state == 'different':
        rename = ("give the recipe and the Tiles' tile_name names of their own" if tile else
                  "give the recipe its own name (recipe.replace(name=...))")
        return f"{rename}, {again}"
    return again[3:]


def refusal(product, state, details, *, input_cal=False, tile=False) -> ConfigError:
    """The error for a product that an action would reuse but must not."""
    name = os.path.basename(os.fspath(product))
    fix = remedy(state, input_cal=input_cal, tile=tile)
    fix = fix[0].upper() + fix[1:]
    if state == 'different':
        shown = '; '.join(details[:6]) + (f' (+{len(details) - 6} more)' if len(details) > 6 else '')
        return ConfigError(f"{name} exists but was made with different inputs ({shown}). {fix}")
    if state == 'unrecorded':
        return ConfigError(f"{name} exists but has no record of how it was made (made before records existed, by "
                           f"a TOML run, or interrupted). {fix}")
    return ConfigError(f"{name} changed after it was recorded ({'; '.join(details)}). {fix}")


# --------------------------------------------------------------------------- inputs
def frames_digest(frames) -> dict:
    """The frames of a product, by name: their number and the hash of their sorted names."""
    names = sorted(os.path.basename(os.fspath(f)) for f in frames)
    return {'n': len(names), 'sha256': hashlib.sha256('\n'.join(names).encode()).hexdigest()}


def _instrument(field):
    """The instrument, and the reference grid (``ref.fits``, by content) the products are made on."""
    return {'instrument': encode(field.instrument), 'pixel_scale': field.pixel_scale,
            'reference': _file_digest(os.path.join(field.path, 'ref.fits'))}


def job_key(job) -> dict:
    """A job (the API's or the engine's) as plain data."""
    value = job.value
    if isinstance(value, (list, tuple)):
        value = [int(v) if isinstance(v, (int, np.integer)) else v for v in value]
    return {'name': job.name, 'kind': job.kind, 'value': value}


def solve_fit(recipe, passes=None, instrument=None):
    """The fit the solve uses: the recipe's, with the N-pass first pass's clip when one is given,
    defaults resolved (the instrument's ignored flags; the tolerance as ``(atol, btol)``)."""
    fit = recipe.fit
    if passes is not None and passes.init_clip is not None:
        clip = passes.init_clip
        changes = {'clip': clip.replace(ignore_flags=None)}
        if clip.ignore_flags is not None:
            changes['ignore_flags'] = clip.ignore_flags
        fit = fit.replace(**changes)
    if fit.ignore_flags is None and instrument is not None:
        fit = fit.replace(ignore_flags=tuple(instrument.default_ignore_flags()))
    return fit.replace(tolerance=fit.atol_btol)


_DIGESTS = {}


def _file_digest(path):
    """``{"sha256": ...}`` of a file's bytes (the path itself when it cannot be read)."""
    try:
        st = os.stat(path)
    except OSError:
        return path
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _DIGESTS:
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for block in iter(lambda: f.read(1 << 22), b''):
                h.update(block)
        _DIGESTS[key] = {'sha256': h.hexdigest()}
    return _DIGESTS[key]


def content_addressed(x):
    """Encoded settings with every file they name replaced by what it holds: a line template or a
    detector / sky map file by the hash of its bytes, a solved sky by its cal's fingerprint. The same
    template reached through another path (another checkout) gives the same inputs."""
    if isinstance(x, dict):
        kind = str(x.get('type', ''))
        if kind.endswith(':Shape') and x.get('kind') == 'template':
            params = dict(x.get('params') or {})
            if isinstance(params.get('file'), str):
                params['file'] = _file_digest(params['file'])
            x = {**x, 'params': params}
        elif (kind.endswith(':DetectorMap') or kind.endswith(':SkyMap')) and isinstance(x.get('source'), str) \
                and os.path.exists(x['source']):
            x = {**x, 'source': _file_digest(x['source'])}
        elif kind.endswith(':SolvedSky') and isinstance(x.get('cal'), str) and os.path.exists(x['cal']):
            x = {**x, 'cal': product_fingerprint(x['cal'])}
        return {k: content_addressed(v) for k, v in x.items()}
    if isinstance(x, list):
        return [content_addressed(v) for v in x]
    return x


def resolved_model(model):
    """The model with each sky term's damping resolved (what the solve uses)."""
    return model.replace(sky=tuple(t.replace(damping=d) for t, d in zip(model.sky, model.sky_dampings())))


def cal_inputs(field, recipe, job, frames, *, passes=None, tile=None, start=None) -> dict:
    """The inputs of a cal (a tile's, with ``tile = {"name", "box", "assign", "halo"}``; a solve
    continued from another cal's solution, with ``start``, that cal's :func:`start_identity`). A key
    is present only when its input is: a cal made without them keeps its fingerprint."""
    n = recipe.numerics
    out = {'kind': 'cal', **_instrument(field), 'job': job_key(job),
           'model': content_addressed(encode(resolved_model(recipe.model))),
           'fit': encode(solve_fit(recipe, passes, field.instrument)),
           'numerics': {'threads': n.threads, 'batch': n.batch, 'rmatvec_threads': n.rmatvec_threads},
           'frames': frames_digest(frames)}
    if tile is not None:
        out['tile'] = tile
    if start is not None:
        out['start'] = start
    return out


def start_identity(path) -> dict:
    """The identity of the cal a solve starts from: ``{"fingerprint": ...}``, its sidecar's, when it has
    a current one (written after the cal, the cal unchanged since), else ``{"sha256": ...}``, the hash
    of its bytes. Its path does not enter: the same cal under another path is the same start."""
    side = read_sidecar(path)
    if side is not None and side.get('fingerprint'):
        st = os.stat(path)
        if side.get('size') == st.st_size and side.get('mtime_ns') in (None, st.st_mtime_ns):
            return {'fingerprint': side['fingerprint']}
    digest = _file_digest(os.fspath(path))
    if not isinstance(digest, dict):
        raise ConfigError(f"start={path}: the cal cannot be read")
    return dict(digest)


def start_label(identity) -> str:
    """:func:`start_identity` as one string, ``fingerprint:<sha256>`` or ``sha256:<sha256>`` (the
    cal's ``solve`` group records it as ``start_identity``)."""
    ((kind, value),) = identity.items()
    return f'{kind}:{value}'


def stitched_inputs(tile_fingerprints, line) -> dict:
    """The inputs of a stitched cal: its tile cals (by fingerprint)."""
    return {'kind': 'stitched', 'tiles': dict(sorted(tile_fingerprints.items())), 'line': bool(line)}


def pass_inputs(init_fingerprint, passes, index, pass_type) -> dict:
    """The inputs of the N-pass product of pass ``index`` (``"sky"`` or ``"off"``)."""
    settings = encode(passes)
    for k in ('n', 'stop_tol', 'keep_moments', 'ends_on_offset', 'init_clip'):
        settings.pop(k, None)
    return {'kind': 'pass', 'init': init_fingerprint, 'index': int(index), 'type': pass_type, 'passes': settings}


def mosaic_inputs(field, recipe, job, cal_fingerprint, frames) -> dict:
    """The inputs of a mosaic: its cal (by fingerprint), the model (the coadd reads its chunk maps,
    offset bases and observation weight, also with a cal made elsewhere), the coadd and its
    numerics, the frames."""
    n = recipe.numerics
    coadd = recipe.coadd
    if coadd.ignore_flags is None:
        coadd = coadd.replace(ignore_flags=tuple(field.instrument.default_ignore_flags()))
    return {'kind': 'mosaic', **_instrument(field), 'job': job_key(job), 'cal': cal_fingerprint,
            'model': content_addressed(encode(resolved_model(recipe.model))),
            'coadd': encode(coadd), 'numerics': {'mosaic_batch': n.mosaic_batch,
                                                        'coadd_batch': n.coadd_batch},
            'frames': frames_digest(frames)}


def product_fingerprint(product) -> str:
    """The fingerprint recorded in ``product``'s sidecar; for a product without one, its name,
    size and modification time (a cal made elsewhere, e.g. ``mosaic(cal=...)``)."""
    side = read_sidecar(product)
    if side is not None:
        return side['fingerprint']
    st = os.stat(product)
    return f'unrecorded:{os.path.abspath(os.fspath(product))}:{st.st_size}:{st.st_mtime_ns}'


# --------------------------------------------------------------------------- the products of an action
@dataclass
class Product:
    """One product of an action: its ``kind`` (``cal``, ``stitched``, ``pass``, ``mosaic``), ``path``,
    job and the thunk of its inputs; its ``state`` once checked (see :func:`check`)."""
    kind: str
    path: str
    job: object
    inputs: object = None
    state: str = 'missing'
    details: list = field(default_factory=list)
    tile: str | None = None
    index: int | None = None
    depends: tuple = ()                # the products it is made from (a mosaic: its cal)


def _cal_frames(cal_path):
    from ..io.calfile import CalFile
    with CalFile(cal_path) as cal:
        return list(cal.reproj_list)


class Book:
    """The products of one action: what each would be made from (checked before the run) and,
    called by the engine as each product is written (``RunSpec.on_product``), its sidecar. The
    same builders serve both, so a product the run writes is current for the same settings later.
    Each solve the engine ends (``kind="solve"``) is entered in the action's record
    (``action_record``: :meth:`selfcal.run.records.Record.add_solve`)."""

    def __init__(self, field, recipe, *, passes=None, tiles=None, record=None, start=None):
        self.field, self.recipe, self.passes, self.tiles = field, recipe, passes, tiles
        self.start = start             # {job name: the cal its solve starts from} (calibrate(start=...))
        self.record = record           # the action record's path (the sidecars name it)
        self.action_record = None      # the action's Record (its solves are entered in it)
        self.expected = {}             # path -> inputs thunk (the products the action knows of)
        self.init_path = {}            # job name -> the cal that holds the first pass's sky
        self.written = []

    def fp_of(self, path) -> str:
        """The fingerprint of a product: for one of the action's, what it is made from (an action runs
        only when each existing product is current, so this is the sidecar's); for another (a cal
        from elsewhere), its sidecar's, or its name, size and time."""
        path = os.fspath(path)
        if path in self.expected:
            return fingerprint(self.expected[path]())
        return product_fingerprint(path)

    # ---- inputs -------------------------------------------------------------------------------
    def cal(self, job, frames, tile=None):
        start = self.start.get(job.name) if self.start else None
        return cal_inputs(self.field, self.recipe, job, frames, passes=self.passes, tile=tile,
                          start=None if start is None else start_identity(start))

    def tile_key(self, tile):
        return {'name': tile.name, 'box': [int(v) for v in tile.bbox], 'assign': self.tiles.assign,
                'halo': int(self.tiles.halo)}

    def mosaic(self, job, cal_path, frames):
        return mosaic_inputs(self.field, self.recipe, job, self.fp_of(cal_path), frames)

    def stitched(self, tile_paths):
        return stitched_inputs({name: self.fp_of(p) for name, p in tile_paths.items()},
                               len(self.recipe.model.sky) > 1)

    def passed(self, job, index, pass_type):
        return pass_inputs(self.fp_of(self.init_path[job.name]), self.passes, index, pass_type)

    # ---- the engine's callback ----------------------------------------------------------------------
    def __call__(self, kind, path, *, job, frames=None, tile=None, tiles=None, cal=None, frame_dir=None,
                 index=None, pass_type=None, solve=None):
        if kind == 'solve':
            if self.action_record is not None:
                self.action_record.add_solve({'cal': os.fspath(path), 'job': job.name, 'tile': tile, **solve})
            return
        if kind == 'cal':
            inputs = self.cal(job, frames, None if tile is None else self.tile_key(tile))
        elif kind == 'stitched':
            inputs = self.stitched(tiles)
        elif kind == 'mosaic':
            frames = [f for f in _cal_frames(cal) if os.path.exists(os.path.join(frame_dir, os.path.basename(f)))]
            inputs = self.mosaic(job, cal, frames)
        elif kind == 'pass':
            inputs = self.passed(job, index, pass_type)
        else:
            raise ValueError(f"unknown product kind {kind!r}")
        write_sidecar(path, inputs, record=self.record)
        self.written.append(os.fspath(path))


def expected_products(plan, book, frames) -> list:
    """Every product ``plan``'s action writes or reuses, in the order the engine makes them, with
    the thunks of their inputs registered in ``book``."""
    from .engine import resolve_tiles, tile_assignment
    from .schedule import schedule
    out = []

    def add(product):
        out.append(product)
        book.expected[product.path] = product.inputs

    recipe, tiles, passes = plan.recipe, plan.tiles, plan.passes
    for spec, ctx in zip(plan.lowered, plan.contexts):
        cal_dir = ctx.pipeline_config.cal_dir
        for job in ctx.jobs():
            if tiles is None:
                elsewhere = plan.action == 'mosaic' and spec.cal_override
                cal = os.fspath(spec.cal_override) if elsewhere else ctx.cal_path(job)
                if not elsewhere:
                    add(Product('cal', cal, job, lambda job=job: book.cal(job, frames)))
                init = cal
                if plan.action == 'mosaic' or (recipe.coadd is not None and passes is None):
                    frame_dir = spec.frames.in_place or ctx.pipeline_config.reproj_dir

                    def mosaic_inputs_(job=job, cal=cal, d=frame_dir):
                        here = [f for f in _cal_frames(cal) if os.path.exists(os.path.join(d, os.path.basename(f)))]
                        return book.mosaic(job, cal, here)
                    add(Product('mosaic', ctx.mosaic_path(job), job, mosaic_inputs_, depends=(cal,)))
            else:
                specs, only = resolve_tiles(spec.tiling)
                assignment = {}

                def tile_frames(name, tiling=spec.tiling, assignment=assignment):
                    if not assignment:
                        assignment.update(tile_assignment(tiling)[3])
                    return assignment[name][0]
                tile_paths = {}
                for tile in specs:
                    path = os.path.join(cal_dir, ctx.tile_cal_file(job, tile))
                    tile_paths[tile.name] = path
                    add(Product('cal', path, job, lambda job=job, tile=tile: book.cal(
                        job, tile_frames(tile.name), book.tile_key(tile)), tile=tile.name))
                init = ctx.stitched_cal_path(job)
                if not only:
                    add(Product('stitched', init, job, lambda paths=tile_paths: book.stitched(paths),
                                depends=tuple(tile_paths.values())))
            book.init_path[job.name] = init
            if passes is not None:
                stem = ctx.pass_stem(job)
                for i, kind in enumerate(schedule(passes.n, passes.order)[1:], start=2):
                    t_ = 'sky' if kind == 'sky' else 'off'
                    add(Product('pass', os.path.join(cal_dir, f'{stem}_pass{i}{t_}.h5'), job,
                                lambda job=job, i=i, t_=t_: book.passed(job, i, t_), index=i, depends=(init,)))
    return out


def remove_product(path):
    """Delete a product and its sidecar."""
    for p in (os.fspath(path), sidecar_path(path)):
        if os.path.exists(p):
            os.remove(p)


# --------------------------------------------------------------------------- adoption
def verify(product, recipe, geom=None) -> list[str]:
    """What is wrong with an existing product as one ``recipe`` makes (empty: nothing): a cal's
    sky terms, offset maps and detector shape; a mosaic's maps. Its inputs are checked by the
    caller (the frames, from the product itself)."""
    path = product.path
    if product.kind == 'mosaic':
        from astropy.io import fits
        try:
            with fits.open(path) as h:
                names = [x.name for x in h]
        except Exception as e:
            return [f"cannot be read ({type(e).__name__}: {e})"]
        return [] if 'MEAN_MAP' in names else [f"has no MEAN_MAP ({names})"]
    from ..io.calfile import CalFile
    try:
        with CalFile(path) as cal:
            names = list(cal.sky_names)
            n_maps = cal.num_maps
            maps = cal.chunk_maps if n_maps else []
            n_frames = cal.n_frames
    except Exception as e:
        return [f"cannot be read ({type(e).__name__}: {e})"]
    problems = []
    if not n_frames and product.kind != 'stitched':          # a stitched cal keeps no frame list
        problems.append("it lists no frames")
    if product.kind == 'pass':
        return problems
    want = [t.name for t in recipe.model.sky]
    if product.kind == 'stitched':                           # it names its maps generically
        if len(names) != len(want):
            problems.append(f"it has {len(names)} sky maps, the recipe {len(want)} sky terms")
    elif names != want:
        problems.append(f"its sky terms are {names}, the recipe's {want}")
    if product.kind == 'cal' and n_maps != len(recipe.model.offsets):
        problems.append(f"it has {n_maps} offset maps, the recipe {len(recipe.model.offsets)} offset terms")
    elif product.kind == 'cal' and maps and geom is not None and tuple(np.shape(maps[0])) != tuple(geom.shape):
        problems.append(f"its chunk map is {np.shape(maps[0])}, the instrument's detector {tuple(geom.shape)}")
    return problems


# --------------------------------------------------------------------------- N-pass intermediates
def clear_stale_intermediates(work_dir, cal_dir, stem, expected):
    """Delete the N-pass intermediates that were made from other inputs.

    The SKY passes keep per-tile moment dumps (``<work_dir>/moments/pass<i>_<tile>.npz``; with
    ``sky_merge="stitch"``, ``<cal_dir>/<stem>_pass<i>sky_<tile>.h5``) and the OFFSET passes export
    the previous sky (``<work_dir>/sky_pass<j>/``); the engine reuses any it finds, so an
    interrupted run can resume. ``expected`` maps ``"pass<i>"`` and ``"sky<j>"`` to the fingerprint
    of what made them; a manifest in ``work_dir`` records the fingerprints the intermediates were
    made with, and those that differ (or are unknown) are deleted before the run.
    """
    manifest = os.path.join(work_dir, 'intermediates.json')
    old = {}
    if os.path.exists(manifest):
        with open(manifest) as f:
            old = json.load(f)
    removed = []
    for key, fp in expected.items():
        if old.get(key) == fp:
            continue
        if key.startswith('pass'):
            i = key[len('pass'):]
            paths = glob.glob(os.path.join(work_dir, 'moments', f'pass{i}_*.npz'))
            paths += glob.glob(os.path.join(cal_dir, f'{stem}_pass{i}sky_*.h5'))
        else:
            paths = [os.path.join(work_dir, f'sky_pass{key[len("sky"):]}')]
        for p in paths:
            if os.path.isdir(p):
                shutil.rmtree(p)
                removed.append(p)
            elif os.path.exists(p):
                os.remove(p)
                removed.append(p)
    if os.path.isdir(work_dir) or expected:
        os.makedirs(work_dir, exist_ok=True)
        with atomic_path(manifest) as tmp:
            with open(tmp, 'w') as f:
                json.dump({**old, **expected}, f, indent=1, sort_keys=True)
    return removed
